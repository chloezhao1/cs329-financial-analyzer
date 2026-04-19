"""
financial_signal_engine.py
==========================

CS329 Financial Report Analyzer -- Signal Extraction Engine
Owner: Nick

Consumes the JSON records produced by `text_preprocessor.py` and produces
per-document Indicator Profiles (Growth, Risk, Cost Pressure, Net Operating
Signal) grounded in the Loughran-McDonald Master Dictionary.

Output also includes sector-relative z-scores when `baseline_stats.json`
exists (written by `build_baseline.py`). The z-score section always
includes a human-readable `reference_label` making it obvious whether
the company was compared to its sector peers or to the full corpus
(for tickers outside sector_map.py).

Design (lexicon-grounded, no ML):

    Layer 1 -- Word-level lookup in Loughran-McDonald Master Dictionary
      Positive words            -> Growth
      Negative + Litigious +    -> Risk
        Constraining words
      Uncertainty words         -> Risk (half weight, epistemic not directional)
      (LM has no Cost Pressure category)

    Layer 2 -- Curated multi-word phrase dictionary
      Supplements LM for multi-word signals like "margin compression"

    Layer 3 -- Context adjustments using preprocessing flags
      has_negation -> flip sign of sentence scores
      has_hedge    -> multiply by 0.5

    Layer 4 -- Section-aware weighting
      forward_guidance > mdna > qa_A > prepared_remarks > qa_Q > risk_factors

    Layer 5 (optional, at batch level) -- Sector-relative z-scores
      Computed via apply_baseline.py if baseline_stats.json is present.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from apply_baseline import BaselineApplier
    _HAS_BASELINE = True
except ImportError:
    _HAS_BASELINE = False
    BaselineApplier = None  # type: ignore

logger = logging.getLogger(__name__)

ENGINE_VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# 1. Loughran-McDonald loader
# ---------------------------------------------------------------------------

LM_CATEGORY_MAP: dict[str, list[str]] = {
    "growth":        ["Positive"],
    "risk":          ["Negative", "Litigious", "Constraining"],
    "uncertainty":   ["Uncertainty"],
    "cost_pressure": [],
}

LM_POSITIVE_BLOCKLIST: set[str] = {
    "able", "effective", "effectively", "efficient", "efficiently",
    "successful", "successfully", "success", "succeed", "achievable",
    "satisfactory", "satisfactorily", "adequate", "adequately",
    "complete", "completed", "completing", "resolved", "resolve",
    "benefit", "beneficial", "favorable", "favorably", "valuable",
    "reward", "rewarded", "enjoy", "enjoyed",
}

LM_RISK_BLOCKLIST: set[str] = {
    "adverse", "adversely", "material", "materially",
    "require", "required", "requires", "requirement", "requirements",
    "law", "laws", "legal", "legally",
    "subject", "regulatory",
}

UNCERTAINTY_RISK_WEIGHT = 0.5


@dataclass
class LMDictionary:
    """Loaded LM categories as lowercase lemma sets."""
    growth:       set[str] = field(default_factory=set)
    risk:         set[str] = field(default_factory=set)
    uncertainty:  set[str] = field(default_factory=set)
    total_words:  int = 0

    @classmethod
    def from_csv(cls, csv_path: Path) -> "LMDictionary":
        if not csv_path.exists():
            raise FileNotFoundError(
                f"LM dictionary not found at {csv_path}. Download from "
                f"https://sraf.nd.edu/loughranmcdonald-master-dictionary/ "
                f"and place at {csv_path}."
            )

        d = cls()
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            required = set()
            for indicator, cats in LM_CATEGORY_MAP.items():
                required.update(cats)
            missing = required - set(cols)
            if missing:
                raise ValueError(
                    f"LM CSV missing required columns: {missing}. "
                    f"Available columns: {cols}"
                )

            for row in reader:
                word = row["Word"].strip().lower()
                if not word:
                    continue
                d.total_words += 1
                if any(_nonzero(row[c]) for c in LM_CATEGORY_MAP["growth"]):
                    d.growth.add(word)
                if any(_nonzero(row[c]) for c in LM_CATEGORY_MAP["risk"]):
                    d.risk.add(word)
                if any(_nonzero(row[c]) for c in LM_CATEGORY_MAP["uncertainty"]):
                    d.uncertainty.add(word)

        removed_g = d.growth & LM_POSITIVE_BLOCKLIST
        d.growth -= LM_POSITIVE_BLOCKLIST
        removed_r = d.risk & LM_RISK_BLOCKLIST
        d.risk -= LM_RISK_BLOCKLIST
        if removed_g:
            logger.info("Removed %d non-growth LM Positive words", len(removed_g))
        if removed_r:
            logger.info("Removed %d RF-boilerplate LM risk words", len(removed_r))

        logger.info(
            "Loaded LM: %d total, %d growth, %d risk, %d uncertainty",
            d.total_words, len(d.growth), len(d.risk), len(d.uncertainty),
        )
        return d


def _nonzero(v: Any) -> bool:
    try:
        return int(str(v).strip()) != 0
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 2. Phrase dictionary
# ---------------------------------------------------------------------------

PHRASE_WEIGHT = 2.0

GROWTH_PHRASES: set[str] = {
    "revenue growth", "net sales increased", "strong demand",
    "market share gains", "record quarter", "accelerating growth",
    "accelerating demand", "raised guidance", "margin expansion",
    "outperformed", "exceeded expectations", "increased adoption",
    "expanded footprint", "robust demand", "double-digit growth",
}

RISK_PHRASES: set[str] = {
    "supply chain disruption", "foreign currency headwinds",
    "macroeconomic uncertainty", "demand weakness", "regulatory scrutiny",
    "material adverse", "competitive pressure", "geopolitical tensions",
    "customer softness", "weaker spending", "elevated risk",
    "adverse effect", "adversely affected", "cyber attack",
    "reputational harm",
}

COST_PRESSURE_PHRASES: set[str] = {
    "margin compression", "margin pressure", "cost inflation",
    "input cost", "wage pressure", "pricing pressure",
    "elevated freight costs", "operating expenses increased",
    "higher costs", "labor costs", "rising costs",
    "higher infrastructure costs", "commodity cost",
    "higher input costs", "cost headwinds",
}


# ---------------------------------------------------------------------------
# 3. Section weights
# ---------------------------------------------------------------------------

SECTION_WEIGHTS: dict[str, float] = {
    "mdna":               1.0,
    "forward_guidance":   1.2,
    "risk_factors":       0.3,
    "prepared_remarks":   1.0,
}

QA_A_WEIGHT = 1.1
QA_Q_WEIGHT = 0.3


def _section_weight(section_name: str) -> float:
    if section_name.startswith("qa_Q"):
        return QA_Q_WEIGHT
    if section_name.startswith("qa_A"):
        return QA_A_WEIGHT
    return SECTION_WEIGHTS.get(section_name, 1.0)


# ---------------------------------------------------------------------------
# 4. Sentence-level scoring
# ---------------------------------------------------------------------------

MIN_TOKENS_FOR_SCORING = 5


def _sentence_lemmas(sentence: dict) -> set[str]:
    lemmas: set[str] = set()
    for tok in sentence.get("tokens", []):
        if tok.get("is_stop"):
            continue
        lemma = tok.get("lemma") or tok.get("text", "").lower()
        if lemma:
            lemmas.add(lemma)
    return lemmas


def _find_phrases(text_lower: str, phrase_set: set[str]) -> list[str]:
    return [p for p in phrase_set if p in text_lower]


@dataclass
class SentenceScore:
    sent_id:              int
    section:              str
    text:                 str
    growth:               float
    risk:                 float
    cost_pressure:        float
    net_score:            float
    has_negation:         bool
    has_hedge:            bool
    lm_growth_hits:       list[str]
    lm_risk_hits:         list[str]
    lm_uncertainty_hits:  list[str]
    phrase_growth_hits:   list[str]
    phrase_risk_hits:     list[str]
    phrase_cost_hits:     list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sent_id":              self.sent_id,
            "section":              self.section,
            "text":                 self.text,
            "growth":               round(self.growth, 3),
            "risk":                 round(self.risk, 3),
            "cost_pressure":        round(self.cost_pressure, 3),
            "net_score":            round(self.net_score, 3),
            "has_negation":         self.has_negation,
            "has_hedge":            self.has_hedge,
            "lm_growth_hits":       self.lm_growth_hits,
            "lm_risk_hits":         self.lm_risk_hits,
            "lm_uncertainty_hits":  self.lm_uncertainty_hits,
            "phrase_growth_hits":   self.phrase_growth_hits,
            "phrase_risk_hits":     self.phrase_risk_hits,
            "phrase_cost_hits":     self.phrase_cost_hits,
        }


class SignalEngine:
    """Lexicon-grounded per-sentence and per-document signal extraction."""

    def __init__(self, lm: LMDictionary):
        self.lm = lm

    def score_sentence(self, sentence: dict) -> SentenceScore | None:
        tokens = sentence.get("tokens", [])
        if len(tokens) < MIN_TOKENS_FOR_SCORING:
            return None

        text = sentence.get("text", "")
        text_lower = text.lower()
        lemmas = _sentence_lemmas(sentence)
        section = sentence.get("section", "")
        has_neg = bool(sentence.get("has_negation"))
        has_hedge = bool(sentence.get("has_hedge"))

        lm_growth = sorted(lemmas & self.lm.growth)
        lm_risk   = sorted(lemmas & self.lm.risk)
        lm_uncert = sorted(lemmas & self.lm.uncertainty)

        ph_growth = _find_phrases(text_lower, GROWTH_PHRASES)
        ph_risk   = _find_phrases(text_lower, RISK_PHRASES)
        ph_cost   = _find_phrases(text_lower, COST_PRESSURE_PHRASES)

        growth = len(lm_growth) + len(ph_growth) * PHRASE_WEIGHT
        risk   = (len(lm_risk)
                  + len(lm_uncert) * UNCERTAINTY_RISK_WEIGHT
                  + len(ph_risk) * PHRASE_WEIGHT)
        cost   = len(ph_cost) * PHRASE_WEIGHT

        if has_neg:
            growth, risk, cost = -growth, -risk, -cost
        if has_hedge:
            growth *= 0.5
            risk   *= 0.5
            cost   *= 0.5

        w = _section_weight(section)
        growth *= w
        risk   *= w
        cost   *= w

        return SentenceScore(
            sent_id=sentence.get("sent_id", -1),
            section=section,
            text=text,
            growth=growth,
            risk=risk,
            cost_pressure=cost,
            net_score=growth - risk,
            has_negation=has_neg,
            has_hedge=has_hedge,
            lm_growth_hits=lm_growth,
            lm_risk_hits=lm_risk,
            lm_uncertainty_hits=lm_uncert,
            phrase_growth_hits=ph_growth,
            phrase_risk_hits=ph_risk,
            phrase_cost_hits=ph_cost,
        )

    def analyze_record(self, record: dict) -> dict:
        sentences = (record.get("processed") or {}).get("sentences") or []
        scored: list[SentenceScore] = []
        for s in sentences:
            result = self.score_sentence(s)
            if result is not None:
                scored.append(result)

        if not scored:
            logger.warning(
                "No scoreable sentences in %s %s", record.get("ticker"),
                record.get("filing_date"),
            )

        n = len(scored) or 1
        doc_growth = sum(s.growth for s in scored) / n
        doc_risk   = sum(s.risk   for s in scored) / n
        doc_cost   = sum(s.cost_pressure for s in scored) / n

        phrase_tallies = {
            "growth": self._tally_phrases(scored, "phrase_growth_hits", "lm_growth_hits"),
            "risk":   self._tally_phrases(scored, "phrase_risk_hits",   "lm_risk_hits"),
            "cost":   self._tally_phrases(scored, "phrase_cost_hits",   None),
        }

        top_sentences = sorted(
            scored,
            key=lambda s: abs(s.net_score) + abs(s.cost_pressure),
            reverse=True,
        )[:20]

        return {
            "ticker":          record.get("ticker", "UNK"),
            "company_name":    record.get("company_name", "Unknown"),
            "form_type":       record.get("form_type", "UNKNOWN"),
            "filing_date":     record.get("filing_date", "Unknown"),
            "source":          record.get("source", "SEC EDGAR"),
            "method": {
                "type":                "loughran_mcdonald_lexicon",
                "engine_version":      ENGINE_VERSION,
                "lm_words_loaded":     self.lm.total_words,
                "lm_growth_words":     len(self.lm.growth),
                "lm_risk_words":       len(self.lm.risk),
                "lm_uncertainty_words": len(self.lm.uncertainty),
                "phrase_counts": {
                    "growth": len(GROWTH_PHRASES),
                    "risk":   len(RISK_PHRASES),
                    "cost":   len(COST_PRESSURE_PHRASES),
                },
                "aggregation":         "mean_per_sentence",
            },
            "scores": {
                "growth":               round(doc_growth, 3),
                "risk":                 round(doc_risk,   3),
                "cost_pressure":        round(doc_cost,   3),
                "net_operating_signal": round(doc_growth - doc_risk, 3),
            },
            "coverage": {
                "scored_sentences":     len(scored),
                "scored_with_hits":     sum(
                    1 for s in scored
                    if s.lm_growth_hits or s.lm_risk_hits or s.lm_uncertainty_hits
                    or s.phrase_growth_hits or s.phrase_risk_hits or s.phrase_cost_hits
                ),
                "sentences_by_section": self._by_section(scored),
            },
            "top_sentences":     [s.to_dict() for s in top_sentences],
            "top_growth_phrases": phrase_tallies["growth"],
            "top_risk_phrases":   phrase_tallies["risk"],
            "top_cost_phrases":   phrase_tallies["cost"],
        }

    @staticmethod
    def _tally_phrases(
        scored: list[SentenceScore],
        phrase_attr: str,
        lm_attr: str | None,
    ) -> list[dict]:
        counts: dict[str, int] = defaultdict(int)
        sources: dict[str, str] = {}
        for s in scored:
            for p in getattr(s, phrase_attr):
                counts[p] += 1
                sources[p] = "phrase"
            if lm_attr:
                for w in getattr(s, lm_attr):
                    counts[w] += 1
                    sources[w] = "lm_word"
        return [
            {"term": term, "source": sources[term], "count": counts[term]}
            for term in sorted(counts, key=lambda k: counts[k], reverse=True)
        ][:25]

    @staticmethod
    def _by_section(scored: list[SentenceScore]) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for s in scored:
            d[s.section] += 1
        return dict(d)


# ---------------------------------------------------------------------------
# 5. Batch runner
# ---------------------------------------------------------------------------

def infer_data_source(base_dir: Path) -> str:
    """
    Tell the frontend which data stage is available, in priority order:
      'data/processed'  -> preprocessing has run, signals can be computed
      'pipeline_output' -> scraping has run but preprocessing has not
      'demo_data'       -> nothing scraped yet; fall back to bundled samples
    """
    processed_dir = base_dir / "data" / "processed"
    if processed_dir.exists() and list(processed_dir.glob("*.processed.json")):
        return "data/processed"
    filings_dir = base_dir / "data" / "filings"
    transcripts_dir = base_dir / "data" / "transcripts"
    if (filings_dir.exists() and list(filings_dir.glob("*.json"))) or (
        transcripts_dir.exists() and list(transcripts_dir.glob("*.json"))
    ):
        return "pipeline_output"
    return "demo_data"


def load_records(base_dir: Path) -> list[dict]:
    processed_dir = base_dir / "data" / "processed"
    if processed_dir.exists():
        records = []
        for path in sorted(processed_dir.glob("*.processed.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.error("Could not read %s: %s", path, e)
        if records:
            return records

    demo_path = base_dir / "demo_data" / "sample_documents.json"
    if demo_path.exists():
        return json.loads(demo_path.read_text(encoding="utf-8"))
    return []


def analyze_records(
    records: list[dict],
    engine: "SignalEngine | None" = None,
    applier: "BaselineApplier | None" = None,
) -> list[dict]:
    """
    Score records, optionally attaching sector-relative z-scores.

    If `engine` is not provided, one is lazily created using the default
    LM dictionary path. This keeps backwards compatibility with callers
    like frontend_app.py that call analyze_records(records) directly.

    If `applier` is not provided and the baseline file exists, one is
    lazily created. This ensures the frontend gets z-scores too.
    """
    if engine is None:
        lm = LMDictionary.from_csv(Path("data/lexicons/loughran_mcdonald.csv"))
        engine = SignalEngine(lm)
    if applier is None and _HAS_BASELINE:
        try:
            applier = BaselineApplier()
        except Exception as e:
            logger.warning("Could not load baseline automatically: %s", e)
            applier = None

    analyses = [engine.analyze_record(r) for r in records]
    if applier is not None:
        analyses = applier.apply_all(analyses)
    return analyses


def build_comparison_rows(analyses: list[dict]) -> list[dict]:
    """Flat rows for the comparison UI (Chloe consumes this)."""
    rows = []
    for a in analyses:
        z = a.get("zscores") or {}
        rows.append({
            "label":                f"{a['ticker']} {a['filing_date']}",
            "ticker":               a["ticker"],
            "filing_date":          a["filing_date"],
            "form_type":            a["form_type"],
            # Raw scores
            "growth":               a["scores"]["growth"],
            "risk":                 a["scores"]["risk"],
            "cost_pressure":        a["scores"]["cost_pressure"],
            "net_operating_signal": a["scores"]["net_operating_signal"],
            # Z-scores + clear reference labeling
            "z_growth":             z.get("growth", 0.0) if z else None,
            "z_risk":               z.get("risk", 0.0) if z else None,
            "z_net":                z.get("net_operating_signal", 0.0) if z else None,
            "z_reference":          z.get("reference", None) if z else None,
            "z_reference_label":    z.get("reference_label", None) if z else None,
            "z_is_sector_specific": z.get("is_sector_specific", False) if z else None,
            "z_reliable":           z.get("reference_reliable", False) if z else None,
            "scored_sentences":     a["coverage"]["scored_sentences"],
        })
    return rows


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="LM-grounded signal extraction.")
    ap.add_argument("--base-dir", type=Path, default=Path("."),
                    help="Project root (default: .)")
    ap.add_argument(
        "--lm-csv", type=Path,
        default=Path("data/lexicons/loughran_mcdonald.csv"),
        help="Path to Loughran-McDonald Master Dictionary CSV",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/signals"),
                    help="Where to write per-document signal JSONs")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N records")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip applying sector-relative z-scores even if "
                         "baseline_stats.json exists")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    lm = LMDictionary.from_csv(args.lm_csv)
    engine = SignalEngine(lm)

    applier: "BaselineApplier | None" = None
    if _HAS_BASELINE and not args.no_baseline:
        applier = BaselineApplier()

    records = load_records(args.base_dir)
    if args.limit:
        records = records[:args.limit]
    logger.info("Analyzing %d records", len(records))

    out_dir = args.base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for rec in records:
        a = engine.analyze_record(rec)
        if applier is not None:
            applier.apply(a)
        analyses.append(a)

        fname = (
            f"{a['ticker']}_{a['form_type']}_{a['filing_date']}.signals.json"
        )
        (out_dir / fname).write_text(
            json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        # Log raw scores + z-score tail. Make unmapped-company fallback
        # obvious by adding a `[CORPUS]` tag when not sector-specific.
        z = a.get("zscores") or {}
        if z and z.get("reference") not in (None, "no_baseline"):
            tag = ""
            if not z.get("is_sector_specific"):
                tag = " [CORPUS FALLBACK — ticker not in sector map]"
            z_tail = (
                f"  (z_net={z['net_operating_signal']:+.2f}"
                f" vs {z['reference']}, n={z['reference_n']}){tag}"
            )
        else:
            z_tail = ""
        logger.info(
            "%-6s %s %-4s  growth=%+.2f  risk=%+.2f  net=%+.2f%s",
            a["ticker"], a["filing_date"], a["form_type"],
            a["scores"]["growth"], a["scores"]["risk"],
            a["scores"]["net_operating_signal"], z_tail,
        )

    index = {
        "engine_version": ENGINE_VERSION,
        "n_records":      len(analyses),
        "baseline_used":  applier is not None,
        "records":        build_comparison_rows(analyses),
    }
    (out_dir / "_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Done. Wrote %d signal files + _index.json to %s",
                len(analyses), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())