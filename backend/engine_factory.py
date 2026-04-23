"""Default signal-engine factory for the API layer.

Central place that decides which engine implementation the dashboard and
pipeline run through. Swapping defaults later (e.g. to the hybrid LLM
engine) is a one-line change here instead of scattered across routers.

The hybrid engine (`financial_signal_engine_LLMv1.HybridSignalEngine`) is
deliberately NOT the default: it requires an Anthropic API key and its
output shape is per-sentence, not per-document, so it does not drop into
`analyze_records`. It is wired in only where a hybrid evaluation path is
explicitly requested.

Why `DefaultV2Engine` exists
----------------------------
`SignalEngineV2` in `financial_signal_engine_v2.py` overrides only
`score_sentence`; it is not a subclass of `SignalEngine` and therefore
has no `analyze_record` / `_tally_phrases` / `_by_section`. But
`analyze_records` in v1 calls `engine.analyze_record(r)` per document.

We cannot modify the engine files. The cleanest fix is a small subclass
here that reuses V2's sentence scorer + LM blocklist but keeps v1's
document-level aggregation machinery, via multiple inheritance.

`DefaultV3Engine`
----------------
`HybridSignalEngineV3` in `financial_signal_engine_v3.py` exposes
`score_batch` only. This subclass inherits `SignalEngine` for
`analyze_record` aggregation only: it calls the V3 batch scorer, then
builds the same document dict shape as the lexicon engine (scores,
top_sentences, phrase tables). Does **not** modify `financial_signal_engine_v3.py`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from financial_signal_engine import (
    COST_PRESSURE_PHRASES,
    GROWTH_PHRASES,
    LMDictionary,
    RISK_PHRASES,
    SentenceScore,
    SignalEngine,
)
from financial_signal_engine_v2 import (
    GROWTH_PHRASES_V2,
    MIN_TOKENS_FOR_SCORING_V2,
    SignalEngineV2,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DefaultV2Engine",
    "DefaultV3Engine",
    "build_default_engine",
    "build_default_v3_engine",
    "load_default_engine",
    "load_default_v3_engine",
    "load_preferred_signal_engine",
    "resolve_v3_model_path",
]


def resolve_v3_model_path(project_root: Path) -> Path:
    """Classifier + TF-IDF bundle produced by `evaluate_v3.py`."""
    p = project_root / "data" / "models" / "finbert_logreg.pkl"
    if p.is_file():
        return p
    cwd_candidate = Path("data/models/finbert_logreg.pkl")
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()
    return p


class DefaultV2Engine(SignalEngineV2, SignalEngine):
    """V2 sentence scoring + v1 document-level aggregation.

    MRO: ``[DefaultV2Engine, SignalEngineV2, SignalEngine, object]``

    - ``__init__``       — SignalEngineV2 (builds a trimmed LM, sets self.lm)
    - ``score_sentence`` — SignalEngineV2 (V2 phrase list, token floor, negation)
    - ``analyze_record`` — SignalEngine   (document aggregation, top sentences,
                                           phrase tallies, section breakdown)
    - ``_tally_phrases`` / ``_by_section`` — SignalEngine helpers
    """


def build_default_engine(lm: LMDictionary) -> DefaultV2Engine:
    """Construct the default (V2 lexicon) signal engine from a loaded LM."""
    return DefaultV2Engine(lm)


def load_default_engine(lm_csv: Path) -> DefaultV2Engine:
    """Load the LM CSV and return a ready-to-use V2 engine."""
    return build_default_engine(LMDictionary.from_csv(lm_csv))


class DefaultV3Engine(SignalEngine):
    """
    V3 `HybridSignalEngineV3` batch scoring + v1 document-level aggregation.

    Produces the same `analyze_record` JSON shape as `SignalEngine` / V2
    so `apply_baseline` z-scores and the React `Analysis` type stay valid.
    """

    def __init__(self, lm: LMDictionary, model_path: Path) -> None:
        from financial_signal_engine_v3 import (  # noqa: WPS433
            V3_VERSION,
            HybridSignalEngineV3,
        )

        super().__init__(lm)
        if not model_path.is_file():
            logger.warning(
                "V3 classifier not found at %s — HybridSignalEngineV3 will "
                "error when embedding tier runs. Run evaluate_v3.py to train.",
                model_path,
            )
        self._v3 = HybridSignalEngineV3(
            lm,
            model_path=model_path,
            use_llm=False,
        )
        self._model_path = model_path
        self._v3_version = V3_VERSION

    @staticmethod
    def _normalize_row(r: dict) -> tuple[float, float, float, float]:
        """Return (growth, risk, cost_pressure, net_score) for aggregation."""
        g = float(r.get("growth", 0.0))
        rk = float(r.get("risk", 0.0))
        c = float(r.get("cost_pressure", 0.0))
        ns = float(r.get("net_score", 0.0))
        method = r.get("method", "")
        # Embedding/LLM tiers may leave growth/risk at 0 while net_score is set.
        if method in ("embedding", "llm", "llm_pending") or (
            abs(g) < 1e-12 and abs(rk) < 1e-12 and abs(ns) > 1e-12
        ):
            g, rk = max(0.0, ns), max(0.0, -ns)
        return g, rk, c, ns

    def analyze_record(self, record: dict) -> dict:  # noqa: PLR0914
        sentences = (record.get("processed") or {}).get("sentences") or []
        filtered = [
            s
            for s in sentences
            if len(s.get("tokens") or []) >= MIN_TOKENS_FOR_SCORING_V2
        ]
        if not filtered:
            logger.warning(
                "No scoreable sentences in %s %s (v3)",
                record.get("ticker"),
                record.get("filing_date"),
            )
        raw_results: list[dict] = self._v3.score_batch(filtered) if filtered else []

        scored: list[SentenceScore] = []
        for sent, r in zip(filtered, raw_results):
            method = r.get("method", "")
            if method == "lexicon":
                ss = self._v3.lexicon_engine.score_sentence(sent)
                if ss is not None:
                    scored.append(ss)
            else:
                g, rk, c, _ns = self._normalize_row(r)
                scored.append(
                    SentenceScore(
                        sent_id=sent.get("sent_id", -1),
                        section=sent.get("section", ""),
                        text=sent.get("text", ""),
                        growth=g,
                        risk=rk,
                        cost_pressure=c,
                        net_score=g - rk,
                        has_negation=bool(sent.get("has_negation")),
                        has_hedge=bool(sent.get("has_hedge")),
                        lm_growth_hits=[],
                        lm_risk_hits=[],
                        lm_uncertainty_hits=[],
                        phrase_growth_hits=[],
                        phrase_risk_hits=[],
                        phrase_cost_hits=[],
                    )
                )

        n = len(scored) or 1
        doc_growth = sum(s.growth for s in scored) / n
        doc_risk = sum(s.risk for s in scored) / n
        doc_cost = sum(s.cost_pressure for s in scored) / n

        phrase_tallies = {
            "growth": self._tally_phrases(scored, "phrase_growth_hits", "lm_growth_hits"),
            "risk": self._tally_phrases(scored, "phrase_risk_hits", "lm_risk_hits"),
            "cost": self._tally_phrases(scored, "phrase_cost_hits", None),
        }

        top_sentences = sorted(
            scored,
            key=lambda s: abs(s.net_score) + abs(s.cost_pressure),
            reverse=True,
        )[:20]

        return {
            "ticker": record.get("ticker", "UNK"),
            "company_name": record.get("company_name", "Unknown"),
            "form_type": record.get("form_type", "UNKNOWN"),
            "filing_date": record.get("filing_date", "Unknown"),
            "source": record.get("source", "SEC EDGAR"),
            "method": {
                "type": "loughran_mcdonald_plus_finbert_v3",
                "engine_version": self._v3_version,
                "engine_id": "v3",
                "signal_engine": "HybridSignalEngineV3",
                "classifier_path": str(self._model_path),
                "use_llm_tier": False,
                "lm_words_loaded": self.lm.total_words,
                "lm_growth_words": len(self.lm.growth),
                "lm_risk_words": len(self.lm.risk),
                "lm_uncertainty_words": len(self.lm.uncertainty),
                "phrase_counts": {
                    "growth": len(GROWTH_PHRASES_V2),
                    "risk": len(RISK_PHRASES),
                    "cost": len(COST_PRESSURE_PHRASES),
                },
                "legacy_v1_phrase_count_growth": len(GROWTH_PHRASES),
                "aggregation": "mean_per_sentence",
            },
            "scores": {
                "growth": round(doc_growth, 3),
                "risk": round(doc_risk, 3),
                "cost_pressure": round(doc_cost, 3),
                "net_operating_signal": round(doc_growth - doc_risk, 3),
            },
            "coverage": {
                "scored_sentences": len(scored),
                "scored_with_hits": sum(
                    1
                    for s in scored
                    if s.lm_growth_hits
                    or s.lm_risk_hits
                    or s.lm_uncertainty_hits
                    or s.phrase_growth_hits
                    or s.phrase_risk_hits
                    or s.phrase_cost_hits
                ),
                "sentences_by_section": self._by_section(scored),
            },
            "top_sentences": [s.to_dict() for s in top_sentences],
            "top_growth_phrases": phrase_tallies["growth"],
            "top_risk_phrases": phrase_tallies["risk"],
            "top_cost_phrases": phrase_tallies["cost"],
        }


def build_default_v3_engine(lm: LMDictionary, project_root: Path) -> DefaultV3Engine:
    path = resolve_v3_model_path(project_root)
    return DefaultV3Engine(lm, model_path=path)


def load_default_v3_engine(lm_csv: Path, project_root: Path) -> DefaultV3Engine:
    return build_default_v3_engine(LMDictionary.from_csv(lm_csv), project_root)


def load_preferred_signal_engine(
    lm_csv: Path, project_root: Path
) -> DefaultV2Engine | DefaultV3Engine:
    """
    Use V3 when it can be constructed (transformers, model bundle, etc.).
    Otherwise fall back to V2 so the API and local dashboard still load.
    """
    try:
        return load_default_v3_engine(lm_csv, project_root)
    except Exception as e:
        logger.warning(
            "V3 engine unavailable; falling back to V2 (%s: %s)",
            type(e).__name__,
            e,
        )
        return load_default_engine(lm_csv)
