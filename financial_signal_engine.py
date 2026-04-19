from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DIMENSIONS = ["growth", "risk", "cost_pressure"]
NEGATION_WORDS = {"not", "no", "never", "without", "unlikely"}
HEDGE_WORDS = {"may", "might", "could", "expect", "anticipate", "likely", "approximately"}

# Cost pressure is not a dedicated Loughran-McDonald label, so we derive it from
# LM-scored negative / constraining vocabulary restricted to cost-like terms.
COST_TERMS = {
    "cost",
    "costs",
    "expense",
    "expenses",
    "inflation",
    "margin",
    "margins",
    "pricing",
    "freight",
    "labor",
    "overhead",
    "input",
}

COMMON_LM_FILENAMES = [
    "LM_MasterDictionary_1993-2024.csv",
    "LM_MasterDictionary_2023.csv",
    "LM_MasterDictionary.csv",
    "loughran_mcdonald.csv",
    "lmdictionary.csv",
]

# Small fallback so the demo app still runs before the real CSV is added.
FALLBACK_LM_ROWS = [
    {"word": "growth", "positive": 1.0},
    {"word": "strong", "positive": 1.0},
    {"word": "improve", "positive": 1.0},
    {"word": "record", "positive": 1.0},
    {"word": "guidance", "strong_modal": 1.0},
    {"word": "expect", "strong_modal": 1.0},
    {"word": "risk", "negative": 1.0, "uncertainty": 1.0},
    {"word": "uncertainty", "negative": 1.0, "uncertainty": 1.0},
    {"word": "headwind", "negative": 1.0, "uncertainty": 1.0},
    {"word": "pressure", "negative": 1.0, "constraining": 1.0},
    {"word": "decline", "negative": 1.0},
    {"word": "softness", "negative": 1.0},
    {"word": "cost", "negative": 1.0, "constraining": 1.0},
    {"word": "expense", "negative": 1.0, "constraining": 1.0},
    {"word": "inflation", "negative": 1.0, "constraining": 1.0},
    {"word": "margin", "negative": 1.0, "constraining": 1.0},
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tokenize_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_lemma(text: str) -> str:
    return re.sub(r"[^a-z_]+", "", text.lower())


def _as_float(value: object) -> float:
    if value in (None, "", "0", 0):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SentenceFeatureBundle:
    vector: np.ndarray
    matched_terms: list[dict]
    aggregates: dict[str, float]


class LMDictionary:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.entries, self.source = self._load_entries()

    def _candidate_paths(self) -> list[Path]:
        candidates = []
        search_dirs = [
            self.base_dir,
            self.base_dir / "data",
            self.base_dir / "data" / "lexicon",
            self.base_dir / "data" / "lexicons",
            self.base_dir / "demo_data",
        ]
        for directory in search_dirs:
            for filename in COMMON_LM_FILENAMES:
                candidates.append(directory / filename)
        return candidates

    def _load_entries(self) -> tuple[dict[str, dict[str, float]], str]:
        for path in self._candidate_paths():
            if not path.exists():
                continue
            if path.suffix.lower() != ".csv":
                continue
            entries = self._load_csv(path)
            if entries:
                return entries, str(path)
        return self._load_fallback(), "built_in_demo_subset"

    def _load_csv(self, path: Path) -> dict[str, dict[str, float]]:
        entries: dict[str, dict[str, float]] = {}
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            field_map = {field.lower(): field for field in (reader.fieldnames or [])}
            word_field = field_map.get("word")
            if not word_field:
                return {}
            for row in reader:
                lemma = _normalize_lemma(str(row.get(word_field, "")))
                if not lemma:
                    continue
                entries[lemma] = {
                    "positive": _as_float(row.get(field_map.get("positive", ""), 0)),
                    "negative": _as_float(row.get(field_map.get("negative", ""), 0)),
                    "uncertainty": _as_float(row.get(field_map.get("uncertainty", ""), 0)),
                    "litigious": _as_float(row.get(field_map.get("litigious", ""), 0)),
                    "strong_modal": _as_float(row.get(field_map.get("strong_modal", ""), 0)),
                    "weak_modal": _as_float(row.get(field_map.get("weak_modal", ""), 0)),
                    "constraining": _as_float(row.get(field_map.get("constraining", ""), 0)),
                }
        return entries

    def _load_fallback(self) -> dict[str, dict[str, float]]:
        entries: dict[str, dict[str, float]] = {}
        for row in FALLBACK_LM_ROWS:
            lemma = row["word"]
            entries[lemma] = {
                "positive": row.get("positive", 0.0),
                "negative": row.get("negative", 0.0),
                "uncertainty": row.get("uncertainty", 0.0),
                "litigious": row.get("litigious", 0.0),
                "strong_modal": row.get("strong_modal", 0.0),
                "weak_modal": row.get("weak_modal", 0.0),
                "constraining": row.get("constraining", 0.0),
            }
        return entries

    def lookup(self, lemma: str) -> dict[str, float] | None:
        return self.entries.get(_normalize_lemma(lemma))


def _sentence_records(record: dict) -> list[dict]:
    processed = record.get("processed", {})
    sentences = processed.get("sentences")
    if sentences:
        return sentences

    raw_text = record.get("raw_text", "")
    return [
        {
            "sent_id": idx,
            "section": "raw_text",
            "text": sentence,
            "tokens": [
                {"lemma": token, "text": token, "is_stop": False}
                for token in re.findall(r"[a-zA-Z']+", sentence.lower())
            ],
            "has_negation": any(word in _normalize_text(sentence).split() for word in NEGATION_WORDS),
            "has_hedge": any(word in _normalize_text(sentence).split() for word in HEDGE_WORDS),
        }
        for idx, sentence in enumerate(_tokenize_sentences(raw_text))
    ]


class TwoHiddenLayerFinancialSignalModel:
    """
    Two-hidden-layer model whose features come from Loughran-McDonald categories.

    The hidden layers remain, but the evidence feeding them is no longer based on
    custom phrase weights. Instead, it comes from actual LM dictionary matches
    across sentence lemmas.
    """

    def __init__(self, base_dir: Path) -> None:
        self.lexicon = LMDictionary(base_dir)
        self.feature_names = [
            "positive_sum",
            "negative_sum",
            "uncertainty_sum",
            "litigious_sum",
            "strong_modal_sum",
            "weak_modal_sum",
            "constraining_sum",
            "positive_count",
            "negative_count",
            "uncertainty_count",
            "cost_term_count",
            "has_hedge",
            "has_negation",
            "length_short",
            "section_risk_factors",
        ]

        self.W1 = np.array(
            [
                [1.9, -0.6, -0.2, 1.2, -0.1, -0.2],
                [-0.6, 2.0, 0.7, -0.2, 1.1, 0.4],
                [-0.4, 1.4, 0.4, -0.2, 1.0, 0.3],
                [-0.2, 1.1, 0.2, -0.1, 0.9, 0.2],
                [1.0, -0.1, -0.1, 0.9, -0.1, -0.1],
                [-0.2, 0.7, 0.1, -0.1, 0.7, 0.1],
                [-0.2, 0.5, 1.8, -0.1, 0.2, 1.0],
                [0.8, -0.1, -0.1, 0.7, -0.1, -0.1],
                [-0.1, 0.8, 0.3, -0.1, 0.7, 0.2],
                [-0.1, 0.8, 0.2, -0.1, 0.7, 0.2],
                [-0.1, 0.3, 1.4, -0.1, 0.1, 0.8],
                [-0.3, 0.5, 0.3, -0.2, 0.5, 0.2],
                [-0.9, 0.8, 0.5, -0.7, 0.6, 0.4],
                [-0.1, 0.1, 0.2, -0.3, 0.2, 0.2],
                [-0.1, 0.9, 0.3, -0.1, 0.7, 0.2],
            ],
            dtype=float,
        )
        self.b1 = np.array([-0.6, -0.6, -0.6, -0.4, -0.4, -0.4], dtype=float)

        self.W2 = np.array(
            [
                [1.9, -0.6, -0.2, 1.0],
                [-0.5, 1.9, 0.4, -0.2],
                [-0.2, 0.4, 1.9, -0.1],
                [1.0, -0.2, -0.1, 0.8],
                [-0.2, 1.0, 0.2, -0.1],
                [-0.1, 0.2, 1.0, -0.1],
            ],
            dtype=float,
        )
        self.b2 = np.array([-0.5, -0.5, -0.5, -0.3], dtype=float)

        self.W3 = np.array(
            [
                [2.2, -0.7, -0.4],
                [-0.7, 2.2, 0.4],
                [-0.4, 0.3, 2.3],
                [1.0, -0.2, -0.1],
            ],
            dtype=float,
        )
        self.b3 = np.array([-0.8, -0.8, -0.8], dtype=float)

    def build_features(self, sentence: dict) -> SentenceFeatureBundle:
        tokens = sentence.get("tokens") or []
        lemmas = []
        for token in tokens:
            lemma = token.get("lemma") or token.get("text") or ""
            lemma = _normalize_lemma(lemma)
            if lemma:
                lemmas.append(lemma)

        aggregates = {
            "positive_sum": 0.0,
            "negative_sum": 0.0,
            "uncertainty_sum": 0.0,
            "litigious_sum": 0.0,
            "strong_modal_sum": 0.0,
            "weak_modal_sum": 0.0,
            "constraining_sum": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
            "uncertainty_count": 0.0,
            "cost_term_count": 0.0,
        }
        matched_terms = []

        for lemma in lemmas:
            scores = self.lexicon.lookup(lemma)
            if not scores:
                continue

            matched_categories = [
                name for name, value in scores.items() if value > 0
            ]
            if not matched_categories:
                continue

            aggregates["positive_sum"] += scores["positive"]
            aggregates["negative_sum"] += scores["negative"]
            aggregates["uncertainty_sum"] += scores["uncertainty"]
            aggregates["litigious_sum"] += scores["litigious"]
            aggregates["strong_modal_sum"] += scores["strong_modal"]
            aggregates["weak_modal_sum"] += scores["weak_modal"]
            aggregates["constraining_sum"] += scores["constraining"]
            if scores["positive"] > 0:
                aggregates["positive_count"] += 1
            if scores["negative"] > 0:
                aggregates["negative_count"] += 1
            if scores["uncertainty"] > 0:
                aggregates["uncertainty_count"] += 1
            if lemma in COST_TERMS and (scores["negative"] > 0 or scores["constraining"] > 0):
                aggregates["cost_term_count"] += 1

            matched_terms.append(
                {
                    "phrase": lemma,
                    "dimension": self._primary_dimension_for_term(lemma, scores),
                    "weight": float(sum(scores.values())),
                    "lm_categories": matched_categories,
                }
            )

        has_hedge = 1.0 if sentence.get("has_hedge") else 0.0
        has_negation = 1.0 if sentence.get("has_negation") else 0.0
        section = str(sentence.get("section", "")).lower()
        length_short = 1.0 if len(lemmas) < 8 else 0.0
        section_risk_factors = 1.0 if "risk" in section else 0.0

        vector = np.array(
            [
                aggregates["positive_sum"],
                aggregates["negative_sum"],
                aggregates["uncertainty_sum"],
                aggregates["litigious_sum"],
                aggregates["strong_modal_sum"],
                aggregates["weak_modal_sum"],
                aggregates["constraining_sum"],
                aggregates["positive_count"],
                aggregates["negative_count"],
                aggregates["uncertainty_count"],
                aggregates["cost_term_count"],
                has_hedge,
                has_negation,
                length_short,
                section_risk_factors,
            ],
            dtype=float,
        )

        return SentenceFeatureBundle(
            vector=vector,
            matched_terms=matched_terms,
            aggregates=aggregates,
        )

    def _primary_dimension_for_term(self, lemma: str, scores: dict[str, float]) -> str:
        if lemma in COST_TERMS and (scores["negative"] > 0 or scores["constraining"] > 0):
            return "cost_pressure"
        if scores["positive"] > 0 or scores["strong_modal"] > 0:
            return "growth"
        return "risk"

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h1 = _sigmoid(np.dot(x, self.W1) + self.b1)
        h2 = _sigmoid(np.dot(h1, self.W2) + self.b2)
        y = _sigmoid(np.dot(h2, self.W3) + self.b3)
        return h1, h2, y

    def score_sentence(self, sentence: dict) -> dict:
        bundle = self.build_features(sentence)
        h1, h2, outputs = self.forward(bundle.vector)

        growth_evidence = bundle.aggregates["positive_sum"] + 0.5 * bundle.aggregates["strong_modal_sum"]
        risk_evidence = (
            bundle.aggregates["negative_sum"]
            + bundle.aggregates["uncertainty_sum"]
            + 0.5 * bundle.aggregates["litigious_sum"]
            + 0.5 * bundle.aggregates["weak_modal_sum"]
        )
        cost_evidence = bundle.aggregates["cost_term_count"] + 0.5 * bundle.aggregates["constraining_sum"]
        evidence_strength = np.array([growth_evidence, risk_evidence, cost_evidence], dtype=float)

        if sentence.get("has_hedge"):
            evidence_strength *= 0.6
        if sentence.get("has_negation"):
            evidence_strength *= -1.0

        scores = outputs * (1.0 + evidence_strength)
        return {
            "feature_vector": bundle.vector,
            "hidden_layer_1": h1,
            "hidden_layer_2": h2,
            "output_vector": outputs,
            "matched_terms": bundle.matched_terms,
            "scores": {
                "growth": float(scores[0]),
                "risk": float(scores[1]),
                "cost_pressure": float(scores[2]),
            },
        }


BASE_DIR = Path(__file__).resolve().parent
MODEL = TwoHiddenLayerFinancialSignalModel(BASE_DIR)


def analyze_record(record: dict) -> dict:
    sentence_results = []
    term_totals: dict[tuple[str, str], dict] = {}
    totals = {"growth": 0.0, "risk": 0.0, "cost_pressure": 0.0}

    for sentence in _sentence_records(record):
        sentence_text = sentence.get("text", "")
        scored = MODEL.score_sentence(sentence)
        matched_terms = scored["matched_terms"]

        if not matched_terms:
            continue

        growth_score = scored["scores"]["growth"]
        risk_score = scored["scores"]["risk"]
        cost_score = scored["scores"]["cost_pressure"]

        totals["growth"] += growth_score
        totals["risk"] += risk_score
        totals["cost_pressure"] += cost_score

        for term_match in matched_terms:
            key = (term_match["dimension"], term_match["phrase"])
            if key not in term_totals:
                term_totals[key] = {
                    "dimension": term_match["dimension"],
                    "phrase": term_match["phrase"],
                    "score": 0.0,
                    "count": 0,
                    "evidence": [],
                }
            term_totals[key]["score"] += term_match["weight"]
            term_totals[key]["count"] += 1
            if len(term_totals[key]["evidence"]) < 3:
                term_totals[key]["evidence"].append(sentence_text)

        sentence_results.append(
            {
                "sent_id": sentence.get("sent_id"),
                "section": sentence.get("section", "unknown"),
                "text": sentence_text,
                "growth_score": round(growth_score, 3),
                "risk_score": round(risk_score, 3),
                "cost_pressure_score": round(cost_score, 3),
                "net_score": round(growth_score - risk_score, 3),
                "matched_phrases": matched_terms,
                "has_negation": bool(sentence.get("has_negation")),
                "has_hedge": bool(sentence.get("has_hedge")),
                "hidden_layer_1": [round(float(value), 3) for value in scored["hidden_layer_1"]],
                "hidden_layer_2": [round(float(value), 3) for value in scored["hidden_layer_2"]],
                "output_vector": [round(float(value), 3) for value in scored["output_vector"]],
            }
        )

    top_terms = sorted(
        term_totals.values(),
        key=lambda item: abs(item["score"]),
        reverse=True,
    )

    return {
        "ticker": record.get("ticker", "UNK"),
        "company_name": record.get("company_name", "Unknown Company"),
        "form_type": record.get("form_type", "UNKNOWN"),
        "filing_date": record.get("filing_date", "Unknown Date"),
        "source": record.get("source", "Unknown Source"),
        "model": {
            "type": "two_hidden_layer_mlp",
            "input_features": MODEL.feature_names,
            "hidden_layer_sizes": [6, 4],
            "output_dimensions": DIMENSIONS,
            "lexicon_source": MODEL.lexicon.source,
        },
        "scores": {
            "growth": round(totals["growth"], 2),
            "risk": round(totals["risk"], 2),
            "cost_pressure": round(totals["cost_pressure"], 2),
            "net_operating_signal": round(totals["growth"] - totals["risk"], 2),
        },
        "sentence_signals": sorted(
            sentence_results,
            key=lambda item: abs(item["net_score"]) + abs(item["cost_pressure_score"]),
            reverse=True,
        ),
        "top_phrases": top_terms,
    }


def _read_json_records_from_directory(directory: Path, pattern: str) -> list[dict]:
    records = []
    for path in sorted(directory.glob(pattern)):
        if path.name.startswith("_"):
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def infer_data_source(base_dir: Path) -> str:
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
        records = _read_json_records_from_directory(processed_dir, "*.processed.json")
        if records:
            return records

    pipeline_records = []
    filings_dir = base_dir / "data" / "filings"
    transcripts_dir = base_dir / "data" / "transcripts"
    if filings_dir.exists():
        pipeline_records.extend(_read_json_records_from_directory(filings_dir, "*.json"))
    if transcripts_dir.exists():
        pipeline_records.extend(_read_json_records_from_directory(transcripts_dir, "*.json"))
    if pipeline_records:
        return pipeline_records

    demo_path = base_dir / "demo_data" / "sample_documents.json"
    if demo_path.exists():
        return json.loads(demo_path.read_text(encoding="utf-8"))

    return []


def analyze_records(records: list[dict]) -> list[dict]:
    return [analyze_record(record) for record in records]


def build_comparison_rows(analyses: list[dict]) -> list[dict]:
    rows = []
    for analysis in analyses:
        rows.append(
            {
                "label": f"{analysis['ticker']} {analysis['filing_date']}",
                "ticker": analysis["ticker"],
                "filing_date": analysis["filing_date"],
                "growth": analysis["scores"]["growth"],
                "risk": analysis["scores"]["risk"],
                "cost_pressure": analysis["scores"]["cost_pressure"],
                "net_operating_signal": analysis["scores"]["net_operating_signal"],
            }
        )
    return rows
