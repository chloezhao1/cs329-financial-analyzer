from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GROWTH_PHRASES = {
    "revenue growth": 2.6,
    "strong demand": 2.2,
    "market share gains": 2.1,
    "raised guidance": 2.4,
    "record revenue": 2.5,
    "margin expansion": 1.8,
    "accelerating demand": 2.2,
    "improved outlook": 2.0,
    "grew": 1.0,
    "growth": 1.0,
    "expand": 1.0,
    "accelerate": 1.2,
}

RISK_PHRASES = {
    "macroeconomic uncertainty": 2.5,
    "demand weakness": 2.4,
    "foreign exchange headwinds": 2.2,
    "supply chain disruption": 2.2,
    "regulatory pressure": 2.0,
    "competitive pressure": 1.9,
    "customer softness": 1.8,
    "weaker spending": 1.9,
    "uncertainty": 1.0,
    "risk": 0.9,
    "headwind": 1.2,
    "decline": 1.2,
    "softness": 1.1,
}

COST_PHRASES = {
    "margin pressure": 2.6,
    "higher costs": 2.0,
    "cost inflation": 2.4,
    "input cost inflation": 2.6,
    "labor costs": 1.8,
    "operating expenses increased": 2.1,
    "pricing pressure": 1.8,
    "elevated freight costs": 2.1,
    "cost pressure": 1.8,
    "expenses": 0.8,
    "cost": 0.7,
    "inflation": 1.1,
}

DIMENSION_LEXICONS = {
    "growth": GROWTH_PHRASES,
    "risk": RISK_PHRASES,
    "cost_pressure": COST_PHRASES,
}

DIMENSIONS = ["growth", "risk", "cost_pressure"]
NEGATION_WORDS = {"not", "no", "never", "without", "unlikely"}
HEDGE_WORDS = {"may", "might", "could", "expect", "anticipate", "likely", "approximately"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tokenize_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


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
            "has_negation": any(word in _normalize_text(sentence).split() for word in NEGATION_WORDS),
            "has_hedge": any(word in _normalize_text(sentence).split() for word in HEDGE_WORDS),
        }
        for idx, sentence in enumerate(_tokenize_sentences(raw_text))
    ]


@dataclass
class SentenceFeatureBundle:
    vector: np.ndarray
    matched_phrases: list[dict]
    weighted_hits: dict[str, float]
    counts: dict[str, int]


class TwoHiddenLayerFinancialSignalModel:
    """
    A small two-hidden-layer MLP-style model.

    Input features are Loughran-McDonald-inspired lexicon matches and sentence-level
    modifiers such as negation and hedging. The two hidden layers learn intermediate
    concepts like optimism, caution, and expense stress before mapping them to
    growth, risk, and cost-pressure outputs.
    """

    def __init__(self) -> None:
        self.feature_names = [
            "growth_weight",
            "risk_weight",
            "cost_weight",
            "growth_count",
            "risk_count",
            "cost_count",
            "has_hedge",
            "has_negation",
            "length_short",
            "section_risk_factors",
        ]

        # Layer 1 learns broad latent concepts from LM-based inputs.
        self.W1 = np.array(
            [
                [1.8, -0.5, -0.3, 1.0, -0.2, -0.2],
                [-0.5, 1.9, 0.2, -0.2, 1.0, 0.3],
                [-0.2, 0.3, 1.9, -0.1, 0.2, 1.1],
                [0.9, -0.2, -0.1, 0.8, -0.1, -0.1],
                [-0.1, 0.9, 0.1, -0.1, 0.8, 0.1],
                [-0.1, 0.1, 0.9, -0.1, 0.1, 0.8],
                [-0.4, 0.5, 0.3, -0.3, 0.6, 0.2],
                [-1.0, 0.8, 0.5, -0.8, 0.6, 0.4],
                [-0.2, 0.1, 0.1, -0.4, 0.2, 0.2],
                [-0.2, 1.0, 0.4, -0.2, 0.8, 0.3],
            ],
            dtype=float,
        )
        self.b1 = np.array([-0.6, -0.6, -0.6, -0.4, -0.4, -0.4], dtype=float)

        # Layer 2 combines those concepts into cleaner financial postures.
        self.W2 = np.array(
            [
                [1.8, -0.6, -0.2, 1.0],
                [-0.4, 1.9, 0.1, -0.2],
                [-0.2, 0.2, 1.9, -0.1],
                [1.1, -0.3, -0.1, 0.8],
                [-0.2, 1.1, 0.2, -0.2],
                [-0.1, 0.2, 1.1, -0.1],
            ],
            dtype=float,
        )
        self.b2 = np.array([-0.5, -0.5, -0.5, -0.3], dtype=float)

        # Final outputs are independent sigmoid heads so one sentence can carry
        # multiple signals at once, which fits the project better than softmax.
        self.W3 = np.array(
            [
                [2.2, -0.7, -0.4],
                [-0.7, 2.3, 0.3],
                [-0.4, 0.3, 2.3],
                [1.0, -0.2, -0.1],
            ],
            dtype=float,
        )
        self.b3 = np.array([-0.8, -0.8, -0.8], dtype=float)

    def build_features(self, sentence: dict) -> SentenceFeatureBundle:
        text = sentence.get("text", "")
        normalized = _normalize_text(text)

        weighted_hits = {dimension: 0.0 for dimension in DIMENSIONS}
        counts = {dimension: 0 for dimension in DIMENSIONS}
        matched_phrases = []

        for dimension, lexicon in DIMENSION_LEXICONS.items():
            for phrase, weight in lexicon.items():
                if not _contains_phrase(normalized, phrase):
                    continue
                weighted_hits[dimension] += weight
                counts[dimension] += 1
                matched_phrases.append(
                    {
                        "phrase": phrase,
                        "dimension": dimension,
                        "weight": weight,
                    }
                )

        has_hedge = 1.0 if sentence.get("has_hedge") else 0.0
        has_negation = 1.0 if sentence.get("has_negation") else 0.0
        section = str(sentence.get("section", "")).lower()
        length_short = 1.0 if len(normalized.split()) < 8 else 0.0
        section_risk_factors = 1.0 if "risk" in section else 0.0

        vector = np.array(
            [
                weighted_hits["growth"],
                weighted_hits["risk"],
                weighted_hits["cost_pressure"],
                float(counts["growth"]),
                float(counts["risk"]),
                float(counts["cost_pressure"]),
                has_hedge,
                has_negation,
                length_short,
                section_risk_factors,
            ],
            dtype=float,
        )

        return SentenceFeatureBundle(
            vector=vector,
            matched_phrases=matched_phrases,
            weighted_hits=weighted_hits,
            counts=counts,
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h1 = _sigmoid(np.dot(x, self.W1) + self.b1)
        h2 = _sigmoid(np.dot(h1, self.W2) + self.b2)
        y = _sigmoid(np.dot(h2, self.W3) + self.b3)
        return h1, h2, y

    def score_sentence(self, sentence: dict) -> dict:
        bundle = self.build_features(sentence)
        h1, h2, outputs = self.forward(bundle.vector)

        # Use lexicon evidence to scale the neural outputs so the model stays grounded
        # in explicit LM-style phrase matches and remains explainable.
        evidence_strength = np.array(
            [
                bundle.weighted_hits["growth"],
                bundle.weighted_hits["risk"],
                bundle.weighted_hits["cost_pressure"],
            ],
            dtype=float,
        )
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
            "matched_phrases": bundle.matched_phrases,
            "scores": {
                "growth": float(scores[0]),
                "risk": float(scores[1]),
                "cost_pressure": float(scores[2]),
            },
        }


MODEL = TwoHiddenLayerFinancialSignalModel()


def analyze_record(record: dict) -> dict:
    sentence_results = []
    phrase_totals: dict[tuple[str, str], dict] = {}
    totals = {"growth": 0.0, "risk": 0.0, "cost_pressure": 0.0}

    for sentence in _sentence_records(record):
        sentence_text = sentence.get("text", "")
        scored = MODEL.score_sentence(sentence)
        matched_phrases = scored["matched_phrases"]

        if not matched_phrases:
            continue

        growth_score = scored["scores"]["growth"]
        risk_score = scored["scores"]["risk"]
        cost_score = scored["scores"]["cost_pressure"]

        totals["growth"] += growth_score
        totals["risk"] += risk_score
        totals["cost_pressure"] += cost_score

        for phrase_match in matched_phrases:
            key = (phrase_match["dimension"], phrase_match["phrase"])
            if key not in phrase_totals:
                phrase_totals[key] = {
                    "dimension": phrase_match["dimension"],
                    "phrase": phrase_match["phrase"],
                    "score": 0.0,
                    "count": 0,
                    "evidence": [],
                }
            phrase_totals[key]["score"] += phrase_match["weight"]
            phrase_totals[key]["count"] += 1
            if len(phrase_totals[key]["evidence"]) < 3:
                phrase_totals[key]["evidence"].append(sentence_text)

        sentence_results.append(
            {
                "sent_id": sentence.get("sent_id"),
                "section": sentence.get("section", "unknown"),
                "text": sentence_text,
                "growth_score": round(growth_score, 3),
                "risk_score": round(risk_score, 3),
                "cost_pressure_score": round(cost_score, 3),
                "net_score": round(growth_score - risk_score, 3),
                "matched_phrases": matched_phrases,
                "has_negation": bool(sentence.get("has_negation")),
                "has_hedge": bool(sentence.get("has_hedge")),
                "hidden_layer_1": [round(float(value), 3) for value in scored["hidden_layer_1"]],
                "hidden_layer_2": [round(float(value), 3) for value in scored["hidden_layer_2"]],
                "output_vector": [round(float(value), 3) for value in scored["output_vector"]],
            }
        )

    top_phrases = sorted(
        phrase_totals.values(),
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
        "top_phrases": top_phrases,
    }


def load_records(base_dir: Path) -> list[dict]:
    processed_dir = base_dir / "data" / "processed"
    if processed_dir.exists():
        records = []
        for path in sorted(processed_dir.glob("*.processed.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        if records:
            return records

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
