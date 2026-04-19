from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


class TwoHiddenLayerFinancialSignalModel:
    """

    """

        self.feature_names = [
            "has_hedge",
            "has_negation",
            "length_short",
            "section_risk_factors",
        ]

        self.W1 = np.array(
            [
            ],
            dtype=float,
        )
        self.b1 = np.array([-0.6, -0.6, -0.6, -0.4, -0.4, -0.4], dtype=float)

        self.W2 = np.array(
            [
            ],
            dtype=float,
        )
        self.b2 = np.array([-0.5, -0.5, -0.5, -0.3], dtype=float)

        self.W3 = np.array(
            [
                [2.2, -0.7, -0.4],
                [-0.4, 0.3, 2.3],
                [1.0, -0.2, -0.1],
            ],
            dtype=float,
        )
        self.b3 = np.array([-0.8, -0.8, -0.8], dtype=float)

    def build_features(self, sentence: dict) -> SentenceFeatureBundle:


                continue
                {
                }
            )

        has_hedge = 1.0 if sentence.get("has_hedge") else 0.0
        has_negation = 1.0 if sentence.get("has_negation") else 0.0
        section = str(sentence.get("section", "")).lower()
        section_risk_factors = 1.0 if "risk" in section else 0.0

        vector = np.array(
            [
                has_hedge,
                has_negation,
                length_short,
                section_risk_factors,
            ],
            dtype=float,
        )

        return SentenceFeatureBundle(
            vector=vector,
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h1 = _sigmoid(np.dot(x, self.W1) + self.b1)
        h2 = _sigmoid(np.dot(h1, self.W2) + self.b2)
        y = _sigmoid(np.dot(h2, self.W3) + self.b3)
        return h1, h2, y

    def score_sentence(self, sentence: dict) -> dict:
        bundle = self.build_features(sentence)
        h1, h2, outputs = self.forward(bundle.vector)

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
            "scores": {
                "growth": float(scores[0]),
                "risk": float(scores[1]),
                "cost_pressure": float(scores[2]),
            },
        }




def analyze_record(record: dict) -> dict:
    sentence_results = []
    totals = {"growth": 0.0, "risk": 0.0, "cost_pressure": 0.0}

    for sentence in _sentence_records(record):
        sentence_text = sentence.get("text", "")
        scored = MODEL.score_sentence(sentence)

            continue

        growth_score = scored["scores"]["growth"]
        risk_score = scored["scores"]["risk"]
        cost_score = scored["scores"]["cost_pressure"]

        totals["growth"] += growth_score
        totals["risk"] += risk_score
        totals["cost_pressure"] += cost_score

                    "score": 0.0,
                    "count": 0,
                    "evidence": [],
                }

        sentence_results.append(
            {
                "sent_id": sentence.get("sent_id"),
                "section": sentence.get("section", "unknown"),
                "text": sentence_text,
                "growth_score": round(growth_score, 3),
                "risk_score": round(risk_score, 3),
                "cost_pressure_score": round(cost_score, 3),
                "net_score": round(growth_score - risk_score, 3),
                "has_negation": bool(sentence.get("has_negation")),
                "has_hedge": bool(sentence.get("has_hedge")),
                "hidden_layer_1": [round(float(value), 3) for value in scored["hidden_layer_1"]],
                "hidden_layer_2": [round(float(value), 3) for value in scored["hidden_layer_2"]],
                "output_vector": [round(float(value), 3) for value in scored["output_vector"]],
            }
        )

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
    }


def load_records(base_dir: Path) -> list[dict]:
    processed_dir = base_dir / "data" / "processed"
    if processed_dir.exists():
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
