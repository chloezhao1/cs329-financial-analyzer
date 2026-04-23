"""
financial_signal_engine_LLMpure.py
CS329 Financial Report Analyzer — Pure LLM Signal Engine

This engine classifies every sentence using only a Claude LLM call.
There is no Loughran-McDonald dictionary lookup, no manual phrase matching,
and no FinBERT embeddings. All sentiment understanding is delegated entirely
to the language model.

Design motivation:
    V1 through V3 all rely on a lexicon tier as the primary or first-pass
    classifier. That means ~33% of sentences are always scored by keyword
    matching, which is fast but brittle. This engine asks: what happens if
    we let the LLM handle everything? It should generalize better to novel
    phrasing and implicit signals but will be slower and more expensive.

Architecture:
    Single tier: Claude (Haiku by default, configurable) classifies every
    sentence in batches. The model returns:
        - label: positive | negative | neutral
        - score: continuous float in [-1.0, 1.0] representing sentiment strength
        - reason: one-sentence explanation

    Output dict format matches other engines for drop-in comparison.

Usage:
    engine = PureLLMSignalEngine(api_key="sk-ant-...")
    results = engine.score_batch(sentences)  # sentences: list of dicts with "text"

Installation:
    pip install anthropic
"""

from __future__ import annotations

import logging
import os
import time

import anthropic

logger = logging.getLogger(__name__)

PURE_LLM_VERSION = "0.1.0"

LLM_MODEL = "claude-haiku-4-5-20251001"

# How many sentences per API call. Larger batches are more token-efficient
# but produce longer outputs that are harder to parse reliably.
BATCH_SIZE = 10

LABEL_TO_NET: dict[str, float] = {
    "positive":  1.0,
    "negative": -1.0,
    "neutral":   0.0,
}

SYSTEM_PROMPT = """\
You are a financial sentiment analyst. Your job is to classify sentences
extracted from corporate earnings reports, 10-Ks, press releases, and
financial news into one of three sentiment categories:

  positive — the sentence conveys good financial news: revenue growth,
             profit increases, raised guidance, improved margins, strong
             demand, successful product launches, or any other clearly
             favourable financial signal.

  negative — the sentence conveys bad financial news: revenue declines,
             losses, cost increases, downgraded guidance, market risk,
             regulatory issues, or any other clearly unfavourable signal.

  neutral  — the sentence is purely factual with no clear directional
             financial signal, or discusses context without positive or
             negative implication.

You must also provide a continuous sentiment score from -1.0 (strongly
negative) to 1.0 (strongly positive). Use intermediate values to reflect
the strength of the signal — a sentence with 0.9 is more bullish than one
with 0.3, even if both are labelled positive. Neutral sentences should be
near 0.0. Use the full range: avoid clustering everything at -1, 0, or 1.

Respond ONLY in this exact format, one line per sentence:
<number>|<label>|<score>|<one-sentence reason>

where <score> is a decimal like 0.75 or -0.40 (no extra text around it).
Do not include any other text, headers, or explanation outside of these lines.\
"""


class PureLLMSignalEngine:
    """
    Pure LLM signal engine — every sentence is classified by Claude.
    No lexicon matching, no embeddings, no manual rules.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model:   Claude model ID. Defaults to Haiku for cost/speed.
        batch_size: sentences per API call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = LLM_MODEL,
        batch_size: int = BATCH_SIZE,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.batch_size = batch_size

        self.stats: dict[str, int] = {
            "total": 0,
            "llm_positive": 0,
            "llm_negative": 0,
            "llm_neutral": 0,
            "api_errors": 0,
            "parse_errors": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_batch(self, sentences: list[dict]) -> list[dict]:
        """
        Score a list of sentence dicts through the LLM.

        Args:
            sentences: list of dicts, each must have a "text" key.

        Returns:
            list of result dicts with keys:
                text, method, label, net_score, score, reason
        """
        if not sentences:
            return []

        texts = [s.get("text", "") for s in sentences]
        results: list[dict] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            batch_results = self._classify_batch(batch)
            results.extend(batch_results)
            if start + self.batch_size < len(texts):
                time.sleep(0.3)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_batch(self, sentences: list[str]) -> list[dict]:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        user_message = f"Classify these sentences:\n\n{numbered}"

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            response_text = message.content[0].text
            return self._parse_response(sentences, response_text)
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            self.stats["api_errors"] += len(sentences)
            return [self._fallback_result(s) for s in sentences]

    def _parse_response(
        self, sentences: list[str], response_text: str
    ) -> list[dict]:
        parsed: dict[int, tuple[str, float, str]] = {}
        for line in response_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0].strip()) - 1
                label = parts[1].strip().lower()
                score = float(parts[2].strip())
                reason = parts[3].strip()
                if label not in ("positive", "negative", "neutral"):
                    label = "neutral"
                # clamp score to [-1, 1]
                score = max(-1.0, min(1.0, score))
                parsed[idx] = (label, score, reason)
            except (ValueError, IndexError):
                self.stats["parse_errors"] += 1
                continue

        results = []
        for i, sentence in enumerate(sentences):
            if i in parsed:
                label, score, reason = parsed[i]
            else:
                self.stats["parse_errors"] += 1
                label, score, reason = "neutral", 0.0, "parse_error"

            self.stats["total"] += 1
            self.stats[f"llm_{label}"] += 1

            results.append({
                "text": sentence,
                "method": "llm_pure",
                "label": label,
                "net_score": score,   # continuous [-1, 1]
                "score": score,
                "reason": reason,
                # these keys keep the format compatible with other engines
                "growth": max(0.0, score),
                "risk": max(0.0, -score),
                "cost_pressure": 0.0,
                "embedding_confidence": None,
                "llm_reason": reason,
            })
        return results

    @staticmethod
    def _fallback_result(sentence: str) -> dict:
        return {
            "text": sentence,
            "method": "llm_pure_error",
            "label": "neutral",
            "net_score": 0.0,
            "score": 0.0,
            "reason": "api_error",
            "growth": 0.0,
            "risk": 0.0,
            "cost_pressure": 0.0,
            "embedding_confidence": None,
            "llm_reason": "api_error",
        }

    def log_stats(self) -> None:
        t = self.stats["total"] or 1
        logger.info(
            "PureLLM stats: %d total | positive=%d (%.1f%%) | "
            "negative=%d (%.1f%%) | neutral=%d (%.1f%%) | "
            "api_errors=%d | parse_errors=%d",
            self.stats["total"],
            self.stats["llm_positive"],
            100 * self.stats["llm_positive"] / t,
            self.stats["llm_negative"],
            100 * self.stats["llm_negative"] / t,
            self.stats["llm_neutral"],
            100 * self.stats["llm_neutral"] / t,
            self.stats["api_errors"],
            self.stats["parse_errors"],
        )
