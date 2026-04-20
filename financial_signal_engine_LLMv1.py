"""
financial_signal_engine_LLMv1.py
 
This file implements a hybrid signal engine that combines the lexicon based
approach (v2) with a Claude LLM fallback for sentences where the lexicon
gets zero signal hits.
 
The motivation comes from the evaluation results:

We saw that the V2 engine only hit 30% of sentences (coverage_rate = 0.3). 
The other 70% defaulted to neutral with no real signal
An LLM can handle these zero hit sentences using contextual understanding
instead of keyword matching
 
How the hybrid works:
    1. Run every sentence through SignalEngineV2 first (this is fast and free)
    2. If the sentence got zero hits we send to Claude for classification
    3. Claude returns positive/negative/neutral + a one sentence reason
    4. Merge results: lexicon label where it had signal, LLM label where it didn't
 
This approach is efficient because we only call the API on sentences the lexicon
could not handle. This keeps costs low and makes the system explainable:

The explanations are something like:
"This sentence was scored by the lexicon because it matched X phrases"
"This sentence was scored by the LLM because the lexicon had no hits"
 
Evaluation:
    Run evaluate_hybrid.py to compare lexicon only vs hybrid side by side
    New evaluation file for this! Don't use the evaluate.py for this script
    Use normal eval file for V2, Use hybrid eval file for LLMV1 engine

"""


from __future__ import annotations
 
import logging
import os
import time
from pathlib import Path
import anthropic
from financial_signal_engine import LMDictionary, SentenceScore
from financial_signal_engine_v2 import SignalEngineV2
 
logger = logging.getLogger(__name__)
 



HYBRID_VERSION = "0.1.0"
 
#how many sentences to send in one API call to save on requests
BATCH_SIZE = 10
 
#Claude model to use, haiku is fast, cheap, good for classification
#appropriate for our task at hand
LLM_MODEL = "claude-haiku-4-5-20251001"
 
#label map back to scores for merging into the pipeline
LLM_LABEL_TO_NET: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral":  0.0,
}
 
 
#LLM classifier
 
#here it calls Claude to classify financial sentences as positive, negative, 
#or neutral, only used when lexicon has ZERO hits on a sentence
class LLMClassifier:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key directly."
            )
        self.client = anthropic.Anthropic(api_key=key)
 
#classify batch of sentences in single api call, returns list of dicts with keys
#sentence, label, reason
    def classify_batch(self, sentences: list[str]) -> list[dict]:
        
        if not sentences:
            return []
 
        # number them so the model can reference them clearly
        numbered = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(sentences)
        )
 
        prompt = f"""You are a financial sentiment classifier. Classify each sentence below as exactly one of: positive, negative, or neutral.
 
Rules:
- positive: the sentence reports growth, profit increases, improved performance, raised guidance, or other clearly good financial news
- negative: the sentence reports losses, declining revenue, risk factors, cost increases, or clearly bad financial news  
- neutral: the sentence is factual with no clear positive or negative financial signal
 
For each sentence respond with ONLY this format (one per line):
<number>|<label>|<one sentence reason>
 
Sentences to classify:
{numbered}"""
 
        try:
            message = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text
            return self._parse_response(sentences, response_text)
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            
            
            #if API fails, return neutral for all sentences in the batch
            return [
                {"sentence": s, "label": "neutral", "reason": "api_error"}
                for s in sentences
            ]
 
#parse the numbered response back into list of results
    def _parse_response(
        self, sentences: list[str], response_text: str
    ) -> list[dict]:
        results = []
        lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
 
        parsed: dict[int, tuple[str, str]] = {}
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 3:
                try:
                    idx = int(parts[0].strip()) - 1
                    label = parts[1].strip().lower()
                    reason = parts[2].strip()
                    if label in ("positive", "negative", "neutral"):
                        parsed[idx] = (label, reason)
                except (ValueError, IndexError):
                    continue
 

        for i, sentence in enumerate(sentences):
            if i in parsed:
                label, reason = parsed[i]
            else:
                #if model doe not return this sentence, default to neutral
                label, reason = "neutral", "parse_error"
            results.append({
                "sentence": sentence,
                "label": label,
                "reason": reason,
            })
        return results
 
 




#Hybrid engine
#this combines SignalEngineV2 with LLM fallback
#description below
#LLM only called when Lexicon doesnt have result, keeps it fast and efficient
class HybridSignalEngine:
    """ 
    For each sentence:
        - IF lexicon has at least one signal hit THEN use lexicon score
        - IF lexicon has zero hits THEN send to LLM for classification
    """
 
    def __init__(self, lm: LMDictionary, api_key: str | None = None):
        self.lexicon_engine = SignalEngineV2(lm)
        self.llm = LLMClassifier(api_key=api_key)
        self.stats = {
            "total": 0,
            "lexicon_hits": 0,
            "llm_fallback": 0,
            "llm_positive": 0,
            "llm_negative": 0,
            "llm_neutral": 0,
        }
 

#score s single sentence, returns dict with all score field
    def score_sentence_hybrid(self, sentence: dict) -> dict:
        result = self.lexicon_engine.score_sentence(sentence)
        has_hit = (
            result is not None and (
                result.lm_growth_hits or result.lm_risk_hits
                or result.lm_uncertainty_hits or result.phrase_growth_hits
                or result.phrase_risk_hits or result.phrase_cost_hits
            )
        )
        self.stats["total"] += 1
 
        if has_hit:
            self.stats["lexicon_hits"] += 1
            return {
                "text": sentence.get("text", ""),
                "method": "lexicon",
                "label": self._net_to_label(result.net_score),
                "net_score": result.net_score,
                "growth": result.growth,
                "risk": result.risk,
                "cost_pressure": result.cost_pressure,
                "llm_reason": None,
            }

#zero hits so will be handled by LLM in batch
        else:
            self.stats["llm_fallback"] += 1
            return {
                "text": sentence.get("text", ""),
                "method": "llm_pending",
                "label": "neutral",
                "net_score": 0.0,
                "growth": 0.0,
                "risk": 0.0,
                "cost_pressure": 0.0,
                "llm_reason": None,
            }
 

#here we score a batch of sentence dicts, it handles llm calls efficiently and 
#does so by batching all zero hit sentences into grouped API calls as described earlier
    def score_batch(self, sentences: list[dict]) -> list[dict]:
    
        #first pass, lexicon scoring
        results = [self.score_sentence_hybrid(s) for s in sentences]
 
        #collect indices that need LLM
        llm_indices = [i for i, r in enumerate(results) if r["method"] == "llm_pending"]
 
        if not llm_indices:
            return results
 
        #batch the LLM calls
        logger.info(
            "Sending %d zero-hit sentences to LLM (batches of %d)",
            len(llm_indices), BATCH_SIZE,
        )
        llm_texts = [results[i]["text"] for i in llm_indices]
 






#i researched delay, not sure if this is correct, come back
        llm_results: list[dict] = []
        for start in range(0, len(llm_texts), BATCH_SIZE):
            batch = llm_texts[start: start + BATCH_SIZE]
            llm_results.extend(self.llm.classify_batch(batch))
            if start + BATCH_SIZE < len(llm_texts):
                time.sleep(0.5)




#merge LLM results back
        for idx, llm_result in zip(llm_indices, llm_results):
            label = llm_result["label"]
            results[idx]["method"] = "llm"
            results[idx]["label"] = label
            results[idx]["net_score"] = LLM_LABEL_TO_NET[label]
            results[idx]["llm_reason"] = llm_result["reason"]
            self.stats[f"llm_{label}"] += 1
 
        return results
 
    @staticmethod
    def _net_to_label(net_score: float, threshold: float = 0.1) -> str:
        if net_score > threshold:
            return "positive"
        if net_score < -threshold:
            return "negative"
        return "neutral"
 
    def log_stats(self) -> None:
        t = self.stats["total"] or 1
        logger.info(
            "Hybrid stats: %d total | lexicon=%d (%.1f%%) | llm=%d (%.1f%%)",
            self.stats["total"],
            self.stats["lexicon_hits"],
            100 * self.stats["lexicon_hits"] / t,
            self.stats["llm_fallback"],
            100 * self.stats["llm_fallback"] / t,
        )
        logger.info(
            "LLM breakdown: positive=%d | negative=%d | neutral=%d",
            self.stats["llm_positive"],
            self.stats["llm_negative"],
            self.stats["llm_neutral"],
        )
 