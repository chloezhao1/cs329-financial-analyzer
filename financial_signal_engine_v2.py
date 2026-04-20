"""
financial_signal_engine_v2.py
CS329 Financial Report Analyzer,  Signal Extraction Engine v2
Owner: Riyaa
Changes on v1 (financial_signal_engine.py, owned by Nick):

Used the evaluate.py to understand bottlenecks and weak points, here are
the changes made summarized based on that.
 
    1. Expanded GROWTH_PHRASES: the biggest issue from evaluation was that
       the engine missed past tense verb constructions like "profit rose" or
       "sales increased." Added ~30 phrases directly based on the error
       sample from evaluate.py.
 
    2. Lowered MIN_TOKENS_FOR_SCORING from 5 to 3: short but meaningful
       sentences like "Sales have risen" were being skipped entirely and
       defaulting to neutral. This will help a lot with the coverage issue 
       in version 1. A substantial amount of no hits came from sentences
       being too short.
 
    3. Fixed negation logic: the original code flipped all scores when
       has_negation was True which caused "not a risk" to accidentally boost
       growth. 
       New logic: negation zeroes out growth if the sentence has growth
       hits, and zeroes out risk if it has risk hits, instead of flipping signs.
 
    4. Removed overly aggressive positive blocklist words: "benefit",
       "favorable", and "valuable" are genuinely positive in financial context
       and were previously excluded.
 
Since we have two versions

Directions to use:
    from financial_signal_engine_v2 import SignalEngineV2, GROWTH_PHRASES_V2
    from financial_signal_engine import LMDictionary
    lm = LMDictionary.from_csv(Path("data/lexicons/loughran_mcdonald.csv"))
    engine = SignalEngineV2(lm)
 
To run evaluation against v2:
    python3 evaluate.py  (after swapping the engine import in evaluate.py)
"""





from __future__ import annotations
 
import logging
from pathlib import Path
from dataclasses import dataclass, field
 
from financial_signal_engine import (
    LMDictionary,
    SentenceScore,
    RISK_PHRASES,
    COST_PRESSURE_PHRASES,
    PHRASE_WEIGHT,
    UNCERTAINTY_RISK_WEIGHT,
    _section_weight,
    _sentence_lemmas,
    _find_phrases,
)
 
logger = logging.getLogger(__name__)
 
ENGINE_VERSION_V2 = "0.2.0-v2"
 
#change 1: lower the minimum token threshol because original was 5 which was too aggressive, short sentences 
#ike "Sales have risen" were being skipped entirely


MIN_TOKENS_FOR_SCORING_V2 = 3
 



#CHANGE 2: expanded growth phrase list. This is based directly on the error sample from evaluate.py
#positive sentence used a past-tense verb that wasn't in the original list. The original list had 15 phrases, 
#mostly forward looking earnings call language. These additions cover the factual reporting style 
#common in PhraseBank.
 
GROWTH_PHRASES_V2: set[str] = {
    #original phrases (Nick's)
    "revenue growth", "net sales increased", "strong demand",
    "market share gains", "record quarter", "accelerating growth",
    "accelerating demand", "raised guidance", "margin expansion",
    "outperformed", "exceeded expectations", "increased adoption",
    "expanded footprint", "robust demand", "double-digit growth",
 

    #new: past-tense profit/income phrases, most common in error sample
    "operating profit rose",
    "operating profit increased",
    "net profit rose",
    "net profit increased",
    "net profit went up",
    "profit rose",
    "profit increased",
    "profit doubled",
    "profit tripled",
    "profit surged",
    "earnings doubled",
    "earnings increased",
    "earnings grew",
 

    #new: sales/revenue verb phrases
    "sales increased",
    "sales rose",
    "sales surged",
    "sales have risen",
    "revenue increased",
    "revenue rose",
    "net sales surged",
    "net sales rose",
    "net sales doubled",
    "income rose",
    "income increased",
    "income doubled",
    "lending volume rose",
    "commission income increased",
 

    #new: share/market phrases seen in errors
    "shares rose",
    "shares closed higher",
    "shares were up",
    "beat analysts",
    "beat expectations",
    "market share increased",
    "subscriber base increased",
    "order book",
    #new: general upward movement
    "went up",
    "up from",
    "year-on-year",
    "raised payout",
    "raised dividend",
    "proposed dividend",
    "demand increased",
    "prices by",
}
 
#CHANGE 4: remove overly aggressive blocklist entries
#"benefit", "favorable", "valuable" are genuinely positive in financial text
 



LM_POSITIVE_BLOCKLIST_V2: set[str] = {
    "able", "effective", "effectively", "efficient", "efficiently",
    "successful", "successfully", "success", "succeed", "achievable",
    "satisfactory", "satisfactorily", "adequate", "adequately",
    "complete", "completed", "completing", "resolved", "resolve",
    #removed: "benefit", "beneficial", "favorable", "favorably",
    #"valuable", "reward", "rewarded", "enjoy", "enjoyed"
}
 






 
#CHANGE 3: improved negation logic
#Original: has_negation flips ALL scores (growth, risk, cost become negative)
#Issue: "not a risk" flipped risk to -risk, then growth - (-risk) inflated
#the net score, accidentally making the sentence look positive
#So, here the negation zeroes out the dimension it applies to instead of flipping it

def _apply_negation_v2(
    growth: float,
    risk: float,
    cost: float,
    has_neg: bool,
    lm_growth_hits: list,
    lm_risk_hits: list,
    ph_growth_hits: list,
    ph_risk_hits: list,
) -> tuple[float, float, float]:
    if not has_neg:
        return growth, risk, cost
    has_growth_signal = bool(lm_growth_hits or ph_growth_hits)
    has_risk_signal = bool(lm_risk_hits or ph_risk_hits)
    if has_growth_signal:
        growth = 0.0
    if has_risk_signal:
        risk = 0.0
    # cost pressure negation still zeroes it out
    cost = 0.0
    return growth, risk, cost
 
 




#V2 engine 
 
class SignalEngineV2:
    """
    New signal engine drops in as a replacement for the first SignalEngine.
    Only overrides score_sentence, analyze_record is inherited behavior
    replicated here to use the updated scorer
    """
 


    def __init__(self, lm: LMDictionary):
        #rebuild growth set using the updated blocklist
        self.lm = LMDictionary(
            growth = lm.growth - LM_POSITIVE_BLOCKLIST_V2,
            risk = lm.risk,
            uncertainty = lm.uncertainty,
            total_words =lm.total_words,
        )
        removed = lm.growth & LM_POSITIVE_BLOCKLIST_V2
        if removed:
            logger.info(
                "V2: updated positive blocklist removed %d words from growth",
                len(removed),
            )
 
    def score_sentence(self, sentence: dict) -> SentenceScore | None:
        tokens = sentence.get("tokens", [])
        #fix 2: lower threshold from 5 to 3
        if len(tokens) < MIN_TOKENS_FOR_SCORING_V2:
            return None
 

        text = sentence.get("text", "")
        text_lower = text.lower()
        lemmas = _sentence_lemmas(sentence)
        section = sentence.get("section", "")
        has_neg = bool(sentence.get("has_negation"))
        has_hedge = bool(sentence.get("has_hedge"))
 
        lm_growth = sorted(lemmas & self.lm.growth)
        lm_risk = sorted(lemmas & self.lm.risk)
        lm_uncert = sorted(lemmas & self.lm.uncertainty)
 






        #fix 1: use expanded phrase list
        ph_growth = _find_phrases(text_lower, GROWTH_PHRASES_V2)
        ph_risk = _find_phrases(text_lower, RISK_PHRASES)
        ph_cost = _find_phrases(text_lower, COST_PRESSURE_PHRASES)
 
        growth = len(lm_growth) + len(ph_growth) * PHRASE_WEIGHT
        risk = (len(lm_risk)
                  + len(lm_uncert) * UNCERTAINTY_RISK_WEIGHT
                  + len(ph_risk) * PHRASE_WEIGHT)
        cost = len(ph_cost) * PHRASE_WEIGHT
 

        #fix 3:negation, zero out the affected dimension
        growth, risk, cost = _apply_negation_v2(
            growth, risk, cost, has_neg,
            lm_growth, lm_risk, ph_growth, ph_risk,
        )
 
        if has_hedge:
            growth *= 0.5
            risk *= 0.5
            cost *= 0.5
 
        w = _section_weight(section)
        growth *= w
        risk *= w
        cost *= w
 



        return SentenceScore(
            sent_id = sentence.get("sent_id", -1),
            section = section,
            text = text,
            growth = growth,
            risk =risk,
            cost_pressure = cost,
            net_score =growth - risk,
            has_negation = has_neg,
            has_hedge = has_hedge,
            lm_growth_hits = lm_growth,
            lm_risk_hits = lm_risk,
            lm_uncertainty_hits = lm_uncert,
            phrase_growth_hits = ph_growth,
            phrase_risk_hits = ph_risk,
            phrase_cost_hits = ph_cost,
        )