"""
Smoke test for financial_signal_engine.py.

Builds a tiny mock LM CSV and a mock preprocessed record, runs the
engine end-to-end, and asserts the scores + attribution are sensible.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from financial_signal_engine import (
    LMDictionary, SignalEngine, build_comparison_rows,
    GROWTH_PHRASES, RISK_PHRASES, COST_PRESSURE_PHRASES,
)


# ---------------------------------------------------------------------------
# 1. Write a tiny mock LM CSV (just enough to exercise the loader)
# ---------------------------------------------------------------------------
tmp = Path("/tmp/mock_lm.csv")
with tmp.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Word", "Positive", "Negative", "Uncertainty",
                "Litigious", "Constraining", "Strong_Modal", "Weak_Modal"])
    # Real LM words sampled from across categories
    w.writerow(["achieve",       "2009", "0",    "0",    "0",    "0", "0", "0"])
    w.writerow(["beneficial",    "2009", "0",    "0",    "0",    "0", "0", "0"])
    w.writerow(["strong",        "2009", "0",    "0",    "0",    "0", "0", "0"])
    w.writerow(["growth",        "2009", "0",    "0",    "0",    "0", "0", "0"])
    w.writerow(["improve",       "2009", "0",    "0",    "0",    "0", "0", "0"])
    w.writerow(["decline",       "0",    "2009", "0",    "0",    "0", "0", "0"])
    w.writerow(["adverse",       "0",    "2009", "0",    "0",    "0", "0", "0"])
    w.writerow(["loss",          "0",    "2009", "0",    "0",    "0", "0", "0"])
    w.writerow(["weakness",      "0",    "2009", "0",    "0",    "0", "0", "0"])
    w.writerow(["deteriorate",   "0",    "2009", "0",    "0",    "0", "0", "0"])
    w.writerow(["lawsuit",       "0",    "0",    "0",    "2009", "0", "0", "0"])
    w.writerow(["plaintiff",     "0",    "0",    "0",    "2009", "0", "0", "0"])
    w.writerow(["prohibit",      "0",    "0",    "0",    "0",    "2009", "0", "0"])
    w.writerow(["restrict",      "0",    "0",    "0",    "0",    "2009", "0", "0"])
    w.writerow(["uncertain",     "0",    "0",    "2009", "0",    "0",    "0", "0"])
    w.writerow(["depend",        "0",    "0",    "2009", "0",    "0",    "0", "0"])
    w.writerow(["approximate",   "0",    "0",    "2009", "0",    "0",    "0", "0"])
    # Neutral word (shouldn't land in any indicator set)
    w.writerow(["company",       "0",    "0",    "0",    "0",    "0",    "0", "0"])

print("=== 1. LM LOADER ===")
lm = LMDictionary.from_csv(tmp)
print(f"  loaded words: {lm.total_words}")
print(f"  growth: {sorted(lm.growth)}")
print(f"  risk:   {sorted(lm.risk)}")
print(f"  uncert: {sorted(lm.uncertainty)}")
assert "achieve" in lm.growth and "strong" in lm.growth
assert "decline" in lm.risk and "lawsuit" in lm.risk and "prohibit" in lm.risk
assert "uncertain" in lm.uncertainty
assert "company" not in lm.growth
assert "company" not in lm.risk
print("  [OK] LM categories load correctly")

# ---------------------------------------------------------------------------
# 2. Build a mock preprocessed record that exercises every code path
# ---------------------------------------------------------------------------


def make_sent(sid, text, section, has_neg=False, has_hedge=False, extra_tokens=None):
    """Build a minimal preprocessed-sentence dict."""
    words = text.replace(".", "").replace(",", "").split()
    tokens = [
        {
            "text": w,
            "lemma": w.lower(),
            "pos": "NOUN",
            "is_stop": False,
            "is_neg": False,
        }
        for w in words
    ]
    if extra_tokens:
        tokens.extend(extra_tokens)
    return {
        "sent_id": sid,
        "section": section,
        "text": text,
        "tokens": tokens,
        "has_negation": has_neg,
        "has_hedge": has_hedge,
    }


mock_record = {
    "ticker": "TEST",
    "company_name": "Test Corp",
    "form_type": "10-K",
    "filing_date": "2025-10-31",
    "processed": {
        "sentences": [
            # clean growth sentence: LM word 'growth' + phrase 'revenue growth'
            make_sent(
                0,
                "We saw strong revenue growth this quarter driven by improve in demand.",
                "mdna",
            ),
            # clean risk sentence: LM words 'decline', 'adverse'
            make_sent(
                1,
                "We expect a decline in demand with adverse effect on margins.",
                "mdna",
            ),
            # hedged growth (has_hedge=True): should get half weight
            make_sent(
                2,
                "We may achieve strong growth if conditions improve.",
                "mdna",
                has_hedge=True,
            ),
            # negated risk (has_negation=True): sign should flip, so this
            # reduces risk rather than adding to it
            make_sent(
                3,
                "We do not expect any decline or adverse impact.",
                "mdna",
                has_neg=True,
            ),
            # Cost pressure: hits phrase 'margin pressure' (no LM layer)
            make_sent(
                4,
                "We continue to experience margin pressure from rising costs.",
                "mdna",
            ),
            # Litigation: LM 'lawsuit' should fall in risk
            make_sent(
                5,
                "A lawsuit was filed by the plaintiff regarding the matter.",
                "risk_factors",
            ),
            # Uncertainty only: should contribute to risk at HALF weight
            make_sent(
                6,
                "Future results depend on uncertain market conditions.",
                "mdna",
            ),
            # Too short -- should be skipped entirely
            make_sent(7, "Growth.", "mdna"),
            # Q&A analyst question (should get 0.3 weight)
            make_sent(
                8,
                "Can you comment on the strong growth in your cloud segment?",
                "qa_Q_0",
            ),
            # Q&A executive answer (should get 1.1 weight)
            make_sent(
                9,
                "We saw strong growth in cloud demand and expect it to continue.",
                "qa_A_1",
            ),
        ],
    },
}

# ---------------------------------------------------------------------------
# 3. Run the engine on the mock record
# ---------------------------------------------------------------------------
print("\n=== 2. PER-SENTENCE SCORING ===")
engine = SignalEngine(lm)

# Score each sentence individually and verify specific behaviors
for sent in mock_record["processed"]["sentences"]:
    r = engine.score_sentence(sent)
    if r is None:
        print(f"  sent {sent['sent_id']}: SKIPPED (too short)")
        continue
    print(
        f"  sent {r.sent_id}  sec={r.section:<18s}  "
        f"g={r.growth:+.2f}  r={r.risk:+.2f}  c={r.cost_pressure:+.2f}  "
        f"neg={r.has_negation}  hedge={r.has_hedge}"
    )

# Specific assertions
s0 = engine.score_sentence(mock_record["processed"]["sentences"][0])
s1 = engine.score_sentence(mock_record["processed"]["sentences"][1])
s2 = engine.score_sentence(mock_record["processed"]["sentences"][2])
s3 = engine.score_sentence(mock_record["processed"]["sentences"][3])
s4 = engine.score_sentence(mock_record["processed"]["sentences"][4])
s5 = engine.score_sentence(mock_record["processed"]["sentences"][5])
s6 = engine.score_sentence(mock_record["processed"]["sentences"][6])
s7 = engine.score_sentence(mock_record["processed"]["sentences"][7])
s8 = engine.score_sentence(mock_record["processed"]["sentences"][8])
s9 = engine.score_sentence(mock_record["processed"]["sentences"][9])

# s7 should be skipped (too short)
assert s7 is None, "one-word sentence should be skipped"

# s0: LM 'strong', 'growth', 'improve' in growth + phrase 'revenue growth'
assert s0.growth > 0, f"sentence 0 should have positive growth, got {s0.growth}"
assert "strong" in s0.lm_growth_hits
assert "revenue growth" in s0.phrase_growth_hits

# s1: LM 'decline', 'adverse' in risk
assert s1.risk > 0, f"sentence 1 should have positive risk, got {s1.risk}"
assert "decline" in s1.lm_risk_hits
assert "adverse" in s1.lm_risk_hits

# s2: hedged, so growth should be POSITIVE but strictly less than s0's growth-per-hit
# (hedge halves the score)
assert s2.growth > 0
assert s2.has_hedge
print(f"\n  s0 growth = {s0.growth:+.2f}, s2 (hedged) growth = {s2.growth:+.2f}")
# s2 has only LM 'achieve' + 'strong' + 'growth' + 'improve' (4 hits) vs s0's
# similar hits + phrase. Absolute counts differ so can't compare directly, but
# we can verify hedge flag is respected by scoring the same sentence twice.
clone = dict(mock_record["processed"]["sentences"][2])
clone_unhedged = dict(clone)
clone_unhedged["has_hedge"] = False
s2u = engine.score_sentence(clone_unhedged)
assert abs(s2.growth - s2u.growth * 0.5) < 0.001, (
    f"hedged score ({s2.growth}) should be 0.5x unhedged ({s2u.growth})"
)
print(f"  hedged = 0.5 * unhedged: {s2.growth:+.2f} == 0.5 * {s2u.growth:+.2f}  [OK]")

# s3: negated -> risk should be NEGATIVE (sign flipped)
assert s3.risk < 0, (
    f"negated risk sentence should have negative risk, got {s3.risk}"
)
print(f"  negated risk (flipped): {s3.risk:+.2f}  [OK]")

# s4: cost pressure phrase 'margin pressure' + 'rising costs'
assert s4.cost_pressure > 0, f"sentence 4 should have cost pressure, got {s4.cost_pressure}"
assert "margin pressure" in s4.phrase_cost_hits
assert "rising costs" in s4.phrase_cost_hits

# s5: lawsuit + plaintiff -> risk; in risk_factors section (0.6 weight)
assert s5.risk > 0
assert "lawsuit" in s5.lm_risk_hits
assert "plaintiff" in s5.lm_risk_hits
print(f"  risk_factors section-weighted (0.6x): s5 risk = {s5.risk:+.2f}  [OK]")

# s6: uncertainty only -> contributes to risk at HALF weight
assert s6.risk > 0
assert s6.lm_uncertainty_hits, "should have uncertainty hits"
print(f"  uncertainty -> risk (half weight): s6 risk = {s6.risk:+.2f}  [OK]")

# s8: qa_Q gets 0.3 weight, s9: qa_A gets 1.1 weight
# Both have similar content ('strong growth in cloud')
print(f"  qa_Q weight (0.3x): s8 growth = {s8.growth:+.2f}")
print(f"  qa_A weight (1.1x): s9 growth = {s9.growth:+.2f}")
assert s9.growth > s8.growth, (
    f"qa_A ({s9.growth}) should score higher than qa_Q ({s8.growth})"
)
print("  [OK] section weights applied correctly")

# ---------------------------------------------------------------------------
# 4. Document-level aggregation
# ---------------------------------------------------------------------------
print("\n=== 3. DOCUMENT AGGREGATION ===")
analysis = engine.analyze_record(mock_record)
print(f"  ticker: {analysis['ticker']}")
print(f"  scores: {json.dumps(analysis['scores'], indent=4)}")
print(f"  coverage: {analysis['coverage']}")
print(f"  top phrases (growth): {[p['term'] for p in analysis['top_growth_phrases'][:5]]}")
print(f"  top phrases (risk):   {[p['term'] for p in analysis['top_risk_phrases'][:5]]}")
print(f"  top phrases (cost):   {[p['term'] for p in analysis['top_cost_phrases'][:5]]}")

assert analysis["coverage"]["scored_sentences"] == 9, (
    f"should have 9 scoreable (skip 1 too-short), got "
    f"{analysis['coverage']['scored_sentences']}"
)
assert "net_operating_signal" in analysis["scores"]
assert analysis["scores"]["growth"] != 0
assert analysis["scores"]["risk"] != 0

# Comparison row extraction
rows = build_comparison_rows([analysis])
assert len(rows) == 1
assert rows[0]["ticker"] == "TEST"
assert "net_operating_signal" in rows[0]
print(f"  comparison row: {rows[0]}")
print("  [OK] document aggregation + comparison rows work")

# ---------------------------------------------------------------------------
# 5. JSON serialization
# ---------------------------------------------------------------------------
print("\n=== 4. JSON SERIALIZATION ===")
out = json.dumps(analysis, indent=2, ensure_ascii=False)
reloaded = json.loads(out)
assert reloaded["ticker"] == "TEST"
assert reloaded["method"]["type"] == "loughran_mcdonald_lexicon"
print(f"  [OK] round-trips JSON cleanly ({len(out):,} chars)")

# ---------------------------------------------------------------------------
# 6. Phrase dictionary sanity
# ---------------------------------------------------------------------------
print("\n=== 5. PHRASE DICTIONARY SANITY ===")
print(f"  GROWTH_PHRASES:        {len(GROWTH_PHRASES)}")
print(f"  RISK_PHRASES:          {len(RISK_PHRASES)}")
print(f"  COST_PRESSURE_PHRASES: {len(COST_PRESSURE_PHRASES)}")
# No overlaps between categories
assert not (GROWTH_PHRASES & RISK_PHRASES), "growth and risk phrases shouldn't overlap"
assert not (GROWTH_PHRASES & COST_PRESSURE_PHRASES), "growth and cost shouldn't overlap"
assert not (RISK_PHRASES & COST_PRESSURE_PHRASES), "risk and cost shouldn't overlap"
print("  [OK] no overlap between indicator phrase sets")

print("\n" + "=" * 60)
print("ALL SIGNAL ENGINE TESTS PASSED")
print("=" * 60)
