"""Quick validation of the preprocessing pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from text_preprocessor import (
    FinancialTextCleaner,
    SECSectionSegmenter,
    TranscriptSegmenter,
    FinancialNLPProcessor,
    PreprocessingPipeline,
)

# -------------------------------------------------------------
# 1. Cleaner: exercise unicode + boilerplate
# -------------------------------------------------------------
cleaner = FinancialTextCleaner()
dirty = (
    "Page 3 of 47\n\n"
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
    "Washington, D.C. 20549\n\n"
    "We expect \u201cstrong\u201d revenue growth \u2014 margins may remain\n"
    "under pressure. us-gaap:Revenues  ............... 12\n\n\n\n"
    "Table of Contents"
)
cleaned = cleaner.clean(dirty)
print("=== CLEANER ===")
print(repr(cleaned))
assert "Page 3 of 47" not in cleaned
assert "\u201c" not in cleaned  # smart quotes normalized
assert "us-gaap:Revenues" not in cleaned
assert "strong" in cleaned
print("  [OK] cleaner normalizes unicode and strips boilerplate")

# -------------------------------------------------------------
# 2. SEC segmenter: synthetic 10-K with Item 1A and Item 7
# -------------------------------------------------------------
mock_10k = """
PART I

Item 1. Business
We design and sell consumer electronics and services.

Item 1A. Risk Factors
Our business is subject to numerous risks. Supply chain disruptions could
materially affect our operations. Foreign currency fluctuations may impact
our reported revenues in international markets.

Item 2. Properties
We own and lease facilities globally.

PART II

Item 7. Management's Discussion and Analysis of Financial Condition and
Results of Operations

We delivered strong performance in fiscal 2024. Revenue grew 8% year over
year driven by services. We expect continued momentum in fiscal 2025 as we
launch new products. Operating margin may compress due to higher input costs.

Item 8. Financial Statements
See index to consolidated financial statements.
"""
seg = SECSectionSegmenter()
sections = seg.segment(mock_10k, "10-K")
print("\n=== SEC SEGMENTER ===")
print(f"  MD&A ({len(sections.mdna)} chars): {sections.mdna[:100]!r}...")
print(f"  Risk Factors ({len(sections.risk_factors)} chars): {sections.risk_factors[:100]!r}...")
print(f"  Forward Guidance ({len(sections.forward_guidance)} chars): {sections.forward_guidance!r}")
assert "strong performance" in sections.mdna
assert "Supply chain disruptions" in sections.risk_factors
assert "expect continued momentum" in sections.forward_guidance.lower() or \
       "expect" in sections.forward_guidance.lower()
# Must NOT leak Item 8 content
assert "See index to consolidated" not in sections.mdna
print("  [OK] MD&A, Risk Factors, Forward Guidance correctly segmented")

# -------------------------------------------------------------
# 3. Transcript segmenter
# -------------------------------------------------------------
mock_transcript = """
Operator: Good afternoon and welcome to the Q3 2024 earnings call.

Tim Cook -- Chief Executive Officer
Thank you. We had a strong quarter with revenue growth across all segments.

Luca Maestri -- Chief Financial Officer
Gross margin was 46%, up 100 basis points year over year.

We will now open the line for questions.

Amit Daryanani -- Evercore ISI -- Analyst
Thanks for taking my question. Can you comment on iPhone demand in China?

Tim Cook -- Chief Executive Officer
We saw strong demand in China, although we remain cautious about the
macroeconomic environment.

Wamsi Mohan -- Bank of America -- Analyst
What about services growth going forward?

Luca Maestri -- Chief Financial Officer
We expect services to continue growing at a healthy rate next quarter.
"""
tseg = TranscriptSegmenter()
tsec = tseg.segment(mock_transcript)
print("\n=== TRANSCRIPT SEGMENTER ===")
print(f"  Prepared remarks: {len(tsec.prepared_remarks)} chars")
print(f"  Q&A turns: {len(tsec.qa_section)}")
for t in tsec.qa_section:
    print(f"    [{t['role']}] {t['speaker']} ({t['title']}): {t['text'][:60]!r}...")
assert len(tsec.qa_section) >= 3, f"expected >=3 turns, got {len(tsec.qa_section)}"
roles = [t["role"] for t in tsec.qa_section]
assert "Q" in roles and "A" in roles, "missing Q or A role classification"
# prepared remarks should NOT contain the analyst lines
assert "Amit Daryanani" not in tsec.prepared_remarks
print("  [OK] prepared/Q&A split + speaker role classification works")

# -------------------------------------------------------------
# 4. NLP processor (will use blank('en') fallback here; real tests need model)
# -------------------------------------------------------------
print("\n=== NLP PROCESSOR ===")
nlp_proc = FinancialNLPProcessor(model="en_core_web_sm")
test_text = (
    "We expect revenue growth to accelerate next quarter. "
    "However, margins may remain under pressure. "
    "We do not anticipate any supply chain disruptions."
)
records, next_sid = nlp_proc.process_section(test_text, "mdna", sent_id_start=0)
print(f"  Processed {len(records)} sentences from test text")
for r in records:
    print(f"    sent_id={r['sent_id']} neg={r['has_negation']} "
          f"hedge={r['has_hedge']} tokens={len(r['tokens'])}")
    print(f"      text: {r['text']!r}")
    print(f"      masked: {r['text_masked']!r}")

# Verify negation detection: sentence 3 has "do not" -> should flag
assert records[2]["has_negation"] is True, \
    f"Expected negation in 'We do not anticipate...', got {records[2]}"
# Verify hedge detection: "expect" is a hedge lemma -> sentence 1 should flag
# (only works if lemmatizer is available; blank('en') gives lemma == text)
# so we check either sentence 1 ('expect') or sentence 2 ('may')
any_hedge = any(r["has_hedge"] for r in records)
assert any_hedge, "Expected at least one sentence flagged as hedged"
print("  [OK] negation + hedge flags fire as expected")

# -------------------------------------------------------------
# 5. End-to-end pipeline on a mock record
# -------------------------------------------------------------
print("\n=== END-TO-END PIPELINE ===")
mock_record = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "cik": "0000320193",
    "form_type": "10-K",
    "filing_date": "2024-11-01",
    "quarter": "Q4",
    "year": "2024",
    "source": "SEC EDGAR",
    "source_url": "https://www.sec.gov/...",
    "accession_number": "0000320193-24-000123",
    "raw_text": mock_10k,
    "text_length": len(mock_10k),
    "collected_at": "2025-04-16T12:00:00Z",
}
pipe = PreprocessingPipeline()
out = pipe.process_record(mock_record)
print("  Keys in output record:", sorted(out.keys()))
assert "processed" in out
assert "raw_text" in out  # original preserved
processed = out["processed"]
print(f"  Sections: {list(processed['sections'].keys())}")
print(f"  Total sentences: {processed['stats']['n_sentences']}")
print(f"  Tokens: {processed['stats']['n_tokens']}")
print(f"  Sentences by section: {processed['stats']['sentences_by_section']}")
print(f"  Preprocessing version: {processed['stats']['preprocessing_version']}")
assert processed["stats"]["n_sentences"] > 0
assert "mdna" in processed["stats"]["sentences_by_section"]
print("  [OK] end-to-end pipeline produces expected structure")

# -------------------------------------------------------------
# 6. Verify JSON serializability (important -- the output must write to disk)
# -------------------------------------------------------------
serialized = json.dumps(out, indent=2, ensure_ascii=False)
assert len(serialized) > 0
roundtrip = json.loads(serialized)
assert roundtrip["ticker"] == "AAPL"
assert len(roundtrip["processed"]["sentences"]) == processed["stats"]["n_sentences"]
print("\n  [OK] output round-trips through JSON cleanly "
      f"({len(serialized):,} chars)")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
