# Text Preprocessing — Handoff to Signal Extraction

**Owner (preprocessing):** Oliver
**Consumer (signal extraction):** Nick/Riyaa/Pranav
**Status:** Validated on real AAPL 10-K and 10-Q. Transcript path tested on synthetic input only — real-data validation pending.

---

## TL;DR

The preprocessing pipeline takes the raw JSON records that the scraper produces and adds a `processed` field containing clean text, section segmentation, and per-sentence NLP features (tokens, lemmas, POS tags, noun phrases, negation/hedging flags, entity-masked text). You read one JSON per filing/transcript and get back everything you need to do phrase-level signal extraction against the Loughran-McDonald lexicon without touching the raw HTML-stripped text.

---

## How to run

```bash
# One-time setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Test that the pipeline is healthy
python smoke_test.py

# Process everything currently in data/filings and data/transcripts
python run_preprocessing.py

# Handy flags while iterating:
python run_preprocessing.py --limit 1       # process one file
python run_preprocessing.py --overwrite     # reprocess existing outputs
python run_preprocessing.py --only-form 10-K
```

Output lands in `data/processed/` with one file per record plus `_index.json` summarizing them all.

---

## The output schema

Every processed file is a superset of the scraper's original record. All original fields (`ticker`, `company_name`, `form_type`, `filing_date`, `raw_text`, etc.) are preserved. A new top-level key `processed` is appended:

```
{
  ...all original scraper fields...,
  "processed": {
    "sections": {...},
    "sentences": [...],
    "stats": {...}
  }
}
```

### `processed.sections`

For SEC filings (`form_type` in `{"10-K", "10-Q", "8-K"}`):

| key | content |
|---|---|
| `mdna` | Full body of Management's Discussion and Analysis (Item 7 for 10-K, Item 2 for 10-Q). String. |
| `risk_factors` | Full body of Risk Factors (Item 1A). String. For 10-Qs this is usually ~200 chars of cross-reference boilerplate. For 10-Ks it's tens of thousands of chars. |
| `forward_guidance` | Sentences from MD&A that contain forward-looking cue phrases ("expect", "anticipate", "may", "outlook", "next quarter", etc.). String, space-joined. |

For transcripts:

| key | content |
|---|---|
| `prepared_remarks` | The executive presentation part of the call. String. |
| `qa_section` | Q&A turns. List of `{"speaker", "title", "role", "text"}` where `role` is `"Q"` (analyst) or `"A"` (exec). |

### `processed.sentences`

A flat list of sentence records across all sections. This is what you'll iterate over for signal extraction. Each record:

| field | type | meaning |
|---|---|---|
| `sent_id` | int | Unique within the document, assigned in processing order. |
| `section` | str | Which section the sentence came from. One of `mdna`, `risk_factors`, `forward_guidance`, `prepared_remarks`, or `qa_Q_N` / `qa_A_N` for transcripts. |
| `text` | str | The original sentence, as spaCy segmented it. |
| `text_masked` | str | Same sentence with named entities replaced by placeholders: `<MONEY>`, `<PCT>`, `<DATE>`, `<NUM>`, `<ORG>`, `<PERSON>`, `<LOC>`. Rest is lowercased. Use this for cross-company comparison. |
| `tokens` | list | Per-token features (see below). |
| `noun_phrases` | list of str | Noun-phrase chunks, lowercased. Example: `["revenue growth", "foreign currencies", "supply chain"]`. |
| `has_negation` | bool | True if the sentence contains a negation particle or negspaCy flagged an entity as negated. |
| `has_hedge` | bool | True if the sentence contains a modal/uncertainty word from Loughran-McDonald's hedge list (`may`, `expect`, `approximately`, `likely`, etc.). |

Each token in `tokens` has:

| field | type | meaning |
|---|---|---|
| `text` | str | Original surface form. |
| `lemma` | str | Lemmatized form, lowercased. This is what to match against Loughran-McDonald. |
| `pos` | str | Universal POS tag (`NOUN`, `VERB`, `ADJ`, etc.). |
| `is_stop` | bool | True if it's a stopword. **Note:** negation particles are explicitly *not* marked as stopwords. |
| `is_neg` | bool | True if the token itself is a negation word. |

### `processed.stats`

| field | meaning |
|---|---|
| `n_sentences` | Total sentences across all sections. |
| `n_tokens` | Total non-space tokens. |
| `n_negated_sentences` | Count where `has_negation` is true. |
| `n_hedged_sentences` | Count where `has_hedge` is true. |
| `sentences_by_section` | Dict mapping section name to sentence count. |
| `preprocessing_version` | Current version is `"0.1.0"`. Bump this if the schema changes so downstream code can detect stale outputs. |

---

## Worked example

Here's a minimal loop showing how you'd read a processed file and do lexicon-based scoring:

```python
import json
from pathlib import Path

# Load the lexicon (you'll build your own dict from Loughran-McDonald CSV)
GROWTH_LEMMAS = {"grow", "increase", "expand", "accelerate", "strong", "record"}
RISK_LEMMAS   = {"decline", "headwind", "disrupt", "weakness", "pressure"}

def score_record(path: Path) -> dict:
    d = json.loads(path.read_text())
    growth = risk = 0
    for sent in d["processed"]["sentences"]:
        # Skip very short sentences (likely subsection headers, not prose)
        if len(sent["tokens"]) < 6:
            continue
        lemmas = {t["lemma"] for t in sent["tokens"] if not t["is_stop"]}
        g_hits = len(lemmas & GROWTH_LEMMAS)
        r_hits = len(lemmas & RISK_LEMMAS)
        # Negated sentences flip sign; hedged sentences get half weight
        sign   = -1 if sent["has_negation"] else 1
        weight = 0.5 if sent["has_hedge"] else 1.0
        growth += sign * weight * g_hits
        risk   += sign * weight * r_hits
    return {
        "ticker":        d["ticker"],
        "date":          d["filing_date"],
        "growth_signal": growth,
        "risk_signal":   risk,
        "net":           growth - risk,
    }

for p in Path("data/processed").glob("*.processed.json"):
    print(score_record(p))
```

This is just an illustrative starter — real scoring will use the full Loughran-McDonald dictionary and phrase-level matching over `noun_phrases`, not only token-level lemma hits.

---

## What you can rely on

- **Every sentence has stable `sent_id`** that's consistent within a document, so you can map signals back to source sentences for explainability ("this growth signal came from sentences 42, 78, 103").
- **`text_masked` is comparable across companies.** If Apple says "$4.2B in Q3 2024" and Microsoft says "$18.9B in FY25", both become `<MONEY> in <DATE>` so your phrase patterns are company-agnostic.
- **Negation and hedging are already flagged** — you do not need to re-detect them. Use the boolean flags to weight scoring appropriately.
- **Lemmas are lowercase and normalized.** Matching `{"grow", "growth", "grew", "growing"}` against the Loughran-McDonald lexicon is a single lookup on `tokens[i]["lemma"]`.

## What you cannot rely on

- **`noun_phrases` is best-effort.** spaCy's noun-chunk detector sometimes merges adjacent chunks ("asia pacific rest" from a section header bleeding into prose) or misses rare multi-word terms. Don't treat absence as meaningful.
- **Short "sentences" sneak in.** Subsection headers like `"Business Seasonality and Product Introductions"` end up as sentences because they lack terminal punctuation. Filter `len(tokens) < 5 or < 6` if this matters for your scoring.
- **Entity masking has some false positives.** spaCy tags `"iPhone"` as `<ORG>`. If product-level signal detection matters to you, you'll need to post-process or use `text` instead of `text_masked` for those cases.
- **Section boundaries are heuristic.** They're accurate on AAPL 10-K/10-Q but other filers may have edge cases. If you see a record with `mdna` of only a few hundred characters, ping me — it's likely a segmentation miss worth fixing upstream.

## Known limitations

1. **Transcript path is validated on synthetic data only.** Real Motley Fool transcripts likely have formatting quirks (inline ads, footer legalese, speaker-line variants) that haven't been tested. Expect minor issues when you first process real transcripts — tell me and we'll tune.
2. **Hedge list is abbreviated.** Using ~30 most common hedge words. The full Loughran-McDonald modal/uncertainty list is ~300 terms. Adding the full list is straightforward if hedge-precision matters for scoring.
3. **`negspacy` only fires on entities.** Pure sentence-level negation ("We do not expect growth") is caught by the lemma-level fallback, which is less precise than syntactic scope detection. Good enough for lexicon scoring; not good enough for fine-grained NLI.
4. **Document-level context is lost.** Each sentence stands alone. Cross-sentence phenomena (anaphora, topic continuation) aren't tracked. The pitch already flagged this under "risks" — mitigate with phrase-level pattern matching rather than single-token scoring.

---

## If something breaks

- **Segmentation looks wrong on a specific filing:** Use the diagnostic pattern from my dev notes — dump every `_ITEM_HEADER` match with surrounding context and see what the regex is picking up. Ping me, we'll tune.
- **Lots of sentences, but signals look noisy:** Check whether your scoring is filtering short sentences (the subsection-header issue) and whether you're respecting the hedge/negation flags.
- **Want a new NLP feature added:** Tell me the shape you need. Adding a field to each sentence record is a ~10-line change, rerunning preprocessing takes ~10 seconds per filing.

---

## Files

| file | purpose |
|---|---|
| `text_preprocessor.py` | All the classes (cleaner, segmenters, NLP processor, pipeline). |
| `run_preprocessing.py` | Batch runner. CLI entry point. |
| `smoke_test.py` | Self-contained test suite. Run this first if anything behaves oddly. |
| `data/processed/_index.json` | Summary of every processed file with per-record stats. |
| `data/processed/*.processed.json` | One enriched record per filing/transcript. |