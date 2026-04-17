# CS329 Financial Report Analyzer — Data Ingestion Module

## Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Master entrypoint — runs both collectors |
| `sec_edgar_collector.py` | Pulls 10-K, 10-Q filings from SEC EDGAR |
| `transcript_scraper.py` | Scrapes earnings call transcripts (Motley Fool / Kaggle) |
| `requirements.txt` | Python dependencies |

## Output Structure

```
data/
  filings/
    AAPL_10-K_2024-11-01.json
    AAPL_10-Q_2024-08-02.json
    _index.json               ← metadata only, no raw text
  transcripts/
    AAPL_EARNINGS_CALL_2024-11-01.json
    _index.json
  _master_index.json          ← combined index across all sources
```

## JSON Record Schema

Every saved file (filing or transcript) follows this schema:

```json
{
  "ticker":           "AAPL",
  "company_name":     "Apple Inc.",
  "cik":              "0000320193",
  "form_type":        "10-K",
  "filing_date":      "2024-11-01",
  "quarter":          "Q4",
  "year":             "2024",
  "source":           "SEC EDGAR",
  "source_url":       "https://...",
  "accession_number": "0000320193-24-000123",
  "raw_text":         "...",
  "text_length":      42815,
  "collected_at":     "2025-04-16T12:00:00Z"
}
```

## Quick Start

```bash
pip install -r requirements.txt

# Basic run — AAPL, MSFT, NVDA, JPM; 2 of each form type
python run_pipeline.py

# Custom tickers and more filings
python run_pipeline.py --tickers AAPL MSFT GOOGL META --forms 10-K 10-Q --max 4

# Use Kaggle transcript dataset instead of live scraping
python run_pipeline.py --kaggle-csv path/to/kaggle_transcripts.csv

# SEC only (no transcript scraping)
python run_pipeline.py --skip-transcripts

# Just transcripts
python run_pipeline.py --skip-sec
```

## SEC EDGAR Notes

- The SEC EDGAR API is free and public but **requires** a descriptive `User-Agent` header
  with your name/email. Update the `User-Agent` string in `sec_edgar_collector.py`.
- Rate limit: ~10 requests/second. The pipeline uses a 0.5s delay by default.
- Text extraction: downloads the primary document (HTM/HTML) and strips tags.
  Some filings may return `[TEXT EXTRACTION FAILED]` if the primary doc is XBRL-only —
  the accession number is still saved so you can revisit manually.

## Transcript Scraping Notes

- **Motley Fool** works with Selenium but is rate-limited and may change its DOM.
  Works best for 2–4 transcripts per ticker per run.
- **Seeking Alpha** requires a paid subscription — a login-based Selenium flow is
  documented as a stub in `transcript_scraper.py`.
- **Kaggle dataset** (`--kaggle-csv`) is the most reliable bulk source:
  https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts


