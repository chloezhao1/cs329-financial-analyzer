"""
Financial Report Analyzer — Data Ingestion Pipeline
CS329 Computational Linguistics, Group 7

Orchestrates:
  1. SEC EDGAR filing collection (10-K, 10-Q)
  2. Earnings call transcript collection (Motley Fool / Kaggle)

Output schema (each record saved as JSON):
  {
    "ticker":          "AAPL",
    "company_name":    "Apple Inc.",
    "cik":             "0000320193",        # SEC filings only
    "form_type":       "10-K",              # or EARNINGS_CALL
    "filing_date":     "2024-11-01",
    "quarter":         "Q4",                # transcripts only
    "year":            "2024",              # transcripts only
    "source":          "SEC EDGAR",
    "source_url":      "https://...",
    "accession_number":"0000320193-24-...", # SEC filings only
    "raw_text":        "...",
    "text_length":     42000,
    "collected_at":    "2025-04-16T12:00:00Z"
  }

Usage:
    python run_pipeline.py
    python run_pipeline.py --tickers AAPL MSFT NVDA JPM --forms 10-K 10-Q --max 2
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, date

from sec_edgar_collector import collect_sec_filings
from transcript_scraper import collect_transcripts
from text_preprocessor import PreprocessingPipeline


BASE_DIR = Path(__file__).resolve().parent


def run_full_pipeline(
    tickers: list[str],
    form_types: list[str],
    max_per_type: int,
    kaggle_pkl: str = None,
    skip_sec: bool = False,
    skip_transcripts: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
):
    start = datetime.utcnow()
    print("\n" + "█"*60)
    print("  CS329 FINANCIAL REPORT ANALYZER — DATA INGESTION")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Forms   : {', '.join(form_types)}")
    print(f"  Max/type: {max_per_type}")
    print("█"*60)

    all_records = []

    # ── Step 1: SEC EDGAR ──────────────────────────────────────────
    if not skip_sec:
        print("\n[STEP 1] SEC EDGAR Filing Collection")
        sec_records = collect_sec_filings(
            tickers=tickers,
            form_types=form_types,
            max_per_type=max_per_type,
            start_date=start_date,
            end_date=end_date,
        )
        all_records.extend(sec_records)
    else:
        print("\n[STEP 1] Skipping SEC EDGAR (--skip-sec)")

    # ── Step 2: Earnings Call Transcripts ─────────────────────────
    if not skip_transcripts:
        print("\n[STEP 2] Earnings Call Transcript Collection")
        transcript_records = collect_transcripts(
            tickers=tickers,
            max_per_ticker=max_per_type,
            kaggle_pkl_path=kaggle_pkl,
            start_date=start_date,
            end_date=end_date,
        )
        all_records.extend(transcript_records)
    else:
        print("\n[STEP 2] Skipping transcripts (--skip-transcripts)")

    # ── Summary ────────────────────────────────────────────────────
    elapsed = (datetime.utcnow() - start).total_seconds()

    print("\n" + "="*60)
    print("  COLLECTION SUMMARY")
    print("="*60)

    by_type = {}
    by_ticker = {}
    failed = 0

    for r in all_records:
        ft = r["form_type"]
        tk = r["ticker"]
        by_type[ft] = by_type.get(ft, 0) + 1
        by_ticker[tk] = by_ticker.get(tk, 0) + 1
        if r.get("raw_text", "").startswith("[TEXT EXTRACTION FAILED]"):
            failed += 1

    for ft, count in sorted(by_type.items()):
        print(f"  {ft:<20} {count} records")

    print()
    for tk, count in sorted(by_ticker.items()):
        print(f"  {tk:<10} {count} documents")

    print(f"\n  Total   : {len(all_records)} documents")
    print(f"  Failed  : {failed} text extractions")
    print(f"  Elapsed : {elapsed:.1f}s")

    # Write combined master index
    master_index_path = BASE_DIR / "data" / "_master_index.json"
    master_index_path.parent.mkdir(exist_ok=True)
    index = [{k: v for k, v in r.items() if k != "raw_text"} for r in all_records]
    with open(master_index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  Master index → {master_index_path}")
    print("="*60 + "\n")

    # ── Step 3: Preprocessing (with disk cache) ────────────────────
    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP 3] Preprocessing collected records (cache: data/processed/)...")
    preprocessor = None
    processed_records = []
    cached = 0

    for record in all_records:
        ticker = record.get("ticker", "UNK")
        form = record.get("form_type", "UNK").replace("/", "-")
        date_str = record.get("filing_date", "unknown")
        cache_path = processed_dir / f"{ticker}_{form}_{date_str}.processed.json"

        if cache_path.exists():
            try:
                processed_records.append(json.loads(cache_path.read_text(encoding="utf-8")))
                cached += 1
                continue
            except Exception:
                pass

        if preprocessor is None:
            preprocessor = PreprocessingPipeline()

        try:
            result = preprocessor.process_record(record)
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            processed_records.append(result)
        except Exception as e:
            print(f"  WARNING: preprocessing failed for {ticker} {form}: {e}")
            processed_records.append(record)

    print(f"  {cached} from cache, {len(processed_records) - cached} newly processed\n")

    return processed_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS329 Financial Data Ingestion Pipeline")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "JPM"],
                        help="Stock tickers to collect")
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"],
                        help="SEC form types (10-K, 10-Q)")
    parser.add_argument("--max", type=int, default=2,
                        help="Max filings per form type per ticker")
    parser.add_argument("--kaggle-pkl", type=str, default=None,
                        help="Path to Kaggle transcripts .pkl file")
    parser.add_argument("--skip-sec", action="store_true",
                        help="Skip SEC EDGAR collection")
    parser.add_argument("--skip-transcripts", action="store_true",
                        help="Skip earnings transcript collection")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Optional start date in YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Optional end date in YYYY-MM-DD")

    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    run_full_pipeline(
        tickers=args.tickers,
        form_types=args.forms,
        max_per_type=args.max,
        kaggle_pkl=args.kaggle_pkl,
        skip_sec=args.skip_sec,
        skip_transcripts=args.skip_transcripts,
        start_date=start_date,
        end_date=end_date,
    )
