"""
SEC EDGAR Filing Collector
Pulls 10-K, 10-Q, and 8-K filings via the SEC EDGAR full-text search API.
Saves structured JSON records with company name, date, report type, and raw text.
"""

import requests
import json
import time
import re
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# ─── Output Directory ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "filings"

HEADERS = {
    "User-Agent": "CS329-FinancialReportAnalyzer chloe.zhao@emory.edu",  # REQUIRED by SEC
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/full-index/"


# ─── Ticker → CIK Lookup ─────────────────────────────────────────────────────

def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Resolve a stock ticker to its SEC CIK number."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
    # Use the company tickers JSON maintained by SEC
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(tickers_url, headers={
        "User-Agent": "CS329-FinancialReportAnalyzer your-email@emory.edu"
    })
    if resp.status_code != 200:
        print(f"  [!] Could not fetch ticker list: {resp.status_code}")
        return None

    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            print(f"  [+] Resolved {ticker} → CIK {cik} ({entry['title']})")
            return cik

    print(f"  [!] Ticker {ticker} not found in SEC database.")
    return None


# ─── Filing List Retrieval ────────────────────────────────────────────────────

def _within_date_range(
    filing_date: str,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    try:
        parsed = datetime.strptime(filing_date, "%Y-%m-%d").date()
    except Exception:
        return True
    if start_date and parsed < start_date:
        return False
    if end_date and parsed > end_date:
        return False
    return True


def get_filings_for_company(
    cik: str,
    form_types: list[str],
    max_per_type: int = 4,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """
    Fetch recent filing metadata for a company from EDGAR submissions endpoint.
    Returns a list of dicts: {accession_number, form_type, filing_date, primary_doc}.
    """
    url = EDGAR_SUBMISSIONS_URL.format(cik=int(cik))
    resp = requests.get(url, headers={
        "User-Agent": "CS329-FinancialReportAnalyzer your-email@emory.edu",
        "Host": "data.sec.gov",
    })
    if resp.status_code != 200:
        print(f"  [!] Could not retrieve submissions for CIK {cik}: {resp.status_code}")
        return []

    data = resp.json()
    company_name = data.get("name", "Unknown")
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    counts = {ft: 0 for ft in form_types}

    for form, acc, date, doc in zip(forms, accessions, dates, primary_docs):
        if (
            form in form_types
            and counts[form] < max_per_type
            and _within_date_range(date, start_date, end_date)
        ):
            results.append({
                "company_name": company_name,
                "cik": cik,
                "form_type": form,
                "filing_date": date,
                "accession_number": acc,
                "primary_document": doc,
            })
            counts[form] += 1

    print(f"  [+] Found {len(results)} filings for {company_name}")
    return results


# ─── Raw Text Extraction ──────────────────────────────────────────────────────

def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> Optional[str]:
    """
    Download and return the raw text of a filing.
    Strips HTML/XBRL tags and returns plain text.
    """
    acc_clean = accession_number.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/{cik}/{acc_clean}/{primary_document}"

    resp = requests.get(base_url, headers={
        "User-Agent": "CS329-FinancialReportAnalyzer your-email@emory.edu"
    })
    if resp.status_code != 200:
        # Try with leading zeros on CIK in path
        base_url = f"https://www.sec.gov/Archives/edgar/full-index/{acc_clean[:4]}/{acc_clean[4:6]}/{primary_document}"
        resp = requests.get(base_url, headers={"User-Agent": "CS329-FinancialReportAnalyzer your-email@emory.edu"})
        if resp.status_code != 200:
            print(f"    [!] Could not fetch filing document: {resp.status_code} — {base_url}")
            return None

    raw = resp.text
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', raw)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove XBRL artifacts
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text


def fetch_filing_text_v2(cik: str, accession_number: str, primary_document: str) -> Optional[str]:
    """
    Alternate fetcher: constructs URL from CIK + accession directly (more reliable).
    """
    cik_nodash = str(int(cik))  # strip leading zeros for path
    acc_nodash = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nodash}/{acc_nodash}/{primary_document}"

    try:
        resp = requests.get(url, headers={
            "User-Agent": "CS329-FinancialReportAnalyzer your-email@emory.edu"
        }, timeout=30)
        if resp.status_code == 200:
            raw = resp.text
            text = re.sub(r'<[^>]+>', ' ', raw)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
            return text if len(text) > 500 else None
        else:
            print(f"    [!] HTTP {resp.status_code} for {url}")
            return None
    except Exception as e:
        print(f"    [!] Exception fetching {url}: {e}")
        return None


# ─── Save Record ─────────────────────────────────────────────────────────────

def save_filing_record(record: dict):
    """Save a structured filing record as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker = record.get("ticker", "UNKNOWN")
    form_type = record["form_type"].replace("/", "_")
    date = record["filing_date"]
    filename = f"{ticker}_{form_type}_{date}.json"
    path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"    [✓] Saved → {path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def collect_sec_filings(
    tickers: list[str],
    form_types: list[str] = ["10-K", "10-Q"],
    max_per_type: int = 2,
    delay: float = 0.5,
    start_date: date | None = None,
    end_date: date | None = None,
):
    """
    Main entry point. For each ticker:
      1. Resolve CIK
      2. Fetch filing list
      3. Download raw text
      4. Save structured JSON record
    """
    print("\n" + "="*60)
    print("  SEC EDGAR Filing Collector — CS329 Financial Report Analyzer")
    print("="*60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    all_records = []

    for ticker in tickers:
        print(f"\n[→] Processing: {ticker}")
        time.sleep(delay)

        cik = get_cik_from_ticker(ticker)
        if not cik:
            continue

        filings = get_filings_for_company(
            cik,
            form_types,
            max_per_type,
            start_date=start_date,
            end_date=end_date,
        )

        for filing in filings:
            print(f"  [↓] Fetching {filing['form_type']} from {filing['filing_date']}...")
            time.sleep(delay)

            text = fetch_filing_text_v2(cik, filing["accession_number"], filing["primary_document"])

            record = {
                "ticker": ticker.upper(),
                "company_name": filing["company_name"],
                "cik": cik,
                "form_type": filing["form_type"],
                "filing_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                "source": "SEC EDGAR",
                "raw_text": text if text else "[TEXT EXTRACTION FAILED]",
                "text_length": len(text) if text else 0,
                "collected_at": datetime.utcnow().isoformat() + "Z",
            }

            save_filing_record(record)
            all_records.append(record)

    # Also write a master index
    index_path = OUTPUT_DIR / "_index.json"
    index = [{k: v for k, v in r.items() if k != "raw_text"} for r in all_records]
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n[✓] Master index saved → {index_path}")
    print(f"[✓] Total records collected: {len(all_records)}")
    return all_records


if __name__ == "__main__":
    # Example: collect filings for a few tech companies
    TICKERS = ["AAPL", "MSFT", "NVDA"]
    collect_sec_filings(
        tickers=TICKERS,
        form_types=["10-K", "10-Q", "8-K"],
        max_per_type=2,
    )
