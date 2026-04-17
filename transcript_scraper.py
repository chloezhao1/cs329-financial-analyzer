"""
Earnings Call Transcript Scraper
Scrapes earnings call transcripts from Motley Fool using Selenium.
Falls back to a Kaggle-sourced transcript dataset if scraping is blocked.

Usage:
    python transcript_scraper.py

Outputs JSON records in data/transcripts/ with the same schema as SEC filings.
"""

import json
import time
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Output Directory ────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/transcripts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Selenium Setup ───────────────────────────────────────────────────────────

def get_driver(headless: bool = True):
    """Initialize a Selenium Chrome WebDriver."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        print(f"[!] Selenium WebDriver init failed: {e}")
        print("    Install: pip install selenium && brew install chromedriver")
        return None


# ─── Motley Fool Scraper ──────────────────────────────────────────────────────

MOTLEY_FOOL_SEARCH = "https://www.fool.com/earnings-call-transcripts/?search={ticker}"
MOTLEY_FOOL_BASE = "https://www.fool.com"

def scrape_motley_fool_transcripts(ticker: str, max_transcripts: int = 4, delay: float = 2.0) -> list[dict]:
    """
    Scrape earnings call transcripts for a ticker from Motley Fool.
    Returns list of structured transcript records.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    print(f"  [→] Scraping Motley Fool for {ticker}...")
    driver = get_driver(headless=True)
    if not driver:
        return []

    records = []

    try:
        search_url = MOTLEY_FOOL_SEARCH.format(ticker=ticker.lower())
        driver.get(search_url)
        time.sleep(delay)

        # Find transcript links on the search/listing page
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='earnings-call-transcript']")
        transcript_urls = []
        for link in links[:max_transcripts]:
            href = link.get_attribute("href")
            if href and "transcript" in href:
                if href.startswith("/"):
                    href = MOTLEY_FOOL_BASE + href
                transcript_urls.append(href)

        transcript_urls = list(dict.fromkeys(transcript_urls))  # deduplicate
        print(f"    [+] Found {len(transcript_urls)} transcript links")

        for url in transcript_urls[:max_transcripts]:
            print(f"    [↓] Fetching: {url}")
            driver.get(url)
            time.sleep(delay)

            record = _parse_motley_fool_page(driver, ticker, url)
            if record:
                records.append(record)
                save_transcript_record(record)

    except Exception as e:
        print(f"    [!] Scraping error: {e}")
    finally:
        driver.quit()

    return records


def _parse_motley_fool_page(driver, ticker: str, url: str) -> Optional[dict]:
    """Extract structured data from a single Motley Fool transcript page."""
    from selenium.webdriver.common.by import By

    try:
        # Title (often contains company name + quarter + year)
        title = ""
        try:
            title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
        except Exception:
            title = driver.title

        # Date
        filing_date = ""
        try:
            date_el = driver.find_element(By.CSS_SELECTOR, "time, [class*='date'], [class*='Date']")
            filing_date = date_el.get_attribute("datetime") or date_el.text.strip()
            # Normalize to YYYY-MM-DD if possible
            filing_date = _normalize_date(filing_date)
        except Exception:
            filing_date = datetime.utcnow().strftime("%Y-%m-%d")

        # Raw transcript text — Motley Fool wraps in article body
        text = ""
        try:
            article = driver.find_element(By.CSS_SELECTOR, "article, [class*='article-body'], .transcript-body")
            text = article.text.strip()
        except Exception:
            # Fallback: grab all paragraph text
            paras = driver.find_elements(By.TAG_NAME, "p")
            text = "\n".join(p.text for p in paras if len(p.text) > 50)

        if len(text) < 200:
            print(f"    [!] Too little text extracted ({len(text)} chars), skipping.")
            return None

        # Parse quarter/year from title if possible
        quarter, year = _parse_quarter_year(title)

        company_name = _parse_company_name(title, ticker)

        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "form_type": "EARNINGS_CALL",
            "filing_date": filing_date,
            "quarter": quarter,
            "year": year,
            "source": "Motley Fool",
            "source_url": url,
            "title": title,
            "raw_text": text,
            "text_length": len(text),
            "collected_at": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"    [!] Parse error: {e}")
        return None


# ─── Seeking Alpha Scraper (stub — requires subscription bypass) ──────────────

def scrape_seeking_alpha_transcripts(ticker: str) -> list[dict]:
    """
    Seeking Alpha requires login for full transcripts.
    This stub documents the approach; use the Kaggle dataset as a practical alternative.
    
    Real implementation would:
    1. Log in via Selenium (credentials in env vars)
    2. Navigate to https://seekingalpha.com/symbol/{TICKER}/earnings/transcripts
    3. Click each transcript link and extract .sa-art article body text
    """
    print(f"  [!] Seeking Alpha requires authentication. Use Kaggle transcript dataset instead.")
    print(f"      Dataset: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts")
    return []


# ─── Kaggle Dataset Loader (offline fallback) ────────────────────────────────

def load_kaggle_transcripts(csv_path: str, tickers: list[str] = None) -> list[dict]:
    """
    Load transcripts from the Kaggle Motley Fool earnings call dataset.
    CSV columns expected: ticker, company_name, date, transcript (or similar).
    
    Dataset: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts
    """
    import csv

    if not os.path.exists(csv_path):
        print(f"[!] Kaggle CSV not found at {csv_path}")
        print("    Download from: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts")
        return []

    records = []
    tickers_upper = [t.upper() for t in tickers] if tickers else None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", row.get("symbol", "")).upper()
            if tickers_upper and ticker not in tickers_upper:
                continue

            text = row.get("transcript", row.get("text", row.get("content", "")))
            date = row.get("date", row.get("filing_date", ""))

            record = {
                "ticker": ticker,
                "company_name": row.get("company_name", row.get("company", "")),
                "form_type": "EARNINGS_CALL",
                "filing_date": _normalize_date(date),
                "quarter": row.get("quarter", ""),
                "year": row.get("year", ""),
                "source": "Kaggle / Motley Fool",
                "raw_text": text,
                "text_length": len(text),
                "collected_at": datetime.utcnow().isoformat() + "Z",
            }
            save_transcript_record(record)
            records.append(record)

    print(f"[✓] Loaded {len(records)} transcripts from Kaggle dataset")
    return records


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_date(date_str: str) -> str:
    """Try to parse various date formats into YYYY-MM-DD."""
    date_str = date_str.strip()
    formats = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str[:len(fmt)], fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return date_str  # return as-is if can't parse


def _parse_quarter_year(title: str) -> tuple[str, str]:
    """Extract Q1/Q2/Q3/Q4 and year from transcript title."""
    q_match = re.search(r'Q([1-4])', title, re.IGNORECASE)
    y_match = re.search(r'(20\d{2})', title)
    return (f"Q{q_match.group(1)}" if q_match else ""), (y_match.group(1) if y_match else "")


def _parse_company_name(title: str, ticker: str) -> str:
    """Best-effort company name extraction from title."""
    # Titles often look like: "Apple (AAPL) Q1 2024 Earnings Call Transcript"
    match = re.match(r'^(.+?)\s*[\(\[]?' + ticker, title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ticker


def save_transcript_record(record: dict):
    """Save a transcript record as JSON."""
    ticker = record.get("ticker", "UNKNOWN")
    quarter = record.get("quarter", "")
    year = record.get("year", "")
    date = record.get("filing_date", "")
    filename = f"{ticker}_EARNINGS_CALL_{date or f'{year}_{quarter}'}.json"
    path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"    [✓] Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def collect_transcripts(
    tickers: list[str],
    max_per_ticker: int = 4,
    kaggle_csv_path: str = None,
):
    print("\n" + "="*60)
    print("  Earnings Call Transcript Collector — CS329")
    print("="*60)

    all_records = []

    # Option 1: Kaggle dataset (recommended for bulk / offline)
    if kaggle_csv_path:
        records = load_kaggle_transcripts(kaggle_csv_path, tickers)
        all_records.extend(records)

    # Option 2: Live Motley Fool scraping
    else:
        for ticker in tickers:
            print(f"\n[→] Scraping transcripts for: {ticker}")
            records = scrape_motley_fool_transcripts(ticker, max_transcripts=max_per_ticker)
            all_records.extend(records)
            time.sleep(2.0)

    # Write master index
    index_path = OUTPUT_DIR / "_index.json"
    index = [{k: v for k, v in r.items() if k != "raw_text"} for r in all_records]
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[✓] Transcript collection complete. {len(all_records)} records.")
    return all_records


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "NVDA"]
    collect_transcripts(
        tickers=TICKERS,
        max_per_ticker=4,
        # kaggle_csv_path="path/to/kaggle_transcripts.csv",  # uncomment if using Kaggle
    )
