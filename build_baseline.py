"""
build_baseline.py
=================

One-time script: scrapes a reference corpus of companies (defined in
sector_map.py), preprocesses them, scores them with the signal engine,
and writes per-sector statistics to `baseline_stats.json`.

The resulting baseline file is used by the signal engine's
`apply_baseline()` function to compute sector-relative z-scores for
ANY company queried later, including companies not in the baseline.

Usage:
    # Scrape + preprocess + score + compute baseline
    python build_baseline.py

    # Skip scraping/preprocessing (use existing data/processed/)
    python build_baseline.py --skip-ingest

    # Use a subset of sectors (for faster iteration / testing)
    python build_baseline.py --sectors Technology Financials

Output: baseline_stats.json at the project root. Commit this file to
the repo so teammates don't need to rerun.

Runtime estimate: ~90 min for 50 tickers (scraping is the bottleneck
due to SEC rate limits). Subsequent runs with --skip-ingest take <10s.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Local imports
from sector_map import BASELINE_TICKERS, SECTOR_MAP, SECTORS, sector_for

try:
    from financial_signal_engine import LMDictionary, SignalEngine, load_records
    _HAS_ENGINE = True
except ImportError as e:
    # Defer the error until main() actually needs the engine. This allows
    # compute_baseline() and helpers to be imported and tested in isolation.
    _HAS_ENGINE = False
    _IMPORT_ERROR = e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_baseline")

BASELINE_FILE = Path("baseline_stats.json")
MIN_COMPANIES_PER_SECTOR = 4      # sectors with fewer are flagged unreliable
INDICATORS = ("growth", "risk")   # cost_pressure optional; drop if removed


def _run_cmd(cmd: list[str]) -> int:
    """Run a subprocess command with live output. Returns exit code."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def ingest_tickers(tickers: list[str], max_per_form: int = 1) -> None:
    """Drive scraping + preprocessing for the baseline universe."""
    # Scrape filings via run_pipeline.py (SEC only; transcripts are optional)
    rc = _run_cmd([
        sys.executable, "run_pipeline.py",
        "--tickers", *tickers,
        "--forms", "10-Q",
        "--max", str(max_per_form),
        "--skip-transcripts",
    ])
    if rc != 0:
        logger.error("Scraping step failed (exit %d). Continuing anyway --- "
                     "some tickers may still have been fetched.", rc)

    # Preprocess whatever got scraped
    rc = _run_cmd([
        sys.executable, "run_preprocessing.py", "--overwrite",
    ])
    if rc != 0:
        logger.error("Preprocessing step failed (exit %d). "
                     "Some records may be missing.", rc)


def compute_baseline(
    analyses: list[dict],
    sector_map: dict[str, str] = SECTOR_MAP,
) -> dict[str, Any]:
    """
    Compute per-sector statistics (mean, stdev, n) for each indicator
    plus a corpus-wide fallback. Result shape:

        {
            "engine_version": "...",
            "n_total_records": 48,
            "indicators":      ["growth", "risk"],
            "sectors": {
                "Technology": {
                    "n": 10,
                    "tickers": ["AAPL", "MSFT", ...],
                    "growth": {"mean": 0.14, "stdev": 0.06},
                    "risk":   {"mean": 0.31, "stdev": 0.05},
                    "reliable": true
                },
                ...
            },
            "_corpus_all": {...same shape...}
        }
    """
    by_sector: dict[str, list[dict]] = {}
    for a in analyses:
        sec = sector_map.get(a["ticker"].upper(), "Unknown")
        by_sector.setdefault(sec, []).append(a)

    out_sectors: dict[str, dict] = {}
    for sec, records in by_sector.items():
        out_sectors[sec] = _sector_stats(sec, records)

    # Also compute a corpus-wide baseline as a fallback for unknown sectors
    corpus_stats = _sector_stats("_corpus_all", analyses)

    # Pull engine version from the first analysis's method metadata (if any)
    engine_version = (
        analyses[0].get("method", {}).get("engine_version", "unknown")
        if analyses else "unknown"
    )

    return {
        "engine_version":   engine_version,
        "n_total_records":  len(analyses),
        "indicators":       list(INDICATORS),
        "min_sector_size":  MIN_COMPANIES_PER_SECTOR,
        "sectors":          out_sectors,
        "_corpus_all":      corpus_stats,
    }


def _sector_stats(sector_name: str, records: list[dict]) -> dict[str, Any]:
    n = len(records)
    tickers = sorted({r["ticker"].upper() for r in records})
    reliable = n >= MIN_COMPANIES_PER_SECTOR

    stats: dict[str, Any] = {
        "n":         n,
        "tickers":   tickers,
        "reliable":  reliable,
    }
    for dim in INDICATORS:
        values = [r["scores"].get(dim, 0.0) for r in records]
        if len(values) >= 2:
            m = mean(values)
            # Guard against zero stdev (all values identical)
            s = stdev(values) if stdev(values) > 1e-9 else 1.0
        elif len(values) == 1:
            m = values[0]
            s = 1.0   # placeholder; z-scores will be 0 against yourself
        else:
            m = 0.0
            s = 1.0
        stats[dim] = {"mean": round(m, 4), "stdev": round(s, 4)}

    if not reliable:
        logger.warning(
            "Sector '%s' has only %d records (need %d). "
            "Stats computed but flagged unreliable.",
            sector_name, n, MIN_COMPANIES_PER_SECTOR,
        )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-ingest", action="store_true",
        help="Use existing data/processed/ records; don't re-scrape.",
    )
    ap.add_argument(
        "--sectors", nargs="+", default=None,
        help="Only process these sectors (faster; for testing).",
    )
    ap.add_argument(
        "--lm-csv", type=Path,
        default=Path("data/lexicons/loughran_mcdonald.csv"),
    )
    ap.add_argument(
        "--out", type=Path, default=BASELINE_FILE,
        help=f"Output path (default: {BASELINE_FILE})",
    )
    args = ap.parse_args()

    # Filter ticker universe if --sectors was passed
    if args.sectors:
        selected_sectors = set(args.sectors)
        missing = selected_sectors - set(SECTORS)
        if missing:
            logger.error("Unknown sector(s): %s. Known: %s",
                         missing, SECTORS)
            return 1
        tickers = [t for t, s in SECTOR_MAP.items() if s in selected_sectors]
        logger.info("Restricted to %d tickers in sector(s): %s",
                    len(tickers), selected_sectors)
    else:
        tickers = BASELINE_TICKERS

    # Stage 1: scrape + preprocess (unless skipped)
    if not args.skip_ingest:
        logger.info("Ingesting %d tickers. This may take 60-90 minutes.",
                    len(tickers))
        ingest_tickers(tickers)
    else:
        logger.info("Skipping ingest; reading existing data/processed/")

    # Stage 2: load processed records + score them
    if not _HAS_ENGINE:
        logger.error(
            "Cannot score records: financial_signal_engine could not be "
            "imported (%s). Place this script in the project root.",
            _IMPORT_ERROR,
        )
        return 1

    logger.info("Loading LM dictionary from %s", args.lm_csv)
    lm = LMDictionary.from_csv(args.lm_csv)
    engine = SignalEngine(lm)

    records = load_records(Path("."))
    if not records:
        logger.error("No processed records found. Nothing to baseline.")
        return 1

    # Restrict to baseline universe (skip anything outside SECTOR_MAP)
    baseline_universe = {t.upper() for t in tickers}
    baseline_records = [
        r for r in records
        if r.get("ticker", "").upper() in baseline_universe
    ]
    logger.info(
        "Found %d processed records; %d are in baseline universe.",
        len(records), len(baseline_records),
    )

    # Handle duplicates (e.g. multiple 10-Qs for same ticker) by keeping the
    # most recent. Baseline should be one data point per company.
    by_ticker: dict[str, dict] = {}
    for r in baseline_records:
        t = r["ticker"].upper()
        prev = by_ticker.get(t)
        if prev is None or r.get("filing_date", "") > prev.get("filing_date", ""):
            by_ticker[t] = r

    deduped = list(by_ticker.values())
    logger.info("After dedup (most-recent per ticker): %d records", len(deduped))

    # Stage 3: score each record
    logger.info("Scoring %d records...", len(deduped))
    analyses = [engine.analyze_record(r) for r in deduped]

    # Log a quick preview
    for a in analyses[:5]:
        logger.info(
            "  %-6s %s  growth=%+.3f  risk=%+.3f  sector=%s",
            a["ticker"], a["filing_date"],
            a["scores"]["growth"], a["scores"]["risk"],
            sector_for(a["ticker"]),
        )
    if len(analyses) > 5:
        logger.info("  ... and %d more", len(analyses) - 5)

    # Stage 4: compute baseline + write
    baseline = compute_baseline(analyses)
    args.out.write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote baseline to %s", args.out)

    # Print a summary table
    print()
    print(f"{'Sector':<15s} {'n':>4s} {'growth μ':>10s} {'growth σ':>10s} "
          f"{'risk μ':>10s} {'risk σ':>10s} {'reliable':>10s}")
    print("-" * 75)
    for sec in sorted(baseline["sectors"]):
        s = baseline["sectors"][sec]
        print(
            f"{sec:<15s} {s['n']:>4d} "
            f"{s['growth']['mean']:>+10.3f} {s['growth']['stdev']:>10.3f} "
            f"{s['risk']['mean']:>+10.3f} {s['risk']['stdev']:>10.3f} "
            f"{'yes' if s['reliable'] else 'NO':>10s}"
        )
    s = baseline["_corpus_all"]
    print("-" * 75)
    print(
        f"{'_corpus_all':<15s} {s['n']:>4d} "
        f"{s['growth']['mean']:>+10.3f} {s['growth']['stdev']:>10.3f} "
        f"{s['risk']['mean']:>+10.3f} {s['risk']['stdev']:>10.3f} "
        f"{'yes' if s['reliable'] else 'NO':>10s}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
