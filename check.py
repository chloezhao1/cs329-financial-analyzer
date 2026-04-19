"""
check.py - score a single company on demand against the sector baseline.

Usage:
    python check.py NFLX
    python check.py SHOP BABA   (multiple at once)
"""
import json
import subprocess
import sys
from pathlib import Path


def check(tickers: list[str]) -> None:
    if not tickers:
        print("Usage: python check.py TICKER [TICKER ...]")
        sys.exit(1)

    # 1. Scrape + preprocess + score
    subprocess.run([
        sys.executable, "run_pipeline.py",
        "--tickers", *tickers,
        "--forms", "10-Q",
        "--max", "1",
        "--skip-transcripts",
    ], check=False)
    subprocess.run([sys.executable, "run_preprocessing.py"], check=False)
    subprocess.run([sys.executable, "financial_signal_engine.py"], check=False)

    # 2. Show the results
    print("\n" + "=" * 70)
    print(f"{'Ticker':<8s} {'Sector':<13s} {'z_growth':>9s} {'z_risk':>8s} {'z_net':>8s}  coverage")
    print("-" * 70)
    for t in tickers:
        matches = list(Path("data/signals").glob(f"{t.upper()}_*.signals.json"))
        if not matches:
            print(f"{t.upper():<8s} NO DATA (scrape may have failed)")
            continue
        # Most recent
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        d = json.loads(latest.read_text())
        z = d.get("zscores", {})
        ref = z.get("reference", "none")
        flag = "" if d["coverage"]["scored_sentences"] > 50 else "  ⚠ low coverage"
        print(
            f"{d['ticker']:<8s} {ref:<13s} "
            f"{z.get('growth', 0):>+9.2f} "
            f"{z.get('risk', 0):>+8.2f} "
            f"{z.get('net_operating_signal', 0):>+8.2f}  "
            f"{d['coverage']['scored_sentences']} sents{flag}"
        )


if __name__ == "__main__":
    check(sys.argv[1:])