"""
run_preprocessing.py
====================

Batch runner: reads data/_master_index.json produced by run_pipeline.py,
dispatches each record through PreprocessingPipeline, and writes enriched
records to data/processed/.

Usage:
    python run_preprocessing.py
    python run_preprocessing.py --data-dir ./data --limit 5
    python run_preprocessing.py --only-form 10-K
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from text_preprocessor import PreprocessingPipeline, PREPROCESSING_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_preprocessing")


def _iter_input_files(data_dir: Path) -> list[Path]:
    """Find all collector-produced JSON files (ignores _index.json files)."""
    out: list[Path] = []
    for sub in ("filings", "transcripts"):
        d = data_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            out.append(p)
    return out



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Root data directory (default: ./data)")
    ap.add_argument("--out-subdir", default="processed",
                    help="Subdirectory under data-dir to write outputs")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N files (for testing)")
    ap.add_argument("--only-form", default=None,
                    help="Only process records whose form_type matches "
                         "(e.g. 10-K, 10-Q, EARNINGS_CALL)")
    ap.add_argument("--model", default="en_core_web_sm",
                    help="spaCy model name (default: en_core_web_sm)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess files that already exist in output dir")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = data_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _iter_input_files(data_dir)
    logger.info("Found %d input files under %s", len(inputs), data_dir)

    pipeline = PreprocessingPipeline(nlp_model=args.model)

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    index_entries: list[dict] = []

    for p in inputs:
        if args.limit and n_done >= args.limit:
            break
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Could not read %s: %s", p, e)
            continue

        form = (record.get("form_type") or "").upper()
        if args.only_form and form != args.only_form.upper():
            continue

        ticker = record.get("ticker", "UNK")
        date = record.get("filing_date", "unknown")
        out_path = out_dir / f"{ticker}_{form}_{date}.processed.json"

        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        try:
            processed = pipeline.process_record(record)
        except Exception as e:
            logger.exception("Failed to process %s: %s", p.name, e)
            continue

        out_path.write_text(
            json.dumps(processed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stats = processed["processed"]["stats"]
        logger.info(
            "%s: %d sentences, %d tokens, %d negated, %d hedged",
            p.name,
            stats["n_sentences"],
            stats["n_tokens"],
            stats["n_negated_sentences"],
            stats["n_hedged_sentences"],
        )
        index_entries.append({
            "ticker": ticker,
            "form_type": form,
            "filing_date": date,
            "source_file": str(p.relative_to(data_dir)),
            "processed_file": str(out_path.relative_to(data_dir)),
            "stats": stats,
        })
        n_done += 1

    # Write processed index
    index_path = out_dir / "_index.json"
    index_path.write_text(
        json.dumps(
            {
                "preprocessing_version": PREPROCESSING_VERSION,
                "n_records":             len(index_entries),
                "records":               index_entries,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    elapsed = time.time() - t0
    logger.info(
        "Done: processed=%d, skipped=%d, elapsed=%.1fs, index=%s",
        n_done, n_skipped, elapsed, index_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
