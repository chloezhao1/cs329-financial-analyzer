"""
apply_baseline.py
=================

Attach sector-relative z-scores to any signal-engine analysis record,
using the baseline stats from `baseline_stats.json`.

When a queried ticker IS in sector_map.py, z-scores are computed against
that sector's baseline. When a ticker is NOT mapped, z-scores fall back
to the corpus-wide baseline (_corpus_all). Output always includes a
human-readable `reference_label` so users know which baseline was used.

Usage:
    from apply_baseline import BaselineApplier
    applier = BaselineApplier()              # loads baseline_stats.json
    enriched = applier.apply(analysis)       # adds "zscores" field
    enriched_list = applier.apply_all(list)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sector_map import sector_for

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_FILE = Path("baseline_stats.json")


def _format_reference_label(reference: str, n: int) -> str:
    """
    Turn an internal reference name into something a human-reader would
    recognize immediately in the output JSON or demo UI.
    """
    if reference == "_corpus_all":
        return (
            f"All sectors combined (n={n}). Ticker not mapped to a "
            f"specific sector, so compared against the full reference "
            f"corpus. Sector-specific comparison would be more precise."
        )
    if reference == "no_baseline":
        return "No baseline available; z-scores are zero."
    # Sector-specific reference
    return f"{reference} sector peers (n={n})"


@dataclass
class BaselineApplier:
    """Attach sector-relative z-scores to signal engine analyses."""
    baseline_file: Path = DEFAULT_BASELINE_FILE
    baseline: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.baseline_file.exists():
            logger.warning(
                "Baseline file %s not found. Z-scores will be 0.0 for all "
                "records. Run `python build_baseline.py` to generate it.",
                self.baseline_file,
            )
            self.baseline = None
            return
        self.baseline = json.loads(self.baseline_file.read_text(encoding="utf-8"))
        logger.info(
            "Loaded baseline from %s (%d sectors, %d total reference records)",
            self.baseline_file,
            len(self.baseline.get("sectors", {})),
            self.baseline.get("n_total_records", 0),
        )

    def _stats_for(self, ticker: str) -> tuple[dict[str, Any], str, bool]:
        """
        Return (stats_dict, reference_name, reliable_flag) for a ticker.
        Falls back to _corpus_all if ticker's sector is unknown or
        insufficiently populated.
        """
        if self.baseline is None:
            return {}, "no_baseline", False

        sector = sector_for(ticker)
        sectors = self.baseline.get("sectors", {})
        if sector in sectors and sectors[sector].get("reliable"):
            return sectors[sector], sector, True

        corpus = self.baseline.get("_corpus_all", {})
        return corpus, "_corpus_all", bool(corpus.get("reliable"))

    def apply(self, analysis: dict) -> dict:
        """
        Attach a `zscores` field to a single analysis dict.

        Shape of the added field:
            {
                "reference":            "Technology" or "_corpus_all",
                "reference_label":      Human-readable comparison desc,
                "reference_n":          10,
                "reference_reliable":   true,
                "is_sector_specific":   true,      # false for _corpus_all
                "growth":               +0.42,
                "risk":                 -0.18,
                "net_operating_signal": +0.60,
            }
        """
        ticker = analysis.get("ticker", "")
        stats, reference, reliable = self._stats_for(ticker)

        n = stats.get("n", 0) if stats else 0
        is_sector_specific = reference not in ("_corpus_all", "no_baseline")

        if not stats:
            analysis["zscores"] = {
                "reference":            reference,
                "reference_label":      _format_reference_label(reference, 0),
                "reference_n":          0,
                "reference_reliable":   False,
                "is_sector_specific":   False,
                "growth":               0.0,
                "risk":                 0.0,
                "net_operating_signal": 0.0,
            }
            return analysis

        def _z(dim: str) -> float:
            d = stats.get(dim, {})
            mu, sigma = d.get("mean", 0.0), d.get("stdev", 1.0)
            if sigma < 1e-9:
                return 0.0
            raw = analysis["scores"].get(dim, 0.0)
            return (raw - mu) / sigma

        zg = _z("growth")
        zr = _z("risk")
        analysis["zscores"] = {
            "reference":            reference,
            "reference_label":      _format_reference_label(reference, n),
            "reference_n":          n,
            "reference_reliable":   reliable,
            "is_sector_specific":   is_sector_specific,
            "growth":               round(zg, 3),
            "risk":                 round(zr, 3),
            # Net z is z_growth - z_risk (above peers on growth AND below
            # peers on risk). Do NOT compute z of the raw net -- that would
            # need a separate net-baseline and is less interpretable.
            "net_operating_signal": round(zg - zr, 3),
        }
        return analysis

    def apply_all(self, analyses: list[dict]) -> list[dict]:
        return [self.apply(a) for a in analyses]
