"""Backend-local resolver for the Loughran-McDonald dictionary CSV.

Duplicated here (not in `financial_signal_engine.py`) to keep the existing
.py files unmodified per project constraints.
"""
from __future__ import annotations

from pathlib import Path


def resolve_lm_csv(base_dir: Path) -> Path:
    root = base_dir.resolve()
    candidates = [
        root / "data" / "lexicons" / "loughran_mcdonald.csv",
        root / "data" / "lexicon" / "loughran_mcdonald.csv",
        root / "data" / "lexicons" / "LM_MasterDictionary.csv",
        root / "data" / "LM_MasterDictionary.csv",
        root / "data" / "loughran_mcdonald.csv",
        root / "Loughran-McDonald_MasterDictionary_1993-2025.csv",
        root / "LM_MasterDictionary_1993-2024.csv",
        root / "LM_MasterDictionary.csv",
        root / "loughran_mcdonald.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Loughran-McDonald dictionary CSV not found. Download from "
        "https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and place "
        f"it at data/lexicons/loughran_mcdonald.csv.\nSearched:\n  {searched}"
    )
