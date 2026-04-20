"""Default signal-engine factory for the API layer.

Central place that decides which engine implementation the dashboard and
pipeline run through. Swapping defaults later (e.g. to the hybrid LLM
engine) is a one-line change here instead of scattered across routers.

The hybrid engine (`financial_signal_engine_LLMv1.HybridSignalEngine`) is
deliberately NOT the default: it requires an Anthropic API key and its
output shape is per-sentence, not per-document, so it does not drop into
`analyze_records`. It is wired in only where a hybrid evaluation path is
explicitly requested.

Why `DefaultV2Engine` exists
----------------------------
`SignalEngineV2` in `financial_signal_engine_v2.py` overrides only
`score_sentence`; it is not a subclass of `SignalEngine` and therefore
has no `analyze_record` / `_tally_phrases` / `_by_section`. But
`analyze_records` in v1 calls `engine.analyze_record(r)` per document.

We cannot modify the engine files. The cleanest fix is a small subclass
here that reuses V2's sentence scorer + LM blocklist but keeps v1's
document-level aggregation machinery, via multiple inheritance.
"""
from __future__ import annotations

from pathlib import Path

from financial_signal_engine import LMDictionary, SignalEngine
from financial_signal_engine_v2 import SignalEngineV2

__all__ = ["DefaultV2Engine", "build_default_engine", "load_default_engine"]


class DefaultV2Engine(SignalEngineV2, SignalEngine):
    """V2 sentence scoring + v1 document-level aggregation.

    MRO: ``[DefaultV2Engine, SignalEngineV2, SignalEngine, object]``

    - ``__init__``       — SignalEngineV2 (builds a trimmed LM, sets self.lm)
    - ``score_sentence`` — SignalEngineV2 (V2 phrase list, token floor, negation)
    - ``analyze_record`` — SignalEngine   (document aggregation, top sentences,
                                           phrase tallies, section breakdown)
    - ``_tally_phrases`` / ``_by_section`` — SignalEngine helpers
    """


def build_default_engine(lm: LMDictionary) -> DefaultV2Engine:
    """Construct the default (V2 lexicon) signal engine from a loaded LM."""
    return DefaultV2Engine(lm)


def load_default_engine(lm_csv: Path) -> DefaultV2Engine:
    """Load the LM CSV and return a ready-to-use V2 engine."""
    return build_default_engine(LMDictionary.from_csv(lm_csv))
