"""
/api/signals/*

Exposes the core signal-engine views consumed by the React dashboard.
Backed entirely by existing, unmodified functions in
`financial_signal_engine.py`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from financial_signal_engine import (
    LMDictionary,
    analyze_records,
    build_comparison_rows,
    infer_data_source,
    load_records,
)

logger = logging.getLogger(__name__)
router = APIRouter()

from backend.engine_factory import (
    DefaultV2Engine,
    DefaultV3Engine,
    load_preferred_signal_engine,
)
from backend.lm_path import resolve_lm_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_FILE = PROJECT_ROOT / "baseline_stats.json"


class _AnalysesCache:
    """Thread-safe in-memory cache for computed analyses.

    Computing analyses is cheap (pure Python over already-preprocessed
    records) but avoiding repeated disk reads + baseline loads on every
    page navigation is a meaningful UX win.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._analyses: list[dict] | None = None
        self._engine: DefaultV2Engine | DefaultV3Engine | None = None

    def _engine_or_build(self) -> DefaultV2Engine | DefaultV3Engine:
        """Build the preferred signal engine (V3, or V2 if V3 cannot load)."""
        if self._engine is None:
            lm_csv = resolve_lm_csv(PROJECT_ROOT)
            self._engine = load_preferred_signal_engine(lm_csv, PROJECT_ROOT)
        return self._engine

    def get(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            if self._analyses is not None and not refresh:
                return self._analyses
            records = load_records(PROJECT_ROOT)
            engine = self._engine_or_build()
            self._analyses = analyze_records(records, engine=engine) if records else []
            return self._analyses

    def invalidate(self) -> None:
        with self._lock:
            self._analyses = None


_CACHE = _AnalysesCache()


def get_cache() -> _AnalysesCache:
    """Exposed so other routers (e.g. /pipeline/run) can invalidate."""
    return _CACHE


@router.get("/analyses")
def list_analyses(refresh: bool = Query(False, description="Force reload from disk")) -> list[dict]:
    """Return every computed analysis for the current dataset."""
    return _CACHE.get(refresh=refresh)


@router.get("/data-source")
def get_data_source() -> dict:
    """Report which data stage is available on disk."""
    return {"data_source": infer_data_source(PROJECT_ROOT)}


class ComparisonRequest(BaseModel):
    labels: list[str] = Field(
        ...,
        description='Analysis labels in the form "{ticker} | {form_type} | {filing_date}".',
    )


def _format_label(a: dict) -> str:
    return f"{a['ticker']} | {a['form_type']} | {a['filing_date']}"


@router.post("/comparison")
def comparison_rows(req: ComparisonRequest) -> list[dict]:
    """Return `build_comparison_rows` for the selected analyses."""
    analyses = _CACHE.get()
    by_label: dict[str, dict] = {_format_label(a): a for a in analyses}
    selected: list[dict] = []
    missing: list[str] = []
    for label in req.labels:
        if label in by_label:
            selected.append(by_label[label])
        else:
            missing.append(label)
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "Some labels are not currently loaded", "missing": missing},
        )
    return build_comparison_rows(selected)


@router.get("/baseline")
def get_baseline() -> dict[str, Any] | None:
    """Return the raw `baseline_stats.json` contents, or null if not built yet."""
    if not BASELINE_FILE.exists():
        return None
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Could not parse baseline file: %s", e)
        raise HTTPException(status_code=500, detail=f"Could not parse baseline file: {e}")


# ---------------------------------------------------------------------------
# Hybrid LLM rescore (opt-in; requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

class HybridRescoreRequest(BaseModel):
    label: str = Field(
        ...,
        description='Analysis label "{ticker} | {form_type} | {filing_date}".',
    )
    max_sentences: int = Field(
        120,
        ge=1,
        le=2000,
        description=(
            "Cap on sentences sent through the engine. LLM fallback only "
            "touches the subset with zero lexicon hits; this cap bounds "
            "wall-clock time on long 10-Ks."
        ),
    )


class HybridSentence(BaseModel):
    text: str
    method: str  # "lexicon" | "llm"
    label: str  # "positive" | "negative" | "neutral"
    net_score: float
    growth: float
    risk: float
    cost_pressure: float
    llm_reason: str | None = None


class HybridRescoreResponse(BaseModel):
    label: str
    ticker: str
    form_type: str
    filing_date: str
    total_sentences: int
    scanned_sentences: int
    lexicon_hits: int
    llm_fallback: int
    llm_positive: int
    llm_negative: int
    llm_neutral: int
    lexicon_coverage_rate: float
    hybrid_coverage_rate: float
    sentences: list[HybridSentence]
    # Aggregated hybrid document scores
    hybrid_growth_score: float
    hybrid_risk_score: float
    hybrid_cost_score: float
    hybrid_net_score: float
    hybrid_positive_count: int
    hybrid_negative_count: int
    hybrid_neutral_count: int


def _find_record_for_label(label: str) -> dict | None:
    """Re-load raw records and return the one matching a dashboard label."""
    raw = load_records(PROJECT_ROOT)
    for r in raw:
        r_label = (
            f"{r.get('ticker', 'UNK')} | "
            f"{r.get('form_type', 'UNKNOWN')} | "
            f"{r.get('filing_date', 'Unknown')}"
        )
        if r_label == label:
            return r
    return None


@router.post("/hybrid", response_model=HybridRescoreResponse)
def hybrid_rescore(req: HybridRescoreRequest) -> HybridRescoreResponse:
    """Run V2 lexicon + Claude LLM fallback on a single document.

    Flow (per HybridSignalEngine):
      1. Every sentence is scored by SignalEngineV2 first (free, fast).
      2. Sentences with zero lexicon hits are batched to Claude for
         positive/negative/neutral classification.
      3. Results are merged: lexicon label when the lexicon spoke,
         LLM label when it did not.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail=(
                "ANTHROPIC_API_KEY is not set. Add it to the project-root "
                ".env file and restart the API server."
            ),
        )

    record = _find_record_for_label(req.label)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Label not found: {req.label}")

    sentences = (record.get("processed") or {}).get("sentences") or []
    if not sentences:
        raise HTTPException(
            status_code=422,
            detail=(
                "Record has no preprocessed sentences. Re-run the pipeline "
                "so preprocessing populates data/processed/."
            ),
        )

    scanned = sentences[: req.max_sentences]

    try:
        from financial_signal_engine_LLMv1 import HybridSignalEngine
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not import HybridSignalEngine: {e}. Is the 'anthropic' "
                "package installed? (`pip install anthropic`)"
            ),
        )

    try:
        lm_csv = resolve_lm_csv(PROJECT_ROOT)
        lm = LMDictionary.from_csv(lm_csv)
        engine = HybridSignalEngine(lm)
    except Exception as e:
        logger.exception("Could not build HybridSignalEngine")
        raise HTTPException(status_code=500, detail=f"Hybrid engine init failed: {e}")

    try:
        results = engine.score_batch(scanned)
    except Exception as e:
        logger.exception("Hybrid score_batch failed")
        raise HTTPException(status_code=500, detail=f"Hybrid scoring failed: {e}")

    stats = engine.stats
    scanned_n = stats.get("total", len(scanned)) or 1
    lexicon_hits = stats.get("lexicon_hits", 0)
    llm_fallback = stats.get("llm_fallback", 0)
    llm_with_label = (
        stats.get("llm_positive", 0) + stats.get("llm_negative", 0)
    )

    # Aggregate hybrid document-level scores from all sentences
    # Normalize by number of sentences (same as SignalEngine.analyze_record)
    hybrid_growth_sum = 0.0
    hybrid_risk_sum = 0.0
    hybrid_cost_sum = 0.0
    hybrid_positive_count = 0
    hybrid_negative_count = 0
    hybrid_neutral_count = 0

    for r in results:
        hybrid_growth_sum += float(r.get("growth", 0.0))
        hybrid_risk_sum += float(r.get("risk", 0.0))
        hybrid_cost_sum += float(r.get("cost_pressure", 0.0))
        lbl = r.get("label", "neutral")
        if lbl == "positive":
            hybrid_positive_count += 1
        elif lbl == "negative":
            hybrid_negative_count += 1
        else:
            hybrid_neutral_count += 1

    # Normalize by number of scored sentences (matching original engine behavior)
    n_scored = len(results) or 1
    hybrid_growth = hybrid_growth_sum / n_scored
    hybrid_risk = hybrid_risk_sum / n_scored
    hybrid_cost = hybrid_cost_sum / n_scored
    hybrid_net = hybrid_growth - hybrid_risk

    return HybridRescoreResponse(
        label=req.label,
        ticker=record.get("ticker", "UNK"),
        form_type=record.get("form_type", "UNKNOWN"),
        filing_date=record.get("filing_date", "Unknown"),
        total_sentences=len(sentences),
        scanned_sentences=len(scanned),
        lexicon_hits=lexicon_hits,
        llm_fallback=llm_fallback,
        llm_positive=stats.get("llm_positive", 0),
        llm_negative=stats.get("llm_negative", 0),
        llm_neutral=stats.get("llm_neutral", 0),
        lexicon_coverage_rate=round(lexicon_hits / scanned_n, 4),
        hybrid_coverage_rate=round((lexicon_hits + llm_with_label) / scanned_n, 4),
        sentences=[
            HybridSentence(
                text=r.get("text", ""),
                method=r.get("method", "lexicon"),
                label=r.get("label", "neutral"),
                net_score=float(r.get("net_score", 0.0)),
                growth=float(r.get("growth", 0.0)),
                risk=float(r.get("risk", 0.0)),
                cost_pressure=float(r.get("cost_pressure", 0.0)),
                llm_reason=r.get("llm_reason"),
            )
            for r in results
        ],
        hybrid_growth_score=round(hybrid_growth, 4),
        hybrid_risk_score=round(hybrid_risk, 4),
        hybrid_cost_score=round(hybrid_cost, 4),
        hybrid_net_score=round(hybrid_net, 4),
        hybrid_positive_count=hybrid_positive_count,
        hybrid_negative_count=hybrid_negative_count,
        hybrid_neutral_count=hybrid_neutral_count,
    )


# ---------------------------------------------------------------------------
# Pure LLM (opt-in; requires ANTHROPIC_API_KEY) — no lexicon / signal engine mix
# ---------------------------------------------------------------------------


class LlmPureRescoreRequest(BaseModel):
    label: str = Field(
        ...,
        description='Analysis label "{ticker} | {form_type} | {filing_date}".',
    )
    max_sentences: int = Field(
        120,
        ge=1,
        le=2000,
        description="Cap on sentences sent through Claude (API cost & latency).",
    )


class LlmPureSentence(BaseModel):
    text: str
    method: str
    label: str
    net_score: float
    growth: float
    risk: float
    cost_pressure: float
    reason: str | None = None


class LlmPureRescoreResponse(BaseModel):
    label: str
    engine: str
    engine_version: str
    ticker: str
    form_type: str
    filing_date: str
    total_sentences: int
    scanned_sentences: int
    sentences: list[LlmPureSentence]
    llm_pure_growth_score: float
    llm_pure_risk_score: float
    llm_pure_cost_score: float
    llm_pure_net_score: float
    llm_positive_count: int
    llm_negative_count: int
    llm_neutral_count: int


@router.post("/llm-pure", response_model=LlmPureRescoreResponse)
def llm_pure_rescore(req: LlmPureRescoreRequest) -> LlmPureRescoreResponse:
    """Score a document with the pure LLM engine only (Claude, every sentence)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail=(
                "ANTHROPIC_API_KEY is not set. Add it to the project-root "
                ".env file and restart the API server."
            ),
        )

    record = _find_record_for_label(req.label)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Label not found: {req.label}")

    sentences = (record.get("processed") or {}).get("sentences") or []
    if not sentences:
        raise HTTPException(
            status_code=422,
            detail=(
                "Record has no preprocessed sentences. Re-run the pipeline "
                "so preprocessing populates data/processed/."
            ),
        )

    scanned = sentences[: req.max_sentences]

    try:
        from financial_signal_engine_LLMpure import PURE_LLM_VERSION, PureLLMSignalEngine
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import PureLLMSignalEngine: {e}.",
        )

    try:
        engine = PureLLMSignalEngine()
    except Exception as e:
        logger.exception("Could not build PureLLMSignalEngine")
        raise HTTPException(status_code=500, detail=f"Pure LLM engine init failed: {e}")

    try:
        results = engine.score_batch(scanned)
    except Exception as e:
        logger.exception("Pure LLM score_batch failed")
        raise HTTPException(status_code=500, detail=f"Pure LLM scoring failed: {e}")

    n = len(results) or 1
    growth_sum = sum(float(r.get("growth", 0.0)) for r in results)
    risk_sum = sum(float(r.get("risk", 0.0)) for r in results)
    cost_sum = sum(float(r.get("cost_pressure", 0.0)) for r in results)
    pos_c = sum(1 for r in results if r.get("label") == "positive")
    neg_c = sum(1 for r in results if r.get("label") == "negative")
    neu_c = sum(1 for r in results if r.get("label") == "neutral")

    return LlmPureRescoreResponse(
        label=req.label,
        engine="PureLLMSignalEngine",
        engine_version=PURE_LLM_VERSION,
        ticker=record.get("ticker", "UNK"),
        form_type=record.get("form_type", "UNKNOWN"),
        filing_date=record.get("filing_date", "Unknown"),
        total_sentences=len(sentences),
        scanned_sentences=len(scanned),
        sentences=[
            LlmPureSentence(
                text=r.get("text", ""),
                method=r.get("method", "llm_pure"),
                label=r.get("label", "neutral"),
                net_score=float(r.get("net_score", 0.0)),
                growth=float(r.get("growth", 0.0)),
                risk=float(r.get("risk", 0.0)),
                cost_pressure=float(r.get("cost_pressure", 0.0)),
                reason=(r.get("reason") or r.get("llm_reason")),
            )
            for r in results
        ],
        llm_pure_growth_score=round(growth_sum / n, 4),
        llm_pure_risk_score=round(risk_sum / n, 4),
        llm_pure_cost_score=round(cost_sum / n, 4),
        llm_pure_net_score=round((growth_sum - risk_sum) / n, 4),
        llm_positive_count=pos_c,
        llm_negative_count=neg_c,
        llm_neutral_count=neu_c,
    )
