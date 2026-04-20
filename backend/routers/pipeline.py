"""
/api/pipeline/*

Wraps the long-running scraping + preprocessing pipeline (`run_full_pipeline`)
so the React frontend can trigger it and then refresh its analyses.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from financial_signal_engine import analyze_records
from run_pipeline import run_full_pipeline

from backend.engine_factory import load_default_engine
from backend.lm_path import resolve_lm_csv
from backend.safe_console import patch_print_for_console
from backend.routers.signals import get_cache

logger = logging.getLogger(__name__)
router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


FormType = Literal["10-K", "10-Q", "EARNINGS_CALL"]


class RunPipelineRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="Uppercase tickers.")
    form_types: list[FormType] = Field(
        default_factory=lambda: ["10-K", "10-Q"],
        description="SEC form types; EARNINGS_CALL routes to the transcript scraper.",
    )
    max_per_type: int = Field(2, ge=1, le=8, description="Max docs per form type per ticker.")
    skip_sec: bool = Field(False, description="Skip SEC EDGAR collection.")
    skip_transcripts: bool = Field(
        True,
        description="Skip transcript scraping (recommended: requires Selenium/Chrome).",
    )
    start_date: date | None = None
    end_date: date | None = None
    kaggle_pkl: str | None = Field(
        None, description="Optional Kaggle transcripts .pkl path for bulk import."
    )


class RunPipelineResponse(BaseModel):
    n_records: int
    n_analyses: int
    tickers: list[str]
    form_types: list[str]


@router.post("/run", response_model=RunPipelineResponse)
def run_pipeline(req: RunPipelineRequest) -> RunPipelineResponse:
    """
    Trigger `run_full_pipeline` synchronously, then invalidate the analyses
    cache so subsequent `/api/signals/analyses` calls reflect new data.

    NOTE: This endpoint is long-running (minutes). For production you would
    move it behind a task queue; for a local dev dashboard, sync is fine.
    """
    tickers = [t.strip().upper() for t in req.tickers if t and t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No usable tickers supplied.")

    sec_forms = [f for f in req.form_types if f != "EARNINGS_CALL"]
    include_transcripts = "EARNINGS_CALL" in req.form_types and not req.skip_transcripts

    logger.info(
        "Pipeline run requested: tickers=%s forms=%s max=%d skip_sec=%s skip_transcripts=%s",
        tickers, req.form_types, req.max_per_type, req.skip_sec, not include_transcripts,
    )

    try:
        with patch_print_for_console():
            records = run_full_pipeline(
                tickers=tickers,
                form_types=sec_forms or ["10-K", "10-Q"],
                max_per_type=req.max_per_type,
                kaggle_pkl=req.kaggle_pkl,
                skip_sec=req.skip_sec or not bool(sec_forms),
                skip_transcripts=not include_transcripts,
                start_date=req.start_date,
                end_date=req.end_date,
            )
            if records:
                engine = load_default_engine(resolve_lm_csv(PROJECT_ROOT))
                analyses = analyze_records(records, engine=engine)
            else:
                analyses = []
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=f"Pipeline run failed: {e}")

    cache = get_cache()
    cache.invalidate()

    return RunPipelineResponse(
        n_records=len(records),
        n_analyses=len(analyses),
        tickers=tickers,
        form_types=list(req.form_types),
    )
