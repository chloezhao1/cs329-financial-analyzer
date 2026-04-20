"""
/api/evaluation/*

Reads cached PhraseBank evaluation results and optionally triggers a fresh
evaluation run by calling `evaluate.run_evaluation`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_OUT = PROJECT_ROOT / "data" / "eval_results.json"


@router.get("/latest")
def latest_results() -> dict:
    """Return the most recent evaluation JSON or a 404 if none exists."""
    if not DEFAULT_EVAL_OUT.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation results yet. Run the evaluation first "
                "(POST /api/evaluation/run or `python evaluate.py`)."
            ),
        )
    try:
        return json.loads(DEFAULT_EVAL_OUT.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse eval JSON: {e}")


class RunEvaluationRequest(BaseModel):
    threshold: float = Field(
        0.1,
        ge=0.0,
        le=2.0,
        description="Net-signal threshold used for positive/negative classification.",
    )


@router.post("/run")
def run_evaluation_endpoint(req: RunEvaluationRequest) -> dict:
    """
    Run the PhraseBank evaluation synchronously.

    WARNING: This loads spaCy + the Financial PhraseBank dataset; the first
    call can take a couple of minutes. Not suitable for a live UI button
    without a loading state. Returns the same JSON written to disk.
    """
    try:
        from evaluate import run_evaluation  # noqa: WPS433

        from backend.lm_path import resolve_lm_csv
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import evaluation deps (likely missing packages): {e}",
        )

    try:
        lm_csv = resolve_lm_csv(PROJECT_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    DEFAULT_EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)

    try:
        run_evaluation(lm_csv=lm_csv, threshold=req.threshold, out_path=DEFAULT_EVAL_OUT)
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    try:
        return json.loads(DEFAULT_EVAL_OUT.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation ran but output JSON could not be read: {e}",
        )
