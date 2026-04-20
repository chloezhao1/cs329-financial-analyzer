"""
/api/sec/*

Thin wrappers around the SEC EDGAR collector and sector map for ad-hoc
lookups from the React frontend. Nothing heavy: no text downloads here.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from sec_edgar_collector import get_cik_from_ticker, get_filings_for_company
from sector_map import (
    SECTOR_MAP,
    SECTORS,
    sector_coverage,
    sector_for,
    tickers_in_sector,
)

logger = logging.getLogger(__name__)
router = APIRouter()

FormType = Literal["10-K", "10-Q", "8-K"]


@router.get("/cik/{ticker}")
def resolve_cik(ticker: str) -> dict:
    """Resolve a ticker to its SEC CIK using the public company_tickers feed."""
    cik = get_cik_from_ticker(ticker.upper())
    if not cik:
        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve CIK for ticker '{ticker}'.",
        )
    return {"ticker": ticker.upper(), "cik": cik}


@router.get("/filings")
def list_filings(
    ticker: str = Query(..., description="Uppercase ticker."),
    forms: list[FormType] = Query(default=["10-K", "10-Q"]),
    max_per_type: int = Query(4, ge=1, le=20),
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """
    List recent filing metadata for a ticker (no text downloads).
    Useful for the frontend to show "what's available" before running
    the full pipeline.
    """
    cik = get_cik_from_ticker(ticker.upper())
    if not cik:
        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve CIK for ticker '{ticker}'.",
        )
    filings = get_filings_for_company(
        cik=cik,
        form_types=list(forms),
        max_per_type=max_per_type,
        start_date=start_date,
        end_date=end_date,
    )
    for f in filings:
        f["ticker"] = ticker.upper()
    return filings


@router.get("/sectors")
def sectors_overview() -> dict:
    """Return the full sector map plus coverage counts and sector list."""
    return {
        "sectors": SECTORS,
        "map": SECTOR_MAP,
        "coverage": sector_coverage(),
    }


@router.get("/sectors/ticker/{ticker}")
def sector_lookup(ticker: str) -> dict:
    """Return the sector assigned to a single ticker (or 'Unknown')."""
    return {"ticker": ticker.upper(), "sector": sector_for(ticker)}


@router.get("/sectors/{sector}/tickers")
def sector_tickers(sector: str) -> dict:
    """Return every ticker mapped to a given sector label."""
    if sector not in SECTORS:
        raise HTTPException(
            status_code=404,
            detail=f"Sector '{sector}' not found. Valid: {SECTORS}",
        )
    return {"sector": sector, "tickers": tickers_in_sector(sector)}
