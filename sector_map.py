"""
sector_map.py
=============

Ticker -> GICS sector mapping for baseline computation and z-score
normalization.

Currently covers ~150 large-cap US public companies across 5 sectors.

Tickers NOT in this map still get scored by the signal engine, but their
z-scores use the "_corpus_all" fallback baseline (the combined mean/stdev
across all 49 reference-corpus companies, spanning all sectors). The
z-score output explicitly labels which reference was used so users know.

Curation notes:
- The first ~10 tickers per sector are in the reference baseline and
  drive the per-sector statistics. Additional tickers route to the
  same sector's baseline at z-score time; they don't change the stats.
- Consumer Staples + Consumer Discretionary are merged here as
  "Consumer" to match our baseline sector labeling.
- Amazon is placed in Technology despite being GICS Consumer
  Discretionary, because its disclosure language is more tech-like.
- Tesla is placed in Consumer (GICS Consumer Discretionary).
- Boundaries between sectors are inherently fuzzy; some companies
  could defensibly belong elsewhere. This is documented as a limitation.
"""
from __future__ import annotations

from collections import Counter

SECTOR_MAP: dict[str, str] = {
    # --- Technology (~30 tickers) ---
    "AAPL":  "Technology",   # baseline
    "MSFT":  "Technology",   # baseline
    "GOOGL": "Technology",   # baseline
    "META":  "Technology",   # baseline
    "NVDA":  "Technology",   # baseline
    "AMZN":  "Technology",   # baseline (Consumer-Disc per GICS, but tech-like)
    "ORCL":  "Technology",   # baseline
    "CRM":   "Technology",   # baseline
    "ADBE":  "Technology",   # baseline
    "CSCO":  "Technology",   # baseline
    "GOOG":  "Technology",
    "AMD":   "Technology",
    "AVGO":  "Technology",
    "INTC":  "Technology",
    "QCOM":  "Technology",
    "TXN":   "Technology",
    "IBM":   "Technology",
    "NFLX":  "Technology",
    "SHOP":  "Technology",
    "SNOW":  "Technology",
    "PLTR":  "Technology",
    "UBER":  "Technology",
    "ABNB":  "Technology",
    "SPOT":  "Technology",
    "COIN":  "Technology",
    "SQ":    "Technology",
    "PYPL":  "Technology",
    "INTU":  "Technology",
    "NOW":   "Technology",
    "MU":    "Technology",

    # --- Financials (~30 tickers) ---
    "JPM":   "Financials",   # baseline
    "BAC":   "Financials",   # baseline
    "WFC":   "Financials",   # baseline
    "GS":    "Financials",   # baseline
    "MS":    "Financials",   # baseline
    "C":     "Financials",   # baseline
    "BLK":   "Financials",   # baseline
    "AXP":   "Financials",   # baseline
    "V":     "Financials",   # baseline
    "MA":    "Financials",   # baseline
    "BRK.B": "Financials",
    "USB":   "Financials",
    "PNC":   "Financials",
    "TFC":   "Financials",
    "SCHW":  "Financials",
    "CB":    "Financials",
    "PGR":   "Financials",
    "AIG":   "Financials",
    "ALL":   "Financials",
    "MET":   "Financials",
    "PRU":   "Financials",
    "COF":   "Financials",
    "TRV":   "Financials",
    "AFL":   "Financials",
    "HIG":   "Financials",
    "SOFI":  "Financials",
    "HOOD":  "Financials",
    "FIS":   "Financials",
    "FITB":  "Financials",

    # --- Healthcare (~30 tickers) ---
    "JNJ":   "Healthcare",   # baseline
    "PFE":   "Healthcare",   # baseline
    "UNH":   "Healthcare",   # baseline
    "LLY":   "Healthcare",   # baseline
    "MRK":   "Healthcare",   # baseline
    "ABBV":  "Healthcare",   # baseline
    "TMO":   "Healthcare",   # baseline
    "DHR":   "Healthcare",   # baseline
    "BMY":   "Healthcare",   # baseline
    "AMGN":  "Healthcare",   # baseline
    "ABT":   "Healthcare",
    "MDT":   "Healthcare",
    "GILD":  "Healthcare",
    "CVS":   "Healthcare",
    "CI":    "Healthcare",
    "ELV":   "Healthcare",
    "HUM":   "Healthcare",
    "ISRG":  "Healthcare",
    "SYK":   "Healthcare",
    "BSX":   "Healthcare",
    "REGN":  "Healthcare",
    "VRTX":  "Healthcare",
    "BIIB":  "Healthcare",
    "MRNA":  "Healthcare",
    "GEHC":  "Healthcare",
    "EW":    "Healthcare",
    "ZTS":   "Healthcare",
    "IQV":   "Healthcare",
    "MCK":   "Healthcare",

    # --- Consumer (~30 tickers, staples + discretionary) ---
    "WMT":   "Consumer",     # baseline
    "COST":  "Consumer",     # baseline
    "HD":    "Consumer",     # baseline
    "MCD":   "Consumer",     # baseline
    "NKE":   "Consumer",     # baseline
    "SBUX":  "Consumer",     # baseline
    "TGT":   "Consumer",     # baseline
    "KO":    "Consumer",     # baseline
    "PEP":   "Consumer",     # baseline
    "PG":    "Consumer",     # baseline
    "TSLA":  "Consumer",     # GICS Consumer Discretionary
    "DIS":   "Consumer",
    "LOW":   "Consumer",
    "CMG":   "Consumer",
    "BKNG":  "Consumer",
    "MDLZ":  "Consumer",
    "MO":    "Consumer",
    "PM":    "Consumer",
    "CL":    "Consumer",
    "KMB":   "Consumer",
    "EL":    "Consumer",
    "F":     "Consumer",
    "GM":    "Consumer",
    "TJX":   "Consumer",
    "ROST":  "Consumer",
    "DG":    "Consumer",
    "DLTR":  "Consumer",
    "YUM":   "Consumer",
    "MAR":   "Consumer",
    "COTY":  "Consumer",

    # --- Energy (~30 tickers) ---
    "XOM":   "Energy",       # baseline
    "CVX":   "Energy",       # baseline
    "COP":   "Energy",       # baseline
    "SLB":   "Energy",       # baseline
    "EOG":   "Energy",       # baseline
    "MPC":   "Energy",       # baseline
    "PSX":   "Energy",       # baseline
    "OXY":   "Energy",       # baseline
    "HES":   "Energy",       # baseline (may not resolve via SEC API)
    "VLO":   "Energy",       # baseline
    "KMI":   "Energy",
    "WMB":   "Energy",
    "MRO":   "Energy",
    "DVN":   "Energy",
    "FANG":  "Energy",
    "HAL":   "Energy",
    "BKR":   "Energy",
    "APA":   "Energy",
    "OKE":   "Energy",
    "ET":    "Energy",
    "EPD":   "Energy",
    "TRGP":  "Energy",
    "LNG":   "Energy",
    "CNQ":   "Energy",
    "SU":    "Energy",
    "CTRA":  "Energy",
    "RRC":   "Energy",
    "EQT":   "Energy",
    "CHK":   "Energy",
    "AR":    "Energy",       # Antero Resources
}


BASELINE_TICKERS: list[str] = list(SECTOR_MAP.keys())

SECTORS: list[str] = sorted(set(SECTOR_MAP.values()))

UNKNOWN_SECTOR = "Unknown"


def sector_for(ticker: str) -> str:
    """Return the sector for a ticker, or 'Unknown' if not mapped."""
    if not ticker:
        return UNKNOWN_SECTOR
    return SECTOR_MAP.get(ticker.upper(), UNKNOWN_SECTOR)


def tickers_in_sector(sector: str) -> list[str]:
    """Return all tickers currently mapped to the given sector."""
    return [t for t, s in SECTOR_MAP.items() if s == sector]


def sector_coverage() -> dict[str, int]:
    """Return {sector: count} of tickers covered."""
    return dict(Counter(SECTOR_MAP.values()))
