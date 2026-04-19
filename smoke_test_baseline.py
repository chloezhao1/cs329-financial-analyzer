"""
Smoke test for the baseline system.

Builds a fake baseline, applies it to mock analyses, and asserts the
z-score math is correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sector_map import sector_for, SECTOR_MAP, SECTORS
from apply_baseline import BaselineApplier
from build_baseline import compute_baseline


# ---------------------------------------------------------------------------
# 1. sector_map sanity
# ---------------------------------------------------------------------------
print("=== 1. SECTOR MAP ===")
print(f"  total tickers:  {len(SECTOR_MAP)}")
print(f"  sectors:        {SECTORS}")
assert sector_for("AAPL") == "Technology"
assert sector_for("aapl") == "Technology"   # case-insensitive
assert sector_for("JPM") == "Financials"
assert sector_for("NOTREAL") == "Unknown"
assert sector_for("") == "Unknown"
print("  [OK] sector_for works case-insensitively with unknown fallback")

# Every sector should have at least 4 companies (min for reliability)
from collections import Counter
sec_counts = Counter(SECTOR_MAP.values())
print(f"  per-sector counts: {dict(sec_counts)}")
for sec, n in sec_counts.items():
    assert n >= 4, f"Sector {sec} has only {n} tickers (need >=4)"
print("  [OK] every sector has >=4 companies")

# ---------------------------------------------------------------------------
# 2. compute_baseline math
# ---------------------------------------------------------------------------
print("\n=== 2. BASELINE COMPUTATION ===")

def mock_analysis(ticker: str, growth: float, risk: float) -> dict:
    return {
        "ticker":      ticker,
        "filing_date": "2026-01-01",
        "form_type":   "10-Q",
        "scores": {"growth": growth, "risk": risk, "cost_pressure": 0.0,
                   "net_operating_signal": growth - risk},
        "method":      {"engine_version": "0.3.0"},
    }

# Build mock analyses: tech cluster, financials cluster with different means
tech = [
    mock_analysis("AAPL",  0.20, 0.30),
    mock_analysis("MSFT",  0.15, 0.28),
    mock_analysis("GOOGL", 0.10, 0.25),
    mock_analysis("META",  0.05, 0.35),
]
fin = [
    mock_analysis("JPM", 0.08, 0.40),
    mock_analysis("BAC", 0.06, 0.45),
    mock_analysis("WFC", 0.10, 0.42),
    mock_analysis("GS",  0.05, 0.48),
]
all_mocks = tech + fin

baseline = compute_baseline(all_mocks)
print(f"  sectors in baseline: {sorted(baseline['sectors'].keys())}")
print(f"  n_total_records:     {baseline['n_total_records']}")

tech_stats = baseline["sectors"]["Technology"]
fin_stats = baseline["sectors"]["Financials"]

print(f"  Technology:  n={tech_stats['n']}  growth_mu={tech_stats['growth']['mean']}  "
      f"risk_mu={tech_stats['risk']['mean']}")
print(f"  Financials:  n={fin_stats['n']}  growth_mu={fin_stats['growth']['mean']}  "
      f"risk_mu={fin_stats['risk']['mean']}")

# Verify means
# Tech growth: (0.20 + 0.15 + 0.10 + 0.05) / 4 = 0.125
assert abs(tech_stats["growth"]["mean"] - 0.125) < 0.001, \
    f"expected 0.125, got {tech_stats['growth']['mean']}"
# Fin growth: (0.08 + 0.06 + 0.10 + 0.05) / 4 = 0.0725
assert abs(fin_stats["growth"]["mean"] - 0.0725) < 0.001
# Tech risk mean: 0.295
assert abs(tech_stats["risk"]["mean"] - 0.295) < 0.001
# Both reliable (n=4 >= min 4)
assert tech_stats["reliable"]
assert fin_stats["reliable"]
print("  [OK] per-sector means computed correctly")

# Sector sizes below min should be flagged
small_baseline = compute_baseline([
    mock_analysis("AAPL", 0.2, 0.3),
    mock_analysis("MSFT", 0.1, 0.2),
])
# Tech has n=2 here -- below the reliability threshold
assert not small_baseline["sectors"]["Technology"]["reliable"], \
    "n=2 sector should be flagged unreliable"
print("  [OK] sectors below min are flagged unreliable")

# ---------------------------------------------------------------------------
# 3. apply_baseline math
# ---------------------------------------------------------------------------
print("\n=== 3. APPLY BASELINE ===")

# Write our mock baseline to a temp file so BaselineApplier can load it
tmp_baseline = Path("/tmp/mock_baseline.json")
tmp_baseline.write_text(json.dumps(baseline))
applier = BaselineApplier(baseline_file=tmp_baseline)

# Test 1: AAPL should get Technology z-scores
aapl = mock_analysis("AAPL", 0.20, 0.30)
applier.apply(aapl)
print(f"  AAPL zscores: {aapl['zscores']}")
assert aapl["zscores"]["reference"] == "Technology"
assert aapl["zscores"]["reference_n"] == 4
assert aapl["zscores"]["reference_reliable"] is True

# Tech growth mean = 0.125, stdev ≈ 0.0645
# AAPL growth = 0.20, so z = (0.20 - 0.125) / 0.0645 ≈ +1.16
assert 0.8 < aapl["zscores"]["growth"] < 1.5, (
    f"AAPL z_growth should be ~+1.16, got {aapl['zscores']['growth']}"
)
print(f"  AAPL z_growth = {aapl['zscores']['growth']:+.2f}  (above Tech peers, as expected)  [OK]")

# Test 2: An unknown ticker should fall back to _corpus_all
sofi = mock_analysis("SOFI", 0.18, 0.29)
applier.apply(sofi)
print(f"  SOFI zscores: {sofi['zscores']}")
assert sofi["zscores"]["reference"] == "_corpus_all", (
    f"unknown ticker should fall back to _corpus_all, got {sofi['zscores']['reference']}"
)
print(f"  [OK] unknown ticker falls back to _corpus_all baseline")

# Test 3: A financials ticker should get Financials z-scores, different from tech
jpm = mock_analysis("JPM", 0.08, 0.40)
applier.apply(jpm)
print(f"  JPM zscores: {jpm['zscores']}")
assert jpm["zscores"]["reference"] == "Financials"
# JPM growth = 0.08, Fin mean = 0.0725, stdev ≈ 0.0222. z ≈ +0.34
# JPM risk = 0.40, Fin mean = 0.4375, stdev ≈ 0.0340. z ≈ -1.10
# Net z = 0.34 - (-1.10) = +1.44 (above peers on growth AND below peers on risk)
print(f"  JPM z_growth = {jpm['zscores']['growth']:+.2f}, "
      f"z_risk = {jpm['zscores']['risk']:+.2f}, "
      f"net_z = {jpm['zscores']['net_operating_signal']:+.2f}")
assert jpm["zscores"]["net_operating_signal"] > 0, (
    "JPM should score positive net z within Financials (above avg growth, below avg risk)"
)
print("  [OK] net_z rewards above-peer growth AND below-peer risk")

# Test 4: The WORST tech stock (META in the mock: low growth, high risk) should
# get negative z's on both
meta = mock_analysis("META", 0.05, 0.35)
applier.apply(meta)
print(f"  META zscores: {meta['zscores']}")
assert meta["zscores"]["growth"] < 0, "META growth (0.05) is below Tech mean (0.125) -> z should be negative"
# META risk = 0.35, Tech risk mean = 0.295, stdev ≈ 0.0436, so z_risk ≈ +1.26
assert meta["zscores"]["risk"] > 0, "META risk (0.35) is above Tech mean (0.295) -> z should be positive"
# Net z = negative growth - positive risk = very negative
assert meta["zscores"]["net_operating_signal"] < 0
print(f"  [OK] META (worst in Tech mock): net_z = {meta['zscores']['net_operating_signal']:+.2f}")

# Test 5: batch apply
batch = [mock_analysis("AAPL", 0.2, 0.3), mock_analysis("MSFT", 0.15, 0.28)]
enriched = applier.apply_all(batch)
assert all("zscores" in e for e in enriched)
print(f"  [OK] apply_all enriches batches")

# Test 6: missing baseline file
missing_applier = BaselineApplier(baseline_file=Path("/tmp/does_not_exist.json"))
result = mock_analysis("AAPL", 0.2, 0.3)
missing_applier.apply(result)
assert result["zscores"]["growth"] == 0.0
assert result["zscores"]["reference"] == "no_baseline"
print("  [OK] graceful degradation when baseline file missing")

# ---------------------------------------------------------------------------
# 4. Round-trip the baseline JSON
# ---------------------------------------------------------------------------
print("\n=== 4. JSON ROUND-TRIP ===")
serialized = json.dumps(baseline, indent=2)
reloaded = json.loads(serialized)
assert reloaded["sectors"]["Technology"]["growth"]["mean"] == 0.125
assert reloaded["_corpus_all"]["n"] == 8
print(f"  [OK] baseline JSON round-trips cleanly ({len(serialized):,} chars)")

print("\n" + "=" * 60)
print("ALL BASELINE SYSTEM TESTS PASSED")
print("=" * 60)
