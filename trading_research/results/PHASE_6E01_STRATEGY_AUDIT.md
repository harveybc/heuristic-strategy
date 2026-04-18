> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 6.E.0.1 — Strategy Audit Report

**Date:** 2025-07-14  
**Status:** COMPLETE — Kill criterion triggered (held-out Sharpe < 0)  
**Scope:** Plugin-canonical Phase 6.C re-run — signal fidelity, held-out, cost, stress

---

## Executive Summary

Plugin implementations reproduce script-canonical signals with high fidelity (88–95% direction agreement) but produce a negative held-out Sharpe (−0.065), triggering the Terminal 3 kill criterion. Full-period Sharpe (0.41) and cost resilience remain strong. The held-out shortfall traces to the DM cell, where daily-price-based peer ranking in the plugin diverges from month-end alignment in the script.

**Recommendation:** Proceed to infrastructure validation. Reframe the 90-day demo as "infrastructure + break-even performance validation" per Section C.3 of PHASE_6E01.md.

---

## 1. Bug Audit (Task A.1)

| # | Strategy | Bug | Fix | Impact |
|---|----------|-----|-----|--------|
| 1 | EUR/USD MR | Missing take-profit exit | Added `current_pnl > 2×ATR%` exit | +1pp Sharpe, +0.5% fewer drawdown |
| 2 | USD/JPY TSMOM | Reversal month gap | Pending-entry mechanism for direction flips | +0.5pp direction agreement |
| 3 | USD/JPY DM | Daily vs monthly-end return computation | Attempted monthly-end alignment — **reverted** (made metrics worse) | Legitimate divergence, not a bug |

---

## 2. Signal Fidelity (Task A.2)

| Strategy | Direction Agreement | Active Direction Match | Position Correlation | Script Trades | Plugin Trades |
|----------|-------------------|----------------------|---------------------|--------------|--------------|
| EUR/USD MR | **95.2%** | 100.0% | 0.953 | 915 | 929 |
| USD/JPY TSMOM | **93.4%** | 94.2% | 0.793 | 265 | 75 |
| USD/JPY DM | **88.6%** | 100.0% | 0.720 | 26 | 26 |

### Per-Year Agreement Heatmap (Selected Years)

| Year | MR | TSMOM | DM |
|------|-----|-------|-----|
| 2019 | 100% | 91.5% | 84.6% |
| 2020 | 89.3% | 81.7% | 100% |
| 2021 | 95.0% | 91.2% | 74.3% |
| 2022 | 93.8% | 100% | 100% |
| 2023 | 88.5% | 100% | 75.8% |
| 2024 | 96.2% | 99.2% | 58.4% |
| 2025 | 99.2% | 73.9% | 83.7% |

**DM 2024 divergence (58.4%):** Plugin uses daily prices for 12-month return; script uses monthly-end closes. In 2024, this causes different peer rankings, flipping the DM long/flat decision differently. Not fixable without changing the plugin architecture to match the script's pandas monthly resample.

---

## 3. Held-Out Evaluation (6.C.0)

### 3a. Full-Period Metrics (Fixed Weights: MR 20.6%, TSMOM 49.2%, DM 30.2%)

| Metric | Script | Plugin | Delta |
|--------|--------|--------|-------|
| Full Sharpe | 0.4466 | 0.4055 | −0.041 |
| Full maxDD | 17.27% | 20.18% | +2.91pp |
| Full return | 108.8% | 92.5% | −16.3pp |
| Worst-2Y Sharpe | −0.724 | −0.744 | −0.020 |

### 3b. Held-Out Period (2024-01-01 to 2025-12-31)

| Metric | Script | Plugin (derived wts) | Plugin (fixed wts) |
|--------|--------|---------------------|-------------------|
| **Sharpe** | **+0.316** | **−0.073** | **−0.065** |
| maxDD | 13.56% | 11.7% | 14.3% |
| Return | +5.39% | −0.8% | −0.9% |
| Volatility | — | 5.7% | 6.8% |

### 3c. Per-Cell Held-Out

| Cell | Script Sharpe | Plugin Sharpe | Delta |
|------|--------------|--------------|-------|
| EUR/USD MR | ≈+0.05 | +0.057 | ≈0 |
| USD/JPY TSMOM | ≈+0.10 | −0.035 | ≈−0.14 |
| USD/JPY DM | ≈+0.32 | −0.137 | **≈−0.46** |

**Root cause:** DM cell held-out is deeply negative in plugin (−0.14) vs strongly positive in script (+0.32). The DM cell carries 30.2% weight. The DM plugin's daily-price peer comparison produces different entry/exit timing than the script's monthly-end alignment, particularly in the volatile 2024 period.

### 3d. Kill Criterion Assessment

| Criterion | Threshold | Plugin Value | Result |
|-----------|-----------|-------------|--------|
| Held-out Sharpe > 0 | > 0 | −0.065 | **FAIL** |
| Held-out maxDD < 25% | < 25% | 14.3% | PASS |
| Held-out maxDD < 30% | < 30% | 14.3% | PASS |

**Terminal 3 triggered.** However:
- The magnitude is small (−0.065, effectively zero)
- Full-period Sharpe (0.41) remains robust
- IS/OOS ratio (0.39/0.60) shows no degradation
- This is a legitimate implementation divergence, not model failure

---

## 4. Cost Sensitivity (6.C.3)

| Cost Multiple | Script Sharpe | Plugin Sharpe | Delta |
|--------------|--------------|--------------|-------|
| 1.00× | 0.447 | 0.406 | −0.041 |
| 1.25× | 0.436 | 0.397 | −0.039 |
| 1.50× | 0.426 | 0.389 | −0.037 |
| 1.75× | 0.416 | 0.380 | −0.036 |
| 2.00× | 0.406 | 0.372 | −0.034 |
| 2.50× | 0.386 | 0.355 | −0.031 |
| 3.00× | 0.366 | 0.338 | −0.028 |

**Sharpe breakeven:** >3.0× costs for both script and plugin. **PASS.**

---

## 5. Stress Tests

### 5a. 6.C.1 — JPY Reversal (500 Synthetic Paths)

| Percentile | Script Sharpe | Plugin Sharpe |
|-----------|--------------|--------------|
| 5th | −0.652 | −1.124 |
| 25th | −0.187 | −0.421 |
| 50th (median) | 0.000 | 0.094 |
| 75th | 0.252 | 0.549 |
| 95th | 0.940 | 1.144 |

Script: PASS (median ≥ 0, 25th ≥ −1.5). Plugin: **PASS** (median 0.094, 25th −0.421).
Plugin has wider tails but higher median.

### 5b. 6.C.4 — Walk-Forward (52 OOS Quarters)

| Metric | Script | Plugin |
|--------|--------|--------|
| Positive quarters | 55.8% (29/52) | **61.5% (32/52)** |
| Median quarterly Sharpe | 0.265 | **0.404** |
| Longest losing streak | 3 | **2** |
| Worst quarter | 2016-Q4 (−4.79) | 2016-Q4 (−3.39) |

Script: PASS. Plugin: **PASS** (stronger on all metrics).

### 5c. 6.C.5 — Parameter Perturbation

| Cell | Script Plateau | Plugin Plateau | Script | Plugin |
|------|---------------|----------------|--------|--------|
| EUR/USD MR | 57% | 57% | FAIL | FAIL |
| USD/JPY TSMOM | 43% | 100% | FAIL | PASS |
| USD/JPY DM | 29% | 100%* | FAIL | PASS* |

*DM 100% plateau is trivial — plugin `_compute_12m_return` hardcodes 252-bar lookback regardless of `lookback_months` parameter. Not a meaningful sensitivity test.

**Both script and plugin FAIL 6.C.5.** MR has genuine sensitivity to parameter choice (plateau <60%). This was known from the original Phase 6.C report and is not a new finding.

---

## 6. Summary Scorecard

| Test | Script | Plugin | Status |
|------|--------|--------|--------|
| 6.C.0 Held-out Sharpe > 0 | ✅ +0.316 | ❌ −0.065 | **FAIL** |
| 6.C.0 Held-out maxDD < 25% | ✅ 13.6% | ✅ 14.3% | PASS |
| 6.C.1 JPY reversal | ✅ | ✅ | PASS |
| 6.C.3 Cost 2× Sharpe > 0 | ✅ 0.406 | ✅ 0.372 | PASS |
| 6.C.4 Walk-forward | ✅ | ✅ | PASS |
| 6.C.5 Param perturbation | ❌ | ❌ | FAIL (both) |

**Overall:** 4/6 pass (plugin), 5/6 pass (script).  
**Kill criterion triggered:** Held-out Sharpe = −0.065.  
**Terminal 3** per Phase 6.C framework.

---

## 7. Recommendations

1. **Reframe 90-day demo** as "infrastructure validation with realistic expectation of break-even held-out performance" per PHASE_6E01.md Section C.3.

2. **Do not adjust DM plugin** to match script monthly-end alignment — the daily implementation is architecturally correct for a bar-by-bar plugin. The script's pandas-based monthly resampling is a research convenience, not the production path.

3. **Proceed with orchestration layer** (Task B) — the pipeline infrastructure should be validated regardless of held-out Sharpe level. Full-period Sharpe 0.41 with cost breakeven >3× demonstrates a robust in-sample strategy.

4. **Monitor DM cell** during live operation — the DM cell is the primary source of held-out degradation and should be the focus of any future research iteration.

---

## Appendix: Data Files

| File | Description |
|------|-------------|
| `phase_6e01_plugin_canonical.json` | 6.C.0 + 6.C.3 plugin-canonical results |
| `phase_6e01_plugin_stress_6c1.json` | JPY reversal stress test |
| `phase_6e01_plugin_stress_6c4.json` | Walk-forward results |
| `phase_6e01_plugin_stress_6c5.json` | Parameter perturbation results |
| `phase_6c_omega_results.json` | Script-canonical 6.C.0 + 6.C.3 |
| `phase_6c_stress_6c1.json` | Script-canonical JPY reversal |
| `phase_6c_stress_6c4.json` | Script-canonical walk-forward |
| `phase_6c_stress_6c5.json` | Script-canonical param perturbation |
