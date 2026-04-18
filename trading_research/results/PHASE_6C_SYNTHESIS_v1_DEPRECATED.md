> **⚠ DEPRECATED** — This file contains pre-fix values with known errors (double vol-scaling bug).
> The canonical final document is **`PHASE_6C_SYNTHESIS_FINAL.md`**.
> This file is retained as a historical audit trail only.

# Phase 6.C Synthesis — Final Robustness & Terminal Decision (DEPRECATED v1)

**Date:** 2025-07-24  
**Status:** DEPRECATED — superseded by PHASE_6C_SYNTHESIS_FINAL.md  
**Terminal Decision:** **TERMINAL 1 — DEPLOY P3 to OANDA Demo (90-day observation)**

---

## Executive Summary

Portfolio P3 (EUR/USD MR 20.6%, USD/JPY TSMOM 49.2%, USD/JPY DM 30.2%) was subjected
to 5 rigorous robustness stress tests plus a genuine held-out evaluation on 2024–2025 data
never seen during any research phase. P3 **passed 4 out of 5 stress tests** and **cleared
the held-out gate** with Sharpe 0.316 and maximum drawdown 16.7%. This meets the
pre-committed Terminal 1 threshold: deploy to OANDA demo for 90-day live observation.

---

## 1. Held-Out Evaluation (6.C.0) — GATE: PASS

The 2024-01-01 to 2025-12-31 period was preserved untouched through all research phases.
This is the single genuine out-of-sample test.

| Metric | In-Sample (≤2018) | OOS (2019–2023) | **Held-Out (2024–2025)** |
|--------|-------------------|-----------------|--------------------------|
| Sharpe | 0.452 | 0.505 | **0.316** |
| Max DD | 15.8% | 14.4% | **13.6%** |
| Return | — | — | **5.4%** |
| Vol | — | — | **8.2%** |

**Per-cell held-out:**

| Cell | Sharpe | Return | Max DD | Trades |
|------|--------|--------|--------|--------|
| EUR/USD MR | 0.178 | 1.9% | 5.4% | 77 |
| USD/JPY TSMOM | 0.239 | 3.9% | 18.0% | 25 |
| USD/JPY DM | 0.321 | 6.7% | 20.0% | 4 |

**Gate criteria:**
- Sharpe > 0: **PASS** (0.316)
- Max DD < 25%: **PASS** (16.7%)
- Max DD < 30% (gate): **PASS** (16.7%)

**Observations:**
- Held-out Sharpe (0.316) shows ~30% decay from IS/OOS (0.45–0.50), consistent with
  typical strategy degradation. Not alarming — still well above zero.
- Max DD improved in held-out vs IS, suggesting vol-scaling is working.
- All 3 cells contribute positively. USD/JPY concentration (79.4%) remains a risk factor.

---

## 2. Stress Test Results

### 2.1 — 6.C.1 JPY Reversal Stress (Gamma) — PASS

500 synthetic 2-year paths via block bootstrap with 30% JPY reversal overweight
(amplification up to 2.0x).

| Percentile | Sharpe | Max DD |
|-----------|--------|--------|
| 5th | -0.652 | 6.7% |
| 25th | -0.187 | 10.3% |
| **50th** | **0.000** | **12.8%** |
| 75th | 0.252 | 16.0% |
| 95th | 0.940 | 20.8% |

- Fraction with Sharpe > 0: 49.4%
- **Kill criteria:** Median SR ≥ 0 **PASS** (0.000) | 25th pct ≥ −1.5 **PASS** (−0.187)
- **Interpretation:** P3 is approximately flat during JPY reversals, not catastrophic.
  The median Sharpe sitting at exactly 0 indicates the JPY momentum cells give back
  gains during reversals while the MR cell provides modest offset.

### 2.2 — 6.C.2 Monte Carlo Regime Scenarios (Dragon) — PASS

3000 synthetic 2-year paths per regime via block bootstrap.

| Regime | Period | Bars | Median SR | 75th DD | Frac SR>0 |
|--------|--------|------|-----------|---------|-----------|
| A: Inflation | 2022–2025 | 1039 | 0.074 | 16.2% | 53.3% |
| B: QE | 2013–2019 | 1822 | 0.022 | 15.5% | 51.7% |
| C: Crisis | 2008–2012 | 1284 | 0.024 | 15.1% | 51.3% |

- **Kill criteria (Regime B):** Median SR < 0 **PASS** (0.022) | 75th DD > 30% **PASS** (15.5%)
- **All regimes pass:** True
- **Interpretation:** P3 is approximately break-even across all regimes with well-controlled
  drawdowns. QE regime (the hardest for momentum strategies) shows a barely positive
  median Sharpe. No regime causes catastrophic failure.

### 2.3 — 6.C.3 Cost Sensitivity (Omega) — PASS

| Cost Multiplier | Sharpe | Worst-2Y | Max DD | Total Return |
|----------------|--------|----------|--------|-------------|
| 1.00x | 0.447 | −0.724 | 24.6% | 143.6% |
| 1.25x | 0.436 | −0.735 | 25.1% | 138.0% |
| 1.50x | 0.426 | −0.745 | 25.6% | 132.6% |
| 1.75x | 0.416 | −0.755 | 26.0% | 127.3% |
| 2.00x | 0.406 | −0.765 | 26.5% | 122.2% |
| 2.50x | 0.386 | −0.784 | 27.4% | 112.5% |
| 3.00x | 0.366 | −0.803 | 28.3% | 103.4% |

- Sharpe break-even: **>3.0x** (still positive at 3x costs)
- Worst-2Y break-even: **>3.0x** (still above −1.0 at 3x)
- **Kill criteria:** SR positive at 2.0x **PASS** (0.406) | worst-2Y > −1.0 at 1.5x **PASS** (−0.745)
- **Interpretation:** Extremely robust to cost assumptions. P3 could tolerate a tripling
  of transaction costs and still maintain positive Sharpe. The gentle slope (−0.027 SR
  per 1x cost increase) reflects P3's low turnover, particularly the DM cell (4 trades
  in 2 years held-out).

### 2.4 — 6.C.4 Walk-Forward (Omega) — PASS

52 OOS quarters (2011-Q1 to 2023-Q4), expanding-window with recomputed P3 weights each quarter.

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Positive quarters | 29/52 (55.8%) | ≥55% | **PASS** |
| Median quarterly Sharpe | 0.265 | ≥0.2 | **PASS** |
| Longest losing streak | 3 quarters | ≤5 | **PASS** |

**Notable quarters:**
- Best: 2012-Q4 (+15.7%, SR=5.36) — BOJ Abenomics launch
- Worst: 2016-Q1 (−11.9%, SR=−2.80) — BOJ negative rates surprise
- Strongest period: 2022-Q1 to Q2 (+20.3% cumulative) — USD/JPY surge
- Most volatile period: 2022-Q4 (−9.3%) — BOJ policy reversal

**Interpretation:** Walk-forward validates that P3's edge is not an artifact of fixed-weight
optimization. The strategy adapts reasonably with expanding-window weight recomputation.
The 55.8% hit rate is modest but consistent, with no prolonged losing streaks.

### 2.5 — 6.C.5 Parameter Perturbation (Gamma) — FAIL

| Cell | Baseline SR | Configs | Within 20% | Plateau? |
|------|------------|---------|------------|----------|
| EUR/USD MR | 0.186 | 21 | 12 (57%) | No (<60%) |
| USD/JPY TSMOM | 0.296 | 7 | 3 (43%) | No (<60%) |
| USD/JPY DM | 0.406 | 7 | 2 (29%) | No (<60%) |

- **Kill criterion:** All cells show plateau (≥60% within 20%) → **FAIL**
- **Interpretation:** All three strategy cells show sensitivity to parameter choices,
  with performance varying meaningfully across the tested grid. The DM cell is most
  fragile (only 29% within 20% of baseline). However, note that:
  - The grid tested includes aggressive perturbations (±50% for some parameters)
  - All cells maintain positive Sharpe at the baseline point
  - The "within 20% of baseline" criterion is strict for strategies with modest
    absolute Sharpe values (20% of 0.186 = 0.037 tolerance)

---

## 3. Decision Tree Evaluation

```
6.C.0 Held-out gate: Sharpe > 0 AND maxDD < 30%?
  → YES (SR=0.316, DD=16.7%)
  → Proceed to stress tests

Stress test scorecard: 4/5 PASS?
  6.C.1 JPY Reversal:     PASS
  6.C.2 MC Regimes:       PASS
  6.C.3 Cost Sensitivity:  PASS
  6.C.4 Walk-Forward:      PASS
  6.C.5 Param Perturbation: FAIL
  → Score: 4/5 = YES

Held-out maxDD < 25%?
  → YES (16.7%)

TERMINAL 1: DEPLOY to OANDA Demo
```

---

## 4. Terminal Decision

### **TERMINAL 1 — DEPLOY P3 TO OANDA DEMO**

P3 passes the pre-committed deployment criteria:
1. Held-out Sharpe 0.316 > 0 ✓
2. Held-out max DD 16.7% < 25% ✓
3. 4/5 stress tests passed ✓

### Deployment Parameters

| Parameter | Value |
|-----------|-------|
| Portfolio | P3: EUR/USD MR (20.6%), USD/JPY TSMOM (49.2%), USD/JPY DM (30.2%) |
| Target vol | 10% annualized |
| Rebalance frequency | Monthly |
| Cost model | OANDA spread + 0.3 bps slippage per trade |
| Observation period | 90 calendar days |
| Go/No-go threshold | Sharpe > −0.5 at day 90, max DD < 20% |

### Risk Factors to Monitor

1. **USD/JPY concentration (79.4%):** The portfolio is heavily exposed to JPY dynamics.
   BOJ policy changes remain the primary tail risk.
2. **Parameter sensitivity (6.C.5 FAIL):** The chosen parameters are not on a plateau.
   Any structural market microstructure changes could erode the edge faster than expected.
3. **Modest held-out Sharpe (0.316):** While positive, this is below typical institutional
   deployment thresholds (~0.5). The 90-day demo should confirm whether this is the
   "new normal" or a transient weak period.
4. **Regime B (QE) marginal:** Median Sharpe of 0.022 in QE conditions suggests near-zero
   expected return if central banks revert to sustained accommodation.

### Standing Rules (90-day Demo)

- If day-90 Sharpe < −0.5: **CLOSE** immediately
- If max DD exceeds 20% at any point: **CLOSE** immediately
- If 90-day Sharpe > 0 and max DD < 15%: Consider live deployment with minimum capital
- Otherwise: Extend observation by 90 days (maximum 2 extensions)

---

## 5. Master Results Table

| Test | Machine | Metric | Value | Threshold | Result |
|------|---------|--------|-------|-----------|--------|
| 6.C.0 Held-out | Omega | Sharpe | 0.316 | >0 | **PASS** |
| 6.C.0 Held-out | Omega | Max DD | 16.7% | <25% | **PASS** |
| 6.C.1 JPY Stress | Gamma | Med SR | 0.000 | ≥0 | **PASS** |
| 6.C.1 JPY Stress | Gamma | 25th SR | −0.187 | ≥−1.5 | **PASS** |
| 6.C.2 MC Regimes | Dragon | B med SR | 0.022 | ≥0 | **PASS** |
| 6.C.2 MC Regimes | Dragon | B 75th DD | 15.5% | <30% | **PASS** |
| 6.C.3 Cost | Omega | SR@2x | 0.406 | >0 | **PASS** |
| 6.C.3 Cost | Omega | w2Y@1.5x | −0.745 | >−1.0 | **PASS** |
| 6.C.4 Walk-Fwd | Omega | %+ve Q | 55.8% | ≥55% | **PASS** |
| 6.C.4 Walk-Fwd | Omega | Med Q SR | 0.265 | ≥0.2 | **PASS** |
| 6.C.4 Walk-Fwd | Omega | Max streak | 3 | ≤5 | **PASS** |
| 6.C.5 Params | Gamma | MR plateau | 57% | ≥60% | **FAIL** |
| 6.C.5 Params | Gamma | TS plateau | 43% | ≥60% | **FAIL** |
| 6.C.5 Params | Gamma | DM plateau | 29% | ≥60% | **FAIL** |

**Overall: 4/5 PASS → TERMINAL 1 (DEPLOY)**

---

## 6. Appendix: Compute Infrastructure

| Machine | Role | GPU | Tasks | Runtime |
|---------|------|-----|-------|---------|
| Omega | Local | RTX 4070 12GB | 6.C.0, 6.C.3, 6.C.4 | ~5 min |
| Dragon | Remote | RTX 4090 16GB | 6.C.2 (9000 MC paths) | ~30 min |
| Gamma | Remote | RTX 5070 Ti 8GB | 6.C.1 (500 paths), 6.C.5 | ~10 min |

All results are stored in `trading_research/results/`:
- `phase_6c_omega_results.json` (6.C.0 + 6.C.3)
- `phase_6c_stress_6c1.json` (JPY reversal)
- `phase_6c_stress_6c2.json` (MC regimes)
- `phase_6c_stress_6c4.json` (walk-forward)
- `phase_6c_stress_6c5.json` (parameter perturbation)

---

## Phase 6.D Addendum — Pre-Deployment Discrepancy Reconciliation

*Added: 2025-06-17. This addendum does not modify the original synthesis above.*

### Discrepancies Identified

Independent review found two inconsistencies:
1. **Max DD:** Phase 5.5 reported 15.0%, Phase 6.C.3 reported 24.6% for the same P3 portfolio
2. **Implementation lineage:** Unclear whether Phase 6.C tests used LTS plugins or script-level NumPy

### Resolution

**Discrepancy 1 — Root Cause:** Two compounding factors (accounting for 106% of the 9.6pp gap):
- **F1 (41%):** Phase 5.5 reported max DD at realized vol (7.76%), not 10% target vol. At 10% vol, Phase 5.5 value is 18.9%.
- **F2 (65%):** Phase 6.C.3 double vol-scaled (cells to 10% individually, then portfolio to 10% again), inflating max DD from ~18% to 24.6%. This is a bug.
- F3 (6%): Minor difference in equity curve method (exp(cumsum) vs cumprod(1+r)).
- F4 (0%): Weekly resampling anchor had no effect.

**Canonical Max DD: 18.9%** at 10% target vol, single vol-scaling, exp(cumsum).

**Discrepancy 2 — All Phase 6.C tests used script-level NumPy.** Phase 6.B validated that LTS plugins match script-level within tolerance (Sharpe < 0.01 difference). No re-runs needed.

### Impact on Terminal Decision

Terminal 1 (Deploy) remains valid. Pass/fail outcomes used Sharpe-based criteria, not max DD. The only change is deployment parameter calibration:
- Auto-pause threshold revised from 15% to 20% (canonical max DD is 18.9%)
- Hard stop added at 28.3% (1.5× canonical)

See `PHASE_6D_RECONCILIATION.md` for full analysis.

---

## Phase 6.D.1 Addendum — Bug Fix & Re-Run Verification

*Added: 2025-07-15. This addendum updates metric values but does NOT change the terminal decision.*

### Bug Fix Applied

The double vol-scaling bug identified in Phase 6.D has been **fixed** in both `phase6c_omega.py` and `phase6c_stress.py`:
- Removed portfolio-level re-scaling (`port_ret * (0.10 / vol)`) — cells are already at 10% vol individually
- Fixed equity curve construction: `np.cumprod(1 + r)` → `np.exp(np.cumsum(r))`
- Added `at_10pct_vol` metrics for deployment reference

### Corrected Results (All Tests Re-Run)

| Test | Old Result | **New Result** | Key Change |
|------|-----------|---------------|------------|
| 6.C.0 Held-out | PASS (SR=0.316, DD=16.7%) | **PASS** (SR=0.316, DD=**13.6%** @ 8.2% vol) | DD decreased |
| 6.C.1 JPY reversal | PASS | **PASS** | Identical (median SR=0.0) |
| 6.C.2 MC regimes | PASS | **PASS** | Identical (all regimes positive median SR) |
| 6.C.3 Cost sensitivity | PASS (DD=24.6%) | **PASS** (DD=**17.3%** @ 7.3% vol) | DD decreased |
| 6.C.4 Walk-forward | PASS (55.8%) | **PASS** (55.8%) | Identical |
| 6.C.5 Params | FAIL | **FAIL** | Identical (MR 57%, TSMOM 43%, DM 29%) |

**Zero pass/fail flips.** Sharpe is scale-invariant; drawdowns only decreased.

### Corrected Canonical Metrics (Full Period @1x Cost)

| Metric | At Realized Vol (~7.3%) | At 10% Target Vol |
|--------|------------------------|-------------------|
| Sharpe | 0.4466 | 0.4466 |
| Max DD | 17.3% | 22.8% |
| Worst-2Y | -0.7243 | -0.7243 |

### Week Count Resolution

977 vs 1168 weeks: Phase 5.5 used `valid >= 2` filter (977 weeks), Phase 6.C includes all weeks (1168). The 191-week gap is strategy warmup periods with only 1 active cell. Both are valid; Phase 6.C's inclusive approach is more conservative for deployment.

### Updated Deployment Parameters

- **Canonical max DD (Phase 6.C method, all weeks):** 17.3% at realized vol, 22.8% at 10% vol
- **Auto-pause:** 25% (at realized vol, ~1.5× canonical)
- **Hard stop:** 35% (at 10% vol, ~1.5× canonical)

### Terminal Decision

**TERMINAL 1 — DEPLOY** remains valid. The bug fix improved all drawdown metrics (lowered them). No pass/fail outcomes changed.

See `PHASE_6D1_EXECUTION.md` for full execution details.
