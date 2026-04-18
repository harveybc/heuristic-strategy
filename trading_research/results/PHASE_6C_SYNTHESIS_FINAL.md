# ⚠️ DEPRECATED — Script-Canonical (Historical Only)

> **Superseded by:** `PHASE_6C_SYNTHESIS_FINAL_v2.md` (plugin-canonical, 2025-07-14)
> **Reason:** Plugin implementations are the production code path. Script-canonical metrics
> are retained for reference only. Do not use for deployment planning.

---

# Phase 6.C Synthesis — Final Robustness & Terminal Decision

**Date:** 2025-07-24 (original) | 2025-07-15 (Phase 6.D.1 corrections applied)
**Consolidated:** 2026-04-17 (Phase 6.D.2 — this is the canonical final document)
**Status:** DEPRECATED — See PHASE_6C_SYNTHESIS_FINAL_v2.md
**Terminal Decision:** **TERMINAL 1 — DEPLOY P3 to OANDA Demo (90-day observation)**

> This document reflects the state after Phase 6.D reconciliation and Phase 6.D.1 bug
> correction. All metrics are corrected and internally consistent. See
> `PHASE_6D_RECONCILIATION.md` and `PHASE_6D1_EXECUTION.md` for the full correction history.

---

## Vol Reporting Convention

All deployment-relevant metrics (max drawdown, absolute returns) are reported at **10%
annualized target vol** unless explicitly noted otherwise. This matches the operational
deployment target. Scale-invariant metrics (Sharpe ratio, worst-2Y Sharpe, correlations)
are identical at any vol level. Where realized portfolio vol differs from 10%, a leverage
factor is applied: `leverage = 10% / realized_vol`.

The portfolio's realized vol is approximately **7.3%** (full period) due to cell-weight
diversification. At 10% target vol, implied leverage is **1.36×**.

---

## Executive Summary

Portfolio P3 (EUR/USD MR 20.6%, USD/JPY TSMOM 49.2%, USD/JPY DM 30.2%) was subjected
to 5 rigorous robustness stress tests plus a genuine held-out evaluation on 2024–2025 data
never seen during any research phase. P3 **passed 4 out of 5 stress tests** and **cleared
the held-out gate** with Sharpe 0.316 and maximum drawdown 16.2% at 10% target vol. This
meets the pre-committed Terminal 1 threshold: deploy to OANDA demo for 90-day live
observation.

**Canonical full-period metrics (10% target vol, 1168 weeks, all weeks included):**

| Metric | Value |
|--------|-------|
| Sharpe | 0.447 |
| Max Drawdown | **22.8%** |
| Worst-2Y Sharpe | −0.724 |
| Total Return | 172.7% |
| Realized Vol | 7.3% (leverage 1.36× to reach 10%) |

---

## 1. Held-Out Evaluation (6.C.0) — GATE: PASS

The 2024-01-01 to 2025-12-31 period was preserved untouched through all research phases.
This is the single genuine out-of-sample test.

| Metric | In-Sample (≤2018) | OOS (2019–2023) | **Held-Out (2024–2025)** |
|--------|-------------------|-----------------|--------------------------|
| Sharpe | 0.452 | 0.505 | **0.316** |
| Max DD | 22.4% | 16.5% | **16.2%** |
| Return | 98.3% | 28.8% | **6.6%** |

*All max DD and return values at 10% target vol.*

**Per-cell held-out (at cell-level 10% vol):**

| Cell | Sharpe | Return | Max DD | Trades |
|------|--------|--------|--------|--------|
| EUR/USD MR | 0.178 | 2.4% | 5.4% | 77 |
| USD/JPY TSMOM | 0.239 | 4.9% | 17.7% | 25 |
| USD/JPY DM | 0.321 | 8.4% | 19.5% | 4 |

**Gate criteria:**
- Sharpe > 0: **PASS** (0.316)
- Max DD < 25%: **PASS** (16.2% at 10% vol)
- Max DD < 30% (gate): **PASS** (16.2%)

**Observations:**
- Held-out Sharpe (0.316) shows ~30% decay from IS/OOS (0.45–0.50), consistent with
  typical strategy degradation. Not alarming — still well above zero.
- Held-out max DD (16.2%) is well below the IS value (22.4%), suggesting the strategy
  has not encountered an extreme stress period in 2024–2025.
- All 3 cells contribute positively. USD/JPY concentration (79.4%) remains a risk factor.

---

## 2. Stress Test Results

### 2.1 — 6.C.1 JPY Reversal Stress (Gamma) — PASS

500 synthetic 2-year paths via block bootstrap with 30% JPY reversal overweight
(amplification up to 2.0x).

| Percentile | Sharpe | Max DD |
|-----------|--------|--------|
| 5th | −0.652 | 0.7% |
| 25th | −0.187 | 3.1% |
| **50th** | **0.000** | **8.1%** |
| 75th | 0.252 | 11.1% |
| 95th | 0.940 | 15.3% |

- Fraction with Sharpe > 0: 49.4%
- **Kill criteria:** Median SR ≥ 0 **PASS** (0.000) | 25th pct ≥ −1.5 **PASS** (−0.187)
- **Interpretation:** P3 is approximately flat during JPY reversals, not catastrophic.
  The median Sharpe sitting at exactly 0 indicates the JPY momentum cells give back
  gains during reversals while the MR cell provides modest offset.

### 2.2 — 6.C.2 Monte Carlo Regime Scenarios (Dragon) — PASS

3000 synthetic 2-year paths per regime via block bootstrap.

| Regime | Period | Bars | Median SR | 75th DD | Frac SR>0 |
|--------|--------|------|-----------|---------|-----------|
| A: Inflation | 2022–2025 | 1039 | 0.074 | 12.0% | 53.3% |
| B: QE | 2013–2019 | 1822 | 0.022 | 11.7% | 51.7% |
| C: Crisis | 2008–2012 | 1284 | 0.024 | 9.1% | 51.3% |

- **Kill criteria (Regime B):** Median SR < 0 **PASS** (0.022) | 75th DD > 30% **PASS** (12.0%)
- **All regimes pass:** True
- **Interpretation:** P3 is approximately break-even across all regimes with well-controlled
  drawdowns. QE regime (the hardest for momentum strategies) shows a barely positive
  median Sharpe. No regime causes catastrophic failure.

### 2.3 — 6.C.3 Cost Sensitivity (Omega) — PASS

| Cost Multiplier | Sharpe | Worst-2Y | Max DD (10% vol) | Total Return (10% vol) |
|----------------|--------|----------|------------------|----------------------|
| 1.00× | 0.447 | −0.724 | 22.8% | 172.7% |
| 1.25× | 0.436 | −0.735 | 23.3% | 166.4% |
| 1.50× | 0.426 | −0.745 | 23.8% | 160.3% |
| 1.75× | 0.416 | −0.755 | 24.3% | 154.4% |
| 2.00× | 0.406 | −0.765 | 24.7% | 148.7% |
| 2.50× | 0.386 | −0.784 | 25.7% | 137.9% |
| 3.00× | 0.366 | −0.803 | 26.6% | 127.7% |

- Sharpe break-even: **>3.0×** (still positive at 3× costs)
- Worst-2Y break-even: **>3.0×** (still above −1.0 at 3×)
- **Kill criteria:** SR positive at 2.0× **PASS** (0.406) | worst-2Y > −1.0 at 1.5× **PASS** (−0.745)
- **Interpretation:** Extremely robust to cost assumptions. P3 could tolerate a tripling
  of transaction costs and still maintain positive Sharpe. The gentle slope (−0.027 SR
  per 1× cost increase) reflects P3's low turnover, particularly the DM cell (4 trades
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
  → YES (SR=0.316, DD=16.2% at 10% vol)
  → Proceed to stress tests

Stress test scorecard: 4/5 PASS?
  6.C.1 JPY Reversal:     PASS
  6.C.2 MC Regimes:       PASS
  6.C.3 Cost Sensitivity:  PASS
  6.C.4 Walk-Forward:      PASS
  6.C.5 Param Perturbation: FAIL
  → Score: 4/5 = YES

Held-out maxDD < 25%?
  → YES (16.2%)

TERMINAL 1: DEPLOY to OANDA Demo
```

---

## 4. Terminal Decision

### **TERMINAL 1 — DEPLOY P3 TO OANDA DEMO**

P3 passes the pre-committed deployment criteria:
1. Held-out Sharpe 0.316 > 0 ✓
2. Held-out max DD 16.2% < 25% (at 10% vol) ✓
3. 4/5 stress tests passed ✓

### Deployment Parameters

| Parameter | Value |
|-----------|-------|
| Portfolio | P3: EUR/USD MR (20.6%), USD/JPY TSMOM (49.2%), USD/JPY DM (30.2%) |
| Target vol | 10% annualized |
| Rebalance frequency | Monthly |
| Cost model | OANDA spread + 0.3 bps slippage per trade |
| Observation period | 90 calendar days |
| Go/No-go threshold | Sharpe > −0.5 at day 90, max DD < 25% (at 10% vol) |

### Canonical Max Drawdown

**22.8%** at 10% annualized target vol (1168 weeks, all weeks included, exp(cumsum) equity).

This is the Phase 6.C methodology (Option B) which includes all calendar weeks — including
strategy warm-up periods when only 1 of 3 cells is active. This is more conservative than
the Phase 5.5 methodology (18.9% over 977 filtered weeks) because a live deployment will
include warm-up periods when the strategy first goes live.

### Risk Thresholds

| Level | Threshold | Calibration |
|-------|-----------|-------------|
| Warning | 17.1% | 75% of canonical (22.8%) |
| Auto-pause | 25.0% | ~1.1× canonical |
| Hard stop | 34.2% | 1.5× canonical |

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
- If max DD exceeds 25% at any point (at 10% vol): **AUTO-PAUSE** — halt new trades, flatten
- If max DD exceeds 34.2%: **HARD STOP** — close all positions immediately
- If 90-day Sharpe > 0 and max DD < 17%: Consider live deployment with minimum capital
- Otherwise: Extend observation by 90 days (maximum 2 extensions)

---

## 5. Master Results Table

| Test | Machine | Metric | Value | Threshold | Result |
|------|---------|--------|-------|-----------|--------|
| 6.C.0 Held-out | Omega | Sharpe | 0.316 | >0 | **PASS** |
| 6.C.0 Held-out | Omega | Max DD (10% vol) | 16.2% | <25% | **PASS** |
| 6.C.1 JPY Stress | Gamma | Med SR | 0.000 | ≥0 | **PASS** |
| 6.C.1 JPY Stress | Gamma | 25th SR | −0.187 | ≥−1.5 | **PASS** |
| 6.C.2 MC Regimes | Dragon | B med SR | 0.022 | ≥0 | **PASS** |
| 6.C.2 MC Regimes | Dragon | B 75th DD | 11.7% | <30% | **PASS** |
| 6.C.3 Cost | Omega | SR@2× | 0.406 | >0 | **PASS** |
| 6.C.3 Cost | Omega | w2Y@1.5× | −0.745 | >−1.0 | **PASS** |
| 6.C.4 Walk-Fwd | Omega | %+ve Q | 55.8% | ≥55% | **PASS** |
| 6.C.4 Walk-Fwd | Omega | Med Q SR | 0.265 | ≥0.2 | **PASS** |
| 6.C.4 Walk-Fwd | Omega | Max streak | 3 | ≤5 | **PASS** |
| 6.C.5 Params | Gamma | MR plateau | 57% | ≥60% | **FAIL** |
| 6.C.5 Params | Gamma | TS plateau | 43% | ≥60% | **FAIL** |
| 6.C.5 Params | Gamma | DM plateau | 29% | ≥60% | **FAIL** |

**Overall: 4/5 PASS → TERMINAL 1 (DEPLOY)**

---

## 6. Appendix

### Compute Infrastructure

| Machine | Role | GPU | Tasks | Runtime |
|---------|------|-----|-------|---------|
| Omega | Local | RTX 4070 12GB | 6.C.0, 6.C.3, 6.C.4 | ~5 min |
| Dragon | Remote | RTX 4090 16GB | 6.C.2 (9000 MC paths) | ~30 min |
| Gamma | Remote | RTX 5070 Ti 8GB | 6.C.1 (500 paths), 6.C.5 | ~10 min |

### Results Files

All results stored in `trading_research/results/`:
- `phase_6c_omega_results.json` — 6.C.0 + 6.C.3 (corrected, Phase 6.D.1)
- `phase_6c_stress_6c1.json` — JPY reversal (corrected)
- `phase_6c_stress_6c2.json` — MC regimes (corrected)
- `phase_6c_stress_6c4.json` — Walk-forward (corrected)
- `phase_6c_stress_6c5.json` — Parameter perturbation (corrected)
- `phase_6d1_results.json` — Phase 6.D.1 correction summary

### Correction History

This document was originally produced on 2025-07-24 with a double vol-scaling bug that
inflated max drawdown figures (e.g., held-out max DD was reported as 16.7% at artificially
forced 10% vol instead of the correct 16.2% at 10% target vol; cost sensitivity max DD was
24.6% instead of the correct 22.8%). The bug was identified in Phase 6.D (reconciliation)
and fixed in Phase 6.D.1 (execution). Phase 6.D.2 consolidated all corrections into this
document. See:
- `PHASE_6D_RECONCILIATION.md` — Discrepancy analysis identifying the bug
- `PHASE_6D1_EXECUTION.md` — Bug fix execution and re-run verification
- `PHASE_6C_SYNTHESIS_v1_DEPRECATED.md` — Original synthesis with pre-fix values (historical)

### Canonical Metric Methodology

- **Vol scaling:** Cell returns are individually scaled to 10% annualized vol (single level).
  No portfolio-level vol re-scaling.
- **Equity curve:** `np.exp(np.cumsum(log_returns))` — correct for log returns.
- **Weekly aggregation:** `.resample("W").sum()` (Sunday anchor), all weeks included
  (no `valid >= 2` filter). This yields 1168 weeks vs Phase 5.5's 977 filtered weeks.
- **Canonical max DD (22.8%):** Computed over all 1168 weeks at 10% target vol.
  The more conservative inclusive method was chosen because live deployment will include
  warm-up periods when not all cells are active.
