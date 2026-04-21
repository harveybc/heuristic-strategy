# Stage II-3 Path A: Adaptive Heuristic Experiments — Synthesis Report

**Project:** Project 2 Part II — Rolling Walk-Forward Trading Research  
**Date:** 2026-04-19  
**Status:** COMPLETE — PA-γ (All experiments FAIL)  

---

## 1. Executive Summary

Stage II-3 evaluated whether GA-optimized `regime_adaptive` and `regime_wfo` strategies, retrained at different frequencies, can produce statistically significant improvement over the fixed-parameter static baseline (Stage II-2 aggregate Sharpe = +0.068).

Five experiments (A1–A5) were executed across three GPU machines (Omega, Dragon, Gamma) totalling 87 walk-forward windows and ~20 hours of compute. **All five experiments fail multiple F-10 kill criteria.** No experiment exceeds the static baseline on any decisive metric. Two experiments (A2, A3) produced zero trades across all windows.

**Outcome Classification: PA-γ (Failure)**

### Key Findings

1. **GA bounds bug:** DEAP `cxBlend(alpha=0.5)` and `mutGaussian(sigma=1.0)` pushed parameters outside defined ranges (e.g. `atr_period=3.88` when range is 7–28). A `_repair_bounds()` clipping fix was applied mid-session; only A3 and A5 ran with the fix.
2. **A1 ≡ A4:** GMM refit with K=9 components had zero effect — identical results through all 11 windows (same seed + same data + regime labels don't alter the GA fitness landscape).
3. **A5 (regime_wfo + bounds fix)** was the best experiment: 102 trades (vs 55 for A1/A4), Mean SR near zero (−0.008 vs −0.588), but still fails K-3 and K-5.
4. **Systemic zero-trade problem:** A2 (47 windows) and A3 (7 windows) produced zero trades in every single test window — the strategies are too conservative for short test periods.
5. **No experiment beat the static baseline** (ΔSR < 0 for all five).

---

## 2. Experimental Design

### 2.1 Common Configuration
- **Data:** Synthetic GBM `eurusd_4h_2005_2024.csv` (31,297 bars)
- **GA Engine:** DEAP (seed=42 for reproducibility)
- **Embargo:** 6 bars between val and test
- **Fitness Function:** Validation Sharpe ratio
- **regime_adaptive params (4):** atr_period (7–28), atr_tp_multiplier (1.0–5.0), atr_sl_multiplier (0.5–3.0), cluster_confidence (0.5–3.0)
- **regime_wfo params (7):** Above 4 + bb_low (0.10–0.45), bb_high (0.55–0.90), atr_ratio_high (0.8–2.0), ema_align_thresh (−1.0–1.0)

### 2.2 Experiments

| Exp | Description | Plugin | Retraining | Windows | Pop | Gen | Machine | Bounds Fix |
|-----|-------------|--------|------------|---------|-----|-----|---------|------------|
| A1 | Yearly GA | regime_adaptive | Annual (expanding) | 11 | 30 | 20 | Omega | No |
| A2 | Quarterly GA | regime_adaptive | Quarterly (expanding) | 47 | 20 | 10 | Gamma | No |
| A3 | Change-point GA | regime_adaptive | Wasserstein trigger | 7 | 30 | 20 | Omega | **Yes** |
| A4 | Yearly GA + GMM refit | regime_adaptive | Annual + GMM(9) | 11 | 30 | 20 | Dragon | No |
| A5 | Yearly regime_wfo GA | regime_wfo | Annual (expanding) | 11 | 30 | 20 | Dragon | **Yes** |

**Not Executed:**
- A6 (NEAT-HPO): `eurusd_mr` is an LTS strategy, not a heuristic-strategy plugin.
- A7 (Weekly): Low priority, insufficient time.

### 2.3 Static Baseline Reference (Stage II-2)

| Metric | Value |
|--------|-------|
| Aggregate Sharpe | +0.068 |
| Mean ± Std | −0.073 ± 0.488 |
| Window Consistency | 54.5% (6/11) |
| Total Trades | 95 |
| Max Drawdown | 8.0% |

---

## 3. Per-Experiment Results

### 3.1 A1 — Yearly GA (`regime_adaptive`)

| Window | Test Year | Test Sharpe | Trades | Max DD | atr_period | tp_mult | sl_mult | conf |
|--------|-----------|-------------|--------|--------|------------|---------|---------|------|
| W1 | 2009 | +0.674 | 2 | 0.2% | 22.5 | 4.66 | 2.73 | 0.72 |
| W2 | 2010 | 0.000 | 0 | 0.0% | **3.9** | 3.87 | 3.20 | 0.47 |
| W3 | 2011 | +0.012 | 12 | 7.4% | 7.0 | 3.89 | **0.24** | 0.92 |
| W4 | 2012 | 0.000 | 1 | 3.3% | 18.6 | 4.14 | 1.56 | 0.63 |
| W5 | 2013 | **−6.363** | 2 | 13.8% | **37.6** | 3.09 | 3.44 | 0.52 |
| W6 | 2014 | +0.356 | 2 | 2.2% | 18.5 | 4.11 | 1.26 | 0.51 |
| W7 | 2015 | +0.026 | 2 | 1.7% | **2.2** | 3.80 | 1.28 | 0.48 |
| W8 | 2016 | 0.000 | 1 | 3.9% | 10.2 | 3.82 | 1.73 | 0.51 |
| W9 | 2017 | −0.317 | 2 | 5.9% | 22.4 | 4.23 | 3.05 | 0.89 |
| W10 | 2018 | +0.314 | 19 | 11.9% | 14.6 | 3.28 | **5.25** | 0.93 |
| W11 | 2019 | −1.167 | 12 | 12.2% | 18.0 | 2.97 | **0.36** | 0.66 |

**Bold** = out-of-bounds parameter values (no bounds fix applied).

- **Aggregate SR:** +0.063
- **Mean SR:** −0.588 ± 1.879
- **Consistency:** 45.5% (5/11)
- **Total Trades:** 55
- **Max Drawdown:** 18.9%

### 3.2 A2 — Quarterly GA (`regime_adaptive`)

47/47 windows complete. **Every window produced 0 trades and test Sharpe = 0.0.**

The quarterly expanding windows have short 1-month test periods (~160 4H bars). Combined with conservative GA-optimized parameters and the bounds bug, the strategy never triggers a trade during any test month.

- **Aggregate SR:** 0.000
- **Total Trades:** 0
- **Verdict:** Dead experiment.

### 3.3 A3 — Change-Point Triggered GA (`regime_adaptive`)

7/7 windows complete. **Every window produced 0 trades** despite the bounds fix being active.

| Window | Test Period | Test Sharpe | Trades | atr_period | conf |
|--------|------------|-------------|--------|------------|------|
| W1 | 2008-12 → 2009-06 | 0.000 | 0 | 21.8 | 0.50 |
| W2 | 2010-06 → 2010-12 | 0.000 | 0 | 9.8 | 0.50 |
| W3 | 2011-03 → 2011-09 | 0.000 | 0 | 9.7 | 0.50 |
| W4 | 2013-06 → 2013-12 | 0.000 | 0 | 17.5 | 0.50 |
| W5 | 2015-09 → 2016-03 | 0.000 | 0 | 21.5 | 0.50 |
| W6 | 2017-06 → 2017-12 | 0.000 | 0 | 16.1 | 0.56 |
| W7 | 2019-03 → 2019-09 | 0.000 | 0 | 22.4 | 0.92 |

Note: `cluster_confidence` converged to 0.50 (lower bound) in 6/7 windows — the GA found no confidence threshold that produces trades during short change-point test windows.

- **Aggregate SR:** 0.000
- **Total Trades:** 0
- **Verdict:** Dead experiment.

### 3.4 A4 — Yearly GA + GMM Refit (`regime_adaptive`)

**100% identical to A1** through all 11 windows. Same parameters, same trades, same Sharpe ratios.

This demonstrates that GMM refit (K=9 components) has zero effect on the GA optimization landscape. The regime labels assigned by GMM do not change the fitness function when the same DEAP seed and data are used.

- **Aggregate SR:** +0.063 (identical to A1)
- **All metrics identical to A1.**

### 3.5 A5 — Yearly GA (`regime_wfo`)

| Window | Test Year | Test Sharpe | Trades | Max DD | atr | tp | sl | bb_low | bb_high | atr_ratio | ema_align |
|--------|-----------|-------------|--------|--------|-----|----|----|--------|---------|-----------|-----------|
| W1 | 2009 | 0.000 | 1 | 5.1% | 19.3 | 4.91 | 2.94 | 0.17 | 0.61 | 1.24 | 0.91 |
| W2 | 2010 | 0.000 | 0 | 0.0% | 15.5 | 5.00 | 2.99 | 0.16 | 0.60 | 1.13 | −1.00 |
| W3 | 2011 | −0.137 | 24 | 22.2% | 9.2 | 1.47 | 3.00 | 0.28 | 0.64 | 1.22 | −1.00 |
| W4 | 2012 | **+0.312** | 4 | 4.2% | 9.2 | 1.56 | 2.91 | 0.30 | 0.64 | 1.36 | −1.00 |
| W5 | 2013 | −0.387 | 6 | 8.0% | 9.5 | 2.47 | 1.23 | 0.31 | 0.63 | 1.43 | 1.00 |
| W6 | 2014 | +0.002 | 5 | 13.3% | 14.3 | 4.14 | 3.00 | 0.44 | 0.89 | 1.41 | 0.98 |
| W7 | 2015 | −0.089 | 25 | 19.5% | 19.6 | 1.00 | 3.00 | 0.13 | 0.55 | 1.67 | −0.29 |
| W8 | 2016 | +0.116 | 11 | 8.4% | 9.2 | 2.00 | 2.91 | 0.31 | 0.63 | 1.49 | −0.31 |
| W9 | 2017 | −0.539 | 2 | 3.9% | 13.5 | 3.17 | 1.34 | 0.10 | 0.90 | 1.46 | 0.37 |
| W10 | 2018 | **+0.372** | 14 | 6.8% | 7.0 | 4.93 | 1.67 | 0.38 | 0.86 | 2.00 | −0.26 |
| W11 | 2019 | **+0.265** | 10 | 6.5% | 15.8 | 5.00 | 1.63 | 0.38 | 0.89 | 2.00 | −0.28 |

- **Aggregate SR:** +0.039
- **Mean SR:** −0.008 ± 0.267
- **Consistency:** 45.5% (5/11)
- **Total Trades:** 102 (best of all experiments)
- **Max Drawdown:** 29.0%

A5 generated nearly 2× the trades of A1/A4 and had a Mean SR near zero (best of all experiments). The bounds fix kept parameters in range. However, W3 and W7 had excessive drawdowns (>19%), and the strategy still fails K-3 and K-5.

---

## 4. F-10 Kill Criteria Evaluation

### 4.1 Kill Criteria Summary

| Criterion | Description | A1 | A2 | A3 | A4 | A5 |
|-----------|-------------|----|----|----|----|-----|
| K-1 | Held-out SR > 0 | Skipped | Skipped | Skipped | Skipped | Skipped |
| K-2 | Worst 2yr rolling SR > −0.9 | **FAIL** (−3.18) | PASS (0.0) | PASS (0.0) | **FAIL** (−3.18) | PASS (−0.21) |
| K-3 | Cost ratio ≥ 2.0 | **FAIL** (−10.0) | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (−10.0) | **FAIL** (−10.0) |
| K-5 | Window consistency ≥ 60% | **FAIL** (45.5%) | **FAIL** (0%) | **FAIL** (0%) | **FAIL** (45.5%) | **FAIL** (45.5%) |
| K-7 | ΔSR vs static > 0 (p<0.10) | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

**K-1 (held-out 2020–2024):** Not evaluated. Per §9 rules, held-out data is touched only for experiments passing all other criteria. No experiment qualifies.

### 4.2 Deflated Sharpe Ratio

DSR accounts for multiple testing across 5 experiments.

| Experiment | Observed SR | Expected Max SR | DSR | Significant (>0.95) |
|------------|-------------|-----------------|-----|---------------------|
| A1 | +0.063 | +1.50 | ≈0.000 | No |
| A2 | 0.000 | +1.50 | ≈0.000 | No |
| A3 | 0.000 | +1.50 | ≈0.000 | No |
| A4 | +0.063 | +1.50 | ≈0.000 | No |
| A5 | +0.039 | +1.50 | ≈0.000 | No |

All DSR values are essentially zero — the observed Sharpe ratios are far below the expected maximum SR under the null hypothesis for 5 independent trials.

### 4.3 K-7: Adaptive vs Static Baseline

| Experiment | Adaptive Agg SR | Static Agg SR | ΔSR | Pass |
|------------|-----------------|---------------|------|------|
| A1 | +0.063 | +0.068 | −0.005 | **FAIL** |
| A2 | 0.000 | +0.068 | −0.068 | **FAIL** |
| A3 | 0.000 | +0.068 | −0.068 | **FAIL** |
| A4 | +0.063 | +0.068 | −0.005 | **FAIL** |
| A5 | +0.039 | +0.068 | −0.029 | **FAIL** |

No experiment improved over the static baseline. The GA optimization either matched (A1/A4) or degraded (A5) aggregate performance, and completely failed for quarterly/change-point windows (A2/A3).

---

## 5. Parameter Stability Analysis

### 5.1 A1/A4 (`regime_adaptive`, 4 params)

| Parameter | Mean | Std | CV | Verdict |
|-----------|------|-----|-----|---------|
| atr_period | 14.5 | 8.8 | 0.604 | ⚠️ HIGH |
| atr_tp_multiplier | 3.85 | 0.49 | 0.128 | ✓ STABLE |
| atr_sl_multiplier | 2.10 | 1.37 | 0.652 | ⚠️ HIGH |
| cluster_confidence | 0.65 | 0.17 | 0.263 | ✓ STABLE |

**Avg CV: 0.41 — AMBIGUOUS.** `atr_period` and `atr_sl_multiplier` show high instability, partly due to bounds bug producing out-of-range values (3.9–37.6 for atr_period with range 7–28).

### 5.2 A5 (`regime_wfo`, 7 params)

| Parameter | Mean | Std | CV | Verdict |
|-----------|------|-----|-----|---------|
| atr_period | 12.9 | 4.2 | 0.321 | ✓ STABLE |
| atr_tp_multiplier | 3.24 | 1.53 | 0.472 | ◐ AMBIGUOUS |
| atr_sl_multiplier | 2.42 | 0.73 | 0.301 | ✓ STABLE |
| bb_low | 0.27 | 0.11 | 0.401 | ◐ AMBIGUOUS |
| bb_high | 0.70 | 0.13 | 0.184 | ✓ STABLE |
| atr_ratio_high | 1.49 | 0.28 | 0.186 | ✓ STABLE |
| ema_align_thresh | −0.11 | 1.07 | **9.472** | ⚠️ HIGH |

**Verdict: CURVE_FITTING_RISK.** `ema_align_thresh` is completely unstable (flips between −1.0 and +1.0), suggesting the GA cannot find a stable value. Core ATR parameters are more stable with the bounds fix.

### 5.3 A3 (`regime_adaptive`, change-point windows)

`cluster_confidence` converged to the lower bound (0.50) in 6/7 windows. The GA found that the only way to avoid catastrophic losses on short change-point windows was to set the confidence threshold at minimum — which produced zero trades.

---

## 6. Cross-Experiment Ranking

| Rank | Experiment | Agg SR | Mean SR | Consistency | Trades | ΔSR vs Static | DSR | Verdict |
|------|-----------|--------|---------|-------------|--------|---------------|-----|---------|
| 1 | A1 | +0.063 | −0.588 | 45.5% | 55 | −0.005 | ≈0 | FAIL |
| 2 | A4 | +0.063 | −0.588 | 45.5% | 55 | −0.005 | ≈0 | FAIL |
| 3 | A5 | +0.039 | −0.008 | 45.5% | 102 | −0.029 | ≈0 | FAIL |
| 4 | A2 | 0.000 | 0.000 | 0% | 0 | −0.068 | ≈0 | FAIL |
| 5 | A3 | 0.000 | 0.000 | 0% | 0 | −0.068 | ≈0 | FAIL |

---

## 7. Held-Out Evaluation (K-1)

**Not conducted.** No experiment passed the prerequisite kill criteria (K-2, K-3, K-5, K-7). Per §9 standing rules, held-out data (2020–2024) remains untouched.

---

## 8. Outcome Classification

Based on F-10 framework:

- **PA-α (Pass):** At least one experiment passes ALL kill criteria including K-1 on held-out data.
- **PA-β (Partial):** Some experiments show improvement but fail one or more criteria.
- **PA-γ (Failure):** All experiments fail multiple criteria.

**Classification: PA-γ (Failure)**

All five experiments fail K-3 (cost ratio), K-5 (window consistency), and K-7 (adaptive vs static). No experiment demonstrates statistically or economically significant improvement over the fixed-parameter baseline.

---

## 9. Key Observations

### 9.1 GA Bounds Bug — Critical Infrastructure Issue

DEAP's `cxBlend(alpha=0.5)` and `mutGaussian(sigma=1.0)` operators do not respect parameter bounds. This was discovered during A1 execution when `atr_period=3.88` (range 7–28) and `atr_period=37.6` appeared in results. A `_repair_bounds()` clipping function was added to `walk_forward_optimizer.py` after crossover and mutation. Only A3 and A5 benefited from the fix.

**Impact:** A1/A4 parameter instability (CV=0.60 for atr_period) is partially attributable to this bug. However, A3 (with fix) still produced 0 trades, and A5 (with fix) still failed all kill criteria, so the bug is not the sole cause of failure.

### 9.2 A1 ≡ A4: GMM Refit Has Zero Effect

A1 (no GMM refit) and A4 (GMM refit K=9) produced bit-identical results through all 11 windows. With the same DEAP seed (42), the GA evolution path is deterministic and the GMM regime labels do not alter the fitness landscape. The regime_adaptive plugin uses cluster labels from the data, but the GA optimizes ATR-based entry/exit parameters — the cluster labels are inputs, not outputs of the optimization.

### 9.3 Zero-Trade Systemic Failure (A2, A3)

47/47 quarterly windows and 7/7 change-point windows produced zero trades. Root cause: the GA-optimized parameters produce entry signals only on rare regime transitions, and short test windows (1–6 months) don't contain enough regime changes to trigger trades. The strategy's reliance on cluster confidence thresholds creates an implicit "do nothing" mode when confidence is near the lower bound.

### 9.4 A5 Best-in-Class But Still Failing

A5 (`regime_wfo`) with bounds fix was the most active strategy (102 trades), had the most stable Mean SR (−0.008), and showed improving performance in later windows (W10: +0.372, W11: +0.265). However:
- MaxDD = 29.0% (excessive)
- K-5 consistency = 45.5% (needs ≥60%)
- Aggregate SR (+0.039) < static baseline (+0.068)

### 9.5 Synthetic Data Limitations

The synthetic GBM data lacks microstructure, vol clustering, and regime persistence found in real FX data. Strategies relying on regime detection may inherently underperform on synthetic data where regimes are statistically indistinguishable.

---

## 10. Recommendations for Stage II-4

1. **Fix GA bounds in all experiments**: Re-run A1/A4 with `_repair_bounds()` to get a clean comparison. The bounds bug makes A1/A2/A4 results unreliable.

2. **Increase minimum trade constraint**: Current `min_trades=5` is insufficient. Consider `min_trades=20` to force the GA toward more active parameter regions.

3. **Revise fitness function**: Pure validation Sharpe over-penalizes drawdowns, pushing the GA toward zero-trade solutions. Consider a composite fitness: `0.7 * val_sharpe + 0.3 * log(num_trades + 1)`.

4. **Test on real data**: Synthetic GBM may not support regime-based strategies. Real EURUSD 4H data would provide meaningful regime transitions.

5. **A5 shows promise**: `regime_wfo` with bounds fix generated meaningful trade activity and improving late-window performance. A targeted follow-up with reduced MaxDD constraint could yield better results.

6. **Abandon A2/A3 configurations**: Quarterly and change-point retraining with 1–6 month test windows are structurally incompatible with low-frequency regime strategies.

---

## Appendix A: Execution Log

| Experiment | Machine | Duration | Windows | Status |
|------------|---------|----------|---------|--------|
| A1 | Omega (RTX 4070) | 5h 17m | 11/11 | Complete |
| A2 | Gamma (RTX 5070 Ti) | 5h 29m | 47/47 | Complete |
| A3 | Omega (RTX 4070) | 3h 28m | 7/7 | Complete |
| A4 | Dragon (RTX 4090) | 3h 08m | 11/11 | Complete |
| A5 | Dragon (RTX 4090) | 3h 04m | 11/11 | Complete |

**Total compute:** ~20h 26m across 3 machines  
**Evaluation output:** `deliverables/evaluation_results.json`
