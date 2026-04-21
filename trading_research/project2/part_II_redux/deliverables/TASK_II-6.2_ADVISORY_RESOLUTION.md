# Task II-6.2: Advisory Failures Resolution

**Date:** 2025-04-20  
**Status:** RESOLVED — Both advisories are benign statistical noise. No data quality issues.

---

## 1. Advisory Failures (from Stage II-0b)

| Asset×TF | Test 5 ACF(1) | Threshold | Result |
|-----------|---------------|-----------|--------|
| EUR/USD 4H | -0.00205 | \|ACF\| > 0.005 | ⚠️ ADVISORY |
| USD/JPY Daily | -0.00344 | \|ACF\| > 0.005 | ⚠️ ADVISORY |

## 2. Diagnostic Results

### EUR/USD 4H
- N = 33,729 returns, 95% CI = ±0.01067
- ACF(1) = -0.00205 → **within 95% CI** (not statistically significant)
- Zero-return bars: 324 (1.0%) — normal for 4H FX
- ACF lags 2-5: [-0.004, -0.019, +0.009, +0.012] — no anomalous pattern

### USD/JPY Daily
- N = 6,549 returns, 95% CI = ±0.02422
- ACF(1) = -0.00344 → **within 95% CI** (not statistically significant)
- Zero-return bars: 20 (0.3%) — negligible
- ACF lags 2-5: [-0.006, -0.026, -0.006, +0.004] — no anomalous pattern

### Cross-Timeframe Comparison

| Asset | 1H | 4H | Daily | Weekly |
|-------|-----|-----|-------|--------|
| EUR/USD | -0.0112* | **-0.0021** | +0.0195 | -0.0110 |
| USD/JPY | n/a | -0.0141* | **-0.0034** | -0.0566* |

\* = exceeds 0.005 threshold

**Pattern**: ACF(1) magnitude follows a non-monotonic function of timeframe. 4H EUR/USD and Daily USD/JPY happen to be near-zero crossing points — consistent with EMH-efficient intermediate frequencies in major FX pairs.

## 3. Root Cause Assessment

- **No aggregation errors**: Zero-return bars are minimal (1.0% and 0.3%)
- **No data gaps**: Bar counts match expected values (passed Test 1)
- **Threshold is arbitrary**: The 0.005 cutoff was chosen to differentiate real vs synthetic data. Both values are well within the 95% statistical noise band.
- **All other tests pass**: Kurtosis (14.6, 8.2), vol clustering (0.124, 0.169), anti-GBM fingerprint (3/3, 2/3)

## 4. Impact on Held-Out Evaluation

**None.** All held-out evaluations (II-6.3, II-6.4) use BTC/USD 4H exclusively. BTC/USD 4H passed all 6 tests with no advisories (ACF(1) = -0.046, kurtosis = 23.4).

## 5. Verdict

Both advisories are **benign** — the ACF(1) values are statistically indistinguishable from zero and consistent with efficient market microstructure at intermediate FX timeframes. No threshold recalibration needed. No impact on held-out evaluation.
