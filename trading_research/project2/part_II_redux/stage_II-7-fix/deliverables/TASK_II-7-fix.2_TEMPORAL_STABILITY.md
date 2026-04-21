# TASK II-7-fix.2: Causal Temporal Stability Test

**Generated:** 2026-04-21 09:41 UTC

## Parameters

| Parameter | Value |
|-----------|-------|
| Method | PCMCI+ |
| Independence test | ParCorr |
| tau_max | 10 |
| pc_alpha | 0.01 |
| alpha_level | 0.05 |
| max_samples | 5000 |
| target_variable | target_fwd_6 |
| n_features | 12 |
| min_bars_threshold | 2000 |

## 1. Sub-Period Results Table

| Asset | Sub-Period | Samples | macd_hist τ=1 | MCI | p-value | Class |
|-------|-----------|---------|--------------|-----|---------|-------|
| BTC | 2017-H2 | 3117 | ✗ NO | — | — | γ |
| BTC | 2018-H1 | 4138 | ✗ NO | — | — | β |
| BTC | 2018-H2 | 4238 | ✗ NO | — | — | γ |
| BTC | 2019-H1 | 4167 | ✓ YES | 0.2777 | 0.0000 | α |
| BTC | 2019-H2 | 4243 | ✗ NO | — | — | γ |
| ETH | 2017-H2 | 3117 | ✓ YES | 0.1177 | 0.0000 | α |
| ETH | 2018-H1 | 4138 | ✗ NO | — | — | β |
| ETH | 2018-H2 | 4238 | ✗ NO | — | — | γ |
| ETH | 2019-H1 | 4167 | ✓ YES | 0.2616 | 0.0000 | α |
| ETH | 2019-H2 | 4243 | ✗ NO | — | — | γ |

## 2. macd_hist τ=1 Stability Classification

### BTC

- Sub-periods completed: **5**  (skipped: 0)
- macd_hist τ=1 present in: **1/5** sub-periods
- MCI sign consistent: **True**
- **CLASSIFICATION: REGIME_SPECIFIC**

> macd_hist τ=1 appeared only in 1 sub-period (likely 2019). Original II-7.2 finding weakened.

### ETH

- Sub-periods completed: **5**  (skipped: 0)
- macd_hist τ=1 present in: **2/5** sub-periods
- MCI sign consistent: **True**
- **CLASSIFICATION: PARTIALLY_STABLE**

> macd_hist τ=1 is present in some but not most sub-periods. Evidence is moderate.

## 3. Other Stable Lagged Links (≥3 sub-periods)

| Feature_Tau | N sub-periods present |
|-------------|----------------------|
| macd_hist_tau1 | 3 |

## 4. Conclusion

- **BTC macd_hist τ=1:** REGIME_SPECIFIC
- **ETH macd_hist τ=1:** PARTIALLY_STABLE

### Data Availability Note

Local 1h parquet files (`btcusdt_1h_2019_2025.parquet`, `ethusdt_1h_2019_2025.parquet`) start from 2019-01-01. Pre-2019 sub-periods (2017-H2, 2018-H1, 2018-H2) required fetching from Binance public API. See `skip_reason` in JSON for any sub-periods that could not be loaded.
