# Stage II-0.6: Asset Selection

**Date:** 2026-04-19  
**Status:** CONFIRMED  

## Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Primary** | BTC/USD 4h | Only α classification; RSI lag-6 causal link (MCI=-0.2459, p<0.001) |
| **Secondary** | None | All FX/equity runs are γ; no benefit from parallel comparison |
| **Path B** | ENABLED | α signal strong enough to justify causal-informed strategies |
| **Feature set** | Technical only | Macro features showed zero causal contribution across all 14 runs |

## Data Available for Primary

- **BTC/USD 4h:** 18,332 bars (2017-08-17 to present)
- **IS period:** 2017-08-17 to 2019-12-31 (~5,200 4h bars)
- **HO period:** 2020-01-01 to 2025+ (~13,100 4h bars)
- **Note:** BTC IS period is shorter than FX (2.4yr vs 15yr) due to asset history. Still provides >5,000 samples for robust analysis.

## Implications for Remaining Stages

- **II-1:** Validate rolling orchestrator on BTC/USD 4h real data
- **II-2:** BTC momentum baseline (no eurusd_mr equivalent) + regime_adaptive (recalibrate GMM for BTC)
- **II-3:** Path A with btc_momentum as strategy_1 (replaces eurusd_mr)
- **II-4:** RPCMCI refinement on BTC/USD 4h
- **II-5:** Path B enabled — Ridge, LightGBM, TFT, CNN, LSTM, TCN with RSI causal feature
