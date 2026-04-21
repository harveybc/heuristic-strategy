# Stage II-2: Static Baselines Report

**Date**: 2025-06-17  
**Asset**: BTC/USD 4h  
**IS Period**: 2017-08-17 to 2019-12-31  
**Trigger**: Yearly (2 windows)  

## Baseline Results

| Baseline | Mean Test Sharpe | Total Trades | Max DD | Notes |
|----------|:----------------:|:------------:|:------:|-------|
| btc_momentum (fixed defaults) | **0.135** | 112 | 0.126 | EMA(12/26) + RSI(14) + ATR(14) |
| regime_adaptive (fixed defaults) | 0.000 | 1 | 0.000 | EURUSD-calibrated GMM → near-zero activity on BTC |
| Buy-and-Hold | -1.487 | — | — | BTC crashed in both test windows (H2 2018 + H2 2019) |
| Random (100 MC trials) | 0.030 | — | — | Coin-flip baseline |
| Zero (no trades) | 0.000 | — | — | Sanity check |

## Key Observations

1. **btc_momentum outperforms all simple baselines** even with default parameters — validates the EMA crossover + RSI filter approach for BTC.
2. **regime_adaptive is unsuitable for BTC without GMM recalibration** — the EURUSD centroids produce almost no tradeable regime signals on BTC data. This is expected and confirms the need for A4/A7 to recalibrate with `--gmm_refit`.
3. **Buy-and-Hold Sharpe is deeply negative** (-1.49) because both test windows (Jul-Dec 2018, Jul-Dec 2019) coincide with BTC drawdowns. Any positive Sharpe from an active strategy already beats BH.
4. **Random baseline is near zero** (0.03) as expected, providing a valid null hypothesis.

## Fixed Default Parameters

### btc_momentum
| Parameter | Value |
|-----------|-------|
| ema_fast | 12 |
| ema_slow | 26 |
| rsi_period | 14 |
| rsi_overbought | 70 |
| rsi_oversold | 30 |
| atr_period | 14 |
| atr_tp_multiplier | 2.5 |
| atr_sl_multiplier | 1.5 |

### regime_adaptive
| Parameter | Value |
|-----------|-------|
| atr_period | 14 |
| atr_tp_multiplier | 2.5 |
| atr_sl_multiplier | 1.5 |
| cluster_confidence | 1.5 |

## Stage Gate: PASS

- ✅ 5 baselines computed
- ✅ btc_momentum fixed > buy-and-hold
- ✅ btc_momentum fixed > random
- ✅ Results archived in `logs/baseline_*` and `deliverables/baselines_yearly.json`

## Files

- `logs/baseline_btc_momentum_fixed/` — full orchestrator output
- `logs/baseline_regime_adaptive_fixed/` — full orchestrator output
- `deliverables/baselines_yearly.json` — BH/Random/Zero results
