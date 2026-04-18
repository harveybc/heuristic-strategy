# Work Plan Execution Report

**Date**: 2025-04-17
**Execution**: Distributed across omega (local), dragon (192.168.1.235), gamma (192.168.0.106)

---

## Phase 0: Diagnostic Infrastructure ✅

All modules built, tested, and validated:
- **Oracle sensitivity test** (`oracle_sensitivity.py`): σ-calibrated noise, 11-point grid
- **Transaction cost model** (`transaction_cost_model.py`): 16 assets, spread+slippage+vol scaling
- **Evaluation harness** (`evaluation_harness.py`): rolling 2Y windows, regime robustness metric

---

## Phase 1: 360-Cell Oracle Sweep → 14 Survivors ✅

### 1.2 Distributed Oracle Sweep
- **Grid**: 12 assets × 4 timeframes × 5 strategies = 240 cells (15min excluded, yfinance 60-day limit)
- **Dragon** (RTX 4090): BTC/USD, ETH/USD, CL, XAU/USD, XAG/USD → 100/100 cells
- **Gamma** (RTX 5070 Ti): EUR/USD, USD/JPY, GBP/USD, AUD/USD → 80/80 cells
- **Omega** (RTX 4070): AUD/JPY, EUR/JPY, GBP/JPY → 60/60 cells
- **All 240 cells evaluated, 0 failures**

### Key Statistics:
| Metric | Count | % |
|--------|-------|---|
| Budget > 0σ | 65 | 27.1% |
| Budget ≥ 0.25σ | 62 | 25.8% |
| Budget ≥ 0.50σ | 55 | 22.9% |
| Budget ≥ 1.00σ | 46 | 19.2% |
| Budget ≥ 1.50σ | 41 | 17.1% |

### 1.3 Naive Baseline Test: 14 of 30 Survive
Kill criterion (< 5 survivors) **NOT triggered**.

| # | Asset | TF | Strategy | Oracle SR | Budget | B&H SR | p-value |
|---|-------|----|----------|-----------|--------|--------|---------|
| 1 | BTC/USD | weekly | momentum | +1.904 | 10.00σ | +0.703 | 0.000 |
| 2 | BTC/USD | weekly | carry_momentum | +1.904 | 10.00σ | +0.703 | 0.000 |
| 3 | XAU/USD | weekly | momentum | +2.009 | 10.00σ | +0.636 | 0.000 |
| 4 | XAU/USD | weekly | carry_momentum | +2.009 | 10.00σ | +0.636 | 0.000 |
| 5 | XAG/USD | weekly | momentum | +0.827 | 10.00σ | +0.350 | 0.000 |
| 6 | XAG/USD | weekly | carry_momentum | +0.827 | 10.00σ | +0.350 | 0.000 |
| 7 | EUR/USD | daily | mean_reversion | +0.469 | 10.00σ | -0.068 | 0.000 |
| 8 | USD/JPY | daily | mean_reversion | +0.368 | 10.00σ | +0.200 | 0.020 |
| 9 | GBP/USD | 4h | vol_regime_switch | +1.760 | 10.00σ | +0.379 | ~0 |
| 10 | AUD/USD | weekly | vol_regime_switch | +0.492 | 10.00σ | -0.050 | 0.040 |
| 11 | EUR/JPY | weekly | vol_regime_switch | +0.535 | 10.00σ | +0.130 | 0.020 |
| 12 | XAU/USD | daily | momentum | +1.396 | 3.00σ | +0.622 | 0.000 |
| 13 | XAU/USD | daily | carry_momentum | +1.396 | 3.00σ | +0.622 | 0.000 |
| 14 | XAU/USD | daily | vol_regime_switch | +1.110 | 3.00σ | +0.622 | 0.000 |

### Tier Structure:
- **Tier 1 (10σ budget)**: BTC weekly, XAU weekly, EUR/USD daily MR, USD/JPY daily MR, GBP/USD 4h, AUD/USD weekly, EUR/JPY weekly — extreme noise tolerance
- **Tier 2 (3-5σ budget)**: XAU daily momentum/carry/vol — strong but more fragile

### Asset Champions:
- **XAU/USD**: 5 surviving cells (weekly + daily), strongest overall
- **BTC/USD**: 2 cells (weekly momentum/carry), very high SR
- **XAG/USD**: 2 cells (weekly momentum/carry)
- **FX pairs**: EUR/USD MR, USD/JPY MR, GBP/USD vol_regime, AUD/USD, EUR/JPY

---

## Phase 2: EUR/USD MR Deep Audit → KILLED ✗

### Findings:
| Criterion | Result | Threshold | Verdict |
|-----------|--------|-----------|---------|
| Rolling robustness | 22.2% | ≥ 40% | **FAIL** |
| Parameter plateau | FRAGILE (narrow spike) | Smooth | **FAIL** |
| Survives 3 bps | NO (dies at ~1.5 bps) | Yes | **FAIL** |
| Structural explanation | Asian session only | Any | Partial |

### Detail:
- Best params: lookback=20, z_entry=2.0, z_exit=0.0
- Best net Sharpe: -0.016 (after costs!)
- Zero-cost Sharpe: +0.275 → costs eat 100%+ of alpha
- Only profitable during Asian session (hours 0-8 UTC, Sharpe +0.80)
- London and NY sessions destroy the signal
- Monday effect strong (+2.84 SR) but rest of week negative
- Autocorrelation ≈ 0 at all lags → no exploitable structure

**Verdict: EUR/USD hourly mean reversion is NOT tradeable. Kill and do not revisit.**

> Note: EUR/USD daily MR survived the oracle sweep (10σ budget) — the DAILY timeframe may still be viable. The hourly implementation specifically fails.

---

## Phase 3: Exogenous Data Enrichment ✅

### Downloaded:
| Source | Series | Status |
|--------|--------|--------|
| FRED | DGS10, DGS2, T10Y2Y, DFF | ✅ 5253-7670 obs |
| Yahoo | VIX, DXY | ✅ ~5280 obs |
| Yahoo | BTC/ETH volume + vol + returns | ✅ 2921 obs |
| CFTC | COT reports | ✗ 403 Forbidden |

### Feature Store:
- 11 assets × daily timeframe = 11 CSV files
- 19 features per FX asset, 25 per crypto asset
- Derived features: yield_spread, VIX_pctile_60d, DXY_ret_20d, log_return, vol_20, z_score_20
- Location: `trading_research/feature_store/`

---

## Next Steps (Phase 4+)

### Priority targets for deeper investigation:
1. **XAU/USD** — 5 surviving cells, strongest asset, test with real parameter optimization
2. **BTC/USD weekly** — highest oracle Sharpe (+1.90), momentum and carry both survive
3. **GBP/USD 4h vol_regime_switch** — high oracle Sharpe (+1.76), unique timeframe/strategy combo
4. **EUR/USD daily MR** — survived oracle sweep but hourly version is killed; daily-only audit needed
5. **USD/JPY daily MR** — similar to EUR/USD, needs own audit

### Key insight:
- **Weekly timeframe dominates** — 8 of 14 survivors are weekly
- **Momentum and carry_momentum are twins** — they share oracle signals in most cells
- **Vol regime switching is the dark horse** — survives in 4 different assets
- **Mean reversion only works for FX on daily** — not hourly
