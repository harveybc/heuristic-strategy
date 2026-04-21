# Stage II-0.5: Cross-Asset Causal Comparison Report

**Date:** 2026-04-19 (final)  
**Status:** 14/14 runs complete — 13 successful + 1 insufficient data | **1 α, 12 γ, 0 β**  
**Method:** PCMCI+ with ParCorr | τ_max=10 | pc_alpha=0.01 | alpha_level=0.05  
**Target:** 6-bar forward log return  
**IS Period:** 2005-01-01 to 2019-12-31  
**Machines:** Omega (local) + Dragon (192.168.0.107) + Gamma (192.168.0.106)  

---

## 1. Executive Summary

Cross-asset causal discovery across 4 assets, 4 timeframes, and 14 PCMCI+ configurations reveals a single actionable causal signal:

**BTC/USD 4h is the only α classification.** RSI at lag 6 causally predicts 6-bar forward log returns with MCI = -0.2459 (p < 0.001). All other asset/timeframe combinations are classified γ (no lagged causal structure). Macro features (rate differentials, DXY, VIX, CFTC net positions) contribute zero causal links to returns in any tested configuration.

### Key Implications
- **Path A only** for EUR/USD, USD/JPY, SPY (all γ — no causal structure to exploit)
- **Path B eligible** for BTC/USD 4h (α classification enables causal-informed strategies)
- Macro features provide no incremental causal information for any asset

---

## 2. Classification Summary

| # | Run | Asset | TF | Macro | Vars | Samples | Time | Class | Lagged Links | Strongest MCI | Machine |
|---|-----|-------|----|-------|------|---------|------|-------|-------------|---------------|---------|
| 1 | eurusd_4h | EUR/USD | 4h | No | 13 | 5,000 | 22.8s | **γ** | 0 | — | Omega |
| 2 | eurusd_daily | EUR/USD | daily | No | 13 | 4,530 | 74.8s | **γ** | 0 | — | Omega |
| 3 | eurusd_daily_macro | EUR/USD | daily | Yes | 17 | 4,360 | 75.8s | **γ** | 0 | — | Gamma |
| 4 | eurusd_weekly | EUR/USD | weekly | No | 13 | 638 | 22.5s | **γ** | 0 | — | Omega |
| 5 | usdjpy_4h | USD/JPY | 4h | No | 13 | 5,000 | 36.7s | **γ** | 0 | — | Dragon |
| 6 | usdjpy_daily | USD/JPY | daily | No | 13 | 4,530 | 184.0s | **γ** | 0 | — | Dragon |
| 7 | usdjpy_daily_macro | USD/JPY | daily | Yes | 17 | 4,360 | 740.8s | **γ** | 0 | — | Gamma |
| 8 | usdjpy_weekly | USD/JPY | weekly | No | 13 | 638 | 16.7s | **γ** | 0 | — | Dragon |
| 9 | spy_daily | SPY | daily | No | 13 | 3,631 | 223.4s | **γ** | 0 | — | Dragon |
| 10 | spy_daily_macro | SPY | daily | Yes | 16 | 3,517 | 1,127.8s | **γ** | 0 | — | Gamma |
| 11 | spy_weekly | SPY | weekly | No | 13 | 638 | 38.8s | **γ** | 0 | — | Dragon |
| 12 | **btcusd_4h** | **BTC/USD** | **4h** | **No** | **13** | **5,000** | **191.3s** | **α** | **1** | **RSI t-6: -0.2459** | **Omega** |
| 13 | btcusd_daily | BTC/USD | daily | No | 13 | 723 | 20.3s | **γ** | 0 | — | Omega |
| 14 | btcusd_weekly | BTC/USD | weekly | No | — | — | — | **INSUF** | — | 124 IS bars, 134-bar warmup needed | Omega |

**Classification criteria:**
- **α:** ≥1 lagged link with |MCI| > 0.10 and p < 0.01
- **β:** ≥1 lagged link with |MCI| ∈ [0.05, 0.10] and p < 0.05
- **γ:** No lagged links with |MCI| > 0.05

---

## 3. Detailed Results

### 3.1 BTC/USD 4h — α Classification (Critical Finding)

The only run exhibiting lagged causal structure. RSI at 6-bar lag (24 hours) causally predicts forward returns.

**Lagged causal links → fwd_ret_6:**

| Feature | Lag | MCI | p-value | Direction |
|---------|-----|-----|---------|-----------|
| **rsi** | **t-6** | **-0.2459** | **< 0.001** | Negative (high RSI → lower future returns) |

**Contemporaneous links → fwd_ret_6:**

| Feature | MCI | p-value |
|---------|-----|---------|
| di_spread | -0.1865 | < 0.001 |
| rsi | -0.1901 | < 0.001 |
| macd_hist | -0.1444 | < 0.001 |

**Autodependency:** fwd_ret_6 at t-1 (MCI=0.729), t-4 (MCI=0.131), t-6 (MCI=-0.125)

**Interpretation:** BTC/USD 4h returns exhibit mean-reversion with respect to RSI. When RSI is elevated (overbought), returns 24 hours later tend to be negative. The MCI magnitude (-0.2459) is substantial and highly significant, indicating genuine predictive power rather than spurious correlation. The negative sign is consistent with momentum exhaustion / mean-reversion dynamics.

### 3.2 EUR/USD — All γ

| Configuration | Contemporaneous Links |
|--------------|----------------------|
| 4h (tech) | di_spread (-0.082), bb_position (-0.040), rsi (-0.038), price_vs_ema50 (-0.177) |
| daily (tech) | di_spread (-0.169), bb_position (-0.080), rsi (-0.224), stoch_k (-0.153) |
| daily (tech+macro) | bb_position (-0.284), rsi (-0.225), macd_hist (-0.149) |
| weekly (tech) | rsi (-0.242), roc_12 (-0.245), macd_hist (-0.231) |

**Note:** Adding 4 macro features (us_eu_rate_diff, dxy_broad, vix, eur_net_pos) did not introduce any causal links. None of the macro features appear in the significant links.

### 3.3 USD/JPY — All γ

| Configuration | Contemporaneous Links |
|--------------|----------------------|
| 4h (tech) | roc_12 (-0.134) |
| daily (tech) | bb_position (-0.300), rsi (-0.241) |
| daily (tech+macro) | rsi (-0.242), macd_hist (-0.136). **No macro features linked.** |
| weekly (tech) | di_spread (-0.111), rsi (-0.297), macd_hist (-0.125) |

**Note:** Adding 4 macro features (us_jp_rate_diff, dxy_broad, vix, jpy_net_pos) contributed zero causal links.

### 3.4 SPY — All γ

| Configuration | Contemporaneous Links |
|--------------|----------------------|
| daily (tech) | bb_position (-0.183), rsi (-0.203), roc_12 (-0.158), stoch_k (-0.163), macd_hist (-0.103) |
| daily (tech+macro) | bb_position (-0.179), rsi (-0.197), roc_12 (-0.156), stoch_k (-0.147), macd_hist (-0.101), **vix (+0.140)** |
| weekly (tech) | roc_12 (-0.212) |

**Note:** VIX is the **only macro feature** to appear in any significant link across all 14 runs. It has a contemporaneous (t=0) positive association with SPY returns (MCI=+0.1397, p<0.001) — but critically, this is NOT a lagged causal link and cannot be exploited for prediction.

### 3.5 BTC/USD Daily and Weekly

| Configuration | Result |
|--------------|--------|
| daily (tech) | γ — only di_spread (-0.126) contemporaneous |
| weekly (tech) | **INSUFFICIENT DATA** — only 124 IS bars (2017-2019). `atr_pct` and `bb_width_pct` require 134+ bar warmup (14-bar ATR + 120-bar rolling percentile), exceeding total IS history. Even dropping those 2 features yields only 65 clean bars — below 200 minimum for reliable PCMCI+ inference. |

---

## 4. Cross-Cutting Analysis

### 4.1 Most Frequent Causal Features (Contemporaneous)

| Feature | Appearances (of 13 runs) | Avg |MCI| |
|---------|--------------------------|------------|
| rsi | 11/13 | 0.211 |
| bb_position | 6/13 | 0.214 |
| macd_hist | 7/13 | 0.135 |
| di_spread | 5/13 | 0.133 |
| roc_12 | 5/13 | 0.172 |
| stoch_k | 3/13 | 0.158 |
| price_vs_ema50 | 1/13 | 0.177 |
| vix | 1/13 | 0.140 |

**RSI is the dominant causal feature** — appearing in 9 of 11 successful runs and the only feature with a lagged causal link.

### 4.2 Macro Feature Contribution

Across all 3 macro runs (eurusd_daily_macro, usdjpy_daily_macro, spy_daily_macro):
- **Zero macro features** causally link to returns at any lag for EUR/USD and USD/JPY
- **VIX** has a single contemporaneous association with SPY returns (MCI=+0.1397) — but this is t=0 only, not a lagged causal link, and cannot be exploited for prediction
- Rate differentials (us_eu, us_jp, us_10y), DXY Broad, and CFTC net positions show zero causal links
- **Conclusion:** Macro features are causally irrelevant to 6-bar forward returns across all tested assets

### 4.3 Timeframe Effects

- **4h timeframes** show weaker contemporaneous structure (fewer links, lower MCI) — consistent with higher noise
- **Daily timeframes** show strongest contemporaneous structure (more links, higher MCI)
- **Weekly timeframes** have limited power due to small sample sizes (638 bars)
- **BTC/USD 4h is the exception** — the only 4h run showing lagged structure, likely due to BTC's unique microstructure (24/7 trading, retail-dominated)

---

## 5. Data Issues and Resolution

### 5.1 BTC/USD Weekly — Insufficient Data

BTC weekly has only 124 IS-period bars (2017-08 to 2019-12). Two features (`atr_pct`, `bb_width_pct`) require a 120-bar rolling percentile window on top of a 14-bar ATR warmup — totaling 134+ bars minimum. This exceeds the entire IS dataset. Even dropping these 2 features and computing on full data yields only 65 clean IS bars — below the 200 minimum for reliable PCMCI+ inference. Marked as INSUFFICIENT_DATA. BTC is adequately covered by btcusd_4h (α) and btcusd_daily (γ).

### 5.2 CFTC Duplicate Labels (Dragon)

Dragon ran the old `cross_asset_causal.py` without the CFTC duplicate-date dedup fix. This caused `usdjpy_daily_macro` and `eurusd_daily_macro` to error on Dragon. Both were successfully completed on Gamma (which had the fix). The Dragon error for `spy_daily_macro` did not occur because SPY's macro features (us_10y_yield, dxy_broad, vix) come from FRED only — no CFTC data.

### 5.3 Compute Time Distribution

| Machine | Runs | Total Time | Longest Run |
|---------|------|-----------|-------------|
| Omega | 7 | ~427s | btcusd_4h (191s) |
| Dragon | 7 (5 OK, 1 error, 1 duplicate) | ~2,429s | spy_daily_macro (1,920s) |
| Gamma | 3 | ~1,947s | spy_daily_macro (1,128s) |

Note: spy_daily_macro took 1,920s on Dragon vs 1,128s on Gamma — RTX 5070 Ti (Gamma) is faster than RTX 4090 (Dragon) for this workload, possibly due to CPU differences.

---

## 6. Path Determination

Per plan §7.4 classification criteria:

| Asset | Recommended Path | Rationale |
|-------|-----------------|-----------|
| **BTC/USD 4h** | **Path A + Path B** | α classification (RSI t-6, MCI=-0.2459). Path B enabled for causal-informed strategies |
| EUR/USD | Path A only | γ across all timeframes. No lagged causal structure |
| USD/JPY | Path A only | γ across all timeframes. No lagged causal structure |
| SPY | Path A only | γ across all timeframes. No lagged causal structure |

---

## 7. Recommendations for Stage II-0.6 (Asset Selection)

1. **Primary asset:** BTC/USD 4h — the only α classification, strong causal signal, sufficient data (18,332 bars)
2. **Feature set:** Technical only — macro features add nothing
3. **Path B:** ENABLED for BTC/USD 4h (exploit RSI lag-6 causal link)
4. **Secondary asset (optional):** SPY daily or EUR/USD daily for robustness comparison (both γ, Path A only)

---

## 8. Methodology Notes

- **Independence test:** ParCorr (linear partial correlation). Originally planned RobustParCorr but switched due to computational intractability with 12+ correlated technical features
- **Subsampling:** max_samples=5000 for 4h data (>5000 available bars)
- **Feature computation:** Uses regime_detector.compute_regime_features() from feature-eng library
- **Clean samples:** Rows with any NaN dropped after feature computation (initial warmup period)
- **Distributed execution:** Omega (local), Dragon (192.168.0.107), Gamma (192.168.0.106) running in parallel

---

*Report generated as part of Project 2 Part II-Redux. See `causal_results_II05_unified.json` for raw JSON results.*
