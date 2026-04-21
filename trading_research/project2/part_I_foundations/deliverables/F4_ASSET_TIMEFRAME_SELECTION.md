# F-4: Asset and Timeframe Selection for Project 2

**Date**: 2025-06-17  
**Scope**: Select primary and secondary assets, timeframes, and data coverage windows for all Project 2 experimental paths  
**Depends on**: F-1 (P1 lessons), F-3 (data catalog), F-6 (causal analysis findings), F-8 (infrastructure audit)  
**Note**: F-2 (Jansen book integration) is deferred — selections here are based on empirical evidence from existing analyses and infrastructure constraints.

---

## 1. Selection Criteria

### 1.1 Hard Constraints (from F-1)

| ID | Constraint | Source |
|----|-----------|--------|
| C-1 | Predictor F1 ≥ 0.91 for predictor-primary strategies | P1 kill criteria |
| C-4 | Worst 2-year Sharpe > −0.9 | P1 evaluation metric |
| C-5 | Cost breakeven ≥ 2× spread | P1 viability threshold |
| HC-1 | Zero-cost data stack preferred (F-3: OANDA + FRED + HistData + yfinance + Binance) | Budget constraint |
| HC-2 | Must have ≥ 15 years of history for robust WFO | Needed for ≥ 5 non-overlapping 3-year windows |
| HC-3 | Must be tradeable via OANDA or Binance (live deployment infrastructure exists in lts) | F-8: broker plugins |

### 1.2 Soft Criteria

| ID | Criterion | Rationale |
|----|-----------|-----------|
| SC-1 | Low transaction costs (≤ 5 bps round-trip) | P1 showed cost sensitivity — strategies lost 50%+ return to costs |
| SC-2 | High liquidity / tight spreads | Related to SC-1 |
| SC-3 | Existing causal / regime analysis available | Reduces Part I work; allows immediate Path A experiments |
| SC-4 | Evidence of mean-reversion or momentum structure | Cross-asset audit findings (F-6) |
| SC-5 | Cross-asset diversification potential | Multi-asset portfolio reduces concentration risk |

---

## 2. Candidate Assessment

### 2.1 FX Pairs

| Pair | History | Cost (bps RT) | Liquidity | Causal Analysis | Structure | Score |
|------|---------|---------------|-----------|-----------------|-----------|-------|
| **EUR/USD** | 20+ yr (OANDA, HistData) | **2** | Highest | ✅ Full (PCMCI+, ICP, TE, GMM K=9, regime analysis) | No significant Hurst. Momentum unprofitable after costs. | **PRIMARY** |
| USD/JPY | 20+ yr (OANDA) | 2-3 | Very high | ❌ None | Unknown | Candidate |
| GBP/USD | 20+ yr (OANDA) | 3-4 | High | ❌ None | Unknown | Candidate |
| AUD/JPY | 15+ yr (OANDA) | **6** | Medium | Partial (cross-asset audit) | Unknown | Low priority (high cost) |
| USD/MXN | 10+ yr (OANDA) | **12** | Low | Partial (cross-asset audit) | Unknown | Exclude (high cost, short history) |

### 2.2 Crypto

| Pair | History | Cost (bps RT) | Liquidity | Analysis | Structure | Score |
|------|---------|---------------|-----------|----------|-----------|-------|
| BTC/USD | ~10 yr (Binance) | **30** | High | Cross-asset audit | Best momentum Sharpe (~0.4 net) | **SECONDARY** (if crypto in scope) |
| ETH/USD | ~8 yr (Binance) | 30 | High | Cross-asset audit | Similar to BTC | Low priority (correlated with BTC) |

### 2.3 Commodities / Equities

| Asset | History | Cost (bps RT) | Source | Analysis | Structure | Score |
|-------|---------|---------------|--------|----------|-----------|-------|
| XAU/USD (Gold) | 20+ yr (OANDA) | **5** | OANDA | Cross-asset audit | No significant structure | Low priority |
| SPY (S&P 500) | 30+ yr (yfinance) | **2** | yfinance | Cross-asset audit | **VR rejected random walk** (mean-reversion signal, VR=0.85, p=0.002) | **SECONDARY** |
| CL (Crude Oil) | 20+ yr (yfinance) | **8** | yfinance | Cross-asset audit | Unknown | Exclude (high cost, not OANDA-tradeable) |

---

## 3. Selected Assets

### 3.1 Primary Asset

**EUR/USD**

**Rationale**:
- Only asset with full causal/regime infrastructure (PCMCI+, GMM K=9, regime_adaptive plugin, regime_wfo plugin)
- Lowest cost (2 bps RT)
- 20+ years of hourly data available
- Live trading infrastructure exists (OANDA broker in lts)
- Continuous with Project 1 — allows direct comparison of static vs adaptive results
- Causal analysis found weak but non-zero signals (bb_position invariant, ema_alignment leading)

**Honest caveat**: EUR/USD showed no profitable momentum strategy after costs, no significant Hurst, and no lagged causal links at 4h. It may be the hardest asset to trade profitably. But it's the one with the most existing infrastructure, which means faster time-to-experiment.

### 3.2 Secondary Assets (for cross-asset validation in Part III+)

| Asset | When to Add | Purpose |
|-------|------------|---------|
| **USD/JPY** | Part III (after EUR/USD baseline) | FX cross-validation: different carry/rate dynamics, same cost structure. Tests whether EUR/USD findings generalize. |
| **SPY** | Part III | Non-FX validation: only asset with statistically significant mean-reversion signal. Different market microstructure (equity vs FX). |
| **BTC/USD** | Part IV or V (if crypto in scope) | Highest momentum signal but 15× cost of EUR/USD. Tests whether adaptive strategies can overcome high costs. |

### 3.3 Excluded Assets (with reasons)

| Asset | Reason |
|-------|--------|
| AUD/JPY | 6 bps cost, limited history, not enough differentiation from EUR/USD |
| USD/MXN | 12 bps cost, <15 yr history, EM liquidity risk |
| GBP/USD | No analysis done, too similar to EUR/USD for diversification value |
| ETH/USD | Highly correlated with BTC, redundant |
| XAU/USD | No significant structure found, 5 bps cost, limited OANDA execution quality |
| CL | 8 bps cost, not OANDA-tradeable via lts |

---

## 4. Timeframe Selection

### 4.1 Assessment

| Timeframe | Data Points (EUR/USD 20yr) | Bars/Year | P1 Experience | Causal Evidence | Rolling Window Feasibility |
|-----------|---------------------------|-----------|---------------|-----------------|---------------------------|
| **1-minute** | ~7M | 350K | Not used | None | Impractical for WFO (too many bars) |
| **5-minute** | ~1.4M | 70K | Not used | None | Marginal (compute-heavy) |
| **1-hour** | ~131K | 6.5K | **Used in P1** (base data) | None directly | ✅ Feasible |
| **4-hour** | ~33K | 1.6K | Used (via resampling) | ✅ **Full PCMCI+ analysis** | ✅ Feasible |
| **Daily** | ~5.2K | 260 | Not used | None | ✅ Feasible but small windows |
| **Weekly** | ~1K | 52 | Not used | None | ❌ Too few bars per window |

### 4.2 Selected Timeframes

| Timeframe | Role | Rationale |
|-----------|------|-----------|
| **4-hour** | **Primary experimental** | All causal analysis was done at 4h. GMM K=9 fitted at 4h. Regime plugins operate at 4h. Direct continuity with existing work. ~1,600 bars/year gives enough data per rolling window (e.g., 3-year train = ~4,800 bars). |
| **1-hour** | **Primary data collection** | Collect at 1h, resample to 4h/daily as needed. Preserves optionality. P1 used 1h as base. |
| **Daily** | **Secondary experimental** | F-6 recommended multi-timeframe causal analysis (CI-2). Daily may reveal lagged causal links that 4h misses. ~260 bars/year means rolling windows must be longer (5-year train). |

### 4.3 Excluded Timeframes

| Timeframe | Reason |
|-----------|--------|
| 1-minute, 5-minute | Too granular for macro features. Compute-prohibitive for rolling WFO. Microstructure noise dominates. No causal analysis. |
| Weekly | Too few bars (52/year). A 3-year rolling window gives only ~156 bars — insufficient for ML training. |

---

## 5. Data Coverage Windows

### 5.1 EUR/USD (Primary)

| Period | Years | Role | Notes |
|--------|-------|------|-------|
| 2005-01 to 2024-12 | 20 | Full dataset | OANDA + HistData. Covers GFC, Euro crisis, COVID, rate hikes. |
| 2005-01 to 2019-12 | 15 | In-sample universe | WFO training + validation windows drawn from here. Matches P1 data period. |
| 2020-01 to 2024-12 | 5 | **Held-out** | Final out-of-sample evaluation. Includes COVID and rate-hike regimes (stress test). |

### 5.2 Rolling Window Design (Preliminary)

For Path A (Adaptive Heuristic) and Path B (ML Rolling):

| Parameter | Conservative | Aggressive |
|-----------|-------------|------------|
| Training window | 5 years (anchored expanding) | 3 years (sliding) |
| Validation window | 1 year | 6 months |
| Step size | 1 year | 6 months |
| Min bars per window (4h) | ~8,000 (5yr) | ~4,800 (3yr) |
| Windows in 15-year IS period | ~10 | ~20 |

**Recommendation**: Start with **anchored expanding, 3-year minimum, 1-year step, 1-year validation**. This gives ~12 WFO folds over 2005-2019, matching heuristic-strategy's existing WFO design.

### 5.3 Secondary Assets

| Asset | Period | Source | Notes |
|-------|--------|--------|-------|
| USD/JPY | 2005-2024 | OANDA | Same coverage as EUR/USD for direct comparison |
| SPY | 1993-2024 | yfinance | 30+ years gives longer history for validation |
| BTC/USD | 2015-2024 | Binance/yfinance | Only 10 years; cannot use same WFO design. Consider shorter windows or reduced fold count. |

---

## 6. Feature Universe by Timeframe

### 6.1 At 4h (Primary)

| Category | Features | Source |
|----------|----------|--------|
| **Technical (P1 validated)** | adx, di_spread, atr_pct, atr_ratio, bb_width_pct, bb_position, rsi, roc_12, price_vs_ema50, ema_alignment, stoch_k, macd_hist | feature-eng plugins |
| **Causal-filtered subset** | rsi, bb_position, adx, macd_hist, ema_alignment | PCMCI+ root causes + leading TE |
| **Regime features** | bb_position, atr_ratio, ema_alignment | GMM K=9 inputs |

### 6.2 At Daily (Secondary, to explore)

| Category | Features | Source |
|----------|----------|--------|
| Same technical as 4h | (recomputed on daily bars) | feature-eng |
| **Macro (new for P2)** | US 10Y yield, USD DXY, VIX, EURUSD implied vol, US-EU rate differential, CFTC EUR net positioning | FRED + Alpha Vantage + CFTC |
| **Cross-asset** | SPY daily return, BTC daily return, Gold daily return | yfinance |

### 6.3 Feature Expansion Roadmap

| Phase | Features Added | When |
|-------|---------------|------|
| Phase 0 (baseline) | P1 technical features only (12) | Part II start |
| Phase 1 | + Causal filter (reduce to 5) | Part II after CI-4 |
| Phase 2 | + Macro features (6-8) | Part II/III |
| Phase 3 | + Cross-asset features (3-4) | Part III |
| Phase 4 | + Autoencoder-compressed (UL-3) | Part III |

---

## 7. Summary Decision Table

| Dimension | Selection | Confidence |
|-----------|-----------|------------|
| **Primary asset** | EUR/USD | HIGH — infrastructure, continuity, analysis all support this |
| **Secondary assets** | USD/JPY, SPY, BTC/USD (in that order) | MEDIUM — pending Part III cross-validation results |
| **Primary timeframe** | 4-hour | HIGH — all causal/regime work done at 4h |
| **Data collection timeframe** | 1-hour | HIGH — preserves resampling optionality |
| **Secondary timeframe** | Daily | MEDIUM — pending CI-2 multi-timeframe causal analysis |
| **In-sample period** | 2005–2019 | HIGH — 15 years, matches P1 |
| **Held-out period** | 2020–2024 | HIGH — stress test with COVID + rate hikes |
| **Rolling window** | Anchored expanding, 3yr min, 1yr step | MEDIUM — may adjust based on early experiments |
| **Initial features** | P1 12 technical → reduce via causal filter to 5 | MEDIUM — pending CI-1 (RPCMCI fix) |

---

## 8. Open Questions

1. **Is crypto in scope for Project 2?** BTC/USD has the best momentum signal but 15× cost of EUR/USD and only 10 years of data. Including it requires separate WFO design.

2. **Should daily timeframe analysis (CI-2) be done before committing to 4h?** If lagged causal links exist at daily but not 4h, the entire experimental setup should shift.

3. **Should we collect data now or at Part II start?** Downloading 20 years of 1h EUR/USD from OANDA/HistData takes time. Could be done as Part I infrastructure prep (F-5 addresses this).
