> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 4 Synthesis Report

**Date**: 2025-07-17  
**Tracks executed**: A, B, C, D — all 4 in parallel across omega, dragon, gamma  
**Status**: ALL TRACKS COMPLETE

---

## Executive Summary

Phase 4 tested four independent hypotheses to find deployable trading strategies. **All four tracks failed to produce strategies that survive the worst-2Y-window kill criterion at the -0.5 Sharpe threshold.** This result, while disappointing, is consistent with the Phase 3.5 finding that all 14 surviving cells were killed by extended-history stress testing.

| Track | Hypothesis | Verdict |
|-------|-----------|---------|
| **A** | Oracle-independent MR is deployable | **NOT DEPLOYABLE** — narrow parameter spike, not a plateau |
| **B** | Regime filter can rescue killed strategies | **MARGINAL** — perfect oracle helps at -1.0 threshold but real classifier fails |
| **C** | Academic strategies survive our kill criteria | **ALL DEAD** — 0/17 asset-strategy combinations survive at -0.5 |
| **D** | GBP/USD 4h VRS has edge on longer data | **NOT VIABLE** — daily proxy fails, 4h data too short |

---

## Track A: Oracle-Independent Mean Reversion Deployment

**Machine**: omega | **Runtime**: ~2 min | **Cells**: EUR/USD daily, USD/JPY daily

### Task A.1 — Parameter Perturbation Audit

| Cell | Baseline SR | Grid Range | % Space ≥ 50% Max | Plateau? |
|------|------------|-----------|-------------------|----------|
| EUR/USD daily | +0.158 | [+0.026, +0.549] | 33% | **NO — Narrow spike** |
| USD/JPY daily | +0.142 | [-0.306, +0.588] | 20% | **NO — Narrow spike** |

**EUR/USD** shows 100% positive SR across the 75-combo grid (mean=+0.223), but only 33% of parameter space achieves ≥50% of maximum. The best SR (+0.549) is concentrated around lookback=10, which is a short, fragile window.

**USD/JPY** is worse — 20% of space has negative SR, the edge over B&H is essentially zero at baseline (-0.024), and the strategy is sensitive to z_entry: z_entry=0.75 gives SR=-0.05 while z_entry=2.25 gives SR=+0.45.

**Verdict**: Neither cell forms a robust plateau. Performance is parameter-dependent, not structurally stable.

### Task A.3 — Position Sizing (for reference)

| Cell | Max DD | Leverage | Position per $10K |
|------|--------|---------|------------------|
| EUR/USD | 27.1% | 0.55× | $5,539 |
| USD/JPY | 29.1% | 0.52× | $5,156 |

### Task A.5 — Monitoring Protocol (moot — deployment not recommended)

Documented but not actionable given Track A failure.

### Track A Decision: **NOT DEPLOYABLE**

---

## Track B: Regime Filter Hypothesis Test

**Machine**: dragon (RTX 4090) | **Runtime**: ~3 min | **Cells**: 5 top survivors from Phase 3

### Task B.1-B.2 — Perfect Regime Oracle Results

The perfect regime oracle uses future information (next 2Y Sharpe) to classify each bar as favorable, neutral, or hostile. This is the **ceiling** — if this doesn't work, no real classifier can.

| Cell | V0 SR | V0 Worst2Y | V1 Worst2Y | V2 Worst2Y | V3 Worst2Y | Rescued at -0.5? |
|------|-------|-----------|-----------|-----------|-----------|-----------------|
| XAU/USD weekly mom | +2.141 | **-3.734** | -3.103 | -2.559 | -2.980 | NO |
| BTC/USD weekly mom | +1.906 | **-1.705** | -2.731 | -1.775 | -2.590 | NO |
| XAU/USD daily mom | +1.652 | **-4.095** | -2.181 | -1.093 | -2.493 | NO |
| XAG/USD weekly mom | +0.916 | **-6.170** | -3.108 | -0.700 | -3.367 | NO |
| EUR/USD daily MR | +0.390 | **-1.116** | -0.664 | -0.634 | -0.707 | NO |

**Key finding**: Even with perfect future knowledge, no cell passes the -0.5 worst-window threshold. The best improvement is EUR/USD daily MR with V2 (reverse in hostile): worst2Y goes from -1.116 to -0.634 — still fails.

At the relaxed -1.0 threshold, 2/5 cells are rescued (XAG/USD weekly V2 at -0.700, EUR/USD daily V1/V2/V3 all pass). But -1.0 represents a catastrophic 2-year drawdown that most allocators wouldn't tolerate.

### Task B.3 — Rescue Viability

| Threshold | Cells Rescued | Notes |
|-----------|--------------|-------|
| -0.5 | 0/5 | None pass all 3 criteria (worst>-0.5, edge>+0.5, trades>30% of base) |
| -0.75 | 2/5 | XAG V2, EUR/USD V1/V2/V3 pass |
| -1.0 | 2/5 | Same cells |

### Task B.4 — Real Regime Classifier

Only 2 cells had sufficient daily feature store data for classification:

| Cell | Model | Balanced Accuracy | Precision | Recall |
|------|-------|------------------|-----------|--------|
| XAU/USD daily mom | GB | 0.747 | 0.118 | 0.757 |
| XAU/USD daily mom | LR | 0.659 | 0.064 | 1.000 |
| EUR/USD daily MR | GB | 0.504 | 0.800 | 0.009 |
| EUR/USD daily MR | LR | 0.500 | 0.000 | 0.000 |

**XAU/USD** shows some signal (GB bal_acc=0.747) with top features: DGS2 (2Y Treasury), Close, DFF (Fed Funds), COT open interest, DXY. But precision is only 11.8% — it would flag many false hostile periods.

**EUR/USD** classifier is essentially random (0.504 bal_acc). The regime is not detectable from available features.

When the real GB classifier is applied to XAU/USD (V1 stand-aside), it achieves SR=+1.618 but worst2Y=-4.095 — **no improvement**, because the classifier doesn't filter the right periods.

### Track B Decision: **REGIME_FILTER_MARGINAL**

The regime is detectable in theory (XAU/USD GB=0.747) but:
- The perfect oracle itself can't rescue cells at -0.5
- The real classifier has too many false positives
- Classification doesn't improve worst-window

---

## Track C: Academic Strategy Replication

**Machine**: gamma (RTX 5070 Ti) | **Runtime**: ~2 min | **3 strategies, 17 asset-strategy pairs**

### Strategy 1: Time-Series Momentum (Moskowitz, Ooi, Pedersen 2012)

Sign of 12-month return × inverse-vol sizing × monthly rebalance.

| Asset | SR | Edge vs B&H | Worst 2Y |
|-------|-----|------------|----------|
| EUR/USD | -0.275 | -0.268 | **-1.797** |
| USD/JPY | +0.291 | +0.190 | **-0.598** |
| GBP/USD | -0.072 | +0.042 | **-1.645** |
| AUD/USD | -0.089 | -0.034 | **-1.221** |
| XAU/USD | +0.419 | -0.212 | **-1.515** |
| BTC/USD | +0.228 | -0.344 | **-1.533** |
| ETH/USD | -0.229 | -0.490 | **-1.282** |
| XAG/USD | +0.071 | -0.275 | **-1.559** |
| CL | NaN | NaN | **-0.861** |

**0/9 survive at -0.5**. Best: USD/JPY at -0.598. TSMOM's 12-month lookback creates massive drawdowns during regime changes.

### Strategy 2: Dual Momentum (Antonacci 2014)

Absolute + relative momentum on FX pairs, monthly rebalance.

| Asset | SR | Edge vs B&H | Worst 2Y |
|-------|-----|------------|----------|
| EUR/USD | -0.216 | -0.210 | **-1.674** |
| USD/JPY | +0.408 | +0.307 | **-0.973** |
| GBP/USD | +0.001 | +0.115 | **-1.102** |
| AUD/USD | +0.206 | +0.261 | **-1.008** |

**0/4 survive at -0.5**. USD/JPY is best at -0.973 (barely survives at -1.0).

### Strategy 3: Cross-Sectional FX Momentum (Menkhoff et al. 2012)

Rank FX by 12-month return, long top half / short bottom half.

| Asset | SR | Edge vs B&H | Worst 2Y |
|-------|-----|------------|----------|
| EUR/USD | +0.116 | +0.122 | **-1.266** |
| USD/JPY | +0.201 | +0.100 | **-0.738** |
| GBP/USD | -0.192 | -0.079 | **-1.460** |
| AUD/USD | -0.175 | -0.121 | **-2.030** |
| Portfolio | -0.069 | — | **-1.756** |

**0/4 survive at -0.5**. The long/short portfolio SR is negative (-0.069).

### Kill Criteria Summary

| Strategy | Survive -0.5 | Survive -0.75 | Survive -1.0 |
|----------|-------------|--------------|-------------|
| TSMOM (9 assets) | 0/9 | 1/9 | 2/9 |
| Dual Momentum (4 FX) | 0/4 | 0/4 | 1/4 |
| XS-FX (4 FX) | 0/4 | 1/4 | 1/4 |
| **Total** | **0/17** | **2/17** | **4/17** |

### Track C Decision: **ALL DEAD**

Academic strategies, implemented as published with no tuning, produce the same result as our oracle-based sweep: **every combination has a catastrophic 2-year window.** This is important because it shows the problem is not our strategy design — it's a structural property of these markets at these frequencies.

---

## Track D: GBP/USD 4h VRS Resolution

**Machine**: gamma | **Runtime**: ~2 min | **Data**: 730 days 4h + 22 years daily

### Data Acquisition

| Source | Bars | Period |
|--------|------|--------|
| yfinance 1h → 4h | 3,178 | 2024-04-17 to 2026-04-17 (730 days) |
| yfinance daily | 5,741 | 2003-12-01 to 2025-12-30 (22+ years) |

### VRS Performance on 4h Data (730 days — too short for definitive testing)

| Noise σ | SR | Edge | Worst 2Y | DD |
|---------|-----|------|----------|-----|
| 0 (perfect) | +0.960 | +0.661 | -0.526 | 9.8% |
| 10 (max noise) | **-0.467** | -0.766 | **-1.058** | 18.1% |

The 4h VRS shows positive SR only at low noise. At σ=10 it's deeply negative. The 730-day window is insufficient for rolling-window analysis.

### VRS Performance on Daily Data (22 years — proper backtest)

| Noise σ | SR | Edge | Worst 2Y | DD |
|---------|-----|------|----------|-----|
| 0 (perfect) | +0.708 | +0.821 | **-1.484** | 31.1% |
| 10 (max noise) | +0.173 | +0.287 | **-0.642** | 20.8% |

At σ=10 on daily: SR=+0.173, worst2Y=-0.642. **Killed** (just barely — threshold is -0.5).

### Regime Analysis (daily, σ=10)

| Period | SR |
|--------|-----|
| Pre-GFC (2003-2007) | **-0.298** |
| GFC (2007-2009) | +0.209 |
| QE (2009-2020) | **+0.442** |
| COVID (2020-2021) | +0.066 |
| Inflation (2022-2025) | **-0.132** |

The strategy only worked during the QE era. Pre-GFC and inflation periods are negative.

### Track D Decision: **VRS_NOT_VIABLE**

---

## Decision Matrix (from work_plan_4.md)

| Track A | Track B | Track C | Track D | → Action |
|---------|---------|---------|---------|----------|
| ❌ FAIL | ❌ MARGINAL | ❌ FAIL | ❌ FAIL | **Row 16: "All fail → Research pivot"** |

Per the work plan decision matrix, when all tracks fail, the recommendation is a **research pivot** — fundamentally different approach rather than incremental refinement.

---

## Key Findings Across All Tracks

### 1. The Worst-Window Problem Is Structural

Every strategy — oracle-assisted, oracle-free, momentum, mean-reversion, vol-regime, academic — has a catastrophic 2-year window somewhere in 20+ years of data. This is not a tuning issue or a strategy issue. **Markets structurally produce multi-year drawdowns that no simple rule-based strategy can avoid.**

### 2. Regime Detection ≠ Regime Avoidance

Track B showed that even a **perfect future-looking regime oracle** cannot rescue strategies at the -0.5 threshold. This means the hostile periods aren't isolated events — they're distributed and overlapping with favorable periods.

### 3. Academic Strategies Fail the Same Test

Moskowitz (2012), Antonacci (2014), and Menkhoff (2012) all fail. These are among the most cited momentum papers in quantitative finance. Our kill criterion is stricter than typical academic evaluation (which uses full-sample Sharpe, not worst-window).

### 4. EUR/USD Daily MR Is the Closest to Viable

Across all 4 tracks, EUR/USD daily mean reversion consistently shows the best worst-window behavior:
- Track A: worst2Y = -1.14 (baseline), but 100% of grid is positive SR
- Track B with perfect oracle V2: worst2Y = -0.63 (best of any combination)
- Track B with perfect oracle V3: worst2Y = -0.71

If the threshold were relaxed to -0.75, EUR/USD daily MR with regime filtering would pass. This remains the most promising candidate for further work.

---

## Recommended Next Steps (Research Pivot Options)

1. **Relax kill criterion to -0.75**: EUR/USD daily MR + regime filter becomes viable. This trades safety margin for deployment opportunity.

2. **Portfolio-level diversification**: Instead of requiring each cell to survive individually, test a multi-asset portfolio where different cells fail at different times.

3. **Adaptive position sizing**: Instead of binary in/out regime filtering, use continuous Kelly-criterion sizing that reduces exposure proportionally to recent performance.

4. **Higher-frequency strategies**: The 2Y worst-window is designed for daily/weekly. At 15min/1h, there are more windows to average over, and structural breaks have less impact per window.

5. **Alternative data sources**: COT and macro features showed promise in Track B classification (XAU/USD GB=0.747). Richer feature sets (options IV, order flow, sentiment) might improve regime detection.

---

## Files Produced

| File | Location |
|------|----------|
| Track A results | `trading_research/results/phase4_track_a_results.json` |
| Track B results | `trading_research/results/phase4_track_b_results.json` |
| Track C results | `trading_research/results/phase4_track_c_results.json` |
| Track D results | `trading_research/results/phase4_track_d_results.json` |
| Track A script | `trading_research/track_a_mr_deploy.py` |
| Track B script | `trading_research/track_b_regime_filter.py` |
| Track C script | `trading_research/track_c_academic.py` |
| Track D script | `trading_research/track_d_gbpusd_vrs.py` |
| This report | `trading_research/results/PHASE_4_SYNTHESIS.md` |
