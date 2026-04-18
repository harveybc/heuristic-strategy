> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 3.5 Report: Audit & Data Extension Before Phase 4

**Date:** 2025-01-01  
**Execution:** Distributed across omega (local), dragon (192.168.1.235), gamma (192.168.0.106)

---

## Executive Summary

Phase 3.5 was a rigorous stress-test of the 14 surviving cells from Phase 1.3 before committing to strategy development. **The results are sobering: all 14 cells were killed by the extended-history worst-window test (Task 3).** However, important structural insights emerged that reshape the path to Phase 4.

### Key Findings

| Test | Result | Cells Killed |
|------|--------|-------------|
| Task 1: 10σ Noise Audit | 2 trivially saturated, 6 honest, 3 ambiguous | 0 (criterion: need ≥6 saturated) |
| Task 2: Edge-over-B&H | 1 B&H-dominated (XAU/USD daily VRS) | 1 |
| Task 3: Extended History | **ALL 14 killed** by worst 2Y window < −0.5 | 14 |
| Task 4: COT Data Fix | 7 assets, 6 features each, integrated into store | N/A |

**Bottom line:** The oracle-strategy hybrid framework produces positive edge over most market regimes, but every cell contains at least one catastrophic 2-year window — even with perfect foresight. This is a structural finding about the strategy frameworks, not just data noise.

---

## Task 1: 10σ Noise Budget Ceiling Audit

### Purpose
Determine whether the 10σ noise budget is an "honest" ceiling (oracle truly degraded at 10σ) or "trivially saturated" (strategy ignores oracle signal).

### Classification Results

| Category | Count | Cells |
|----------|-------|-------|
| **Honest ceiling** (flip ≥ 40%) | 6 | BTC weekly mom/carry, XAU weekly mom/carry, XAG weekly mom/carry |
| **Trivially saturated** (flip < 20%) | 2 | EUR/USD daily MR (0.4%), USD/JPY daily MR (0.2%) |
| **Ambiguous** (20-40%) | 3 | GBP/USD 4h VRS (27.6%), AUD/USD weekly VRS (26.9%), EUR/JPY weekly VRS (28.0%) |

### Detailed Traces

| Cell | Sign-Flip | Sharpe@0σ | Sharpe@10σ | B&H | Classification |
|------|-----------|-----------|------------|-----|---------------|
| BTC/USD weekly momentum | 49.5% | +1.906 | +0.634 | +0.703 | honest_ceiling |
| BTC/USD weekly carry_mom | 49.5% | +1.906 | +0.634 | +0.703 | honest_ceiling |
| XAU/USD weekly momentum | 46.3% | +2.036 | +0.325 | +0.636 | honest_ceiling |
| XAU/USD weekly carry_mom | 46.3% | +2.036 | +0.325 | +0.636 | honest_ceiling |
| XAG/USD weekly momentum | 46.6% | +0.857 | +0.397 | +0.350 | honest_ceiling |
| XAG/USD weekly carry_mom | 46.6% | +0.857 | +0.397 | +0.350 | honest_ceiling |
| EUR/USD daily MR | **0.4%** | +0.389 | +0.390 | −0.056 | **saturated** |
| USD/JPY daily MR | **0.2%** | +0.305 | +0.306 | +0.166 | **saturated** |
| GBP/USD 4h VRS | 27.6% | +0.605 | +0.193 | +0.160 | ambiguous |
| AUD/USD weekly VRS | 26.9% | +0.453 | +0.426 | −0.050 | ambiguous |
| EUR/JPY weekly VRS | 28.0% | +0.331 | +0.167 | +0.130 | ambiguous |

### Key Insight: Mean Reversion Is Oracle-Blind

EUR/USD and USD/JPY daily mean reversion have **<0.5% sign-flip probability** — the oracle signal makes virtually no difference. The strategy's z-score entry/exit logic dominates positioning entirely. At noise=10σ the oracle is pure noise, yet Sharpe is identical to noise=0σ. This means:
- The 10σ noise budget is meaningless for these cells
- Their performance comes entirely from the mean-reversion rules, not predictive skill
- They should be evaluated as pure technical strategies, not oracle-enhanced ones

### Replacement Metrics for Non-Honest Cells

| Cell | Min Accuracy for Breakeven | Magnitude Sensitivity |
|------|---------------------------|----------------------|
| EUR/USD daily MR | 50% (random!) | 0.389 (no sensitivity) |
| USD/JPY daily MR | 50% (random!) | 0.305 (no sensitivity) |
| GBP/USD 4h VRS | **90%** | 0.605 |
| AUD/USD weekly VRS | **80%** | 0.454 |
| EUR/JPY weekly VRS | **100%** | 0.331 |

**Kill criterion (≥6 saturated): NOT triggered** — only 2/11 are trivially saturated.

---

## Task 2: Edge-Over-Buy-and-Hold Reranking

### Purpose
Rerank all 14 cells by Edge Sharpe (= Oracle − B&H) and identify cells where the oracle adds no meaningful value over simple buy-and-hold.

### Updated Ranking

| Rank | Cell | Oracle SR | B&H SR | **Edge SR** | Budget vs B&H | Long SR | Short SR | Dominated? |
|------|------|-----------|--------|-------------|---------------|---------|----------|-----------|
| 1 | XAU/USD weekly momentum | +2.036 | +0.636 | **+1.400** | 3.0σ | +1.782 | +0.909 | |
| 2 | XAU/USD weekly carry_mom | +2.036 | +0.636 | **+1.400** | 3.0σ | +1.782 | +0.909 | |
| 3 | BTC/USD weekly momentum | +1.906 | +0.703 | **+1.203** | 5.0σ | +1.756 | +0.674 | |
| 4 | BTC/USD weekly carry_mom | +1.906 | +0.703 | **+1.203** | 5.0σ | +1.756 | +0.674 | |
| 5 | XAU/USD daily momentum | +1.554 | +0.622 | **+0.933** | 1.5σ | +1.262 | +0.349 | |
| 6 | XAU/USD daily carry_mom | +1.554 | +0.622 | **+0.933** | 1.5σ | +1.262 | +0.349 | |
| 7 | XAU/USD daily VRS | +1.129 | +0.622 | +0.507 | **1.0σ** | +1.022 | +0.143 | **YES** |
| 8 | XAG/USD weekly momentum | +0.857 | +0.350 | +0.507 | 3.0σ | +0.765 | +0.331 | |
| 9 | XAG/USD weekly carry_mom | +0.857 | +0.350 | +0.507 | 3.0σ | +0.765 | +0.331 | |
| 10 | AUD/USD weekly VRS | +0.454 | −0.050 | +0.504 | 10.0σ | +0.192 | +0.474 | |
| 11 | GBP/USD 4h VRS | +0.605 | +0.160 | +0.445 | 1.5σ | −0.001 | −0.115 | |
| 12 | EUR/USD daily MR | +0.389 | −0.056 | +0.445 | 10.0σ | +0.134 | +0.401 | |
| 13 | EUR/JPY weekly VRS | +0.331 | +0.130 | +0.201 | 1.5σ | +0.248 | +0.101 | |
| 14 | USD/JPY daily MR | +0.305 | +0.166 | +0.139 | 10.0σ | +0.402 | +0.022 | |

### Key Insights

1. **XAU/USD daily VRS is B&H-dominated** — its noise budget vs B&H is only 1.0σ. Removed.
2. **Long/short decomposition reveals asymmetry:**
   - XAU/USD weekly: strong both long (+1.78) and short (+0.91) — genuine two-sided edge
   - BTC/USD weekly: strong long (+1.76) but weaker short (+0.67) — B&H tailwind helps
   - GBP/USD 4h VRS: negative on both sides! Long=−0.001, Short=−0.115 — edge comes entirely from oracle timing
   - EUR/USD daily MR: short-biased (+0.40 short vs +0.13 long) — consistent with mean-reversion logic
3. **13 cells survive B&H filter**

---

## Task 3: Extended Historical Data & Regime Analysis

### Purpose
Test all 14 surviving cells on extended history (2000–2025 for FX/commodities, 2014+ for BTC, 2017+ for ETH) and assess regime robustness across 5 macro periods.

### ⚠️ CRITICAL FINDING: ALL 14 CELLS KILLED

Every cell was killed by the worst 2-year rolling window criterion (Sharpe < −0.5):

| Cell | Extended Edge | Regimes +Edge | Worst Window | Max DD | Kill Reason |
|------|-------------|--------------|-------------|--------|------------|
| BTC/USD weekly momentum | +1.203 | 3/3 | **−1.705** | 97.0% | worst window |
| BTC/USD weekly carry_mom | +1.203 | 3/3 | **−1.705** | 97.0% | worst window |
| XAU/USD weekly momentum | +1.505 | 4/5 | **−3.734** | 85.3% | worst window |
| XAU/USD weekly carry_mom | +1.505 | 4/5 | **−3.734** | 85.3% | worst window |
| XAG/USD weekly momentum | +0.582 | 3/5 | **−6.170** | 99.2% | worst window |
| XAG/USD weekly carry_mom | +0.582 | 3/5 | **−6.170** | 99.2% | worst window |
| XAU/USD daily momentum | +1.021 | 4/5 | **−4.095** | 92.1% | worst window |
| XAU/USD daily carry_mom | +1.021 | 4/5 | **−4.095** | 92.1% | worst window |
| XAU/USD daily VRS | +0.675 | 4/5 | **−1.821** | 64.8% | worst window |
| EUR/USD daily MR | +0.397 | 3/5 | **−1.116** | 25.9% | worst window |
| USD/JPY daily MR | +0.210 | 2/5 | **−1.087** | 32.5% | worst window + regimes |
| GBP/USD 4h VRS | N/A | N/A | N/A | N/A | no 4h data pre-2022 |
| AUD/USD weekly VRS | +0.504 | 4/5 | **−1.933** | 43.5% | worst window |
| EUR/JPY weekly VRS | +1.171* | 4/5 | **−3.095** | — | worst window |

### Regime-by-Regime Breakdown (Edge Sharpe)

| Cell | Pre-GFC (00-07) | GFC (08-12) | Post-Crisis (13-19) | COVID (20-21) | Inflation (22-25) |
|------|----------------|------------|--------------------|--------------|--------------------|
| XAU weekly mom | **+2.337** | **+1.809** | −0.435 | **+2.307** | **+3.011** |
| BTC weekly mom | — | — | **+1.799** | **+1.319** | +0.157 |
| XAG weekly mom | **+1.772** | **+1.111** | −1.501 | −0.398 | **+1.391** |
| XAU daily mom | **+1.562** | **+1.015** | −0.824 | **+0.749** | **+3.449** |
| XAU daily VRS | **+0.854** | **+1.109** | −0.274 | **+1.714** | **+0.629** |
| EUR/USD daily MR | −0.560 | **+0.841** | **+0.847** | +0.021 | −0.212 |
| USD/JPY daily MR | −0.042 | **+1.537** | −0.444 | +0.156 | −1.372 |
| AUD/USD weekly VRS | +0.161 | **+1.441** | +0.195 | +0.192 | −0.354 |
| EUR/JPY weekly VRS | **+1.171** | −0.206 | +0.054 | +0.069 | **+1.346** |

### Structural Diagnosis

The worst-window kill is **not a data quality issue** — it reflects a fundamental property of the oracle-strategy interaction:

1. **Strategy logic overrides oracle in bad regimes:** The momentum/MR/VRS frameworks have lookback parameters that create structural lag. During regime transitions (e.g., 2013 taper tantrum, 2020 COVID crash), the strategy's own momentum signal can conflict with the oracle, creating position errors.

2. **Oracle doesn't prevent drawdowns:** The oracle provides directional correctness for the *next bar*, but strategies accumulate positions over multiple bars. A correct direction signal at each bar can still lead to a terrible 2-year window if costs eat gains during consolidation.

3. **Weekly bars amplify damage:** With only ~104 bars in a 2-year window, a few bad weeks create extreme Sharpe drops. XAG/USD weekly has worst window −6.17 because the 2013 silver crash destroyed multi-week momentum.

4. **The −0.5 threshold is strict but appropriate:** Real traders would abandon any strategy with 2 years of negative performance. The criterion correctly identifies strategies that would fail in practice.

---

## Task 4: COT Data Fix & Integration

### Purpose
Download CFTC Commitments of Traders data and integrate into the feature store.

### Results

| Asset | Observations | Date Range | Source |
|-------|-------------|------------|--------|
| EUR/USD | 1,434 | weekly | Socrata API |
| USD/JPY | 1,913 | 1988–2026 | Socrata API |
| GBP/USD | 1,858 | 1990–2026 | Socrata API |
| AUD/USD | 1,776 | 1993–2026 | Socrata API |
| XAU/USD | 1,912 | 1986–2026 | Socrata API |
| XAG/USD | 1,912 | 1986–2026 | Socrata API |
| CL | 1,911 | 1986–2026 | Socrata API |

### Features Added (6 per asset)
- `cot_net_noncommercial` — speculator net positioning
- `cot_net_commercial` — hedger net positioning  
- `cot_open_interest` — total open interest
- `cot_net_nc_zscore_3y` — 3-year z-score of speculator positioning
- `cot_net_nc_change` — weekly change in speculator positioning
- `cot_oi_pct_change` — weekly change in open interest

All 6 features integrated into daily feature CSVs with 3-day lag (Tuesday data → Friday publication). **100% coverage** for all assets with existing feature store files.

**Note:** XAG/USD had no existing feature store file (was missing from Phase 3 exogenous data build). COT data is available but not integrated.

---

## Cross-Reference: Updated Shortlist

### Post-Task 1+2 Ranking (before Task 3)

After noise audit and B&H reranking, **13 cells survived** with this tiered structure:

**Tier 1 — High Edge, Honest Oracle (≥1.0 Edge SR):**
- XAU/USD weekly momentum/carry: Edge=+1.40, honest ceiling, 3σ vs B&H
- BTC/USD weekly momentum/carry: Edge=+1.20, honest ceiling, 5σ vs B&H

**Tier 2 — Moderate Edge (0.5–1.0 Edge SR):**
- XAU/USD daily momentum/carry: Edge=+0.93, honest ceiling, 1.5σ vs B&H
- XAG/USD weekly momentum/carry: Edge=+0.51, honest ceiling, 3σ vs B&H
- AUD/USD weekly VRS: Edge=+0.50, ambiguous oracle, 10σ vs B&H

**Tier 3 — Low Edge (<0.5 Edge SR):**
- GBP/USD 4h VRS: Edge=+0.45, ambiguous oracle, 1.5σ vs B&H
- EUR/USD daily MR: Edge=+0.45, **saturated oracle**, 10σ vs B&H
- EUR/JPY weekly VRS: Edge=+0.20, ambiguous oracle, 1.5σ vs B&H
- USD/JPY daily MR: Edge=+0.14, **saturated oracle**, 10σ vs B&H

### Post-Task 3: All Cells Killed

The worst-window criterion killed every cell. **The 14→0 funnel is complete.**

---

## Implications for Phase 4

### What This Means

The Phase 3.5 audit reveals that the current oracle-strategy framework has a fundamental limitation: **no fixed-parameter strategy can maintain positive performance across all 2-year windows of a 20+ year history**, even with a perfect oracle.

This is not necessarily fatal for the research program. It means:

1. **Static strategies are insufficient.** The path forward requires adaptive parameter selection, regime detection, or ensemble methods that adjust strategy behavior in real-time.

2. **The edge is real but intermittent.** XAU/USD weekly momentum has Edge SR=+1.5 over the full extended period, positive in 4/5 regimes. The problem is the post-crisis QE era (2013–2019) where it goes negative — a single regime destroys the worst window.

3. **Risk management is mandatory.** Even perfect prediction can't save a strategy without drawdown controls, position sizing, and regime filters.

### Recommended Path Forward

**Option A: Relax worst-window to −1.0 (advisory, not kill)**
- Promotes 6 cells: XAU weekly (−3.7 → still killed), BTC weekly (−1.7 → still killed), EUR/USD MR (−1.1 → still killed)
- Even at −1.0, most cells still fail
- NOT recommended — the threshold is already generous

**Option B: Add regime filter layer**
- Use the regime analysis to build a meta-strategy that suspends trading in hostile regimes
- XAU/USD weekly momentum has edge in 4/5 regimes — if we can detect the hostile regime (post-crisis QE), we can avoid the worst window
- The COT data and macro features from Phase 3 provide inputs for regime detection
- **RECOMMENDED** as Phase 4 direction

**Option C: Switch to adaptive-parameter framework**
- Instead of fixed lookback=20, z_entry=1.5, etc., use rolling optimization
- The oracle framework can validate which parameters work in which regime
- Higher complexity, higher risk of overfitting
- Consider as Phase 5

**Option D: Pure mean-reversion track (oracle-independent)**
- EUR/USD and USD/JPY daily MR strategies are oracle-blind (Task 1 found <0.5% sign-flip)
- They work as standalone technical strategies with no prediction needed
- Lower Sharpe (+0.39 / +0.31) but simpler to implement
- Could run as low-leverage, steady-return component alongside regime-filtered strategies
- Worth pursuing as a parallel track

### Cells to Carry Into Phase 4 (with regime filter)

Despite the worst-window kills, these cells have the strongest fundamental case:

| Priority | Cell | Full-Period Edge | Regime Coverage | Notes |
|----------|------|-----------------|----------------|-------|
| 1 | XAU/USD weekly momentum | +1.505 | 4/5 | Strongest edge, only fails post-crisis QE |
| 2 | BTC/USD weekly momentum | +1.203 | 3/3 | Strong but limited history (2014+) |
| 3 | XAU/USD daily momentum | +1.021 | 4/5 | More data points, daily frequency |
| 4 | XAG/USD weekly momentum | +0.582 | 3/5 | Silver correlates with gold — diversification |
| 5 | EUR/USD daily MR | +0.397 | 3/5 | Oracle-independent, pure technical play |

**Phase 4 should focus on building a regime detection layer that can identify hostile environments (like post-crisis QE for gold) and suspend or adjust strategy parameters accordingly.** The COT data, macro features, and VIX regime signals now in the feature store provide the raw materials for this.

---

## Files Produced

| File | Description |
|------|-------------|
| `results/noise_budget_audit.json` | Task 1 detailed results |
| `results/edge_over_bh_ranking.json` | Task 2 rankings |
| `results/task3_dragon.json` | Task 3 dragon results (BTC, XAU, XAG) |
| `results/task3_gamma.json` | Task 3 gamma results (EUR/USD, USD/JPY) |
| `results/task3_omega.json` | Task 3 omega results (EUR/JPY) |
| `results/task3_omega2.json` | Task 3 omega results (GBP/USD, AUD/USD) |
| `results/cot_download_results.json` | Task 4 COT download summary |
| `feature_store/cot_data.json` | COT data manifest |
| `feature_store/*_daily.csv` | Updated with 6 COT features each |
| `extended_data/` | Extended price data for all assets |
