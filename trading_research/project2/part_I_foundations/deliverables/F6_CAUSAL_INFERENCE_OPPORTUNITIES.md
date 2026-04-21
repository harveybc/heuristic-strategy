# F-6: Causal Inference Opportunities for Project 2

**Date**: 2025-06-17  
**Scope**: Assessment of causal inference methods, existing work, and realistic integration points for Project 2 adaptive strategies  
**Depends on**: F-1 (Project 1 lessons), F-8 (Infrastructure audit)  
**Honesty note**: This document reports what the causal-inference repo actually found, including null results.

---

## 1. What Already Exists

The `causal-inference/` repo contains a substantial body of completed analysis on EUR/USD hourly data (2005–2020, 93,084 bars resampled to 4h = 24,063 bars). Two generations of analysis were run:

### 1.1 V1 Methods (causal_regime_analysis.py)

| Method | Library | Status | Finding |
|--------|---------|--------|---------|
| NOTEARS (DAG learning) | gCastle | ❌ Failed / empty | No graph learned |
| ICP (Invariant Causal Prediction) | Manual Wald test | ✅ Ran | Only `bb_position` invariant (p=0.272, sign_consistency=0.83) |
| DoWhy refutation | DoWhy | ✅ Ran | All 6 tested features passed placebo + random cause tests |
| CausalForestDML | EconML | ✅ Ran | No significant heterogeneous effects |
| Transfer Entropy | Binned estimator | ✅ Ran | Only `ema_alignment` has positive net TE (feature → return) |

### 1.2 V2 Methods (causal_regime_analysis_v2.py)

| Method | Library | Status | Finding |
|--------|---------|--------|---------|
| **PCMCI+** | tigramite (RobustParCorr) | ✅ Ran | `rsi`, `bb_position`, `macd_hist` have contemporaneous (lag-0) causal links to returns. **No lagged links found.** |
| **RPCMCI** (regime-dependent causal graphs) | tigramite | ❌ Failed (numpy type bug) | Never completed — regime-dependent causal structure unknown |
| ICP (repeated) | Manual | ✅ Ran | Same result: only `bb_position` invariant |
| CausalForestDML (upgraded 5-fold) | EconML | ✅ Ran | No significant HTEs. ATE CI crosses zero. |
| Transfer Entropy | Binned | ✅ Ran | `ema_alignment` only positive net TE (replicated V1) |
| Sensitivity Analysis | DoWhy | ✅ Ran | **All features score 1/4 on robustness** — placebo test fails for all |

### 1.3 Additional Analyses

| Script | Finding |
|--------|---------|
| `momentum_analysis.py` | Return autocorrelation at +0.663 is overlapping-window artifact. 1-bar autocorrelation ≈ 0. Combined momentum+bb_position: Sharpe 6.67 but likely look-ahead bias (contemporaneous bb_position). Rolling stability poor. |
| `cross_asset_audit.py` | No asset has significant Hurst ≠ 0.5. BTC/ETH best momentum Sharpe (~0.4 net). EUR/USD momentum unprofitable after costs. SPY shows mean-reversion. |
| `cross_asset_comparison.py` | Comprehensive 10-metric comparison across 8 assets. |
| `nfp_event_response_poc.py` | NFP volatility spike only 1.07× normal (p=0.47). No significant post-NFP drift. Event-driven thesis **not supported**. |
| `cluster_regime_analysis.py` | GMM K=9 on 3 features (bb_position, atr_ratio, ema_alignment). Cluster→action mapping based on 6-bar return and win rate. |

---

## 2. Honest Assessment of Current Findings

### 2.1 What Was Established

1. **bb_position is the only ICP-invariant feature** across market regimes — its relationship to returns does not change with regime. But its ATE is small (+0.012) and sensitivity analysis rates it WEAK.

2. **ema_alignment is the only leading indicator** by transfer entropy — all other features are lagging (driven BY returns, not driving returns).

3. **rsi and bb_position are root causes in the PCMCI+ causal graph** — they have high out-degree and zero/low in-degree. But their links to returns are **contemporaneous only** (lag-0), meaning they move WITH returns, not before them.

4. **No lagged causal feature → return link was found by PCMCI+** at any lag from 1 to 10 (40 hours). This is the single most important finding: the features examined do not Granger-cause returns at the 4h timeframe.

5. **The +0.663 return autocorrelation is mechanical** — it's overlapping 6-bar returns. The 1-bar (4h) autocorrelation is near zero.

### 2.2 What This Means for Project 2

The causal analysis results are **sobering but not disqualifying**:

- **For Path A (Adaptive Heuristic)**: The regime clusters (K=9 GMM) are built on causally-validated features (bb_position, atr_ratio, ema_alignment). The `regime_adaptive` and `regime_wfo` plugins in heuristic-strategy already use these. The causal evidence supports the feature *selection* even if it doesn't prove predictability.

- **For Path B (Supervised ML Rolling)**: The lack of lagged causal links means pure feature-to-return prediction is very hard at 4h. Rolling retraining may pick up time-varying contemporaneous relationships, but this is fragile. Need to test at different timeframes (daily, weekly) where autocorrelation structures may differ.

- **For Path C (RL)**: RL agents observe contemporaneous features + price and learn a policy. The contemporaneous causal links (rsi, bb_position → return) suggest the state representation is informative, even if not predictive in a lead-lag sense. RL may capture nonlinear decision boundaries that linear causal methods miss.

### 2.3 Critical Gaps in the Analysis

| Gap | Impact | Resolution |
|-----|--------|------------|
| RPCMCI never completed | Don't know if causal structure changes across regimes | Fix numpy bug and re-run — this is the most valuable missing analysis |
| Only 4h timeframe tested | May miss causal relationships at other scales | Extend to 1h, daily, weekly |
| Only 12 technical features tested | Macro/fundamental features untested | Add FRED/economic features per F-3 catalog |
| No conditional independence testing between regimes | Regime transitions may have different causal drivers | Part of RPCMCI fix |
| Sensitivity analysis failed all features | Raises questions about entire causal approach | Need to understand why placebo test fails (may be power issue with small effect sizes) |

---

## 3. Causal Inference Opportunities for Project 2

### 3.1 Opportunity CI-1: Fix and Complete RPCMCI (HIGH Priority)

**What**: Fix the numpy type bug in `run_rpcmci_only.py` and run RPCMCI with 3 regimes on the 12-feature set.

**Why**: This is the single most valuable unfulfilled analysis. If causal structure changes across regimes, it provides the theoretical foundation for regime-adaptive strategies (Path A) and for knowing *when* to retrain (Path B).

**Effort**: Small (bug fix) + compute time  
**Part**: I (foundations) and II (Path A)

### 3.2 Opportunity CI-2: Multi-Timeframe Causal Analysis (HIGH Priority)

**What**: Run PCMCI+ on {1h, 4h, daily, weekly} resampled data to test whether lagged causal links emerge at different scales.

**Why**: The absence of lagged links at 4h does not mean they don't exist at other timeframes. Daily/weekly may show momentum or mean-reversion causal structures that hourly misses. This directly informs F-4 (asset/timeframe selection).

**Effort**: Medium (data resampling + 4× PCMCI runs)  
**Part**: I (complement to F-4) and II

### 3.3 Opportunity CI-3: Macro Feature Causal Testing (MEDIUM Priority)

**What**: Add economic/macro features from F-3 catalog (FRED: interest rates, yield curves, CPI, unemployment; CFTC CoT positioning) to the causal graph.

**Why**: Project 1 used only technical features. Macro features may have genuine lagged causal relationships to FX returns (e.g., interest rate differentials → carry trade returns). This is the "expand input universe" direction.

**Effort**: Medium (feature engineering for macro data + PCMCI run)  
**Part**: II and III

### 3.4 Opportunity CI-4: Causal Feature Selection for Rolling Retraining (MEDIUM Priority)

**What**: Use the PCMCI+ causal graph as a feature selection filter before training ML models. Only include features with direct or indirect causal paths to the target.

**Why**: Reduces overfitting in rolling retraining by excluding features that are noise or downstream effects. The current analysis shows price_vs_ema50, bb_width_pct, and roc_12 are DOWNSTREAM — they should not be inputs to a predictive model because they are effects, not causes.

**Implementation**:
```
Input features = {f : f has out-degree > 0 in PCMCI+ graph} ∪ {f : net_TE(f→return) > 0}
           = {rsi, bb_position, adx, macd_hist, ema_alignment}
Exclude    = {price_vs_ema50, bb_width_pct, roc_12}  (downstream/effects)
```

**Effort**: Small (filter logic) but depends on CI-1 (RPCMCI may reveal regime-specific features)  
**Part**: II and III

### 3.5 Opportunity CI-5: Causal Discovery as Retraining Trigger (HIGH Priority)

**What**: Run rolling-window PCMCI+ and trigger model retraining when the causal graph structure changes (new edges appear, existing edges disappear, or edge strengths shift beyond threshold).

**Why**: This addresses gap G-7 (no performance monitoring → retrain trigger). Instead of retraining on a fixed schedule, retrain when the *data-generating process itself changes* — a much more principled approach.

**Implementation sketch**:
1. Run PCMCI+ on rolling 2-year windows, stepping by 1 month
2. Track: set of significant edges, MCI values, graph out-degrees
3. Define "structural break" as: edge set Jaccard distance > 0.3 from previous window, or any MCI value change > 50%
4. When structural break detected → trigger rolling retraining for that window

**Effort**: Large (rolling PCMCI is compute-heavy: ~10 min per window × ~100+ windows)  
**Part**: II and III — this is the most novel contribution potential

### 3.6 Opportunity CI-6: Causal Forest for Heterogeneous Strategy Allocation (LOW Priority)

**What**: Use CausalForestDML to estimate conditional average treatment effects (CATE) — i.e., which market conditions make a strategy profitable vs unprofitable.

**Why**: Current analysis found no significant HTEs, but this was using technical features only. With macro features and regime labels, CATEs may emerge. This enables "allocate capital to strategy X when CATE > threshold."

**Effort**: Medium  
**Part**: III (evaluation) and potentially IV

### 3.7 Opportunity CI-7: Double Machine Learning for Confounding Control (LOW Priority)

**What**: Use DoWhy + EconML's Double ML to estimate the true effect of strategy signals (e.g., momentum signal → return) after controlling for confounders (volatility regime, time-of-day, day-of-week).

**Why**: Backtesting conflates signal effect with confounders. Double ML isolates the causal effect of the signal itself. If the causal effect is zero after controlling for confounders, the signal is spurious.

**Effort**: Medium  
**Part**: III

---

## 4. Integration Points with Project 2 Parts

| Opportunity | Part I | Part II (Path A) | Part III (Path B) | Part IV (Path C/RL) | Part V (Synthetic) | Part VI (NEAT) |
|-------------|--------|-----------------|-------------------|--------------------|--------------------|----------------|
| CI-1: RPCMCI fix | ✅ Do now | ✅ Regime structure | ✅ Feature selection | ✅ State design | — | — |
| CI-2: Multi-timeframe | ✅ Informs F-4 | — | ✅ Timeframe choice | — | — | — |
| CI-3: Macro features | — | ✅ Expand features | ✅ Expand features | ✅ Expand state | — | — |
| CI-4: Causal feature filter | — | ✅ Feature set | ✅ Feature set | ✅ State pruning | — | — |
| CI-5: Causal retrain trigger | — | ✅ Re-optimization | ✅ When to retrain | — | — | — |
| CI-6: Causal Forest CATE | — | — | ✅ Conditional eval | — | — | — |
| CI-7: Double ML confounding | — | — | ✅ Signal validation | — | — | — |

---

## 5. Constraints and Honest Caveats

1. **The strongest finding is a null result**: No lagged feature→return causal link was found at 4h. All 7 opportunities above are attempts to work around or extend beyond this null result. If multi-timeframe analysis (CI-2) also returns null, the causal inference direction may have limited value for prediction.

2. **RPCMCI failure is not trivial**: The bug may be in tigramite's interface, not just a numpy casting issue. Regime-dependent causal discovery is computationally expensive and may not converge.

3. **Sensitivity analysis undermines the CausalForest results**: All features scored 1/4 on robustness. This means placebo treatments (random noise) show similar "effects" to real features. The effect sizes may be too small for causal methods to reliably detect.

4. **Compute budget matters**: Rolling PCMCI+ (CI-5) on 100+ windows could take 15-20 hours. Need to allocate Dragon or Gamma for this.

5. **Look-ahead risk in contemporaneous links**: The PCMCI+ lag-0 links (rsi → return, bb_position → return) cannot be used for prediction directly — they are simultaneous. Any strategy using these features must use lagged values (t-1), which reduces the signal strength.

6. **The momentum finding is cautionary**: The +0.663 autocorrelation that motivated momentum analysis turned out to be an overlapping-window artifact. This pattern of "exciting preliminary finding → artifact upon scrutiny" should be expected for other signals too.

---

## 6. Recommended Priority Order for Project 2

| Priority | Opportunity | When | Effort | Expected Value |
|----------|-------------|------|--------|---------------|
| 1 | CI-1: Fix RPCMCI | Part I (now) | Small | HIGH — fills biggest gap |
| 2 | CI-2: Multi-timeframe PCMCI | Part I (before F-4) | Medium | HIGH — may find lagged links |
| 3 | CI-4: Causal feature filter | Part II start | Small | MEDIUM — reduces feature set |
| 4 | CI-5: Causal retrain trigger | Part II/III | Large | HIGH if CI-2 finds links |
| 5 | CI-3: Macro features | Part II/III | Medium | MEDIUM — expands input universe |
| 6 | CI-6: Causal Forest CATE | Part III | Medium | LOW — current results unpromising |
| 7 | CI-7: Double ML confounding | Part III | Medium | LOW — academic interest |

---

## 7. Open Questions for User

1. **RPCMCI bug**: Should we attempt the fix now (Part I) or defer to Part II? The bug appears to be in numpy type handling of tigramite's return value.

2. **Multi-timeframe scope**: Should CI-2 test all 4 timeframes (1h, 4h, daily, weekly) or focus on daily + weekly which are more likely to show macro-driven causal links?

3. **Compute allocation**: CI-5 (rolling PCMCI) needs a dedicated machine for ~15-20 hours. Assign to Dragon?

4. **Feature expansion**: Should macro features (CI-3) use the zero-cost stack from F-3 (FRED + Alpha Vantage) or is there budget for premium data?
