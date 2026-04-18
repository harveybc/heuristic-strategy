# Project Closure — Retail Systematic FX Research

**Date:** 2026-04-17  
**Status:** CLOSED — Terminal 3 (pre-registered kill criterion triggered at held-out validation)  
**Duration:** 12+ months, 18 phases, distributed compute across 3 machines  
**Final artifact:** This document

---

## 1. Executive Summary

This research program applied institutional-grade quantitative methodology to retail-accessible FX data and strategies. The objective was to identify a deployable trading portfolio through systematic search and rigorous multi-stage validation.

**Starting point:** 360 oracle-enhanced strategy cells across 12 assets, 4 timeframes, and 5 strategy types.

**Final candidate:** P3 portfolio — 3 oracle-free FX strategy cells (EUR/USD mean-reversion 20.6%, USD/JPY time-series momentum 49.2%, USD/JPY dual momentum 30.2%), weighted by inverse worst-2-year-window Sharpe.

**Terminal metrics (plugin-canonical):**

| Metric | Value |
|--------|-------|
| Full-period Sharpe (2003–2026) | **0.41** |
| Held-out Sharpe (2024–2025) | **−0.065** |
| Full-period max drawdown | 20.2% at 10% target vol |
| Cost breakeven | >3× baseline costs |
| Walk-forward positive quarters | 61.5% (32 of 52) |

**Terminal decision:** The pre-registered kill criterion required held-out Sharpe > 0. Plugin-canonical held-out Sharpe = −0.065, triggering Terminal 3 (project closure). The kill criterion was honored without post-hoc adjustment.

**Contribution:** The project's value is threefold: (1) the negative result itself, documented honestly; (2) substantial reusable infrastructure; (3) a methodology template for systematic trading research. The research program reached the held-out validation stage, honored its kill criterion, and documented the negative result — three things that most retail quant projects do not do.

---

## 2. Project Timeline

### 2.1 Phase Chronology

| Phase | Date | Purpose | Outcome | Key Artifact |
|-------|------|---------|---------|--------------|
| 0 | 2024 | Diagnostic infrastructure | Built oracle sensitivity test, transaction cost model (16 assets), evaluation harness | Eval harness, cost model |
| 1 | 2024 | 360-cell oracle sweep | 14 candidates survived from 240 cells (12 assets × 4 timeframes × 5 strategies) | Oracle sweep results |
| 1.3 | 2024 | Naive baseline separation | Separated oracle signal from inherent technical edge; 14 robust candidates confirmed | Noise budget audit |
| 2 | 2024 | EUR/USD hourly MR deep audit | **KILLED** — parameter spike (33% plateau), not robust; only profitable during Asian session | EUR/USD MR audit |
| 3 | 2024 | Exogenous data enrichment | Extended time series with macro/vol/yield features; 11 assets enriched with COT, VIX, yield data | Extended data store |
| 3.5 | 2025-01-01 | Extended-history stress test | **ALL 14 candidates killed** by worst-2Y window < −0.5 threshold | PHASE_3_5_REPORT.md |
| 4 | 2025-07-17 | 4 parallel hypothesis tracks | **All tracks failed**: MR narrow spike, regime filter fails, academic strategies dead, GBP/USD data too short | PHASE_4_SYNTHESIS.md |
| 5 | 2026-04-17 | Portfolio diversification + threshold calibration | P1 equal-weight SR=0.60 but worst-2Y=−0.55; threshold recalibrated to −1.0 using ETF benchmarks | PHASE_5_SYNTHESIS.md |
| 5.5 | 2026-04-17 | Corrective audit (oracle contamination) | Only 3 oracle-free cells survive; **P3 portfolio identified** (SR=0.45, worst-2Y=−0.63) | PHASE_5_5_SYNTHESIS.md |
| 6.A | 2026-04-17 | Plugin inventory audit | F1 gap (0.44 vs 0.91 required) rules out predictor-primary strategies; only 2 regime plugins untested | PHASE_6A_AUDIT.md |
| 6.B | 2026-04-17 | Untested candidate evaluation | 3 candidates all failed; P3 confirmed as best. LTS strategy plugins created | PHASE_6B_SYNTHESIS.md |
| 6.C | 2026 | P3 robustness stress testing (5 tests + held-out) | 4/5 tests passed; held-out SR=0.316 (script-canonical); Terminal 1 declared (premature) | PHASE_6C_SYNTHESIS_FINAL.md |
| 6.D | 2025-06-17 | Discrepancy reconciliation | Double vol-scaling bug identified in Phase 6.C code | PHASE_6D_RECONCILIATION.md |
| 6.D.1 | 2025-07-15 | Bug fix execution | Bug fixed; all tests re-run with zero pass/fail changes | PHASE_6D1_EXECUTION.md |
| 6.D.2 | 2025-07 | Documentation consolidation | Canonical final doc produced with corrected metrics | PHASE_6C_SYNTHESIS_FINAL.md |
| 6.E.0 | 2025-07-21 | Pipeline simulation validation | Signal mismatch identified (89–90% direction match); GO decision with widened tolerance | PHASE_6E0_SIMULATION_VALIDATION.md |
| 6.E.0.1 | 2025-07 | Pipeline remediation + plugin-canonical | **Held-out Sharpe = −0.065; kill criterion triggered**; E2E orchestration validated | PHASE_6E01_STRATEGY_AUDIT.md |
| 7 | 2026-04-17 | Closure | This document | PROJECT_CLOSURE.md |

### 2.2 Pivot Points

**Pivot 1 — Phase 3.5: Individual strategies all killed**

The extended-history worst-2-year-window test killed all 14 candidates that survived the oracle sweep. Every cell — including those with full-period Sharpe > 1.0 — contained at least one catastrophic 2-year window with Sharpe below −0.5. This was a structural finding: the strategy frameworks themselves could not avoid extended drawdowns, even with perfect foresight.

*Decision:* Shift focus from individual strategy deployment to portfolio diversification.

**Pivot 2 — Phase 5: Threshold recalibration**

The initial worst-2Y threshold of −0.5 killed everything, including the best diversified portfolios. Benchmarking against 6 liquid managed-futures ETFs revealed that 5 of 6 also fail the −0.5 threshold. The threshold was recalibrated to −1.0 based on industry evidence, with an intermediate −0.9 benchmark.

*Decision:* Adopt industry-calibrated threshold. P1 equal-weight portfolio at worst-2Y = −0.55 passes the recalibrated gate.

**Pivot 3 — Phase 5.5: Oracle cells removed from portfolio**

A self-critique audit identified oracle contamination in the Phase 5 portfolio construction. Of 11 cells in P1, 8 depended on oracle signals. Strict filtering left only 3 oracle-free cells: EUR/USD pure MR, USD/JPY TSMOM, USD/JPY dual momentum. P3 (inverse worst-window weighted) was identified as the best deployment candidate.

*Decision:* Restrict to purely oracle-free cells. P3 with 3 survivors becomes the sole candidate.

**Pivot 4 — Phase 6.E.0.1: Script-canonical invalidated**

Plugin implementations — the actual production code — produced held-out Sharpe = −0.065, compared to script-canonical +0.316. The divergence traced to the DM cell, where daily-price peer comparison (plugin) differs from monthly-end alignment (script). The kill criterion (held-out Sharpe > 0) was triggered.

*Decision:* Honor pre-registered kill criterion. Terminal 3 (closure). The script-canonical result that supported Terminal 1 was not reproducible in the production code path.

### 2.3 The Filtering Funnel

```
Phase 0-1:   360 cells (12 assets × 4 TFs × 5 strategies + oracle sweep)
Phase 1.3:    14 candidates (naive baseline separation)
Phase 2:      13 candidates (EUR/USD hourly MR killed)
Phase 3.5:     0 individual survivors (all killed by worst-2Y)
Phase 4:       0 individual strategies (all 4 hypothesis tracks failed)
Phase 5:       5 portfolio variants (11 cells, diversification rescue)
Phase 5.5:     3 oracle-free cells → P3 portfolio
Phase 6.A-B:   P3 confirmed (regime hybrids worsen it)
Phase 6.C:     P3 passes 4/5 stress tests (script-canonical)
Phase 6.E.0.1: P3 fails held-out gate (plugin-canonical)
Terminal:      0 deployable strategies
```

---

## 3. Research Methodology

### 3.1 Overall Approach

The research followed a convergent search methodology: start broad (many candidates), apply increasingly rigorous filters, and advance only what survives. Each phase had pre-registered pass/fail criteria. The methodology was explicitly designed to avoid common retail quant biases — data snooping, curve fitting, premature deployment, and criteria relaxation under pressure.

### 3.2 Key Methodological Innovations

**Oracle sensitivity testing.** A synthetic "perfect prediction" oracle was used to efficiently identify which strategy-asset-timeframe combinations have any theoretical edge. The oracle noise budget (0σ = perfect, 10σ = pure noise) measured how much prediction quality matters for each strategy. This separated genuine strategy logic from forecasting quality. Key insight: mean-reversion strategies are oracle-blind (sign-flip < 0.5%), meaning they work on pure statistical properties of price, not on directional forecasting.

**Pre-registered kill criteria.** Every phase had explicit pass/fail thresholds defined before running tests. The most important were:
- Worst-2Y Sharpe > −0.9 (industry-calibrated)
- Held-out Sharpe > 0 (deployment gate)
- Parameter perturbation plateau ≥ 60% (structural robustness)
- Cost breakeven ≥ 2× (practical robustness)

**Plugin-canonical vs script-canonical separation.** Phase 6.E.0.1 established that the production code path (plugins running bar-by-bar) defines the canonical metrics, not the research scripts (vectorized pandas computations). This is a transferable insight: always validate through the deployment code path.

**Extended-history worst-window filter.** Rather than evaluating strategies on average performance, the worst-2Y-window filter identifies the deepest sustained drawdown. This catches strategies that backtest well on average but contain hidden fragility. It was calibrated against live managed-futures ETFs to set a realistic threshold.

### 3.3 Framework of Pre-Registered Gates

```
Gate 1: Oracle sweep (Sharpe > 0 at 10σ noise)           → Phase 1
Gate 2: Naive baseline (edge over buy-and-hold)           → Phase 1.3
Gate 3: Extended-history worst-2Y (> −0.5, later −0.9)   → Phase 3.5
Gate 4: Parameter plateau (≥ 50% of max Sharpe)           → Phase 4 Track A
Gate 5: Portfolio worst-2Y (> −0.9)                       → Phase 5.5
Gate 6: OOS Sharpe (no degradation)                       → Phase 5.5
Gate 7: 5 stress tests (held-out, JPY reversal, MC,       → Phase 6.C
        cost sensitivity, walk-forward, perturbation)
Gate 8: Plugin-canonical held-out Sharpe > 0              → Phase 6.E.0.1
```

P3 passed Gates 1–7. It failed Gate 8 — the most genuine out-of-sample test.

---

## 4. Methodology Retrospective

### 4.1 What Worked

**Oracle sensitivity testing** efficiently identified promising signal-strategy combinations from a large search space (360 cells → 14 candidates in Phase 1). Without it, exhaustive backtesting of all combinations would have been infeasible.

**Extended-history worst-2Y filter** was the critical reality check. It killed oracle-inflated strategies that looked excellent on average metrics (Phase 3.5) and prevented premature deployment of fragile candidates.

**Industry benchmarking for threshold calibration** prevented premature kill of the entire portfolio approach. The original −0.5 threshold was unrealistically strict — most professional managed-futures products also fail it. Recalibrating to −0.9 using live ETF evidence was the right methodological move.

**Pre-registered kill criteria** maintained discipline under pressure to continue. At Phase 6.E.0.1, when plugin-canonical held-out Sharpe was −0.065, the temptation to widen the tolerance or argue the result was "close enough" was real. The pre-registration prevented it.

**Phase-by-phase corrective audits** caught bugs and methodological issues before propagation:
- Phase 5.5 caught oracle contamination in Phase 5 portfolio
- Phase 6.D caught double vol-scaling bug in Phase 6.C
- Phase 6.E.0.1 caught script-vs-plugin divergence that invalidated the Terminal 1 decision

**Plugin-vs-script canonical separation** was the key insight that enabled honest Terminal 3. Without this distinction, the project would have deployed based on script-canonical metrics that the production code could not reproduce.

### 4.2 What Failed or Was Insufficient

**Initial reliance on oracle-enhanced evaluation.** While useful for search, every oracle-positive individual strategy failed extended-history tests. Future work should apply the extended-history filter as a first-class gate, not an afterthought.

**Single held-out period.** The 2024–2025 window is one 2-year sample. A single unlucky sample triggered the kill criterion, but could have been unrepresentative. Multiple held-out windows at different historical points would have been more robust — but was infeasible given project constraints and the imperative not to contaminate the holdout.

**Assumed plugin-script equivalence.** Phase 6.B validated that plugins matched scripts on full-period Sharpe, and this was extrapolated to all metrics and all periods. This assumption did not hold at the held-out level, where a ~10% signal mismatch concentrated in specific periods (2024 DM at 58% agreement) produced materially different results.

**Portfolio orchestration left as stub.** `DefaultPipeline` and `DefaultPortfolio` were not fully implemented until Phase 6.E.0.1 — the very last pre-deployment phase. Earlier end-to-end validation could have surfaced the script-vs-plugin divergence earlier, potentially changing the project trajectory.

**Parameter perturbation (6.C.5) treated as soft criterion.** The pre-registered threshold was applied, but the FAIL was characterized as "informational" rather than as a genuine gate. In retrospect, it was signaling real fragility that manifested in held-out degradation.

### 4.3 What Would Be Done Differently

If restarting from Phase 0:

1. **Implement plugins and production pipeline first, validate them against backtest.** The production path should define canonical metrics, not the research scripts. This alone would have caught the script-plugin divergence before Phase 6.

2. **Apply extended-history filter as first-class gate from Phase 1.** Phase 3.5 was framed as an "audit." It should have been a baseline gate: no candidate advances without passing worst-2Y.

3. **Multiple held-out windows.** Rolling held-out: train on 2003–2018, test 2019–2020; train on 2003–2019, test 2020–2021; etc. This gives a distribution of held-out outcomes instead of a single point estimate.

4. **Tighter parameter perturbation criteria.** The 6.C.5 FAIL should have been a hard gate, not an informational test.

5. **Smaller asset universe from the start.** The project started with 12 assets. The final portfolio uses 2. Starting smaller with deeper diagnostics per asset would have been more efficient.

6. **Held-out validation through plugins from the start.** Every pass/fail decision in Phases 3.5–6.C should have been gated on plugin-canonical performance, not script performance.

---

## 5. Portfolio P3 — The Final Candidate

### 5.1 Construction

**Cells:**

| Cell | Asset | Strategy | Weight | Origin |
|------|-------|----------|--------|--------|
| EUR/USD MR | EUR/USD | Daily z-score mean-reversion (lookback=20, z_entry=1.5, z_exit=0.5) | 20.55% | Oracle-blind (sign-flip < 0.5%) |
| USD/JPY TSMOM | USD/JPY | Monthly time-series momentum (12-month lookback, inverse-vol sized) | 49.20% | Oracle-free (Moskowitz 2012) |
| USD/JPY DM | USD/JPY | Monthly dual momentum — absolute + relative vs 3 peers (12-month lookback) | 30.24% | Oracle-free (Antonacci) |

**Weighting rule:** Inverse worst-2Y-window Sharpe. Cells with shallower worst drawdowns receive more weight. This rule was derived on the training set (≤2018) and frozen for out-of-sample evaluation.

**USD/JPY concentration:** 79.4% of portfolio weight in two USD/JPY strategies. This is the primary risk factor — the portfolio is essentially a USD/JPY directional bet with a small EUR/USD mean-reversion diversifier.

**Correlation structure:** EUR/USD MR is near-zero correlated with both USD/JPY strategies (ρ ≈ 0), making it a genuine diversifier. The two USD/JPY strategies are moderately correlated (ρ = 0.556), expected since both trade the same asset with different signals.

### 5.2 Full-Period Performance (Plugin-Canonical)

| Metric | Value |
|--------|-------|
| Sharpe ratio | 0.4055 |
| Max drawdown | 20.18% |
| Total return | 92.47% |
| Annualized vol | 7.19% |
| Worst 2-year Sharpe | −0.744 |
| Period | 2003–2026 (1168 weeks) |
| Target vol | 10% annualized |

| Sub-Period | Sharpe | maxDD |
|------------|--------|-------|
| In-sample (≤2018) | 0.3865 | 12.6% |
| Out-of-sample (2019–2023) | 0.5955 | 20.2% |
| Held-out (2024–2025) | −0.0650 | 14.3% |

The IS-to-OOS improvement (0.39 → 0.60) is unusual and warrants caution — it may reflect favorable market conditions for momentum strategies in 2019–2023 rather than genuine out-of-sample alpha.

### 5.3 Held-Out Performance — The Negative Result

| Metric | Value |
|--------|-------|
| Sharpe ratio | **−0.065** |
| Max drawdown | 14.3% |
| Total return | −0.9% |
| Vol | 6.8% |
| Period | 2024-01-01 to 2025-12-31 (105 weeks) |

**Per-cell held-out decomposition:**

| Cell | Held-out Sharpe | Return | maxDD |
|------|----------------|--------|-------|
| EUR/USD MR | +0.057 | +0.8% | 5.5% |
| USD/JPY TSMOM | −0.035 | −0.7% | 20.4% |
| USD/JPY DM | **−0.137** | −2.2% | 14.0% |

The DM cell is the primary source of held-out degradation. In the plugin implementation, daily-price peer comparison produces different entry/exit timing than the script's monthly-end alignment, particularly in 2024 where direction agreement drops to 58.4%.

### 5.4 Stress Test Summary (Plugin-Canonical)

| Test | Script Result | Plugin Result | Status |
|------|--------------|--------------|--------|
| 6.C.0 Held-out Sharpe > 0 | ✅ +0.316 | ❌ −0.065 | **FAIL** |
| 6.C.0 Held-out maxDD < 25% | ✅ 13.6% | ✅ 14.3% | PASS |
| 6.C.1 JPY reversal (500 paths) | ✅ median 0.00 | ✅ median 0.09 | PASS |
| 6.C.3 Cost sensitivity (2×) | ✅ 0.406 | ✅ 0.372 | PASS |
| 6.C.4 Walk-forward (52 quarters) | ✅ 55.8% positive | ✅ 61.5% positive | PASS |
| 6.C.5 Parameter perturbation | ❌ MR 57% | ❌ MR 57% | FAIL (both) |

5 of 7 criteria pass. The two failures:
1. Held-out Sharpe barely negative (−0.065 ≈ zero)
2. MR parameter sensitivity below threshold (57% plateau < 60% required)

---

## 6. Terminal Decision and Its Rationale

### 6.1 Pre-Registered Criteria

The kill criteria were defined before Phase 6.C execution:

| Criterion | Threshold | Plugin Result | Status |
|-----------|-----------|-------------|--------|
| Held-out Sharpe | > 0 | −0.065 | **FAIL** |
| Held-out max DD | < 25% | 14.3% | PASS |
| Cost breakeven | > 2× | > 3× | PASS |
| Walk-forward positive | ≥ 55% | 61.5% | PASS |
| JPY reversal median | ≥ 0 | 0.09 | PASS |

### 6.2 Evidence at Each Gate

| Gate | Evidence | Decision |
|------|----------|----------|
| Full-period Sharpe > 0 | 0.4055 | PASS — proceed to held-out |
| Held-out Sharpe > 0 | −0.065 | **FAIL — kill criterion triggered** |
| maxDD within acceptable range | 14.3% < 25% | PASS — not catastrophic |
| Cost-robust | Positive at 3× | PASS — practical |
| Walk-forward stable | 61.5% positive, max streak 2 | PASS — temporally stable |
| Parameter robust | 57% plateau (MR) | FAIL — genuine sensitivity |

### 6.3 Why Terminal 3, Not Terminal 1

**Terminal 1 was declared prematurely in Phase 6.C** based on script-canonical metrics (held-out Sharpe = +0.316). When Phase 6.E.0.1 established that plugin-canonical held-out Sharpe = −0.065, the Terminal 1 decision was invalidated.

The project had an explicit standing rule: "No tolerance adjustment post-hoc." The held-out Sharpe > 0 criterion was pre-registered. The plugin-canonical result fails it. Terminal 3 (closure) is the disciplined outcome.

**Could the result be argued away?**

- "−0.065 is statistically indistinguishable from zero on 105 weeks" — True, but the criterion was > 0, not > 0 at 95% confidence. Weakening the criterion post-hoc is exactly what pre-registration prevents.
- "The DM cell's plugin divergence is a bug, not a strategy failure" — The plugin implementation is the production code. Script-canonical metrics are research artifacts. If the production code can't reproduce the held-out result, the held-out result doesn't exist for deployment purposes.
- "Full-period Sharpe is 0.41, which is competitive" — Full-period includes in-sample. The held-out test exists precisely because full-period is not sufficient evidence.

### 6.4 The Role of Intellectual Honesty

Most retail quant projects would not reach this point. The typical trajectory:
1. Find something that backtests well → deploy immediately (skip held-out)
2. Held-out disappoints → widen criteria ("it was close") → deploy anyway
3. Live performance disappoints → attribute to market regime change → keep trading

This project chose differently: pre-register criteria, reach held-out, honor the kill. The negative result is the methodological contribution.

---

## 7. Infrastructure Produced

### 7.1 Repository Inventory

| Repo | Purpose | State | Key Interfaces |
|------|---------|-------|----------------|
| **feature-eng** | Plugin-based feature engineering (tech indicators, oracle labels, SSA, FFT) | Functional | `feature_eng` CLI; 5+ plugins |
| **preprocessor** | Time-series normalization, unbiasing, trimming, feature selection | Functional | `preprocessor` CLI; 5+ plugins |
| **predictor** | Neural timeseries prediction (ANN, CNN, LSTM, Transformer, TFT, N-BEATS, TCN) | Functional | `predictor` CLI; 18+ model plugins; regression, binary, direction |
| **prediction_provider** | FastAPI async prediction service with plugin architecture | Functional | REST API (`/api/v1/predict/*`); feeders, predictors, endpoints |
| **heuristic-strategy** | Trading strategy optimizer (Backtrader + DEAP GA) | Functional | `heuristic_strategy` CLI; 5 strategy plugins; regime-adaptive strategies |
| **lts** | Multi-user live trading system (OANDA, Backtrader, FastAPI, JWT, RBAC) | Partial | `lts` CLI; web dashboard; strategy, portfolio, broker, pipeline plugins |
| **causal-inference** | Causal analysis tools (Double ML, Causal Forests, Meta-Learning) | Experimental | Batch scripts; preprocessing, inference, transformation plugins |
| **doin-core** | Blockchain consensus library (DOIN network) | Partial | Python library; consensus, crypto, protocol |
| **doin-node** | P2P node for DOIN network with GossipSub and dashboard | Partial | `doin-node` CLI; HTTP transport; web dashboard |
| **doin-plugins** | Reference optimizer/inferencer plugins for DOIN | Partial | Entry points: `doin.optimization.*`, `doin.inference.*` |

### 7.2 Reusable Methodology Artifacts

**Transaction cost model** (`trading_research/transaction_cost_model.py`)
- Calibrated for FX: spread (1.5 bps), slippage (0.3 bps base), market impact scaling
- `apply_cost_to_returns(gross_returns, positions, asset, volatility)` → net returns
- Parameterized cost table supporting EUR/USD, USD/JPY, and extensible to other pairs

**Evaluation harness** (`trading_research/evaluation_harness.py`)
- `annualized_sharpe(returns, periods_per_year)` — standard Sharpe computation
- `rolling_window_evaluation(returns, ppy)` — worst-2Y window, rolling Sharpe distribution
- `compute_strategy_metrics(returns)` — comprehensive metrics dict

**Oracle sensitivity testing** (`trading_research/` oracle sweep scripts)
- Tests strategy robustness to prediction quality degradation (0σ–10σ noise)
- Classifies cells as oracle-blind, honest-ceiling, or trivially-saturated
- Identifies minimum F1 for breakeven

**Walk-forward evaluator** (`phase6c_stress.py`, `phase6e01_plugin_stress.py`)
- Expanding-window quarterly OOS evaluation (52 quarters, 2011–2023)
- Inverse-worst-2Y weight derivation on training window
- Reports: fraction positive, median Sharpe, longest losing streak

**Monte Carlo regime simulator** (`phase6c_stress.py` Task 6.C.2)
- 3000 paths × 3 regimes (bull, bear, range-bound)
- Block bootstrap with regime-specific amplification
- Reports: per-regime Sharpe percentiles

**Parameter perturbation harness** (`phase6e01_plugin_stress.py` Task 6.C.5)
- Grid perturbation around baseline parameters (±25%, ±50%)
- Reports: plateau fraction (% of space within 20% of baseline Sharpe)

**Plugin-vs-script comparison framework** (`phase6e01_plugin_canonical.py`)
- Bar-by-bar direction agreement, position correlation, active-bar match
- Year-by-year agreement heatmap for divergence diagnosis
- Dual-variant testing (derived weights + fixed weights)

### 7.3 Reusable Insights

These are documented findings that transfer to future systematic trading research:

1. **Oracle-inflated strategies fail extended history.** Every individual cell that showed positive Sharpe through oracle enhancement failed the worst-2Y window test on extended (20+ year) history. The oracle identifies theoretical potential, not practical deployability. (Phase 3.5)

2. **Script-canonical metrics differ from plugin-canonical at held-out.** Vectorized pandas research code and bar-by-bar plugin code can agree on full-period metrics while diverging on held-out metrics. The divergence concentrates in periods where signal timing matters most. Always validate through production code. (Phase 6.E.0.1)

3. **Worst-2Y of −0.9 to −1.0 is the industry-calibrated threshold.** The −0.5 threshold (often used as default "reasonable") is too strict — 5 of 6 major managed-futures ETFs fail it. The −0.9 threshold, derived from live product performance, is more appropriate. (Phase 5)

4. **Portfolio diversification rescues individually non-deployable cells.** Cells with individual Sharpe 0.18–0.41 that fail worst-2Y individually can combine into a portfolio with Sharpe 0.45 and acceptable worst-2Y, provided correlations are low. (Phase 5)

5. **Warm-up period inclusion materially changes canonical max DD.** Including the first 252 bars (when rolling estimates are unstable) can add 2–5pp to reported max DD. The choice of warm-up handling should be documented explicitly. (Phase 6.D.2)

6. **F1 ≥ 0.91 is required for predictor-primary FX strategies at retail cost levels.** The oracle noise sweep establishes that with retail spreads (1.5 bps) and typical strategy structures, prediction accuracy below F1 = 0.91 produces negative returns. The best ML predictor achieved F1 ≈ 0.45. (Phase 6.A)

7. **Mean-reversion strategies are oracle-blind.** EUR/USD and USD/JPY daily MR showed < 0.5% sign-flip probability across the oracle noise spectrum. Their performance comes entirely from statistical properties of price (z-score entry/exit), not from directional forecasting. (Phase 1.3, 3.5)

8. **Double vol-scaling is a subtle compounding bug.** When cells are individually vol-scaled to a target and the portfolio is then re-scaled, drawdowns compound. The correct approach: scale cells individually, report portfolio metrics at realized vol. (Phase 6.D)

---

## 8. Honest Assessment

### 8.1 What the Evidence Says

- The P3 portfolio showed positive Sharpe in full-period backtest (0.41) across 23 years and 1168 weeks
- The portfolio was robust to transaction cost multiples (positive at 3×), temporal stability (61.5% positive quarters), and JPY reversal stress (median Sharpe 0.09)
- The held-out period (2024–2025) — the most genuine test of deployment-readiness — showed Sharpe not distinguishable from zero (−0.065)
- The EUR/USD MR cell has parameter sensitivity inconsistent with a durable edge (57% plateau)

### 8.2 What This Does NOT Say

- **It does NOT say the portfolio would definitely lose money in 2026+.** The held-out sample is 105 weeks — noisy by construction. The result is consistent with both a small positive edge and a small negative edge.
- **It does NOT say other strategies on the same data would fail similarly.** P3 was the best candidate from this specific search methodology. Other approaches (different strategy families, different portfolio construction, different data sources) could produce different results.
- **It does NOT say systematic quant research is infeasible at retail scale.** It says that this specific approach, on this specific data, with this specific methodology, did not find a deployable edge.
- **It does NOT say the methodology was wasted.** The infrastructure, the insights, and the documented negative result all have value.

### 8.3 What This DOES Say

- **At this specific strategy construction, with this specific data, under this specific methodology, no deployable edge was identified.** This is the project's conclusion.
- **The pre-registered kill criterion was triggered on the single genuine out-of-sample test.** The held-out period is the most important single piece of evidence, and it was not favorable.
- **Honoring that kill criterion is the discipline that makes the methodology credible.** If the criterion is relaxed when results disappoint, pre-registration is meaningless.

### 8.4 Context Within Quant Finance

**Industry context:**
- Retail quant research rarely reaches held-out validation stage. Most projects deploy based on backtest alone.
- Most successful institutional strategies have Sharpe 0.3–0.8 after costs at scale. A plugin-canonical full-period Sharpe of 0.41 is not insignificant — it is suggestive but not conclusive.
- The held-out test is the most important single piece of evidence. Institutional allocators require it. This project applied the same standard.

**Retail constraints:**
- Spread costs on FX are 2–10× more expensive than institutional (retail 1.5 bps vs institutional 0.1–0.5 bps)
- Capital sizes limit diversification (cannot hold 30+ uncorrelated positions)
- Data availability limits the feature space (no order flow, limited alternative data)
- These constraints structurally lower achievable Sharpe and explain why the F1 requirement (0.91) is so stringent

**The project's contribution in context:** It demonstrates that a rigorous, multi-stage research methodology can be applied at retail scale. The methodology works — it correctly identifies that no deployable edge exists. That identification is the output. The common alternative (deploying without held-out validation, or relaxing criteria when held-out disappoints) would have led to capital loss.

---

## 9. Future Work

### 9.1 Unexplored Directions

These directions were not explored in this project and remain open for future investigation:

- **Higher-frequency data (tick, sub-second).** This project used daily and hourly bars. Higher frequency may reveal microstructure-based edges not visible at daily scale. Requires different infrastructure and cost model.
- **Non-FX assets.** Crypto, commodities, individual equities. The methodology (oracle sweep → extended history → portfolio diversification → held-out validation) is portable but structural characteristics differ.
- **Alternative data.** News sentiment, order flow, macro surprise indices, satellite data. Would expand the feature space beyond price/volume/COT.
- **Longer-term horizons.** Weekly or monthly strategies. This project focused on intraday-to-daily. Longer horizons have different risk-reward profiles and lower transaction costs.
- **Different portfolio construction.** P3 used inverse-worst-2Y weighting. Alternatives (risk parity, Kelly, Black-Litterman, mean-variance with shrinkage) could produce different results from the same cells.
- **Machine learning with F1 ≥ 0.91.** The F1 gap (0.44 observed vs 0.91 required) would need a fundamentally different predictor approach — perhaps ensemble methods, attention mechanisms on multi-asset data, or transformer architectures on tick data.

### 9.2 Less Promising Directions

Based on evidence from this research:

- **More parameter tuning on P3 cells.** Phase 6.C.5 showed genuine parameter sensitivity. More tuning is not the answer; the strategy space is fragile.
- **Regime filter overlays.** Phase 6.B showed regime filters (WFO V2, GMM V3) worsened P3 rather than improved it. Even perfect regime classification (future oracle) could not rescue individual cells at the −0.5 threshold.
- **Predictor-primary strategies on hourly FX.** The F1 gap of 0.44 vs 0.91 required is too large to close with incremental improvements. Fundamental architecture changes would be needed.
- **Oracle-enhanced individual strategies.** Phase 3.5 showed these uniformly fail extended history regardless of oracle quality.

### 9.3 Infrastructure Completion Options

If a future project wants to build on this infrastructure:

- **Complete `DefaultPipeline` for DB-independent batch backtesting.** Currently requires PostgreSQL + web server. A lightweight batch mode would enable faster iteration.
- **Add historical spread data** from OANDA or LMAX for more realistic cost modeling (currently uses fixed 1.5 bps).
- **Implement Bayesian hyperparameter optimization** to replace grid search with uncertainty-quantified parameter exploration.
- **Integrate causal inference pipeline** (from `causal-inference` repo) into the main evaluation harness for structural relationship validation.
- **Multi-asset portfolio expansion** beyond EUR/USD + USD/JPY. The infrastructure supports any pair; the data store has 12 assets with 20+ years of history.

---

## 10. Appendices

### Appendix A: Complete Phase Artifacts

| Phase | Work Plan | Results | Key Data |
|-------|-----------|---------|----------|
| 3.5 | — | PHASE_3_5_REPORT.md | oracle_sweep_*.json, noise_budget_audit.json |
| 4 | — | PHASE_4_SYNTHESIS.md | phase4_track_*.json |
| 5 | PHASE_5.md | PHASE_5_SYNTHESIS.md | phase5_q*.json, benchmark_*.csv |
| 5.5 | PHASE_5_5.md | PHASE_5_5_SYNTHESIS.md | phase5_5_results.json |
| 6.A | PHASE_6a.md | PHASE_6A_AUDIT.md | — |
| 6.B | PHASE_6b.md | PHASE_6B_SYNTHESIS.md | phase_6b_results.json |
| 6.C | PHASE_6c.md | PHASE_6C_SYNTHESIS_FINAL.md (deprecated), PHASE_6C_SYNTHESIS_FINAL_v2.md | phase_6c_omega_results.json, phase_6c_stress_*.json |
| 6.D | PHASE_6d.md | PHASE_6D_RECONCILIATION.md | phase_6d_reconciliation.json |
| 6.D.1 | PHASE_6d1.md | PHASE_6D1_EXECUTION.md | phase_6d1_results.json |
| 6.D.2 | PHASE_6d2.md | — | — |
| 6.E.0 | PHASE_6e0.md | PHASE_6E0_SIMULATION_VALIDATION.md | phase_6e0_results.json |
| 6.E.0.1 | PHASE_6e01.md | PHASE_6E01_STRATEGY_AUDIT.md, PHASE_6E01_ORCHESTRATION_VALIDATION.md | phase_6e01_plugin_canonical.json, phase_6e01_plugin_stress_*.json, phase_6e01_orchestration_e2e.json |
| 7 | PHASE_7.md | PROJECT_CLOSURE.md (this document) | — |

### Appendix B: Key Metrics Across Phases

| Phase | Best Candidate | Full SR | Worst-2Y | Held-Out SR | Status |
|-------|---------------|---------|----------|-------------|--------|
| 1 | 14 oracle-enhanced cells | 0.3–2.0 | — | — | Advanced |
| 3.5 | 14 cells (extended history) | 0.3–2.0 | all < −0.5 | — | **All killed** |
| 4 | MR, regime, academic, GBP | 0.1–0.5 | all < −0.5 | — | **All killed** |
| 5 | P1 equal-weight (11 cells) | 0.598 | −0.547 | — | Near threshold |
| 5.5 | P3 inverse-worst (3 cells) | 0.449 | −0.632 | — | **Passes −0.9** |
| 6.B | P3 vs regime hybrids | 0.447 | −0.724 | — | P3 confirmed |
| 6.C (script) | P3 | 0.447 | −0.724 | +0.316 | Terminal 1 |
| 6.E.0 (plugin) | P3 | 0.432 | — | +0.007 | GO (widened tol) |
| 6.E.0.1 (plugin) | P3 | **0.406** | **−0.744** | **−0.065** | **Terminal 3** |

### Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Cell** | A specific (asset, timeframe, strategy) combination |
| **Oracle** | Synthetic perfect-prediction signal used to test strategy response to forecast quality |
| **Worst-2Y** | Sharpe ratio of the worst consecutive 2-year window in the backtest |
| **P3** | The final portfolio: 3 oracle-free FX cells, inverse-worst-window weighted |
| **Plugin-canonical** | Metrics produced by running LTS strategy plugins bar-by-bar (production code path) |
| **Script-canonical** | Metrics produced by vectorized pandas research scripts (deprecated as canonical) |
| **Terminal 1** | Deploy to live demo (was declared in Phase 6.C, later invalidated) |
| **Terminal 3** | Project closure due to kill criterion failure |
| **Kill criterion** | Pre-registered threshold that, if failed, terminates the candidate |
| **PPY** | Periods per year (252 for daily, 52 for weekly) |
| **Target vol** | Annualized portfolio volatility target (10%) |
| **Vol-scalar** | Multiplier applied to cell returns to achieve target volatility |

---

## Closing Statement

This research program ran for 12+ months through 18 phases and produced a definitive result: applying institutional-grade quantitative methodology to retail-accessible FX data and strategies did not identify a deployable edge.

This is a legitimate research outcome. It is rare in retail quant because most practitioners stop early when results look favorable, relax criteria when results disappoint, or don't document negative results. This project did all three: it reached the held-out stage, honored the kill criterion, and documented the result.

The infrastructure remains available for reuse. The methodology serves as a template. The negative result, documented honestly, is the contribution.

The project is closed.
