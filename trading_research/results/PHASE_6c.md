# Work Plan: Phase 6.C — Robustness Validation of P3 Portfolio

**Prior state:** Phase 6.B complete with Scenario S4 outcome — P3 confirmed as best available candidate. H1 (regime_wfo standalone), H2-refit (regime_adaptive standalone), and H3 (P3+regime filter) all failed kill criteria. The predictor-based alternatives were ruled out by Phase 6.A audit (F1 gap 0.44 vs 0.91 required). The plugin porting work completed successfully with LTS plugins now available for USD/JPY TSMOM and USD/JPY Dual Momentum.

**Purpose:** Subject the P3 portfolio to five rigorous robustness stress tests before final deployment decision. This is the last research phase. After Phase 6.C, the decision is binary: deploy to OANDA demo or close the project.

**Duration:** 10-12 days.

**Decision thresholds:**
- **DEPLOY to OANDA demo:** P3 passes 4 of 5 stress tests (worst-2Y stays > −0.9 in each), AND on the 2024-2025 held-out period achieves Sharpe > 0 and max DD < 25%
- **DEPLOY EUR/USD MR only (Terminal 2):** P3 portfolio fails stress tests but EUR/USD MR cell alone passes individually
- **CLOSE PROJECT (Terminal 3):** P3 fails 3+ stress tests OR held-out period shows material degradation (Sharpe < 0 or max DD > 30%)

---

## Critical Anomaly from Phase 6.B to Address First

Phase 6.B reported H3 trade count anomaly: the regime filter was supposed to reduce trades but instead increased USD/JPY TSMOM trades by 203% and Dual Momentum trades by 1238%. This indicates a bug in H3 implementation, not a genuine strategy result.

While H3 is no longer pursued (it failed kill criteria regardless), the anomaly suggests a potential issue in the plugin integration layer that combines daily price data with 4h-designed regime classification. If this integration is used anywhere in Phase 6.C (it should not be, but must be verified), the results would be unreliable.

**Phase 6.C preflight check:** verify that P3 plugin evaluation in the LTS pipeline does NOT use any regime filter component. The three P3 plugins (eurusd_mr_strategy, usdjpy_tsmom_strategy, usdjpy_dual_momentum_strategy) should operate independently with no regime-related logic. If any cross-contamination exists, fix before Phase 6.C.

---

## Stress Test Overview

Five independent stress tests, each with its own kill criterion. P3 must pass at least 4 of 5 to warrant deployment.

| Test | Hypothesis Tested | Machine | Days |
|------|------------------|---------|------|
| 6.C.1 | JPY reversal (concentration risk) | Gamma | 3-4 |
| 6.C.2 | Monte Carlo regime scenarios | Dragon | 3-4 |
| 6.C.3 | Cost sensitivity | Omega | 2 |
| 6.C.4 | Expanding-window walk-forward | Dragon | 3-4 |
| 6.C.5 | Parameter perturbation sensitivity | Gamma | 2-3 |

Additionally:
| Evaluation | Purpose | Machine | Days |
|------------|---------|---------|------|
| 6.C.0 | 2024-2025 held-out evaluation | Omega | 1-2 |
| 6.C.6 | Synthesis and terminal decision | Omega | 2 |

Parallelization: 6.C.1 and 6.C.5 run sequentially on Gamma. 6.C.2 and 6.C.4 run sequentially on Dragon. 6.C.0 and 6.C.3 run on Omega. Omega finishes early and transitions to synthesis.

---

## Task 6.C.0 — 2024-2025 Held-Out Evaluation (Days 1-2, Omega)

### Purpose
The 2024-2025 period has been preserved untouched across all prior phases. This is the single genuine out-of-sample test available. It runs first because its result can short-circuit later tests: if P3 already fails on held-out, extensive stress testing is wasted effort.

### Task 6.C.0.1 — Run P3 portfolio on 2024-2025

**Implementation:**
- Use the three LTS plugins (eurusd_mr_strategy, usdjpy_tsmom_strategy, usdjpy_dual_momentum_strategy) ported in Phase 6.B
- Run through backtrader_simulation_broker over 2024-01-01 to 2025-12-31
- Apply P3 weights (20.6% / 49.2% / 30.2%)
- Use standardized cost model

### Task 6.C.0.2 — Compute held-out metrics

- Full-period Sharpe
- Max drawdown within the 2-year window
- Per-cell contribution to portfolio return
- Number of trades per cell
- Realized USD/JPY concentration
- Comparison to Phase 6.B IS and OOS (2019-2023) results

### Gate to subsequent tasks

**If held-out Sharpe < 0 OR max DD > 30%:** Phase 6.C effectively concludes here. Proceed directly to 6.C.6 synthesis with "held-out failure" as the primary finding. Terminal 3 (close project) becomes the likely outcome, though remaining tests should still be run for complete documentation.

**If held-out Sharpe > 0 AND max DD < 30%:** continue with stress tests 6.C.1-6.C.5.

### Deliverable 6.C.0
Held-out evaluation report with clear pass/fail determination.

### Important caveat
Even if held-out passes, it is still a single 2-year window. One favorable period does not prove robustness. The stress tests remain necessary.

---

## Task 6.C.1 — JPY Reversal Stress Test (Days 3-6, Gamma)

### Purpose
P3 has 71-80% effective USD/JPY exposure. The largest unresolved risk is a persistent JPY strengthening regime (BoJ normalization post-2024). This test simulates that scenario explicitly to see if P3 survives.

### Task 6.C.1.1 — Identify historical JPY reversal episodes

Extract from 2003-2023 data:
- 1998-like carry unwind episodes (multi-month JPY strengthening of 15%+)
- 2007-2008 risk-off USD/JPY decline
- 2011 post-earthquake JPY surge
- 2016 post-Brexit JPY spike
- 2020 COVID initial risk-off JPY strength

For each episode, extract: duration, magnitude, volatility characteristics, concurrent EUR/USD behavior.

### Task 6.C.1.2 — Generate synthetic price paths with amplified JPY reversals

Build 500 synthetic 2-year paths for USD/JPY using block bootstrap:
- Base: resample 2-year blocks from historical 2003-2023 data
- Perturbation: replace 20-30% of sampled blocks with JPY-reversal episodes (from 6.C.1.1), optionally amplified by 1.5x or 2x
- EUR/USD paths: resample conditionally to preserve correlation structure with synthetic USD/JPY
- Preserve cross-asset correlation structure using Cholesky decomposition of historical correlation matrix

### Task 6.C.1.3 — Run P3 through synthetic paths

For each of the 500 synthetic paths:
- Run P3 portfolio through LTS simulation
- Record: Sharpe, worst-2Y (the full path is 2Y, so worst-2Y = path Sharpe), max DD
- Aggregate distribution across 500 paths

### Task 6.C.1.4 — Pass/fail determination

**Kill criterion:** median synthetic Sharpe < 0 OR 25th percentile synthetic worst-2Y < −1.5

**Pass:** median Sharpe ≥ 0 AND 25th percentile worst-2Y ≥ −1.2

If pass: P3 has enough resilience to JPY reversals in its historical-block-space. If fail: concentration risk is materially severe and deployment is not justified.

### Deliverable 6.C.1
Distribution of P3 performance across 500 synthetic stress paths. Pass/fail determination with specific percentiles.

### Machine assignment
Gamma (RTX 5070 Ti, 12GB). The computation is embarrassingly parallel across paths. Batch of 500 paths can leverage GPU for matrix operations in Cholesky decomposition and correlated noise generation.

---

## Task 6.C.2 — Monte Carlo Regime Scenarios (Days 3-6, Dragon)

### Purpose
Phase 5.5 found P3 performs well in crisis regimes (GFC +2.05) and poorly in QE regimes (pre-GFC −0.31). A forward-looking regime switch to QE-like conditions could be catastrophic. This test quantifies expected performance under distinct regime assumptions.

### Task 6.C.2.1 — Parameterize three regime models

For each of three historical regimes, fit return-generating process to USD/JPY and EUR/USD pairs:
- **Regime A — Inflation regime (2022-2025):** current regime, high vol, JPY weakness
- **Regime B — QE regime (2013-2019):** low vol, range-bound FX, strategies historically failed
- **Regime C — Crisis regime (2008-2012):** high vol, strong trends, strategies historically worked

Model: multivariate GARCH with regime-specific mean returns, volatility clustering, correlation structure. Validate fit by comparing simulated 1000-bar paths to actual regime data statistics (autocorrelation of absolute returns, tail indices, etc.).

### Task 6.C.2.2 — Generate 10,000 forward paths per regime

For each regime, generate 10,000 forward 2-year paths using the fitted model.

### Task 6.C.2.3 — Run P3 through all 30,000 paths

Execute P3 portfolio on each path. Aggregate by regime.

### Task 6.C.2.4 — Report percentile distributions

Per regime, report: 5th, 25th, 50th, 75th, 95th percentiles of:
- Path Sharpe
- Path max drawdown
- Worst single-quarter Sharpe within path

### Kill criterion

**Fail:** Regime B (QE-like) median Sharpe < 0 OR 25th percentile max DD > 30%

**Pass:** All three regimes show median Sharpe > 0 AND 25th percentile max DD < 25%

The QE regime is the crucial test. If P3 has materially negative expected value under QE conditions, deploying now is a bet against regime switch — unjustified without explicit macro view.

### Deliverable 6.C.2
Regime-conditional performance distributions with explicit attention to Regime B.

### Machine assignment
Dragon (RTX 4090, 16GB). 30,000 Monte Carlo paths benefit from GPU acceleration for matrix operations in the GARCH simulation. Dragon's larger VRAM helps with batch processing.

---

## Task 6.C.3 — Cost Sensitivity (Days 3-4, Omega)

### Purpose
The standardized cost model used throughout Phase 3.5-5.5 assumes specific spreads and slippage. Real OANDA spreads during low-liquidity hours (Asian session, holiday periods) can be 2-3x modeled. This test quantifies how much cost buffer P3 has.

### Task 6.C.3.1 — Re-run P3 at multiple cost levels

Using full 2003-2023 history:
- 1.0x baseline (Phase 5.5 assumption)
- 1.25x, 1.5x, 1.75x, 2.0x, 2.5x, 3.0x
- Also test 1.0x daytime + 3.0x during Asian session (more realistic)

### Task 6.C.3.2 — Plot degradation curves

- Sharpe vs cost multiplier
- worst-2Y vs cost multiplier
- max DD vs cost multiplier
- Identify the cost multiplier at which each metric crosses a critical threshold

### Task 6.C.3.3 — Calibrate against actual OANDA spread history

If possible, pull OANDA historical spread data (publicly available for recent periods) and determine the typical actual multiplier. Compare to where P3 breaks even.

### Kill criterion

**Fail:** Sharpe crosses zero before cost multiplier 2.0x OR worst-2Y crosses −1.0 before multiplier 1.5x

**Pass:** Sharpe remains positive at 2.0x AND worst-2Y better than −1.0 at 1.5x

### Deliverable 6.C.3
Cost sensitivity curves and break-even cost multiplier for each metric.

### Machine assignment
Omega. This is sequential computation (full history per cost level, 7-9 levels total) but computationally light. Fits well with Omega's role as the orchestration machine while heavy compute runs elsewhere.

---

## Task 6.C.4 — Expanding-Window Walk-Forward (Days 7-10, Dragon)

### Purpose
Phase 5.5 used a single IS/OOS split (2003-2018 train, 2019-2023 test). This produced a single OOS number. Real robustness requires many OOS evaluations. Walk-forward with quarterly re-evaluation produces ~60 OOS quarters instead of one.

### Task 6.C.4.1 — Walk-forward setup

Expanding-window schedule:
- Initial training: 2003-01 to 2010-12 (8 years minimum)
- Test quarter: 2011-Q1
- Extend training to 2011-Q1, test 2011-Q2
- Continue quarterly through 2023-Q4
- 52 out-of-sample quarters total (2011-2023)
- Re-compute P3 weights at each training extension (using inverse worst-window rule from Phase 5.5)

Note: this re-computes weights, which changes them slightly each quarter as more data enters training. The Phase 5.5 weights (20.6/49.2/30.2) are only the final-period weights.

### Task 6.C.4.2 — Evaluate each OOS quarter

Per quarter, record:
- Portfolio quarterly return and Sharpe (note: 63 days per quarter is statistically thin)
- Max drawdown within quarter
- Cell-level contribution

### Task 6.C.4.3 — Aggregate statistics

- Distribution of quarterly Sharpe values
- Fraction of quarters with positive returns
- Worst quarter
- Longest consecutive losing-quarter streak
- Are bad quarters clustered (consecutive) or distributed?

### Kill criterion

**Fail:** Fewer than 55% of OOS quarters have positive returns OR median quarterly Sharpe < 0.2 OR longest losing streak > 6 consecutive quarters

**Pass:** ≥55% positive quarters AND median quarterly Sharpe ≥ 0.2 AND longest streak ≤ 5 quarters

### Deliverable 6.C.4
Walk-forward performance distribution with explicit analysis of bad-quarter clustering.

### Machine assignment
Dragon. Walk-forward can be partially parallelized across quarters (each quarter's evaluation is independent once its weights are determined). Dragon has the compute to run this efficiently.

---

## Task 6.C.5 — Parameter Perturbation Sensitivity (Days 7-9, Gamma)

### Purpose
The Phase 5.5 P3 strategies have parameters (MR lookback, z-score thresholds, TSMOM lookback months, Dual Momentum lookback). If performance degrades sharply when parameters are perturbed, the Phase 5.5 results were optimized to a narrow spike and will not generalize.

### Task 6.C.5.1 — Define perturbation grid

For each P3 cell, perturb parameters by ±10%, ±25%, ±50% from baseline:

**EUR/USD MR:**
- lookback: 20 (baseline), perturb to {10, 15, 18, 20, 22, 25, 30}
- z_entry: 1.5 (baseline), perturb to {0.75, 1.125, 1.35, 1.5, 1.65, 1.875, 2.25}
- z_exit: 0.5 (baseline), perturb to {0.25, 0.375, 0.45, 0.5, 0.55, 0.625, 0.75}

**USD/JPY TSMOM:**
- lookback months: 12 (baseline), perturb to {6, 9, 11, 12, 13, 15, 18}
- no-trade threshold: 0.02, perturb to {0.01, 0.015, 0.018, 0.02, 0.022, 0.025, 0.03}

**USD/JPY Dual Momentum:**
- lookback months: 12, same grid as TSMOM

### Task 6.C.5.2 — Run grid over full history

For each parameter combination (single cell at a time, not all cells simultaneously — that's combinatorial explosion), compute Sharpe and worst-2Y.

Total runs: ~50 configurations per cell × 3 cells = ~150 backtests. Each backtest is light, so total time is modest.

### Task 6.C.5.3 — Evaluate plateau vs spike

For each cell, plot Sharpe surface across parameter space:
- If surface is smooth with broad plateau: robust parameters, generalization expected
- If surface has narrow spike at baseline: overfit, fragile

Quantitative test: fraction of perturbed configurations within 20% of baseline Sharpe. If >60%, plateau confirmed. If <40%, fragile.

### Kill criterion

**Fail:** Any P3 cell shows narrow-spike pattern (< 40% of perturbed configurations within 20% of baseline)

**Pass:** All three cells show plateau (≥60% of perturbations within 20% of baseline)

### Deliverable 6.C.5
Parameter surfaces for all three P3 cells with plateau-vs-spike analysis.

### Machine assignment
Gamma. Light compute per run, many independent runs. Fits well with Gamma after JPY stress test completes.

---

## Task 6.C.6 — Synthesis and Terminal Decision (Days 10-12, Omega)

### Task 6.C.6.1 — Aggregate all results

Build master table:

| Test | Pass? | Key Metric | Notes |
|------|-------|-----------|-------|
| 6.C.0 (held-out 2024-2025) | ? | Sharpe, max DD | Genuine OOS |
| 6.C.1 (JPY reversal) | ? | Median Sharpe, 25th pct worst-2Y | Concentration risk |
| 6.C.2 (MC regimes) | ? | Regime B median Sharpe | QE regime survival |
| 6.C.3 (cost sensitivity) | ? | Break-even cost multiplier | Production buffer |
| 6.C.4 (walk-forward) | ? | % positive quarters, worst streak | Temporal stability |
| 6.C.5 (parameter) | ? | Plateau vs spike | Overfitting check |

### Task 6.C.6.2 — Terminal decision

**Decision tree:**

1. If held-out (6.C.0) fails: **Terminal 3** regardless of other tests. Document in synthesis.

2. If held-out passes and 4+ of 5 stress tests pass: **Terminal 1 (Deploy to OANDA demo).** Write deployment plan with specific observation period and criteria.

3. If held-out passes and 2-3 stress tests pass: **Evaluate case-by-case.** Consider Terminal 2 (EUR/USD MR only) if failure is concentrated in USD/JPY-related tests (6.C.1, 6.C.2). Consider Terminal 3 if failures are systemic.

4. If held-out passes but 0-1 stress tests pass: **Terminal 3 (Close project).** The portfolio has fundamental robustness issues despite favorable held-out.

### Task 6.C.6.3 — Write synthesis document

`PHASE_6C_SYNTHESIS.md` containing:
- Results from all six evaluations
- Master table with pass/fail per test
- Terminal decision with supporting evidence
- If Terminal 1: specific OANDA demo deployment plan (observation period, monitoring, stop rules)
- If Terminal 2: specific EUR/USD MR only deployment plan
- If Terminal 3: project closure research document structure (Phase 0-6 comprehensive summary)

### Deliverable 6.C.6
Phase 6.C synthesis with terminal state commitment. This is the final analytical document of the project. Subsequent work is either operational (deployment monitoring) or closure documentation, not more research.

---

## Execution Schedule

```
Day 1-2:   6.C.0 held-out evaluation (Omega)
           Preflight check on H3 anomaly (Omega)
           Environment verification (all machines)

Day 3:     6.C.0 gate assessment (Omega)
           If pass: launch 6.C.1 (Gamma), 6.C.2 (Dragon), 6.C.3 (Omega)

Day 3-6:   6.C.1 JPY reversal stress (Gamma)
           6.C.2 Monte Carlo regimes (Dragon)
           6.C.3 cost sensitivity (Omega, completes Day 4)

Day 4-5:   Omega pivots to 6.C.6 synthesis preparation while 6.C.1, 6.C.2 run

Day 7-9:   6.C.5 parameter perturbation (Gamma, after 6.C.1)
           6.C.4 walk-forward (Dragon, after 6.C.2)

Day 10-12: 6.C.6 synthesis (Omega)
           Final terminal decision
```

Parallelization maximizes wall-clock efficiency:
- Gamma: 6.C.1 (Days 3-6) → 6.C.5 (Days 7-9)
- Dragon: 6.C.2 (Days 3-6) → 6.C.4 (Days 7-10)
- Omega: 6.C.0 (Days 1-2) → 6.C.3 (Days 3-4) → 6.C.6 synthesis prep/writing (Days 4-12)

---

## Standing Rules for Phase 6.C

1. **Held-out period is used in 6.C.0 and then consumed.** No further OOS period exists after this phase. All subsequent validation is live trading.

2. **Pre-registered kill criteria are not adjustable.** If results fall near thresholds, the pre-registered thresholds govern. Post-hoc adjustment is prohibited.

3. **Tests run independently.** Results of one test do not justify modifying another test's criteria. The only exception is 6.C.0 acting as early stopping if held-out fails severely.

4. **Synthesis must address ALL tests' results.** Even if terminal decision is driven by one test, the synthesis reports all six evaluations.

5. **If Terminal 1 results from this phase, the deployment plan has a built-in 90-day observation period.** No real capital before 90 days of live demo data. This is non-negotiable.

6. **If Terminal 3, the closure document is comprehensive.** It covers Phases 0-6 with honest narration, including acknowledgment of what was learned despite not finding a deployable edge.

---

## Machine Assignment Summary

| Machine | Primary Tasks | Secondary Tasks |
|---------|--------------|----------------|
| Omega (RTX 4070, 12GB, 4x slower, full conda env) | 6.C.0 held-out, 6.C.3 cost sensitivity, 6.C.6 synthesis | Orchestration, consolidation |
| Gamma (RTX 5070 Ti, 12GB, fast) | 6.C.1 JPY reversal, 6.C.5 parameter perturbation | GPU for block bootstrap Cholesky operations |
| Dragon (RTX 4090, 16GB, fastest) | 6.C.2 MC regime scenarios, 6.C.4 walk-forward | Heavy Monte Carlo with GARCH simulation |

Omega's 4x slower speed is compensated by assigning it the lightest-compute tasks (held-out evaluation is a single 2-year backtest, cost sensitivity is sequential but light, synthesis is writing).

Dragon and Gamma each have two tasks in sequence. Both complete their second task by Day 9-10, allowing Omega 2-3 days of dedicated synthesis time.

---

## Honest Acknowledgments About This Plan

1. **The 500 synthetic paths in 6.C.1 and 30,000 MC paths in 6.C.2 use historical statistics.** They cannot simulate genuinely novel market conditions (true black swans). The stress tests cover the space of historically-observed behavior, not all possible futures.

2. **Parameter perturbation in 6.C.5 tests locally smooth perturbations.** It does not test whether fundamentally different parameters would work better — that's Phase 7 territory and only makes sense if Phase 6.C terminal decision is Terminal 3 with appetite for restart.

3. **Walk-forward in 6.C.4 recomputes weights each quarter.** This is more rigorous than Phase 5.5's single split but also exposes P3 weights to fresh information each quarter. The worst quarters will likely be those right after regime transitions.

4. **I (plan author) am not running the code.** My distribution of tasks and machine assignments are based on stated capabilities. If actual execution reveals mismatches (e.g., Dragon's GARCH simulation runs out of memory, or Gamma's block bootstrap takes longer than estimated), the agent should flag and rebalance rather than forcing adherence to this schedule.

5. **This is the last planning document for this project.** Subsequent documents are 6.C.6 synthesis, deployment operational plan (if Terminal 1), monitoring reports, or closure research document. No Phase 6.D, no Phase 7, without new project scope explicitly defined.

If Phase 6.C terminal decision is Terminal 3 and user later wants to explore alternative directions (different asset universe, different data sources, different strategy paradigms), that would be a new project with its own scoping. The current research program converges at Phase 6.C regardless of outcome.