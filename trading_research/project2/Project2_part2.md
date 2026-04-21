# Project 2 — Part II Execution Plan

**Previous state:** Part I Foundations complete (11 deliverables). One critical blocker remains (G-1: rolling orchestrator). F-6 discovered null result for lagged causal links at 4h — foundational finding that shapes Part II structure.

**Purpose:** Build rolling retraining infrastructure, validate with static baseline replay, execute Path A (adaptive heuristic) experiments, and gate entry to Path B on multi-timeframe causal re-examination (CI-2) before committing compute to prediction-based approaches that may be foundationally compromised.

**Shape:** Five sequential stages, each with its own user-facing gate. Part II does not blindly execute all three paths. It commits to Path A, performs CI-2 interim check, and only then commits to Path B based on updated evidence. Path C (RL) is explicitly deferred to Part III.

---

## 0. Reframing Based on Part I Findings

Two reframings from Part I's discoveries need explicit user acknowledgment before Part II execution begins.

### 0.1 The predictability question

F-6 found no lagged causal structure at 4h. Different paths depend on this differently:

| Path | Depends on predictability? | Fallback if predictability fails |
|------|---------------------------|----------------------------------|
| Path A (Adaptive Heuristic) | NO | Works on regime characterization + adaptive parameters; regimes are states, not predictions |
| Path B (Supervised ML) | YES | If no lagged causal links at any timeframe, supervised ML predicts noise |
| Path C (RL) | MIXED | RL can learn contemporaneous features → policy, but learned policy may not generalize without lagged structure |

**Implication:** Path A is the right starting point. Path B entry should depend on multi-timeframe causal analysis (CI-2) results. Part II is structured accordingly.

### 0.2 The F-4 selection under uncertainty

F-4 selected EUR/USD 4h as primary knowing F-6 had returned null there. Two interpretations:

- **Interpretation X:** EUR/USD 4h is genuinely uncausable; any work at this combination is predestined to null. Reconsider asset/timeframe.
- **Interpretation Y:** 4h doesn't show lagged links but contemporaneous relationships exist; adaptive heuristic (Path A) can still work; other timeframes may show lagged links (CI-2 will test).

Part II adopts Interpretation Y with CI-2 as the test of whether Interpretation X should govern Path B decisions.

---

## 1. Stage Structure

Part II is organized in five sequential stages, each with its own gate. No stage commits to the next without user review.

| Stage | Focus | Primary Machines | Gate to Next Stage |
|-------|-------|------------------|---------------------|
| **II-1** | Infrastructure build (orchestrator, data, windows, embargo, IC analysis) | Omega primary | Sprint 0 items complete and pilot run validated |
| **II-2** | Static baseline replay through orchestrator | Omega | Orchestrator reproduces P1 plugin-canonical results within tolerance |
| **II-3** | Path A execution (adaptive heuristic across 7 configurations) | Omega + Gamma parallel | Kill criteria evaluated per F-10; best Path A configuration identified |
| **II-4** | CI-2 interim analysis (multi-timeframe causal re-examination) | Dragon | Outcome classified (CI-α / CI-β / CI-γ); Path B go/no-go decision |
| **II-5** | Path B execution (conditional on II-4 outcome) | Dragon + Gamma | Path B results evaluated; Part II synthesis |

Path C (RL) is **not** in Part II. Deferred to Part III pending:
- Part II evidence from Path A and Path B
- gym-fx gymnasium migration (per F-9)
- Orchestration modernization

Including Path C in Part II would force commitment to the most expensive path before cheaper paths produce evidence.

---

## 2. Stage II-1: Infrastructure Build

### 2.1 Rolling Orchestrator (G-1 — the critical blocker)

**Build:** `trading_research/project2/part_II/infrastructure/rolling_orchestrator.py`

**Required capabilities:**

1. Load window manifest (JSON per F-5 §5.3)
2. For each window: slice raw data, invoke preprocessor with `--fit_on train_only`, invoke feature-eng, invoke model training, capture metrics, log per F-5 §7 CSV format
3. Manage per-window state isolation (no cross-contamination)
4. Support configurable embargo between train and validation per F-2 §3.1
5. Handle per-window failures gracefully (log, skip, continue — not silent pass)
6. Aggregate cross-window metrics at end of run
7. Support rolling GMM re-fit (UL-1) as optional preprocessing step per window
8. Support change-point triggered windows (UL-2) as alternative to fixed schedule
9. Log exact data window, feature set version, and model state hash per window for full reproducibility
10. Separate "training completed" from "model deployed for next window" — support rollback if retrained model fails validation

**Design constraints informed by Part I findings:**

- Must be plugin-compatible with heuristic-strategy AND predictor (Path A uses the former, Path B the latter)
- Must produce F-5 §7 CSV format so downstream F-10 evaluation works without adaptation
- Must be testable in isolation before dispatching experiments

**Validation (pilot run):**

Before declaring orchestrator complete, run single-window end-to-end test:
- One window: train 2015-2018, val 2019, test 2020
- Heuristic strategy `eurusd_mr` with fixed (not optimized) parameters from P1
- Verify: data slicing correct, no look-ahead, metrics computed, CSV logged, embargo applied correctly
- This is F-5 §10 pilot validation

**Deliverable II-1.1:** `rolling_orchestrator.py` + `ORCHESTRATOR_DESIGN.md` documenting API, design decisions, and pilot validation results

### 2.2 Data Acquisition and Preparation

Build and execute the scripts F-5 §9.1 calls for:

**II-1.2.a** `download_data.py`
- EUR/USD 1h bars 2005-01-01 to 2024-12-31
- Source: HistData bulk (2005-2020) + OANDA v20 API (2020-2024, top-up)
- Data quality checks per F-5 §2.3: missing bar %, duplicates, price sanity, weekend filter, timezone
- Output: `data/raw/eurusd_1h_2005_2024.csv`

**II-1.2.b** `resample_data.py`
- 1h → 4h (primary experimental timeframe)
- 1h → daily (for CI-2 and secondary analysis)
- Proper OHLCV aggregation: O=first, H=max, L=min, C=last, V=sum
- Output: `data/processed/eurusd_{4h,daily}_2005_2024.csv`

**II-1.2.c** `window_manifest_generator.py`
- Generate anchored expanding windows per F-5 §5.2 Design A
- Parameters: train_min=3yr, val=1yr, test=1yr, step=1yr, start=2005, end=2019
- Held-out period: 2020-2024 (preserved, not in windows)
- Output: `data/windows/window_manifest.json` per F-5 §5.3 schema

**II-1.2.d** Macro/calendar data prep (deferred from Phase 2 features per F-4 §6.3):
- FRED macro download via `fredapi`: US 10Y yield, DXY, VIX, CPI, unemployment
- CFTC EUR positioning via `cot_reports` package
- Alignment to target timeframes with forward-fill (no look-ahead)
- Output: `data/raw/macro_fred_monthly.csv`, `data/raw/cftc_eur_weekly.csv`
- **Note:** Not blocking Stage II-3 Path A initial runs (Phase 0 features first). Needed for Stage II-4 CI-3.

### 2.3 Feature Engineering Validation

**II-1.3.a** Fix G-4 (feature-eng entry_points `.get()` → `.select()` per F-8 small gap). 30 minute fix.

**II-1.3.b** Run feature-eng with existing 12 technical features on full 2005-2024 EUR/USD 4h data. Verify:
- Zero NaN after warmup period (~60 bars)
- Feature distributions reasonable (no outliers suggesting data errors)
- Output format compatible with rolling orchestrator input

**II-1.3.c** IC analysis (F-2 §3.3 MEDIUM priority addition)
- Compute Information Coefficient per feature vs forward returns at horizons (1, 6, 24 bars)
- Compute rolling 1-year IC mean, std, IR (IC/std) per feature
- Output: `ic_analysis_phase0.csv` + interpretation document

**This is diagnostic only — does not gate model training.** But provides baseline expectation of feature predictive power before committing compute. If IC analysis shows surprisingly high IC contradicting F-6, escalate for re-examination.

### 2.4 Embargo Implementation

Per F-2 §3.1:

```
Train:     [window_start, train_end]
EMBARGO:   [train_end, train_end + embargo_bars]   # exclude both from val and test
Validation:[train_end + embargo_bars, val_end]
```

Default `embargo_bars = 6` (matches 6-bar forward-return horizon per F-10). Configurable per experiment.

Add to `window_manifest_generator.py`. Verify embargo respected by orchestrator in pilot run.

### 2.5 Deliverable II-1

`STAGE_II-1_INFRASTRUCTURE_REPORT.md`:

- All scripts built and tested, with file paths
- Pilot single-window run results (metrics, logs, CSV output)
- Data quality summary (gaps, duplicates, outliers handled)
- IC analysis findings — flag any feature with IC contradicting F-6
- Embargo verified in place
- Go/no-go recommendation for Stage II-2

### 2.6 Gate to Stage II-2

User reviews II-1 report and confirms:

- Rolling orchestrator functional (pilot passed)
- All data downloaded without gaps
- Window manifest generated correctly
- Embargo implemented
- IC analysis complete

If orchestrator has fundamental design issues uncovered during pilot, escalate via `ESCALATION_II-1.md` and revise before Stage II-2.

---

## 3. Stage II-2: Static Baseline Replay

### 3.1 Purpose

Validate that the Part II rolling orchestrator, configured with fixed P1 parameters, reproduces Project 1 plugin-canonical results within tolerance. This is **infrastructure validation**, not research. If it doesn't match, the orchestrator has a bug and must be fixed before Stage II-3.

### 3.2 Procedure

Configure orchestrator with fixed `eurusd_mr` parameters from P1 final spec (the plugin-canonical values documented in F-1). Run through all in-sample windows 2005-2019 with no optimization.

Compare output to Project 1 plugin-canonical results documented in F-1:

**Success criteria:**

- Sharpe within ±0.05 of P1 plugin-canonical
- Max DD within ±3 percentage points
- Trade count within ±10%

If deviation exceeds tolerance: orchestrator has a bug. Fix before Stage II-3.

### 3.3 Baseline benchmarks

For F-10 §3 comparisons in later stages:

- **Buy-and-hold:** long EUR/USD across all windows, compute Sharpe, MDD, return
- **Random entry:** same trade frequency as eurusd_mr, random direction; 100 simulations to compute distribution statistics
- **Zero (flat):** verifies zero cost baseline

### 3.4 Deliverable II-2

`STAGE_II-2_BASELINE_VALIDATION.md`:

- Orchestrator replay vs P1 plugin-canonical results, window-by-window
- Deviation analysis if any deviations found
- Benchmark metrics documented for later comparisons
- Go/no-go for Stage II-3

### 3.5 Gate to Stage II-3

Orchestrator reproduces P1 within tolerance. Benchmarks documented. If reproduction fails, fix and re-run before proceeding.

---

## 4. Stage II-3: Path A — Adaptive Heuristic Execution

### 4.1 Purpose

Test whether adaptive (rolling re-optimization) of heuristic strategy parameters outperforms static. This is the least-risky path given F-6 null findings because Path A does not require predictive features — it requires only detectable regime structure, which exists per GMM K=9.

### 4.2 Configuration matrix

Execute experiments across strategy × optimizer × retraining-trigger combinations:

| Experiment | Strategy Plugin | Optimizer | Retraining Trigger | Priority |
|------------|-----------------|-----------|--------------------|----------|
| A1 | eurusd_mr (P1) | DEAP GA | Fixed yearly | HIGH (baseline) |
| A2 | eurusd_mr (P1) | DEAP GA | Fixed monthly | HIGH |
| A3 | eurusd_mr (P1) | DEAP GA | Change-point trigger (UL-2 Wasserstein) | MEDIUM |
| A4 | regime_adaptive | DEAP GA | Fixed yearly + rolling GMM re-fit (UL-1) | HIGH |
| A5 | regime_wfo | DEAP GA | Fixed yearly | MEDIUM |
| A6 | eurusd_mr (P1) | NEAT-HPO (System B) | Fixed yearly | MEDIUM (DEAP vs NEAT comparison) |
| A7 | regime_adaptive | DEAP GA | Fixed weekly | LOW (aggressive frequency test) |

**A1-A4 must run.** A5-A7 run if compute and time permit.

### 4.3 Per-experiment procedure

**Per window:**

1. Extract window data via orchestrator
2. If UL-1 rolling GMM: re-fit GMM on train portion, update regime centroids for this window
3. Optimize strategy parameters on train window (DEAP GA or NEAT-HPO per config)
4. Select best parameters on validation window
5. Evaluate on test window
6. Log per F-5 §7 format

**Cross-window:**

- Aggregate test-period metrics
- Compute parameter stability CV per parameter
- Compute retrain improvement (F-10 §2.4)
- Apply all kill criteria K-1 through K-7 from F-10 §4.1

### 4.4 Evaluation against F-10 framework

For each experiment, produce the F-10 §6.1 per-experiment report. Specifically evaluate:

- **K-1 (held-out Sharpe):** evaluated once at end of Stage II-3, per experiment, on 2020-2024 data
- **K-2 (worst 2-year Sharpe):** worst 2-year rolling Sharpe within full OOS
- **K-3 (cost ratio):** gross PnL / total costs, must be ≥ 2.0
- **K-5 (window consistency):** fraction of test windows with positive Sharpe, must be ≥ 60%
- **K-6 (train-test MAE divergence):** N/A for heuristic (no MAE), skip
- **K-7 (adaptive vs static delta):** **critical test of Project 2 hypothesis** — adaptive must beat P1 static baseline by ΔSR > 0 with p < 0.10 bootstrap

Statistical tests per F-10 §5:

- Ledoit-Wolf on adaptive vs static Sharpe difference
- Bootstrap confidence interval on ΔSR (circular block, 10K resamples, block=20)
- Deflated Sharpe Ratio accounting for multiple experiments tested (7 experiments in Path A)

### 4.5 Cross-experiment comparison

Produce F-10 §6.2 cross-path comparison table:

- Rank all 7 Path A variants by held-out Sharpe
- Identify best-performing retraining trigger (yearly vs monthly vs change-point vs weekly)
- Identify best-performing strategy plugin (eurusd_mr vs regime_adaptive vs regime_wfo)
- Identify if NEAT-HPO outperforms DEAP GA (Part VI preview)

### 4.6 Parameter stability diagnostic

For each experiment, track parameter changes between consecutive retrainings:

- Compute CV of each parameter across windows
- If average CV > 0.6 across parameters: flag as potentially curve-fitting
- If CV < 0.4: parameters shifting modestly, consistent with genuine adaptation
- If between 0.4-0.6: ambiguous, document but do not kill

**Note:** CV threshold from F-10 may be too strict given F-6 findings. If most experiments fail only on CV, reconsider threshold in Part II synthesis (not during Stage II-3).

### 4.7 Deliverable II-3

`STAGE_II-3_PATH_A_SYNTHESIS.md`:

- Per-experiment results following F-10 §6.1 template (one section per experiment)
- Cross-experiment comparison following F-10 §6.2 template
- Parameter stability diagnostic per experiment
- Best Path A configuration identified (or "no configuration passed gates")
- Explicit kill criteria evaluation table per experiment
- Verdict per experiment: PASS / FAIL / INCONCLUSIVE

### 4.8 Stage II-3 decision point

Three possible outcomes:

**Outcome PA-α (Path A success):** At least one Path A configuration passes all F-10 kill criteria including K-7. Best configuration documented. Proceed to Stage II-4 for Path B/C decision.

**Outcome PA-β (Path A partial):** No configuration passes all gates, but some show promising patterns (e.g., passes K-1 and K-5 but fails K-7 marginally). Document as "adaptive advantage not demonstrated but not disproven." Proceed to Stage II-4.

**Outcome PA-γ (Path A failure):** All configurations fail multiple gates. Adaptive heuristic does not work on EUR/USD 4h. Escalate to user for Part II-level decision: continue to Path B despite Path A failure, or terminate Part II and reconsider asset/timeframe at CI-2 or beyond.

---

## 5. Stage II-4: CI-2 Interim Analysis

### 5.1 Purpose

Before committing compute to Path B (supervised ML rolling retraining), re-examine the causal null finding from F-6 at additional timeframes. This is the test of whether Path B is worth running at all.

This stage corrects the ordering error in Part I where F-4 was published before CI-2 completed. In Part II, CI-2 gates Path B — no predictive path executes without updated causal evidence.

### 5.2 CI-2 execution

Per F-6 §3.2:

- Run PCMCI+ with RobustParCorr on EUR/USD resampled to daily and weekly
- Same 12 technical features as F-6 (for comparability)
- Search for lagged causal links at lags 1, 2, 3, 5, 10
- Report significance, MCI strengths, edge types

If compute allows, also run at 1h to check if higher-frequency shows different structure.

### 5.3 CI-3 macro feature causal testing

Per F-6 §3.3:

- Add FRED macro features (US-EU rate differential, DXY, VIX) and CFTC positioning to feature set
- Re-run PCMCI+ at daily timeframe (where macro data aligns naturally with forward-fill)
- Test specifically: does US-EU rate differential Granger-cause EUR/USD return?
- Test specifically: do CFTC net positions predict next-week return?

### 5.4 RPCMCI fix attempt (CI-1 from F-6)

Attempt to fix the numpy type bug in `run_rpcmci_only.py`. Time-box this: if fix is not straightforward within reasonable bounds, document the bug and move on. RPCMCI is nice-to-have, not Part II blocker.

### 5.5 Interpretation — three possible outcomes

**Outcome CI-α (strong lagged links at daily/weekly):** CI-2 finds lagged feature → return causal links at daily or weekly timeframe. Supports Path B at those timeframes. Recommend shifting Path B to daily/weekly from 4h.

**Outcome CI-β (weak lagged links with macro):** CI-2 confirms null at technical features alone, but CI-3 finds macro features causally lead returns. Supports Path B at daily with expanded feature set (Phase 2 per F-4 §6.3).

**Outcome CI-γ (null at all timeframes):** CI-2 confirms no lagged structure at any tested timeframe or feature set. Path B is predictably non-viable — predicting noise. Recommend limiting Part II to Path A results and closing Path B before compute investment.

### 5.6 Deliverable II-4

`STAGE_II-4_CI_2_RESULTS.md`:

- Multi-timeframe PCMCI+ results (daily, weekly, optionally 1h)
- Macro feature causal analysis results (if CI-3 run)
- RPCMCI attempt status (fixed / time-boxed / blocked)
- Explicit classification of outcome: CI-α, CI-β, or CI-γ
- Recommendation for Path B scope based on outcome

### 5.7 Stage II-4 decision point

Decision tree:

- **If CI-α:** Proceed to Stage II-5 Path B at the timeframe where lagged links were found. Update F-4 asset/timeframe selection accordingly. Document the shift.
- **If CI-β:** Proceed to Stage II-5 Path B at daily with expanded macro feature set.
- **If CI-γ:** **Halt Path B.** Document outcome. Part II concludes with Path A results only. User decides: accept partial scope and draft Part III, reconsider Part II direction, or propose Part III in different direction (e.g., different asset universe).

This gate exists specifically to prevent the known failure mode of "run expensive predictive models on unpredictable data because the plan said to."

---

## 6. Stage II-5: Path B — Supervised ML Rolling (Conditional)

### 6.1 Entry condition

Only entered if Stage II-4 outcome is CI-α or CI-β. If CI-γ, Path B is skipped and Part II concludes with Stage II-3 results.

### 6.2 Purpose

Test whether supervised ML models with rolling retraining can extract the causal signal identified by CI-2 into profitable strategies. Per F-2 Cap 11 guidance: include linear/tree baselines before committing to neural complexity.

### 6.3 Configuration matrix

At timeframe(s) where lagged links were found per CI-2:

| Experiment | Model | Feature Set | Target | Priority |
|------------|-------|-------------|--------|----------|
| B1 | Ridge regression | Phase 1 (causal-filtered) | Forward return regression | HIGH (linear baseline) |
| B2 | LightGBM | Phase 1 | Forward return regression | HIGH (tree baseline) |
| B3 | TFT (best P1 model) | Phase 1 | Forward return regression | HIGH |
| B4 | CNN (P1 plugin) | Phase 1 | Forward return regression | MEDIUM |
| B5 | LSTM (P1 plugin) | Phase 1 | Forward return regression | MEDIUM |
| B6 | TCN (P1 plugin) | Phase 1 | Forward return regression | MEDIUM |
| B7 | TFT | Phase 2 (Phase 1 + macro) | Forward return regression | HIGH (if CI-β) |
| B8 | LightGBM | Phase 2 | Forward return regression | MEDIUM |
| B9 | Best of B1-B8 | Phase 1 or 2 | Direction classification (binary) | LOW |

**Rationale for linear/tree baselines (B1, B2):** Per Jansen Cap 11 guidance and F-2 §3.2. If simple models achieve within 95% of neural model IC, neural complexity isn't justified.

### 6.4 Per-experiment procedure

Per F-5 §6.2:

1. Window extraction via orchestrator
2. Preprocessor normalization (fit on train, apply to val/test) with embargo
3. Model training on train, validation via early stopping (patience=10)
4. Prediction on test
5. Signal-to-trade via heuristic-strategy `ls_pred_strategy` plugin (or equivalent)
6. Backtest on test with realistic costs
7. Log per F-5 §7 format

### 6.5 Evaluation

Same F-10 framework as Stage II-3. Additionally:

- **K-4 (F1 ≥ 0.91)** applies only to binary classifier (B9)
- **K-6 (train-test MAE divergence)** applies to all regression experiments — divergence > 3× in > 50% of windows triggers kill
- **K-7 (adaptive > static)** — compare rolling to single-fit-all baseline

### 6.6 Model comparison analysis

- Rank models by held-out Sharpe
- Baseline comparison: does neural complexity (B3-B6) justify over linear/tree (B1-B2)?
  - If IC(best neural) < 1.05 × IC(best baseline): neural not justified
- Feature set comparison: does Phase 2 (macro) outperform Phase 1 (technical only)?
- SHAP analysis on best LightGBM model for feature importance (per F-2 §3.4)

### 6.7 Deliverable II-5

`STAGE_II-5_PATH_B_SYNTHESIS.md`:

Same structure as Stage II-3 deliverable but for Path B experiments. Plus baseline-vs-neural comparison and Phase 1-vs-Phase 2 comparison.

### 6.8 Stage II-5 decision point

- **If Path B passes gates:** winning configuration identified. Go/no-go for Part III (which would include Path C RL, multi-asset expansion, synthetic augmentation).
- **If Path B fails gates:** Path B does not add value over Path A. Decide whether to attempt Path C in Part III despite Path B failure, or conclude Project 2 with Path A as best.

---

## 7. Part II Final Synthesis

After Stage II-5 (or Stage II-4 if outcome CI-γ terminates Path B), produce:

`PART_II_FINAL_SYNTHESIS.md`:

- Summary of all stages executed
- Best Path A configuration with held-out metrics
- Best Path B configuration with held-out metrics (if executed)
- CI-2 findings and their strategic implications
- Comparison table: Path A vs Path B vs P1 static vs buy-and-hold
- Core hypothesis verdict: "Adaptive strategies outperform static" — SUPPORTED / NOT SUPPORTED / PARTIALLY
- Recommendation for Part III scope:
  - Both paths strong: Part III = Path C (RL) + multi-asset + synthetic augmentation
  - Only Path A strong: Part III = multi-asset Path A expansion + possibly constrained Path C
  - Neither strong: Part III = fundamental reconsideration (different asset universe, different paradigm) or project closure consideration

---

## 8. Machine Assignment

### Stage II-1 (Infrastructure):
- Omega: orchestrator build, data download, window manifest, pilot validation
- Dragon: idle (pre-position for II-4 CI-2 infrastructure)
- Gamma: idle

### Stage II-2 (Baseline):
- Omega: baseline replay and validation
- Dragon, Gamma: idle

### Stage II-3 (Path A):
- Omega: experiments A1, A2, A3 (serial execution)
- Gamma: experiments A4, A5, A6, A7 (parallel execution)
- Dragon: CI-2 infrastructure prep, RPCMCI bug investigation

### Stage II-4 (CI-2):
- Dragon: PCMCI+ at daily/weekly (compute-heavy)
- Omega: macro feature preparation and alignment
- Gamma: idle or support Dragon

### Stage II-5 (Path B, conditional):
- Dragon: neural model experiments B3-B6 (GPU-heavy)
- Omega: linear/tree baselines B1-B2 (CPU-friendly)
- Gamma: Phase 2 experiments B7-B8, and B9 binary classifier

---

## 9. Standing Rules for Part II

1. **F-10 kill criteria are pre-registered and not modifiable during Part II.** If a criterion fails, the experiment fails. Post-hoc adjustment is not permitted.

2. **Held-out 2020-2024 data touched exactly once per experiment, after all in-sample tuning.** No iteration on held-out. If accidentally used for tuning, disclose and mark contaminated.

3. **Each stage has its own gate.** Part II does not barrel through all stages. Stage-level decisions can pause or redirect.

4. **CI-γ outcome halts Path B.** This is the most important standing rule. If multi-timeframe causal analysis returns null at all tested timeframes with all tested feature sets, the evidence-based decision is to not run ML prediction models. Part II honors this outcome.

5. **Statistical tests are honest.** Deflated Sharpe accounts for all experiments run in the path (not just final chosen one). Multiple-comparison correction within each path per F-10 §5.1.

6. **User decision at each stage gate.** The agent reports stage outcome and recommendation. User makes the call on proceeding, redirecting, or terminating. The plan does not run stages autonomously.

7. **Escalation protocol from Part I preserved.** If an agent encounters blocker, produce `ESCALATION_II-[stage].md` and stop. Do not silently work around issues.

---

## 10. Honest Acknowledgments Specific to Part II

1. **F-6 null finding is heavier evidence than Part I synthesis acknowledged.** Part II treats CI-2 as go/no-go for predictive paths because if F-6's null extends to all timeframes, running Path B is compute waste. This correction was not fully reflected in F-11.

2. **Path A may succeed while still leaving the project short of commercial viability.** A successful Path A likely produces modest Sharpe (0.15-0.35 held-out per P1 precedent). User's stated productization goal (lts + prediction-provider as commercial products) typically requires stronger Sharpe. Partial Path A success may not reach that bar.

3. **The orchestrator is critical path.** If G-1 has design issues, every downstream stage is affected. Stage II-1 validation must be thorough. A bad orchestrator contaminates all experiment data across all stages.

4. **Parameter stability gate (F-10 CV < 0.6) may be too strict.** True adaptation to regime shifts may produce parameter changes that appear unstable by CV measure. If this gate kills experiments that otherwise pass held-out criteria, reconsider threshold in Part II synthesis based on distribution of observed CVs. Do not adjust gates during Stage II-3.

5. **The configuration matrix has 7 Path A experiments; conditional 9 Path B experiments.** 16 experiments total. Deflated Sharpe penalty is material. Even if one experiment shows Sharpe 0.5, DSR may be 0.2 after accounting for 16 tests. Realistic expectation for best adaptive configuration: DSR in 0.1-0.3 range.

6. **My Part I plan had F-4 and F-6 in parallel when they shouldn't have been.** Causal analysis should have preceded asset/timeframe selection, not run alongside. Part II corrects this by placing CI-2 before Path B commitment. In Part III planning (if project continues), cause-before-selection ordering should be standard.

7. **Path C is appropriately deferred.** Initially I planned all three paths for Part II. Part I's discovery of null causal structure + F-9 gym-fx gymnasium migration needed + F-8 orchestration gaps make Path C premature. Part III is the right place for Path C, with Path A/B evidence informing RL design.

---

## 11. Dependency Graph

```
II-1 Infrastructure (Omega)
    ├── Orchestrator build (G-1)
    ├── Data download + resample
    ├── Window manifest
    ├── Feature-eng validation + IC analysis
    ├── Embargo implementation
    └── PILOT RUN validates all above
         │
         ▼
II-2 Baseline Replay (Omega)
    ├── Fixed-parameter P1 replay through orchestrator
    ├── Comparison to P1 plugin-canonical
    └── Buy-and-hold/random/zero benchmarks
         │
         ▼
II-3 Path A Experiments (Omega + Gamma parallel)
    ├── A1, A2, A3 (Omega serial)
    ├── A4, A5, A6, A7 (Gamma parallel)
    ├── F-10 evaluation per experiment
    └── Cross-experiment comparison
         │
         ▼
II-4 CI-2 Analysis (Dragon)
    ├── PCMCI+ at daily + weekly
    ├── Optionally CI-3 macro features
    ├── Optionally RPCMCI fix
    └── Outcome classification: CI-α / CI-β / CI-γ
         │
         ├── CI-α or CI-β ──▶ II-5 Path B (Dragon + Gamma)
         │                         └── B1-B9 experiments
         │                              │
         │                              ▼
         └── CI-γ ────────────────▶ Part II Final Synthesis
                                         (Path A only)
```

---

## 12. Immediate Next Actions

User approves this Part II plan or requests adjustments.

Upon approval:

1. **Agent begins Stage II-1 on Omega.** Parallel: Dragon begins RPCMCI investigation as pre-work for Stage II-4.

2. **When Stage II-1 completes:** notify user for II-1 gate review. User approves or requests fixes.

3. **Each subsequent stage begins only after user approves previous stage gate.**

4. **If any stage produces `ESCALATION_II-[stage].md`:** user reviews before continuation.

Part II concludes with `PART_II_FINAL_SYNTHESIS.md` and recommendation for Part III scope (or Project 2 closure consideration if results warrant).