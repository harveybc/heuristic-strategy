# Work Plan: Phase 6.B — Evaluation of Untested Candidates

**Prior state:** Phase 6.A audit complete. Found that Phase 3.5-5.5 tested only oracle-free rule-based strategies through the rigorous framework. Predictor-based strategies were tested separately through WFO/GA framework and all produced negative returns over 14 years. Structural F1 gap (0.44 observed vs 0.91 required) makes pure-predictor strategies non-viable. Only 3 untested combinations identified as worth evaluating. P3 strategies exist only as NumPy functions, not as pipeline plugins.

**Purpose:** Formally evaluate the 3 high-priority untested combinations with Phase 5.5 rigor. Determine if any improves over P3 or if P3 remains the best available candidate. Complete the plugin porting work required for eventual deployment regardless of result.

**Duration:** 6-8 days total.

**Decision thresholds (from Phase 5.5 benchmark calibration):**
- Deploy threshold: worst-2Y > −0.9 under all stress tests AND OOS Sharpe remains positive
- Close threshold: worst-2Y < −1.5 under realistic perturbations OR OOS Sharpe goes negative

---

## Critical Corrections to Audit Framing

Before execution, three points from the audit need correction or explicit handling:

### Correction 1: Terminal state from Phase 5.5 is Terminal 2.5, not Terminal 1

The audit's section A.6 states: "In all cases, the terminal decision from Phase 5.5 (TERMINAL 1: Deploy) remains valid."

This is incorrect. The original Phase 5.5 synthesis declared Terminal 1, but its own self-critique identified 11 issues, 4 of which were material. Subsequent review converted this to Terminal 2.5 — staged validation before deployment. Phase 6.B operates under Terminal 2.5 framing: no deployment is automatic, all candidates (P3 included) require the robustness validation that Phase 6 provides before any OANDA demo work begins.

### Correction 2: GMM centroids in plugin_regime_adaptive are in-sample fitted

The audit correctly flagged this in "Honest Acknowledgments" point 3. Phase 6.B must handle this explicitly: the centroids were fitted on 15 years of EUR/USD data that overlaps with any test period we use. This is a structural look-ahead issue that cannot be fully resolved without refitting centroids only on training data.

### Correction 3: plugin_regime_wfo V2 features were causally validated using full dataset

If the causal inference pipeline (PCMCI+, RPCMCI, ICP, DML, TE, DoWhy) ran over the complete available data including what has been used as test periods in prior phases, then the V2 feature selection has seen test data indirectly. This is a less severe issue than Correction 2 (feature selection is a weaker form of peeking than model parameter fitting), but it must be acknowledged.

### Correction 4: Timeframe incompatibility between H3 components

plugin_regime_wfo operates on 4h bars (resampled from 1h). P3 cells operate at daily frequency (pure_mr) and monthly rebalance (tsmom, dual_momentum). H3 (P3 + regime filter) requires a decision about how 4h regime classification filters daily/monthly decisions. Phase 6.B must specify this explicitly before testing.

---

## Task 6.B.1 — Plugin Porting (Days 1-2)

### Purpose
Make P3 cells runnable through the real pipeline. Required regardless of Phase 6.B test outcomes, because any deployment (P3 as-is or P3 modified) needs this porting done.

### Task 6.B.1.1 — Port USD/JPY TSMOM to LTS plugin

**Machine:** Omega (has full conda env)

Create `lts/app/plugins/strategy/usdjpy_tsmom_strategy.py` implementing:
- Signal: sign of trailing 12-month return on USD/JPY
- Position sizing: inverse trailing 60-day volatility
- Rebalance: monthly on first trading day
- No-trade band: if |12m return| < 0.02, flat
- Use existing LTS broker abstraction (backtrader_simulation_broker for backtesting, oanda_broker available for later)

**Validation:** run on 2003-2025 EUR/USD... wait, USD/JPY data. Run on 2003-2025 USD/JPY daily data through backtrader_simulation_broker. Compare Sharpe, worst-2Y, and trade count to Phase 5.5 `run_tsmom()` results. If discrepancy exceeds 0.05 Sharpe or ±3 trades, porting is incorrect and must be fixed before continuing.

### Task 6.B.1.2 — Port USD/JPY Dual Momentum to LTS plugin

**Machine:** Omega

Create `lts/app/plugins/strategy/usdjpy_dual_momentum_strategy.py`:
- Absolute momentum: 12-month return > DGS2 (use FRED data already in feature store)
- For single-asset case, reduces to "long if absolute momentum positive, flat otherwise"
- Monthly rebalance
- Validation same as 6.B.1.1

### Task 6.B.1.3 — Verify existing EUR/USD MR plugin produces consistent results

**Machine:** Omega

`lts/app/plugins/strategy/eurusd_mr_strategy.py` already exists. Run it through backtrader_simulation_broker on full 2003-2025 EUR/USD data. Compare to Phase 5.5 `run_pure_mr()` script-level function. Document any discrepancies.

### Gate to Task 6.B.2
All three P3 cells reproduce script-level Phase 5.5 results within tolerance through the LTS pipeline. If any fails, fix before proceeding. This gate also confirms the pipeline is operationally sound for the new candidates in 6.B.2.

### Deliverable 6.B.1
Three LTS strategy plugins validated against Phase 5.5 script-level results. Discrepancy report if any.

---

## Task 6.B.2 — Evaluate H1: plugin_regime_wfo Standalone (Days 2-4, parallel with 6.B.1 tail)

### Purpose
Evaluate the causally-informed regime strategy as a standalone cell with Phase 5.5 rigor. This is the highest-priority untested combination per the audit.

### Task 6.B.2.1 — Adapt evaluation harness for 4h timeframe

**Machine:** Gamma

plugin_regime_wfo operates on 4h bars. The Phase 5.5 evaluation harness was designed for daily/weekly. Adapt it:
- Rolling 2Y window calculation for 4h data (~4380 bars per 2Y)
- Cost model adjusted for 4h trading frequency (more bars = more potential trades, but regime_wfo only acts on regime transitions so actual trade count is modest)
- Regime breakdown labels adjusted to 4h timestamps

Validate the adapted harness by running it on an oracle-free 4h baseline (simple momentum on 4h EUR/USD) and confirming results are consistent with expected magnitudes.

### Task 6.B.2.2 — Run plugin_regime_wfo through evaluation harness

**Machine:** Gamma

- Execute plugin_regime_wfo on EUR/USD 4h data for full history available
- Record: full-sample Sharpe, rolling 2Y worst Sharpe, max drawdown, regime breakdown per macro period, trades per year, hit rate
- Split into IS (2003-2018) and OOS (2019-2023), with 2024-2025 held out as in Phase 5.5 methodology

**Critical note on look-ahead:** The features used (bb_position, atr_ratio, ema_alignment) were selected via causal analysis that may have seen post-2018 data. Document this as a caveat in the result, but proceed with the evaluation. The threshold-based regime classification logic itself is not fitted on the data — it uses fixed thresholds. This is meaningfully different from H2 (GMM) where centroids are explicitly fitted.

### Task 6.B.2.3 — Compare H1 to P3 candidate criteria

**Machine:** Gamma (compute), Omega (synthesis)

Does H1 meet any of:
- Standalone worst-2Y > −0.9 (passes deployment threshold)
- Standalone worst-2Y better than P3 cells' worst-2Y individually (would be useful as portfolio cell)
- Positive Sharpe that survives cost sensitivity to 2x baseline costs

If yes on any: H1 is a viable new cell candidate for an expanded portfolio.
If no on all: H1 provides no improvement over P3, archive with documented results.

### Deliverable 6.B.2
H1 evaluation report with rolling windows, regime breakdown, cost sensitivity, kill criteria pass/fail for each threshold.

---

## Task 6.B.3 — Evaluate H2: plugin_regime_adaptive Standalone (Days 3-5, parallel with H1 tail)

### Purpose
Evaluate the GMM-based regime strategy. Direct comparison with H1 reveals whether threshold vs GMM classification matters, and whether the in-sample GMM fitting helps or hurts generalization.

### Task 6.B.3.1 — Handle the in-sample centroid issue

**Machine:** Dragon (has the conda env via verification, and compute for potentially refitting)

Two parallel evaluations must be done:

**H2-original:** run plugin_regime_adaptive as-is, with hardcoded centroids. Report results with explicit "in-sample fitted centroids" caveat. This shows what the existing plugin would do if deployed.

**H2-refit:** refit GMM centroids on training period only (2003-2018), evaluate on 2019-2023 OOS. Compare to H2-original. The gap reveals how much the in-sample fitting inflates reported performance.

If H2-refit significantly underperforms H2-original, the original results are artifacts of look-ahead and should not be trusted for deployment decisions. If H2-refit is similar to H2-original, the GMM classification is genuinely robust.

### Task 6.B.3.2 — Run both H2 variants through evaluation harness

**Machine:** Dragon

Same metrics as H1. Cost sensitivity. Regime breakdown.

### Task 6.B.3.3 — Compare H1 vs H2

**Machine:** Dragon (compute), Omega (synthesis)

If H1 and H2 give similar results, the classification method (threshold vs GMM) doesn't matter materially. Pick the simpler (H1, threshold-based) as the preferred candidate.

If H2 significantly outperforms H1, the GMM classification captures something threshold-based misses. But only H2-refit's performance counts for deployment evaluation.

If H1 significantly outperforms H2, the hardcoded thresholds of H1 may themselves reflect implicit in-sample knowledge. Investigate whether V2 thresholds were also derived from full dataset analysis.

### Deliverable 6.B.3
H2 evaluation with both original and refit variants. Comparison to H1. Preferred candidate identified.

---

## Task 6.B.4 — Evaluate H3: P3 + Regime Filter Hybrid (Days 4-6)

### Purpose
Test if adding regime_wfo as a meta-filter over P3 cells improves worst-2Y. This is the only hybrid approach structurally defensible given the F1 gap for predictor-primary strategies.

### Task 6.B.4.1 — Resolve timeframe incompatibility

**Machine:** Omega

plugin_regime_wfo operates on 4h bars. P3 cells operate daily (pure_mr) and monthly (tsmom, dual_momentum).

Design decision required: how does 4h regime classification filter daily/monthly decisions?

Three options, each testable:

**Option A — Snapshot at cell decision time:** when pure_mr evaluates daily (say, 22:00 UTC), take regime classification at that moment. If unfavorable, suppress trade.

**Option B — Majority-in-window:** when tsmom evaluates monthly, take majority regime classification over past month. If more than 50% of 4h bars were in unfavorable regimes, suppress.

**Option C — Daily aggregation:** resample regime classification from 4h to daily (mode, or most-recent-favorable), then apply as filter.

Test Option A for all three P3 cells. If time permits, also test Option C. Option B is third priority.

### Task 6.B.4.2 — Run H3 variants through evaluation harness

**Machine:** Dragon (parallel with H2.3) and Omega (synthesis)

For each variant (H3-A, H3-C):
- Run P3 portfolio with regime filter applied to each cell
- Compare worst-2Y to P3 baseline (−0.632 from Phase 5.5)
- Compare Sharpe to P3 baseline (+0.449)
- Count of trades filtered out per cell
- Cost impact of filtering (fewer trades = lower cost burden)

**Kill criteria for H3 being valuable:**
- Must improve worst-2Y by at least 20% (from −0.632 to better than −0.51)
- Must not reduce Sharpe by more than 20% (from +0.449 to better than +0.359)
- Must not reduce any cell's trade count by more than 70%

If filter kills too many trades, the strategy degenerates to "stand aside most of the time" which may have high Sharpe but is not robust.

### Task 6.B.4.3 — Concentration check

**Machine:** Omega

The 79.5% USD/JPY concentration in P3 was a key risk from Phase 5.5. If H3 filter reduces USD/JPY cells more than EUR/USD MR, it might inadvertently worsen the concentration (now EUR/USD becomes dominant, but total USD/JPY trades reduced). Report realized concentration under H3.

If H3 accidentally reduces concentration (filter kills more USD/JPY trades because USD/JPY was in unfavorable regime more often), this is a secondary benefit worth noting.

### Deliverable 6.B.4
H3 evaluation with variants A and C, comparison to P3 baseline, kill criteria pass/fail, concentration analysis.

---

## Task 6.B.5 — Synthesis and Terminal Decision (Days 6-8)

### Decision Framework

Combining results from Tasks 6.B.2, 6.B.3, 6.B.4:

| Scenario | H1/H2 Standalone Result | H3 Result | Recommended Action |
|----------|-------------------------|-----------|--------------------|
| S1 | H1 or H2-refit passes deployment threshold | Improves P3 worst-2Y | Expanded portfolio: P3 + H1/H2 + regime filter. Proceed to Phase 6.C robustness. |
| S2 | H1 or H2-refit passes deployment threshold | No improvement to P3 | Expanded portfolio: P3 + H1/H2 as new cell. Proceed to Phase 6.C robustness for expanded portfolio. |
| S3 | H1/H2 fail | Improves P3 worst-2Y | Deploy P3 + regime filter hybrid. Proceed to Phase 6.C robustness for hybrid. |
| S4 | H1/H2 fail | No improvement to P3 | Proceed to Phase 6.C robustness for original P3 as per Terminal 2.5 original plan. |

In all scenarios, Phase 6.C (robustness stress tests from prior Phase 6 plan) follows. No deployment is automatic — robustness validation still required.

### Task 6.B.5.1 — Write synthesis document

**Machine:** Omega

Produce `PHASE_6B_SYNTHESIS.md` with:
- Results from all 3 candidates (H1, H2, H3) with honest caveats
- Porting completion status (6.B.1 gate)
- Recommendation: which candidate proceeds to Phase 6.C robustness testing
- Explicit handling of look-ahead issues (H1 feature selection, H2 centroid fitting)
- Explicit acknowledgment of in-sample vs out-of-sample performance for H2-original vs H2-refit

### Deliverable 6.B.5
Synthesis document with terminal candidate selection for Phase 6.C.

---

## Machine Assignment Summary

Given the user's machine specifications:
- **Omega (RTX 4070, 12GB, 4x slower, full conda env):** plugin porting (6.B.1), synthesis (6.B.5), orchestration. Small-to-medium backtests only. NOT in critical path for heavy work.
- **Dragon (RTX 4090, 16GB, fast):** H2 (both variants, GMM refit needs GPU), H3 evaluation. Heaviest computation.
- **Gamma (RTX 5070 Ti, 12GB, fast):** H1 evaluation, harness adaptation. Second heavy workstation.

Parallelization plan:
- Day 1-2: Omega does 6.B.1 porting. Dragon and Gamma verify their environments are ready (install dependencies matching Omega's conda env where needed for the predictor/causal inference code).
- Day 2-4: Gamma runs 6.B.2 (H1). Dragon runs 6.B.3 (H2). Omega continues porting and starts harness adaptation verification.
- Day 4-6: H1 and H2 complete. Dragon picks up 6.B.4 (H3). Gamma available for sensitivity analyses or re-runs if needed.
- Day 6-8: Omega does synthesis. All results consolidated.

---

## Execution Schedule

```
Day 1:    6.B.1 (Omega, plugin porting)
          Environment verification (Dragon, Gamma)
Day 2:    6.B.1 validation (Omega)
          6.B.2.1 harness adaptation (Gamma)
          6.B.3.1 H2 refit setup (Dragon)
Day 3-4:  6.B.2.2-3 H1 evaluation (Gamma)
          6.B.3.2 H2 evaluation both variants (Dragon)
          6.B.1 completion check (Omega)
Day 4-5:  6.B.4.1-2 H3 evaluation (Dragon primary, Omega synthesis)
          6.B.2-3 results compilation (Omega)
Day 6-7:  6.B.5 synthesis (Omega)
          Cross-check with Gamma and Dragon if anomalies found
Day 7-8:  Synthesis refinement, Phase 6.C input document production
```

---

## Standing Rules for Phase 6.B

1. **Plugin-level evaluation is mandatory.** Script-level NumPy functions no longer count. Any candidate that cannot be evaluated at the plugin level is not deployable.

2. **H2-original and H2-refit are reported separately.** Deployment decisions are based on H2-refit only. H2-original is shown for comparison to reveal the look-ahead effect.

3. **Kill criteria are pre-registered.** The thresholds in the decision framework are fixed before execution. Post-hoc threshold adjustment to preserve favorable conclusions is not acceptable.

4. **2024-2025 held-out period is preserved.** Evaluation uses 2003-2018 (train) / 2019-2023 (test). 2024-2025 remains untouched for Phase 6.C or live validation.

5. **Honest caveats must be in the executive summary, not buried.** The look-ahead issues in H1 features and H2 centroids are material and must be prominent in the synthesis, not relegated to appendices.

6. **No new strategies beyond the 3 audit-identified candidates.** If during execution the agent has ideas for additional candidates, flag them for Phase 7 consideration, do not execute in Phase 6.B.

---

## What This Phase Produces

At end of Phase 6.B, one of the following states is established:

**State 1:** New expanded portfolio with H1 or H2 as additional cell. Move to Phase 6.C robustness testing.

**State 2:** P3 + regime filter hybrid confirmed better than P3 alone. Move to Phase 6.C robustness testing for hybrid.

**State 3:** P3 confirmed as best available candidate. Move to Phase 6.C robustness testing for P3 as originally specified.

All three states are legitimate outcomes. The point of Phase 6.B is to ensure that Phase 6.C robustness testing happens on the genuinely best candidate, not on a default assumption.

---

## Honest Acknowledgments About This Plan

1. **The look-ahead issue in H2 centroids may not be fully resolvable** even with refitting, because feature selection (which features to use in the GMM) may also have been informed by full-dataset analysis. H2-refit only addresses the clustering parameters, not feature selection. Treat even H2-refit results with appropriate skepticism.

2. **Phase 6.B assumes the causal inference pipeline results are reliable.** If the PCMCI+/RPCMCI/ICP/DML/TE/DoWhy analyses had methodological issues we have not audited, then plugin_regime_wfo V2 features may not be genuinely causal. This is outside Phase 6.B scope but relevant caveat.

3. **The F1 gap finding (0.44 vs 0.91) is robust evidence** that predictor-primary strategies are non-viable. This plan respects that finding and does not re-test predictor-primary variants. If the gap analysis itself was flawed, this plan would be misdirected, but based on audit documentation the analysis appears sound.

4. **Phase 6.B is the last pre-deployment research phase.** After Phase 6.C (robustness), the decision is binary: deploy to OANDA demo or close. There is no Phase 6.D planning further research. Phase 6.B must be executed with this finality in mind.

5. **I may be wrong about Correction 1 (Terminal 2.5 vs Terminal 1).** If the user's intent was actually Terminal 1 and my prior review over-corrected, this plan is more conservative than needed. The user should confirm which terminal state framing governs Phase 6.B execution before the agent proceeds.