# Work Plan: Phase 6.A — Strategy × Predictor Combination Audit

**Prior state:** Phase 5.5 produced P3 portfolio (oracle-free strategies: EUR/USD pure MR + USD/JPY TSMOM + USD/JPY Dual Momentum). User's infrastructure includes mature causal inference pipeline (6 methods), trained CNN direction and binary predictors, and multiple heuristic-strategy plugins (regime_adaptive, regime_wfo with V2 causal features, direction_atr, long_short_predictions, api_predictions). It is unclear which combinations of these resources have been formally evaluated in the Phase 3.5-5.5 framework.

**Purpose of this phase:** Before assuming P3 is the best available candidate, systematically map which strategy × predictor combinations have been rigorously tested and which have not. Identify untested combinations that are plausibly competitive with P3 and become candidates for Phase 6.B formal evaluation.

**Duration:** 2 days. This is a pure audit — no new backtests are run, just inventory and classification of existing work.

**Output:** A structured document listing every strategy × predictor combination, its test status, and its priority for Phase 6.B formal testing.

---

## Why This Audit Exists

Phase 3.5-5.5 tested strategy families (momentum, mean reversion, vol regime switch) across multiple assets and timeframes. These tests used generic strategy specifications, not the specific plugin implementations in `heuristic-strategy`. They also did not systematically test the user's own trained predictors (CNN direction, CNN binary) connected to prediction-consuming strategies.

This creates a gap: combinations that are natural given the user's infrastructure may never have been evaluated with Phase 5.5 rigor. Before committing to P3, we must rule out that a better combination exists and was simply never tested.

The audit is deliberately narrow: it does not generate new evidence, it maps existing evidence. This keeps Phase 6.A cheap and fast.

---

## Task A.1 — Inventory Existing Strategy Plugins

**Machine:** Omega (requires access to repo code)

For each strategy plugin in `heuristic-strategy/plugins/`, document:

- Plugin name and file
- What it consumes (prices only, price + features, prediction API, both)
- If it consumes predictions: what prediction type (direction, binary, regression), what cadence (daily, monthly)
- If it uses features: which features, and whether they include the V2 causal features (bb_position, atr_ratio, ema_alignment)
- Documented test history: has this plugin ever been evaluated with Phase 3.5-5.5 methodology?

Expected plugins based on user's description:
- `plugin_api_predictions`
- `plugin_direction_atr`
- `plugin_long_short_predictions`
- `plugin_regime_adaptive`
- `plugin_regime_wfo` (V2 with causal features)

Plus any additional plugins not previously mentioned that the audit discovers.

**Deliverable A.1:** Table of plugins with columns: name, inputs consumed, prediction type if any, features used, test status (tested / partially tested / never tested in Phase 3.5-5.5 framework).

---

## Task A.2 — Inventory Existing Predictor Plugins

**Machine:** Omega

For each predictor plugin in `predictor/predictor_plugins/`, document:

- Plugin name and file
- Architecture type (CNN, TCN, LSTM, Transformer, TFT, NBEATS, etc.)
- Variant (direction / binary / regression)
- Training status: is there a trained model available? When was it last trained? On what data split?
- Performance metrics from its last training run (accuracy, F1, MAE, whichever applies)
- Whether this predictor has ever been connected to a strategy and evaluated in trading terms

This includes CNN direction, CNN binary, TCN, and any others in the repo.

**Deliverable A.2:** Table of predictors with columns: name, architecture, variant, trained yes/no, prediction metrics, ever-used-in-trading-eval yes/no.

---

## Task A.3 — Cross-Reference: Combinations Matrix

**Machine:** Omega

Build a matrix where rows are strategy plugins and columns are predictor options (including "no predictor / oracle-free" as a column). Each cell records:

- Whether this combination has been formally evaluated with the Phase 3.5-5.5 framework (rolling windows, cost model, worst-2Y, regime breakdown)
- If yes: what were the results, and where are they documented
- If no: is this combination compatible (i.e., does it make sense to test)?

Example row structure:

| Strategy Plugin | No Predictor | CNN Direction | CNN Binary | TCN Direction | plugin_regime_wfo as filter |
|----------------|--------------|---------------|------------|---------------|------------------------------|
| pure_mr (port)  | ✓ tested (P5.5) | incompatible | incompatible | incompatible | NOT TESTED |
| tsmom (port)    | ✓ tested (P5.5) | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| dual_momentum (port) | ✓ tested (P5.5) | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| plugin_direction_atr | incompatible | NOT TESTED? | NOT TESTED? | NOT TESTED? | N/A |
| plugin_regime_wfo (standalone) | N/A | N/A | N/A | N/A | ✓ or NOT TESTED |
| plugin_api_predictions | incompatible | NOT TESTED? | NOT TESTED? | NOT TESTED? | NOT TESTED |

Fill in actually what is tested vs not based on audit findings.

**Deliverable A.3:** Complete compatibility matrix with test status per cell.

---

## Task A.4 — Classify Untested Combinations by Priority

**Machine:** Omega

For each combination in Task A.3 that is compatible and NOT tested, assign a priority based on:

**Priority HIGH (test in Phase 6.B):**
- Combinations that directly use the user's trained CNN direction predictor (which per user showed predictive improvement)
- Combinations that use V2 causal features via plugin_regime_wfo
- Combinations that plausibly address P3's weaknesses (especially USD/JPY concentration — e.g., a strategy that trades more diverse pairs)
- Hybrid: P3 base cells + plugin_regime_wfo as meta-filter

**Priority MEDIUM:**
- Combinations using CNN binary or TCN (second-order predictors with less direct evidence of predictive value)
- Combinations using predictors on assets other than the Phase 5.5 focus (EUR/USD, USD/JPY)

**Priority LOW or SKIP:**
- Combinations using architectures that underperformed in predictor validation
- Combinations where data plumbing is incomplete and would require significant new development
- Combinations structurally similar to already-killed Phase 3.5-5.5 cells

**Deliverable A.4:** Ranked list of untested combinations with priority assignment and brief justification for each.

---

## Task A.5 — Sanity Check on P3 Testing

**Machine:** Omega

A narrower audit question: was P3 itself tested with the actual plugins, or only with NumPy function equivalents?

Per earlier conversation, P3 cells exist as loose functions in `phase5_5_corrective_audit.py`, not as heuristic-strategy plugins. This audit confirms:

- Do any of pure_mr, TSMOM, Dual Momentum exist as plugins?
- If yes, were they used in Phase 5.5, or were the script-level functions used?
- If no (most likely), then even P3's "tested" status is based on script-level evaluation, not plugin-level evaluation

This matters because Phase 6.B will need to port these strategies to plugins anyway to test them in the real pipeline. The audit confirms this porting work is required.

**Deliverable A.5:** Clear statement about P3's implementation status (plugin vs script).

---

## Task A.6 — Consolidate Findings into Phase 6.B Input

**Machine:** Omega

Produce a single document `PHASE_6A_AUDIT.md` containing:

1. Inventory of plugins (A.1, A.2)
2. Combinations matrix with test status (A.3)
3. Priority-ranked list of untested combinations (A.4)
4. P3 implementation status (A.5)
5. **Recommendation:** how many untested combinations to include in Phase 6.B (should be 3-6, not 20)
6. **List of plugin porting work required** before Phase 6.B can run

This document becomes the input for writing Phase 6.B work plan with exact scope.

---

## Execution Details

**Total duration:** 2 days.

**Single machine:** Omega. The audit is pure documentation work — reading code, consulting previous phase reports, classifying. No compute needed. Dragon and Gamma are idle during Phase 6.A and should be verified-ready for Phase 6.B (dependencies installed, GPU drivers working, disk space available).

**No backtests run.** If a combination's test status is ambiguous ("was this tested or not?"), the answer is NOT TESTED for purposes of this audit. Phase 6.B will definitively test anything classified as unclear.

**No new code written.** Phase 6.A only reads and classifies what exists. Any code needed (plugin porting, new strategy wiring) is Phase 6.B work.

---

## What Phase 6.A Enables

After Phase 6.A, I (the plan author) will have:

- Exact list of untested combinations that deserve formal evaluation
- Specific plugin porting work identified (not guessed)
- Clear priority ordering so Phase 6.B can be scoped appropriately
- Confirmation of P3's implementation status

With that, Phase 6.B will be tightly scoped:
- Port the specific plugins needed (2-4 plugins, not all 20)
- Run Phase 5.5-equivalent evaluation on specific candidate combinations (3-6 combinations, not all 30+ possible)
- Compare results against P3 directly
- Identify best candidate for robustness stress testing (Phase 6.C)

Phase 6.A's 2 days save probably 5-7 days of misdirected work in Phase 6.B.

---

## Honest Acknowledgments

This audit assumes that existing documentation is accurate and complete. If there are tests that were run but not properly documented, the audit will classify them as NOT TESTED and duplicate work may occur in Phase 6.B. This is acceptable — better to re-test than to rely on undocumented claims.

The audit does not validate the correctness of past tests. If Phase 5.5 made methodological errors that affected strategy rankings, Phase 6.A does not catch those (Phase 5.5 self-critique already covered that ground). Phase 6.A simply maps what was evaluated, not whether evaluations were correct.

If Phase 6.A reveals that most plausibly-good combinations were already tested and killed, then P3 is indeed the best available candidate and Phase 6.B reduces to pure robustness validation of P3 (essentially the Phase 6 plan from before your causal inference information). That would be a valid and valuable outcome.

---

## Deliverable Summary

Single document: `PHASE_6A_AUDIT.md`

Sections required:
- Plugin inventory (strategies + predictors)
- Combinations matrix with test status
- Priority-ranked untested combinations
- P3 implementation status
- Recommendation for Phase 6.B scope
- List of required plugin porting work

Upon completion of Phase 6.A, the audit document is sent to me and I write Phase 6.B with precise scope based on actual findings rather than assumptions.