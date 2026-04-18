# Work Plan: Phase 6.E.0.1 — Pipeline Remediation

**Prior state:** Phase 6.E.0 completed with formal GO decision but three material issues identified on review:
1. **Signal mismatch (~10%):** Plugin direction signals and script direction signals agree only 89-90% of bars. Phase 6.C canonical metrics were derived from scripts; deployment will run plugins. These are de facto two different strategies.
2. **Held-out performance collapsed:** Pipeline-canonical held-out Sharpe is 0.007 (vs script-canonical 0.316). Pipeline return is 0.11% vs script 5.4%. The "PASS" was only achieved by widening tolerance from ±0.05 to ±0.35 post-registration, which violated Phase 6.E.0 standing rules.
3. **Portfolio orchestration untested:** `DefaultPortfolio` and `DefaultPipeline` are stubs. Validation drove plugins directly through historical data, bypassing the DB-driven pipeline that will run in live deployment.

**Purpose:** Resolve all three issues before Phase 6.E.1 live demo. Establish plugin-canonical metrics that match what deployment will actually produce. Validate the full end-to-end pipeline including orchestration layer.

**Duration:** 5-8 days.

**Machines:** Primarily Omega (pipeline compute). Dragon available for re-runs of stress tests if needed. Gamma available for parameter perturbation re-run if needed.

**Hard gate to Phase 6.E.1:** All three issues resolved with documented canonical plugin-based metrics, portfolio orchestration layer functional end-to-end, and honest expectations calibrated to plugin-canonical numbers (which may be materially lower than script-canonical).

---

## Context: Why This Phase Exists

Phase 6.E.0 was supposed to be the last pre-deployment validation. Its design assumed plugins matched scripts within narrow tolerance (±0.05 Sharpe). The actual results showed wider divergence, and rather than treating this as failure, tolerance was relaxed post-hoc to allow formal PASS. This violates the scientific discipline that has served the project well through 6+ prior phases.

The honest interpretation of Phase 6.E.0 results is: the pipeline implementation behaves differently enough from the script implementation that the Phase 6.C canonical metrics (Sharpe 0.447 full-period, 0.316 held-out, max DD 22.8%) are not representative of what deployment will produce. The pipeline may still produce an acceptable deployable strategy, but we don't currently know what its canonical metrics are.

Phase 6.E.0.1 does three things:
1. **Task A:** Eliminate the signal mismatch by choosing one implementation as source of truth
2. **Task B:** Implement and validate the portfolio orchestration layer end-to-end
3. **Task C:** Re-establish canonical metrics using whichever implementation is chosen as source of truth

After this phase, deployment runs a strategy with known canonical behavior, not an approximation.

---

## Task 6.E.0.1.A — Resolve Signal Mismatch

### Decision: Plugin version becomes source of truth

The three options considered in Claude's review:
- **Option 1:** Rewrite plugins to exactly match vectorized script logic (incremental but functionally identical)
- **Option 2:** Declare plugins canonical, re-run Phase 6.C with plugins, use those as canonical
- **Option 2b (new):** Rewrite scripts to match plugins exactly, verify they give same result, then plugins are validated against their now-equivalent scripts

**Chosen: Option 2.** Rationale:
- Plugins are what deployment will run. Deployment metrics should match what deployment actually produces.
- Attempting to make plugins match scripts (Option 1) introduces risk of forced complexity to replicate vectorized idioms in incremental form.
- The ~10% direction mismatch is between valid but different implementations; there is no objective "correct" version. Plugins were carefully designed for the live incremental data-flow; scripts were designed for batch analysis.
- Option 2b (rewriting scripts to match plugins) is possible but adds work without operational benefit since deployment doesn't use scripts.

**Consequence:** Canonical metrics will change. They may be worse than script-canonical. This is acceptable — we want accurate expectations, not flattering ones.

### Task 6.E.0.1.A.1 — Verify plugin implementations are correct (not buggy)

Before making plugins canonical, confirm plugins implement the intended strategy logic correctly, just differently from scripts. The 10% mismatch could be:
- **Path A:** Both plugins and scripts correctly implement the strategy; the 10% divergence is due to legitimate incremental-vs-vectorized timing differences (e.g., when exactly a rolling window contains enough data to produce a signal). In this case, plugins are canonical-worthy.
- **Path B:** Plugins have bugs that produce divergent signals. In this case, plugins need fixing before being declared canonical.

**Investigation:**

For each of the three plugins (EUR/USD MR, USD/JPY TSMOM, USD/JPY Dual Momentum), produce a side-by-side comparison document:

- Spec from Phase 5.5 and Phase 6.C: what the strategy is supposed to do
- Script implementation: what the script actually does
- Plugin implementation: what the plugin actually does
- Annotated diff showing specific logic differences (initialization, warm-up, signal generation, exit conditions, vol-sizing)
- Classification per difference: (a) legitimate incremental vs vectorized adaptation, (b) subtle bug, or (c) intentional design choice

For each difference classified as (b) — subtle bug — fix the plugin and re-run the signal comparison.

After bug fixes, signal direction match should be noticeably higher than 89-90%. If still around 90%, remaining gap is Path A (legitimate).

**Deliverable:** `PHASE_6E01_STRATEGY_AUDIT.md` documenting what each plugin does differently from each script, with classification and any bug fixes applied.

### Task 6.E.0.1.A.2 — Establish plugin-canonical metrics by re-running Phase 6.C

Once plugin logic is confirmed (Task 6.E.0.1.A.1 complete), re-run Phase 6.C tests through plugins to produce plugin-canonical metrics. 

**Full period (Phase 6.C.3 equivalent):**
- Run P3 (20.6% / 49.2% / 30.2%) through pipeline on 2003-2023 full history
- Target vol: 10% annualized
- Cost model: 1.0× baseline
- Produce: Sharpe, Max DD, Worst-2Y, Total Return, Realized Vol
- These become plugin-canonical metrics replacing Phase 6.C canonical

**Held-out (Phase 6.C.0 equivalent):**
- Same config but 2024-01-01 to 2025-12-31
- Produce: Sharpe, Max DD, Return, per-cell contribution
- This becomes plugin-canonical held-out metrics

**Stress tests (6.C.1, 6.C.2, 6.C.4):** 
- Re-run each using plugin implementations instead of script
- Same pre-registered kill criteria
- Pass/fail may change based on plugin-canonical numbers
- Machine assignment: 6.C.1 on Gamma, 6.C.2 on Dragon, 6.C.4 on Dragon (as in Phase 6.C)
- Use **original** pre-registered tolerances and thresholds — do not adjust

**Cost sensitivity (6.C.3):**
- Re-run on Omega
- Same cost multiplier grid

**Parameter perturbation (6.C.5):**
- This was relative, may not need full re-run
- If relative plateau fractions shift significantly when using plugin implementations, re-run; else cite existing result

### Task 6.E.0.1.A.3 — Compare script-canonical vs plugin-canonical

Produce comparison table:

| Metric | Script-Canonical | Plugin-Canonical | Delta | Notes |
|--------|------------------|------------------|-------|-------|
| Full-period Sharpe | 0.447 | ? | ? | |
| Full-period Max DD @10% vol | 22.8% | ? | ? | |
| Held-out Sharpe | 0.316 | ? | ? | |
| Held-out Max DD | 16.2% | ? | ? | |
| Worst-2Y Sharpe | -0.724 | ? | ? | |
| 6.C.1 median Sharpe | 0.000 | ? | ? | |
| 6.C.2 Regime B median | 0.022 | ? | ? | |
| 6.C.3 Sharpe @ 3x cost | 0.366 | ? | ? | |
| 6.C.4 positive quarters | 55.8% | ? | ? | |

**Kill criterion for Phase 6.E.0.1.A:** If any Phase 6.C pre-registered kill criterion fails when re-evaluated with plugin-canonical:
- 6.C.0 held-out Sharpe < 0 → Terminal 3
- 6.C.1 median Sharpe < 0 → investigate, may re-evaluate Terminal state
- 6.C.2 Regime B median Sharpe < 0 → same
- 6.C.4 positive quarters < 55% → same

If any kill criterion fails: escalate to user immediately. Terminal decision may need revision.

If all kill criteria still pass with plugin-canonical: proceed to Task B with plugin-canonical as the deployment reference.

### Deliverable 6.E.0.1.A
- `PHASE_6E01_STRATEGY_AUDIT.md` — documented plugin-vs-script differences
- Plugin-canonical metric table
- Kill criteria re-evaluation
- Go/No-Go for continuing to Task B

---

## Task 6.E.0.1.B — Validate Portfolio Orchestration End-to-End

### Purpose
The Phase 6.E.0 validation drove plugins directly with historical data, bypassing `DefaultPortfolio` and `DefaultPipeline`. Live deployment uses the full orchestration: database stores price data and state, pipeline orchestrator loads data and dispatches to plugins, plugins emit signals, portfolio manager translates signals to broker orders, broker executes. Nothing downstream of direct plugin driving has been end-to-end validated.

### Task 6.E.0.1.B.1 — Assess current state of orchestration layer

Audit the repos involved:
- `lts/portfolio/` — DefaultPortfolio implementation
- `lts/pipeline/` — DefaultPipeline implementation
- Related database schemas and access layer

Document for each:
- Current implementation status (stub / partial / functional)
- Required capabilities for deployment (load price data, dispatch to strategies, aggregate signals, manage weights, submit orders)
- Gaps between current state and deployment-required state

### Task 6.E.0.1.B.2 — Implement minimum deployable orchestration

Based on gap assessment, implement what is needed for deployment (and only that — no expanded scope):

- Database-driven data loading for EUR/USD and USD/JPY daily bars
- Pipeline that on trigger: loads current market state, dispatches to three plugins with their respective data series, collects signals, aggregates to portfolio orders per P3 weights
- Portfolio-level P3 weight management (20.6% / 49.2% / 30.2%) with realized-vol to target-vol leverage calculation
- Order submission to broker (for 6.E.0.1: to `backtrader_simulation_broker`; for 6.E.1 live: to OANDA)
- Logging at each layer

### Task 6.E.0.1.B.3 — End-to-end backtest through full pipeline

Run the complete pipeline (DB → pipeline → plugins → portfolio → broker) on:
- Full history 2003-2023: should match plugin-canonical from Task A
- Held-out 2024-2025: should match plugin-canonical from Task A

**Tolerance:** ±0.02 Sharpe, ±1pp max DD from plugin-canonical (tighter than Phase 6.E.0 because now we're validating orchestration layer, not comparing different implementations).

**If results diverge beyond tolerance from plugin-canonical:** the orchestration layer has bugs. Fix before proceeding.

### Task 6.E.0.1.B.4 — Validate operational behaviors

Through the full pipeline:
- Portfolio-level vol-scaling produces realized 10% target vol on historical data
- Monthly rebalance orders are correctly sized and submitted
- Concurrent cell signals on same day produce correct portfolio-level orders (no conflicts)
- Logging at pipeline level captures per-cell attribution
- Broker fills are correctly propagated back to portfolio state
- Next signal generation uses updated portfolio state

### Task 6.E.0.1.B.5 — Edge cases through full pipeline

Previously tested edge cases only at plugin-direct-drive level:
- DST transition days
- Weekend gaps over SL/TP
- Broker rejection and graceful handling

Re-run through the full pipeline. Orchestration layer must handle these correctly.

### Deliverable 6.E.0.1.B
- `PHASE_6E01_ORCHESTRATION_VALIDATION.md` — full end-to-end pipeline tested
- Orchestration layer functional and matches plugin-canonical results
- Operational behaviors validated
- Handoff-ready for live deployment

---

## Task 6.E.0.1.C — Update Documentation and Deployment Parameters

### Task 6.E.0.1.C.1 — Produce new canonical synthesis

Replace `PHASE_6C_SYNTHESIS_FINAL.md` as the authoritative document with `PHASE_6C_SYNTHESIS_FINAL_v2.md` (or similar), containing:
- Plugin-canonical metrics throughout
- Honest comparison to script-canonical with explanation of why plugins are source of truth
- Updated deployment parameters calibrated to plugin-canonical max DD

Deprecate the v1 (script-canonical) version with clear header marking it historical.

### Task 6.E.0.1.C.2 — Recalibrate deployment parameters

Based on plugin-canonical max DD (which may differ from 22.8%), recalibrate:
- Auto-pause threshold (~1.1× canonical max DD)
- Hard stop threshold (1.5× canonical)
- Warning level (75% of canonical)
- Go/No-go threshold at day 90

If plugin-canonical max DD is higher: thresholds shift up.
If plugin-canonical max DD is lower: thresholds shift down.

### Task 6.E.0.1.C.3 — Update Phase 6.E operational plan

If plugin-canonical Sharpe is materially different from 0.316:
- Recalibrate expected 90-day performance
- Update Go/No-go day-90 criteria
- Update "success looks like" expectations in the operational plan

If plugin-canonical held-out Sharpe is near zero (as Phase 6.E.0 suggested):
- Consider whether Terminal 1 is still the correct decision
- Or whether the 90-day demo should be re-framed as "infrastructure validation with realistic expectation of break-even performance"

### Deliverable 6.E.0.1.C
- Updated canonical synthesis
- Recalibrated deployment parameters
- Updated operational plan
- Clear documentation of what changed and why

---

## Standing Rules for Phase 6.E.0.1

1. **No tolerance adjustment post-hoc.** Pre-registered tolerances from Phase 6.E.0 work plan (±0.05 Sharpe full-period) are re-adopted. If results fail, document failure; do not widen tolerance.

2. **Plugin-canonical is the new source of truth.** Once Task A re-runs Phase 6.C with plugins, those numbers govern. Script-canonical becomes historical.

3. **Kill criteria from Phase 6.C remain in force.** Re-evaluated with plugin-canonical. If any fails, escalate immediately.

4. **No new research.** No new strategies, no parameter tuning, no alternative portfolio weights.

5. **Orchestration layer implementation is scope-constrained.** Only what's needed for deployment. Not a full LTS refactor.

6. **If plugin-canonical suggests Terminal 1 is no longer justified, escalate to user.** Agent does not unilaterally change terminal decision.

7. **This is the last pre-deployment phase.** If Phase 6.E.0.1 reveals further issues requiring remediation, that is escalation point, not automatic Phase 6.E.0.2.

---

## Execution Schedule

```
Day 1-2: Task A.1 — Audit plugin-vs-script differences (Omega)
         Bug fixes if identified
         Re-run signal comparison after any fixes

Day 3-4: Task A.2 — Re-run Phase 6.C tests with plugins
         Day 3 morning: 6.C.3 (cost) and 6.C.0 (held-out) on Omega
         Day 3 afternoon onward: 6.C.1 launch on Gamma, 6.C.2 launch on Dragon, 6.C.4 on Dragon
         Day 4: complete stress tests, parameter perturbation if needed

Day 5:   Task A.3 — Compare script-canonical vs plugin-canonical
         Kill criteria evaluation
         Go/No-Go for continuing to Task B

Day 6-7: Task B — Orchestration layer implementation and validation (Omega)

Day 8:   Task C — Documentation updates and deployment parameter recalibration

```

Realistic range: 5-8 days. Lower end if plugins are bug-free and orchestration layer is simpler than expected. Upper end if plugin fixes require iteration or orchestration layer needs substantial implementation work.

---

## Possible Outcomes

### Outcome A: Plugins valid, held-out collapse was real (~40% probability)
- Plugin implementations have no significant bugs
- Plugin-canonical held-out Sharpe is indeed near zero (0.0-0.1 range)
- Full-period metrics are modestly worse than script-canonical
- This means the strategy's "live-realistic" performance is weaker than Phase 6.C claimed
- User decision: proceed with lower expectations, or re-evaluate Terminal state

### Outcome B: Bugs found in plugins, fix narrows the gap (~30%)
- Task A.1 reveals specific bugs
- After fixes, signal match improves from 90% to 95%+
- Plugin-canonical metrics approach script-canonical
- Held-out Sharpe recovers to 0.2-0.3 range
- Proceed to Task B with updated plugin-canonical

### Outcome C: Plugins valid, held-out collapse was sample noise (~20%)
- Plugins have no bugs
- Held-out collapse was concentrated on specific 2024-2025 events where 10% signal mismatch happened to hurt
- Full-period plugin-canonical is close to script-canonical
- Held-out plugin-canonical remains near zero (bad luck sample)
- User decision: Terminal 1 still reasonable given full-period evidence, but held-out is not encouraging

### Outcome D: Substantial orchestration layer work required (~10%)
- Task B reveals that DefaultPortfolio and DefaultPipeline need meaningful implementation
- Extends Phase 6.E.0.1 by several days
- Separate decision about whether to proceed after implementation or consider this overhead as signal that project is not operationally ready

---

## Honest Acknowledgments

1. **This phase exists because Phase 6.E.0 bent its own rules.** Tolerance was widened post-hoc to permit a PASS that reflected a 97% collapse in held-out return (5.4% → 0.11%). That is not a clean outcome. The remediation is to establish canonical metrics using the implementation that will actually run in deployment, then honestly accept whatever numbers result.

2. **The project may need to accept lower expected performance.** If plugin-canonical Sharpe is 0.30 full-period and 0.00 held-out, that is the real strategy. Deployment is worthwhile only if this is acceptable.

3. **The 90-day demo's value may change.** If plugin-canonical held-out is near zero, the 90-day live demo is primarily infrastructure validation rather than performance validation. Go/No-go criteria should reflect this.

4. **Orchestration layer as stubs is a real operational risk.** Phase 6.E.0 marked this as a "limitation" without treating it as a deployment blocker. It is a deployment blocker. Running live demo with stub orchestration is either (a) running a different system than intended, or (b) requiring the stubs to become functional under live pressure — neither is acceptable.

5. **This is the correction cycle completing.** Phase 6.D found a script-level bug. Phase 6.D.1 fixed it. Phase 6.E.0 revealed a plugin-vs-script mismatch and orchestration stub issue. Phase 6.E.0.1 is the analogous correction. If Phase 6.E.0.1 reveals further undiscovered issues, pattern suggests project complexity has outpaced per-phase validation quality, and more substantial rework may be needed. But the next-most-likely outcome is that Phase 6.E.0.1 cleanly resolves the three issues, and live demo begins with accurate expectations.

---

## Immediate Next Actions

1. Task A.1 begins: side-by-side audit of each plugin vs its script
2. Within 2 days: plugin bug fixes applied if any identified
3. Days 3-4: plugin-canonical Phase 6.C re-runs across three machines
4. Day 5: decision point based on plugin-canonical kill criteria
5. Days 6-8: orchestration layer and documentation

After Phase 6.E.0.1, Phase 6.E.1 live deployment begins with plugin-canonical expectations and validated orchestration.

---

## What Success Looks Like

At end of Phase 6.E.0.1:

**Task A success:** 
- Each plugin audited vs its script
- Any bugs fixed, signal match improved to >95% if possible
- Plugin-canonical Phase 6.C metrics produced
- Kill criteria re-evaluated; if pass, confidence in Terminal 1; if fail, escalation to user

**Task B success:**
- Orchestration layer functional end-to-end
- Full pipeline backtest matches plugin-canonical within tight tolerance
- No stubs remaining that block deployment

**Task C success:**
- New canonical synthesis document reflects plugin-based reality
- Deployment parameters calibrated to plugin-canonical max DD
- Operational plan updated with realistic expectations
- Script-canonical documents preserved as audit trail but marked historical

**Phase gate:** all three tasks pass → Phase 6.E.1 live demo deployment can begin with:
- Known plugin-canonical metrics that match what deployment will produce
- Validated orchestration layer
- Honestly calibrated expectations