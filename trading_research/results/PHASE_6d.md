# Work Plan: Phase 6.D — Pre-Deployment Discrepancy Clarification

**Prior state:** Phase 6.C complete. Terminal decision formally is Terminal 1 (Deploy P3 to OANDA Demo), with 4/5 stress tests passed and held-out evaluation Sharpe 0.316 / max DD 16.7%. However, independent review of the synthesis identified two numerical discrepancies that must be clarified before deployment commitment.

**Purpose:** Resolve specific numerical inconsistencies between Phase 5.5, Phase 6.B, and Phase 6.C reported metrics. Determine whether discrepancies are explainable (different implementations, cost models, periods) or indicate methodological issues. No new research, only reconciliation.

**Duration:** 1-2 days. This is targeted investigation, not exploration.

**Scope boundary:** Phase 6.D only resolves the two identified discrepancies. It does not re-run Phase 6.C tests, does not introduce new candidates, does not re-evaluate P3 strategies. If the discrepancies reveal fundamental issues, those issues are escalated to the user for a decision about project direction — but investigation itself stays narrow.

---

## The Two Discrepancies Requiring Resolution

### Discrepancy 1: Max Drawdown Magnitude

Reported P3 max drawdown values across phases:

| Source | Max DD Reported | Context |
|--------|-----------------|---------|
| Phase 5.5 synthesis | 15.0% | Full-history P3 portfolio, script-level NumPy |
| Phase 6.B synthesis (6.B.1 validation) | Not explicitly stated | Portfolio-level but focus on per-cell Sharpe |
| Phase 6.C held-out (6.C.0, 2024-2025) | 16.7% | LTS plugin pipeline, held-out period |
| Phase 6.C cost sensitivity (6.C.3 @ 1.0x) | 24.6% | LTS plugin pipeline, full 2003-2023 history |

The concerning comparison is **15.0% (Phase 5.5 full history) vs 24.6% (Phase 6.C full history)**. Both claim to evaluate the same strategy (P3) on the same data (2003-2023) with the same cost assumptions (1.0x baseline). A discrepancy of 9.6 percentage points in max DD requires explanation.

### Discrepancy 2: Implementation Lineage for 6.C.3

Phase 6.B explicitly ported P3 cells to LTS plugins (Task 6.B.1) and validated that plugin-based results matched script-based results within tolerance. Phase 6.C.0 (held-out) used these plugins. But Phase 6.C.3 (cost sensitivity) implementation is not explicitly stated in the synthesis.

If 6.C.3 uses the LTS plugin pipeline: results are consistent with deployed implementation, good.
If 6.C.3 uses script-level NumPy functions: results may not reflect deployment reality.

The synthesis's reported Sharpe at 1.0x cost (0.447) matches both Phase 5.5 (0.449, script) and Phase 6.B plugin-level Sharpe (0.4466) — so this alone doesn't disambiguate which implementation was used. We need explicit confirmation.

---

## Task 6.D.1 — Reproduce Max DD for Full History Through LTS Plugin Pipeline

**Machine:** Omega

### Task 6.D.1.1 — Exact reproduction
- Run P3 portfolio through LTS simulation with the three plugins (eurusd_mr_strategy, usdjpy_tsmom_strategy, usdjpy_dual_momentum_strategy) from Phase 6.B
- Period: full 2003-01-01 to 2023-12-31
- Cost model: 1.0x baseline (same as Phase 5.5 and Phase 6.C.3)
- Weights: P3 (20.6% / 49.2% / 30.2%)
- Measure max drawdown using consistent methodology (peak-to-trough on cumulative equity curve)

### Task 6.D.1.2 — Compare to both reference values
- Does the result match Phase 5.5's 15.0%?
- Does it match Phase 6.C.3's 24.6%?
- If different from both, it's a third value requiring separate explanation

### Task 6.D.1.3 — Identify the source of any discrepancy
If 15.0% ≠ 24.6%, possible causes to investigate:

**Candidate A: Different cost model implementations**
- Phase 5.5 may have used simpler flat spread
- Phase 6.C.3 may have used the full cost model with slippage and volatility scaling
- Verify by running both cost models on the same strategy and data

**Candidate B: Different drawdown calculation methodology**
- Peak-to-trough on equity curve vs peak-to-trough on returns series
- Daily vs weekly calculation frequency
- Currency basis (USD PnL vs percentage of capital)

**Candidate C: Different portfolio aggregation method**
- Script-level forward-filling weekly returns to daily (Phase 5.5)
- Plugin-level native daily execution with weekly-at-monthly rebalance (Phase 6.B/C)
- These produce different return series with different drawdown profiles

**Candidate D: Subtle strategy implementation differences**
- Phase 6.B validation showed plugins reproduce Sharpe within tolerance, but drawdown may be more sensitive to subtle timing differences
- Monthly rebalance day choice (1st trading day vs end of month)
- Vol-sizing lookback window exact bar count

### Task 6.D.1.4 — Produce reconciliation

Present a table clearly documenting:
- The "canonical" P3 max DD value (from LTS plugin pipeline, which is deployment-representative)
- Why other values were reported
- Which value should govern the deployment decision and monitoring parameters

### Deliverable 6.D.1

Report containing:
- The definitive max DD value from plugin-based evaluation
- Root cause of the Phase 5.5 vs Phase 6.C.3 discrepancy
- Implications for deployment (auto-pause threshold, capital sizing, risk warnings)

---

## Task 6.D.2 — Confirm Implementation Used in 6.C.3 and Other Phase 6.C Tests

**Machine:** Omega

### Task 6.D.2.1 — Explicit implementation audit

For each Phase 6.C test, document:
- Which code file generated the results
- Whether it uses LTS plugins or script-level NumPy functions
- Whether it uses the standardized cost model module or an inline cost calculation

Tests to audit:
- 6.C.0 (held-out 2024-2025)
- 6.C.1 (JPY reversal, 500 paths, Gamma)
- 6.C.2 (MC regimes, 9000 paths, Dragon)
- 6.C.3 (cost sensitivity, Omega)
- 6.C.4 (walk-forward, 52 quarters, Omega)
- 6.C.5 (parameter perturbation, Gamma)

### Task 6.D.2.2 — Identify inconsistencies

If some tests use plugins and others use scripts, document which. This is material because:
- Plugin-based results are what deployment will actually produce
- Script-based results may have been close to plugins but not identical
- Stress tests that run on scripts may report different magnitudes than plugins would

### Task 6.D.2.3 — Determine impact on terminal decision

For any test that used scripts instead of plugins:
- Estimate whether re-running with plugins would plausibly flip the pass/fail outcome
- If a "pass" was marginal (e.g., 6.C.1 median Sharpe = 0.000 exact) and plugins might differ, flag as "re-run recommended"
- If a "pass" has large margin, flag as "plugin re-run is nice-to-have but not blocking"

### Deliverable 6.D.2

Implementation audit table showing plugin vs script for each Phase 6.C test, with impact assessment on terminal decision robustness.

---

## Task 6.D.3 — Conditional Re-Run of Affected Tests

**Machines:** Per test origin — Omega, Gamma, or Dragon

### Trigger
Only execute this task if Task 6.D.2 reveals that one or more stress tests used script-level implementation AND had marginal pass/fail outcomes.

### Priority ordering for re-runs
If needed, re-run in this priority order:
1. **6.C.1 (JPY reversal)** — median Sharpe = 0.000 is literally at threshold, highest risk of flip
2. **6.C.2 Regime B (QE)** — median Sharpe = 0.022 is 0.022 above threshold, second highest risk
3. **6.C.4 walk-forward** — 55.8% vs 55% threshold, third highest risk

Skip if margins are large (6.C.3 cost, held-out 6.C.0).

### Kill criterion for Phase 6.D

If a re-run with correct plugin implementation flips any stress test from PASS to FAIL:
- Terminal decision becomes ambiguous
- Escalate to user for decision on whether to:
  - Accept the result and downgrade to Terminal 2 (EUR/USD MR only if that cell alone still passes)
  - Accept the result as Terminal 3 (close project)
  - Modify deployment plan to account for weaker evidence

### Deliverable 6.D.3

If triggered: updated test results with plugin-based implementation, revised pass/fail table, revised terminal decision recommendation.

---

## Task 6.D.4 — Deployment Parameter Calibration from Reconciled Values

**Machine:** Omega

### Purpose
Once Tasks 6.D.1 and 6.D.2 establish the canonical max DD value, calibrate deployment parameters honestly.

### Task 6.D.4.1 — Auto-pause threshold

If canonical max DD is 24.6% (not 15.0%):
- Auto-pause at 15% means the strategy would have been paused multiple times in backtest
- Either: raise auto-pause to 25% (accept historical reality), or keep 15% with documented expectation that pauses will occur and resume protocol is necessary
- User's prior preference was Option C from Phase 5.5: 15% auto-pause with resume protocol. Reconfirm this choice given accurate max DD information.

### Task 6.D.4.2 — Position sizing

Max DD drives position sizing relative to account. If true max DD is 24.6%:
- At 0.5% risk per trade and historical max DD patterns, what is the expected worst-case demo account drawdown?
- Should initial position sizing be reduced to target 15% worst-case DD instead of 25%?

### Task 6.D.4.3 — Monitoring alerts

Set alert thresholds based on canonical metrics:
- 50% of max DD: warning
- 75% of max DD: serious warning
- 100% of max DD: auto-pause
- All thresholds documented with backtest-derived expected frequency of crossing

### Deliverable 6.D.4

Calibrated deployment parameters document ready to feed into the OANDA demo deployment plan.

---

## Task 6.D.5 — Update Terminal Decision Documentation

**Machine:** Omega

### Task 6.D.5.1 — Revise PHASE_6C_SYNTHESIS.md

Add a Phase 6.D addendum section at the end documenting:
- Identified discrepancies
- Resolution of each
- Updated canonical values
- Whether terminal decision (Terminal 1) remains valid after reconciliation
- Calibrated deployment parameters

Do NOT edit the original synthesis body — only add an addendum. The original is historical record.

### Task 6.D.5.2 — Produce PHASE_6D_RECONCILIATION.md

Stand-alone document with:
- The two discrepancies as identified by review
- Investigation findings for each
- Root cause determination
- Corrected values and implications
- Whether any re-runs were triggered
- Final disposition: Terminal 1 confirmed, Terminal 1 with revised parameters, Terminal 2, or Terminal 3

### Deliverable 6.D.5

Updated synthesis with Phase 6.D addendum, plus standalone reconciliation document.

---

## Execution Schedule

```
Day 1:  6.D.1 max DD reproduction (Omega)
        6.D.2 implementation audit (Omega, parallel)
        6.D.3 decision on whether re-runs needed
        
Day 2:  6.D.3 re-runs if triggered (assigned machines per test)
        6.D.4 parameter calibration (Omega)
        6.D.5 documentation update (Omega)
```

If no re-runs are triggered (discrepancies are explainable without affecting pass/fail), Phase 6.D completes in 1 day.

If re-runs are triggered, Phase 6.D takes 2 days with the re-runs running on their original machines in parallel with documentation.

---

## Standing Rules for Phase 6.D

1. **No new research.** Only reconciliation of existing results.
2. **No threshold modification.** The pre-registered kill criteria from Phase 6.C remain in force. If re-runs cause a pass to flip to fail, that flip is honored.
3. **Documentation is primary output.** The value of Phase 6.D is epistemic clarity, not improved results.
4. **User decision required if re-runs flip outcomes.** The agent should not unilaterally change the terminal decision if reconciliation reveals issues. Escalate with clear framing.
5. **If discrepancies are fully explainable with no impact**, proceed to deployment planning (Phase 6.E — OANDA demo plan) with confidence.

---

## Possible Phase 6.D Outcomes

### Outcome A: Discrepancies explainable, Terminal 1 confirmed
- Max DD difference is due to implementation details (cost model, aggregation method)
- All Phase 6.C tests used consistent implementation
- Canonical max DD value is documented
- Deployment parameters calibrated to canonical value
- **Proceed to OANDA demo deployment with confidence**

### Outcome B: Discrepancies explainable, parameters need recalibration
- Max DD is genuinely 24.6% (not 15%)
- Auto-pause threshold and position sizing must be updated
- Terminal 1 stands but demo deployment expectations are revised downward
- **Proceed with revised parameters**

### Outcome C: Discrepancies indicate re-runs needed
- Some Phase 6.C tests used scripts; plugin-based re-runs flip marginal tests
- Terminal 1 may become Terminal 2 or Terminal 3
- **User decision required on how to proceed**

### Outcome D: Discrepancies reveal deeper issue
- Unable to reconcile max DD with either 15% or 24.6%
- Plugin vs script divergence is larger than expected
- Indicates possible implementation bug
- **Escalate to user; may require Phase 6.B re-audit of plugin correctness**

Outcomes A and B are most likely (estimated 70-85% combined probability) given the pattern of modular pipeline discrepancies being usually cost model or aggregation differences. Outcome C is possible (10-20%) if certain tests used scripts. Outcome D is unlikely (<10%) but must be acknowledged.

---

## Honest Acknowledgments

1. **This phase should have been included in Phase 6.C.** The discrepancies I identified in review should have been caught as part of standard result validation. The fact that they were not in the original synthesis reflects either a gap in the Phase 6.C execution (rushed final synthesis) or limitations of the agent's self-audit (similar to Phase 5 where self-critique happened but was not integrated into conclusions).

2. **Phase 6.D is short but critical.** A 1-2 day reconciliation before commitment prevents deploying with mis-calibrated parameters, which could produce live behavior that surprises the user and triggers unnecessary pauses.

3. **If the canonical max DD is genuinely 24.6%**, this changes the project's self-narrative but not its fundamental validity. P3 is still the best available candidate per Phase 6.B. A 24.6% max DD is within the range of deployed managed futures strategies, just worse than the 15% previously claimed.

4. **I am not running the code.** The agent executes. If the agent encounters issues that suggest my task decomposition is wrong (e.g., 6.D.1 cannot reproduce either value), the agent should flag and re-scope rather than force the plan.

5. **This is the last pre-deployment check.** After Phase 6.D, the next work is either Phase 6.E OANDA demo deployment plan (if Terminal 1 confirmed) or revised terminal state document. No further research phases exist in the current project scope.