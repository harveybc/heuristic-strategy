# Work Plan: Phase 6.D.1 — Bug Fix and Re-Execution

**Prior state:** Phase 6.D identified a double vol-scaling bug in `phase6c_omega.py` that inflated max drawdown in Phase 6.C results by a factor of ~1.36. Root cause analysis established canonical max DD as 18.9% at 10% target vol. Sharpe-based pass/fail criteria were unaffected, so Terminal 1 decision remains valid. However, reported max DD values across Phase 6.C tests (including held-out 16.7%) are inflated relative to canonical methodology. Documentation contains internally inconsistent numbers.

**Purpose:** Fix the identified bug in the evaluation code, re-execute Phase 6.C tests that were affected, and produce a clean set of canonical metrics before deployment. No new research, no new tests, no threshold changes — only correction and re-verification.

**Duration:** 1-2 days.

**Hard constraint:** No pass/fail criterion is modified during this phase. The pre-registered Phase 6.C thresholds remain fixed. The only outputs that can change are reported metric values, not pass/fail determinations.

**Scope boundary:** Phase 6.D.1 only corrects and re-runs. It does NOT reopen investigation of stress test design, does NOT consider additional candidates, does NOT reinterpret Phase 6.C.5 failure. If the bug fix reveals that any test result changes pass/fail, that is escalated to user for terminal decision review — but the scope of Phase 6.D.1 work itself stays narrow.

---

## Context: Why This Phase Exists

Phase 6.D did everything except the final step: applying the correction. The agent:
1. ✅ Identified the bug (double vol-scaling in `eval_portfolio_daily()`)
2. ✅ Quantified impact on max DD (factor of ~1.36)
3. ✅ Established canonical methodology (single vol-scaling, exp(cumsum))
4. ✅ Calibrated deployment parameters against canonical 18.9% max DD
5. ❌ **Did not fix the bug in the actual code**
6. ❌ **Did not re-run the affected tests with corrected code**

This left documentation with a mix of values: Phase 6.C tables report numbers from buggy code, Phase 6.D addendum explains the bug but does not overwrite the numbers. For a deployment decision with real stakes, the canonical numbers should be in the primary documentation, not in addenda explaining why the primary numbers are wrong.

Additionally, Phase 6.D did not investigate a subtle inconsistency visible in the reconciliation JSON: Phase 5.5 reproduction shows 977 weeks, Phase 6.C.3 reproduction shows 1168 weeks. The two analyses may have used different period endpoints, which is a separate issue to verify.

---

## Task 6.D.1.1 — Code Fix (Day 1, Omega, 2-3 hours)

### Task 6.D.1.1.A — Locate the double vol-scaling

In `trading_research/phase6c_omega.py`, locate:
- `eval_cell()` function: scales each cell to 10% daily volatility
- `eval_portfolio_daily()` function: aggregates cells and scales portfolio to 10% weekly volatility

The bug: each cell is already at 10% vol after `eval_cell()`. Applying portfolio-level scaling then amplifies by the reciprocal of the diversification factor (~1/0.73 ≈ 1.37).

### Task 6.D.1.1.B — Decide correct behavior

Two correct approaches:
- **Option 1:** Scale cells to 10% vol, leave portfolio at resulting realized vol (7.3% weekly). Report metrics at realized vol and state explicitly.
- **Option 2:** Do not scale cells individually. Scale only the final portfolio to 10% vol. This is the "target vol portfolio" approach used in most institutional managed futures.

Phase 6.D established canonical max DD at 10% vol via Option 2 approach (though computed post-hoc). For consistency with deployment (which targets 10% vol at portfolio level), use Option 2:

**Corrected behavior:**
1. Each cell produces raw strategy returns without vol scaling
2. Combine cells with P3 weights (20.6% / 49.2% / 30.2%)
3. Compute realized portfolio vol
4. Scale entire portfolio return series by `(0.10 / realized_vol_annualized)` to target 10% annualized vol
5. Compute metrics from this scaled series

### Task 6.D.1.1.C — Implement the fix

Modify `phase6c_omega.py` and any other files that use the same flawed pattern:
- Remove the per-cell 10% vol scaling inside `eval_cell()` OR
- Remove the portfolio-level 10% vol scaling in `eval_portfolio_daily()`

Choose the modification that matches the canonical Option 2 approach above. Document the change clearly in a code comment citing Phase 6.D.

Also fix the equity curve computation: use `exp(cumsum(log_returns))` consistently, not `cumprod(1 + log_returns)`. The latter treats log returns as arithmetic returns and introduces bias.

### Task 6.D.1.1.D — Validate fix with spot check

Run the corrected code on full 2003-2025 history. Confirm:
- Max DD matches canonical 18.9% (±0.5pp tolerance, may differ slightly due to exact period boundaries)
- Sharpe matches Phase 5.5 (0.449) and Phase 6.C original (0.447) within 0.02

If either metric differs materially from expectation, the fix may have introduced a new issue. Investigate before proceeding.

### Deliverable 6.D.1.1
Corrected `phase6c_omega.py` (and related files if needed) with inline comments referencing Phase 6.D root cause. Validation output confirming canonical metrics reproduced.

---

## Task 6.D.1.2 — Re-Execute Affected Phase 6.C Tests (Day 1, distributed)

Only tests whose implementation used the buggy pattern need re-execution. Per Phase 6.D implementation audit:

| Test | File | Re-run required? | Reason |
|------|------|------------------|--------|
| 6.C.0 Held-Out | phase6c_omega.py | **YES** | Uses `eval_portfolio_daily()` with bug |
| 6.C.1 JPY Reversal | phase6c_stress.py | Verify | Need to audit if same bug present |
| 6.C.2 MC Regimes | phase6c_stress.py | Verify | Need to audit if same bug present |
| 6.C.3 Cost Sensitivity | phase6c_omega.py | **YES** | Uses buggy code directly |
| 6.C.4 Walk-Forward | phase6c_stress.py | Verify | Need to audit if same bug present |
| 6.C.5 Param Perturbation | phase6c_stress.py | Verify | Relative metric, may be unaffected |

### Task 6.D.1.2.A — Audit phase6c_stress.py for same bug

**Machine:** Omega

Check whether `phase6c_stress.py` (used by 6.C.1, 6.C.2, 6.C.4, 6.C.5) has the same double vol-scaling pattern. If yes, apply equivalent fix. If no, confirm those tests are unaffected and document the confirmation.

### Task 6.D.1.2.B — Re-run 6.C.0 Held-Out with corrected code

**Machine:** Omega

Rerun held-out evaluation on 2024-2025 data. Report:
- Corrected Sharpe (expected near 0.316, same as before since bug affected DD not Sharpe)
- **Corrected max DD** (expected lower than 16.7%, likely ~12-13%)
- Per-cell metrics with corrected methodology
- Confirm pass/fail status unchanged

### Task 6.D.1.2.C — Re-run 6.C.3 Cost Sensitivity with corrected code

**Machine:** Omega

Rerun the full cost multiplier grid (1.0x, 1.25x, 1.5x, 1.75x, 2.0x, 2.5x, 3.0x). Report:
- Corrected Sharpe at each cost level (expected near original values)
- **Corrected max DD at each cost level** (expected lower than reported)
- Corrected worst-2Y values (may differ slightly)
- Confirm pass/fail status unchanged

### Task 6.D.1.2.D — Re-run 6.C.1, 6.C.2, 6.C.4, 6.C.5 IF audit in 6.D.1.2.A reveals same bug

**Machines:** As originally assigned (Gamma for 6.C.1 and 6.C.5, Dragon for 6.C.2 and 6.C.4)

If the bug is present in phase6c_stress.py:
- Fix same way as phase6c_omega.py
- Re-run all affected stress tests
- Confirm pass/fail status unchanged across all tests

If the bug is not present: document this explicitly in 6.D.1.2.A output, no re-runs needed.

### Kill criterion for Phase 6.D.1

If re-runs reveal that ANY test changes pass/fail status after bug correction:
- Escalate to user immediately
- Do not proceed to synthesis
- User decides whether Terminal 1 remains valid or requires downgrade

Expected outcome: no pass/fail changes (Sharpe-based criteria unaffected by max DD bug, by agent's Phase 6.D analysis). But this must be verified, not assumed.

### Deliverable 6.D.1.2
Corrected test results for all affected Phase 6.C tests. Explicit confirmation of pass/fail stability (or escalation if any flipped).

---

## Task 6.D.1.3 — Investigate Week Count Discrepancy (Day 1, Omega, 1-2 hours)

### The specific question
`phase_6d_reconciliation.json` shows:
- Phase 5.5 reproduction: 977 weeks
- Phase 6.C.3 reproduction: 1168 weeks

A gap of 191 weeks (~3.7 years). This was not addressed in the Phase 6.D root cause analysis but affects the max DD comparison: more history means more opportunities for a worse DD.

### Task 6.D.1.3.A — Identify period endpoints for each reproduction

Check data loading in the reconciliation script:
- What start date and end date did Phase 5.5 reproduction use?
- What start date and end date did Phase 6.C.3 reproduction use?
- Are both ranges intended to be "full history" or did one include additional data?

### Task 6.D.1.3.B — Determine if this is a fifth factor (F5)

The 9.6pp gap was decomposed into F1-F4 summing to 107% (with -1.1pp interaction). If period length is an additional unacknowledged factor (F5: longer period → higher historical max DD by chance), the previous factor decomposition may be slightly overfit.

Compute max DD on Phase 6.C.3's full 1168-week period using Phase 5.5's methodology. If this differs from 18.9% by more than 1-2pp, there is genuine period-length effect.

### Task 6.D.1.3.C — Report canonical max DD for the deployment-relevant period

The relevant period for deployment is "full history through 2023" (since 2024-2025 is held-out). Establish canonical max DD for this specific period using corrected methodology. This is the number that should feed into auto-pause and deployment parameter calibration.

### Deliverable 6.D.1.3
Period endpoint documentation. Canonical max DD confirmed for full 2003-2023 training period (the relevant reference for deployment).

---

## Task 6.D.1.4 — Update Documentation (Day 2, Omega, 2-3 hours)

### Task 6.D.1.4.A — Produce corrected PHASE_6C_SYNTHESIS.md

Create a new version of the Phase 6.C synthesis that:
- Contains the original analysis structure (unchanged)
- Replaces buggy metric values with canonical ones
- Retains Phase 6.D addendum but restructures as "Documentation of Correction" explaining why numbers differ from earlier versions
- Adds clear version indicator: "v2 — post-6.D.1 correction"

The original `PHASE_6C_SYNTHESIS.md` is preserved as `PHASE_6C_SYNTHESIS_v1_DEPRECATED.md` with note at top: "This version contained a vol-scaling bug; see v2 for canonical values."

### Task 6.D.1.4.B — Update Phase 6.D reconciliation with corrected downstream impact

Add a section to `PHASE_6D_RECONCILIATION.md` (or produce `PHASE_6D1_EXECUTION.md`) documenting:
- What code was fixed
- What tests were re-run
- Before/after metric comparisons
- Confirmation that pass/fail status unchanged (or escalation details if not)

### Task 6.D.1.4.C — Update deployment parameter calibration

If canonical max DD for the deployment-relevant period (2003-2023) differs from the 18.9% established in Phase 6.D:
- Recalibrate auto-pause, hard stop, warning levels
- Update position sizing guidance accordingly

If canonical max DD is approximately 18.9% (as expected): confirm existing deployment parameters are correct, no changes needed.

### Deliverable 6.D.1.4
Clean, internally consistent Phase 6.C and 6.D documentation with canonical values throughout. Deployment parameters confirmed or recalibrated.

---

## Task 6.D.1.5 — Final Verification Summary (Day 2, Omega, 1 hour)

Produce `PHASE_6D1_SUMMARY.md` containing:

1. Bug description (one paragraph, citing Phase 6.D analysis)
2. Fix applied (file, function, change summary)
3. Tests re-executed with before/after comparison table
4. Pass/fail status unchanged confirmation (or escalation)
5. Final canonical metrics for all Phase 6.C tests
6. Updated deployment parameters (if any changed)
7. Confirmation that Terminal 1 remains valid

This document is the single source of truth summarizing Phase 6.D.1 work, and becomes the "last checkpoint" before Phase 6.E deployment plan.

### Deliverable 6.D.1.5
One-page summary suitable for rapid review of all Phase 6.D.1 work.

---

## Execution Schedule

```
Day 1 morning:   6.D.1.1 code fix + validation (Omega, 2-3h)
                 6.D.1.3 week count audit (Omega, 1-2h, can run parallel)
                 
Day 1 afternoon: 6.D.1.2.A audit phase6c_stress.py (Omega, 1h)
                 6.D.1.2.B re-run 6.C.0 held-out (Omega, <30min)
                 6.D.1.2.C re-run 6.C.3 cost sensitivity (Omega, <30min)
                 If 6.D.1.2.A confirms bug in stress.py: launch re-runs on Gamma and Dragon

Day 2 morning:   If needed, complete re-runs on Gamma/Dragon
                 6.D.1.4 documentation updates (Omega)
                 
Day 2 afternoon: 6.D.1.5 final verification summary (Omega)
                 Sign-off: ready for Phase 6.E deployment plan
```

If phase6c_stress.py does not have the bug, Phase 6.D.1 completes in 1 day. If it does and 4 stress tests need re-running, the full 2 days is used with Gamma and Dragon active on Day 1 afternoon through Day 2 morning.

---

## Standing Rules for Phase 6.D.1

1. **No threshold changes.** Pre-registered Phase 6.C kill criteria remain fixed. If a bug-corrected result crosses a threshold in either direction, the pre-registered threshold governs.

2. **No scope creep.** Phase 6.D.1 only corrects and re-verifies. It does not introduce new tests, new candidates, or new analyses beyond fixing the identified bug.

3. **All corrected results are first-class.** The canonical documentation after Phase 6.D.1 replaces buggy values, not just supplements them. Deprecated versions are preserved for audit trail but clearly marked.

4. **Escalation is the correct response to surprises.** If bug fix changes any pass/fail outcome, stop and escalate. Do not silently update terminal decision.

5. **Deployment does not begin until Phase 6.D.1 is complete.** The OANDA demo deployment plan (Phase 6.E) must work from corrected metrics, not a mix of original and corrected.

---

## Expected Outcomes and Their Implications

### Outcome A (most likely, ~75% probability)
- Bug fix applied, re-runs complete
- Max DD values decrease across tests (e.g., 6.C.0 held-out goes from 16.7% to ~12-13%, 6.C.3 cost sensitivity DD column all shift down by ~25%)
- Sharpe values unchanged
- All pass/fail statuses unchanged
- Canonical max DD for 2003-2023 period confirmed at 18.9%
- Deployment parameters from Phase 6.D remain valid (auto-pause 20%, hard stop 28.3%)
- **Proceed to Phase 6.E deployment plan**

### Outcome B (possible, ~15% probability)
- Bug fix reveals additional anomaly not captured in Phase 6.D analysis
- One or more tests show meaningful metric changes beyond max DD
- Pass/fail status for marginal tests (especially 6.C.1 median Sharpe = 0.000) needs re-evaluation
- May require user decision on Terminal state

### Outcome C (less likely, ~8% probability)
- Week count discrepancy reveals that Phase 5.5 and Phase 6.C.3 used different periods
- Canonical max DD for 2003-2023 period differs from 18.9%
- Deployment parameters need recalibration
- Terminal 1 likely remains valid but thresholds shift

### Outcome D (unlikely, ~2% probability)
- Bug fix reveals that more code beyond phase6c_omega.py has the same pattern, potentially affecting Phase 5.5 results
- This would require re-examining the canonical 18.9% itself
- Would extend Phase 6.D.1 significantly or require separate investigation

---

## Honest Acknowledgments About This Plan

1. **This phase should have been part of Phase 6.D.** The agent identified the bug but did not complete the correction. Phase 6.D.1 exists to finish that work. The pattern of "identify issues but not fully integrate corrections" has appeared multiple times in this project — worth noting for future work quality.

2. **The 75% probability estimate for Outcome A reflects that the bug is well-characterized.** If the root cause analysis in Phase 6.D was correct (factor decomposition summed to 107%), then fixing the primary factor (F2: double vol-scaling) should produce predictable results. Deviation from expectation would indicate the analysis missed something.

3. **Two full days for this work may be conservative.** Actual re-execution time is short (total compute is <1 hour). The schedule allows time for surprises and careful documentation. If everything goes smoothly, Phase 6.D.1 completes in half a day.

4. **Phase 6.D.1 is the last pre-deployment checkpoint.** After this, any issues discovered during the OANDA demo observation period are live-trading findings, not research findings. The project's research phase closes with Phase 6.D.1 synthesis.

5. **I may be overcautious.** The agent's Phase 6.D analysis was rigorous, and an argument could be made that deployment could proceed with acknowledged documentation inconsistency. The choice to require Phase 6.D.1 reflects preference for clean documentation over speed. If the user wishes to skip Phase 6.D.1 and proceed directly to Phase 6.E, that is a defensible choice — but the user should explicitly opt into that path knowing the documentation has resolved caveats.