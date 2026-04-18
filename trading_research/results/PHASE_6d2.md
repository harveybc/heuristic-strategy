# Work Plan: Phase 6.D.2 — Final Documentation Consolidation

**Prior state:** Phase 6.D.1 complete. Double vol-scaling bug fixed in `phase6c_omega.py` and `phase6c_stress.py`. All Phase 6.C tests re-run with corrected code. Zero pass/fail flips. Terminal 1 (Deploy P3 to OANDA Demo) remains valid. However, the current `PHASE_6C_SYNTHESIS.md` contains a mixture of pre-fix and post-fix values: main body retains original buggy metrics (held-out max DD 16.7%, cost sensitivity 24.6%, auto-pause 20%), while corrected values appear in Phase 6.D and 6.D.1 addenda. Additionally, canonical max DD has two candidate values (18.9% vs 22.8%) depending on methodology choice.

**Purpose:** Produce a single internally consistent final document with canonical values throughout. Resolve the canonical max DD choice explicitly. Preserve audit trail. This is the final documentation deliverable before Phase 6.E deployment plan.

**Duration:** 4-6 hours. Single day.

**Machine:** Omega only (documentation work, no compute required).

**User decision required before execution:** Which canonical max DD to adopt (see Task 6.D.2.1). The agent should not make this choice unilaterally — it affects deployment risk thresholds and user should consciously choose.

---

## Context: Why This Phase Exists

The Phase 6.C "v2" produced after 6.D.1 is actually v1 with addenda appended. Section 1 of the synthesis reports held-out max DD as "13.6%" in one table and "16.7%" in the gate text (old value not updated). Section 4 states auto-pause at 20% but the 6.D.1 addendum revises it to 25% or 35%. A reader would need to read all addenda in sequence and mentally reconcile values to know the current state.

This kind of documentation drift is the pattern noted across the project: the agent identifies and fixes issues but does not always re-integrate corrections into the primary document. For a deployment decision with real stakes, the primary document must stand alone with canonical values.

Phase 6.D.2 is not new research. It is consolidation — the final step of the correction cycle begun in Phase 6.D.

---

## Decision Required Before Execution

### Decision: Which canonical max DD to adopt?

Two defensible candidates emerged from 6.D.1:

**Option A: 18.9%** — Phase 5.5 methodology with `valid >= 2` filter (requires ≥2 cells active for a week to count)
- Excludes 191 warm-up weeks where only 1 cell produces signals
- Max DD at 10% target vol over 977 weeks
- More representative of "steady-state portfolio" behavior

**Option B: 22.8%** — Phase 6.C methodology including all weeks
- Includes warm-up periods and single-cell episodes
- Max DD at 10% target vol over 1168 weeks
- More conservative for deployment — captures what can happen when cells are inactive

**Recommendation:** Option B (22.8%) because:
1. Real deployment will include warm-up periods when the strategy first goes live
2. Conservative risk calibration is appropriate for first-ever live deployment
3. User's implicit preference through the project has been toward rigor over convenience

**Option A is defensible if:** user prefers alignment with Phase 5.5 methodology as the canonical reference, or considers warm-up drawdowns not representative of expected live behavior.

**User must confirm Option A or B before agent begins Task 6.D.2.2.** If user does not specify, agent defaults to Option B per recommendation above.

### Corresponding deployment parameters

| Parameter | Option A (18.9%) | Option B (22.8%) |
|-----------|------------------|------------------|
| Canonical max DD (10% target vol) | 18.9% | 22.8% |
| Warning level (75% of canonical) | 14.2% | 17.1% |
| Auto-pause (approx 1.1× canonical) | 20% | 25% |
| Hard stop (1.5× canonical) | 28.3% | 34.2% |

---

## Task 6.D.2.1 — Audit Current Documentation Inconsistencies (30 min)

### Purpose
Before rewriting, catalog every place in `PHASE_6C_SYNTHESIS.md` where a pre-fix or post-fix value appears. This ensures nothing is missed in the rewrite.

### Task 6.D.2.1.A — Build inconsistency inventory

Read current `PHASE_6C_SYNTHESIS.md` end-to-end. For each numerical claim related to max DD, vol, auto-pause thresholds, or deployment parameters, record:
- Section and line reference
- Current value shown
- Whether it is pre-fix, post-fix (realized vol), post-fix (10% vol), or other
- Whether it needs replacement with canonical value

Expected findings (from review notes):
- Section 1 held-out table: "Max DD" column → shows 13.6% (corrected) but "Vol" shows 8.2% and "In-Sample Max DD" shows 15.8% (?)
- Section 1 held-out gate criteria text: references 16.7% (pre-fix)
- Section 2.3 cost sensitivity Max DD column: all values from pre-fix (24.6%, 25.1%, etc.)
- Section 4 deployment parameters: "Go/No-go threshold: max DD < 20%" (pre-fix)
- Section 4 Standing Rules: "max DD exceeds 20%" (pre-fix)
- Section 5 master table: mixed
- Phase 6.D addendum: 18.9% canonical
- Phase 6.D.1 addendum: 22.8% canonical

### Deliverable 6.D.2.1
Inventory table of all numeric claims requiring update, with pre-fix and post-fix values clearly identified.

---

## Task 6.D.2.2 — Produce PHASE_6C_SYNTHESIS_FINAL.md (2-3 hours)

### Purpose
Single authoritative final document containing all canonical values in the main body.

### Task 6.D.2.2.A — Structural decisions

**File naming:**
- New canonical document: `PHASE_6C_SYNTHESIS_FINAL.md`
- Deprecated document: current `PHASE_6C_SYNTHESIS.md` renamed to `PHASE_6C_SYNTHESIS_v1_DEPRECATED.md` with header note: "This version contains pre-correction values. See PHASE_6C_SYNTHESIS_FINAL.md for canonical state."
- Audit trail: `PHASE_6D_RECONCILIATION.md` and `PHASE_6D1_EXECUTION.md` remain in place as historical record

**Document structure:**
Follow the original Phase 6.C synthesis structure (sections 1-6), but with all values updated to canonical. No embedded addenda. Any material from Phase 6.D or 6.D.1 that belongs in the main narrative is integrated directly.

### Task 6.D.2.2.B — Content updates per section

**Section 1 — Held-out evaluation:**
- Table: update Max DD to canonical values (13.6% at realized vol 8.2%, or 16.2% at 10% vol, both from 6.D.1 results)
- Gate criteria text: replace "16.7%" with the correct canonical value based on which vol reporting convention the final document uses
- Per-cell table: verify values are consistent with corrected code

**Convention decision:** the final document should consistently report metrics at ONE vol level per metric. Options:
- Report at realized vol throughout (with a note giving 10%-vol equivalents)
- Report at 10% target vol throughout (cleaner for deployment since 10% is the operational target)

**Recommendation:** Report at 10% target vol for deployment-relevant metrics (max DD, absolute PnL), and scale-invariant metrics (Sharpe, correlations) are the same at any vol. Include a single "Vol reporting convention" note at the start of the metrics section.

**Section 2 — Stress test results:**
- 2.1 JPY reversal: values unchanged (was identical post-fix)
- 2.2 MC regimes: values unchanged (was identical post-fix)
- 2.3 Cost sensitivity: update the Max DD column with canonical values from 6.D.1 re-run. The Sharpe and worst-2Y columns remain as originally reported (scale-invariant)
- 2.4 Walk-forward: values unchanged
- 2.5 Parameter perturbation: values unchanged (FAIL status remains)

**Section 3 — Decision tree:**
- Update any max DD references to canonical values
- Gate criterion "max DD < 25%" or "< 30%" in Phase 6.C text remains unchanged (these are the pre-registered thresholds, not metric values)
- Pass/fail conclusions unchanged

**Section 4 — Terminal decision:**
- Update "Go/No-go threshold" for max DD from 20% to the value chosen based on canonical decision (Option A or B)
- Update Standing Rules max DD threshold consistently
- Update "Risk Factors to Monitor" with canonical values
- Canonical max DD stated explicitly with methodology note

**Section 5 — Master results table:**
- Update "Max DD" row values to canonical
- All other values remain (scale-invariant or unchanged)

**Section 6 — Appendix:**
- Compute infrastructure unchanged
- File references updated: point to corrected Phase 6.C results files where applicable

**New Section (if needed) — Canonical Metric Methodology:**
- Document the vol reporting convention used
- Document the canonical max DD choice (Option A or B) with rationale
- Reference Phase 6.D and 6.D.1 as the source of these definitions

### Task 6.D.2.2.C — Remove addenda from main synthesis

The Phase 6.D and 6.D.1 addenda are not included in `PHASE_6C_SYNTHESIS_FINAL.md`. Their content relevant to final state is integrated into the main body. Their audit-trail content remains in their standalone files.

One short paragraph at the start of the final synthesis can reference audit trail: "This document reflects the state after Phase 6.D reconciliation and Phase 6.D.1 bug correction. See PHASE_6D_RECONCILIATION.md and PHASE_6D1_EXECUTION.md for the full correction history."

### Deliverable 6.D.2.2
`PHASE_6C_SYNTHESIS_FINAL.md` — single authoritative document with canonical values throughout. No internal contradictions, no embedded addenda.

---

## Task 6.D.2.3 — Produce Project-Level Summary Document (1-2 hours)

### Purpose
In addition to the Phase 6.C final synthesis, produce a one-layer-up document summarizing the full research program. This serves as the entry point for anyone encountering the project fresh (including future-you in 6 months).

### Task 6.D.2.3.A — Create PROJECT_RESEARCH_SUMMARY.md

Structure:

**Section 1: Project objective**
- Build profitable automated trading agent via OANDA API
- Constraints evolved: initially PDT-compliant, later flexible
- Final deliverable path: P3 portfolio on OANDA demo → potential real capital

**Section 2: Phases completed and their essential findings**

| Phase | Purpose | Key Finding |
|-------|---------|-------------|
| 0 | Diagnostic infrastructure | Oracle sensitivity, cost model, rolling window harness built |
| 1 | Asset × timeframe × strategy sweep | Weekly timeframes dominate; 65 cells with budget > 0 |
| 2 | EUR/USD MR audit | Hourly MR not tradeable (parameter spike, dies at 1.5 bps) |
| 3 | Exogenous data enrichment | FRED, ECB, DXY, VIX, COT integrated |
| 3.5 | Cross-asset comparison audit | All 14 initial cells killed by extended-history worst-window |
| 4 | Four parallel tracks (MR deploy, regime filter, academic, GBP/USD) | Threshold of -0.5 too strict vs industry; diversification rescue possible |
| 5 | Three decisive questions | Portfolio diversification works (P3 identified) |
| 5.5 | Corrective audit of Phase 5 | 11 issues found; P3 refined with oracle-free cells only |
| 6.A | Audit of untested combinations | F1 gap rules out predictor-primary; 3 regime-based candidates identified |
| 6.B | Evaluation of untested combinations | All 3 candidates failed; P3 confirmed as best available |
| 6.C | Robustness validation of P3 | 4/5 stress tests pass; Terminal 1 |
| 6.D | Discrepancy reconciliation | Double vol-scaling bug identified |
| 6.D.1 | Bug fix and re-run | Bug fixed, zero pass/fail flips |
| 6.D.2 | Documentation consolidation (this phase) | Canonical final document produced |

**Section 3: Final state**
- Terminal decision: Terminal 1 (Deploy P3 to OANDA Demo for 90-day observation)
- Portfolio: P3 (EUR/USD MR 20.6%, USD/JPY TSMOM 49.2%, USD/JPY DM 30.2%)
- Canonical metrics (updated for user's choice of Option A or B)
- Risk calibration: auto-pause, hard stop, warning levels
- Material caveats: USD/JPY concentration, 6.C.5 parameter sensitivity, modest Sharpe

**Section 4: What was NOT achieved**
- No predictor-primary strategy viable (F1 gap)
- No alternative-asset research (BTC, ETH, crypto, commodities) passed all criteria
- No causal-enhanced strategy outperformed oracle-free baseline
- Portfolio concentration unresolved structurally

**Section 5: Honest self-assessment**
- Sharpe 0.316 held-out is modest by institutional standards
- USD/JPY concentration is real structural risk
- Parameter sensitivity (6.C.5 FAIL) suggests fragility
- Multiple phases of correction suggest methodological rigor improved over time rather than being present from start
- Live deployment is the next source of evidence

**Section 6: What this project accomplishes**
Regardless of live deployment outcome:
- Comprehensive documented research on retail systematic trading with public data
- Reusable infrastructure (feature-eng, predictor, prediction_provider, heuristic-strategy, lts, causal-inference)
- Portfolio piece demonstrating rigorous methodology with honest negative results
- Calibrated expectations for what's achievable at this scale

**Section 7: References**
- Link to Phase 6.C final synthesis as authoritative technical reference
- Link to earlier phase reports for historical record
- Code repos list

### Deliverable 6.D.2.3
`PROJECT_RESEARCH_SUMMARY.md` — executive-level overview of the entire project, written so that a reader unfamiliar with the details can understand what was done and what the final state is.

---

## Task 6.D.2.4 — Preserve Audit Trail (30 min)

### Task 6.D.2.4.A — File organization

Create clear directory structure or file naming indicating which documents are canonical vs historical:

**Canonical (current state):**
- `PHASE_6C_SYNTHESIS_FINAL.md`
- `PROJECT_RESEARCH_SUMMARY.md`

**Historical (preserved for audit trail):**
- `PHASE_6C_SYNTHESIS_v1_DEPRECATED.md` (original synthesis with pre-fix values)
- `PHASE_6D_RECONCILIATION.md` (unchanged)
- `PHASE_6D1_EXECUTION.md` (unchanged)
- All prior phase reports (3.5, 4, 5, 5.5, 6.A, 6.B) unchanged

At the top of each deprecated/historical file, add a brief header indicating its status: "This is a historical artifact. See PHASE_6C_SYNTHESIS_FINAL.md for current canonical state."

### Task 6.D.2.4.B — Update any cross-references

If other documents reference the old synthesis file path, update references to point to the final. Audit-trail references remain pointing to historical files.

### Deliverable 6.D.2.4
File organization with clear canonical-vs-historical marking.

---

## Task 6.D.2.5 — Final Verification (30 min)

### Task 6.D.2.5.A — Consistency check

Read `PHASE_6C_SYNTHESIS_FINAL.md` end-to-end. For every numerical claim about max DD, cross-check that it matches the canonical value chosen. For every threshold reference (auto-pause, hard stop, warning), confirm alignment.

### Task 6.D.2.5.B — Cross-check against PROJECT_RESEARCH_SUMMARY.md

Ensure numerical consistency between the summary document and the technical synthesis. Any value mentioned in both places must match.

### Task 6.D.2.5.C — Produce confirmation

Short sign-off note at the end of `PROJECT_RESEARCH_SUMMARY.md` or as separate file: "Documentation consolidation complete. Ready for Phase 6.E deployment planning."

### Deliverable 6.D.2.5
Final consistency verification. Sign-off.

---

## Standing Rules for Phase 6.D.2

1. **No re-running of backtests.** All values used come from Phase 6.D.1 outputs or earlier canonical reports. No new computation.

2. **User chooses canonical max DD.** The agent does not make this choice unilaterally. If user has not specified, agent asks before Task 6.D.2.2 begins.

3. **Deprecated documents are preserved, not deleted.** Audit trail is valuable. Mark clearly, do not remove.

4. **Final document is atomic.** It either contains all canonical values correctly, or it does not. No partial updates that leave some sections stale.

5. **No scope creep.** Phase 6.D.2 is consolidation only. It does not re-analyze results, question canonical choices beyond what user decided, or introduce new metrics.

6. **Deployment plan does not start until 6.D.2 is complete.** Phase 6.E (OANDA demo deployment plan) uses `PHASE_6C_SYNTHESIS_FINAL.md` as its input.

---

## Execution Schedule

```
Hour 1:     User confirms canonical max DD choice (Option A or B)
            Task 6.D.2.1 audit current inconsistencies
            
Hours 2-4:  Task 6.D.2.2 produce FINAL synthesis
            Task 6.D.2.3 produce PROJECT_RESEARCH_SUMMARY

Hour 5:     Task 6.D.2.4 preserve audit trail, file organization
            Task 6.D.2.5 final consistency verification

Hour 6:     Sign-off, handoff to Phase 6.E
```

Total wall-clock: 5-6 hours including user decision and thorough review. If user pre-specifies canonical choice, execution time is closer to 4 hours.

---

## Honest Acknowledgments About This Plan

1. **This phase should have been the final step of Phase 6.D.1.** When the bug was fixed and tests were re-run, the synthesis should have been updated inline. That it requires a dedicated phase now reflects the pattern this project has shown: technical correction is rigorous, but integration into final documentation is often incomplete on first pass.

2. **I am making a judgment call on format conventions.** Reporting at 10% target vol throughout deployment-relevant sections is my recommendation based on institutional convention; realized vol reporting is equally defensible. If user prefers realized vol as the primary convention, the plan can be executed with that choice.

3. **PROJECT_RESEARCH_SUMMARY.md is scope expansion compared to the strict "documentation consolidation" mandate.** I included it because in my assessment, a project of this size with this many phases warrants an executive summary. If user considers this out of scope, Task 6.D.2.3 can be skipped and only the Phase 6.C final synthesis produced.

4. **The canonical max DD choice has material downstream effects.** Option B (22.8%) implies auto-pause at 25%, which means the strategy could go to 25% drawdown live before auto-pause triggers. On a $10k demo, that is $2,500 down before intervention. User should choose consciously, not by default.

5. **This is the end of research-phase documentation.** After Phase 6.D.2, the next document is Phase 6.E deployment plan, followed by operational monitoring reports. No new analytical phases exist under the current project scope.

---

## What Success Looks Like

At end of Phase 6.D.2:

- **Single canonical technical document:** `PHASE_6C_SYNTHESIS_FINAL.md` with all values consistent, canonical max DD clearly stated, deployment parameters calibrated.

- **Executive summary document:** `PROJECT_RESEARCH_SUMMARY.md` providing entry point to the project for any reader.

- **Preserved audit trail:** all historical documents clearly marked as superseded but available for review.

- **Ready for Phase 6.E:** deployment plan can be written directly from canonical documents without needing to cross-reference multiple sources.

The agent can then confirm: "Documentation consolidated. Ready for deployment planning. Canonical max DD is X%, auto-pause at Y%, hard stop at Z%. Phase 6.E work plan can begin."