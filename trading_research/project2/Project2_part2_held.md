# Project 2 — Stage II-6: Held-Out Validation Plan

**Purpose:** Before Part III, validate that Part II-Redux Path A and Path B "PASS" verdicts hold on held-out 2020-2025 data. Current Part II-Redux results are IS-only. Without held-out validation, we repeat Project 1's false-positive pattern where IS-positive strategies collapsed OOS.

**Scope:** Narrow and focused. This is NOT new research — it is validation of existing claims.

**Gate to Part III:** Held-out evidence either supports Part II-Redux claims or invalidates them. Part III scope depends on outcome.

---

## 0. Why This Stage Exists

### 0.1 What Part II-Redux actually delivered

Re-reading the deliverables carefully:

- **Path A "passing" experiments (A1, A2, A6)** evaluated on in-sample 2017-2019 BTC data. A2 has 24 monthly rolling windows all within 2017-2019.
- **Path B "passing" experiment (B3 TFT)** evaluated on same 2017-2019 window structure.
- **Held-out 2020-2025 was never touched.** No stage II-3 or II-5 experiment computed metrics on 2020-2025.

### 0.2 Why this matters

The entire motivation of pre-registering a held-out period is Project 1's discovery that plugin-canonical 2024-2025 held-out Sharpe was -0.065 despite positive full-period metrics. Part II-Redux re-created this exact pattern: positive in-sample, held-out untested.

The F-10 evaluation framework pre-registered K-1 (held-out Sharpe > 0) as primary kill criterion. Part II-Redux deliverables mark K-1 as "PASS" but what was actually measured is IS aggregate Sharpe, not held-out. This is a methodology gap, not a finding.

### 0.3 Other gaps this stage addresses

- **Deflated Sharpe Ratio never computed.** F-10 §5.2 specifies DSR accounting for multiple testing. With 17 experiments run, DSR penalty is material. A2 raw Sharpe +0.156 may be DSR 0.05-0.10.
- **A4 regime_adaptive produced $52M equity from 2 trades.** Stage II-3 dismissed this as "likely a bug" and moved on. A bug that produces 5200× capital gain from 2 trades is a serious infrastructure issue, not a result to list as FAIL and skip.
- **Stage II-0b advisory failures unresolved.** EUR/USD 4h and USD/JPY daily failed Test 5 (autocorrelation). Handwaved as "expected for efficient markets." But the test was designed precisely to catch data issues, and the excuse pattern is concerning.

### 0.4 What this stage is NOT

- Not new strategy research. No new experiments beyond validation.
- Not scope expansion. Same assets, same Path A/B configurations that passed.
- Not a second chance for failed experiments. Those stay failed.

This is pure validation: do the claimed winners hold up on data they have not seen?

---

## 1. Agent Contract (reinforced for this stage)

Same rules as Part II-Redux. Particularly:

- **No synthetic fallback ever.** If held-out data has gaps, escalate.
- **No repeat of IS data in held-out evaluation.** 2017-2019 is IS. 2020-2025 is held-out. Period.
- **No post-hoc threshold adjustment.** F-10 kill criteria apply as pre-registered.
- **Explicit escalation for the A4 bug.** This is infrastructure issue, must be understood before any further experiments.

---

## 2. Stage Structure

Five sub-tasks, sequential:

| Task | Purpose | Machine |
|------|---------|---------|
| **II-6.1** | A4 equity bug investigation | Omega |
| **II-6.2** | Stage II-0b advisory failures resolution | Omega |
| **II-6.3** | Held-out rolling evaluation of Path A winners | Omega + Gamma |
| **II-6.4** | Held-out rolling evaluation of Path B winner | Dragon |
| **II-6.5** | Deflated Sharpe Ratio + final verdict | Omega |

Each task has its own deliverable and gate. User reviews between tasks.

---

## 3. Task II-6.1: A4 Equity Bug Investigation

### 3.1 Purpose

Stage II-3 reported A4 (regime_adaptive_gmm yearly) produced $52.1M final equity from 2 trades on a $10K initial. This is a 5200× return that Stage II-3 attributed to "a compounding bug in the regime plugin."

Before any further experiments, understand what happened. If the orchestrator or position-sizing logic has a bug that inflates equity by 1000×, the bug may be affecting other experiments (less dramatically, harder to detect).

### 3.2 Procedure

1. Locate A4 experiment logs (`logs/<A4_experiment_id>/`)
2. Extract the 2 trades: entry price, exit price, position size, P&L per trade
3. Compute expected equity trajectory from these trades:
   - $10,000 starting capital
   - Position size per trade (should be 1× capital or configured fraction)
   - Realistic P&L contribution per trade
4. Compare to actual logged equity
5. If divergence >10%, identify source:
   - Position size calculation wrong?
   - P&L formula wrong?
   - Compounding applied incorrectly (geometric vs arithmetic)?
   - Leverage or contract size applied twice?
   - Currency/pip conversion error specific to BTC?

### 3.3 Scope of investigation

- Review `heuristic-strategy/plugins/regime_adaptive_gmm/` plugin code
- Review rolling orchestrator equity tracking (`infrastructure/rolling_orchestrator.py`)
- Check if same bug exists in other plugins (btc_momentum, regime_wfo)

### 3.4 Actions on finding bug

- **If bug isolated to regime_adaptive_gmm plugin:** document, confirm other experiments unaffected, proceed.
- **If bug in orchestrator or shared code:** STOP. Re-examine A1, A2, A6, B3 results with corrected code. May invalidate "passing" verdicts.
- **If bug is in P&L calculation broadly:** Stage II-6 halts. User decides on full re-run with fixed code.

### 3.5 Deliverable II-6.1

`TASK_II-6.1_A4_BUG_INVESTIGATION.md`:
- Trade-by-trade reconstruction
- Root cause identification
- Scope assessment (isolated vs shared code)
- Recommendation: proceed, re-run subset, or halt

### 3.6 Gate

Bug understood. Scope of impact known. User approves continuation or requires re-run.

---

## 4. Task II-6.2: Advisory Failures Resolution

### 4.1 Purpose

Stage II-0b marked EUR/USD 4h and USD/JPY daily with "advisory failures" on Test 5 (return autocorrelation). These were accepted without investigation. Since these assets are γ-classified (no Path A/B planned on them), the failures don't block Part II-Redux. But they matter for:

- Future cross-asset work in Part III
- Confidence in data validation process generally
- Understanding whether threshold calibration was wrong or data has real issue

### 4.2 Procedure

1. Inspect raw EUR/USD 4h bars around periods where ACF(1) was near-zero
2. Check for aggregation errors (bars with artificial zero returns from resampling)
3. Compare EUR/USD 4h to EUR/USD 1h ACF(1) — should be similar magnitude if real
4. Compare to known-good source (re-download EUR/USD 4h from OANDA if credentials arrive, cross-check)
5. Same for USD/JPY daily

### 4.3 Actions on finding

- **If data quality issue:** document, re-download affected periods, re-validate
- **If threshold calibration issue:** threshold too strict for intermediate timeframes; document but keep original for future
- **If genuine EMH-consistent data:** confirm and close the advisory

### 4.4 Deliverable II-6.2

`TASK_II-6.2_ADVISORY_RESOLUTION.md`:
- Diagnostic findings per asset
- Root cause (data issue / threshold / genuine)
- Action taken (re-download / accept / adjust)

### 4.5 Gate

Advisory failures resolved or explicitly accepted with evidence.

---

## 5. Task II-6.3: Path A Held-Out Rolling Evaluation

### 5.1 Purpose

Evaluate Path A "passing" experiments (A1, A2, A6) on held-out 2020-2025 BTC/USD 4h data using rolling retraining that mirrors how they would be deployed in practice.

### 5.2 Procedure

For each of A1, A2, A6:

1. **Starting point:** parameters the GA would have found using data up to 2019-12-31 only
2. **Retraining schedule:** per experiment's original configuration
   - A1 yearly: retrain using data through year N-1, evaluate on year N, for N ∈ {2020, 2021, 2022, 2023, 2024, 2025}
   - A2 monthly: retrain monthly, evaluate monthly, for all months 2020-01 through 2025-12
   - A6 yearly HPO: same as A1
3. **Protocol at each retraining:**
   - Training data: expanding window ending at retraining date (earliest 2017-08 through current-1 day)
   - Validation window: last year before test
   - Test window: next period (month or year per experiment)
   - Embargo: 6 bars between train and validation
4. **Cost model:** same 10 bps round-trip as IS
5. **Metrics per test period:** Sharpe, trades, max DD, cost ratio, equity

### 5.3 Pre-registered kill criteria (F-10)

Applied to held-out evaluation:

| Criterion | Threshold | If fail |
|-----------|-----------|---------|
| K-1: Held-out aggregate Sharpe | > 0 | Experiment fails held-out |
| K-2: Worst 2-year rolling Sharpe in HO | > -0.9 | Experiment fails K-2 |
| K-3: HO cost ratio | ≥ 2.0 | Experiment fails K-3 |
| K-5: HO window consistency | ≥ 60% | Experiment fails K-5 |
| K-7: Adaptive HO Sharpe > static HO Sharpe + 0.15 | p < 0.10 | Experiment fails K-7 |

### 5.4 Comparison

Results table for held-out:

| Exp | IS Agg SR | HO Agg SR | IS Consistency | HO Consistency | IS MaxDD | HO MaxDD | HO Verdict |
|-----|-----------|-----------|----------------|----------------|----------|----------|------------|
| A1 | +0.155 | ? | 100% | ? | 18.5% | ? | ? |
| A2 | +0.156 | ? | 67% | ? | 11.7% | ? | ? |
| A6 | +0.142 | ? | 100% | ? | 22.8% | ? | ? |

Divergence analysis:
- Expected similar pattern if strategy has genuine edge
- Large IS-HO gap indicates overfitting or regime change

### 5.5 Machine assignment

- Omega: A1 and A6 (yearly, 6 retrainings each — light compute)
- Gamma: A2 (monthly, 72 retrainings over 6 years — heavier but CPU-bound GA)

### 5.6 Deliverable II-6.3

`TASK_II-6.3_PATH_A_HELD_OUT.md`:
- Per-experiment held-out metrics vs IS metrics
- K-1 through K-7 evaluation on held-out
- IS-HO divergence analysis per experiment
- Updated verdict per experiment: HELD_OUT_PASS / HELD_OUT_FAIL

### 5.7 Gate

Held-out evaluation complete for all Path A winners. Results documented.

---

## 6. Task II-6.4: Path B Held-Out Rolling Evaluation

### 6.1 Purpose

Same as II-6.3 but for B3 (TFT regression), the single Path B experiment that passed IS kill criteria.

### 6.2 Procedure

1. **Starting point:** TFT model trained on 2017-08 through 2019-12 data
2. **Retraining schedule:** per B3's original configuration (yearly)
3. **Protocol at each retraining:**
   - Training data: expanding window ending at retraining date
   - Validation: last year before test, with 6-bar embargo
   - Test: next year
   - Model trained from scratch per window (not warm-start, matches original B3 protocol)
4. **Signal-to-trade:** Same ls_pred_strategy framework as original
5. **Cost model:** 10 bps round-trip

### 6.3 Pre-registered kill criteria

Same K-1 through K-7 as II-6.3. Additionally for regression ML:
- K-4 F1 ≥ 0.91 not applicable (B3 is regression, not binary)
- K-6: Train-test MAE ratio < 3× in > 50% of held-out windows

### 6.4 Comparison

| Metric | IS (W1 2018 + W2 2019) | HO (2020-2025) |
|--------|------------------------|-----------------|
| Aggregate Sharpe | +0.024 | ? |
| Mean per-window Sharpe | +0.024 | ? |
| Max DD | 30.8% | ? |
| Total trades | 95 | ? |
| Cost ratio | 4.75 avg | ? |
| Window consistency | 100% (2/2) | ? (6/6 possible) |

### 6.5 Machine assignment

- Dragon: TFT training per year (GPU-heavy, ~20-30 min per window per model, 6 windows = ~2-3 hours)

### 6.6 Deliverable II-6.4

`TASK_II-6.4_PATH_B_HELD_OUT.md`:
- Held-out metrics per window
- K-1, K-2, K-3, K-5, K-6, K-7 evaluation
- IS-HO divergence analysis
- Updated verdict: HELD_OUT_PASS / HELD_OUT_FAIL

### 6.7 Gate

Held-out evaluation of B3 complete.

---

## 7. Task II-6.5: Deflated Sharpe Ratio + Final Verdict

### 7.1 Purpose

Compute DSR accounting for multiple testing, produce final held-out verdict for Part II-Redux, and recommend Part III scope.

### 7.2 DSR computation

Per F-10 §5.2 and Bailey-López de Prado (2014):

**Inputs:**
- Total experiments in Part II-Redux: 17 (7 Path A + 10 Path B)
- Best observed aggregate Sharpe (IS or HO, whichever context)
- Returns distribution skewness and kurtosis
- Length of return series

**Formula (simplified):**

```
DSR = ( SR_observed - E[max(SR)] ) / σ(SR)
```

where E[max(SR)] is the expected maximum Sharpe when N=17 strategies are all null, computed as:

```
E[max(SR)] ≈ σ(SR) × √(2 ln(N))
```

adjusted for non-normality of returns.

**Output:** DSR with 95% confidence interval. Interpretation:
- DSR > 0 with CI > 0: strategy is genuinely profitable, not luck
- DSR > 0 but CI includes 0: marginal, cannot claim
- DSR < 0: strategy is consistent with luck given multiple testing

### 7.3 Compute DSR for

- A2 (best Path A by statistical evidence) on both IS and HO
- B3 (only passing Path B) on both IS and HO

### 7.4 Final verdict matrix

| Strategy | IS Sharpe | HO Sharpe | DSR | Verdict |
|----------|-----------|-----------|-----|---------|
| A2 btc_momentum monthly | +0.156 | ? | ? | ? |
| B3 TFT regression | +0.024 | ? | ? | ? |

Four possible verdicts per strategy:

1. **HELD_OUT_VALIDATED_DSR_POSITIVE:** HO Sharpe > 0, DSR CI > 0. Genuine edge.
2. **HELD_OUT_VALIDATED_DSR_MARGINAL:** HO Sharpe > 0, DSR CI includes 0. Edge exists but not distinguishable from luck.
3. **HELD_OUT_FAILED:** HO Sharpe ≤ 0 or kill criteria trigger. Previous IS "pass" was overfitting.
4. **INFRASTRUCTURE_INVALID:** A4 bug or similar compromises methodology.

### 7.5 Part III recommendations based on verdict

- **Both A2 and B3 validated:** Part III proceeds with full ambition — Path C (RL), ensembles (A+B), synthetic augmentation, expansion to Stage II-0.5 β candidates if any emerge
- **Only A2 validated:** Part III focuses on Path A refinement, multi-asset expansion within Path A, possibly Path C-light (hybrid heuristic-RL). Path B deprioritized.
- **Only B3 validated:** Part III focuses on ML refinement, ensemble with A as confirmation filter, Path C with supervised learning state.
- **Neither validated:** Part II-Redux closes as null. Same as Project 1 outcome. Part III scope is fundamental reconsideration: different asset universe, different approach paradigm, or project closure.
- **Infrastructure invalid:** Everything halts until infrastructure fixed, then re-run affected experiments.

### 7.6 Deliverable II-6.5

`TASK_II-6.5_FINAL_VERDICT.md`:
- DSR computations with full methodology
- Final verdict per strategy
- Part III scope recommendation
- Any outstanding issues or caveats
- Honest comparison IS vs HO for A2 and B3

### 7.7 Gate

Final verdict rendered. User reviews. Decision on Part III proceed/halt/redirect.

---

## 8. Part II-Redux Final Synthesis (Updated)

Overwrite or supersede existing `PART_II_REDUX_FINAL_SYNTHESIS.md` with new version that:

- Integrates Task II-6.5 held-out findings
- Presents IS vs HO comparison as primary evidence
- Applies DSR-adjusted conclusions
- Documents A4 bug resolution
- Renders honest verdict based on held-out, not IS

If verdict is HELD_OUT_FAILED for both, the synthesis honestly documents this as null result matching Project 1 pattern.

If verdict is validated, synthesis provides substantive basis for Part III ambition.

---

## 9. Machine Assignment Summary

| Task | Omega | Dragon | Gamma |
|------|-------|--------|-------|
| II-6.1 | A4 bug investigation | idle | idle |
| II-6.2 | Advisory resolution | idle | idle |
| II-6.3 | A1, A6 HO (yearly) | idle | A2 HO (monthly) |
| II-6.4 | idle | B3 HO (TFT training) | idle |
| II-6.5 | DSR + synthesis | idle | idle |

Dragon is idle for most tasks because II-6.3 is CPU-bound GA. II-6.4 is the only GPU-heavy task.

---

## 10. Dependency Graph

```
II-6.1 A4 Bug Investigation
   │
   ├── Bug isolated → proceed
   └── Bug pervasive → ESCALATION, halt
   │
   ▼
II-6.2 Advisory Resolution (parallel with II-6.3 start)
   │
   ▼
II-6.3 Path A Held-Out Evaluation (Omega A1/A6 + Gamma A2)
   │
   ▼ (parallel after II-6.3 Dragon load)
II-6.4 Path B Held-Out Evaluation (Dragon B3)
   │
   ▼
II-6.5 DSR + Final Verdict (Omega)
   │
   ├── Both validated → Part III with full ambition
   ├── One validated → Part III with narrowed scope
   ├── Neither validated → Part II-Redux null closure
   └── Infrastructure invalid → halt, re-run affected
```

---

## 11. Standing Rules Preserved

1. F-10 kill criteria applied as pre-registered (no modification)
2. Held-out 2020-2025 data touched exactly once per experiment at final evaluation
3. No synthetic fallback under any circumstance
4. Each task has user gate before next task
5. Escalation protocol active
6. Agent contract from Part II-Redux §2 still applies

---

## 12. Honest Acknowledgments

1. **This stage should have been part of Part II-Redux original plan.** Held-out evaluation is not an afterthought — it is the point. I failed to specify it explicitly in the stage sequence, and the agent (reasonably) interpreted "Path A execution" as IS evaluation only. My plan design error.

2. **The probability that A2 and B3 held-out results are disappointing is substantial.** Project 1 pattern: IS positive, HO negative. Project 2 Part II-Redux IS pattern looks similar. I estimate ~40-50% probability that HO evaluation reveals Sharpe close to zero or negative for both.

3. **DSR penalty will substantially reduce claimed Sharpes.** With 17 experiments tested, multiple-testing adjustment is material. A raw +0.156 may become DSR 0.05-0.10, at the edge of significance.

4. **A4 bug investigation may reveal larger issues.** A $52M equity from 2 trades is not a small bug. The investigation may reveal that equity tracking, position sizing, or compounding is systematically wrong across experiments. In that case, everything re-runs.

5. **Good outcome of this stage is evidence-based conclusion.** Whether outcome is "Path A is viable" or "Path A also failed held-out," both are valuable. The bad outcome would be skipping this stage and proceeding to Part III on false confidence.

6. **This stage takes finite time, unlike previous stages.** II-6.1 A4 investigation is 2-4 hours typically. II-6.3 A2 monthly is ~72 retrainings which depends on GA time per retraining. II-6.4 B3 TFT training is GPU-bound, maybe 3-4 hours. II-6.5 DSR computation is minutes. Whole stage plausibly 1-2 days of compute.

---

## 13. Immediate Next Actions

**For user:**

1. Approve this plan
2. Start agent: "Execute Stage II-6. Begin with Task II-6.1 A4 bug investigation. Report findings before proceeding to II-6.2."
3. Review each task deliverable at its gate before approving next task

**For agent at II-6.1 start:**

1. Locate A4 experiment logs
2. Reconstruct trades and equity trajectory
3. Investigate root cause in plugin and orchestrator code
4. Assess scope (isolated bug vs shared infrastructure bug)
5. Produce `TASK_II-6.1_A4_BUG_INVESTIGATION.md`
6. Halt at user gate

---

## 14. What Good Looks Like

Best outcome of Stage II-6:

- A4 bug understood, isolated to regime_adaptive_gmm plugin, does not affect other experiments
- Advisory failures explained (genuine EMH-consistent intermediate-timeframe near-zero autocorrelation, not data corruption)
- A2 held-out Sharpe positive (even if modest), consistency >50% across 6 years of held-out
- B3 held-out Sharpe positive, consistency decent
- DSR for A2 and B3 shows CI > 0 at 95%
- Verdict: genuine edge exists, Part III proceeds with confidence

Worst outcome:

- A4 bug is in shared infrastructure — invalidates other experiments
- A2 held-out Sharpe is near zero or negative — IS was overfitting to 2017-2019 regime
- B3 held-out collapses similarly
- DSR CIs span zero or negative
- Verdict: Part II-Redux is Project 1 redux. Null result. Project 2 closes or redirects fundamentally.

Either outcome is valuable scientific result. The bad scenario is not running this stage at all and building Part III on IS-only evidence.

---

## 15. Approval

User reviews plan, approves, and starts agent on Task II-6.1.