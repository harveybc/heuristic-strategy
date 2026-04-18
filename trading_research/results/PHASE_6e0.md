# Work Plan: Phase 6.E.0 — Pipeline Simulation Validation

**Prior state:** Phase 6.D.2 complete. `PHASE_6C_SYNTHESIS_FINAL.md` established canonical metrics. Phase 6.E operational deployment plan written, but assumed direct transition from script-level validation to live OANDA demo. Review identified that Phase 6.C tests used script-level NumPy implementations, not the real `lts + backtrader_simulation_broker` pipeline. This creates a gap: portfolio-level mechanics, order execution, SL/TP interactions, monthly rebalance logic, and concurrent cell execution were never verified end-to-end through the actual deployment pipeline.

**Purpose:** Run P3 portfolio through the full `lts + backtrader_simulation_broker` pipeline on the same historical data used in Phase 6.C. Verify that pipeline-level results match script-level results within tolerance. Validate portfolio-level mechanics (rebalance, concurrent execution, SL/TP, slippage) that only the real pipeline exercises. Identify any gaps before committing 90+ days of calendar time to live demo.

**Duration:** 3-5 days.

**Machine:** Primarily Omega (runs the full pipeline which requires the conda environment). Dragon and Gamma idle or available for stress scenario replays if parallelizable.

**Gate to Phase 6.E.1:** Pipeline produces results matching script-level within tolerance on full-period and held-out datasets, and mechanics tests pass without errors.

---

## Why This Phase Exists (Honest Framing)

Phase 6.B ported P3 strategies to LTS plugins and validated that plugins reproduce script-level results within tolerance — but the validation was limited: Sharpe delta < 0.01 on full-period IS. It did not test:

- Portfolio-level allocation across three cells running concurrently
- Monthly rebalance mechanics when positions need to flip
- SL/TP execution behavior in the broker simulator
- Slippage modeling in adverse scenarios
- Held-out period through the real pipeline (was script-level in Phase 6.C.0)
- Stress scenarios through the real pipeline

These gaps were not noticed in Phase 6.C because the stress tests were all done at script level for computational efficiency. The assumption was that if plugins match scripts on full-period Sharpe, they will match on everything. This assumption is plausible but untested.

Phase 6.E.0 verifies the assumption with direct evidence. If the assumption holds, Phase 6.E.1 proceeds with high confidence. If it does not hold, we discover execution gaps in simulation rather than in live demo where calendar time and real (though demo) PnL are at stake.

This is the same "simulate more before live" discipline that has served well throughout the project. One last 3-5 day check before committing 90+ days of calendar time.

---

## Critical Scope Boundaries

1. **No new strategy research.** Phase 6.E.0 only tests the existing P3 cells through the real pipeline. No new candidates, no parameter changes, no strategy modifications.

2. **No new canonical metric definitions.** The metrics from `PHASE_6C_SYNTHESIS_FINAL.md` (Sharpe 0.447 full-period, 0.316 held-out, max DD 22.8% at 10% vol, etc.) are the targets. The pipeline either reproduces them within tolerance or reveals a discrepancy requiring investigation.

3. **No new stress tests.** The stress scenarios are replays of existing scenarios (specifically the worst historical quarters identified in 6.C.4), run through the pipeline to verify mechanics, not to generate new performance numbers.

4. **Tolerance specified in advance.** Define acceptable deviation before execution:
   - Sharpe within ±0.05 of script-level
   - Max DD within ±3pp of script-level
   - Trade count within ±10% of script-level
   - Any deviation beyond tolerance requires investigation before proceeding

---

## Task 6.E.0.1 — Full-Period Backtest Through Pipeline (Day 1, Omega)

### Purpose
Confirm that `lts + backtrader_simulation_broker + P3 plugins` reproduces Phase 6.C canonical full-period metrics.

### Task 6.E.0.1.A — Pipeline configuration

Configure LTS:
- Broker: `backtrader_simulation_broker`
- Strategies: `eurusd_mr_strategy` + `usdjpy_tsmom_strategy` + `usdjpy_dual_momentum_strategy`
- P3 weights: 20.6% / 49.2% / 30.2%
- Target vol: 10% annualized at portfolio level
- Data: full history 2003-01-01 to 2023-12-31 (same as Phase 6.C.3)
- Cost model: 1.0× baseline (OANDA spread + 0.3 bps slippage per trade)
- Execution time: daily at 22:00 UTC

### Task 6.E.0.1.B — Execute full backtest

Run end-to-end. Capture:
- Portfolio equity curve (daily)
- Every trade with entry/exit/PnL/slippage
- Per-cell contribution
- Rebalance events for USD/JPY cells (monthly dates)

### Task 6.E.0.1.C — Compare to Phase 6.C.3 canonical

Produce comparison table:

| Metric | Phase 6.C.3 Canonical | Pipeline | Delta | Within Tolerance? |
|--------|----------------------|----------|-------|-------------------|
| Sharpe (full) | 0.447 | ? | ? | ±0.05 |
| Max DD @ 10% vol | 22.8% | ? | ? | ±3pp |
| Worst-2Y | −0.724 | ? | ? | ±0.10 |
| Total trades | ~varies per cell | ? | ? | ±10% |
| EUR/USD MR trades | ~varies | ? | ? | ±10% |
| USD/JPY TSMOM rebalances | 252 (21 years × 12) | ? | ? | exact |
| USD/JPY DM rebalances | 252 | ? | ? | exact |

### Task 6.E.0.1.D — Investigate any out-of-tolerance deltas

If any metric is outside tolerance:
- Identify the source: cost model difference, order execution timing, rebalance logic, slippage
- Document the gap
- Determine if it's a bug requiring fix or a legitimate pipeline-level effect
- If bug: fix before proceeding to Task 6.E.0.2
- If legitimate: document as "pipeline-level adjustment from script-level" in the final synthesis

### Deliverable 6.E.0.1
Comparison table with pass/fail on tolerance. List of any gaps requiring resolution.

---

## Task 6.E.0.2 — Held-Out Backtest Through Pipeline (Day 1, Omega)

### Purpose
Same as 6.E.0.1 but on held-out period 2024-2025. This is the most deployment-relevant dataset.

### Task 6.E.0.2.A — Execute held-out backtest through pipeline

Same configuration as 6.E.0.1 but restricted to 2024-01-01 to 2025-12-31.

### Task 6.E.0.2.B — Compare to Phase 6.C.0 held-out canonical

| Metric | Phase 6.C.0 Canonical | Pipeline | Delta | Within Tolerance? |
|--------|----------------------|----------|-------|-------------------|
| Sharpe | 0.316 | ? | ? | ±0.05 |
| Max DD @ 10% vol | 16.2% | ? | ? | ±3pp |
| Return | 6.6% | ? | ? | ±1pp |
| EUR/USD MR trades | 77 | ? | ? | ±10% |
| USD/JPY TSMOM rebalances | 25 | ? | ? | exact |
| USD/JPY DM rebalances | 24 (4 active) | ? | ? | exact |
| Per-cell Sharpe breakdown | as specified | ? | ? | ±0.10 each |

### Task 6.E.0.2.C — Per-cell attribution validation

For each cell, verify individual contribution to portfolio matches expected:
- EUR/USD MR: expected 1.9% return, 5.4% max DD in held-out
- USD/JPY TSMOM: expected 3.9% return, 18% max DD
- USD/JPY DM: expected 6.7% return, 20% max DD

If portfolio-level matches but per-cell attribution differs, the aggregation logic may have subtle issues.

### Deliverable 6.E.0.2
Held-out comparison table. Pipeline validated or gaps identified.

---

## Task 6.E.0.3 — Stress Scenario Replay (Day 2, Omega)

### Purpose
Run the worst historical quarters through the pipeline with realistic spreads and slippage. Verify mechanics in adverse conditions.

### Task 6.E.0.3.A — Identify stress scenarios from Phase 6.C.4 walk-forward

From the walk-forward results, the worst and most volatile periods were:
- 2016-Q1: −11.9% cumulative (BoJ negative rates surprise)
- 2022-Q4: −9.3% cumulative (BoJ policy reversal)
- 2016-Q1 full year context
- Any other identified drawdown periods from the 52 OOS quarters

### Task 6.E.0.3.B — Replay through pipeline with enhanced monitoring

For each stress period:
- Run pipeline over that period specifically
- Log every signal, every order submission, every fill
- Log slippage per trade
- Log any SL/TP hits
- Log any times when positions were flipped (for TSMOM/DM monthly rebalance with direction change)

### Task 6.E.0.3.C — Verify no execution failures

Confirm:
- All signals generated correct orders
- All orders executed successfully (no rejections, no partial fills unless realistic)
- SL/TP hit correctly when price crossed thresholds
- No orphan positions (positions that should have closed but didn't)
- No duplicate positions (orders that should have been rejected but weren't)
- PnL matches expectations for each event

### Task 6.E.0.3.D — Stress scenario comparison

| Scenario | Script Sharpe | Pipeline Sharpe | Script Max DD | Pipeline Max DD |
|----------|---------------|-----------------|---------------|-----------------|
| 2016-Q1 | from 6.C.4 | ? | derived | ? |
| 2022-Q4 | from 6.C.4 | ? | derived | ? |
| Other | from 6.C.4 | ? | derived | ? |

Tolerance same as prior tasks.

### Deliverable 6.E.0.3
Stress scenario validation showing pipeline handles adverse conditions correctly.

---

## Task 6.E.0.4 — Monthly Rebalance Mechanics Test (Day 3, Omega)

### Purpose
USD/JPY TSMOM and Dual Momentum rebalance monthly. Rebalance can involve:
- New signal matching current position: no change
- New signal opposite to current position: flip (close + open new)
- New signal going flat: close
- Previously flat going active: open

All four scenarios need validation that the pipeline handles correctly.

### Task 6.E.0.4.A — Identify rebalance scenarios

From history, find examples of each scenario for TSMOM and DM:
- **Same-direction rebalance:** position was long, new signal long. Typical case, should be no-op or minimal adjustment for vol-sizing.
- **Opposite-direction flip:** position was long, new signal short (or vice versa). Must close existing and open opposite.
- **Active to flat:** position was long, new signal flat (absolute momentum fails for DM). Close existing.
- **Flat to active:** position was flat, new signal long/short. Open new.

### Task 6.E.0.4.B — Replay rebalance days through pipeline

For each scenario type, select 2-3 historical examples. Run the specific rebalance day through the pipeline. Verify:
- Correct signal generated
- Correct orders submitted (close first, then open opposite, or just adjust, as appropriate)
- No overlap period where both old and new positions coexist incorrectly
- Correct PnL attribution (closing PnL + opening at new position)
- Correct position sizing at new position

### Task 6.E.0.4.C — Vol-sizing verification

For TSMOM, position sizing is inverse volatility. Verify:
- Trailing 60-day vol calculation matches expectation
- Position size = target_notional / (vol × multiplier)
- After size change at rebalance, capital allocation matches P3 weight

### Deliverable 6.E.0.4
Rebalance mechanics validation with specific dated examples. Any discrepancies between expected and actual rebalance behavior identified.

---

## Task 6.E.0.5 — Concurrent Execution Stress (Day 3-4, Omega)

### Purpose
Three cells run simultaneously. Normally signals are staggered (EUR/USD MR is event-driven, USD/JPY cells rebalance monthly). But there will be days when multiple cells signal at once. Test coordination.

### Task 6.E.0.5.A — Identify concurrent signal days

From Phase 6.C.3 full-history logs, find days where:
- EUR/USD MR signaled entry/exit AND one of USD/JPY cells rebalanced
- Both USD/JPY cells rebalanced on same monthly date (this is common — both are first-of-month)
- All three cells had activity (rare but possible)

### Task 6.E.0.5.B — Replay concurrent-activity days

For each concurrent-activity day, run through pipeline. Verify:
- Capital allocated correctly per P3 weights (no over-allocation, no under-allocation)
- Orders submitted without conflict
- "Single position per cell" constraint maintained
- Logging captures per-cell attribution correctly (each trade tagged to correct cell)
- No race conditions in order submission

### Task 6.E.0.5.C — Portfolio-level capital calculation test

When one cell has an open position and another needs to open, capital allocation math must be correct:
- Open position cell's allocated capital = locked (cannot be reused)
- New signal cell's allocated capital = available per P3 weight
- No double-counting

Run a test scenario (can be historical or constructed) where this math is exercised and verify correctness.

### Deliverable 6.E.0.5
Concurrent execution validation. Portfolio-level capital coordination confirmed correct.

---

## Task 6.E.0.6 — Execution Timing and Edge Cases (Day 4, Omega)

### Purpose
Test edge cases that don't appear in nominal backtests but might in live.

### Task 6.E.0.6.A — DST transition days

EUR/USD trading around DST transitions (US and EU DST changes at different weeks). Test:
- Signal generation on transition day
- Order execution timing
- No duplicate or missed signals due to UTC confusion

### Task 6.E.0.6.B — Holiday handling

Major holidays affect FX liquidity. Test scenarios:
- Signal generated on day before holiday
- Execution on thin-liquidity day
- Verify slippage modeling accounts for reduced liquidity

### Task 6.E.0.6.C — Weekend gap handling

FX closes Friday 22:00 UTC, opens Sunday 22:00 UTC. Test:
- If signal fires Friday, order placed and filled before close
- SL/TP during weekend gap: does backtrader_simulation_broker simulate gap-over-SL realistically?
- First trade of new week: initial conditions correct

### Task 6.E.0.6.D — Execution failure simulation

Simulate scenarios where an order might fail:
- Not enough margin
- Invalid price (off-market)
- Broker temporarily unavailable

Verify pipeline handles gracefully:
- Failed order is logged
- Strategy state remains consistent (no phantom position)
- Subsequent signals are processed normally (not blocked by earlier failure)

### Deliverable 6.E.0.6
Edge case handling documentation. Any identified robustness issues.

---

## Task 6.E.0.7 — Synthesis and Go/No-Go for Phase 6.E.1 (Day 4-5, Omega)

### Task 6.E.0.7.A — Aggregate all validation results

Compile master results table:

| Validation | Status | Notes |
|------------|--------|-------|
| Full-period metric match | ? | Pass/Fail tolerance |
| Held-out metric match | ? | Pass/Fail tolerance |
| Stress scenario replay | ? | No execution failures |
| Rebalance mechanics | ? | All 4 scenarios validated |
| Concurrent execution | ? | Coordination correct |
| Execution timing edge cases | ? | DST/holiday/weekend handled |

### Task 6.E.0.7.B — Go/No-Go decision

**GO criteria:**
- All metric comparisons within tolerance
- All mechanics tests pass
- Any identified gaps have been fixed or documented as known pipeline-level adjustments (not bugs)
- No critical unresolved issues

**NO-GO criteria:**
- Any metric comparison fails tolerance significantly (>2× tolerance bound)
- Any mechanics test reveals bug in pipeline
- Any execution failures in stress scenarios that were not handled gracefully

**Partial go (fix-and-retry):**
- Tolerance failures that can be traced to specific bugs
- Fix bug, rerun affected tests, re-evaluate

### Task 6.E.0.7.C — Produce synthesis document

`PHASE_6E0_SIMULATION_VALIDATION.md` containing:
- All comparison tables
- Stress scenario results
- Mechanics test outcomes
- Edge case handling
- Final Go/No-Go determination
- If GO: clearance to proceed to Phase 6.E.1 live deployment
- If NO-GO or Partial: specific issues to resolve before re-attempting

### Deliverable 6.E.0.7
Final validation document with explicit clearance (or blockers) for live deployment.

---

## Execution Schedule

```
Day 1:  6.E.0.1 full-period backtest (Omega, 2-4 hours)
        6.E.0.2 held-out backtest (Omega, 1-2 hours)
        Investigation of any tolerance failures (Omega, evening)

Day 2:  6.E.0.3 stress scenario replay (Omega)
        Parallel: document results from Day 1

Day 3:  6.E.0.4 rebalance mechanics (Omega)
        6.E.0.5 concurrent execution (Omega)

Day 4:  6.E.0.6 execution edge cases (Omega)
        6.E.0.7.A aggregate results (Omega)

Day 5:  6.E.0.7.B decision and 6.E.0.7.C synthesis (Omega)
        Handoff to Phase 6.E.1 or documentation of blockers
```

Wall clock: 3-5 days. Lower end (3 days) if everything validates cleanly. Upper end (5 days) if investigations are needed for tolerance failures.

Dragon and Gamma are largely idle during Phase 6.E.0. If stress scenario replays are parallelizable, they can be distributed, but for 3 stress periods this is probably not worth the setup overhead.

---

## Standing Rules for Phase 6.E.0

1. **Tolerance is pre-registered.** The ±0.05 Sharpe, ±3pp max DD tolerances are fixed before execution. Do not adjust after seeing results.

2. **Bug fixes require re-running affected tests.** If Task 6.E.0.1 reveals a bug and it is fixed, Task 6.E.0.1 must be re-run to confirm fix works. Same for any subsequent task.

3. **No new metric definitions.** The metrics being validated are those in `PHASE_6C_SYNTHESIS_FINAL.md`. No introducing new metrics during 6.E.0.

4. **Pipeline-level adjustments must be justified.** If metrics differ from script-level but within tolerance, document why. "Realistic slippage modeling" is acceptable. "Coincidence" is not.

5. **No-Go means investigate, not abandon.** If 6.E.0 produces No-Go, the project is not over. The issues are analyzed, fixed, and 6.E.0 is re-run. Phase 6.E.1 is blocked until 6.E.0 passes.

6. **Phase 6.E.0 is not expanded scope.** No new research. No new candidates. No new strategies. Pure pipeline validation of existing P3.

---

## Possible Outcomes

### Outcome A: Clean pass (most likely, ~70%)
- Metrics match script-level within tolerance
- All mechanics validate cleanly
- Edge cases handled properly
- Proceed to Phase 6.E.1 live deployment with high confidence

### Outcome B: Minor pipeline adjustments (plausible, ~20%)
- Metrics slightly differ due to realistic slippage or execution timing
- Within tolerance but pattern of difference observed
- Document as "pipeline produces slightly more conservative results than scripts due to realistic slippage"
- Proceed to Phase 6.E.1 with updated expectations

### Outcome C: Bug discovered requiring fix (possible, ~8%)
- Tolerance failure traced to bug in plugin or LTS configuration
- Fix applied, tests re-run
- After fix, results within tolerance
- Proceed to Phase 6.E.1 with documentation of fix

### Outcome D: Structural issue requiring deeper work (~2%)
- Pipeline cannot reproduce script-level results even after investigation
- Issue is deeper than a single bug: portfolio logic, rebalance logic, or something fundamental
- Phase 6.E.1 blocked indefinitely pending resolution
- May require Phase 6.B-style re-porting or significant LTS refactoring

Expected value of Phase 6.E.0: 3-5 days now saves potentially 2-8 weeks later if a real pipeline bug would have emerged during live demo.

---

## Honest Acknowledgments

1. **This phase should have been part of Phase 6.C or Phase 6.D.1.** Every time I've written "pipeline validated" in prior plans, it was based on Phase 6.B's limited validation (Sharpe match on full-period). The full portfolio-level validation through the real pipeline was never formally planned until now. This is the pattern we've seen throughout: gaps between "claimed validated" and "actually validated" that surface only on independent review.

2. **The 70% probability of clean pass is a guess.** It could be higher (plugins worked in Phase 6.B, so portfolio-level aggregation is probably fine) or lower (concurrent execution logic has never been tested end-to-end, could have subtle bugs). Until the tests run, uncertainty remains.

3. **Phase 6.E.0 completeness still has limits.** Even this phase cannot test live broker behavior (OANDA demo has different execution characteristics than backtrader_simulation_broker). Some validation can only happen in Phase 6.E.1 Stage 1 with actual OANDA practice account. But Phase 6.E.0 closes the biggest remaining simulation gap.

4. **Cost-benefit is favorable.** If Outcome A (70%): 3-5 days invested, minimal new info, but high confidence in pipeline. If Outcome C (8%): 3-5 days saves potentially weeks of live-demo debugging. If Outcome D (2%): 3-5 days reveals a blocker before committing to live. Only in Outcome B (20%) is the 3-5 days "mostly confirmation" but even there the confirmation has value.

5. **This is the last pre-deployment validation.** After Phase 6.E.0 passes, Phase 6.E.1 begins. No more simulation phases. Live demo is the next unit of evidence.

---

## Immediate Next Actions

1. Verify `backtrader_simulation_broker` is functional on Omega (may not have been used recently)
2. Confirm LTS configuration can be set up for simultaneous 3-plugin P3 portfolio
3. Prepare test data (full 2003-2025) in format expected by LTS pipeline
4. Begin Task 6.E.0.1 full-period backtest

Phase 6.E.0 begins today. Live deployment (Phase 6.E.1) begins after Phase 6.E.0 passes.