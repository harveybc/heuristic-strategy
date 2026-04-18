> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

> **Historical audit trail.** Corrections from this report are incorporated into
> **`PHASE_6C_SYNTHESIS_FINAL.md`** (the canonical final document).

# Phase 6.D.1 — Bug Fix Execution Report

**Date:** 2025-07-15
**Status:** COMPLETE
**Outcome:** Bug fixed, all tests re-run, no pass/fail changes

---

## 1. Critical Analysis of Work Plan

### Issues Identified in PHASE_6d1.md

1. **Option 2 (recommended) was wrong.** The plan recommended "no cell scaling, only portfolio scaling." This would change cell risk contributions vs Phase 5.5. The correct fix is: **keep cell-level vol-scaling, remove portfolio-level re-scaling** — matching Phase 5.5's established methodology where cells are individually normalized to 10% vol for equal risk contribution, and the portfolio is reported at realized vol.

2. **Underestimated scope.** The plan marked stress tests (6.C.1, 6.C.2) as "verify" status, but code reading confirmed `run_p3_on_returns()` has the identical double-scaling bug. All tests using this function were affected.

3. **6.C.4 partial bug missed.** Walk-forward test uses `eval_cell_returns()` (cell scaling) but builds portfolio manually without portfolio-level re-scaling. Only the `cumprod(1+r)` equity bug applied, not the double scaling.

4. **6.C.5 correctly identified as unaffected.** Parameter perturbation uses per-cell metrics with relative plateau fractions — absolute scaling doesn't change pass/fail.

### Self-Critique of Phase 6.D

- Phase 6.D identified the bug but didn't fix it — unacceptable for pre-deployment work
- The "canonical 18.9%" was computed in a standalone reconciliation script, never verified by running the actual corrected pipeline
- The week count discrepancy (977 vs 1168) was visible in Phase 6.D's own output JSON but not mentioned in the report

---

## 2. Bug Fix (6.D.1.1)

### Root Cause
Double vol-scaling: cell returns scaled to 10% vol individually in `eval_cell()` / `eval_cell_returns()`, then portfolio returns scaled to 10% vol again in `eval_portfolio_daily()` / `run_p3_on_returns()`. Additionally, equity curves used `np.cumprod(1 + r)` instead of `np.exp(np.cumsum(r))` (correct for log returns).

### Fix Applied

**`phase6c_omega.py`:**
- `eval_portfolio_daily()`: Removed `port_ret = port_ret * (0.10 / vol)`. Portfolio now reported at realized vol. Added `at_10pct_vol` sub-dict with leverage-adjusted metrics for deployment reference. Fixed equity to `np.exp(np.cumsum(port_ret))`.
- `run_p3_full()` per-cell metrics: Fixed `np.cumprod(1 + rets)` → `np.exp(np.cumsum(rets))`.
- `task_6c0()` and `task_6c3()` output: Added `at_10pct_vol` fields.

**`phase6c_stress.py`:**
- `run_p3_on_returns()`: Removed `port = port * (0.10 / vol)`. Added `at_10pct_vol` output. Fixed equity to `np.exp(np.cumsum(port))`.
- `task_6c4()`: Fixed `np.cumprod(1 + port_daily)` → `np.exp(np.cumsum(port_daily))`.

### What Was NOT Changed
- `eval_cell()` / `eval_cell_returns()`: Cell-level vol-scaling to 10% is correct and matches Phase 5.5. Kept as-is.
- Strategy implementations: No changes.
- P3 weight computation: No changes.

---

## 3. Week Count Discrepancy (6.D.1.3)

### Finding
The 977 vs 1168 week gap is a **methodological difference**, not a bug:

| Method | Resampling | Filter | Weeks | Period |
|--------|-----------|--------|-------|--------|
| Phase 5.5 | W-FRI | `valid >= 2` | 977 | 2004-02-06 to 2026-04-17 |
| Phase 6.C | W (Sunday) | none | 1168 | 2003-12-07 to 2026-04-19 |
| Phase 6.C + filter | W (Sunday) | `valid >= 2` | 977 | 2004-02-08 to 2026-04-19 |

The 191-week difference: 185 weeks where only 1 cell has non-zero returns (strategy warmup periods, DM flat periods) + 6 weeks where all cells are zero. Phase 6.C's approach is more conservative for deployment (includes all periods), while Phase 5.5 evaluates only periods with meaningful multi-cell diversification.

**Resolution:** Not a bug. Both are valid. Phase 6.C's inclusive approach is appropriate for stress testing.

---

## 4. Re-Run Results

### Before vs After Comparison

| Test | Machine | Old Result | New Result | Key Metric Changes |
|------|---------|-----------|------------|-------------------|
| 6.C.0 Held-out | Omega | PASS | **PASS** | maxDD: 16.7% → **13.6%** (realized vol 8.2%) |
| 6.C.1 JPY reversal | Gamma | PASS | **PASS** | median SR: 0.000 → 0.000, 25th: -0.187 → -0.187 |
| 6.C.2 MC regimes | Dragon | PASS | **PASS** | A: medSR 0.074→0.074, B: 0.022→0.022, C: 0.024→0.024 |
| 6.C.3 Cost | Omega | PASS | **PASS** | @1x maxDD: 24.6% → **17.3%** (realized vol 7.3%) |
| 6.C.4 Walk-forward | Omega | PASS | **PASS** | 55.8% positive, medQSR=0.265, streak=3 (identical) |
| 6.C.5 Params | Gamma | FAIL | **FAIL** | MR 57%, TSMOM 43%, DM 29% (identical) |

**Zero pass/fail flips.** As predicted: Sharpe ratios are scale-invariant, and drawdowns only decreased when the artificial portfolio-level inflation was removed.

### Corrected Canonical Metrics (Full Period @1x Cost)

| Metric | At Realized Vol (~7.3%) | At 10% Target Vol |
|--------|------------------------|-------------------|
| Sharpe | 0.4466 | 0.4466 |
| Max DD | 17.3% | 22.8% |
| Worst-2Y | -0.7243 | -0.7243 |
| Total Return | 108.8% | 172.7% |

**Note on canonical max DD:** The corrected Phase 6.C max DD at 10% vol is **22.8%**, not the 18.9% from Phase 5.5/6.D reconciliation. The difference is due to Phase 6.C including all 1168 weeks (no `valid >= 2` filter), which captures worse drawdown episodes during single-cell periods. For deployment risk limits:
- **Auto-pause (1.5x canonical):** 34.2% (at 10% vol)
- **Hard stop (conservative):** 25% (at realized vol)

### 6.C.0 Held-Out Detail

| Metric | Realized Vol | At 10% Vol |
|--------|-------------|------------|
| Sharpe | 0.3163 | 0.3163 |
| Max DD | 13.6% | 16.2% |
| Return | 5.4% | 6.6% |
| Vol | 8.2% | — |
| Leverage | — | 1.22x |

---

## 5. Files Modified

1. `trading_research/phase6c_omega.py` — Bug fix + at_10pct_vol output
2. `trading_research/phase6c_stress.py` — Bug fix + at_10pct_vol output
3. Deployed fixed `phase6c_stress.py` to Dragon (192.168.1.235) and Gamma (192.168.0.106) at `/tmp/trading_research/`

---

## 6. Conclusion

The double vol-scaling bug has been eliminated from the Phase 6.C codebase. All six stress tests have been re-run with corrected code. Pass/fail outcomes are unchanged (5 PASS, 1 FAIL on 6.C.5 parameter sensitivity). Max drawdown figures are now consistent with Phase 5.5 methodology: reported at realized vol, with at-10%-vol metrics available for deployment reference.

The only remaining known limitation is 6.C.5 FAIL (parameter sensitivity), which is a genuine property of the strategies — not a bug artifact.
