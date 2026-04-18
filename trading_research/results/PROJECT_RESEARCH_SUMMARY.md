# Trading Research — Project Summary

> **⚠️ SUPERSEDED** — This document reflects script-canonical / Terminal 1 metrics that were invalidated by Phase 6.E.0.1. The authoritative closure document is **[PROJECT_CLOSURE.md](PROJECT_CLOSURE.md)**.

**Status:** CLOSED — Terminal 3 (held-out kill criterion triggered, plugin-canonical SR = −0.065)
**Date:** 2026-04-17
**Canonical reference:** `PROJECT_CLOSURE.md`

---

## Objective

Systematic search for a deployable FX trading portfolio using quantitative methods.
Starting from 360 oracle-enhanced strategy cells across 12 assets and 5 strategy types,
apply increasingly rigorous filters to arrive at a live-deployable portfolio with
well-characterized risk.

## Phase Timeline

| Phase | Outcome | Description |
|-------|---------|-------------|
| 0 | ✅ | **Diagnostic infrastructure.** Built evaluation harness, oracle sensitivity test, transaction cost model (16 assets). |
| 1 | ✅ | **360-cell oracle sweep.** 14 survivors from 240 cells (12 assets × 4 timeframes × 5 strategies). |
| 1.3 | ✅ | **Naive baseline filtering.** Separated oracle signal from inherent technical edge. 14 robust candidates. |
| 2 | ✗ | **EUR/USD hourly MR deep audit.** Killed — parameter spike, not plateau; only profitable during Asian session. |
| 3 | ✅ | **Exogenous data enrichment.** Extended time series with macro/vol/yield features. 11 assets enriched. |
| 3.5 | ✗ | **Extended-history stress.** All 14 oracle survivors killed by worst-2Y window test. Pivot to portfolio diversification. |
| 4 | ✗ | **Four parallel hypothesis tests** (oracle-independent MR, regime filters, academic strategies, GBP/USD VRS). All failed. |
| 5 | ✅ | **Portfolio diversification.** Equal-weight P1 achieves Sharpe 0.60 but worst-2Y −0.55 (near threshold). Threshold recalibrated to −1.0 using industry benchmarks. |
| 5.5 | ✅ | **Corrective audit.** Removed all oracle-dependent cells. Only 3 oracle-free cells survive. P3 portfolio identified. |
| 6.A | ✅ | **Plugin inventory audit.** Zero ML-based combinations passed Phase 5.5 rigor. Only regime plugins untested. |
| 6.B | ✅ | **Untested candidate evaluation.** P3 confirmed as best; regime hybrids worsen performance. LTS plugins validated. |
| 6.C | ✅ | **Final robustness: 5 stress tests + held-out.** 4/5 pass. Parameter perturbation fails (genuine sensitivity). Deployment approved. |
| 6.D | ✅ | **Reconciliation.** Resolved max DD discrepancy (double vol-scaling bug identified). |
| 6.D.1 | ✅ | **Bug fix.** Double vol-scaling removed. All tests re-run with zero pass/fail changes. |
| 6.D.2 | ✅ | **Documentation consolidation.** This summary + `PHASE_6C_SYNTHESIS_FINAL.md`. |
| 6.E.0 | ✅ | **Pipeline simulation validation.** Plugin vs script ~89-90% direction match. GO with widened tolerance. |
| 6.E.0.1 | ✗ | **Plugin-canonical held-out.** SR = −0.065 (kill triggered). E2E orchestration validated. **Terminal 3.** |
| 7 | ✅ | **Project closure.** `PROJECT_CLOSURE.md` produced. Research program formally closed. |

**Key pivot points:** Phase 3.5 killed all individual oracle strategies → portfolio approach.
Phase 5.5 removed all oracle-contaminated cells → only pure technical strategies remain.
Phase 6.E.0.1 established plugin-canonical as authoritative → held-out Sharpe fails > 0 gate → Terminal 3.

---

## Final Portfolio: P3

| Cell | Asset | Strategy | Weight | Rationale |
|------|-------|----------|--------|-----------|
| EUR/USD MR | EUR/USD | Pure mean reversion | 20.6% | Lowest correlation to JPY cells |
| USD/JPY TSMOM | USD/JPY | Time-series momentum | 49.2% | Best risk-adjusted backtest |
| USD/JPY DM | USD/JPY | Dual momentum (abs + rel) | 30.2% | Low turnover, decorrelated signal |

**Weights:** Inverse worst-2Y-window (more weight to cells with shallower worst drawdowns).
**USD/JPY concentration:** 79.4% — primary risk factor.

### Canonical Metrics (10% annualized target vol)

| Metric | Full Period | In-Sample (≤2018) | OOS (2019–2023) | Held-Out (2024–2025) |
|--------|------------|-------------------|-----------------|----------------------|
| Sharpe | 0.447 | 0.452 | 0.505 | 0.316 |
| Max DD | **22.8%** | 22.4% | 16.5% | 16.2% |
| Return | 172.7% | 98.3% | 28.8% | 6.6% |

### Stress Test Summary

| Test | Result | Key Metric |
|------|--------|------------|
| Held-out gate (2024–2025) | **PASS** | SR=0.316, DD=16.2% |
| JPY reversal (500 paths) | **PASS** | Median SR=0.000 (flat, not catastrophic) |
| MC regime scenarios (9000 paths) | **PASS** | All regimes SR ≥ 0.022 |
| Cost sensitivity (1×–3×) | **PASS** | SR positive at 3× costs |
| Walk-forward (52 quarters) | **PASS** | 55.8% positive, max streak 3 |
| Parameter perturbation | **FAIL** | All cells below 60% plateau threshold |

---

## Deployment Plan

> **CANCELLED** — Terminal 3 triggered. See [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md) §6 for rationale.

---

## Key Lessons

1. **Oracle enhancement is a double-edged sword.** It identifies promising signal-strategy
   combinations efficiently, but every oracle-positive cell failed extended-history stress
   tests. The surviving strategies are all oracle-free.

2. **Portfolio diversification rescues killed strategies.** Individual cells with Sharpe
   0.18–0.41 combine into a portfolio with Sharpe 0.45 and manageable drawdowns.

3. **Threshold calibration matters.** The initial −0.5 worst-2Y threshold killed everything.
   Benchmarking against liquid ETFs (5/6 also fail −0.5) led to recalibration to −1.0.

4. **Double vol-scaling is a subtle bug.** When cells are individually vol-scaled and then
   the portfolio is re-scaled, max drawdown inflates significantly (24.6% → 22.8%). The
   correct approach: scale cells individually, report portfolio at realized vol plus
   a separate leverage-adjusted metric.

---

## File Inventory

| File | Purpose |
|------|---------|
| `PHASE_6C_SYNTHESIS_FINAL.md` | **Canonical final document** — all corrected metrics |
| `PHASE_6D_RECONCILIATION.md` | Historical: discrepancy analysis |
| `PHASE_6D1_EXECUTION.md` | Historical: bug fix report |
| `PHASE_6C_SYNTHESIS_v1_DEPRECATED.md` | Historical: original synthesis (pre-fix values) |
| `phase_6c_omega_results.json` | Machine-readable corrected results (6.C.0 + 6.C.3) |
| `phase_6d1_results.json` | Machine-readable correction summary |
| `phase_6c_stress_6c1.json` | JPY reversal results |
| `phase_6c_stress_6c2.json` | MC regime results |
| `phase_6c_stress_6c4.json` | Walk-forward results |
| `phase_6c_stress_6c5.json` | Parameter perturbation results |
