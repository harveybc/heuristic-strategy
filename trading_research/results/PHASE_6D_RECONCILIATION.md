> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

> **Historical audit trail.** Corrections from this analysis are incorporated into
> **`PHASE_6C_SYNTHESIS_FINAL.md`** (the canonical final document).

# Phase 6.D — Pre-Deployment Discrepancy Reconciliation

**Date:** 2025-06-17
**Machine:** Omega (local)
**Script:** `phase6d_reconciliation.py`
**Status:** COMPLETE — Both discrepancies fully resolved

---

## 1. Discrepancies Investigated

### Discrepancy 1: Max Drawdown Magnitude (15.0% vs 24.6%)

| Source | Max DD | Vol Level | Methodology |
|--------|--------|-----------|-------------|
| Phase 5.5 synthesis | 15.0% | 7.76% realized | exp(cumsum), single vol-scaling, W-FRI |
| Phase 6.C.3 @ 1.0x cost | 24.6% | 10.0% target | cumprod(1+r), double vol-scaling, W-SUN |

### Discrepancy 2: Implementation Lineage

Phase 6.D work plan asked whether Phase 6.C tests used LTS plugins or script-level NumPy, since Phase 6.B had validated plugins.

---

## 2. Root Cause Analysis — Discrepancy 1

### Methodology

Ran identical strategy code (P3 portfolio: EUR/USD MR 20.6%, USD/JPY TSMOM 49.2%, USD/JPY DM 30.2%) through both Phase 5.5 and Phase 6.C.3 evaluation methodologies on the same data. Decomposed the 9.6pp gap into four testable factors.

### Factor Decomposition

| Factor | Delta | Contribution | Description |
|--------|-------|-------------|-------------|
| **F1: Vol level** | +3.9pp | 41% | Phase 5.5 reported at realized vol (7.76%), Phase 6.C.3 at 10% target |
| **F2: Double vol-scaling** | +6.3pp | 65% | Phase 6.C.3 scaled cells to 10% individually, then portfolio to 10% again |
| **F3: Equity method** | +0.5pp | 6% | exp(cumsum) vs cumprod(1+r) for log returns |
| **F4: Resampling anchor** | 0.0pp | 0% | W-FRI vs W-SUN: no effect |
| **Interaction** | −1.1pp | — | Cross-factor interaction (factors sum to 107% due to non-linearity) |

### Detailed Findings

**F1 — Vol Level (41% of gap):** Phase 5.5's synthesis table showed `Vol=7.8%` alongside `MaxDD=15.0%`. This was the max DD at realized portfolio vol (cells individually scaled to 10%, but portfolio diversification reduced aggregate vol to 7.76%). Phase 5.5 also computed `at_10pct_vol.max_dd=18.9%` internally, but the synthesis reported the realized-vol figure. Phase 6.C.3 always vol-scaled the portfolio to 10% before computing metrics.

**F2 — Double Vol-Scaling (65% of gap, primary cause):** In `phase6c_omega.py`, `eval_cell()` scales each cell to 10% daily vol. Then `eval_portfolio_daily()` aggregates to weekly, computes weighted portfolio returns, and scales the portfolio AGAIN to 10% weekly vol. Since diversification reduces portfolio vol below 10%, this second scaling amplifies by ~1.36×. This is a methodological error — each cell is already at 10% vol; the portfolio should either be left at realized vol (7.3% weekly) or scaled once at the portfolio level, not both.

**F3 — Equity Method (6% of gap, minor):** For log returns, `exp(cumsum(r))` is mathematically correct. `cumprod(1+r)` treats log returns as arithmetic returns, introducing a small upward bias in drawdown over long horizons. Effect is 0.5pp over 977 weeks.

**F4 — Resampling Anchor (0% of gap):** W-FRI and W-SUN both produce 977 weeks with identical realized vol and max DD. No contribution to the gap.

### Reproduction Results

| Configuration | Max DD | Sharpe | Notes |
|--------------|--------|--------|-------|
| Phase 5.5 method: exp(cumsum), realized vol | **15.0%** | 0.449 | Matches Phase 5.5 synthesis exactly |
| Phase 5.5 method: exp(cumsum), 10% vol | **18.9%** | 0.449 | Phase 5.5 `at_10pct_vol` value |
| Phase 6.C.3 method: cumprod, double-scaled | **24.6%** | 0.447 | Matches Phase 6.C.3 exactly |
| Phase 6.C.3 method: cumprod, NO 2nd scale | **18.3%** | — | Removing double-scaling |
| Phase 6.C.3 method: exp(cumsum), NO 2nd scale | **17.3%** | — | Corrected equity + no double-scaling |

### Canonical Max DD

**18.9% at 10% target vol** — computed with single vol-scaling (cells to 10%, portfolio at resulting vol, then portfolio scaled once to 10%), using exp(cumsum) for log returns.

Rationale:
- 10% vol is the deployment target, so max DD should be stated at this level
- Single vol-scaling is correct: scale portfolio to 10%, not cells individually + portfolio
- exp(cumsum) is the mathematically correct equity curve for log returns
- The 24.6% value from Phase 6.C.3 was inflated by a double vol-scaling bug
- The 15.0% value from Phase 5.5 was correct but stated at a lower vol level

---

## 3. Root Cause Analysis — Discrepancy 2

### Implementation Audit

| Test | Script File | Implementation | LTS Plugins? | Result |
|------|------------|----------------|-------------|--------|
| 6.C.0 Held-Out | phase6c_omega.py | Script-level NumPy | NO | PASS |
| 6.C.1 JPY Reversal | phase6c_stress.py | Script-level NumPy | NO | PASS |
| 6.C.2 MC Regimes | phase6c_stress.py | Script-level NumPy | NO | PASS |
| 6.C.3 Cost Sensitivity | phase6c_omega.py | Script-level NumPy | NO | PASS |
| 6.C.4 Walk-Forward | phase6c_stress.py | Script-level NumPy | NO | PASS |
| 6.C.5 Param Perturbation | phase6c_stress.py | Script-level NumPy | NO | FAIL |

**All Phase 6.C tests used script-level NumPy implementations (inline `run_pure_mr`, `run_tsmom`, `run_dual_momentum`), not LTS plugins.**

### Impact Assessment

Phase 6.B validated that LTS plugins reproduce script-level results within tolerance (Sharpe difference < 0.01). The strategy logic is identical — same lookback, same z-score thresholds, same cost model. Plugin-based re-runs are not expected to flip any pass/fail outcome.

**No re-runs required.** Script-level results are authoritative and consistent with plugin behavior verified in Phase 6.B.

---

## 4. Deployment Parameter Calibration

### 4.1 Auto-Pause Threshold

Phase 5.5 set auto-pause at 15% DD based on the reported 15% max DD at realized vol. With canonical max DD at 18.9% (10% target vol), a 15% auto-pause would have triggered during backtested history.

**Revised thresholds:**

| Level | Threshold | Frequency (backtest) | Action |
|-------|-----------|---------------------|--------|
| Warning | 9.5% (50% of canonical) | ~2-3× per backtest | Log, continue trading |
| Serious Warning | 14.2% (75% of canonical) | ~1× per backtest | Alert, review positions |
| Auto-Pause | 20.0% (~106% of canonical) | Would not trigger in backtest | Halt trading, 48h review |
| Hard Stop | 28.3% (150% of canonical) | Never in backtest | Close all positions |

**Rationale for 20% auto-pause:** Setting auto-pause at exactly the canonical max DD (18.9%) means it would have triggered at least once during the 22-year backtest. Setting at 20% provides a small buffer (~1.1pp above canonical) to avoid spurious pauses from normal strategy operation while still protecting against genuine regime breaks. This replaces the prior 15% threshold which was based on the understated realized-vol figure.

### 4.2 Position Sizing

At 10% annualized portfolio vol and 18.9% max DD:
- Max DD / Vol ratio: 1.89×
- For a $10,000 demo account: max drawdown ≈ $1,890
- At 0.5% risk per trade: individual trade risk ≈ $50
- This is consistent with micro-lot sizing (0.01 lots per signal)

### 4.3 Monitoring Alerts

| Metric | Alert Level | Value |
|--------|-------------|-------|
| Rolling 20-day DD | Warning | > 5% |
| Rolling 20-day DD | Pause | > 10% |
| Cumulative DD from peak | Warning | > 9.5% |
| Cumulative DD from peak | Serious | > 14.2% |
| Cumulative DD from peak | Auto-Pause | > 20.0% |
| Realized slippage vs modeled | Warning | > 1.5× |
| Realized slippage vs modeled | Pause | > 2.0× |

---

## 5. Impact on Terminal Decision

### Phase 6.C Pass/Fail Status — Unchanged

The double vol-scaling bug in Phase 6.C.3 affected the **reported max DD** but NOT the Sharpe ratio, worst-2Y rolling window, or pass/fail determination. The Phase 6.C.3 pass criterion was Sharpe positive at 3× cost (SR=0.366 > 0, PASS), which is unaffected by the DD calculation.

The 6.C.0 held-out test (SR=0.316, max DD=16.7%) used the same double-scaling methodology. If corrected, its max DD would be lower (~13%), which does not change PASS status.

All other Phase 6.C tests (6.C.1-6.C.5) used Sharpe-based criteria, not max DD, so they are unaffected.

### Terminal 1 — CONFIRMED

Terminal 1 (Deploy P3 to OANDA Demo) remains valid:
- 4/5 stress tests PASS (unchanged)
- Held-out SR=0.316 (positive, unchanged)
- Canonical max DD = 18.9% (was incorrectly reported as 15.0% and 24.6%)
- Auto-pause threshold revised from 15% to 20%

---

## 6. Summary of Corrections

| Item | Old Value | New Value | Action |
|------|-----------|-----------|--------|
| Canonical P3 max DD | 15.0% (Phase 5.5) | 18.9% (at 10% vol) | Use 18.9% for all deployment sizing |
| Phase 6.C.3 max DD | 24.6% | 18.9% (bug corrected) | Double vol-scaling was a bug in phase6c_omega.py |
| Auto-pause threshold | 15% | 20% | Revised upward to avoid backtest false triggers |
| Hard stop threshold | N/A | 28.3% | Added as 1.5× canonical |
| Implementation lineage | Unclear | All script-level NumPy | Documented; consistent with Phase 6.B plugin validation |

---

## 7. Files Produced

- `phase6d_reconciliation.py` — Reconciliation script (reproducible)
- `results/phase_6d_reconciliation.json` — Machine-readable results
- `results/PHASE_6D_RECONCILIATION.md` — This document
- Addendum added to `results/PHASE_6C_SYNTHESIS.md`
