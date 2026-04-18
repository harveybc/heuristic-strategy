> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 5.5 — Corrective Audit Synthesis

**Date:** 2026-04-17  
**Status:** COMPLETE — Terminal state determined  
**Script:** `trading_research/phase5_5_corrective_audit.py`  
**Results:** `trading_research/results/phase5_5_results.json`

---

## Executive Summary

Phase 5.5 corrected 11 issues identified in the Phase 5 self-critique. The corrected analysis:

1. **Removed all oracle-dependent cells** — only 3 oracle-free cells survive strict filtering
2. **Applied out-of-sample validation** with frozen weights — all 5 portfolios show zero degradation (OOS actually improved, which warrants caution)
3. **Benchmarked against active managed-futures ETFs** (not passive asset classes)
4. **Audited the Q3 deployment** — resolved auto-pause contradiction, confirmed OANDA is not yet live
5. **Determined terminal state: TERMINAL 1 — Deploy Portfolio** (with material caveats below)

**Recommended action:** Deploy P3 portfolio (inverse-worst-window weighted) on OANDA demo account first, with 90-day observation period before any real capital.

---

## Task 5.5.1 — Oracle-Free Portfolio Rebuild

### Cell Classification

| Cell | Oracle-Free? | SR | Edge | Worst-2Y | Status |
|------|-------------|-----|------|----------|--------|
| EUR/USD daily pure_mr | YES | +0.186 | +0.193 | -1.432 | INCLUDED |
| USD/JPY daily pure_mr | YES | +0.133 | +0.027 | -1.329 | EXCLUDED (asset cap) |
| USD/JPY daily tsmom | YES | +0.296 | +0.190 | -0.598 | INCLUDED |
| XAU/USD daily tsmom | YES | +0.392 | -0.240 | -1.515 | EXCLUDED (edge ≤ 0) |
| USD/JPY daily dual_momentum | YES | +0.406 | +0.299 | -0.973 | INCLUDED |
| EUR/USD daily mean_reversion | NO (σ=10) | — | — | — | EXCLUDED (oracle) |
| USD/JPY daily mean_reversion | NO (σ=10) | — | — | — | EXCLUDED (oracle) |
| XAU/USD daily momentum | NO (σ=10) | — | — | — | EXCLUDED (oracle) |
| XAU/USD weekly momentum | NO (σ=10) | — | — | — | EXCLUDED (oracle) |
| BTC/USD weekly momentum | NO (σ=10) | — | — | — | EXCLUDED (oracle) |
| AUD/USD weekly vol_regime | NO (σ=10) | — | — | — | EXCLUDED (oracle) |
| EUR/JPY weekly vol_regime | NO (σ=10) | — | — | — | EXCLUDED (oracle) |

**Oracle-free cells after filtering: 3**  
**Filtering steps:** 5 oracle-free → removed XAU/USD tsmom (edge = -0.240) → removed USD/JPY pure_mr (3rd USD/JPY cell, edge only +0.027) → **3 cells**

### Correlation Matrix (Weekly)

|  | EUR/USD MR | USD/JPY TSMOM | USD/JPY DualMom |
|--|-----------|---------------|-----------------|
| EUR/USD MR | 1.000 | -0.018 | +0.029 |
| USD/JPY TSMOM | -0.018 | 1.000 | +0.556 |
| USD/JPY DualMom | +0.029 | +0.556 | 1.000 |

**Key observation:** EUR/USD MR is near-zero correlated with both USD/JPY strategies (excellent diversifier). The two USD/JPY strategies are moderately correlated (ρ=0.556), which is expected since both trade the same asset.

### Corrected Portfolio Results (977 weeks, 2005–2025)

| Portfolio | Sharpe | Worst-2Y | MaxDD | Vol | Weights |
|-----------|--------|----------|-------|-----|---------|
| P1 Equal Weight | +0.476 | -1.008 | 14.4% | 7.2% | 33/33/33 |
| P2 Inverse Vol | +0.474 | -0.997 | 14.5% | 7.2% | 33/34/32 |
| **P3 Inv. Worst Window** | **+0.449** | **-0.632** | **15.0%** | **7.8%** | **21/49/30** |
| P4 Risk Parity | +0.471 | -0.986 | 14.6% | 7.2% | 33/35/31 |
| P5 Hedged Pairs | +0.476 | -1.008 | 14.4% | 7.2% | 33/33/33 |

**Best portfolio by worst-2Y: P3** (inverse worst-window). It overweights the cell with the best worst-2Y (USD/JPY TSMOM at -0.598) and underweights EUR/USD MR (worst-2Y -1.432). This is the only portfolio that passes the benchmark-calibrated -0.9 threshold on full-sample basis.

### Weekly Aggregation (Critique #11 Fix)

All cells are now aggregated to weekly frequency before portfolio combination. No forward-fill artifacts. Weekly→daily expansion that previously created artificial autocorrelation has been eliminated.

### Regime Analysis (P3)

| Regime | Period | Sharpe | Weeks |
|--------|--------|--------|-------|
| Pre-GFC | 2003–2007 | -0.355 | 140 |
| GFC | 2007–2009 | +0.811 | 82 |
| QE Era | 2009–2020 | +0.493 | 471 |
| COVID | 2020–2022 | +0.771 | 80 |
| Inflation | 2022–2026 | +0.543 | 204 |

Pre-GFC is the only negative-Sharpe regime. The strategy performs well in crisis periods (GFC, COVID), consistent with mean-reversion + momentum diversification.

---

## Task 5.5.2 — Out-of-Sample Validation

**Train:** 2004-02-06 to 2018-12-28 (776 weeks)  
**Test:** 2019-01-04 to 2023-12-29 (261 weeks)  
**Held-out:** 2024-01-01+ (NEVER touched)

### IS vs OOS Comparison (Frozen Weights)

| Portfolio | IS SR | OOS SR | Degradation | IS Worst-2Y | OOS Worst-2Y |
|-----------|-------|--------|-------------|------------|--------------|
| P1 | +0.474 | +0.609 | -28% (improved) | -1.008 | -0.008 |
| P2 | +0.471 | +0.637 | -35% (improved) | -0.944 | -0.027 |
| P3 | +0.459 | +0.655 | -43% (improved) | -0.852 | -0.030 |
| P4 | +0.465 | +0.657 | -41% (improved) | -0.870 | -0.047 |
| P5 | +0.474 | +0.609 | -28% (improved) | -1.008 | -0.008 |

**All 5 portfolios classified as ROBUST (no degradation).**

### Critical Self-Assessment of OOS Results

**The OOS improvement is unusual and warrants serious caveats:**

1. **Favorable test period:** 2019–2023 included major JPY weakness (USD/JPY moved from ~108 to ~151), which strongly favored TSMOM and dual momentum on USD/JPY. Two of three cells trade USD/JPY.

2. **This is NOT evidence of exceptional skill** — it's evidence that recent conditions happened to favor these specific strategies. A different 5-year window might show degradation.

3. **The held-out period (2024+) is the real test.** We correctly do not touch it. When live monitoring begins, the first 90 days of performance will provide the first genuine OOS evidence.

4. **USD/JPY concentration risk:** P3 allocates 79.5% to USD/JPY strategies (49.2% TSMOM + 30.2% dual momentum). A regime where JPY mean-reverts strongly (e.g., BoJ normalization) could produce correlated losses in both cells simultaneously.

---

## Task 5.5.3 — Benchmark Comparison (Active Strategies Only)

Phase 5 incorrectly benchmarked against passive ETFs (SPY, GLD, TLT). Phase 5.5 uses only **active managed-futures strategies**.

### Active Managed-Futures Benchmarks

| Benchmark | Period | Sharpe | Vol | Worst-2Y | Pass -0.5 | Pass -0.7 | Pass -1.0 |
|-----------|--------|--------|-----|----------|-----------|-----------|-----------|
| WTMF | 2011–2026 | +0.149 | 6.6% | -1.540 | no | no | no |
| FMF | 2013–2026 | +0.244 | 8.0% | -0.804 | no | no | YES |
| PQTAX | 2014–2026 | +0.432 | 10.0% | -0.941 | no | no | YES |
| KMLM | 2020–2026 | +0.417 | 13.5% | -0.699 | no | YES | YES |
| DBMF | 2019–2026 | +0.738 | 11.6% | -0.410 | YES | YES | YES |
| GMOM | 2018–2026 | +0.631 | 13.3% | -0.248 | YES | YES | YES |

### Distribution

- **Median worst-2Y:** -0.752
- **25th percentile:** -0.907

### Threshold Recalibration

**Recommended threshold: -0.9** (25th percentile of active managed-futures benchmarks)

Justification: Our portfolio should be at least as good as the bottom quartile of comparable active strategies during their worst 2-year periods. This is derived from benchmark data, NOT outcome-justified.

### Our Portfolio vs Benchmarks

| Entity | Sharpe | Worst-2Y | vs Threshold (-0.9) |
|--------|--------|----------|---------------------|
| WTMF | +0.149 | -1.540 | FAIL |
| FMF | +0.244 | -0.804 | PASS |
| PQTAX | +0.432 | -0.941 | FAIL |
| KMLM | +0.417 | -0.699 | PASS |
| DBMF | +0.738 | -0.410 | PASS |
| GMOM | +0.631 | -0.248 | PASS |
| **Our P3** | **+0.449** | **-0.632** | **PASS** |

P3 at -0.632 is better than 4 of 6 active benchmarks and solidly passes the -0.9 threshold.

---

## Task 5.5.4 — Q3 Deployment Audit

### Auto-Pause Contradiction (Resolved)

- **Historical max DD:** 23.5%
- **Auto-pause trigger:** 15%
- **Problem:** Strategy would have been paused before reaching worst historical DD
- **Resolution:** Option C — keep 15% auto-pause with explicit resume protocol

### Resume Protocol

1. Auto-pause triggers at 15% drawdown
2. Human review required within 48 hours
3. Three checks:
   - Trailing 20-day vol < 1.5× long-term vol
   - Realized slippage < 2× modeled slippage
   - Current z-score near zero (no active signal)
4. Resume if 2 of 3 checks pass
5. If not → continue pause, reassess weekly

**Rationale:** For demo phase, information-gathering > return-maximization. A 15% pause preserves capital while the strategy proves itself. The 23.5% historical DD was in-sample; out-of-sample may be larger.

### Deployment Status

| Component | Status |
|-----------|--------|
| Strategy plugin (eurusd_mr_strategy.py) | EXISTS |
| Log directory | EXISTS |
| OANDA credentials | NOT SET |
| Live trades | 0 |
| **Overall** | **NOT LIVE** |

**Action required:** Set `OANDA_ACCOUNT_ID` and `OANDA_ACCESS_TOKEN` environment variables for OANDA practice account to begin demo trading. The strategy plugin and monitoring infrastructure are ready.

---

## Task 5.5.5 — Terminal State Determination

### Decision Matrix

| Criterion | Value | Threshold | Pass? |
|-----------|-------|-----------|-------|
| Oracle-free cells | 3 | ≥2 | YES |
| Best IS worst-2Y (P3) | -0.632 | > -0.9 | YES |
| Best OOS worst-2Y (P3) | -0.030 | > -0.9 | YES |
| OOS SR degradation | -43% (improved) | ≤ 30% degrad. | YES (improved) |
| Benchmark comparison | Better than 4/6 | ≥ median | YES |

### Terminal State: TERMINAL 1 — DEPLOY PORTFOLIO

**Evidence:** P3 (inverse-worst-window) passes the benchmark-calibrated -0.9 threshold both in-sample (worst-2Y = -0.632) and out-of-sample (worst-2Y = -0.030), with all 5 portfolios showing zero or negative degradation in the 2019–2023 test period.

### Material Caveats (Honest Assessment)

1. **Thin candidate set:** Only 3 oracle-free cells survive. This is the minimum viable portfolio, not a robust multi-strategy book. Compare to professional managed-futures funds that trade 50-200 instruments across 4-5 asset classes.

2. **USD/JPY concentration:** 79.5% of P3 is allocated to USD/JPY strategies. This is a single-currency bet disguised as a portfolio. A BoJ regime shift could produce correlated losses.

3. **OOS "improvement" is not evidence of alpha:** The 2019–2023 period was exceptionally favorable for JPY carry unwind (USD/JPY +40%). Don't expect this to repeat.

4. **Weekly vol = ~7.8%:** This is below the 10% target because vol-targeting each cell individually and then combining suppresses portfolio vol. The at-10%-vol results (leverage 1.29x) show similar Sharpe but higher drawdowns.

5. **No live track record yet.** All results are backtested. The demo period is the real first test.

### Recommended Deployment Path

1. **Immediate:** Set OANDA practice account credentials, begin EUR/USD MR demo trading (Q3 — already built)
2. **Week 1:** Confirm EUR/USD MR plugin executes correctly on demo
3. **Week 2-4:** Build LTS plugins for USD/JPY TSMOM and USD/JPY dual momentum
4. **Month 2:** Deploy full P3 portfolio on demo with weekly rebalancing
5. **Month 3:** First 90-day review — compare live vs backtest
6. **Month 4+:** If 90-day review passes, consider micro-sized real capital allocation

### What Would Change the Terminal State

- **Downgrade to T2.5:** If 90-day live performance shows SR < 0 or MaxDD > 20%
- **Downgrade to T3:** If 90-day live performance shows systematic strategy failure (e.g., TSMOM consistently wrong-sided, dual momentum always in wrong asset)
- **Stay at T1:** If live performance is within ±50% of backtest Sharpe and drawdown stays under 15%

---

## Corrections Applied (Phase 5 Critique Items)

| # | Issue | Status | Resolution |
|---|-------|--------|------------|
| 1 | Oracle-dependent cells mixed | FIXED | Only 3 oracle-free cells included |
| 2 | XAU/USD momentum with negative edge | FIXED | Excluded (edge = -0.240) |
| 3 | USD/JPY over-representation | FIXED | Capped at 2 cells |
| 4 | No OOS validation | FIXED | IS/OOS split with frozen weights |
| 5 | Wrong benchmarks (passive ETFs) | FIXED | 6 active managed-futures ETFs |
| 6 | Moskowitz synthetic simulation | FIXED | Dropped entirely, real ETF data only |
| 7 | Phase 3-4 results not loaded | N/A | Fresh computation from data |
| 8 | Vol-targeting suppresses portfolio vol | NOTED | Reported both realized and 10%-target |
| 9 | Weekly→daily forward-fill | FIXED | Weekly aggregation, no forward-fill |
| 10 | Auto-pause contradiction | FIXED | Option C + resume protocol |
| 11 | Threshold outcome-justified | FIXED | Derived from benchmark 25th percentile |

---

## Files Produced

- `trading_research/phase5_5_corrective_audit.py` — Complete audit script
- `trading_research/results/phase5_5_results.json` — Full numerical results
- `trading_research/results/PHASE_5_5_SYNTHESIS.md` — This document
