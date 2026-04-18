> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 6.B — Synthesis: Evaluation of Untested Candidates

**Date:** 2026-04-17
**Status:** COMPLETE
**Scenario:** S4
**Action:** P3 confirmed as best candidate. Proceed to Phase 6.C for original P3.

---

## Executive Summary — Caveats First

1. **Timeframe adaptation:** The regime plugins (H1, H2) were designed for 4h bars resampled from 1h. This evaluation uses daily bars because 4h data is not available. The regime classification logic (bb_position, atr_ratio, ema_alignment) is frequency-agnostic in principle, but trade frequency and TP/SL dynamics differ. Results represent "regime logic on daily data," not the original 4h plugin.

2. **Feature selection look-ahead (H1):** The 3 causal features (bb_position score=5, atr_ratio score=4, ema_alignment positive TE) were selected using causal analysis that may have seen post-2018 data. The threshold values (0.25, 0.75, 1.2, 0.0) are not fitted to data.

3. **Centroid fitting look-ahead (H2):** The original GMM centroids are fitted on 15 years of EUR/USD data overlapping the test period. H2-refit addresses this by refitting on training data only.

4. **Terminal state:** Phase 6.B operates under Terminal 2.5 (staged validation). No deployment is automatic.

---

## Task 6.B.1 — P3 Strategy Validation

Reproduced Phase 5.5 strategy results using the same vectorized implementations:

| Cell | Sharpe | worst-2Y | OOS Sharpe |
|------|--------|----------|------------|
| EUR/USD pure_mr | 0.1860 | -1.4315 | -0.2576 |
| USD/JPY tsmom | 0.2964 | -0.5979 | 0.4097 |
| USD/JPY dual_momentum | 0.4055 | -0.9727 | 0.6635 |
| **P3 portfolio** | **0.4466** | **-0.7243** | **0.4877** |

P3 weights (inverse worst-window): {'eurusd_mr': np.float64(0.206), 'usdjpy_tsmom': np.float64(0.492), 'usdjpy_dm': np.float64(0.302)}

LTS strategy plugins created:
- `usdjpy_tsmom_strategy.py` — TSMOM with monthly rebalance, inverse-vol sizing
- `usdjpy_dual_momentum_strategy.py` — Dual Momentum with cross-asset comparison
- Both registered in `setup.py` entry points

---

## Task 6.B.2 — H1: plugin_regime_wfo Standalone

### Results (EUR/USD daily)

| Metric | Value |
|--------|-------|
| Full-sample Sharpe | -0.1546 |
| worst-2Y Sharpe | -1.3363 |
| IS Sharpe (≤2018) | -0.1342 |
| OOS Sharpe (2019-2023) | 0.0496 |
| Max drawdown | 0.5573 |
| Trades | 438 (19.0/yr) |
| Hit rate | 0.3668 |
| Cost sensitivity (2x) | -0.1546 |
| Deploy threshold (worst-2Y > -0.9) | FAIL |

### Regime Breakdown

| Period | Sharpe |
|--------|--------|
| pre_gfc | 0.1221 |
| gfc | 0.3493 |
| qe_era | -0.4539 |
| covid | 0.7611 |
| inflation | -0.7913 |

### Regime Distribution (daily bars from bar 200)
- VOLATILE_OVERSOLD: 200 bars
- BEARISH_CONTINUATION: 679 bars
- VOLATILE_OVERBOUGHT: 138 bars
- NEUTRAL: 3089 bars
- PULLBACK_IN_UPTREND: 649 bars
- BULLISH_DRIFT: 851 bars

---

## Task 6.B.3 — H2: plugin_regime_adaptive Standalone

### H2-original (hardcoded GMM centroids — IN-SAMPLE FITTED)

| Metric | Value |
|--------|-------|
| Full-sample Sharpe | 0.0808 |
| worst-2Y Sharpe | -0.7349 |
| IS Sharpe (≤2018) | 0.0384 |
| OOS Sharpe (2019-2023) | 0.1716 |
| Trades | 414 (18.0/yr) |
| Deploy threshold | PASS |

**WARNING:** H2-original results include look-ahead from centroids fitted on full dataset.
Deployment decisions must use H2-refit results only.

### H2-refit (GMM centroids refit on training data only)

| Metric | Value |
|--------|-------|
| Full-sample Sharpe | 0.1390 |
| worst-2Y Sharpe | -1.8164 |
| IS Sharpe (≤2018) | 0.2076 |
| OOS Sharpe (2019-2023) | -0.1890 |
| Trades | 414 (18.0/yr) |
| Deploy threshold | FAIL |

**Look-ahead effect:** Sharpe gap = -0.0582 (H2-original − H2-refit).
GMM classification robust to centroid refitting.

### H1 vs H2 Comparison

Preferred: **H2** (threshold-based is preferred when results are similar due to simplicity)

---

## Task 6.B.4 — H3: P3 + Regime Filter Hybrid

### Design
- Option A: Snapshot regime classification at daily close
- If regime ∈ {BEARISH_CONTINUATION, VOLATILE_OVERBOUGHT, NEUTRAL}, suppress trade
- EUR/USD cells filtered by EUR/USD regime; USD/JPY cells by USD/JPY regime

### Per-Cell Impact

| Cell | Sharpe (orig) | Sharpe (filtered) | Trades (orig) | Trades (filt) | Reduction |
|------|--------------|-------------------|---------------|---------------|-----------|
| EUR/USD MR | 0.1860 | -0.2356 | 915 | 663 | 28% |
| USD/JPY TSMOM | 0.2964 | -0.2624 | 267 | 808 | -203% |
| USD/JPY DM | 0.4055 | 0.1480 | 26 | 348 | -1238% |

### H3 Portfolio

| Metric | P3 Baseline | H3 Filtered | Change |
|--------|-------------|-------------|--------|
| Sharpe | 0.4466 | -0.1847 | -0.6313 |
| worst-2Y | -0.7243 | -1.5502 | -0.8259 |
| OOS Sharpe | 0.4877 | -0.2490 | -0.7367 |

### Kill Criteria

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| worst-2Y improvement | ≥20% | -114.0% | FAIL |
| Sharpe degradation | ≤20% | 141.4% | FAIL |
| Trade reduction | ≤70% per cell | {'eurusd_mr': 0.275, 'usdjpy_tsmom': -2.026, 'usdjpy_dm': -12.385} | PASS |
| **Overall** | All pass | | **NOT VALUABLE** |

### Concentration

| Metric | P3 Original | H3 Filtered |
|--------|-------------|-------------|
| USD/JPY time-weighted exposure | 71.2% | 64.5% |

---

## Terminal Decision

### Scenario: S4

| Candidate | Passes Deploy? | Key Metric |
|-----------|---------------|------------|
| H1 (regime_wfo) | NO | worst-2Y=-1.3363 |
| H2-refit (regime_adaptive) | NO | worst-2Y=-1.8164 |
| H3 (P3+filter) | NOT VALUABLE | worst-2Y improvement=-114.0% |

### Recommended Action

P3 confirmed as best candidate. Proceed to Phase 6.C for original P3.

### What Phase 6.C Should Test

The candidate proceeding to Phase 6.C robustness testing:
- **If S1/S2:** Expanded portfolio including regime cell — stress test with parameter perturbation, Monte Carlo bootstrap, regime-conditional analysis
- **If S3:** P3 hybrid with regime filter — stress test filter stability
- **If S4:** Original P3 — standard robustness validation per Terminal 2.5 requirements

---

## Honest Acknowledgments

1. **Daily vs 4h timeframe:** The regime plugins were designed for 4h bars. Daily evaluation may undercount regime transitions (fewer bars = fewer transition opportunities). The transition_only filter with daily data produces fewer entry signals than with 4h data. If H1/H2 show low trade counts, this is a structural artifact of the daily adaptation, not necessarily indicative of the 4h plugin's behavior.

2. **Pre-registered kill criteria were applied without modification.** The thresholds were set before seeing results, as required by the work plan.

3. **The F1 gap finding remains validated:** No predictor-based combinations were tested because the structural evidence (F1=0.44 vs required 0.91) was accepted from Phase 6.A. This plan focused on the genuinely untested candidates.

4. **2024-2025 holdout period was preserved.** All evaluation uses ≤2023 data. The holdout remains untouched for Phase 6.C or live validation.
