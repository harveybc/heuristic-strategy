# Work Plan: Phase 3.5 — Audit & Data Extension Before Phase 4

**Status of prior work:** Phase 0–3 complete. 14 cells survived naive baseline filter. EUR/USD hourly MR killed with evidence. Phase 3 exogenous data integrated except COT (403 Forbidden).

**Purpose of this phase:** Before investing weeks in Phase 4 strategy development, resolve four methodological concerns that could invalidate up to half of the surviving cells. This phase takes 4–6 days and prevents building strategies on artifacts.

**Execution context:** Three machines available with NVIDIA RTX GPUs (8GB, 12GB, 16GB VRAM):
- `omega` (local, RTX 4070, 12GB) — orchestration + FX pairs
- `dragon` (192.168.1.235, RTX 4090, 16GB) — crypto + commodities
- `gamma` (192.168.0.106, RTX 5070 Ti, 8GB) — FX majors

Most audits in this phase are CPU/IO bound (data downloads, sign-flip counting, statistical recomputation). GPU parallelism is only required for Task 3 (data extension + oracle recomputation on longer series). Plan distribution accordingly.

---

## Strategic Principles (Same as Prior Phase)

1. Every hypothesis has a kill criterion defined before execution.
2. Negative results are first-class outcomes — document them.
3. Reuse existing infrastructure (oracle sensitivity module, cost model, evaluation harness).
4. Never tune on the 2024–2025 held-out test year.
5. Report uncertainty, not just point estimates.

---

## Task 1: Audit the 10σ Noise Budget Ceiling (1–2 days, omega)

**Problem:** Seven of 14 surviving cells show noise budget = 10.00σ, which is the grid maximum. This is suspicious. Either these cells tolerate absurd noise levels (unlikely), or the noise injection mechanism is ineffective for certain strategy/timeframe combinations, or the oracle leaks future information.

### Task 1.1 — Manual trace of BTC/USD weekly momentum at noise = 10σ

Pick this cell as the canonical case. Run the oracle sensitivity test with full logging:

- For each weekly bar in the test period, record: (a) oracle's true predicted log-return, (b) noisy prediction after injection at noise=10σ, (c) action taken (long/short/flat), (d) action that would have been taken at noise=0.
- Compute the **sign-flip probability**: fraction of bars where action_at_noise_10σ ≠ action_at_noise_0.
- Compute the **signal-to-noise ratio at decision time:** |true_prediction| / injected_noise_std, per bar.

**Expected outcome if the metric is honest:**
- Sign-flip probability at 10σ should be close to 50% (the noise completely dominates, strategy becomes random).
- If strategy is still profitable at 50% sign-flip, it means the win/loss asymmetry (SL/TP levels) carries the PnL, not the prediction. That would be a different kind of edge worth knowing about, but NOT what the oracle test claims to measure.

**Expected outcome if the metric is broken:**
- Sign-flip probability at 10σ is <20%.
- This means on weekly bars with large drifts, noise of 10×σ(weekly return) rarely flips the sign of the prediction, because the drift itself is many σ.
- The noise budget metric is then saturated trivially and tells us nothing about predictor quality requirements.

### Task 1.2 — Repeat trace for 3 additional 10σ cells

Pick representative samples:
- XAU/USD weekly momentum (commodity, long horizon)
- GBP/USD 4h vol_regime_switch (FX, shorter horizon — important because this is the only non-weekly 10σ cell)
- EUR/USD daily mean_reversion (FX, MR strategy, different family than the others)

Same instrumentation. Compare sign-flip probabilities.

### Task 1.3 — Decision point

Based on Tasks 1.1–1.2, classify each 10σ cell into:
- **Honest ceiling:** sign-flip probability at 10σ is ≥40%, strategy remains profitable due to real structural edge. Keep in the shortlist.
- **Trivially saturated:** sign-flip probability at 10σ is <20%. Noise budget metric is meaningless for this cell. Replace the metric with alternative (see Task 1.4).
- **Ambiguous:** sign-flip probability 20–40%. Re-evaluate after Task 2.

### Task 1.4 — Replacement metric for saturated cells

For cells that fail Task 1.3, compute instead:
- **Minimum prediction accuracy needed:** fraction of bars where the predictor must correctly call the direction for the strategy to break even net of costs. This is a more interpretable number (e.g., "needs 54% directional accuracy to be profitable").
- **Magnitude sensitivity:** how much does net Sharpe degrade if predictions are scaled by 0.5, 0.75, 1.25, 1.5? (Multiplicative miscalibration rather than additive noise.)

These two metrics together give a more robust picture of predictor requirements for trend-driven weekly cells.

### Deliverable Task 1
Report with:
- Sign-flip probabilities for each 10σ cell
- Classification (honest / trivially saturated / ambiguous)
- Replacement metrics for saturated cells
- Updated shortlist with revised budget characterization

### Kill criterion
If 6+ of 7 10σ cells turn out to be trivially saturated, the weekly-timeframe result is mostly an artifact of methodology. In that case, reprioritize Phase 4 toward the Tier 2 (3σ) XAU cells and defer weekly cells until after Phase 4 initial results are in.

### Machine assignment
Omega. Single-machine task, CPU-bound, uses existing oracle_sensitivity.py infrastructure with added logging.

---

## Task 2: Edge-Over-Buy-and-Hold Reranking (1 day, omega)

**Problem:** Current ranking uses raw oracle Sharpe. But on assets with strong secular trends (BTC, XAU in bull periods), a strategy can have high absolute Sharpe mostly by riding the trend. The real question is whether the strategy adds value *over passive exposure*.

### Task 2.1 — Recompute rankings with edge-over-B&H metric

For all 14 surviving cells:

- Edge Sharpe = Oracle Sharpe − B&H Sharpe (both computed on identical period and with identical cost assumptions)
- Edge at noise budget = Sharpe at that cell's noise budget level − B&H Sharpe
- Noise budget relative to B&H: noise level at which Oracle Sharpe drops to B&H Sharpe (not to 0.3)

### Task 2.2 — Re-rank and identify B&H-dominated cells

A cell is B&H-dominated if its noise budget relative to B&H is ≤1σ. At that point, any realistic predictor performs no better than buying and holding the asset, and the strategy is not worth building.

### Task 2.3 — Compute short-side separately

Some cells (especially in crypto bull markets) have most of their edge from long-only trades. Run the oracle strategy in three modes:

- Long-only
- Short-only
- Long+short

Report Sharpe for each mode. A strategy with Sharpe 1.9 long+short but Sharpe 0.4 short-only is effectively a "time-the-trend" strategy, which is simpler and more honest framing than a full long/short system.

### Deliverable Task 2
Updated ranking table with columns:
- Cell
- Raw oracle Sharpe (existing)
- B&H Sharpe (existing)
- **Edge Sharpe (new, primary ranking metric)**
- Long-only / short-only / long+short Sharpes
- Is B&H-dominated (yes/no)

### Kill criterion
Cells that are B&H-dominated are removed from Phase 4 shortlist. If fewer than 5 cells survive both Task 1 and Task 2, re-examine whether the problem is methodological before concluding no edge exists.

### Machine assignment
Omega. Single-machine, CPU-bound, uses evaluation_harness.py.

---

## Task 3: Historical Data Extension for Weekly Cells (2–3 days, distributed)

**Problem:** Weekly strategies evaluated on 2018–2025 data have ~365 weekly bars. Rolling 2Y windows contain ~104 bars each. This is statistically thin for robustness claims.

### Task 3.1 — Extend historical data

Target depth per asset:
- **BTC/USD**: extend to 2014-01-01 (Bitfinex or CryptoCompare; yfinance has gaps)
- **ETH/USD**: extend to 2017-08-01 (inception of liquid trading)
- **XAU/USD**: extend to 2000-01-01 (LBMA fix data or continuous futures GC)
- **XAG/USD**: extend to 2000-01-01 (same sources)
- **EUR/USD, USD/JPY, GBP/USD, AUD/USD**: extend to 2000-01-01 (OANDA, Dukascopy, or your existing long-history source)
- **AUD/JPY, EUR/JPY, GBP/JPY**: extend to 2000-01-01 (same)

Data quality checks:
- No gaps larger than 1 week
- No obvious price errors (rolling z-score > 10 on returns flagged for inspection)
- Consistent session/timezone handling across sources when splicing

### Task 3.2 — Recompute oracle sweep on extended weekly data

For each of the 14 surviving cells that uses weekly or daily timeframe (i.e., all of them), rerun the oracle sweep with extended history.

Output per cell:
- New oracle Sharpe on full history
- New B&H Sharpe on full history
- New Edge Sharpe
- **Rolling regime robustness over full history with 2Y windows, 6-month step**
- Drawdown statistics including worst historical drawdown

### Task 3.3 — Regime change detection

For cells with long history, explicitly test whether the edge is consistent across macro regimes. Define regimes:
- 2000–2007: pre-GFC, USD strength / EUR weakness
- 2008–2012: GFC + Euro crisis
- 2013–2019: post-crisis QE era
- 2020–2021: COVID + stimulus
- 2022–2025: inflation shock + rate hikes

Report oracle and edge Sharpe per regime. A cell with positive edge in 4 of 5 regimes is far more trustworthy than one that only works in 2020–2022.

### Deliverable Task 3
Extended history database + recomputed oracle sweep on all 14 cells + per-regime Sharpe breakdown.

### Kill criterion (per cell, not for phase)
A cell is killed from Phase 4 shortlist if:
- Edge Sharpe on extended history drops by more than 50% vs 2018–2025 result, OR
- Edge is positive in fewer than 3 of 5 regimes, OR
- Worst 2Y rolling window Sharpe is worse than −0.5 (catastrophic loss of edge)

### Machine assignment
Distributed:
- **Dragon (RTX 4090, 16GB)**: BTC/USD, ETH/USD, XAU/USD, XAG/USD, CL (crypto + commodities, same as before)
- **Gamma (RTX 5070 Ti, 8GB)**: EUR/USD, USD/JPY, GBP/USD, AUD/USD
- **Omega (RTX 4070, 12GB)**: AUD/JPY, EUR/JPY, GBP/JPY + orchestration + aggregation

Note: the oracle sweep itself is not GPU-heavy (no model training). GPU allocation is for parallelism of the 360-cell recomputation, not compute intensity per cell. If using CPU-only workers, the work is still embarrassingly parallel across cells.

---

## Task 4: Fix COT Data Download (half day, omega)

**Problem:** CFTC COT reports returned 403 Forbidden in Phase 3. COT positioning is a legitimately valuable feature for FX and commodities.

### Task 4.1 — Diagnose the 403

Likely causes (in order of probability):
1. Missing or wrong User-Agent header
2. Attempting to hit a deprecated endpoint URL
3. Rate limiting (less likely for a single request)
4. Geographic/VPN blocking (unlikely for CFTC)

### Task 4.2 — Use correct modern endpoint

The CFTC provides COT data through:
- **Socrata API (modern, recommended):** `https://publicreporting.cftc.gov/resource/`
  - Legacy combined futures: `6dca-aqww.json`
  - Disaggregated futures: `72hh-3qpy.json`
  - TFF futures: `gpe5-46if.json`
- **Direct historical ZIPs (legacy):** `https://www.cftc.gov/files/dea/history/deacot[YEAR].zip`

Use Socrata for recent data (2016+) and direct ZIPs for older history (1986–2015). Splice together.

### Task 4.3 — Ingest into feature store

For each relevant asset, extract:
- Net non-commercial position (speculator positioning)
- Net commercial position (hedger positioning)
- Open interest
- Normalized z-score of net-non-commercial position over trailing 156 weeks (3 years)

CFTC contract codes:
- EUR/USD → 099741 (EUR FX)
- USD/JPY → 097741 (JPY FX)
- GBP/USD → 096742 (GBP FX)
- AUD/USD → 232741 (AUD FX)
- XAU/USD → 088691 (Gold)
- XAG/USD → 084691 (Silver)
- CL (Crude) → 067651

Propagate weekly values as step function to daily/4h/hourly bars (value changes on Tuesday-of-the-week, publicly available Friday, so lagged 3 business days to avoid look-ahead).

### Task 4.4 — Update feature store

Add COT-derived columns to the daily feature CSVs for all relevant assets. Document the lag convention clearly.

### Deliverable Task 4
Updated feature store with COT features for FX majors, XAU, XAG, CL. Document with column names, update frequency, and lag handling.

### Machine assignment
Omega. Pure data engineering, no GPU needed.

---

## Execution Order and Parallelism

```
Day 1:        Task 1.1–1.2 (omega)         ‖  Task 2 (omega background)  ‖  Task 4 (omega background)
Day 2:        Task 1.3–1.4 (omega)         ‖  Task 3.1 starts (all 3 machines, data download)
Day 3:        Task 3.2 (distributed)
Day 4:        Task 3.2 continues + Task 3.3 (distributed)
Day 5:        Consolidation, cross-task analysis, updated shortlist
Day 6:        Buffer for unexpected issues
```

Tasks 1, 2, and 4 can run concurrently on omega because they are lightweight and use different parts of the pipeline (1 = oracle internals, 2 = ranking recomputation, 4 = data ingestion).

Task 3 is the long pole and fans out across all three machines. Data download (Task 3.1) is IO-bound and can saturate before GPU parallelism helps — it may be worth doing the downloads on omega and distributing only the recomputation (Task 3.2).

---

## Integration Step (End of Phase 3.5)

At the close of this phase, produce a single **Updated Shortlist** document with:

For each cell that survived Tasks 1, 2, and 3:
- Asset, timeframe, strategy family
- Edge Sharpe (not raw oracle Sharpe)
- Honest noise budget (or replacement metric for saturated cells)
- Regime robustness across 5 macro regimes
- Worst 2Y rolling Sharpe
- Structural hypothesis for why the edge exists (one paragraph)
- Priority ranking for Phase 4

**Expected shortlist size:** 3–7 cells. If more, you are being too permissive. If fewer than 3, re-examine whether any were killed unfairly.

**Expected dominant candidates (prior belief, to be validated):**
- XAU/USD daily momentum — best structural story, plentiful data
- XAU/USD weekly momentum — extended history crucial here
- GBP/USD 4h vol_regime_switch — only non-weekly non-daily survivor, unique
- Possibly BTC/USD weekly momentum if it clears edge-over-B&H and extended history

---

## Standing Rules for This Phase

1. **Report sign-flip probabilities with the noise budget.** A noise budget without its sign-flip context is misleading.
2. **Always compare to B&H, not just to zero.** Sharpe > 0 is not the right bar on trending assets.
3. **Document kills with evidence.** When a cell is removed from the shortlist, record the specific criterion and value that triggered the kill. Future agents will want to audit these decisions.
4. **Do not peek at 2024–2025.** Extended history adds older data, not newer. The held-out test year remains inviolate.
5. **Preserve the original 14-cell results.** Write Phase 3.5 outputs alongside, not overwriting. Comparison between phases is itself evidence.
6. **Flag surprises immediately.** If any result strongly contradicts a prior finding (e.g., a cell that looked strong collapses entirely on extended history), surface it for human review before proceeding.

---

## What Success Looks Like

At the end of Phase 3.5, you have one of three outcomes:

1. **Shortlist of 3–7 high-confidence cells** with Edge Sharpe > 0.4, regime robustness > 0.5, positive edge in ≥3 of 5 macro regimes, and structural explanations. Phase 4 can proceed with genuine candidates.

2. **Shortlist of 1–2 cells** — the Tier 1 weekly cells were mostly saturation artifacts, but Tier 2 XAU daily and maybe GBP/USD 4h survive with honest signal. Phase 4 proceeds but narrower in scope.

3. **No cells survive all four audits.** This would be a strong signal that retail edge on public data is beyond reach at current methodology. Decision point: either pivot to alternative data / different paradigm, or close the project with documented negative result. Either outcome is valid.

All three are better than proceeding to Phase 4 on unaudited signals.