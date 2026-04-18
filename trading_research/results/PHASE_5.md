# Work Plan: Phase 5 — Three Decisive Questions Before Final Decision

**Prior state:** Phase 4 complete. All 17 strategy-asset combinations killed by worst-2Y-window criterion at −0.5 Sharpe, including academic strategies (Moskowitz 2012, Antonacci 2014, Menkhoff 2012). Regime filter with perfect oracle cannot rescue cells at that threshold. Framework-level structural finding, not tuning issue.

**Purpose of this phase:** Answer three specific questions that determine the final disposition of the project. Each question has a concrete experiment, a defined kill criterion, and produces binary actionable output. This is not another research expansion — it is the convergence phase.

**Duration:** 1 week total. If results warrant deployment, additional time is for live monitoring, not new research.

**Execution context:** Three machines available (omega RTX 4070, dragon RTX 4090, gamma RTX 5070 Ti). Phase 5 is lighter on compute than prior phases because no new large-scale sweeps are required.

---

## Guiding Principle for This Phase

Phase 4 demonstrated that individual strategies do not survive strict single-strategy worst-window criteria. Phase 5 does not try to find a better strategy. It asks three questions that the entire prior pipeline has implicitly assumed answers to, but never explicitly tested:

1. Does diversification across already-evaluated strategies change the picture?
2. Is our kill criterion calibrated against real-world deployed strategies, or is it self-imposed?
3. Can our best-available candidate (EUR/USD daily MR) be operationally deployed even if suboptimal on paper?

Answering these three questions converges the project to one of three honest terminal states: deployment, closure with documented research, or acknowledged need for resources beyond the project's scope.

---

## Question 1 — Does Portfolio Diversification Rescue Killed Strategies? (2 days, dragon + omega)

### Rationale
Phase 4 evaluated each (asset, timeframe, strategy) cell independently. But institutional CTAs and trend-following funds do not require each strategy to survive individually — they require the *portfolio* to survive. If the worst 2Y windows of different cells occur in different calendar periods, diversification mechanically improves portfolio worst-window metrics even when component worst-windows are all bad.

This question has a binary answer: either temporal diversification of cells produces an acceptable portfolio, or it does not. We test this directly with the cells already in hand.

### Task 1.1 — Identify candidate cells for portfolio construction
From Phase 3.5 and Phase 4 results, select up to 12 cells meeting:
- Edge Sharpe over B&H > 0
- Worst 2Y window better than −3.0 (exclude catastrophic outliers like XAG weekly at −6.17)
- Extended history available (to compute worst 2Y correctly)

Expected candidates:
- EUR/USD daily MR
- USD/JPY daily MR
- XAU/USD daily momentum
- XAU/USD weekly momentum
- BTC/USD weekly momentum
- AUD/USD weekly VRS
- EUR/JPY weekly VRS
- USD/JPY TSMOM (Phase 4 Track C)
- USD/JPY Dual Momentum (Phase 4 Track C)
- EUR/USD Cross-Sectional FX (Phase 4 Track C)
- GBP/USD 4h VRS if data allows
- Plus any other cell with Edge > 0 from prior phases

### Task 1.2 — Compute per-cell return series
For each candidate cell, reconstruct the daily return series of the strategy over the full extended history (or common overlap period, whichever is longer).

Return series must be:
- Net of transaction costs (use standardized cost model from Phase 0)
- Position-sized consistently across cells (e.g., each cell targets 10% annualized volatility)
- Aligned to common calendar dates with forward-fill for lower-frequency strategies (weekly → daily)

### Task 1.3 — Compute correlation matrix and temporal overlap of bad windows
For each pair of cells:
- Pairwise correlation of daily returns
- Correlation of rolling 2Y Sharpe series (this captures whether cells have bad periods at the same times)
- Calendar months where both cells are simultaneously in their worst 10% of performance

Output: correlation matrix and a heatmap of "simultaneous bad periods" between cell pairs.

### Task 1.4 — Construct candidate portfolios
Evaluate these portfolio weighting schemes:

**P1 — Equal weight**: 1/N across all surviving cells
**P2 — Inverse volatility weight**: weight inversely proportional to cell's realized volatility
**P3 — Inverse worst-window weight**: weight inversely proportional to cell's worst 2Y magnitude (heavier weight on more stable cells)
**P4 — Risk parity**: weights adjusted so each cell contributes equal marginal variance to portfolio
**P5 — Hedged pairs**: select 3-4 cell pairs with low or negative correlation, equal weight within pairs

For each portfolio, compute:
- Full-period Sharpe net of costs and rebalancing friction
- Worst 2Y rolling window Sharpe
- Max drawdown
- Edge over equal-weight buy-and-hold of the same assets
- Turnover and implied rebalancing cost

### Task 1.5 — Regime-by-regime portfolio analysis
For each portfolio, compute Sharpe per macro regime (Pre-GFC, GFC, Post-Crisis QE, COVID, Inflation). A robust portfolio has positive Sharpe in at least 4 of 5 regimes.

### Task 1.6 — Decision

**Outcome 1 — Portfolio rescues the problem (worst 2Y > −0.5):**
- Any of P1-P5 produces Edge Sharpe > 0.4 AND worst 2Y > −0.5 AND positive in 4+ regimes
- This is the path to deployment: implement the portfolio as Phase 5 deliverable

**Outcome 2 — Portfolio meaningfully improves but does not cross threshold (worst 2Y between −0.5 and −0.8):**
- Portfolio improves over component strategies but still fails strict criterion
- Feeds into Question 2 decision: is our threshold realistic?

**Outcome 3 — Portfolio does not improve materially:**
- Cell correlations are too high or bad periods too synchronized
- Diversification hypothesis rejected; move to other questions

### Kill criterion for Question 1
If all five portfolio variants produce worst 2Y < −1.5 (worse than most individual killed cells), the diversification hypothesis is falsified and Question 1 produces "no" as final answer.

### Machine assignment
- **Dragon (RTX 4090)**: compute cell return series, correlation matrix, and run portfolio constructions (CPU-bound; GPU not strictly required but dragon has the capacity for potential parallelization of bootstrap confidence intervals on portfolio metrics).
- **Omega**: orchestration, report writing, comparison with prior phase results.

### Deliverables
- `phase5_q1_portfolio_results.json` with metrics for P1-P5
- Correlation matrix and temporal overlap visualization
- Regime breakdown per portfolio
- Clear outcome classification (Outcome 1, 2, or 3)

---

## Question 2 — Is Our Kill Criterion Industry-Standard or Self-Imposed? (2 days, omega)

### Rationale
Phase 4 found that academic strategies with thousands of citations (Moskowitz TSMOM, Antonacci Dual Momentum) fail our −0.5 worst-2Y threshold. These strategies are deployed in production with real capital at AQR, Man Group, Winton, Dunn Capital, and hundreds of CTAs. This is an apparent contradiction: either the industry tolerates worse worst-windows than we do, or they apply additional layers (leverage, portfolio effects, overlays) that our test does not include.

Resolving this contradiction is essential. If our criterion is stricter than reality, we are rejecting strategies that the market actually rewards. If our criterion matches reality, then the strategies we killed are genuinely unsuitable and we have learned the true state of affairs.

### Task 2.1 — Industry benchmark research
Review publicly available performance data for:
- **SG CTA Index**: aggregate index of managed futures CTAs, public monthly data since 2000
- **SG Trend Index**: subset of trend-following CTAs
- **Credit Suisse Managed Futures Index** (or equivalent)
- Individual fund factsheets where publicly available: AQR Managed Futures, Man AHL Diversified, Winton Diversified (from regulatory filings, fund marketing materials, or Morningstar)

For each, extract or compute:
- Full-period Sharpe
- Worst 2Y rolling Sharpe (same metric we use)
- Max drawdown
- Length and severity of worst drawdown period

### Task 2.2 — Academic strategy worst-window in original papers
Re-read the Moskowitz, Antonacci, and Menkhoff papers. For each:
- What metric did they report for "bad periods"?
- Did they report worst 2Y Sharpe explicitly?
- If not, can we compute it from their reported returns?
- What out-of-sample drawdown did they acknowledge?

Specifically: the Moskowitz 2012 TSMOM paper reported annual returns 1985-2009 for 58 instruments. The paper's table of annual returns allows us to compute worst 2Y Sharpe of their own published strategy.

### Task 2.3 — Compare industry worst-2Y to our threshold
Produce a table:

| Entity | Worst 2Y Sharpe | Period of worst window | Our threshold check |
|--------|----------------|----------------------|---------------------|
| SG CTA Index | [compute] | [date range] | pass/fail at −0.5 |
| AQR Managed Futures | [compute if available] | [date range] | pass/fail at −0.5 |
| Man AHL Diversified | [compute if available] | [date range] | pass/fail at −0.5 |
| Moskowitz 2012 TSMOM | [compute from paper] | [date range] | pass/fail at −0.5 |
| Our best portfolio (from Q1) | [from Q1] | [date range] | pass/fail at −0.5 |

### Task 2.4 — Interpret
Three cases to distinguish:

**Case A — Industry benchmarks also have worst 2Y < −0.5:**
- Our threshold is stricter than what deployed capital accepts
- Recalibrate threshold to match industry reality (propose −0.75 or −1.0 as new default)
- Previously killed strategies may be viable at recalibrated threshold
- This changes the conclusion of Phase 4 substantially

**Case B — Industry benchmarks have worst 2Y > −0.5 consistently:**
- Our threshold is appropriate
- Our strategies genuinely underperform the real deployed universe
- Gap suggests missing ingredients: more asset diversification, overlay strategies, capital allocation skill
- Confirms need for Question 3 deployment-as-modest-product or closure

**Case C — Industry benchmarks are variable with some passing and others failing:**
- Worst 2Y is one criterion among several used in practice
- Need to evaluate our strategies against multiple criteria, not just worst 2Y
- Add criteria like Calmar ratio, recovery time from worst drawdown, percentage of rolling 3Y periods profitable

### Task 2.5 — Decision
Produce a recommendation for the "true" kill criterion to use in Phase 5 final synthesis. Document with evidence from Task 2.3. This may confirm our current threshold or propose a recalibration. Any recalibration must be justified by external benchmarks, not by the convenience of reviving our own strategies.

### Kill criterion for Question 2
None — all three outcomes produce actionable information. If external data is unavailable or paywalled, fall back to published academic paper returns (Moskowitz 2012 at minimum is freely available).

### Machine assignment
Omega only. This is research and analysis work, not computation.

### Deliverables
- `phase5_q2_benchmark_comparison.md` with the comparison table
- Clear recommendation on threshold calibration
- Evidence package (paper references, index data sources)

---

## Question 3 — Can EUR/USD Daily MR Be Operationally Deployed? (2 days setup + ongoing monitoring)

### Rationale
Across all Phase 4 tracks, EUR/USD daily MR consistently has the best or near-best worst-window behavior. It is oracle-independent (no predictor needed), has 100% positive Sharpe across its parameter grid, and a max drawdown (27%) that is painful but survivable with conservative sizing. The question is not whether its backtest is impressive — it is whether the operational stack works in live conditions.

This task does not produce a "winning strategy." It produces operational validation of the live-trading pipeline and accumulates real performance data in parallel with other work.

### Task 3.1 — Strategy implementation
Build the EUR/USD daily MR as a Backtrader plugin with the following specification:
- Entry: z-score of price vs rolling mean crosses ±z_entry threshold
- Exit: z-score returns to ±z_exit or max holding bars reached, whichever first
- SL: fixed ATR-multiple below/above entry
- TP: opposite z_exit level as price target (natural MR target)
- Maximum one position at a time (strict)
- Parameters use the Phase 3.5 audit "plateau center" values, not the spike maximum

### Task 3.2 — Realistic sizing for a modest product
Target parameters for a deployable version:
- Target annualized Sharpe 0.2-0.4 (based on Phase 4 audit at realistic parameters)
- Maximum expected drawdown 15% of account
- Position sizing: risk per trade = 0.5% of account, floor at broker's minimum lot
- Account size assumption: document explicitly (e.g., "assume $10k demo account")
- Leverage: no higher than 2× regardless of margin availability

### Task 3.3 — OANDA demo deployment via LTS
Integrate the plugin into the existing live-trading-system and deploy to OANDA demo account.

Configuration:
- Asset: EUR/USD
- Timeframe: daily (end-of-day decisions)
- Order types: market entry, stop-loss order, take-profit order
- Execution time: run once per day at a consistent UTC hour (e.g., 22:00 UTC right after NY close)

### Task 3.4 — Logging and monitoring infrastructure
For every trade:
- Timestamp of signal generation
- Signal features at entry (z-score, rolling mean, volatility)
- Entry price, SL level, TP level, position size
- Exit reason (SL hit, TP hit, max holding bars, manual), exit price
- Realized slippage vs modeled assumption
- PnL in pips, in account currency, and as % of account

Produce weekly auto-generated performance reports:
- Trades executed vs backtest expected frequency
- Cumulative PnL vs backtest simulation on same period
- Drawdown tracker with alerts at 5%, 10%, 15% thresholds

### Task 3.5 — Alert and pause protocol
Auto-pause the strategy (require manual resume) if:
- Account drawdown exceeds 15%
- 10 consecutive losing trades
- Realized slippage averages >2× modeled over 10+ trades
- Divergence from backtest distribution exceeds 2σ after 30+ trades

Pause is not kill. It triggers human review to decide continue, adjust, or stop.

### Task 3.6 — Minimum observation period
Commit to running demo for at least 90 calendar days before drawing conclusions. EUR/USD daily MR expected trade frequency is ~15-25 trades per year, so 90 days yields approximately 4-6 trades. This is not statistically conclusive but validates the operational pipeline.

The strategy runs in parallel with Phase 5 analysis; results feed the final project synthesis document.

### Kill criteria for Question 3
- Operational: if LTS cannot execute to OANDA reliably within 1 week of setup, this indicates infrastructure issues that must be addressed before any deployment is claimed viable.
- Statistical: if after 90 days realized performance is more than 3σ below backtest expectation, investigate slippage, spread, or broker-specific issues.

### Machine assignment
- **Omega**: plugin development, LTS integration, deployment, monitoring dashboard. Light load, runs as background service.
- Other machines: idle for Q3, available for Q1 parallelization.

### Deliverables
- Deployed live strategy on OANDA demo within 1 week of Question 3 start
- Weekly performance report auto-generated
- 90-day monitoring commitment with documented results

---

## Execution Schedule

```
Week 1:
  Mon-Tue:    Q1 portfolio construction (dragon + omega)   ‖  Q2 benchmark research (omega background)   ‖  Q3.1-3.2 implementation (omega background)
  Wed-Thu:    Q1 portfolio results, Q2 decision            ‖  Q3.3-3.4 deployment
  Fri:        Q3 first live trades possible; synthesis begins

Week 2+:      Q3 ongoing monitoring; other questions concluded
              Phase 5 final synthesis and decision produced end of week 1 based on Q1 + Q2
              Q3 results feed into monthly updates to synthesis
```

Questions 1 and 2 complete within Week 1. Question 3 produces its first operational results within Week 1 but statistical evaluation requires the 90-day observation window and feeds into an updated synthesis at that point.

---

## Phase 5 Final Synthesis Decision Matrix

At the end of Week 1, combine Q1 and Q2 results (Q3 provides ongoing data) into one of these terminal states:

| Q1 portfolio result | Q2 threshold finding | Terminal state |
|---------------------|---------------------|----------------|
| Outcome 1 (rescues at −0.5) | Case A (threshold too strict) OR Case B | **DEPLOY PORTFOLIO** — build multi-strategy product as Phase 6 |
| Outcome 2 (partial rescue) | Case A (recalibrate) | **DEPLOY RECALIBRATED PORTFOLIO** — at new threshold, portfolio is viable |
| Outcome 2 (partial rescue) | Case B (threshold correct) | **DEPLOY Q3 ONLY** — portfolio insufficient, EUR/USD MR is the modest product |
| Outcome 3 (no rescue) | Case A (recalibrate) | **REEVALUATE INDIVIDUAL CELLS** at new threshold; possibly deploy 1-2 individual cells |
| Outcome 3 (no rescue) | Case B (threshold correct) | **DEPLOY Q3 ONLY OR CLOSE** — depending on Q3 live performance after 90 days |
| Any outcome | Case C (multi-criteria) | **APPLY NEW CRITERIA** to existing results, then follow matching row |

Every terminal state is either "deploy something" (even if modest), "close with documented research," or "reevaluate with better-calibrated criteria." There is no "try another strategy family" option — Phase 5 is the convergence, not another exploration branch.

---

## Standing Rules for Phase 5

1. **No new strategies are developed in Phase 5.** All work uses cells already evaluated in prior phases. New feature engineering, new ML models, new asset classes are out of scope.
2. **Threshold recalibration must be evidence-based.** Any change from −0.5 to a different value requires external benchmark justification from Q2.
3. **Q3 deployment is not contingent on Q1/Q2 outcomes.** It runs regardless, because operational validation has independent value.
4. **No peeking at 2024-2025 for backtest tuning.** Live trading in Q3 obviously uses real 2025-2026 data, but any backtest recalibration uses prior data only.
5. **All questions converge to a terminal state.** The project moves to Phase 6 deployment or to documented closure. "Phase 5.5 to investigate something new" is explicitly not allowed without compelling justification from Q1-Q3 results.
6. **Honest documentation of negative results.** If the project closes, the closure document is itself a deliverable with research value.

---

## What Success Looks Like

Phase 5 succeeds if it produces an unambiguous terminal state. The options are:

**Terminal 1 — Portfolio deployment:**
- Multi-strategy portfolio of 5-10 cells passes either −0.5 original threshold or recalibrated threshold
- Phase 6 is the implementation and deployment of this portfolio
- Project has a working product

**Terminal 2 — Single-strategy deployment:**
- EUR/USD daily MR operates live with acceptable performance
- Modest scope, single asset, single strategy
- Project has a small working product

**Terminal 3 — Documented closure:**
- All three questions produce outcomes that do not support deployment at responsible scale
- Project closes with comprehensive research document covering findings across all phases
- Publishable research contribution on the difficulty of retail systematic trading with public data
- Portfolio piece demonstrating rigorous methodology

All three terminal states are legitimate outcomes. None require continuing to invest time in search of a better strategy. The evidence accumulated across Phases 0-4 is sufficient to commit to a terminal state in Phase 5.

---

## Files to Produce

- `phase5_q1_portfolio_results.json` and `.md` report
- `phase5_q2_benchmark_comparison.md`
- `phase5_q3_deployment_config.md` — deployment specification for EUR/USD MR
- `phase5_q3_live_monitoring.md` — updated weekly during monitoring period
- `phase5_synthesis.md` — final terminal-state decision with full reasoning
- If Terminal 3: `project_closure_research_document.md` — comprehensive findings across all phases

Phase 5 is complete when `phase5_synthesis.md` exists and names one of Terminal 1, 2, or 3 with supporting evidence.