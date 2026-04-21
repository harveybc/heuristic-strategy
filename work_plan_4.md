# Work Plan: Phase 4 — Parallel Tracks Toward Viable Deployment

**Prior state:** Phase 3.5 complete. 14 cells killed by extended-history worst-window criterion. Structural finding: no fixed-parameter strategy survives 25 years without at least one 2-year window of Sharpe < −0.5, even with perfect oracle. Framework itself is the limitation, not predictor quality.

**Purpose of this phase:** Pursue three parallel tracks that address distinct questions raised by Phase 3.5. No single track is betting the project — each has independent value and independent kill criteria.

**Execution context:** Three machines (omega RTX 4070 12GB, dragon RTX 4090 16GB, gamma RTX 5070 Ti 8GB). Tracks are largely independent and can run in parallel.

---

## Guiding Principle for This Phase

Phase 3.5 established that **static single-strategy frameworks are structurally insufficient**. Phase 4 does not try to fix this by finding a better predictor. It pursues three fundamentally different responses to that finding:

1. **Track A** asks: *does a working strategy already exist in our results, just with modest expectations?* Deploys the oracle-independent MR finding immediately to demo.
2. **Track B** asks: *would regime-awareness rescue the strategies that almost worked?* Tests the hypothesis with a regime oracle before committing to build one.
3. **Track C** asks: *are we being fooled by our own methodology?* Replicates well-known academic strategies to see if they survive our stress test.

Each track produces an actionable outcome regardless of whether the others succeed.

---

## Track A — Deploy Oracle-Independent Mean Reversion (1 week, omega)

### Rationale
Phase 3.5 Task 1 found that EUR/USD and USD/JPY daily MR have sign-flip probability < 0.5% — the oracle signal is irrelevant. These are pure rule-based technical strategies with demonstrated Edge Sharpe over B&H (+0.45 and +0.14). Their worst 2Y windows (−1.1 and −1.1) are poor but not catastrophic with proper sizing. They are **deployable now** and do not depend on any predictor.

This is the project's MVP: first live-capable output, validates the deployment stack end-to-end, starts accumulating real performance data in parallel with research.

### Task A.1 — Reproduce and parameter-audit
- Reload EUR/USD and USD/JPY daily MR results from Phase 3.5
- Confirm exact parameters (lookback, z_entry, z_exit, SL multiple, TP multiple, max holding bars)
- Run parameter-perturbation audit as in Phase 3.5 Task 2.2: ±50% sweep on each parameter, plot Sharpe surface
- Kill criterion: if perturbation reveals a sharp spike (narrow plateau), the result was overfit and Track A stops
- Expected outcome: smooth plateau consistent with a genuine technical phenomenon

### Task A.2 — Build Backtrader plugin
- Implement as a static-rule strategy plugin compatible with LTS architecture
- No prediction-provider dependency (this is the distinguishing feature)
- Include SL, TP, max holding bars, single-position-at-a-time constraint
- Unit tests for entry/exit logic at z-score thresholds
- Integration test on Phase 3.5 data to match reported Edge Sharpe within 0.05

### Task A.3 — Risk-calibrated position sizing
Given worst historical window of −1.1 Sharpe with 26–33% max drawdown:
- Size positions such that a full worst-window repeat produces ≤15% account drawdown
- This typically implies risking 0.5–1% of account per trade, depending on volatility normalization
- Document assumed account size, target drawdown tolerance, and resulting position size formula

### Task A.4 — OANDA demo deployment via LTS
- Deploy to OANDA demo account through existing live-trading-system
- Configure for EUR/USD and USD/JPY daily timeframe
- Set up logging: each trade records timestamp, signal basis, entry, SL, TP, exit, PnL, slippage vs expected
- Expected trade frequency: 15–30 per year per pair (modest)

### Task A.5 — Live monitoring protocol
- Weekly review: trades executed vs backtest distribution
- Monthly review: Sharpe, drawdown, trade stats vs backtest
- Alert criteria (auto-pause the strategy):
  - Drawdown exceeds 1.5× worst historical 2Y window's drawdown
  - 10 consecutive losing trades
  - Slippage exceeds 2× modeled assumption for 10+ trades
- Pause does not mean kill — it means review and decide explicitly

### Deliverable Track A
- Working Backtrader plugin deployed to OANDA demo
- First live trades within 1 week of Track A start
- Weekly performance report auto-generated
- Documented position sizing rationale

### Kill criteria
- Parameter plateau is narrow (spike, not plateau)
- Live performance after 3 months deviates from backtest by more than 2σ of expected distribution
- Slippage structurally higher than modeled (indicates broker/execution issue)

### Machine assignment
Omega. Task is light: plugin dev, deployment, monitoring. No GPU required.

---

## Track B — Test the Regime-Filter Hypothesis (1 week, dragon + omega)

### Rationale
Phase 3.5 showed XAU/USD weekly momentum has Edge Sharpe +1.5 in 4 of 5 macro regimes, but fails catastrophically in the QE era (2013–2019). If a regime filter could reliably detect and avoid hostile regimes, the worst-window problem might dissolve.

Before investing weeks building a real regime detector, we test the *ceiling*: if a **perfect regime oracle** doesn't solve the problem, no real detector will. This is a go/no-go experiment.

### Task B.1 — Define regime oracle
For each cell, implement a lookahead function that classifies every time point as:
- **Favorable**: next 2-year rolling window will have Sharpe ≥ +0.3 on this strategy
- **Hostile**: next 2-year rolling window will have Sharpe < −0.3 on this strategy
- **Neutral**: otherwise

This uses future information — it is not deployable, but it is the upper bound of what any regime filter could achieve.

### Task B.2 — Apply regime oracle to top 5 cells
Run on extended history:
- XAU/USD weekly momentum (highest edge, 1 failed regime)
- BTC/USD weekly momentum (highest edge in its window)
- XAU/USD daily momentum (more data points)
- XAG/USD weekly momentum (worst historical crash, good stress test)
- EUR/USD daily MR (oracle-independent, as control — regime filter should matter less here)

Strategy variants to evaluate:
- **V0 (baseline)**: trade always, per Phase 3.5 result
- **V1 (stand-aside)**: trade only when regime is favorable or neutral, flat when hostile
- **V2 (reverse)**: trade normally in favorable, reverse signal in hostile, flat in neutral
- **V3 (reduce size)**: trade full size in favorable, half size in neutral, flat in hostile

Per cell per variant, compute:
- Full-period Edge Sharpe
- Worst 2Y rolling window Sharpe
- Regime robustness
- Max drawdown
- Trades per year

### Task B.3 — Determine rescue viability
For each cell, compare V0 to V1/V2/V3. A rescue is viable if:
- Worst 2Y window Sharpe improves to > −0.5 (crosses the kill threshold)
- Edge Sharpe remains > +0.5 after filter application
- Filter does not reduce trades by more than 70% (strategy still has meaningful signal)

### Task B.4 — Feasibility analysis of real regime detection
For cells where perfect regime oracle rescues the strategy:
- Analyze which features would be needed to approximate the regime oracle
- Use Phase 3 feature store: macro (DGS10-DGS2, VIX, DXY), COT positioning, realized vol, regime clusters
- Train a simple classifier (logistic regression, gradient boosting) on macro features with target = regime label from oracle
- Report classification accuracy, precision, recall on out-of-sample windows
- Report strategy performance when using *real* classifier instead of oracle

### Task B.5 — Decision

**Outcome 1: Regime oracle rescues strategies AND real classifier achieves ≥70% of oracle benefit.**
- Proceed to full Phase 4 strategy development with regime filter as mandatory layer
- This is the path forward for the primary project goal

**Outcome 2: Regime oracle rescues strategies BUT real classifier captures < 50% of oracle benefit.**
- Regime is detectable in principle but not with available features
- Invest in better regime features (alternative data, sentiment, etc.) or accept that this path is gated on data acquisition

**Outcome 3: Regime oracle does NOT rescue strategies.**
- The worst-window problem is structural, not regime-based
- Pivot to Track C findings + Track A deployment + reconsider strategy frameworks entirely

### Deliverable Track B
- Report with V0/V1/V2/V3 results for 5 cells
- Regime classifier results with held-out evaluation
- Binary decision: "regime filter is viable" or "regime filter does not solve the problem"

### Kill criteria
- Perfect regime oracle fails to bring any cell's worst window above −0.5 (meaning regime-awareness is not the right frame)
- Real classifier for regime detection achieves < 55% balanced accuracy (barely better than chance, no usable signal)

### Machine assignment
- **Dragon (RTX 4090)**: Run Tasks B.1–B.3 for all 5 cells in parallel. Oracle application is CPU-bound but classifier training in Task B.4 benefits from GPU if using gradient boosting frameworks with GPU support (XGBoost GPU, LightGBM GPU).
- **Omega**: Consolidation, analysis, report writing.
- **Gamma**: Idle during Track B unless needed for overflow or cross-validation.

---

## Track C — Replicate Academic Strategies to Validate Methodology (1 week, gamma + dragon)

### Rationale
Phase 3.5 killed 14 of 14 cells. Before concluding that retail trading is impossible, we must validate that our methodology isn't too strict. If well-documented academic strategies also fail under our test, our criterion may need recalibration. If they pass, we have blueprints for what actually works.

### Task C.1 — Select academic strategies to replicate

**Strategy 1: Time-Series Momentum (Moskowitz, Ooi, Pedersen 2012)**
- Trades 58 futures/instruments (we use a reduced basket of available OANDA instruments)
- Sign of past 12-month return determines direction
- Position sized by inverse volatility
- Rebalanced monthly
- Reported historical Sharpe ~1.0 for the aggregate portfolio

**Strategy 2: Dual Momentum (Antonacci 2014)**
- Absolute momentum: invest only if asset 12m return > risk-free rate
- Relative momentum: among qualifying assets, invest in strongest
- Monthly rebalance
- Simple to replicate, widely studied, public performance data

**Strategy 3: Cross-Sectional Momentum within FX**
- Rank FX pairs by recent returns
- Go long top quartile, short bottom quartile
- Documented in Menkhoff et al. 2012 "Currency Momentum Strategies"

These three together span: multi-asset trend-following, simple momentum, and within-asset-class cross-sectional. If any of these survive our test framework, our methodology is validated.

### Task C.2 — Implement each strategy

For each strategy:
- Pure rule implementation, no ML, no predictors
- Same transaction cost model as Phase 3.5
- Same evaluation harness (rolling windows, regime robustness, worst-window criterion)
- Extended history 2000–2025 where available

### Task C.3 — Apply full Phase 3.5 kill criteria

Run each academic strategy through:
- Edge vs B&H check
- Rolling 2Y window analysis
- Regime-by-regime breakdown
- Worst window threshold

### Task C.4 — Interpret results

**Outcome 1: At least one academic strategy survives all criteria.**
- Our methodology is validated and we have working templates
- Phase 4+ can adopt or extend these strategies
- Specifically, the successful strategy's design features (multi-asset diversification, monthly rebalance, vol-inverse sizing) are the likely missing ingredients in our 14 killed cells

**Outcome 2: Academic strategies also fail our criteria.**
- Either our criteria are too strict (recalibrate worst-window threshold?) OR published academic results do not replicate out-of-sample (well-documented finding in the "alpha decay" literature)
- Compare the failure mode: if academic strategies fail similarly to our cells (bad QE-era performance, etc.), there is a common structural story worth investigating
- If academic strategies fail for different reasons, our criteria may need per-strategy calibration

### Task C.5 — Compare with our killed cells

For whichever academic strategy performs best (even if killed):
- Side-by-side comparison with closest analog among our Phase 3.5 cells
- What does the academic version do differently?
- Common patterns: number of assets, rebalance frequency, position sizing method, entry/exit rules
- This informs what to change in future strategy design

### Deliverable Track C
- 3 academic strategies implemented and evaluated with our harness
- Comparative table: our 14 cells vs 3 academic strategies, same metrics
- Recommendation: which (if any) academic design features should be adopted in future work

### Kill criteria
- None for the track itself — all three possible outcomes (validated, recalibration needed, no-replication) produce actionable information

### Machine assignment
- **Gamma (RTX 5070 Ti)**: Implement and backtest Strategy 1 (Time-Series Momentum) — most computationally intensive due to 58-asset basket
- **Dragon**: Implement and backtest Strategy 2 and 3
- **Omega**: Consolidation, comparison with Phase 3.5 cells, report

---

## Track D (Conditional) — GBP/USD 4h VRS Resolution (2 days, gamma)

### Rationale
This cell was neither killed nor confirmed in Phase 3.5 — it was excluded due to missing pre-2022 4h data. Its long-only negative + short-only negative + long+short positive pattern is statistically unusual and merits resolution.

### Task D.1 — Acquire 4h data 2003–2025
- Dukascopy historical tick data (free, reliable, 2003+)
- Aggregate ticks to 4h bars
- Validate against 2022–2025 yfinance data for period overlap
- Align session boundaries consistently (the same convention used in original data)

### Task D.2 — Apply Phase 3.5 full audit
- Oracle sensitivity with corrected noise model
- Extended history worst-window analysis
- Regime-by-regime breakdown
- Long/short decomposition over full history

### Task D.3 — Conclude
- If cell survives extended history: add to Track B candidate list
- If cell fails similarly to others: document and close
- If cell shows the same unusual long/short pattern over 22 years: flag for deeper investigation (possible structural market property worth understanding)

### Machine assignment
Gamma, running after Track C Strategy 1 completes.

---

## Execution Order and Dependencies

```
Week 1:
  Mon-Tue: Track A.1–A.3 (omega)         ‖  Track B.1–B.2 (dragon)       ‖  Track C.1–C.2 (gamma + dragon)
  Wed-Thu: Track A.4 deploy (omega)      ‖  Track B.3–B.4 (dragon+omega) ‖  Track C.3–C.4 (gamma + dragon)
  Fri:     Track A.5 monitoring setup    ‖  Track B.5 decision           ‖  Track C.5 comparison
           Track D starts on gamma

Week 2:
  Mon-Tue: Track D continues and concludes (gamma)
  Wed-Fri: Integration, synthesis, Phase 4 decision
```

All three main tracks are independent and can run concurrently. Track D is conditional on gamma availability after Track C.

---

## Phase 4 Decision Matrix

At the end of Week 2, synthesize results across tracks into a single decision:

| Track A result | Track B result | Track C result | Recommended Phase 5 Path |
|----------------|----------------|----------------|--------------------------|
| Plateau valid, deploying | Regime filter works | ≥1 academic strategy survives | Build regime-filtered momentum portfolio using academic design + MR as stable complement. This is the strongest outcome. |
| Plateau valid, deploying | Regime filter works | All academic fail | Regime filter path is valid but our results are unusual; investigate why academic strategies fail in our harness before full commitment. |
| Plateau valid, deploying | Regime filter fails | ≥1 academic strategy survives | Abandon regime filter path. Adopt academic strategy design. MR continues as deployed baseline. |
| Plateau valid, deploying | Regime filter fails | All academic fail | Retail trading with our data may be structurally harder than literature suggests. Deployed MR continues; research phase ends or pivots to alt-data. |
| Plateau fragile, not deploying | Regime filter works | ≥1 survives | Focus entirely on regime-filtered momentum + academic replication. Restart strategy development. |
| Plateau fragile, not deploying | Regime filter fails | All academic fail | Strong signal to close project with documented negative result. |

---

## Standing Rules for Phase 4

1. **No track depends on another succeeding.** Each produces value independently.
2. **Paper trading is not optional for Track A.** The purpose is operational validation, not just backtest.
3. **Regime oracle in Track B must use strict lookahead.** Any leakage invalidates the ceiling test.
4. **Academic strategies in Track C are implemented as published, not tuned.** We are testing replication, not optimization.
5. **Worst-window threshold of −0.5 remains the default** for evaluating Track B/C/D, but tracks must also report results at −0.75 and −1.0 to support recalibration discussion.
6. **All tracks report to the same harness.** Any new performance metric must be added to the shared evaluation module, not computed ad hoc.
7. **If Track B finds regime filter works, do not skip to production.** Still needs a real classifier, still needs out-of-sample validation on 2024–2025.
8. **Document before extending.** Each track produces a written report before any follow-up work is considered.

---

## What Success Looks Like

At the end of Phase 4 (2 weeks), one of the following outcomes is realized:

**Outcome 1 — Strong success:**
- EUR/USD and USD/JPY MR deployed live on OANDA demo
- Regime filter validated as viable path
- At least one academic strategy replicates
- Clear Phase 5 plan: build regime-aware momentum strategy, MR continues as complement

**Outcome 2 — Partial success:**
- MR deployed to demo, generating data
- Regime filter or academic replication provides direction for further work
- Clear narrowed scope for Phase 5

**Outcome 3 — Deployed-only:**
- MR deployed and functional but neither regime filter nor academic strategies offer new paths
- Project continues with demo MR as primary output
- Research effort paused or redirected to alternative data sources

**Outcome 4 — Closure with evidence:**
- All tracks conclude negatively or inconclusively
- Project has comprehensive documented findings about the limits of retail strategy research with public data
- Close with publishable or portfolio-worthy research output

All four outcomes preserve the value of work done. None require abandoning the prior investment. The project is now too mature to either force success or pretend failure.

---

## Files to Produce

Each track produces at minimum:
- `phase4_track_{a,b,c,d}_report.md` — narrative findings, decision outcomes, kill criteria evaluated
- `phase4_track_{a,b,c,d}_results.json` — structured metrics for programmatic comparison
- Code artifacts (plugins, strategy implementations, regime oracle) in their respective repo paths
- Update to feature store or evaluation harness only where additions are genuinely reusable across tracks

Phase 4 concludes with a single `phase4_synthesis.md` using the decision matrix above to determine Phase 5 direction.