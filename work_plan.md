# Work Plan: Profitable Trading Agent Research

## Context Reminder

**Goal (non-negotiable):** Build a profitable automated trading agent that operates under realistic broker conditions (OANDA-style REST API, live or simulated). The existing `live-trading-system` (LTS) repo already handles execution against such APIs using Backtrader strategies fed by a `prediction-provider` that wraps pre-trained Keras models.

**What is flexible:**
- Asset class (EUR/USD was starting point, not requirement)
- Timeframe / bar periodicity
- Strategy style: static rule-based with predictions OR dynamic (NEAT + action replay)
- ML / no ML
- Causal inference / no causal inference
- Pattern Day Trading constraint (can be applied *after* finding an edge; not a design constraint during research)

**What is fixed:**
- Must be executable through a broker REST API (OANDA compatible)
- Single position at a time
- SL and TP on every order
- Reusable architecture: prediction-provider → strategy plugin → Backtrader → LTS execution
- Re-training cadence is not a concern — pre-trained models can be refreshed weekly or as needed

**Prior evidence gathered (DO NOT repeat):**
- EUR/USD at 1h with public features has ~no exploitable edge (multiple strategies failed)
- Point prediction (regression on returns or levels) cannot beat naive on EUR/USD 1h
- Binary classification of SL/TP hits lacks required F1
- Causal rules on regime clusters overfit
- Event-driven PoC on NFP showed weak/no signal at 1h (but was methodologically flawed — no real surprise data, wrong window alignment)
- Cross-asset audit (2018–2025, with costs, rolling windows):
  - ETH/USD, BTC/USD, CL (crude) show momentum edge, but ETH and CL show clear decay in 2023–2025
  - SPY mean reversion does not beat buy-and-hold
  - EUR/USD mean reversion shows Sharpe ~0.63 net (unaudited, suspicious)
  - No asset has statistically significant Hurst vs shuffled null
  - Oracle sensitivity test was inconclusive (resolution problem at high noise)

---

## Strategic Principles for This Phase

1. **Every hypothesis must have a kill criterion defined before the experiment.** Write down in advance what result would make you stop pursuing that path. If you can't define it, you don't understand the hypothesis.

2. **Broaden the search space before deepening any one path.** We have evidence that the current search space (EUR/USD 1h, predict-then-trade) is nearly empty. Before committing weeks to one new direction, sweep multiple axes cheaply.

3. **Oracle-first methodology is mandatory.** For every candidate strategy/asset/timeframe combination, first verify that a perfect oracle makes money net of realistic costs. Only then invest in building predictors.

4. **Reuse the existing pipeline.** Autoencoder, feature extractor, prediction-provider interface, NEAT/DEAP optimizer with early stopping, OLAP cube, Backtrader plugin system — all of these transfer across assets and timeframes. Do not rebuild.

5. **Transaction costs are part of the signal, not an afterthought.** Every backtest from this point forward includes spread + commission + slippage model appropriate to the asset class.

6. **Regime stability beats peak performance.** A Sharpe 0.4 strategy that works in 70% of 2-year windows is more valuable than a Sharpe 0.8 strategy that works in 40% of windows.

---

## Phase 0: Fix the Diagnostic Infrastructure (3–5 days)

Before any new research, the tools we use to evaluate candidates must be trustworthy.

### Task 0.1 — Rebuild the oracle sensitivity test

The prior test hit ceiling at noise=5.0 for all assets, which is uninformative. Rework:

- Parameterize noise not as multiplicative factor on signal magnitude but as **independent additive Gaussian noise on the oracle's predicted log-return**, measured in units of the **target horizon's realized volatility**.
- Noise grid: `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]` in units of σ(return | horizon).
- Output per asset/timeframe: the noise level at which net Sharpe (after costs) drops below 0.3. Call this the **noise budget**.
- The noise budget tells us how accurate a predictor needs to be. A large noise budget = forgiving market. A tiny noise budget = requires near-perfect prediction.

**Kill criterion for the methodology:** if after this rework the test still saturates for all assets at the same noise level, the oracle test is fundamentally broken and must be rethought.

### Task 0.2 — Standardize the cost model

Create a single `transaction_cost_model.py` module used by every backtest going forward. Per asset class:

| Asset class | Spread (bps, round-trip) | Commission | Slippage model |
|---|---|---|---|
| FX major (EUR/USD, USD/JPY, etc.) | 1.5 | 0 | 0.3 bps × sqrt(volatility ratio) |
| FX cross / emerging | 4 | 0 | 0.5 bps × sqrt(vol ratio) |
| Equity index ETF (SPY) | 1 | $0.005/share | 0.5 bps |
| Equity index futures (ES, NQ) | 0.5 tick | $2.25/round | 0.5 tick |
| Commodity futures (CL, GC) | 1 tick | $2.50/round | 1 tick in stress |
| Crypto major (BTC, ETH) | 10 | 10 bps taker | 15 bps in volatile regimes |
| Crypto alt (SOL, BNB) | 20 | 10 bps | 30 bps in volatile regimes |

Slippage model adds extra cost proportional to current-bar volatility relative to trailing-30-day average.

### Task 0.3 — Rolling-window evaluation harness

Every strategy evaluation from this point on reports:

- Full-period Sharpe (net of costs)
- Rolling 2-year Sharpe with 6-month step
- **Fraction of rolling windows with Sharpe > 0.3** (call this **regime robustness**)
- Max drawdown full-period
- Max drawdown worst 2-year window
- Trades per year
- Hit rate, avg winner, avg loser

A strategy is only "interesting" if regime robustness ≥ 0.5 AND full-period Sharpe ≥ 0.4 net of costs.

---

## Phase 1: Broad Sweep — Asset × Timeframe × Strategy Family (2 weeks)

This is the core exploration. We systematically evaluate combinations we have NOT tried.

### Task 1.1 — Define the search grid

**Assets (12):**
- FX majors: EUR/USD, USD/JPY, GBP/USD, AUD/USD
- FX cross: AUD/JPY, EUR/JPY, GBP/JPY
- Commodities: XAU/USD (gold), XAG/USD (silver), CL (crude via CFD or futures proxy)
- Crypto: BTC/USD, ETH/USD

Note: all 12 are tradeable via OANDA or similar broker APIs (OANDA offers FX, metals, oil CFDs, and some crypto CFDs depending on jurisdiction; verify exact coverage for your deployment region).

**Timeframes (5):**
- 15 min
- 1 h
- 4 h
- Daily
- Weekly

**Strategy families (6), each with ORACLE first, then real predictor later:**
1. **Directional momentum:** enter in direction of N-bar return if magnitude > threshold
2. **Mean reversion:** enter counter to N-bar move if z-score > threshold
3. **Breakout:** enter on break of N-bar high/low with volatility filter
4. **Carry + momentum composite:** (for FX only) interest differential + momentum
5. **Volatility regime switch:** momentum in low-vol regime, MR in high-vol regime (classified by your existing clustering)
6. **Event-driven (macro calendar):** only for FX and gold; requires macro surprise data

### Task 1.2 — Oracle sweep

For each (asset × timeframe × strategy family) = 12 × 5 × 6 = **360 cells**, compute:

- Net Sharpe with perfect oracle
- Net Sharpe with oracle + noise at levels from Phase 0 grid
- Trades per year (to check feasibility vs PDT if ever applied)

**Implementation note:** each cell is a fast computation — the oracle knows future prices so there is no model training. Expect total runtime of a few hours, not days.

Rank cells by **noise budget** (from Phase 0 methodology): the noise level at which Sharpe drops below 0.3.

### Task 1.3 — Naive baseline per cell

For each of the top 30 cells by noise budget, also run:
- Buy-and-hold Sharpe (for the period)
- Random entry with same average trade frequency and SL/TP (100 seeds, report distribution)

Only keep cells where the oracle Sharpe materially exceeds buy-and-hold AND random baseline (p < 0.05 vs random distribution).

**Kill criterion:** if fewer than 5 cells survive these filters, the search space is empty and we have evidence the retail trading problem may be unsolvable with public data. In that case, stop and reconsider scope (see Phase 5).

### Deliverable Phase 1
A ranked table of surviving cells with: asset, timeframe, strategy family, oracle Sharpe, noise budget, regime robustness, trades/year, buy-and-hold comparison.

---

## Phase 2: Deep Audit of the EUR/USD MR Anomaly (parallel to Phase 1, 3–5 days)

The ~0.63 net Sharpe on EUR/USD mean reversion is the one genuinely surprising signal from prior work. It must be either confirmed or killed before being trusted.

### Task 2.1 — Full characterization
Compute with the standardized harness:
- Which specific MR parameters produced the 0.63? (lookback, entry z-score, exit z-score, SL, TP, holding horizon)
- Trades per year, hit rate, avg winner vs loser
- Rolling 2-year Sharpe across 2005–2025 (use full history, not just 2018+)
- Max drawdown and underwater period
- Performance in specific stress regimes: 2008 crisis, 2011–12 euro crisis, 2015 SNB shock, 2020 COVID, 2022 inflation shock

### Task 2.2 — Robustness to parameter perturbation
For each parameter, sweep ±50% around the "optimum" and plot Sharpe. A real edge has a smooth plateau. An overfit has a sharp spike. If performance collapses outside the exact parameters, it's overfitting.

### Task 2.3 — Cost sensitivity
Re-run at spread levels 1, 2, 3, 5 bps round-trip. A genuine edge survives 3 bps. A fragile one dies at 3 bps.

### Task 2.4 — Structural explanation test
Ask: **why** would EUR/USD mean-revert at this specific timeframe and parameters? Candidate explanations:
- Positioning extremes (retail traders buy dips / sell rips) → predict via COT or retail sentiment data
- Liquidity asymmetries at specific times of day → check time-of-day distribution of trades
- Session-boundary effects (London close, NY open) → test with session filter

A signal without a structural explanation is a signal you don't trust.

**Kill criterion for Phase 2:** if rolling robustness < 0.4 OR parameter plateau is narrow OR signal dies at 3 bps cost OR no structural explanation exists, discard EUR/USD MR and do not revisit.

---

## Phase 3: Exogenous Data Enrichment (1 week, parallel-safe)

Regardless of which assets/timeframes survive Phase 1, add these features to the research dataset. Prior investigation established these sources — now is the time to actually integrate.

### Task 3.1 — Always-on macro features
- DGS10, DGS2, T10Y2Y, DFF from FRED API (free, daily, propagate to intraday bars)
- ECB deposit rate, Bund 10Y yield (ECB SDW / Bundesbank, free)
- DXY synthetic (from your own FX data if available for the 6 basket pairs; otherwise Twelve Data)
- VIX from Yahoo Finance (free, daily)
- CFTC COT reports for FX and commodities (free, weekly, propagate as step function)

### Task 3.2 — Economic calendar with surprise data
- Obtain 10+ years of: actual, forecast (pre-release consensus), previous for high-impact events: NFP, US CPI, US PCE, FOMC decisions, EZ CPI, ECB decisions, key PMIs
- Source: 1 month of Trading Economics Pro API (~$50) OR scraped ForexFactory history
- Normalize surprises as z-scores using historical distribution of (actual − forecast) for that specific event
- Store as event table keyed on timestamp_utc

### Task 3.3 — Crypto-specific features (if crypto survives Phase 1)
- On-chain: active addresses, transaction count, exchange net flows (Glassnode free tier or CryptoQuant)
- Futures basis and funding rates (Binance, Bybit public APIs, free)
- Grayscale / ETF flow data if BTC/ETH

### Deliverable Phase 3
A unified feature store aligned to each (asset, timeframe) combination from Phase 1 survivors, containing price, technical features (existing), regime cluster (existing), and new exogenous features.

---

## Phase 4: Strategy Development on Surviving Cells (3–4 weeks)

Only for cells that passed Phase 1 kill criteria.

### Task 4.1 — Real predictor training per surviving cell
For each (asset, timeframe, strategy family) that survived:
- Train the predictor implied by that strategy family (direction classifier, quantile regressor, or conditional response model)
- Use your existing pipeline (autoencoder features + regime clusters + exogenous features from Phase 3)
- Evaluate predictor quality vs naive AND vs the noise budget from Phase 0
- **If the real predictor's effective noise exceeds the cell's noise budget, stop working on that cell.** No amount of strategy tuning compensates for predictor quality below threshold.

### Task 4.2 — Static strategy with real predictor
Implement as a new Backtrader plugin per surviving cell, wrap in prediction-provider interface, backtest with standardized harness.

### Task 4.3 — Dynamic strategy with NEAT
For the top 2–3 cells from Task 4.2, implement NEAT-based dynamic parameter control over the strategy (as already done in prior work, but with early stopping on validation).

Key improvements over prior NEAT attempts:
- Walk-forward validation, not single split
- Fitness function = rolling-window Sharpe minus drawdown penalty, not just total PnL
- Penalize number of trades beyond broker-realistic thresholds
- Early stopping with patience=10 epochs on validation champion

### Task 4.4 — Optional: event-driven overlay
If event-driven from Phase 1 showed signal AND Phase 3 delivers surprise data AND timeframe allows (4h+), build event-driven variant of top cells as additional strategy to combine.

### Deliverable Phase 4
One or more candidate strategies with:
- Full-period net Sharpe > 0.5
- Regime robustness > 0.6
- Validated on out-of-sample test year
- Implemented as LTS-compatible Backtrader plugin with prediction-provider

---

## Phase 5: Honest Decision Point (after Phase 1, formally revisited after Phase 4)

After Phase 1, you will know whether any cells have oracle-level edge net of costs. After Phase 4, you will know whether real predictors can realize that edge.

### Decision tree

**If Phase 1 yields zero surviving cells:**
Stop. Document findings. Possible pivots:
- Broader asset universe (individual equities, less liquid FX crosses)
- Alternative data sources beyond this plan (social sentiment, news LLM embeddings, satellite)
- Completely different trading paradigm (market making, arbitrage between correlated assets, statistical arbitrage across a basket)

**If Phase 1 yields surviving cells but Phase 4 predictors fail to hit noise budget:**
This means the edge is real but unrealizable with current modeling approach. Options:
- Invest in better feature engineering (domain-specific, alternative data)
- Try different model families (sequence models for price, graph models for cross-asset, LLM-based sentiment)
- Reduce ambition: can a lower-Sharpe but robust version be profitable at sufficient leverage within broker limits?

**If Phase 4 yields a viable strategy:**
Move to paper trading on OANDA demo for 2–3 months minimum before any real capital. Monitor live performance vs backtest expectations. Any deviation > 2σ from backtest distribution = stop and diagnose.

---

## Execution Order and Parallelism

```
Week 1:     Phase 0 (diagnostic fix) — blocks everything else
Week 2–3:   Phase 1 (broad sweep)   ‖   Phase 2 (EUR/USD MR audit)   ‖   Phase 3 (data enrichment)
Week 4:     Integrate Phase 3 data into surviving Phase 1 cells
Week 5–7:   Phase 4 (strategy development on survivors)
Week 8:     Phase 5 decision, documentation, demo deployment if viable
```

Phases 1, 2, 3 run in parallel because they share no critical path — Phase 1 is compute-bound on oracle sweeps, Phase 2 is a focused audit, Phase 3 is data engineering.

---

## Standing Rules for the Agent

1. **Never skip the standardized cost model.** Any backtest result reported without costs is invalid and will be rejected.
2. **Never report a single-split Sharpe.** Always rolling-window with regime robustness.
3. **Always state the kill criterion before running.** If a result is ambiguous and no kill criterion was set, it means the experiment was poorly designed.
4. **Report negative results as first-class outcomes.** "X does not work" with evidence is a deliverable, not a failure.
5. **Do not tune on the test set.** 2024–2025 is reserved as final test. Hyperparameter selection uses the designated validation periods only.
6. **Keep a running log of cells killed and why.** This prevents accidentally re-exploring dead paths.
7. **Prefer simpler strategies with structural explanations over complex ones with statistical fit.** If you can't explain *why* the edge exists, assume it's overfitting.
8. **Ask before scope-expanding.** Do not add new assets, timeframes, or strategy families mid-plan without explicit confirmation.

---

## What Success Looks Like

At the end of this plan, you have one of three outcomes, all of which are valuable:

1. **Viable strategy found:** one or more (asset, timeframe, strategy) combinations with net Sharpe > 0.5, regime robustness > 0.6, validated on held-out test year, deployable via LTS to OANDA demo. Objective achieved.

2. **No viable strategy but clear evidence:** comprehensive documentation of what doesn't work across 60+ combinations, with oracle budgets, cost sensitivities, and regime stability for each. This is a publishable research contribution and a strong professional portfolio piece, and it informs whether a completely different approach (alt-data, different timescale, market-making) is warranted.

3. **Partial: edge exists but not realizable with current predictors.** Forward plan becomes focused on model improvements or alternative data, with a clear quantitative target from the noise budget analysis.

All three outcomes are better than the current state, which is "some things don't work on EUR/USD 1h." The current plan transforms that anecdote into a systematic map of the problem.