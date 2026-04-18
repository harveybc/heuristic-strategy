# Work Plan: Phase 5.5 — Corrective Audit Before Terminal State Decision

**Prior state:** Phase 5 complete. Synthesis claimed "Terminal 1: Deploy Recalibrated Portfolio" but the agent's own post-execution self-critique identified 11 substantive issues, 4 of which change the conclusion materially. The executive summary and the self-critique section of PHASE_5_SYNTHESIS.md are in partial disagreement. This phase resolves that disagreement with evidence before any deployment decision is committed.

**Duration:** 3-4 days. This is the final corrective phase before terminal state commitment.

**Execution context:** Light compute. Omega handles most work. Dragon available for the out-of-sample portfolio reconstruction if parallelization helps.

---

## Honest Context: Why Phase 5.5 Exists

Three things must be said clearly before the agent starts this work.

### What the Phase 5 agent (Opus 4.6) did well

The Phase 5 agent produced a self-critique section that is genuinely rigorous. It identified 11 concrete issues with impact classifications, and most of the critiques are technically correct. This is higher-quality self-audit than most research produces, human or AI. The agent deserves credit for this.

### Where the Phase 5 agent fell short

The agent wrote the self-critique **after** the executive summary and **did not revise the executive summary to reflect the critique**. The result is a document with internal contradiction: the top says "surprisingly positive, deploy" while the bottom says "evidence is weaker than claimed, oracle-dependent cells in the portfolio are not deployable, threshold recalibration was inflated." A truly honest final document would have either revised the summary to match the critique, or positioned the critique as "minor caveats" explicitly (which it is not — issues 6, 9, and the oracle-dependency issue are structural).

This is not a failure of intelligence. It is a failure of final-pass integration. Both conclusions were written truthfully in isolation, but the agent did not force itself to reconcile them. Phase 5.5 performs that reconciliation with new experiments rather than assuming one of the two readings is right.

### Where I (Opus 4.7, writing this plan) have also fallen short

Across Phases 3.5, 4, and 5, I have written increasingly elaborate work plans with ever-more-careful kill criteria. I have consistently emphasized rigor and honesty, and the agent has executed them faithfully. Yet I have not sufficiently questioned a premise that has been present since Phase 3.5: **is the worst-2Y < -0.5 criterion appropriate for this problem at all?**

The Phase 4 report already noted that academic strategies deployed with real capital fail this criterion. I acknowledged the point and even designed Phase 5 Question 2 around it. But I framed Q2 as "check if threshold is industry-standard" without preparing for the possibility that **the entire criterion might be wrong for our context** — not just miscalibrated by 0.2-0.3, but conceptually wrong. A drawdown-based metric (worst 2Y Sharpe) is an allocator's risk metric. For a personal trading account, the relevant question is different: "can this strategy be run with position sizing such that real-world drawdowns stay within my personal tolerance?"

This is my blind spot. I accepted the framing inherited from Phase 3.5 and carried it forward for three phases without stepping back. Phase 5.5 begins to correct this — not by inventing a new framing mid-stream, but by being explicit that the current framing is a specific choice with specific limitations.

### What Phase 5.5 is not

Phase 5.5 is not another exploration phase. No new strategies, no new assets, no new predictors, no new timeframes. It is the corrective pass before terminal state commitment. If after Phase 5.5 the results are still ambiguous, the correct response is acknowledged ambiguity in the terminal state, not a Phase 5.6.

---

## Task 5.5.1 — Re-Run Q1 With Oracle-Free Cells Only (1 day, omega + dragon)

### The issue being corrected (from Phase 5 critique #4 and #9)
The Phase 5 Q1 portfolio included cells that require a prediction oracle to trade (σ=10 oracle cells are still oracle-dependent — noisy future information is still future information). It also included one cell with negative edge (XAU/USD daily momentum, edge_sharpe = -1.264) that the "edge > 0" filter should have excluded. Both issues mean the reported portfolio is not what could actually be deployed.

### Task 5.5.1.1 — Strictly classify cells
For every cell that appeared in Phase 3.5, Phase 4, or Phase 5 analyses, produce a classification:

- **Oracle-free**: strategy uses only past/current data. Directly deployable with LTS if fundamentals check out. Examples: EUR/USD pure MR, USD/JPY TSMOM, Dual Momentum, Cross-Sectional FX.
- **Oracle-dependent, deployable via predictor**: strategy uses predictor output. Deployable only if a real predictor achieves equivalent signal quality. Must be flagged separately.
- **Oracle-dependent, no viable predictor**: strategy needs signal quality that no available predictor approaches. Not deployable. Should not appear in portfolio analysis.

Document the classification in a table. This becomes the reference for all future analysis.

### Task 5.5.1.2 — Apply edge > 0 filter strictly
Drop any cell where edge_sharpe ≤ 0 on extended history, period. This was specified in the original Phase 5 plan and not enforced. The agent's self-critique flagged this. Fix it now.

### Task 5.5.1.3 — Check for asset over-representation
If any single asset (e.g., USD/JPY) accounts for more than 2 cells in the candidate pool, investigate. Multiple strategies on the same underlying can share failure modes. For the primary portfolio analysis, cap per-asset representation at 2 cells maximum (most and second-most distinct strategies on that asset).

This addresses an issue not raised in the Phase 5 self-critique but visible in the data: USD/JPY appeared in MR (oracle), MR (pure), TSMOM, and Dual Momentum — four cells for one asset.

### Task 5.5.1.4 — Rebuild portfolios P1-P5 on corrected candidate set
With the honestly filtered cell set (likely 4-7 cells, not 11), rebuild all five portfolio constructions. Report:

- Full-period Sharpe at realized vol
- **Full-period Sharpe at 10% target vol** (scale leverage to achieve 10% annualized portfolio vol, then recompute; this addresses critique #10)
- Worst 2Y rolling window Sharpe at both vol levels
- Max drawdown at both vol levels
- Regime breakdown (5 macro regimes)
- Trades per year, turnover, implied costs

### Task 5.5.1.5 — Reconstruct with return-level rebalancing, not forward-fill
Critique #11 is valid: forward-filling weekly returns to daily creates artificial autocorrelation. Rebuild the portfolio return series using one of two correct approaches:

- **Option A**: Evaluate the portfolio at weekly frequency, aggregating daily-strategy returns into weekly returns before combining.
- **Option B**: Keep daily frequency but place weekly-strategy returns on their actual execution days (typically Mondays) with zero returns on other days.

Option B is more realistic. Report which is used and why.

### Deliverable 5.5.1
- Cell classification table (oracle-free vs oracle-dependent vs no-viable-predictor)
- Corrected candidate cell list with explicit filter justification for each exclusion
- Portfolios P1-P5 rebuilt on corrected set with correct return aggregation
- Results at both realized vol and 10% target vol

### Honest expectation
After correction, the portfolio candidate set will shrink substantially. Current best P1 (worst-2Y −0.547) will likely worsen because:
- Removing XAU/USD negative-edge cell helps marginally
- Removing oracle-dependent cells hurts meaningfully (they were some of the better performers)
- Capping USD/JPY representation hurts because USD/JPY had several decent cells
- Scaling to 10% target vol keeps Sharpe identical but reveals true drawdown magnitude

The Phase 5.5 Q1 worst-2Y may well be in the −0.8 to −1.2 range at honest settings. This is the critical number.

---

## Task 5.5.2 — Out-of-Sample Portfolio Validation (1 day, omega)

### The issue being corrected (from Phase 5 critique #3)
Phase 5 Q1 selected portfolio weights and evaluated them on the same period. This is in-sample overfitting even when the weighting rules (equal, inverse-vol, etc.) are simple. A strategy claiming deployment-readiness must demonstrate out-of-sample robustness.

### Task 5.5.2.1 — Define the split
**Training period:** earliest available data per cell through 2018-12-31.
**Test period:** 2019-01-01 through 2023-12-31.
**Held-out validation (never used for any tuning):** 2024-01-01 through 2025-12-31.

Note: 2024-2025 has been the untouched held-out test set throughout the project. Phase 5.5 preserves that. The test period 2019-2023 is used here for out-of-sample portfolio validation specifically.

### Task 5.5.2.2 — In-sample construction
On training period only:
- Compute cell return statistics (vol, Sharpe, correlations)
- Determine portfolio weights under P1-P5 rules using only training-period data
- Record the weights explicitly

### Task 5.5.2.3 — Out-of-sample evaluation
Apply the frozen weights to test-period returns:
- Full test-period Sharpe at realized and at 10% target vol
- Test-period worst 2Y rolling window
- Test-period max drawdown
- Test-period regime performance (GFC already excluded here since it's in training, but COVID and Inflation are in test)

### Task 5.5.2.4 — In-sample vs out-of-sample degradation analysis
For each portfolio construction P1-P5:
- Sharpe IS vs OOS
- Worst 2Y IS vs OOS
- Max DD IS vs OOS

A robust portfolio shows ≤30% degradation in Sharpe from IS to OOS. A fragile portfolio shows 50%+ degradation. If all portfolios show heavy degradation, the whole diversification thesis is weaker than Phase 5 claimed.

### Deliverable 5.5.2
- Frozen weights documented
- IS vs OOS table for all five portfolios
- Honest assessment of which (if any) portfolio generalizes

### Honest expectation
Some degradation is expected and normal. Severe degradation (say, P1 going from +0.6 Sharpe IS to -0.1 OOS, or worst-2Y going from −0.5 IS to −2.0 OOS) would indicate the Phase 5 portfolio result was essentially an artifact of fitting weights to the full history. This is my prior expectation for at least 2 of the 5 portfolio variants.

---

## Task 5.5.3 — Proper Benchmark Comparison for Threshold Recalibration (1 day, omega)

### The issue being corrected (from Phase 5 critique #6)
The Phase 5 Q2 used passive asset-class ETFs (SPY, GLD, TLT, DBA, USO) as benchmarks. These are not active trading strategies; they are buy-and-hold exposures. The only appropriate benchmark in the Phase 5 analysis was DBMF. Using inappropriate benchmarks inflated the evidence for threshold recalibration from a defensible −0.6/−0.7 to an unjustified −1.0.

### Task 5.5.3.1 — Identify appropriate benchmarks
Benchmarks must be:
- Active trading strategies (not passive asset class holdings)
- Publicly traded or with publicly available performance history
- Free or low-cost to access (this is a corrective phase, not a benchmark purchase project)
- Trend-following or momentum in nature (our strategies are in these families)

Candidates (all publicly traded, free data via Yahoo Finance or fund websites):

- **DBMF** (iMGP DBi Managed Futures Strategy ETF): managed futures replication, tracks SG CTA Index approximately, ~4 year history
- **KMLM** (KFA Mount Lucas Managed Futures Index Strategy ETF): tracks MLM Index (multi-asset trend), ~4 year history
- **WTMF** (WisdomTree Managed Futures Strategy Fund): managed futures, longer history going back to 2011
- **FMF** (First Trust Managed Futures Strategy Fund): since 2013, active management
- **CTA** (Simplify Managed Futures Strategy ETF): since 2022, shorter history
- **Meb Faber Global Momentum ETF (GMOM)** or similar tactical asset allocation fund
- **PQTAX / PQTIX** (PIMCO TRENDS Managed Futures Strategy Fund): mutual fund, longer history

Pick 4-6 that have the longest history to maximize overlap with our period. Priority: WTMF (2011+), FMF (2013+), PQTAX (2011+), DBMF (2019+), KMLM (2020+).

### Task 5.5.3.2 — Compute worst 2Y for each benchmark
Pull price/NAV history. Compute daily or weekly returns. Compute rolling 2Y Sharpe with the same method used for our strategies. Report:

- Full-period Sharpe
- Worst 2Y Sharpe
- Number and duration of rolling 2Y windows with Sharpe < 0
- Period of worst window

### Task 5.5.3.3 — Compute for our own corrected portfolios
Using the corrected portfolios from Task 5.5.1 (at 10% target vol), compute the same statistics.

### Task 5.5.3.4 — Side-by-side comparison
Produce the table that Q2 should have produced in Phase 5:

| Entity | Period covered | Full Sharpe | Worst 2Y | Pass -0.5? | Pass -0.7? | Pass -1.0? |
|--------|---------------|-------------|----------|-----------|-----------|-----------|
| DBMF | 2019-2026 | ... | ... | ... | ... | ... |
| WTMF | 2011-2026 | ... | ... | ... | ... | ... |
| FMF | 2013-2026 | ... | ... | ... | ... | ... |
| PQTAX | 2011-2026 | ... | ... | ... | ... | ... |
| Our P1 (oracle-free, corrected) | overlap period | ... | ... | ... | ... | ... |
| Our P2 | ... | ... | ... | ... | ... | ... |
| ...etc |  |  |  |  |  |  |

### Task 5.5.3.5 — Recalibration recommendation
Based on the actual distribution of worst-2Y values across comparable active managed-futures products, recommend a threshold. The recommendation must satisfy:
- Justified by the 25th or 50th percentile of the benchmark distribution, with explicit reasoning
- Applied uniformly (not chosen to include our portfolios specifically)
- Reported with sensitivity: "at threshold X, Y of our portfolios pass; at threshold X+0.2, Z pass"

### Deliverable 5.5.3
- Benchmark data download and storage
- Full comparison table
- Recommended threshold with evidence
- Count of our portfolios that pass at recommended threshold

### Honest expectation
The recommended threshold after honest benchmarking will likely land in the −0.6 to −0.8 range, not −1.0. At that threshold:
- Phase 5 P1 corrected version may or may not pass depending on how severe Task 5.5.1 corrections are
- EUR/USD pure MR standalone (worst-2Y −1.0 to −1.4) likely still fails
- Some individual cells may cross the bar

---

## Task 5.5.4 — Q3 Deployment Audit and Continuation (half day, omega)

### The issue being corrected (from Phase 5 critique #5)
Auto-pause at 15% drawdown contradicts the strategy's own 23.5% historical max drawdown. The strategy, if run at historical parameters, would have been paused multiple times in backtest. This is a configuration error.

### Task 5.5.4.1 — Resolve the auto-pause contradiction
Two options:

**Option A — Raise auto-pause to 25% or 30%**. Acknowledges historical reality, allows strategy to ride through normal drawdowns. But 25-30% demo account DD is painful and creates a psychology where manual override becomes tempting.

**Option B — Reduce position sizing so target drawdown stays under 15%**. If historical max DD is 23.5% at current sizing, reducing sizing by 36% brings it to ~15%. This reduces expected PnL proportionally but respects the auto-pause.

**Option C — Keep 15% auto-pause but define resume protocol explicitly**. If strategy pauses at 15% DD, documented process for review (human decision within 48h, specific criteria for resume: has strategy entered new signal regime? has volatility normalized? are costs within modeled range?).

Recommendation: Option C is most honest for demo (information gathering) phase. Option B for eventual real-capital deployment. Option A is not recommended because it normalizes catastrophic drawdowns.

### Task 5.5.4.2 — Verify Q3 deployment is actually running
The Phase 5 synthesis is dated 2026-04-17 (today) and declares the observation period running 2026-04-17 to 2026-07-16. Confirm:
- Is the OANDA demo account actually executing trades?
- Is the logging infrastructure capturing data?
- Has any trade been generated since deployment?

If deployment is not actually live, this is the priority operational task — not a research question, an execution one.

### Task 5.5.4.3 — Continue monitoring with corrected configuration
Apply resolution from 5.5.4.1. Continue 90-day observation window. Phase 5.5 does not wait for Q3 completion; Q3 provides ongoing data that feeds into eventual terminal state review.

### Deliverable 5.5.4
- Auto-pause configuration decision with rationale
- Confirmation of live deployment status
- Updated monitoring document

---

## Task 5.5.5 — Revised Synthesis and Terminal State (half day, omega)

### The issue being corrected (from overall Phase 5 structure)
Phase 5 synthesis declared Terminal 1 (Deploy Recalibrated Portfolio) without incorporating its own self-critique. The revised synthesis must reconcile the two readings of the data and commit to a terminal state that honestly reflects the corrected evidence.

### Task 5.5.5.1 — Apply corrected evidence to decision matrix
Using results from Tasks 5.5.1-5.5.3:

- Is there an oracle-free portfolio with edge>0 cells only that passes IS validation at honestly recalibrated threshold?
- Does that portfolio also pass OOS validation?
- At 10% target vol, is the max drawdown operationally acceptable?

### Task 5.5.5.2 — Commit to terminal state
The terminal states available are unchanged from Phase 5:

**Terminal 1 — Portfolio deployment**: requires corrected portfolio passes OOS with worst-2Y above recalibrated threshold. Evidence must be unambiguous.

**Terminal 2 — Single-strategy deployment**: EUR/USD pure MR continues on demo. Expectations are modest (Sharpe 0.2-0.3, max DD 20-25% at retail sizing). Not a scalable product but a validated operational output.

**Terminal 3 — Documented closure**: corrected evidence does not support deployment at honest thresholds. Project closes with comprehensive research document.

**Terminal 2.5 (new, honest)**: Terminal 2 (EUR/USD MR deployed) continues while remaining open to rejoining Terminal 1 if Q3 live performance over 90 days provides operational evidence supporting portfolio expansion. This is not "try again" — it is "deploy the defensible, watch the live data, revisit portfolio if concrete evidence emerges."

### Task 5.5.5.3 — Write the final synthesis
One document, no internal contradictions. Executive summary must reflect the same findings as the detailed analysis. If there are caveats, they appear in the executive summary with the correct weight, not buried at the bottom.

### Deliverable 5.5.5
- `PHASE_5_5_SYNTHESIS.md`: single document with internal consistency
- Terminal state commitment with supporting evidence
- If Terminal 3: `PROJECT_CLOSURE_RESEARCH_DOCUMENT.md` comprehensive findings across all phases

---

## Execution Schedule

```
Day 1: Task 5.5.1 (cell classification + corrected portfolio construction)
         - Omega: classification and filtering
         - Dragon: portfolio reconstruction with target-vol scaling
Day 2: Task 5.5.2 (out-of-sample validation)
         - Omega: IS/OOS split, weight freezing, OOS evaluation
Day 3: Task 5.5.3 (proper benchmarks)
         - Omega: data download, comparison table, threshold recommendation
       Task 5.5.4 (Q3 audit, parallel, half day)
Day 4: Task 5.5.5 (synthesis)
         - Omega: final document, terminal state decision
```

Machine load is much lighter than prior phases because no large sweeps or model training are required.

---

## Standing Rules for Phase 5.5

1. **Critical self-audit must be reflected in conclusions, not just documented separately.** If the synthesis mentions caveats, the executive summary acknowledges them with appropriate weight.

2. **No new strategies, no new cells, no new assets.** Phase 5.5 works only on material already generated in Phases 3.5-5.

3. **Threshold recalibration must be benchmark-justified, not outcome-justified.** The recommended threshold comes from the distribution of worst-2Y across appropriate public benchmarks, not from "what value makes our portfolios pass."

4. **Out-of-sample is mandatory for any deployment claim.** In-sample results, no matter how strong, do not by themselves support deployment.

5. **Honest labeling of operational vs research output.** EUR/USD MR on demo is operational validation (stack works, live conditions OK), not yet research evidence (90 days with ~6 trades is too few for statistical claims). Do not conflate the two.

6. **Terminal state commitment is final.** If Phase 5.5 concludes Terminal 2 or 3, there is no Phase 5.6. A Phase 6 exists only for Terminal 1 implementation.

---

## What Honest Success Looks Like

At the end of Phase 5.5, one of these is true:

**Honest Terminal 1**: Corrected oracle-free portfolio, selected via IS construction and validated OOS, passes recalibrated threshold derived from ≥4 active managed-futures benchmarks. Phase 6 builds and deploys this portfolio.

**Honest Terminal 2 or 2.5**: EUR/USD MR continues on demo as deployed product. Portfolio ambitions set aside. Project output is a working, modest, single-strategy system plus comprehensive research documentation of what does and does not work.

**Honest Terminal 3**: Corrected evidence does not support portfolio deployment and EUR/USD MR alone is judged insufficient to justify continued capital and effort. Project closes. Comprehensive research document is the primary output.

Any of these three outcomes is a legitimate, honest, complete end to the research phase. The failure mode Phase 5.5 is designed to prevent is **a deployment commitment based on evidence the agent itself has already identified as weak**. That is the one outcome that is not acceptable.

---

## A Note on This Plan's Own Limitations

I have written this plan to be rigorous, but it has limits I should acknowledge:

- I am assuming the Phase 5 data files (JSON results, the cells loaded, the codebase) are accessible and correct. If the foundations from Phase 5 have their own issues beyond the 11 the agent identified, Phase 5.5 will not catch them.

- I am assuming 3-4 days is enough time. If Task 5.5.3 benchmark downloads have unexpected issues (NAV data quality, missing periods), timing slips. Not catastrophic, but possible.

- The "Terminal 2.5" I introduced is my own creation, not industry-standard. It is an honest intermediate state that reflects reality (EUR/USD MR is demo-deployed regardless of other findings). I present it as an option because pretending it doesn't exist would itself be dishonest.

- I may have missed issues in the Phase 5 synthesis beyond the 11 the agent listed. The agent's critique was thorough but no critique is exhaustive. If new issues emerge during Phase 5.5 execution, the agent should flag them and we handle them then.

If the agent identifies something I missed during execution, flagging it is correct behavior, not insubordination.