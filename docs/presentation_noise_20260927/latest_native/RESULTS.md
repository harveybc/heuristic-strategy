# Native heuristic-strategy sweep

Status: COMPLETE and independently VERIFIED, 241/241 simulations. Four native
PNG exports visually inspected. Focused regression suite: 74 passed.

## Correction of previous attribution

The previous conditional runner was NOT the user's native strategy. It used the
same TP/SL geometry, but added an ordered-crossing entry filter, replaced exit
variant E with first-passage exit, and replaced CLOSE-only TP/SL handling with
intrabar bracket orders. The previous numeric ledger verification did not prove
strategy equivalence. Prior plots are withdrawn as native-strategy evidence,
with original files/results preserved for audit.

At the frozen TP=0.9 and SL=2 multipliers, that extra entry gate is redundant:
the chosen TP is inside the forecast extremum and the SL outside the opposite
extremum. A 1000-path test confirms this. Do not attribute the observed PnL
difference to that gate; early-exit policy and intrabar execution are the
demonstrated behavioral changes. No separate effect size is claimed for either.

Source of native trading decisions: `app/plugins/plugin_long_short_predictions.py`,
`Plugin.HeuristicStrategy.next`. The adapter calls it directly. Neither that
file nor `app/policies/prediction_entry_exit.py` was changed for this correction.

## Exact scope of fidelity

- Native extrema-based long forecast entry direction, TP, SL and frequency.
- Native E early exit: weighted short/long minima for long positions and maxima
  for shorts, compared to SL, as declared in `Plugin.plugin_params`.
- Native TP/SL checks at current CLOSE, with market execution at the next OPEN.
  No ordered-crossing admission filter or intrabar bracket is added.
- Plugin source and resolved parameters are retained before dispatch.
- Real native `next()` and instrumented `next()` produce identical orders in
  the parity fixture. Negative cases distinguish it from the replaced engine.

This is not a byte-identical execution of the stock GA/evaluate_candidate
launcher: the owner-requested five-percent margin, currency conversion and
actual-fill cap remain explicit adapters. Broker accounting charges the same
declared commission/friction/swap as the prior sweep. Reported trade PnL uses
the broker ledger rather than the plugin's display, which deducts swap again.
Terminal liquidation executes on an additional bar. No change is hidden as a
signal repair; no parameter is selected using sweep performance.

TP/SL naturally vary with long predictions in the native function. For a buy,
TP = current + 0.9*(max(long forecasts)-current); SL = current -
2*max(current-min(long forecasts),10*0.00001). Shorts use the symmetric rule.
The faulty attribution was not evidence that this formula itself was wrong.

## Data and experiment

Same source CSV, 13,590 common exact-timestamp H6/H144 origins, twelve forecast
columns, three stored Gaussian draws, four conditional sweeps and 21 levels.
H6 persistence MAE 0.0017059896983075775; H144 0.007522118469462839 USD/EUR.
Calibrate each family's amplitude at its endpoint; the other family's amplitude
and realization stay fixed. Endpoint equivalence holds in the three-seed mean.
Noisy future prices are not persistence, even with the same MAE.

Cash starts at USD 10,000, leverage 100:1, initial margin at most 5% of balance,
allocation cap 1,000,000 EUR. Costs: USD 7/100,000 EUR/side commission;
0.00010 USD/EUR/side friction; USD 10/100,000 EUR/calendar-day swap, both directions.
Sharpe is calculated from calendar-daily equity returns, annualized by sqrt(365).
This CLOSE-only historical simulator is not a bid/ask tick replay or live result.

## Pilot and acceptance

Five native pilot cells completed in 26.648 CPU seconds, projecting 1285 for the
241 unique full-run cells under the 3600 CPU-second bound. Pilot independently
reconstructed raw targets, input fingerprints, every native decision including
no-order decisions, fills, margin, cash, costs and MAEs. Mutation tests reject
rehashed false position/prediction, missing decision, variant and profit.

Both-ideal pilot: net USD 604,943.05, Sharpe 6.16041, 417 closed trades. Commission
USD 34,432.89, friction USD 49,189.84, swap USD 30,347.72. These figures describe the
declared native CLOSE-only simulation, not achievable live-trading performance.

## Final results

Mean of three paired noise seeds. Ratios refer to each endpoint's persistence
MAE; ratio one is still noisy future prices, not actual persistence.

| Short ratio | Long ratio | Net profit USD mean (SD) | Sharpe mean |
| --- | --- | --- | --- |
| 0 | 0 | 604,943.05 (0.00) | 6.160411 |
| 1 | 0 | 93,301.51 (12,225.41) | 3.894485 |
| 0 | 1 | 37,196.23 (36,502.68) | 1.611698 |
| 1 | 1 | 21,842.55 (14,026.31) | 1.337063 |

Actual persistence: zero entries, zero profit, undefined Sharpe. Both-ideal is
417 trades with 191 net wins and 226 net losses, gross USD 718,913.50 minus
USD 113,970.45 costs. It is not a no-loss oracle; sparse future closing prices,
the native rule and next-open fills do not optimize all possible trades.

The large earlier comparison (USD 31,482.73 ideal under the replacement engine)
does not characterize this plugin. No decomposition between early-exit changes
and execution changes is claimed; both were removed together.

The corrected curves are not forced monotonic. In particular, short-noise ratio
0.1 with long ideal averages about USD 607,051.85, slightly above the noiseless
USD 604,943.05, while individual noise seeds vary. Holding the other signal
noisy also allows larger nonmonotonic responses. This residual observation is
present in the actual plugin output, not removed by choosing a seed, fitting an
exponential curve or changing the policy. The experiment does not prove that
zero MAE maximizes this particular heuristic's realized profit.

Full-run outputs: `run_out/native_conditional_20260927_full`. CPU 1461.693 seconds,
under the frozen 3600-second budget; about 1.1 GiB retained. Independent checks
cover all 241 cells, 60,494 entry margin bounds, all native decisions/no-orders,
raw timestamp targets, forecasts consumed, fills, daily equity, fees and funding.
Maximum independent MAE discrepancy 3.47e-18; maximum margin fraction
0.050000000000008094 is floating-point roundoff. All final positions flat and no
rejected orders. Native plugin, policy and adapter match their frozen source.

Manifest SHA256:
`958726415c09ced866c7d8164dfe4e49b99b4ccb8e0e5e7059fa26effdbba94f`.
Presentation receipt binds verification, the summary CSV and all four PNGs.
The full per-seed contrasts and costs are in `conditional_paired.csv`,
`conditional_summary.csv` and `execution_diagnostics.csv`.

New presentation files in `predictor/docs/`:
- `barrido_estrategia_largo_profit.png`
- `barrido_estrategia_largo_sharpe.png`
- `barrido_estrategia_corto_profit.png`
- `barrido_estrategia_corto_sharpe.png`

Do not substitute the earlier `barrido_condicional_*` files for these native
results. Their original files remain available only with the withdrawal in
CONDITIONAL_RESULTS.md.
