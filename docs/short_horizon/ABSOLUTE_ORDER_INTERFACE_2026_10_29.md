# Absolute order execution and early-close contract

Status: native execution component verified on synthetic positions. This is
not a learned policy, an optimized strategy or a financial performance result.
The default `BarEnv` path and historical artifacts remain unchanged.

## Interface

`app.short_horizon.expert_guided.absolute.AbsoluteBarEnv` is opt-in and shares
the native `BarEnv.step`, quote-side protection, costs, financing and ledger.
Its entry-preparation hook replaces multiplier/clipping rules with absolute
levels. `EnvConfig` must explicitly use `bid_ask_v3`, no daily cap (`None`),
leverage50 and margin-based `rel_volume=0.1`. Calendar: wait12h after weekly
open; no entries at/after forced-flat12h before weekly close.

Actions are HOLD=0, LONG=1, SHORT=2, CLOSE=3. Opens require a pure
`entry_plan(FillContext) -> AbsoluteOrder`; HOLD/CLOSE must not provide one.
Context contains direction, current balance, actual entry/liquidation quotes,
decision timestamp and fill timestamp. Templates capture decision-time
references and resolve at the next open; future intrabar prices are not passed.
This API cannot prevent a malicious callback from reading an external file;
causal signal production remains a separate controller obligation.

The order contains absolute `tp`, `sl`, normalized `q`, a monetary
`risk_budget`, and a per-unit `funding_reserve_per_unit`. Margin is
`q * 0.1 * balance`, subject to native integer unit rounding; `q` never
multiplies rel_volume twice. Both quote-side brackets, tick alignment, broker
minimum distance, minimum units, margin and modeled stop loss are checked.
Small orders are refused, never rounded up to increase risk. Stops cannot
guarantee the nominal risk budget across a gap.

There is one open position. Another opening request while occupied becomes
HOLD with a recorded refusal, not an implicit reversal. `q` is stored on the
position. State feedback is prohibited by this interface. Decisions must
advance sequentially; requesting a final result before full replay is refused.
Missing hours inside an open session are refused, not interpolated/compressed.

## Early close

At next open, automatic SL/TP protection takes precedence over CLOSE. Otherwise
CLOSE executes at that open before the next intrabar range. An early close is
independent of the horizon which opened the position. Funding and costs use
that actual exit time and executable quote side. A gap already through SL
cannot be rescued retroactively. In ambiguous intrabar touches, SL wins.

Tests include both directions, six entries in a day, last admissible entry,
forced-flat boundary, no reversal, original-position volume, current balance,
bad levels, minimum order rejection, missing bars and causal callback context.
Two identical market paths with different SL distances require different exit
choices: the tight stop favors closing now, the wide stop survives recovery.

## Evidence

74 focused tests pass across `test_absolute_order_execution.py`,
`test_quote_side_execution.py`, `test_corrected_execution_cost_modes.py` and
`unit_tests/test_expert_guided_env.py`. Legacy replay of two120bar runs,
18 trades each, remains byte-identical:
`51e9aa9226c49238d20198da7df37abb6f70c65c1d7a021655f1c0e9a772c219`.

The cross-repository runner in agent-multi,
`tools/verify_absolute_early_close.py`, verifies 1,744 fixed synthetic positions
and 14,784 possible early exits against independent cash/first-barrier
calculations. See agent-multi work-plan document131 and its pinned manifest
for exact source identities, controls and deterministic replay. The engine
here owns execution; agent-multi owns risk proposals, evaluation and evidence.
