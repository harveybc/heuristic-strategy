# Synthetic elapsed-hour microexperiment, 2026-09-30

Retsu, integration owner. Lane B. Repository `heuristic-strategy`. Branch `satoshi/strategy-support-20260930`.

Every number in this note is **SYNTHETIC**. The fixture is constructed OHLC. The run stays on CPU, off any broker network, off live capital, off an EURUSD backtest, off the 241-cell sweep, and off the optimizer. B0 stays **NOT_STARTED**.

Measured records: `docs/audits/evidence/STRATEGY_MICRO_20260930/PRE.json`, `POST.json`, and `MICRO.json`.

## PRE, repair, POST

PRE ran the F2 and F3 tests against the unrepaired elapsed-hour code. Pytest collected 17 tests: **1 passed, 16 failed**. F2 failed (2). F3 failed (14). The passing test was the market-gap case: a missing exact bar excluded that origin and did not raise. PRE.json keeps that failure. It was not rewritten after the repair.

The repair is in `app/strategy_support.py`.

- A horizon is a positive unique Python `int`. `True`, `1.9`, `"1"`, duplicates, NaN, and inf are rejected. `int()` is not used to coerce a horizon.
- Prices that the generator or the DEV fit actually consume must be finite. A missing exact bar is a gap: that origin is excluded and the call does not raise.
- The caller declares `source_timezone`. Naive stamps are localized in that zone and converted to UTC. Ambiguous or nonexistent civil times raise. The zone of a historical CSV is not inferred.
- `calibration_set(origins)` still screens origin clocks only, so the existing one-argument tests keep their meaning.
- `admit_elapsed_hour_calibration` and `fit_development_parameters` admit an origin only when its targets, and the scale and residual derived from those targets, sit strictly before 2019-05-16 00:00 UTC. An explicit origin of 2019-05-15 23:00 with horizon 1 raises, because the target is the reserved bar. Mutating that reserved close from 1 to 999 leaves the fitted DEV parameters unchanged. The fit is a per-horizon mean absolute residual. `fits_noise_model` is false. No noise model is fit on or after 2019-05-16.

POST, after the repair, collected the contract tests, the microexperiment, and `tests/unit_tests/test_strategy_support_20260930.py`. Pytest result: **28 passed, 0 failed**, 1.41 seconds. The 17 F2/F3 contract tests passed. Two microexperiment tests passed. Nine existing strategy-support tests passed.

## What the microexperiment runs

`app/elapsed_hour_harness.py` runs only when `prediction_generator` is `elapsed_hours` and `exit_variant` is `E`. The offset unit on this path is **hours**. `create_hourly_predictions` and `create_daily_predictions` stay on `process_data` with unit **rows**.

The harness calls `create_elapsed_hour_predictions`, renames `elapsed_{h}h` onto `Prediction_h_1..6` and `Prediction_d_1..6` in horizon order (1..6h, then 24, 48, 72, 96, 120, 144h), and runs the real `Plugin` from `app/plugins/plugin_long_short_predictions.py`. Decisions stay close-only. Fills stay next-open market orders. No protective order is added. `tp_multiplier` stays 0.9 and `sl_multiplier` stays 2.0.

The label frame drops every bar at or after 2019-05-16 00:00 UTC before the generator reads a price. The fixture still contains a sentinel bar at that instant with open and close 999. That bar is absent from the broker feed and from the four support origins.

## Declared capital

The plugin leaves broker margin unset, so the simulated broker requires full notional. Order size is `min(1_000_000, cash * 2)`. At the plugin's cash of 10000, size is 20000 and a buy of that notional does not fill. Raising cash without hitting the size cap still sizes at twice cash, so the buy still exceeds cash.

The four contrast arms therefore use one declared **SYNTHETIC** cash of **2_000_000**. Size then caps at 1_000_000, and the same sizer and the same full-notional check both accept it. Spread, slippage, the commission rate, swap, and the TP/SL multipliers are the plugin's values. Leverage is not passed into the broker as a correction.

A separate observation at cash 10000, same fixture and same variant E, is recorded in `MICRO.json` as `broker_cash_check`. It is not a fifth contrast arm. Requests were long, short, long. Closed trades: **1**, the short opened 2019-05-01 04:00 and closed 07:00, volume 20000. The 00:00 long was requested and did not become a closed trade. Realized PnL **396.3219999999965**. Pending position **0**. Marked equity **10396.571999999998**. Cash **10396.571999999998**.

## Variant E trace

Variant E can use both families on an exit: `0.6 * short + 0.4 * long` compared with the stop. The long family is what `calculate_entry_geometry` sees, so it sets the side, the take-profit, and the stop. The trace below is the measured decision stream. Expected sides were literals in the test, fixed before the run.

Long-family geometry, measured on both `ideal/ideal` and `persistence/ideal` (SYNTHETIC):

| Decision bar | Request | Take-profit | Stop | Short-family level that was not used |
| --- | --- | --- | --- | --- |
| 2019-05-01 00:00 | buy, long | 1.018 | 0.9998 | a short-family extreme of 1.030 would have produced 1.027 |
| 2019-05-01 04:00 | sell, short | 1.002 | 1.0202 | a short-family extreme of 0.990 would have produced 0.993 |
| 2019-05-01 08:00 | buy, long | 1.009 | 0.9998 | |

Size on each of those requests was 1_000_000. Position before each request was 0. The first fill price was 1.000015.

At 2019-05-01 09:00 the ideal arm is long 1_000_000, stop 0.9998, and the close 1.002 is between the stop and the take-profit 1.009. Ideal short predictions are 0.99, 0.995, 1.0, 1.0, 1.0, 1.0. Ideal long minimum is 1.003. The variant E blend is `0.6 * 0.99 + 0.4 * 1.003 = 0.9952`, which is below 0.9998, so the plugin requests `close`. The long minimum alone, 1.003, is not below 0.9998. The short family is what makes this exit fire.

On `persistence/ideal` the same bar is also long 1_000_000 with the same stop. The short family is the origin close 1.002 on all six horizons. The blend is `0.6 * 1.002 + 0.4 * 1.003 = 1.0024`, which is not below 0.9998. The request is `none`. At 10:00 the close is 0.990, the stop is touched, and the plugin requests `close`. The ideal arm is already flat on that bar and requests nothing.

`ideal/persistence` and `persistence/persistence` request no entry at 00:00. Flat long-family predictions do not open these trades. The short family changes the exit. The long family sets the entry and the TP/SL.

## Four arms

Same twelve horizons, same four support origins (2019-05-01 00:00, 04:00, 08:00, 09:00 UTC), same configured costs, same declared cash 2_000_000. Pair order is short family / long family. MAE is absolute error against the future close on those origins. The naive error on the same rows is the origin close against that future close. The aggregation is the unweighted mean of the twelve horizon MAEs (`macro_average_of_per_horizon_mae`).

Configured costs, identical on every arm (SYNTHETIC): spread 2 pips, slippage 1 pip, pip cost 0.00001, commission 0.00007 per unit of price, swap 10 per lot per day, TP multiplier 0.9, SL multiplier 2.0.

| Arm | Closed trades | Realized PnL | Marked equity | Cash | Pending units | MAE macro | Naive macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ideal/ideal | 3 | 50453.09666666601 | 2050486.4299999997 | 2050486.4299999997 | 0 | 0.0 | 0.009208333333333327 |
| persistence/ideal | 3 | 39449.69999999956 | 2039487.2 | 2039487.2 | 0 | 0.005312500000000002 | 0.009208333333333327 |
| ideal/persistence | 0 | 0.0 | 2000000.0 | 2000000.0 | 0 | 0.0038958333333333254 | 0.009208333333333327 |
| persistence/persistence | 0 | 0.0 | 2000000.0 | 2000000.0 | 0 | 0.009208333333333327 | 0.009208333333333327 |

Pending position, marked equity, and realized PnL are separate fields. On these four arms the book is flat at the last in-feed close, so marked equity equals cash, and realized PnL is the sum of closed-trade PnL. End exposure is 0.

`ideal/ideal` trades, volume 1_000_000 each (SYNTHETIC):

| Open | Close | PnL | Pips | Bars | Commission path |
| --- | --- | --- | --- | --- | --- |
| 2019-05-01 00:00 | 03:00 | 29815.39999999983 | 2996.9999999999827 | 3 | take-profit |
| 2019-05-01 04:00 | 07:00 | 19816.09999999982 | 1996.999999999982 | 3 | take-profit |
| 2019-05-01 08:00 | 10:00 | 821.59666666636 | 96.99999999996932 | 2 | early exit |

Measured cost totals for that arm: gross PnL 50909.999999999345, commission 423.56999999999994, swap 33.33333333333337. Six fills.

`persistence/ideal` keeps the first two trades and replaces the third (SYNTHETIC):

| Open | Close | PnL | Pips | Bars |
| --- | --- | --- | --- | --- |
| 2019-05-01 00:00 | 03:00 | 29815.39999999983 | 2996.9999999999827 | 3 |
| 2019-05-01 04:00 | 07:00 | 19816.09999999982 | 1996.999999999982 | 3 |
| 2019-05-01 08:00 | 11:00 | -10181.800000000094 | -1003.0000000000093 | 3 |

Measured cost totals: gross PnL 39909.999999999556, commission 422.7999999999993, swap 37.5. Six fills.

On this fixture the early exit closed before the later stop print. That is a property of this turn. This lane does not treat the ideal arm's realized PnL as a maximum the signal is required to achieve.

Per-horizon naive MAE on the shared origins (SYNTHETIC), same for every arm: 1h 0.006000000000000005, 2h 0.016750000000000015, 3h 0.013750000000000012, 4h 0.01050000000000001, 5h 0.00874999999999998, 6h 0.008000000000000007, 24h 0.006749999999999978, 48h 0.013000000000000012, 72h 0.007500000000000007, 96h 0.00649999999999995, 120h 0.006749999999999978, 144h 0.006249999999999978. Ideal MAE is 0 on each horizon. Persistence/persistence MAE equals that naive series. The macro average of the twelve naive horizons is 0.009208333333333327.

Equal MAE does not imply equal decisions. Persistence and ideal-plus-noise at the same MAE are different inputs. No sweep was run.

## Pending position at a cut feed

A second SYNTHETIC fixture keeps one complete origin at 2019-05-01 00:00 and sets `broker_session_end` to 2019-05-01 01:00, still before the reserved cut. The long is requested at 00:00 and fills at 01:00, price 1.000015, size 1_000_000. The 01:00 close is 1.001, between stop 0.9998 and take-profit 1.018, and that bar is not itself a twelve-horizon origin. Closed trades: **0**. Realized PnL: **0**. Pending units: **1_000_000**. Marked equity: **2000914.99895**. Cash: **999914.9989499999**.

`stop()` calls `close()` while that position is open. Position before `stop()` is 1_000_000. Position after `stop()` is 1_000_000. Closed-trade count stays 0. `stop_close_obtained_fill` is false. The marked equity is the broker value at the last in-feed close. It is not a post-cut broker event. Trades that would have crossed a later cut are not deleted. No forced liquidation is applied.

## B0 successor, not run

B0 stays **NOT_STARTED**. The historic 3600 CPU-s remainder is not a B0 budget, and this lane did not spend it. A later real-data request could be reviewed against this successor and was not executed here:

- Cost basis of the real plugin: spread 2 pips, slippage 1 pip, pip cost 0.00001, commission passed as `commission_per_lot / 100000` (0.00007) and charged as a percentage of price, swap 10 per lot per day, TP multiplier 0.9, SL multiplier 2.0, exit variant E, close-only decision, next-open market fill, no protective orders. The commission is not rewritten into a literal 7 currency-unit fee.
- Terminal convention: **pending positions stay open; no forced liquidation.**
- The declared SYNTHETIC cash of 2_000_000 above is a property of this microexperiment's full-notional broker check. It is recorded so a later review can see why the plugin's 10000 cash did not let the long fill.

`close()` inside `stop()` was observed on the pending-position fixture and did not obtain a fill.
