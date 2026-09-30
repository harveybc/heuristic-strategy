# Continuous synthetic hourly pilot

The actor is an offline strategy reviewer. This is a mechanism check, not
financial calibration. The trajectory has 360 synthetic hourly OHLC bars from
2019-05-01 through 2019-05-15; 216 consecutive decision origins have exact
future labels for all 1-6h and 24-144h horizons. Scale is fitted only to the
earlier synthetic DEV frame. Noise affects forecasts, never OHLC, fills or costs.

| Requirement | Evidence |
| --- | --- |
| Complete hourly support, paired per-horizon MAE/naive, DEV separation | `test_every_eligible_hour_has_all_forecasts_and_separate_dev`, `test_dev_overlap_and_incomplete_trajectory_are_rejected` |
| Variant E, successor margin/costs, next-open fills | `test_paired_pilot_records_actual_causes_fills_and_exposure`; pilot `fills`, `ledger`, `book` |
| Actual cause and why/no early close under fixed controls | Pilot `events` include predicate evaluation, cause, TP/SL, cash, equity and exposure; test recomputes variant-E predicate |
| Bounded, no B0/market/broker, no retained-evidence overwrite | `crispdm-run -m 2G -t 120`; test writes to `tmp_path`; CLI requires `--output` |

The fixed pilot cells are `ideal_ideal`, `persistence_persistence`, and
`short_noise_long_ideal` (intensity 1, seed 42). A capped standalone execution
took 5.7 s wall time. The first had 216 origins, 6 closed trades, 5 TP and 1
SL close, 0 early closes, final marked equity 14882.01. Persistence had 216
origins, no entries or closes, final equity 10000.00. The noisy-short control
had 216 origins, 6 closed trades, all 6 prediction-triggered early closes,
final marked equity 9682.21. All three ended flat. These numbers are synthetic
observations, not utility estimates; ideal forecasts use future truth by design.
For ideal/ideal, variant E was evaluated 127 times and never triggered; 20
occupied late bars had no complete 144h forecast, so variant E was not evaluated
there. For noisy-short/ideal-long, it was evaluated 8 times and triggered 6.

`pilot.json` retains all 12 per-horizon MAE/naive pairs and complete per-bar
events/fills at the caller's output path. The 44-cell grid was not run or
estimated from PnL. The earlier four-origin sweep and retained evidence remain
untouched. A broader baseline test has one pre-existing failure:
`test_backtrader_adapter_delegates_to_frozen_core` builds a dummy without
`accounting_convention`. The focused continuous/source/replay run is 6 passed.

Next action: review the pilot and only authorize a separate capped full-grid
measurement if its additional computation is warranted.
