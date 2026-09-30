# Strategy support corrections, 2026-09-30

Retsu, integration owner. Repository `heuristic-strategy`. Branch `satoshi/strategy-support-20260930`.

No financial or model performance was measured.

Nothing here places an order, starts B0, reruns the retained 241-cell sweep, or fits a noise model. The replication baseline is still the existing plugin: a decision at the bar close, a market fill at the next open. Protective broker orders are a separate named experiment. No bracket order was added.

---

## What changed

- Variant E is the baseline in `app/config.py`, in `Plugin.plugin_params`, and as the default argument of `HeuristicStrategy.__init__`. The signature default was `D`. The comment that called D the default was removed. `historical_run_recovered` is false.
- `create_hourly_predictions` and `create_daily_predictions` are unchanged in behavior and are still what `process_data` calls. Their offset unit is named **rows**. They were not deleted.
- `app/strategy_support.py` adds `create_elapsed_hour_predictions`. A horizon of 144 is 144 elapsed hours. An origin that lacks the exact target bar is excluded. This generator is not wired into auto-generation or into any scorer.
- `derive_development_support` reports, for each supplied development origin, the exact latest elapsed-hour target, whether that origin is purged, and trade-exit / cost support. It does not move the reserved window.
- `calibration_set` accepts development timestamps only. A fixture that contains 2019-05-16 00:00 or later is rejected. The constructor does not fit noise, correlation, or a scale.
- `cpu_reconciliation` records the subtraction below and the status `B0_NOT_STARTED`.

## Variant E resolution

The plugin contradicted itself. The comment said D was the default. The nested `HeuristicStrategy.__init__` signature defaulted to `D`. `plugin_params` said `E`. Launchers that read `plugin_params` already received E.

The baseline is now E in the config, in `plugin_params`, and in the signature. `baseline_config()` returns the same fact:

- `exit_variant`: `E`
- `historical_run_recovered`: false
- `resolution`: `explicit_resolution_not_recovered_historical_run`

This is an explicit resolution, not a recovered historical run.

The retained 241-cell manifest is not in this worktree. The design points at `docs/noise_price_sweep/` and a retained run directory; neither is present here, and this lane did not open another checkout. `sweep_241_exit_variant` is **NOT_CHECKED**. This note does not claim that sweep used E.

## Elapsed hours versus legacy rows

| Path | Where | Unit | What 144 means |
| --- | --- | --- | --- |
| Legacy reproduction | `create_hourly_predictions`, `create_daily_predictions` | rows | the daily generator's sixth step is row offset `6 * 24`, not a timestamp 144 hours later |
| New generator | `create_elapsed_hour_predictions` | hours | column `elapsed_144h` is `CLOSE` at `t + 144 hours`, or the origin is omitted |

On a three-row synthetic frame whose last timestamp is exactly 144 hours after the first, the elapsed column equals that bar's close. Both legacy generators return an empty frame on that input, because 144 rows are not there. A one-row legacy step is the next dataframe row even when that row is three clock hours away; the elapsed generator will not call that gap one hour.

`process_data` still auto-generates with the legacy row-offset functions. Switching it would have changed the replication baseline.

## Synthetic support population

These counts are `synthetic_elapsed_support_frame` only. They are not market results.

The frame has nine constructed timestamps. The development population is five supplied origins, not every bar before the cut. Bars at 2019-05-15 20:00, 2019-05-15 22:00, and 2019-05-15 23:00 are target bars. They are not members of the five.

Reserved window, not rewritten: **2019-05-16 00:00** onward. `reserved_window_rewritten` is false.

Trade duration is **UNBOUNDED_TRADE_DURATION**. `HeuristicStrategy.next` closes on a close-only take-profit, a close-only stop, or a variant early exit. `max_trades_per_5days` counts entries. It does not limit how long a position may be held. `stop()` closes a position only because the feed ended. No maximum holding period exists in the plugin, so none was invented, and a six-day purge was not treated as a holding cap. Censoring is **REPORTED_NO_TIME_CENSOR**: a 144-hour target does not censor the trade. Swap scales with bars held and the exit cost is paid at the exit, so cost support is the same unbounded interval. Every supplied origin is **NOT_SEPARATED_BY_ORIGIN_CUT**, including origins whose 144-hour target ends before the cut. None of them is reserved-safe.

Longest target horizon: 144 hours.

| Development origin | Latest target (`t + 144h`) | Exact bar present | Purged |
| --- | --- | --- | --- |
| 2019-05-09 20:00 | 2019-05-15 20:00 | yes | no |
| 2019-05-09 22:00 | 2019-05-15 22:00 | yes | no |
| 2019-05-09 23:00 | 2019-05-15 23:00 | yes | no |
| 2019-05-10 00:00 | 2019-05-16 00:00 | yes | yes |
| 2019-05-15 12:00 | 2019-05-21 12:00 | no | yes |

| Count | Value |
| --- | ---: |
| Development origins | 5 |
| Purged (target timestamp on or after 2019-05-16 00:00) | 2 |
| Exact latest target bar present | 4 |
| Missing exact latest target bar | 1 |
| Latest target timestamp strictly before the cut | 3 |
| Not separated by the origin cut | 5 |

Entire target support means every required elapsed-hour bar exists and the latest of them is strictly before the cut.

- Required horizons 1h and 144h: 2 origins. The latest such origin is 2019-05-09 23:00. The 20:00 origin lacks the exact +1h bar, so it is not in that set.
- Required horizons 1, 2, 3, 4, 5, 6, 24, 48, 72, 96, 120, and 144 hours: 0 origins on this fixture. The alternative origin is none. The purge count stays 2.

The alternative origin is only a report. It does not replace the design's reserved window.

## Prior-access audit

Reserved rows were already read. Naming the window reserved does not make earlier reads into non-use.

This lane searched the worktree for `tests/data/eurusd_hour_2005_2020.csv` and for timestamps on or after 2019-05-16, and it scanned CSV first fields once. It did not backtest the file. The scan itself read the reserved timestamps' presence. It did not score them.

The 2026-09-29 design records that the OHLC file spans 2005-05-02 00:00 through 2020-04-29 22:00 (93,084 data rows) and that the span was verified. That verification is a read of the reserved window. This lane's scan agrees: one header line, 93,084 data rows, 5,923 of them on or after 2019-05-16 00:00, last timestamp 2020-04-29 22:00. That is an access count, not a market result.

Full-file loads put the reserved rows in the address space before any later year filter drops or slices them:

- `run_wfo.py` loads the whole file, then masks by year. The default last test year is 2019. `app/walk_forward_optimizer.py` defaults the last test year to 2020.
- `run_phase_b_cnn.py`, `run_phase_c_ensemble.py`, `run_phase_d_neat.py`, and `run_oracle_ceiling.py` each load the whole file, then loop test years 2006 through 2019.
- `deploy_wfo.sh` copies the whole OHLC file.

Committed `wfo_results.json` has a 2019 fold with 6,188 test bars, the calendar-year slice of the loaded file. The file contains 2019-05-16, so those bars are inside that slice. The committed WFO trade for 2019 opens 2019-04-27, before the cut. One pre-cut trade timestamp does not show that the reserved bars were absent from the loaded frame. The WFO trade files have no first-field timestamp on or after the cut.

Committed artifacts that do contain timestamps on or after 2019-05-16 (first-field scan; not a performance table):

| File | Rows at or after the cut | Last such first field |
| --- | ---: | --- |
| `tests/data/eurusd_hour_2005_2020.csv` | 5923 | 2020-04-29 22:00 |
| `tests/data/phase_1c_direction_test_ohlc.csv` | 5923 | 2020-04-29 22:00 |
| `tests/data/phase_1_base_d3.csv` | 5211 | 2020-03-19 06:00 |
| `tests/data/phase_2_3_base_d3.csv` | 5211 | 2020-03-19 06:00 |
| `tests/data/phase_2_3_base_d3_last_year.csv` | 5211 | 2020-03-19 06:00 |
| `tests/data/ideal_predictions_hourly_d3.csv` | 5067 | 2020-03-11 06:00 |
| `tests/data/ideal_predictions_daily_d3.csv` | 5067 | 2020-03-11 06:00 |
| `cnn_predictions_15yr.csv` | 1504 | 2020-04-29 21:00 |
| `ensemble_predictions_15yr.csv` | 1504 | 2020-04-29 21:00 |
| `oracle_ceiling_trades.csv` | 148 | 2019-12-30 22:00 |
| `phase_b_cnn_trades.csv` | 134 | 2019-12-31 06:00 |
| `tests/data/ann_predictions_hourly_d3.csv` | 127 | 2019-05-23 10:00 |
| `tests/data/lstm_predictions_hourly_d3.csv` | 121 | 2019-05-23 04:00 |
| `phase_c_ensemble_trades.csv` | 67 | 2019-12-30 05:00 |
| `phase_d_neat_trades.csv` | 21 | 2019-12-30 05:00 |
| `trades.csv` | 1 | 2019-07-02 19:00 |

## CPU reconciliation

The historic ceiling is 3,600 CPU seconds. It is not a fresh allowance.

The design records 1,461.693 CPU seconds for the retained 241 cells.

Subtraction: `3600 - 1461.693 = 2138.307`.

The remainder is not permission to spend. Status: **B0_NOT_STARTED**. B0 was not started.

No financial or model performance was measured.
