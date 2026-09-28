# Conditional short/long noise sweeps

Owner approved 2026-09-27 before execution. No changes to trading rules or
parameters from the verified margin5 sweep: same source, masks, origins,
seeds 42/43/44, fees, margin cap, leverage, TP/SL and early exits.

## Four conditions

| Name | Vary | Hold fixed |
| --- | --- | --- |
| long_naive | long forecast noise | short MAE equals short naive |
| short_naive | short forecast noise | long MAE equals long naive |
| long_ideal | long forecast noise | short future closes without noise |
| short_ideal | short forecast noise | long future closes without noise |

Short = columns at 1-6 hours; long = available 24-144 hour points. H6 and H144
are calibration endpoints, not a claim that all intermediate errors match their
own naive. With retained standard-normal draws z, each family uses its own sigma
reference: naive endpoint MAE / mean over seeds of mean(abs(endpoint z)).
Multiply that reference by the pre-existing ratios
0,.05,.1,.15,.2,.3,.4,.5,.6,.75,.9,.95,1,1.05,1.1,1.25,1.5,1.75,2,2.5,3.
The fixed family's multiplier is either 1 (equivalent MAE) or 0 (no noise).
Draws do not change within a seed; noise is never applied to actions.

One shared origin population: exact source-clock H6/H144 timestamps, intermediate
unavailable targets absent. No original dataset or reserve is expanded. The
equivalent-naive inputs remain noisy future prices, NOT actual persistence.
Actual persistence is a separate measured control. No forecast model is trained.

There are 252 logical curve points and 240 unique noisy-input computations across
seeds, plus one persistence control. Identical (short ratio,long ratio,seed)
combinations reuse a single new result and explicitly name it in the curve map.
This is not independent replication of a common point. Separately test that
constructing the common point from either varying-axis interface yields exactly
the same forecast array, and verify all logical points against their intended
ratios and referenced results. No arbitrary old scores may be substituted.

## Outputs and interpretation

Four white presentation PNGs: long/short variation, separately profit/Sharpe;
each compares fixed-naive versus fixed-ideal as two colored curves with
between-seed SD. X uses measured mean MAE / its paired naive, with a line at 1.
No footer comments, corrective heading, internal jargon or asterisks. Record
actual noise sigma and every horizon's MAE in CSV, not just the normalized axis.
Report seed-paired changes versus the corresponding curve's ratio-1 point.
Effects are conditional on the other signal and this strategy/market sample.
No universal profitability threshold or exponential law is presumed or fitted.

## Tests before implementation

- Fixed family remains byte-identical through all varied levels; target NaNs
  remain NaN, never noise-only synthetic points.
- At ratio 1, endpoint MAE averaged across seeds equals its own naive.
- Ideal controls really have zero errors in the held-fixed family.
- Both principal curves meet at (1,1), controls meet at (0,0), with explicit
  common-result identity; no duplicate execution is counted as a new replica.
- Reject forged mappings, changing the supposedly fixed noise or omitting any
  logical cell even if file digests were consistently updated.
- Reuse independent raw-timestamp, prediction-consumption, native-protection,
  cash/equity, fee, five-percent-margin and metric verification for every job.

Pilot: ratios 0/1 and seed 42, five unique backtests including persistence.
Then full sequential CPU run, projected under 900 backtest CPU seconds, with
incremental records and no GPU or live service. All source and design bytes are
retained before dispatch. Prior simultaneous-noise runs are preserved.
