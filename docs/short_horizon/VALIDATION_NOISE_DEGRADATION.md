# Business-day RF-L on VALIDATION 2024: ideal predictions, then Gaussian noise

**DEVELOPMENT · VALIDATION 2024 · oracle/noise study, not a model result.** No
model was trained or selected. The forecasts are the realised 2024 block
high/low/first-touch, then the same values with additive Gaussian noise and
label flips. TEST (2025+) stays closed; the loader refuses it at read time.
Owner follow-up of 2026-10-19 to [REQUIRED_ACCURACY_FRONTIER.md](REQUIRED_ACCURACY_FRONTIER.md).

## What ran

- **Strategy:** the recommended business-day RF-L from the contract. Blocks are
  24 h, rollover-aligned and clipped to the tradable window. A limit order sits
  at the predicted first extreme (inset 0.10 of the predicted range), followed by
  a bracket: TP inset 0.10, stop 0.25 of the range beyond the first extreme.
  Unfilled orders and open positions close at block end.
- **Execution, data and costs:** the simulator, costs and parameters are
  unchanged from `67760a6`. Lake file sha256 `c6628375…4d2e`. 259 business-day
  blocks in 2024.
- **Inputs:** a first-touch label (which extreme comes first) and the high/low
  excursion.
- **Range sweep:** Gaussian noise on the predicted high and low, σ ∈ {0, 1, 2, 3,
  5, 7.5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 300, 400, 600} pips.
  First touch is kept perfect. The grid first stopped at 120 pips. Profit had not
  reached zero there, so the grid was extended once (owner instruction: "up to
  where performance is clearly lost"). Common random numbers keep every earlier
  cell identical.
- **First-touch sweep:** flip probability p = 0, 0.05, …, 0.50, with the range
  kept perfect. It is reported as the measured accuracy on non-ambiguous blocks
  (p = 0.5 measures 0.515).
- **Joint sweep:** σ ∈ {0, 5, 10, 20, 30, 40, 60, 80, 120, 200, 400} pips × p ∈
  {0, .1, .2, .3, .4, .5}.
- **Seeds:** five per noisy cell (42–46). Curves show the mean with a min–max band.
- **Costs:** every cell ran at ×1 and ×2 costs.
- **Achieved forecast metrics per cell, on the same rows:** high MAE and low MAE
  in pips; range MAE as a multiple of the NAIVE_MEAN20 range forecast (17.9 pips
  on these rows); first-touch accuracy; and Brier score. Brier assumes the
  forecaster states confidence 1 − p in its own label, i.e. it is perfectly
  calibrated, which is optimistic.
- **Business metrics:** net profit, end equity, max drawdown (hourly
  marked-to-market), Sharpe (calendar-day returns including flat days, √365),
  profit factor, trades and win rate.
- **Tables:** all per-cell values are in `results/validation_2024/validation_degradation_summary.csv`,
  generated from the retained cells (`validation_cells.csv.gz`, sha `3b43b2df…3d6c`).

## Figures

- `results/validation_2024/validation_degradation_range.png`: net profit, Sharpe
  and max drawdown against achieved range MAE / naive (symlog), with a top axis
  in pips.
- `results/validation_2024/validation_degradation_firsttouch.png`: the same three
  panels against achieved first-touch accuracy (1.0 → 0.5).
- `results/validation_2024/validation_degradation_joint.png`: heatmap of mean net
  profit with the 0-contour, at ×1 and ×2.
- `results/validation_2024/validation_equity_2024.png`: 2024 equity for the ideal
  forecast, light noise (σ 5 pips), medium noise (σ 30 pips), the joint cell
  closest to break-even (σ 400 pips, p 0.3), the naive-range baseline and
  NO_TRADE. Seed 42.

## Reference points (×1 costs, 2024)

| Forecast | Net USD | Sharpe | Max DD USD | PF | Trades | Win |
|---|---:|---:|---:|---:|---:|---:|
| Ideal (MAE 0, FT 1.00) | 38,701 | 17.8 | 161 | ∞ | 259 | 1.00 |
| σ 5 pips (MAE 3.9 pips, 0.22× naive) | 30,186 | 14.7 | 165 | 54.7 | 216 | 0.93 |
| σ 15 pips (MAE 11.7, 0.65×) | 12,138 | 7.8 | 274 | 7.8 | 131 | 0.74 |
| σ 30 pips (MAE 23.4, 1.31×) | 5,362 | 4.7 | 317 | 5.1 | 75 | 0.67 |
| σ 120 pips (MAE 93.8, 5.23×) | 1,144 | 2.1 | 252 | 3.5 | 26 | 0.69 |
| Perfect range, FT 0.515 (coin flip) | 23,332 | 13.2 | 216 | 20.9 | 259 | 0.82 |
| NAIVE_MEAN20 range + naive first touch (FT 0.50) | −2,206 | −1.9 | 2,347 | — | 125 | — |
| NO_TRADE | 0 | 0 | 0 | — | 0 | — |

At ×2 costs the ideal is 33,424 and the naive baseline −4,973.

## Break-even levels read off the curves

Linear interpolation between grid points (`app/short_horizon/analysis.zero_crossing`).
Accuracy is the measured value; ratio is achieved range MAE / naive MAE (17.9 pips).

| Sweep | ×1 costs | ×2 costs |
|---|---|---|
| Range only, first touch perfect: zero crossing | **none** up to 26.2× (σ 600 pips); mean still +266 | **none** on the mean up to 26.2× (+178); single seeds go negative from 8.7× (σ 200) |
| Range only: profit down to 50 % of ideal | 0.42× (≈ 7.5 pips MAE) | 0.40× (≈ 7.2 pips) |
| Range only: profit down to 10 % of ideal | 1.83× (≈ 33 pips) | 1.59× (≈ 29 pips) |
| First touch only, range perfect: zero crossing | **none** down to accuracy 0.515 (+23,332) | **none** (+17,748) |
| First touch only: lowest profit on grid | 60 % of ideal at FT 0.515 | 53 % of ideal |
| Joint 0-contour: FT ≥ 0.92 | no crossing up to 17.4× | no crossing |
| Joint 0-contour: FT 0.82 | no crossing up to 17.4× | 16.4× |
| Joint 0-contour: FT 0.71 | 16.1× | 3.2× (≈ 58 pips) |
| Joint 0-contour: FT 0.61 | 6.8× (≈ 123 pips) | 2.1× (≈ 37 pips) |
| Joint 0-contour: FT 0.52 | 2.5× (≈ 45 pips) | 1.5× (≈ 26 pips) |

Cost sensitivity at the ×1 break-even points: they all lose at ×2. For example,
at FT 0.52 / 2.6× (σ 60 pips, p 0.5) the mean is −111 USD at ×1 and −970 USD at ×2.
Doubling costs moves the contour roughly 40–80 % toward more accurate forecasts:
1.5× instead of 2.5× at FT 0.52, and 3.2× instead of 16.1× at FT 0.71.

## How to read this — the standing caveat

**Additive Gaussian noise centred on the truth is optimistic compared with a
real model, because the truth stays inside the error distribution.**

- Here it is extreme. With first touch perfect, RF-L stays profitable at a range
  error 26 times the naive's.
- With the range perfect, it stays profitable at a coin-flip first touch
  (win rate 0.82).
- Each input, when exact, leaks enough of the future path to pay for the other
  input being noise: an exact first touch says the block maximum comes after the
  minimum, and an exact range says where both extremes are.
- A real forecaster has neither property. Its errors are not centred on the
  realised value of each block, and its first-touch call carries no such
  conditional information.
- The break-even numbers above are therefore **upper bounds** on how much error
  is tolerable. They are not targets.

For realistic requirements, use the blend frontier in
[REQUIRED_ACCURACY_FRONTIER.md](REQUIRED_ACCURACY_FRONTIER.md). There, forecasts
move from the naive toward the truth, so errors shrink toward the naive instead
of being centred on the truth. For 24 h RF-L on VALIDATION 2024 it requires:

- first-touch accuracy ≥ 0.82 with no range skill (MAE 1.0× naive);
- ≥ 0.63 at 0.8× naive;
- any first-touch accuracy only once range MAE is ≤ 0.6× naive.

The naive-range baseline itself, at FT 0.50, loses 2,206 USD at ×1 and 4,973 USD
at ×2. A model must beat that on identical rows before any of these curves matter.

## Compute and limits

- **Compute:** 984 cells, 13.9 backtest CPU-seconds in two
  `crispdm-run -q -m 2G -t 120s` chunks. Reporting ran in one more bounded call.
- **Development data:** 2024 only, one market path. Seeds vary only the noise,
  so the seed band is not a confidence interval. 2024 is DEVELOPMENT, not a holdout.
- **Costs:** ASSUMED_NOT_BROKER_VERIFIED: 2 pip spread, 1 pip slippage per fill,
  USD 7/lot round trip, USD 10/lot per rollover. No spread widening at rollover.
- **Intrabar and clock:** hourly OHLC cannot order events inside a bar, so the
  stop-first and no-TP-on-limit-fill-bar rules are conventions. The vendor clock
  is under repair.
- **Brier:** assumes a perfectly calibrated stated confidence of 1 − p.
- **Equity figure:** the "near break-even" curve is the joint cell whose mean is
  closest to zero. The 1-D sweeps have no zero crossing to pick from.
