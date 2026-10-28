# Business-day RF-L on VALIDATION 2024: ideal predictions, then Gaussian noise

> **Superseded axis (owner, 2026-10-19).** The owner rejected first-touch accuracy as an axis. The current
> design consumes only a predicted price path and is reported against price MAE in
> [PRICE_PATH_NOISE_SWEEP.md](PRICE_PATH_NOISE_SWEEP.md). This page is kept as the record of the earlier sweeps.

**DEVELOPMENT · VALIDATION 2024 · oracle/noise study, not a model result.** No
model was trained or selected. The forecasts are the realised 2024 block
high/low/first-touch, then the same values with additive Gaussian noise and
label flips. TEST (2025+) stays closed; the loader refuses it at read time.
Owner follow-up of 2026-10-19 to [REQUIRED_ACCURACY_FRONTIER.md](REQUIRED_ACCURACY_FRONTIER.md).

## How to read these curves (read this first)

The first study added Gaussian noise to one input while keeping the other one
perfect. That design **cannot produce a usable accuracy requirement for RF-L**.
Two mechanical effects explain why. Both are measured in
`results/validation_2024/validation_participation.png`.

**(1) Free abstention.** As σ grows, noise pushes the predicted extreme outward
about half the time.
- RF-L's limit then sits beyond the true extreme and never fills. The block
  simply has no trade, with no loss.
- At σ 600 pips, RF-L submits 109 orders on 259 eligible blocks and only 14 fill;
  95 are never reached. Win rate on the trades that remain stays around 0.71.
- Net profit decays toward zero (1.03 USD per eligible block, against 149 USD
  ideal) but cannot turn negative, because the strategy quietly stops trading.
- RF-M uses market entry and has no unfilled orders. It still abstains at
  decision time: its geometry and cost-edge checks read the truth-centred
  forecast and drop blocks whose true move is small. Orders fall from 237 to 57
  at σ 600 pips.

**(2) Exact-extreme fill.** With the range perfect, RF-L's limit sits at the
block's true extreme plus the inset.
- Every fill is at nearly the block's best price.
- The stop lies beyond an extreme that is never exceeded, so it can never be
  hit: zero stop exits on the whole first-touch sweep.
- Hence +22,749 USD and a win rate of 0.82 at a coin-flip first touch (achieved
  0.498). The losers are block-end time exits, never stops.
- RF-M shows the same property, not through the fill but through the bracket.
  Its TP inside the true extreme is always reached and its stop beyond it never
  is, so its win rate is 0.997 at a coin-flip first touch.
- This is a property of perfect knowledge of the extremes, not robustness to
  classification error.

Consequences:
- The one-input Gaussian sweeps below are kept for completeness. Their "no
  break-even" results must not be quoted as requirements.
- The requirements come from the sweeps in §"Requirements to quote", where the
  other input is held at a realistic level (the naive range or 0.8× of it; a
  first-touch accuracy of 0.60 or 0.70), not at perfection.

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
- Follow-up figures (see §"Requirements to quote"):
  `validation_participation.png` (fills and orders per eligible block, and net
  profit per eligible block, for every one-input sweep);
  `validation_degradation_rfm_range.png` (RF-M Gaussian range sweep with a
  fill-rate panel); and `validation_degradation_realistic.png` (first-touch
  sweeps at naive and 0.8× range; range sweeps at first touch 0.70 and 0.60).
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

## Requirements to quote (follow-up: RF-M range sweep and realistic sweeps)

Same data, costs, simulator and seeds (42–46): grid `val24x`, 2,366 cells, 34.4
CPU-seconds in three bounded chunks. Each first-touch level is set **exactly**
on the 2024 non-ambiguous blocks. The flips are nested, so lower accuracy flips a
superset of the blocks flipped at higher accuracy. Range at "1.0×" is the
NAIVE_MEAN20 forecast itself; "0.8×" is the blend `naive + 0.2·(truth − naive)`.
Figures: `results/validation_2024/validation_degradation_realistic.png`,
`results/validation_2024/validation_degradation_rfm_range.png` and
`results/validation_2024/validation_participation.png`. Tables:
`validation_break_even_realistic.json` and `validation_participation_summary.csv`.

**(b) RF-M (market entry), Gaussian range noise, first touch perfect.** No
crossing up to 26.2× naive MAE (+382 USD at ×1, +265 at ×2), because of
decision-time abstention and the perfect first touch. The fill rate falls from
0.92 to 0.22. Like RF-L, this sweep yields no requirement. The contract has no
RF-L participation-floor variant, so none was run.

**(c) Realistic sweeps: break-even read on the mean, and on the worst of 5 seeds
(all seeds > 0).**

| Required | Held fixed | RF-L ×1 mean (all seeds) | RF-L ×2 mean (all seeds) | RF-M ×1 mean (all seeds) | RF-M ×2 mean (all seeds) |
|---|---|---:|---:|---:|---:|
| first-touch accuracy ≥ | range = naive (1.0×) | **0.715** (0.731) | 0.816 (0.834) | **0.703** (0.766) | 0.875 (0.917) |
| first-touch accuracy ≥ | range = 0.8× naive | **0.580** (0.596) | 0.726 (0.753) | **0.542** (0.623) | 0.685 (0.757) |
| range MAE / naive ≤ | first touch 0.70 | **0.974×** (0.948×) | 0.752× (0.719×) | **0.995×** (0.918×) | 0.811× (0.766×) |
| range MAE / naive ≤ | first touch 0.60 | **0.837×** (0.811×) | 0.609× (0.589×) | **0.877×** (0.762×) | 0.752× (0.709×) |

Fill rate at the ×1 break-evens is about 0.36–0.45 for RF-L and 1.0 for RF-M.
Range MAE in pips = ratio × 17.9; for example, 0.837× ≈ 15.0 pips.

Plain reading for 2024 business days at base costs:
- With no range skill, the first-touch classifier must be right on about
  **70–72 %** of blocks to break even on average, or **73–77 %** to be
  profitable on every seed.
- A range forecast with 20 % less error than the naive lowers that to about
  **54–58 %** (mean), or **60–62 %** (every seed).
- At **60 %** first-touch accuracy, the range forecast must beat the naive's
  MAE by about **12–16 %** (mean), or **19–24 %** (every seed).
- **Doubling costs** raises every one of these requirements a lot: 0.82–0.92
  first-touch accuracy with no range skill.
- RF-L and RF-M need about the same accuracy. RF-L trades about 40 % of blocks
  where RF-M trades all of them.

Gaussian range noise at a fixed first touch still looks lenient (RF-L at FT 0.70
breaks even at 15.0× naive MAE). That is free abstention again: the fill rate
there is 0.07. It is reported in the JSON but must not be quoted.

## Break-even levels from the first study (perfect-other-input sweeps; not requirements)

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

## Additional caveat

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
  `crispdm-run -q -m 2G -t 120s` chunks for the first study. The follow-up
  (`val24x`) ran 2,366 cells in 34.4 CPU-seconds over three chunks. Reporting
  ran in a few more bounded calls.
- **Exact flips:** first-touch levels in the follow-up are set exactly by
  flipping the k non-ambiguous 2024 blocks with the smallest seeded uniforms.
  The realistic range levels are the naive itself or a blend toward the truth,
  so the "realistic" sweeps are still constructions, not models.
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
