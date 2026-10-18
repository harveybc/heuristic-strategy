# Required-accuracy frontier for the short-horizon range-first strategy (S1)

**This is an oracle/noise study, not a model result.** Every forecast below is
the realised future (the oracle) or a deliberately degraded copy of it. No model
was trained or evaluated. Results are DEVELOPMENT evidence on TRAIN 2020-2023 and
VALIDATION 2024 only; TEST (2025-01-01 and later) was never read: the loader
filters at read time and refuses any window reaching 2025. The strategy, labels,
costs and parameters are frozen in [STRATEGY_CONTRACT.md](STRATEGY_CONTRACT.md);
every number here comes from [results/FRONTIER_TABLES.md](results/FRONTIER_TABLES.md),
which is generated from the retained cell table `results/cells.csv.gz`.

Standing limitation: the lake file's bar clock is a vendor clock labelled UTC
(weekly open at 16:00–17:00 of the label); bar-timestamp semantics are under
repair across the programme. Costs are ASSUMED_NOT_BROKER_VERIFIED.

## Setup in one paragraph

EURUSD H1 (`market_data/forex/g10/eurusd/1h.parquet`, sha256 `c6628375…4d2e`).
Blocks of 6 market hours and business days (24 h, rollover-aligned), both inside
each session's tradable window (entries from open + 12 h, flat by close − 12 h).
Targets per block: high, low and which of the two is reached first (same-bar ties
are AMBIGUOUS: 142 of 3,304 TRAIN blocks at 6 h, 5 of 1,020 at 24 h). Two
executions: RF-M (market entry toward the second extreme, bracket beyond the
first) and RF-L (limit order at the predicted first extreme, then bracket).
Stop-first on ambiguous bars; TP is never credited on an intrabar limit-fill bar.
USD 10,000, risk 1 % per trade to the stop, notional capped at 5× capital,
non-compounding. Three seeds per noisy cell; "robust" = all three seeds net > 0.

## Costs assumed (all ASSUMED_NOT_BROKER_VERIFIED)

| Cost | Base (×1) | Charged |
|---|---|---|
| Spread | 2.0 pips | half per fill |
| Slippage | 1.0 pip | per fill |
| Commission | USD 7 per 100k lot, round trip | half per fill |
| Financing | USD 10 per 100k lot per rollover, both directions | entry < rollover ≤ exit |

Round trip ≈ 4.7 pips plus financing. Sensitivity runs at ×0, 0.5, 1.5, 2 and 3.
Rollover-time spread widening is not modelled.

## Headline results

**Oracle (perfect high, low and first touch), base costs:**

| Block | Strategy | TRAIN 2020-2023 net USD (Sharpe) | VALIDATION 2024 net USD (Sharpe) | Trades/yr |
|---|---|---:|---:|---:|
| 6 h | RF-L | 350,292 (10.1) | 65,847 (15.3) | ~810 |
| 6 h | RF-M | 239,231 (10.8) | 46,829 (15.1) | ~650–710 |
| 24 h | RF-L | 181,956 (12.8) | 38,701 (17.8) | ~255 |
| 24 h | RF-M | 118,387 (12.2) | 25,145 (15.4) | ~240 |
| 6 h | comparator (existing heuristic, perfect closes) | 20,545 (2.7) | 4,397 (2.8) | ~650–710 |
| 24 h | comparator (existing heuristic, perfect closes) | 19,335 (3.9) | 4,819 (3.9) | ~235 |

Sharpe is on calendar-day equity returns including flat days and weekends,
sample SD, zero risk-free rate, annualized by √365. Maximum oracle drawdown is
USD 77–177 (intrabar marked-to-market only); the oracle never hits a stop.

**Baselines on the same blocks, base costs — every one loses or is flat:**
NO_TRADE 0; BLOCK_BUY −70,480 / −21,886 (6 h, TRAIN / VALIDATION) and −22,638 /
−9,595 (24 h); SEGMENT_BUY −6,886 / −5,488; naive range forecasts fed to the same
strategy (NAIVE_MEAN20 RF-L) −38,922 / −12,338 (6 h) and −11,854 / −2,206 (24 h);
the comparator's persistence naive makes no trade. Full table: generated §2.

## The frontier

Two families of degraded forecasts were run, and they disagree. That
disagreement is the main methodological result.

1. **Gaussian noise centred on the truth** (σ in pips, or as a fraction of the
   block's realised range): RF stays robustly profitable up to σ = 30 pips (the
   grid maximum, MAE ≈ 24 pips — about 1.8× the 6 h naive MAE) even with
   first-touch at a coin flip (one exception: 6 h VALIDATION 2024 at a coin flip
   holds only to 15 pips). Such a forecast still contains the true extreme,
   so a TP placed inside it is reached about half the time for free. **No real
   model has that property, so this family overstates what is allowed.** Kept for
   comparability with the 2026-09 sweeps (generated §5); not used for the frontier.
2. **Blend between naive and truth** (`ŷ = naive + w·(truth − naive)` for high and
   low, with the naive being the mean excursion of the last 20 completed blocks),
   with the first-touch label flipped with probability p. Range MAE is exactly
   (1 − w) × the naive MAE on the same rows. This is the family behind the frontier.

**Approximate requirement (blend family, RF-L, robust = all seeds > 0, base costs).**
Each cell gives the minimum first-touch accuracy that stays profitable at that
range MAE. Grid steps: w = 0.1, p ∈ {0, .05, .1, .2, .3, .4, .5}; bounds are
bracketed, not interpolated.

| Range MAE vs naive MEAN20 | 6 h TRAIN (naive 13.4 pips) | 6 h VAL 2024 (naive 10.1) | 24 h TRAIN (naive 24.2) | 24 h VAL 2024 (naive 17.9) |
|---|---:|---:|---:|---:|
| 1.0× (no range skill) | FT ≥ 0.80 | FT ≥ 0.91 | FT ≥ 0.69 | FT ≥ 0.82 |
| 0.9× | ≥ 0.80 | ≥ 0.91 | ≥ 0.69 | ≥ 0.71 |
| 0.8× | ≥ 0.70 | ≥ 0.81 | ≥ 0.60 | ≥ 0.63 |
| 0.7× | ≥ 0.60 | ≥ 0.70 | any (0.50) | ≥ 0.63 |
| 0.6× | any | ≥ 0.60 | any | any |
| ≤ 0.5× | any | any | any | any |

RF-M has the same frontier within one grid step (it is slightly more tolerant at
VALIDATION 2024: FT ≥ 0.60 at 0.7×, any at 0.6×; at 24 h it needs FT ≥ 0.71 at
1.0×). Read it as an **iso-profit trade-off**: first-touch accuracy and range
accuracy substitute for each other. Without any range skill the first-touch call
alone must be right about 80–90 % of the time at 6 h, or 70–82 % on business days;
with a range forecast that removes 40–50 % of the naive error, a first-touch
coin flip is enough (the range information then carries the trade). 2024 is
stricter than 2020-2023 at 6 h (quieter: mean 6 h range 28 vs 38 pips, while the
pip cost of a round trip is unchanged).

**Cost sensitivity (generated §6).** The 24 h variant is markedly more cost-robust:
at w = 0.3, p = 0.3 RF-L is robust on TRAIN up to ×2 (mean still positive at ×3,
not on every seed) and on 2024 up to ×2, while the 6 h variant at the same cell
turns negative at ×2 (TRAIN) and ×1.5 (2024). Turnover drives this: about 7,900×
capital per year at 6 h versus about 2,200× at 24 h, so every assumed per-fill cost
is paid about 1,600 times a year at 6 h. The comparator's own oracle already
loses money at ×2 costs at 6 h, and at ×3 at 24 h on TRAIN.

**Comparator verdict.** The existing heuristic with 2–12 h horizons is the weak
option: its oracle earns 6–17× less than range-first, it loses at ×2 costs (6 h),
and under the blend family it needs per-horizon MAE ≤ 0.1× (6 h) or ≤ 0.2× (24 h)
of persistence before it is profitable, even though blended forecasts always have
the right sign (directional accuracy 1.00). The cause is in the reused geometry:
`calculate_entry_geometry` scales the stop with the forecast's own excursion, so
a shrunk forecast — what regression models produce — gets stops of 1–2 pips that
the next bar's noise takes out.

## Recommendation for the forecasting lane

1. Prefer the **business-day (24 h) range-first** variant as the first target: it
   needs less accuracy than 6 h on TRAIN, holds up better under costs, and trades
   about a quarter as often.
2. Train two heads per block: a **first-touch classifier** and a **high/low
   excursion regressor** (relative to the decision price). Report on VALIDATION
   2024: first-touch accuracy on non-ambiguous blocks, and range MAE as a ratio to
   NAIVE_MEAN20 on identical rows (13.4/10.1 pips at 6 h, 24.2/17.9 pips at 24 h).
   Locate the pair on the table above.
3. A model that lands in a profitable cell is **necessary, not sufficient**. Its
   actual predictions must then be backtested through this same simulator (the
   harness accepts `H_hat`, `L_hat` and `ft_hat` arrays; a sealed prediction-CSV
   adapter is the next step) and compared with the naive on identical rows. Its
   errors will not be a scaled copy of the naive error the way the blend's are.

## Limits

- Oracle/noise only; both degradation families are constructions, not models.
  The blend assumes every forecast lies between the naive and the truth, a
  structural assumption that real errors will violate in either direction.
- The Gaussian relative family scales noise by the realised range (future volatility).
- Parameters (inset 0.10, stop buffer 0.25, edge 1.5× costs, 1 % risk, 5× cap)
  were fixed before scores and never tuned. Nothing was selected on 2024, but 2024
  was seen, so it remains DEVELOPMENT, not a holdout.
- Costs are assumed, fixed per fill, with no rollover-time spread widening, no
  triple-swap day, no rejected or partial fills, no requotes. At about 1,600 fills
  a year at 6 h, the true cost per fill decides the outcome; this must be verified
  against broker data before any claim.
- Hourly OHLC cannot order events inside a bar; the stop-first and
  no-TP-on-limit-fill-bar rules are conservative conventions, not tick truth.
- The calendar is derived from observed bar gaps (> 24 h) as a proxy for a
  published broker schedule; vendor clock semantics are unverified.
- Seeds vary only the noise; the market history is one path. Across-seed spread
  is not a confidence interval.

## Reproduce (CPU only, bounded)

```bash
PY=<python with numpy, pandas, pyarrow, matplotlib, trading-contracts>
LAKE=<financial-data checkout>/market_data/forex/g10/eurusd/1h.parquet
OUT=<scratch>/hs-short-horizon-run
crispdm-run -q -m 2G -t 120s -n hs-oracle -- env PYTHONPATH=. $PY -m pytest tests/unit_tests/test_short_horizon.py -q
for spec in "baselines 3 --save-equity" "comparator 2 --save-equity" "rf 8 --save-equity" "costs 3"; do
  set -- $spec
  for i in $(seq 0 $(($2-1))); do
    crispdm-run -q -m 2G -t 120s -n hs-oracle -- env PYTHONPATH=. $PY \
      tools/short_horizon_oracle.py run --data $LAKE --output $OUT --grid $1 --chunk $i/$2 $3
  done
done
crispdm-run -q -m 2G -t 120s -n hs-oracle -- env PYTHONPATH=. $PY tools/short_horizon_oracle.py merge --output $OUT
crispdm-run -q -m 2G -t 120s -n hs-oracle -- env PYTHONPATH=. $PY tools/short_horizon_report.py $OUT
```

Measured cost: 16 chunks, 333.3 backtest CPU-seconds in total (no chunk above 29 s),
peak memory under the 2 GiB cap, 7,536 cell-split rows. `results/MANIFEST.json` pins
the cell-table digest (`b1e72d50…9643`) and the input digest.
