# Short-horizon strategy contract (S0) — DEVELOPMENT

Status: frozen 2026-10-18 before any score was computed. Owner business change
of 2026-10-18: the day-trading constraint behind the old short+long (≤72 h)
strategy no longer binds, so that strategy is obsolete for this business. This
contract defines its successor family and the oracle/noise study (S1) that asks
how accurate forecasts must be for it to pay after costs. **No model is trained
or evaluated here; every number derived from this contract is an oracle/noise
result, labelled DEVELOPMENT.** No real-money trading, no TEST rows.

## 1. Data, splits and clock

- Instrument EURUSD, hourly OHLC. Source: the admitted financial-data lake file
  `market_data/forex/g10/eurusd/1h.parquet` (HistData 1-minute resample,
  provenance sha256 `c6628375b2872f16dcdbd6b1b90ed8e68bc5bdd852efdef01f2515b8712d4e2e`),
  read-only, filtered **at read** to `[2019-12-01, 2025-01-01)`. A row on or
  after 2025-01-01 is a hard error. December 2019 is used only for the calendar
  of the first 2020 week and for naive warm-up history.
- Splits by **block containment**: TRAIN = blocks inside `[2020-01-01, 2024-01-01)`,
  VALIDATION = blocks inside `[2024-01-01, 2025-01-01)`. A block crossing a split
  boundary belongs to neither. Positions are flat at every block end, so each
  split is an independent backtest from USD 10,000. TEST (2025-01-01+) stays closed.
- Clock: the file labels bars UTC, but the weekly open falls at 16:00–17:00 of
  that label (a vendor clock, DST-shifting). Bar-timestamp/clock semantics are
  under repair programme-wide; this study uses the label arithmetically and is
  DEVELOPMENT evidence with that limitation. A bar labelled `t` covers `[t, t+1h)`.

## 2. Business calendar (configured, never guessed)

- The calendar is a versioned JSON (`config/short_horizon/eurusd_h1_calendar.dev.json`)
  of **session segments** `{segment_id, market_open O, market_close C}`. It is
  derived once, by a declared rule, from observed bars: a segment is a maximal
  run of bars with no gap > 24 h; `O` = first bar label, `C` = last bar label + 1 h.
  Holidays with a closure > 24 h (e.g. 1 January) therefore split a week into two
  segments, each with its own guards (conservative). No weekday name is read
  anywhere in the code. This derivation is a DEVELOPMENT proxy for a published
  broker schedule; the final segment of 2024 is truncated at the split boundary,
  which only moves its forced flat earlier (conservative).
- Operational window per segment: entries allowed in `[S, X)`, forced flat by `F`,
  with `S = O + entry_delay_after_open_hours` (12), `F = C − force_flat_before_close_hours`
  (12) and `X = F − entry_cutoff_before_flat_hours` (0 → `X = F`). All three are
  explicit config parameters. Bars outside every segment are a hard error; no
  bar is ever invented (no weekend or gap filling).
- Daily rollover instants for financing: `ρ_k = O + 24k`, k ≥ 1, inside the
  segment (the business day starts at the session open). ASSUMED; the broker's
  rollover clock is not verified.

## 3. Block grid in market time

For block size `B ∈ {6, 24}` hours (both run in the same harness), block `k` of a
segment is `[max(O + kB, S), min(O + (k+1)B, F))`, kept only if its length is at
least `B/2`. B = 24 is the business-day variant (rollover-aligned day clipped to
the operational window); B = 6 tiles it into quarters. The grid restarts at each
segment open, so it is market time, never wall-clock time across a closure. The
block's bars are the observed bars with label in `[start, end − 1h]`; a block with
fewer than `ceil(length/2)` observed bars is skipped (status `INSUFFICIENT_BARS`)
for every strategy alike.

Decision time is the block start `t0`. The decision reference price `r` is the
close of the last observed bar ending at or before `t0` inside the same segment.

## 4. Causal labels (labelling only, never strategy inputs)

From the block's own bars only, i.e. the future interval `(t0, end]`:

- `H = max(HIGH)`, `L = min(LOW)` over the block's bars; `U = H − r`, `D = r − L`.
- First touch: `i_H` = first bar index with `HIGH == H`, `i_L` = first with
  `LOW == L`. `LOW_FIRST` if `i_L < i_H`, `HIGH_FIRST` if `i_H < i_L`,
  `AMBIGUOUS` if equal (both extremes in one hourly bar: OHLC cannot order them).
- Perturbing any bar outside a block cannot change its labels; perturbing a bar
  inside changes that block only (tested).
- Predictions are binary on first touch. For `AMBIGUOUS` truth the "oracle"
  label is a fixed coin (seed 20261018) identical in every cell, so the oracle
  never learns ambiguity from the future. First-touch accuracy is reported on
  non-ambiguous blocks.

## 5. Strategy (i): range-first (RF)

Inputs per block: predicted `Ĥ, L̂` (prices) and `F̂ ∈ {LOW_FIRST, HIGH_FIRST}`.
`R̂ = Ĥ − L̂`; `R̂ ≤ 0` or non-finite → no trade. Fixed parameters (set before
scores, never tuned): inset `i = 0.10`, stop buffer `b = 0.25`, minimum edge
`m = 1.5` round-trip costs, risk fraction 1 %, leverage cap 5×.

- **RF-M (market + bracket).** `F̂ = LOW_FIRST` → long at the block's first open;
  `TP = Ĥ − iR̂`, `SL = L̂ − bR̂`. `F̂ = HIGH_FIRST` → short; `TP = L̂ + iR̂`,
  `SL = Ĥ + bR̂`. Required at decision: `SL < r < TP` (long; mirrored short) and
  reward to TP ≥ `m` × round-trip cost. At the fill the same geometry is
  re-checked against the actual open; violated → entry refused (no fill).
- **RF-L (limit at the first extreme).** `LOW_FIRST` → buy limit `E = L̂ + iR̂`,
  then bracket `TP = Ĥ − iR̂`, `SL = L̂ − bR̂`; `HIGH_FIRST` mirrored (sell limit
  `E = Ĥ − iR̂`). If `E` is not better than `r` (marketable) the order becomes
  RF-M. Edge check on `TP − E`.
- **Wrong first touch.** RF-L waits for the first extreme; if the predicted-second
  extreme actually came first, the limit fills after the move it was meant to
  catch, the TP is no longer ahead, and the position is closed by the time exit
  at block end (or the stop). With an exact `L̂` that loss is bounded by the inset
  plus costs. RF-M is insensitive to the first-touch label when `Ĥ, L̂` are exact
  (its stop lies beyond the true extreme); the label matters only under range noise.
- **Exits.** Native bracket: TP limit and SL stop evaluated on every bar after the
  fill. Unfilled limits are cancelled at block end; an open position is closed at
  the close of the block's last bar (quote at block end). Flat at every block end,
  hence flat before `F` and never holding across a closure.
- **Sizing.** `units = floor_1000(min(0.01·C0 / |entry_ref − SL|, 5·C0 / entry_ref))`
  with `C0` = initial capital USD 10,000 (non-compounding, so cells are comparable);
  `entry_ref` = `r` (market) or `E` (limit). Zero units → no trade.
- **One net position**: one order per block, flat between blocks.

## 6. Intrabar ambiguity rule (hourly OHLC)

Per bar in position, in this order: (1) the OPEN is observed first: an open at or
through the stop fills the stop at the open (gap), an open at or through the TP
fills the TP at the open; (2) if both `LOW ≤ SL` and `HIGH ≥ TP` (long; mirrored
short) the **stop fills first** (conservative, recorded as ambiguous); (3)
otherwise whichever was touched fills at its own price. On the bar where a limit
entry fills intrabar (the open did not already cross it) only the stop may
trigger — the TP is not credited on the fill bar because its order relative to
the fill is unknown. A limit crossed at the open fills at the open and the whole
bar counts as after-fill.

## 7. Execution, costs and financing — all ASSUMED_NOT_BROKER_VERIFIED

Prices are the vendor OHLC as given (treated as mid). Costs are explicit cash
debits, each charged exactly once per fill, never by clipping to OHLC:

| Cost | Base value | Charged |
|---|---|---|
| Spread | 0.00020 price (2 pips) full spread | half per fill |
| Slippage | 0.00010 price (1 pip) | per fill |
| Commission | USD 7 per 100k lot round trip | USD 3.5/lot per fill |
| Financing | USD 10 per 100k lot per rollover, both directions | per rollover ρ with entry < ρ ≤ exit |

Values follow the programme's instrument config `i6dmo-eurusd-costs-2026-10-15.v1`
(sha `f2443141…`, all ASSUMED_NOT_BROKER_VERIFIED). Base round trip = 4.7 pips +
financing. Sensitivity: every cost × {0, 0.5, 1, 1.5, 2, 3}. Latency: the decision
at `t0` executes at the OPEN of the first block bar (declared zero added latency
beyond that quote; slippage charged). Swap is debited to equity, not merely
subtracted from a report. Rollover-time spread widening is NOT modelled (limitation).

## 8. Strategy (ii): the existing heuristic, short horizons (cheap comparator)

`app/policies/prediction_entry_exit.calculate_entry_geometry` (committed, unchanged)
is called with close forecasts at horizons `h ∈ {2,3,4,6}` (B = 6) or
`{2,3,4,6,8,12}` (B = 24) hours after `t0` (target = close of the bar labelled
`t0 + h − 1h`; absent bar → absent forecast, never filled). Parametrization only:
`profit_threshold` = `m` × round-trip cost expressed in the policy's 0.00001 point
unit (70.5 points); every other parameter at its plugin default. Corrections of the
2026-09-29 audit applied by the harness, not the policy: entry needs the ordered
forecast samples to reach TP before SL (chronological, not extrema); both
directions are evaluated from one family; TP/SL are native intrabar orders under
§6; no prediction-based early exit (the self-contradicting variant E is not used);
time exit at block end; same sizing and costs as RF.

## 9. Baselines on identical rows

Same blocks, costs and sizing caps: `NO_TRADE`; `BLOCK_BUY` / `BLOCK_SELL` (enter
at each block open, exit at block end); `SEGMENT_BUY` / `SEGMENT_SELL` (enter at
`S`, exit at `F`, paying financing); naive range forecasts fed to RF:
`NAIVE_PERSIST` (previous completed block's `U, D` and first touch) and
`NAIVE_MEAN20` (mean `U, D` of the last 20 completed blocks, majority first touch).
A block is "completed" when its end ≤ `t0`. Comparator naive: persistence close
for every horizon, with per-horizon MAE reported.

## 10. Noise model (S1)

Common random numbers: per seed (42, 43, 44) one standard-normal pair `(z_H, z_L)`
and one uniform `u` per block, reused across all levels. `Ĥ = H + σ z_H`,
`L̂ = L + σ z_L`; `σ = f · (H − L)` (relative family) or `σ` = fixed pips (absolute
family). `F̂` = true label flipped when `u < p`. The comparator gets
`ĉ_h = c_h + σ z_h` with the same σ families. Measured per cell: MAE of `Ĥ`, `L̂`
in pips, first-touch accuracy, per-horizon MAE (comparator), and the business
metrics. Noise scaled by the realised range is itself future information about
volatility; the absolute family has no such leak. Neither is a model.

## 11. Metrics

Net profit, end equity, max drawdown (USD and % on hourly marked-to-market
equity), Sharpe and Sortino on calendar-day equity returns **including flat days
and weekends**, sample SD, zero risk-free rate, annualized by √365; trades,
win rate, profit factor (net per-trade), turnover (fill notional / C0), exposure
(share of split bars in a position), cost breakdown (spread, slippage, commission,
financing), refused entries, ambiguous stop-first bars. A cell is **profitable**
when the mean net profit over its three seeds is > 0; **robust** when all three are.

## 12. Plugin decision

Range-first **needs a new plugin**; it cannot be a parametrization:

- `plugin_long_short_predictions` consumes only per-horizon CLOSE paths
  (`Prediction_h_*`, `Prediction_d_*`); its entry geometry is the extrema of the
  long family (`calculate_entry_geometry`), TP/SL are close-only market signals
  (`next()` checks `close[0]`), it has no limit entry, no block-end time exit and a
  weekday-free calendar only through `max_trades_per_5days`. There is no input
  slot for (high, low, first-touch) and no order type for "buy at the predicted low".
- `plugin_direction_atr` takes a binary direction and ATR-scaled distances, closes
  on `weekday() == 4 and hour >= 20` (weekday name guess) and checks TP on the close.
- Both rely on Backtrader brackets that the 2026-09 corrections had to patch at
  broker level for entry-bar protection and stop-first.

Implementation in this lane: `app/short_horizon/` — causal labels, the configured
calendar, an explicit hourly-OHLC event simulator with the §6 rule and the
policies — tested without Backtrader. Registering it under
`heuristic_strategy.plugins` (a Backtrader wrapper plus a `setup.py` entry) is
deferred until a real forecaster exists; nothing here is wired to a venue.

## 13. Harness reuse

`run_oracle_ceiling.py` monkey-patches `direction_atr` around a sibling repository's
oracle and writes the committed root records; `sweep_noise.py` hard-codes absolute
paths and needs a running prediction service. Neither can host a (high, low,
first-touch) target without rewriting it, so the new harness
`tools/short_horizon_oracle.py` reuses their method — fixed oracle, monotone noise
grid, accuracy reported next to profit — plus the committed entry-geometry policy,
and writes only under its own output directory.

## 14. Business rule: every order carries SL and TP (owner, 2026-10-19)

**Toda orden lleva SL y TP calculados y efectivos desde la entrada; el cierre anticipado del agente es adicional, nunca sustituto.**

- **Levels are broker-side orders.** SL and TP are set at entry as native
  broker orders. A strategy may move the SL (for a trailing stop) but never
  remove it.
- **Typed refusal.** The simulator refuses an order without a side, positive
  integer units, a finite SL and a finite TP on the correct sides of the entry
  (`OrderRuleError`).
- **No symbolic levels.** Each distance must be at least the broker minimum (0,5
  pip, ASSUMED) and at most `effectiveness_c` × the causal block sigma.
  - Causal block sigma = mean range of the last 20 completed blocks.
  - `effectiveness_c` = 1,5, fixed a priori, so each level has a real chance of
    being hit within the block.
  - Strategies filter such orders before sending them (counted).
- **Per-run reporting.** Each run reports the shares of exits by TP, SL, early
  close and time/forced flat. A TP hit in fewer than 5 % of ≥ 20 trades is flagged
  `tp_rarely_hit`.
- **Retro-check.** Committed RF-M/RF-L orders always set both levels (tested).
- **Passive baselines are excluded.** BLOCK_BUY/SELL and SEGMENT_BUY/SELL of the
  first study are reference benchmarks, not strategy orders. They used placeholder
  levels and cannot be run under this rule.

## 15. Business rule: volume in units from `rel_volume` (owner, 2026-10-19)

- **Units are integer base-currency (EUR) units**, as the broker API takes them:
  `units_max = floor(balance_USD · rel_volume · leverage / price_USD_per_EUR)` and
  `units = floor(f · units_max)`, with f ∈ (0, 1].
- **Balance is the current balance at each entry**, so sizing compounds. Below
  the broker minimum units, the order is not traded (counted).
- **`rel_volume`:** default 0,10, bounded (0, 0,20]. It and `f` are declared in a
  manifest for a later sweep or DEAP search on the inner validation year only,
  never on 2024.
- **`rel_volume_basis = margin`** (owner-confirmed legacy convention). `notional`,
  i.e. without the leverage factor, is a one-flag alternative, and every run
  reports which basis it used.
- **Leverage is an account scenario, not a tuned parameter:** 1:10, 1:50 and 1:100
  (default, matches the plugins). With margin = 10 % of balance, the notional is
  1×, 5× and 10× the balance respectively.
  - Each run reports margin used, max exposure (notional/balance) and margin-call
    events. Closeout is ASSUMED at equity ≤ 50 % of the margin used.
  - Equity reaching zero is ruin: no new orders, no Sharpe, invalid.
- **Legacy observation (plugins not edited).** `app/heuristic_strategy.py` and the
  `direction_atr` / `api_predictions` plugins compute
  `max_from_cash = cash · rel_volume · leverage` and compare it with a unit
  count. For EURUSD, cash is USD and units are EUR, so the USD→EUR division by
  price is missing there. It is applied here.
- **The old sizing** (1 % risk to the stop, notional ≤ 5× the *initial* capital,
  non-compounding) is kept only as a labelled reference arm for the earlier studies.
- **Every order record** in outputs carries units, side, entry, SL and TP.
