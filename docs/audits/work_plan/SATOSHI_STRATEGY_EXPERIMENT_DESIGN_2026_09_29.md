# Business-experiment design: exit-only interventions and the short/long noise factorial

Satoshi, successor technical lead. 2026-09-29.
Repository: `heuristic-strategy`. Branch `satoshi/strategy-experiment-design-20260929`,
from tip `79977b2`.

**Nothing executes under this document.** No broker call, no new sweep, no GA,
no GPU, no service, no parameter selected from any score. The only computation
performed to write it was read-only characterization of committed input CSVs
(§2); no strategy, broker or backtest was run.

This is a **separate lane**. It does not replace the original strategy, and no
replacement strategy is proposed. Where a convention is not in the code, it is
named here as absent rather than supplied.

---

## 1. What the existing strategy actually does, according to its own code

Source of trading decisions: `app/plugins/plugin_long_short_predictions.py`,
`Plugin.HeuristicStrategy.next`, delegating the arithmetic to
`app/policies/prediction_entry_exit.py`. Registered as `default` and
`ls_pred_strategy` in `setup.py`.

### 1.1 Two forecast families, and how many horizons each has

The strategy does not declare a horizon count. It **counts columns** at
construction (`plugin_long_short_predictions.py:254-256`):

- short family = columns named `Prediction_h_*` → `num_hourly_preds`
- long family = columns named `Prediction_d_*` → `num_daily_preds`

So the horizons are whatever the input file supplies. The conventions actually
present in the repository are:

| Source | Short family | Long family |
| --- | --- | --- |
| `app/config.py:19` `time_horizon: 6` with `app/data_processor.py` generators | next 6 **rows** (`create_hourly_predictions`) | rows at +24, +48, … +144 (`create_daily_predictions`) |
| Committed fixtures `tests/data/ideal_predictions_{hourly,daily}_d3.csv` | `Prediction_h_1..6` = `CLOSE[t+1h] … CLOSE[t+6h]`, verified exactly (MAE 0.0 to 2.3e-07) | `Prediction_d_1..6` ≈ `CLOSE[t+24h] … CLOSE[t+144h]`, verified only approximately (MAE 3.5e-06 at 24 h rising to 4.86e-03 at 144 h) |
| Retained sweep tooling `tools/price_noise_sweep.py:24-25` | offsets 1,2,3,4,5,6 | offsets 24,48,72,96,120,144 |

Recovered convention: **twelve horizons, six short (1–6) and six long
(24–144)**, in the units of the supplying file.

Two defects in the code's own generator, named:

- `create_hourly_predictions` flattens **every** column of the base frame
  (`OPEN,LOW,HIGH,CLOSE`), so auto-generation with `time_horizon: 6` yields
  24 values per row, which `evaluate_candidate` then renames positionally to
  `Prediction_h_1..24`. Auto-generated mode is therefore not a six-horizon
  close-price path. Only file-supplied inputs match the fixture convention.
- Both generators use **row offsets**, not elapsed clock time. The retained
  audit (`docs/noise_price_sweep/AUDIT_20260927.md`, finding 3) measured that
  on this data a 144-row offset spans 191–316 elapsed hours, median 192. Which
  of the two is intended is an **open owner decision**, unresolved in code.

### 1.2 Entry — long family only

`plugin_long_short_predictions.py:341-390` → `calculate_entry_geometry`
(`prediction_entry_exit.py:73-121`). Evaluated at each bar's **CLOSE**; the
market order fills at the next bar's OPEN.

With `c` the current close, `P` the long-family path, `pip = pip_cost`:

```
profit_buy   = (max(P) - c) / pip
drawdown_buy = max((c - min(P)) / pip, min_drawdown_pips)
rr_buy       = profit_buy / drawdown_buy
profit_sell  = (c - min(P)) / pip
drawdown_sell= max((max(P) - c) / pip, min_drawdown_pips)
rr_sell      = profit_sell / drawdown_sell
long  if profit_buy  >= profit_threshold and rr_buy >= rr_sell
short if profit_sell >= profit_threshold and rr_sell >  rr_buy
else no trade
```

Three facts that follow from the code and matter to any design:

1. **The short family plays no part in entry.** Line 359 passes only
   `daily_preds`. Short forecasts affect exits and nothing else.
2. Only path **extrema** are used. Reordering the six long samples leaves the
   entry unchanged; there is no first-passage or path-ordering logic.
3. There is a **dead zone**: when only `short_signal` holds but
   `rr_sell <= rr_buy`, neither branch fires and no trade is taken.

Gating before entry: a position must be flat (line 288 returns early while in
position), and `max_trades_per_5days` (default 3) is enforced over entry
timestamps within the last 5 days (lines 341-342).

### 1.3 Initial stop and target

Set once at entry from the same long-family extrema, never trailed, never
re-armed (`prediction_entry_exit.py:89-90` and the symmetric sell lines):

```
long:  TP = c + tp_multiplier * profit_buy  * pip     SL = c - sl_multiplier * drawdown_buy  * pip
short: TP = c - tp_multiplier * profit_sell * pip     SL = c + sl_multiplier * drawdown_sell * pip
```

At the resolved defaults `tp_multiplier=0.9`, `sl_multiplier=2.0` the target sits
**inside** the forecast extremum and the stop **outside** the opposite extremum.

**TP and SL are close-only signals, not protective orders.** Lines 293/297 (long)
and 315/319 (short) compare the current close and submit a market order for the
next open. The retained audit measured seven trades whose bars touched SL,
did not touch TP and recovered before the close — those bars did not trigger the
mandatory branch. There is no bracket order, no intrabar stop, and no
bid/ask replay anywhere in `app/`.

### 1.4 Early exit — the seven variants the code defines

`should_early_close` (`prediction_entry_exit.py:124-182`). All compare forecast
levels against the **stop price**, not against entry or expectation. For a long
(shorts are the sign mirror):

| Variant | Rule (long) |
| --- | --- |
| A | `min(short + long) < SL` |
| B | `min(long) < SL` |
| C | `min(short) < SL` |
| D | `min(short) < SL` **and** `min(long) < SL` |
| E | `0.6*min(short) + 0.4*min(long) < SL`; if one family is empty, either trigger |
| F | `min(short) < SL - 0.5*abs(SL-entry)` **or** `min(long) < SL` |
| G | never (TP/SL only) |

The resolved default is **E** (`plugin_params`, line 44). The source contradicts
itself three ways: the comment block at lines 30-37 says "D = both must agree
(DEFAULT)", the nested `HeuristicStrategy.__init__` signature defaults to `'D'`,
and `plugin_params` says `'E'`. Any launcher that reads `plugin_params` — the
stock `evaluate_candidate` and the retained native adapter — gets **E**. This is
a resolved value, not a recovery of a historical configuration.

### 1.5 Sizing — and where capital feedback enters

`compute_legacy_order_size` (`prediction_entry_exit.py:184-203`):

```
rr >= upper_rr_threshold  -> max_order_volume            (1,000,000)
rr <= lower_rr_threshold  -> min_order_volume            (   10,000)
otherwise                 -> linear interpolation in rr
then                      -> min(that, available_cash * rel_volume * leverage)
```

`compute_size` passes `self.broker.getcash()` (line 371 path). With
`rel_volume=0.02, leverage=100` the cap is `2 x cash` in notional.
**This is capital feedback**: the cap is a function of the running account, so an
exit change that raises equity raises every later position size.

The docstring on `rel_volume` says "uses max 2% of balance for each order". The
formula does not do that; the audit already recorded that 0.02 is not a
guarantee of 2% capital at risk to the stop. **A margin-fraction convention is
absent from `app/`.** The 5%-initial-margin reading is an owner instruction
implemented only in `tools/` (`margin_order_size`, `corrected_price_noise_sweep.py:70-75`),
where the cap becomes `min(cap, equity*0.05*leverage/price)` using
`broker.getvalue()` — still capital feedback, now on equity rather than cash.

### 1.6 Cost conventions

Two different sets exist, and they disagree.

Stock launcher, `plugin_long_short_predictions.py:153-167`:

| Item | Value | Mechanism |
| --- | --- | --- |
| Starting cash | 10,000 | `setcash` |
| Commission | `commission_per_lot/100000` = 7e-05 per unit | `setcommission` |
| Spread + slippage | `(spread_pips + slippage_pips) * pip_cost / 2` = 1.5e-05 per side | `set_slippage_fixed`, **clipped to the bar range by backtrader** (audit: 2,899 of 24,786 fills paid less than configured) |
| Swap | `duration_bars/24 * volume/100000 * swap_per_lot_per_day` | subtracted in `notify_trade` (lines 454-457) from the **reported** trade PnL only, never debited to the broker |

The swap treatment means the stock plugin's printed trade PnL and the broker's
equity are different quantities; the retained native results record that the
plugin display deducts swap a second time relative to a broker that charges it.

Retained execution adapter, `tools/` (the accounting actually used by the
verified sweeps): commission and friction as one explicit cash debit
`7/100000 + 0.00010` per unit per side (`ExplicitCosts`), swap 10 USD per 100,000
units per calendar day in both directions debited to the broker, leverage 100:1,
starting cash 10,000, allocation cap 1,000,000 units.

### 1.7 Fitness, seeding and out-of-sample machinery

- Fitness is single-objective, `weights=(1.0,)` on **profit** = final broker
  value − 10,000 (`app/optimizer.py:97`, `evaluate_candidate` return).
- The GA **is** seeded: `random.seed(42)` at `app/optimizer.py:115` and
  `app/walk_forward_optimizer.py:108`. `AGENTS.md`'s statement that "GA runs are
  stochastic and unseeded" is contradicted by the code. No numpy seed is set,
  and numpy is used only for `np.std` in reporting.
- The only out-of-sample machinery is `run_wfo.py` / `app/walk_forward_optimizer.py`:
  anchored rolling folds, defaults `train_years=3`, test years 2009–2019,
  `min_trades=10`, GA per fold, one evaluation per unseen year. It defaults to
  the `regime_wfo` plugin, which needs no predictions; it is **not** wired to
  `ls_pred_strategy`.

### 1.8 Conventions that are absent (named as absent, not supplied)

- No bid/ask spread, no protective/intrabar orders, no order type but market.
- No unclipped fixed-friction implementation in `app/`.
- No margin-fraction or capital-at-risk convention in `app/`.
- No per-horizon naive baseline anywhere in `app/`. The only naive references in
  the repository live in `tools/` and are **endpoint-only** (H6 and H144).
- No development/reserved-evaluation split in the single-run path.
- No currency conversion: USD-denominated quote-currency P&L is treated as the
  account currency in `app/`.
- No uncertainty or confidence channel is consumed; only point paths.
- No error-structure model of any kind: the retained noise generator is
  `np.random.default_rng(seed).standard_normal(shape)`, independent across both
  rows and columns by construction.
- `pip_cost = 1e-05` means "pip" in this code is a fifth-decimal point on
  EURUSD. `profit_threshold=5` is therefore 0.00005 (half a conventional pip)
  and `min_drawdown_pips=10` is 0.0001 (one conventional pip). The parameter
  names overstate the thresholds tenfold.

### 1.9 What the prior sweep already established, and what it did not

`docs/noise_price_sweep/` and `run_out/native_conditional_20260927_full` hold a
**completed and independently verified** short/long factorial against this exact
plugin: 241 unique cells, 13,590 origins, 4 conditional slices × 21 ratios ×
3 seeds, 1,461.693 CPU s, manifest SHA256
`958726415c09ced866c7d8164dfe4e49b99b4ccb8e0e5e7059fa26effdbba94f`. Its own
records already state the limits this design must repair: equal MAE is not the
persistence signal; endpoint-only calibration; iid noise; equity-coupled sizing;
close-only execution; non-monotonic response not removed.

---

## 2. Measured characterization of the real inputs (read-only, 2026-09-29)

Computed today from committed CSVs under `tests/data/` against
`phase_2_3_base_d3.csv` closes, with the same admission rule as the retained
tooling (`elapsed_targets`: an origin is admissible when its H6 and H144
elapsed-hour targets both exist). No strategy was run.

### 2.1 Real per-horizon error against per-horizon naive

Arm R1, ANN short + ANN long, 4,546 admissible origins,
2018-05-16 01:00 … 2019-05-15 12:00:

| Horizon | model MAE | naive MAE | ratio | n |
| --- | ---: | ---: | ---: | ---: |
| 1 h | 0.00074331 | 0.00068227 | 1.089 | 4546 |
| 2 h | 0.00112338 | 0.00098122 | 1.145 | 4546 |
| 3 h | 0.00136457 | 0.00122280 | 1.116 | 4546 |
| 4 h | 0.00156857 | 0.00142750 | 1.099 | 4546 |
| 5 h | 0.00209878 | 0.00160824 | 1.305 | 4546 |
| 6 h | 0.00215799 | 0.00177731 | 1.214 | 4546 |
| 24 h | 0.00421807 | 0.00374345 | 1.127 | 3612 |
| 48 h | 0.00563735 | 0.00543565 | 1.037 | 2390 |
| 72 h | 0.00542054 | 0.00523724 | 1.035 | 2094 |
| 96 h | 0.00512535 | 0.00503800 | 1.017 | 2103 |
| 120 h | 0.00607112 | 0.00603525 | 1.006 | 3306 |
| 144 h | 0.00704699 | 0.00705759 | **0.998** | 4546 |

This single table is the empirical case for requirement 2: the ratio to naive
runs from **0.998 at H144 to 1.305 at H5**. One denominator reports this arm as
"about naive" and hides that it is 30% worse than naive at H5, the horizon the
early-exit rule weights most heavily under variant E.

Other real arms measured, same method:

| Arm | Window | Origins | Ratio to naive, range over 12 horizons |
| --- | --- | ---: | --- |
| R1 ANN short + ANN long | 2018-05-16 … 2019-05-15 | 4,546 | 0.998 – 1.305 |
| R2 LSTM short + ANN long | 2018-05-16 … 2019-05-15 | 4,546 | 0.998 – 10.09 (short family 3.9–10.1) |
| R3 CNN short + CNN long (`app/config.py` defaults) | 2017-04-07 … 2018-03-21 | 4,393 | 1.613 – 18.63 |

R3 is the pair the shipped default config points at. It carries a **level bias**
of −0.00530 (short) and −0.00949 (long) against the contemporaneous close, with
SD 0.0156 / 0.0149; its MAE is nearly flat in the target offset, which is the
signature of a scale or de-normalization defect rather than a forecast. Declared
here, **not silently corrected**.

### 2.2 Real error structure — what iid noise does not reproduce

Pairwise error correlations, arm R1 (nan-aware, pairs with n>200):

| Block | Mean correlation |
| --- | ---: |
| within short family (1–6 h) | **0.690** |
| within long family (24–144 h) | **0.699** |
| short × long cross block | **0.224** |

Temporal autocorrelation of the error series, arm R1:

| Horizon | lag 1 | lag 2 | lag 6 | lag 24 |
| --- | ---: | ---: | ---: | ---: |
| 1 h | 0.192 | 0.056 | 0.079 | 0.055 |
| 6 h | 0.850 | 0.683 | 0.052 | −0.002 |
| 24 h | 0.948 | 0.895 | 0.687 | −0.042 |
| 144 h | 0.983 | 0.966 | 0.899 | 0.617 |

Arm R3 is more extreme still: within-short mean 0.994, within-long 0.787,
cross-block 0.671, lag-1 autocorrelation 0.995 at every horizon.

The retained generator produces **zero** for all of these in expectation. This is
the quantified content of requirement 4: matching a mean absolute error while
setting every one of these numbers to zero changes what the entry extremum and
the variant-E weighted minimum see on consecutive bars, not merely the error's
size.

---

## 3. Design (a) — fixed entries, sizing and initial stop/target, exit-only interventions

**Question it answers.** What does an exit rule contribute when nothing upstream
is allowed to move?

**Why a frozen ledger is possible at all.** `calculate_entry_geometry` depends
only on the current close and the long-family path. It never reads the exit
variant, the position or the account. Therefore direction, TP and SL at a given
origin are **exit-independent and can be frozen exactly** — this is an acceptance
test below, not an assumption. What is *not* exit-independent in the native
machine is the *set* of origins that become entries, because a position blocks
new entries and `max_trades_per_5days` counts realized entries.

### 3.1 The unit-trade harness

Each frozen entry is replayed as an **independent unit trade**. No position
blocking, no frequency coupling, no account:

- entry at the next bar's OPEN after the origin, as the native code does;
- **size fixed at 100,000 units for every unit trade** — constant, never a
  function of cash or equity;
- TP and SL exactly the frozen ledger values;
- exit by the treatment's rule, evaluated at each bar's CLOSE with the market
  fill on the next OPEN, exactly as `HeuristicStrategy.next` does;
- **censoring**: a unit trade that reaches neither barrier is liquidated at the
  CLOSE of the bar carrying its own H144 elapsed-hour target. This terminator is
  taken from the strategy's own long horizon; censoring counts are reported per
  cell and never dropped.
- costs: the explicit adapter set of §1.6 (`7/100000 + 0.00010` per unit per
  side, swap 10 USD per 100,000 units per calendar day) — one convention, stated.

### 3.2 Treatments: only the rules the code defines

`exit_variant ∈ {A, B, C, D, E, F, G}`. Seven levels, no invented eighth. G is
the exit-free reference; E is the resolved production default.

### 3.3 Ledger bases — measured, not assumed away

The ledger is part of the treatment definition, so it is varied rather than
fixed silently:

| Ledger | Definition | Role |
| --- | --- | --- |
| **L0** | every origin where the native entry gate fires, ignoring position state and the frequency cap | primary; exit-agnostic by construction |
| **LG** | the entries the native sequential machine actually takes under variant G | sensitivity: strategy-realizable, exit-free basis |
| **LE** | the entries it takes under variant E | sensitivity: strategy-realizable, production basis |

LG and LE are read from the retained per-cell `decisions.json` / `fills.csv`
where a matching cell already exists, and regenerated otherwise. Reporting all
three makes ledger dependence a measurement. The design never claims L0 is the
strategy's entry set.

### 3.4 Input arms crossed into design (a)

| Code | Input | Seeds |
| --- | --- | --- |
| P0 | actual persistence, `CLOSE[t]` repeated over 12 columns | none (deterministic) |
| O0 | constructed oracle, `CLOSE[t+h]` for the 12 elapsed-hour offsets | none |
| R1, R2, R3 | the real model arms of §2.1, on their own windows | none (single realization) |
| N-IID | oracle + iid noise at per-horizon naive calibration | 42, 43, 44 |
| N-COR | oracle + correlated noise per §4.4 | 42, 43, 44 |

P0 and O0 are controls, not treatments. The `ideal_predictions_*` fixtures are
**not** used as the oracle — §1.1 shows the daily fixture is only approximate;
they may be run as a separate named arm but never labelled ideal.

### 3.5 Reported quantities

Per cell: the per-unit-trade distribution of pips and USD (mean, SD, median,
quartiles, n), separated cost decomposition (commission, friction, swap),
censoring count, count of early exits attributable to each family, the barrier
that terminated each unit, and the **12-vector of per-horizon MAE and its
12-vector of per-horizon naive ratios** on that cell's own origin population.

Exit contrasts are reported as **paired differences against variant G on the
same ledger, same input arm, same seed**, never as cross-arm means.

---

## 4. Design (b) — the original strategy's short-noise × long-noise factorial

**Question it answers.** How does the whole strategy, with its own sequential
single-position machine and its own equity-coupled sizing, respond when the
short family and the long family degrade separately and together?

The strategy defines the two factors itself (§1.1): factor **short** = the
`Prediction_h_*` family, factor **long** = the `Prediction_d_*` family. Entry
reads only long; early exit reads both. That asymmetry is the reason the factorial
is worth running at all, and it is the strategy's property, not a design choice.

### 4.1 What is carried forward unchanged from the verified prior run

Same plugin source, same resolved variant E, same base CSV and admission rule,
same 21 ratio levels `0,.05,.1,.15,.2,.3,.4,.5,.6,.75,.9,.95,1,1.05,1.1,1.25,1.5,1.75,2,2.5,3`,
same seeds `42,43,44` with draws fixed across levels within a seed, same explicit
costs, same margin cap, same close-only execution, same "manifest written before
any backtest" and per-cell artifact retention discipline. The prior run is
preserved, never overwritten.

### 4.2 Structure of the successor grid

| Stage | Cells | Purpose |
| --- | ---: | --- |
| B0 pilot | 5 | ratios {0,1}, seed 42, plus the persistence control; projects the rest |
| B1 four slices | 241 | the prior conditional design (`long_naive`, `short_naive`, `long_ideal`, `short_ideal`) re-run under **correlated** noise |
| B2 crossed core | 75 | a genuine crossed factorial, short ratio × long ratio ∈ {0, 0.5, 1, 1.5, 2}², 3 seeds |
| B3 real arms | 8 | R1, R2, R3 and P0, each at both sizing modes |
| B4 iid regression | 241 | the retained iid cells, reproduced byte-for-byte as a regression against the stored draws |

B2 is the part the prior design lacked: four slices through the (short, long)
plane cannot show an interaction, and the prior results already contain a
non-monotonic response (short ratio 0.1 with long ideal averaging above the
noiseless cell) that a slice design cannot attribute. The full 21×21×3 = 1,323
crossed grid is **not** proposed: at the measured 6.07 CPU s per cell it is
about 8,000 CPU s, beyond the frozen 3,600 s bound, and the bound is not to be
shrunk or evaded — a larger grid must be requested explicitly.

### 4.3 Two sizing modes, never combined

Every cell in B1–B3 is run twice:

- **EQUITY_FEEDBACK** — the strategy's own sizing: RR interpolation, then the
  margin cap on `broker.getvalue()`. This is the whole-strategy number.
- **FIXED_UNITS** — constant 100,000 units per entry, no dependence on cash or
  equity. Same entries, same barriers, no compounding.

They appear in **separate tables with separate column names**
(`net_usd_equity_feedback`, `net_usd_fixed_units`). They are never summed,
averaged or plotted on one axis. Each cell additionally reports total units
traded and mean entry notional, so a reader can see how much of an
EQUITY_FEEDBACK difference is compounding rather than exit quality.

### 4.4 The correlated-noise generator

Replaces `standard_normal(shape)`. Construction, per seed:

1. Draw `w[t] ~ N(0, I_12)` iid in `t`.
2. Give each column its own AR(1) memory:
   `u[t,j] = phi_j * u[t-1,j] + sqrt(1 - phi_j^2) * w[t,j]`, with `phi_j` the
   measured lag-1 autocorrelation of that horizon's real error (§2.2).
3. Impose cross-horizon structure: `z[t,:] = L @ u[t,:]` with `L = chol(C)` and
   `C` the measured 12×12 real error correlation matrix (§2.2), nearest
   positive-definite if needed, with the adjustment recorded in the manifest.
4. Rescale each column to unit mean-absolute value so the existing per-horizon
   calibration arithmetic is unchanged.

`C` and `phi` are **estimates from one real arm on one window**, recorded in the
manifest with their source and origin count. They are not a law. The iid
generator remains a named arm (N-IID, stage B4) so the iid-versus-correlated
difference is measured rather than asserted.

### 4.5 Per-horizon calibration replaces endpoint calibration

The prior design calibrated amplitude at H6 and H144 only. The successor
calibrates and reports **all twelve**: for each horizon `j`,
`sigma_j = naive_MAE_j / mean_over_seeds(mean |z[:,j]|)`, so ratio 1 means
"MAE equals naive at that horizon", horizon by horizon. The manifest and
`sweep.csv` carry all 12 naive denominators, all 12 realized MAEs and all 12
ratios for every cell. The single normalized x-axis is retained only as a
presentation convenience and is labelled with the horizon it refers to.

The standing qualification is restated in the plan and in every output: **an
error equal to a naive's error is not the same as a naive's predictions.** At
ratio 1 the inputs are still future prices plus noise; the retained audit
measured 74.9% directional accuracy at the H144 marker against 49.8% for
always-up on the same rows. P0, actual persistence, is a separate measured
control — and it takes zero entries and makes zero profit, because `CLOSE[t]`
repeated gives `max(P) = min(P) = c`, so `profit_buy = profit_sell = 0 <
profit_threshold`. That is a property of the entry rule, recorded, not repaired.

---

## 5. The five requirements, per design, with evidence

| # | Requirement | Design (a) | Design (b) |
| --- | --- | --- | --- |
| 1 | Real persistence inputs, not only oracle + MAE-matched noise | **Satisfied in part.** Arms R1/R2/R3 are real committed model outputs with real error structure (§2.1), plus the P0 persistence and O0 oracle controls. Limits in 5.1. | **Satisfied in part**, stage B3, same arms and same limits. |
| 2 | Per-horizon naive denominators | **Satisfied.** 12 denominators recomputed on each cell's own origin population; evidence that one denominator hides the answer is the 0.998–1.305 spread measured today. | **Satisfied**, §4.5; replaces the prior endpoint-only calibration. |
| 3 | Paired seeds | **Satisfied for the synthetic arms**: seeds 42/43/44, draws fixed across levels within a seed, and additionally reused across exit variants and ledgers so every contrast is seed-paired. **Not applicable** to R1/R2/R3/P0/O0 — single realizations, so no SD is reported for them. | **Satisfied**, identically; contrasts reported as seed-paired differences against each curve's own ratio-1 point. |
| 4 | Temporally and cross-horizon correlated errors | **Satisfied by construction** via the N-COR arm; targets measured today (within-short 0.690, within-long 0.699, cross 0.224, lag-1 0.192→0.983). N-IID retained as the named contrast. | **Satisfied**, §4.4, with stage B4 as the iid regression so the difference is measured. |
| 5 | Whole-strategy capital feedback reported separately | **Satisfied by design**: the unit-trade harness has no account at all — size is constant, so no exit effect can compound. Verified by the ×10-cash acceptance test. | **Satisfied**: every cell run in both FIXED_UNITS and EQUITY_FEEDBACK, reported in separate tables, never combined, with units-traded and mean-notional diagnostics. |

### 5.1 What requirement 1 cannot yet reach, precisely

- A real **short and long family from the same model on the same window** exists
  only for ANN (R1) and for the level-biased CNN pair (R3). R2 mixes two models.
- **No real arm covers the synthetic arms' window.** R1/R2 span 4,546 origins in
  2018-05-16…2019-05-15; R3 spans 4,393 origins in 2017-04-07…2018-03-21; the
  synthetic sweeps span 13,590 origins. Real-arm and synthetic-arm scores are
  therefore **NOT_COMPARABLE** unless restricted to a common origin population,
  and the design forbids the unrestricted comparison.
- **No real arm has more than one realization**, so requirement 3 cannot be met
  on real inputs. Producing seeded real inputs means training predictor models,
  which is not authorized here and is not requested by this document.
- R3's level bias (§2.1) is a defect of the committed file. It is reported as an
  arm property; no de-biasing transform is applied, because a de-biased file
  would no longer be a real model output.

### 5.2 Other limits that no part of this design removes

- Execution stays **close-only**. Nothing here becomes a broker or tick replay,
  and the audit's seven recovering-SL bars remain unmeasured as a profit effect.
- Whether the horizon convention is market bars or elapsed hours is still an
  **open owner decision**. This design uses elapsed-hour targets and names the
  plugin's own generator as row-offset based.
- The observed non-monotonic response is not to be smoothed, seed-selected or
  fitted to an exponential law. No profitability threshold is presumed.
- The variant-E default and the TP/SL multipliers are **inputs**, not outcomes.
  No parameter may be chosen from any row produced under this design; that is
  already a blocked action in `docs/noise_price_sweep/PROJECT_METHOD_STATE.json`.

---

## 6. Development selection versus reserved evaluation — which rows belong to which

### 6.1 Development rows (selection, plotting and inspection permitted)

| Rows | Window |
| --- | --- |
| All design (a) unit-trade cells: 7 variants × 3 ledgers × 7 input arms | R1/R2 window 2018-05-16…2019-05-15; R3 window 2017-04-07…2018-03-21; synthetic arms on the sweep's 13,590-origin population, all ≤ 2019-05-15 |
| Design (b) stages B0, B1, B2, B4 | the sweep's existing origin population |
| Design (b) stage B3 real arms | each arm's own window as above |

These rows may be looked at and may be used to choose which exit variants,
which noise structure and which sizing diagnostics go forward.

### 6.2 Reserved evaluation rows (no looking, no selection, nothing executed here)

`tests/data/eurusd_hour_2005_2020.csv` spans 2005-05-02 00:00 … 2020-04-29 22:00
(93,084 rows), verified today. Reserved window: **2019-05-16 00:00 …
2020-04-29 22:00** — strictly after the last development origin (2019-05-15 12:00)
and after every real arm's last row.

- **Rows currently in this set: none.** Nothing is executed on it under this
  order, and no result from it exists.
- Its single future use: **one** pass, **one** configuration, chosen in writing
  and committed before that pass runs, reported once, with the same 12-horizon
  naive table and the same FIXED_UNITS / EQUITY_FEEDBACK separation.
- A second, **separate** evaluation route is the existing anchored WFO
  (`run_wfo.py`, `train_years=3`, per-year unseen folds). It is not mixed with
  the reserved-window route, and its folds are not development rows either — it
  requires wiring `ls_pred_strategy` into a harness that currently defaults to
  `regime_wfo`, which is unbuilt.
- Hard rule: **no parameter, exit variant, ledger basis, noise structure or
  input arm may be selected using a reserved row.** If a reserved-window result
  disagrees with development, the disagreement is reported; the development
  choice is not revised and re-evaluated on the same reserved rows.

---

## 7. Acceptance tests to be red before any module exists

Following this repository's own S9 convention.

1. **Ledger identity.** Frozen direction, TP and SL for every origin equal
   `calculate_entry_geometry`'s output exactly, and are **invariant across all
   seven exit variants** — proving TP/SL are exit-independent rather than assuming it.
2. **Exit-only isolation.** Across the seven variants on one ledger, entry
   timestamps, directions, entry prices and unit sizes are byte-identical; only
   exit times, exit prices and costs differ.
3. **No capital feedback in FIXED_UNITS.** Multiplying initial cash by ten leaves
   every unit trade's pips and size identical. Fails if any sizing path reads
   cash or equity.
4. **Capital feedback is present in EQUITY_FEEDBACK.** The same perturbation
   *does* change sizes — so mode 5 is a real separation, not a relabelling.
5. **Correlated generator fidelity.** The realized 12×12 correlation matrix and
   the realized lag-1 vector match the manifest targets within a declared
   tolerance; any nearest-PD adjustment to `C` is recorded.
6. **iid regression.** The N-IID arm reproduces the retained
   `standard_normal_seed*.npy` digests and the prior cells' MAEs exactly.
7. **Per-horizon naive.** All 12 denominators, recomputed independently from raw
   CSV bytes, match the manifest to better than 1e-15.
8. **Real-arm arithmetic.** The §2 tables reproduce from raw bytes.
9. **Censoring is reported, not dropped.** A synthetic arm engineered so no unit
   reaches a barrier yields a cell whose censoring count equals its unit count.
10. **Refusals that must fail loudly.** A cell whose supposedly fixed family
    changed; a ledger with a substituted size; a cross-window real-vs-synthetic
    comparison; a single-denominator normalization; an iid draw labelled
    correlated; FIXED_UNITS and EQUITY_FEEDBACK figures combined into one number.

---

## 8. Budget and what would have to be requested

Measured today on the coordinator: 16 cores, 30 GiB total with 21 GiB available,
load average 0.43, 114 GiB free on `/home`. This document consumed reading plus
three read-only characterization passes over committed CSVs.

Projections for a successor execution, from the retained measurement of
6.07 CPU s per native cell (1,461.693 s / 241 cells):

| Stage | Cells | Projected CPU s | Basis |
| --- | ---: | ---: | --- |
| B0 pilot | 5 | ~30 | measured rate |
| B1 four slices, correlated | 241 | ~1,465 | measured rate |
| B2 crossed core | 75 | ~455 | measured rate |
| B3 real arms, both modes | 8 | ~50 | measured rate |
| B4 iid regression | 241 | ~1,465 | measured rate |
| Design (a) unit trades | 147 ledger×variant×arm cells | **unmeasured** | the unit-trade replay is not the sequential backtest; no figure is asserted before B0 measures it |

Retained disk: ~1.1 GiB per 241-cell sweep as measured, so roughly 3 GiB for
B1+B4 plus the smaller stages.

The frozen bound from the prior run is **3,600 CPU s**. B1+B2+B3 fits it;
B1+B2+B3+B4 does not, and design (a) is unmeasured. Therefore the successor runs
**B0 first**, then admits only what its measured projection fits inside the
existing bound. **The cap is not shrunk to evade a refusal, and no extra
authority is invented**: if the admitted set does not cover B4 or design (a), the
command and the exact request are prepared and the owner decides.

---

Signed: **Satoshi, successor technical lead**, 2026-09-29.
Nothing in this document has been executed. No broker call, no sweep, no
service, no VM, no shared memory, no host limit touched.
