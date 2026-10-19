# Trend + volatility → TP/SL: does a causal trend forecast add anything once volatility sets the levels?

**DEVELOPMENT.** Owner question 2026-10-19, answered as an explicit ablation that separates the two
channels. Earlier rounds tested it only implicitly (per-side excursions Û/D̂).

Tables: `results/trend_vol/TABLAS_TENDENCIA_VOL.md`. Figures:
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/trend_vol/tendencia_vol_B24.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/trend_vol/tendencia_vol_B6.png

## 1. The component: VolTrendStrategy (`app/short_horizon/voltrend.py`)

One simulable, parametrized component. Every weighting is a declared, bounded parameter in a JSON
schema (`results/trend_vol/VOLTREND_SCHEMA.json`), with defaults and a manifest hash
(`DEFAULTS_MANIFEST.json`); there is no hidden constant.

| Parameter | Default (a priori) | Bounds | Role |
|---|---|---|---|
| `m` | 2 | [0,5; 4] | SL = entry − m·σ̂ |
| `c_tp` | 1 | [0; 3] | vol multiple in the TP |
| `w_trend` | 1 | [0; 2] | weight of \|trend̂\| in the TP |
| `theta` | 0 | [0; 50] | abstain when \|trend̂\| < θ·σ̂ |
| `tp_form` | additive | {additive, clipped} | TP = entry + w·\|trend̂\| + c_tp·σ̂, or min(max(w·\|trend̂\|, min dist), c_tp·σ̂) |
| `pos_a, pos_b, pos_c, pos_min` | 1, 0, 0, 0,25 | bounded | percentual_order_size = clip(a + b·\|trend̂\|/σ̂ − c·σ̂/σ̄, pos_min, 1) |
| `rel_volume`, `leverage` | 0,10, 100 | (0; 0,2], {10, 50, 100} | owner sizing |
| `block_hours`, `time_stop_hours` | 24, 24 | {6, 24} | block = forced flat |
| `trend_model`, `vol_model` | t0, HAR | ids | model choice |
| `max_orders_per_day`, `min_trades_per_week` | 4, 3 | | frequency rule |

Direction = sign(trend̂) (the control t0 uses the fixed coin). The objective adapter
`VolTrendStrategy.evaluate(params, period, seed) → {sharpe, max_dd, net, orders_per_week, valid, …}`
is deterministic, validates the schema, enforces the SL/TP/units rules, the daily cap, the frequency
rule and the ruin stop-out inside the simulation, and penalizes an invalid configuration (Sharpe −10,
DD 100) instead of dropping it. Tests: `tests/unit_tests/test_short_horizon_voltrend.py`.

## 2. Models (all causal)

- **Trend** (target = signed block drift, close at block end − reference): t0 zero drift (control,
  coin direction); t1 AR(3) on hourly returns + HAR volatility terms as exogenous, weekly expanding
  ridge; t2 ridge on 6 h / 1 d / 1 w / 1 m returns + calendar dummies, weekly; t3 DLinear
  (one-sided MA(25) trend / seasonal decomposition of the 168-h return window, linear heads, no
  centred filter), fit on 2020–2022 with the ridge alpha chosen on 2023 (alpha 100 for both B), no
  refit; t4 oracle (realized drift) as the ceiling.
- **Volatility** σ̂ = half the forecast block range: MEAN20 naive, HAR (weekly), v-oracle (realized).

## 3. Ablation at the defaults (1:100, costs ×1; every cell valid by frequency)

| B | Trend | Dir. acc. 2023 / 2024 | Drift MAE vs zero-drift naive (pips) | Sharpe with σ̂ MEAN20 (2023 / 2024) | with σ̂ HAR | with σ̂ oracle |
|---|---|---|---|---|---|---|
| 24 h | t0 coin | 0,455 / 0,498 | 35,5 / 24,8 (= naive) | −1,66 / −2,30 | −1,68 / −1,65 | −0,22 / −0,84 |
| 24 h | t1 AR+HAR | 0,483 / 0,494 | 36,1 / 25,1 | −0,41 / −1,26 | **+0,07** / −1,21 | −0,15 / +0,84 |
| 24 h | t2 ridge | 0,508 / 0,490 | 35,8 / 24,9 | −0,60 / −1,27 | −0,73 / −1,29 | −0,36 / +0,54 |
| 24 h | t3 DLinear | 0,525 / 0,494 | 40,2 / 30,0 (worse than zero) | −0,56 / −2,27 | −0,43 / −2,24 | −1,02 / −0,38 |
| 24 h | t4 oracle | 0,996 / 1,000 | 0 | 12,45 / 10,91 | 12,33 / 10,49 | 12,44 / 11,23 |
| 6 h | t0 coin | 0,522 / 0,506 | 17,7 / 14,3 | −3,76 / −4,53 | −3,53 / −3,25 | −1,43 / −1,63 |
| 6 h | t1 | 0,506 / 0,522 | 17,8 / 14,3 | −3,61 / −4,50 | −3,42 / −4,64 | −1,32 / −2,56 |
| 6 h | t2 | 0,515 / 0,540 | 17,8 / 14,3 | −3,25 / −3,32 | −2,46 / −3,69 | −0,23 / −1,58 |
| 6 h | t3 DLinear | 0,529 / 0,477 | 18,9 / 15,2 | −1,75 / −4,88 | −2,35 / −4,80 | −2,07 / −2,80 |
| 6 h | t4 oracle | 0,994 / 0,996 | 0 | 15,65 / 15,33 | 15,58 / 14,90 | 16,55 / 15,90 |

σ̂ MAE × MEAN20: HAR 0,855 (2023) / 0,833 (2024) at 24 h, 0,783 / 0,803 at 6 h. Exit mix for the
causal cells: ≈ 33 % TP / 10 % SL / 55 % time at 24 h. c_tp ∈ {0,5; 1; 1,5; 2} and the clipped TP
form move the causal cells by ≤ 1 Sharpe without changing their sign pattern (tables). Leverage
changes only the net scale.

**Degradation curves** (one channel degraded, the other ideal): the trend-magnitude break-even is at
drift MAE ≈ 35 pips (2023) / 25 pips (2024) at 24 h and 17,6 / 14,2 at 6 h — i.e. exactly the
zero-drift naive, because the curve keeps the oracle sign; the direction curve with the oracle
magnitude in the TP reaches zero only at ≈ 0,50 (24 h) and 0,53–0,54 (6 h), which is **optimistic**
(a TP placed at the realized \|drift\| is itself information); the vol channel alone (t4 × MEAN20 →
oracle) moves Sharpe by < 1.

## 4. Optimized row: NSGA-II on 2020–2022 (DEAP, pop 24, 25 generations, bounded CPU)

Genes = every schema parameter plus the model ids (t0–t3 and MEAN20/HAR; the oracles are excluded).
Objectives: maximize Sharpe, minimize max drawdown %. Fronts: `front_B24.csv`, `front_B6.csv`.

| B | Front | Best decision Sharpe (2020–2022) | Chosen point (best 2023 Sharpe) | 2023 | 2024, once |
|---|---|---|---|---|---|
| 24 h | 4 points, all t1 (3 × HAR, 1 × MEAN20), clipped TP, c_tp 0,26–0,44, m 1,16–1,22, θ ≈ 0,07 | −0,68 | t1 × HAR | −1,60 (invalid: 2,4–3,2 orders/wk) | −2,08, invalid |
| 6 h | 8 points, t1/t2 × MEAN20, m ≈ 2,7, θ ≈ 0,14 | +0,53 (valid) | t1 × MEAN20 clipped | −1,26 (invalid) | −1,03, invalid |

The optimizer finds nothing that survives 2023: the only positive decision-period Sharpe (+0,53 at
6 h) is in-sample and reverses on 2023 and 2024, and the abstention threshold it learns pushes the
frequency below the rule.

## 5. Verdict

**No causal trend model adds anything over zero drift once volatility sets the levels, and the
volatility channel alone cannot cross zero.** The expectation is confirmed honestly:
- direction accuracy of t1–t3 is 0,48–0,54 against a coin at 0,45–0,52 on the same rows — within
  sampling noise of the coin; their drift MAE is at or above the zero-drift naive (DLinear is worse);
- with the coin (t0) the vol channel yields Sharpe −0,2 … −2,3 even with the vol oracle; HAR vs
  MEAN20 changes the result by < 0,5 Sharpe;
- the only positive cells (t1 × HAR 2023 +0,07; t1/t2 × v-oracle 2024 +0,84 / +0,54) are not
  reproduced across periods and the latter need the realized volatility;
- the oracle trend row (Sharpe 11–16) shows the prize is entirely in the direction/magnitude of the
  drift, which none of the causal models captures at 1 h resolution.

## Limits
- One instrument and one market path per year; 2024 is DEVELOPMENT (looked at once here).
- Small models (ridge, AR(3), DLinear with closed-form ridge heads) — larger nets were out of scope
  for this bounded run; the direction curve's magnitude is optimistic by construction.
- Costs ASSUMED_NOT_BROKER_VERIFIED; vendor clock under repair.
- Compute: prep 5 s, ablation 31 s, NSGA 28 + 61 s CPU under `crispdm-run -m 2G -t 120s`.
