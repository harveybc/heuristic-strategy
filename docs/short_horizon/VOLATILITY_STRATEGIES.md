# Direction-free strategies on predicted volatility — V1 and V2

**DEVELOPMENT · VALIDATION 2024 primary, TRAIN 2020-2023 copy, multiplier
sensitivity on 2023 only · estudio oráculo/ruido, no es un resultado de modelo ·
TEST closed.**

**Question:** how accurate must the *volatility* predictor (excursions Û, D̂,
range R̂ = Û + D̂) be for a strategy that needs no direction?
Definitions are in STRATEGY_CONTRACT §16; the plan context is in
[INTEGRATED_STRATEGY_PLAN.md](INTEGRATED_STRATEGY_PLAN.md).

## What ran

- **Strategies:**
  - **V1:** OCO breakout with a broker-side trailing SL and fixed TP.
  - **V2:** range reversion at a causal anchor.
  - For the "regardless of direction" question: **RF-M and RF-L with a
    random (coin-flip) first touch**, so they carry no direction information.
- **Volatility-predictor error families**, applied to Û and D̂ per block:
  - i.i.d. Gaussian around the truth — **optimistic upper bound**;
  - **A1r** = naive MEAN20 + w·(truth − naive) + 0,5-pip ε; w = 0 *is* the naive;
  - **A3r** = A1r with an extra error that shrinks with skill: noise
    0,5·(1 − w)·MAE_naive, then two points beyond the naive.
  - x is the achieved MAE of Û, of D̂ and of R̂, each as a multiple of MEAN20,
    with the persistence multiple in brackets.
- **Execution:**
  - realistic profile, SL/TP hard rule (orders pre-filtered), owner sizing
    (rel_volume 0,10 × percentual_order_size 1,0 × current balance × leverage /
    price, compounding), daily cap 4, stop-out at ruin;
  - frequency rule: ≥ 3 trades/week (variant 4), ≥ 3 valid seeds.
- **Scenarios:** leverage 1:10, 1:50 and 1:100 at ×1 costs, and 1:100 at ×2.
- **Size and compute:** 10.560 runs (B = 24 and 6, both splits, 5 seeds), plus
  540 sensitivity runs on 2023. 711,8 backtest CPU-seconds in 14
  `crispdm-run -m 2G -t 120s` chunks.

## Break-even (VALIDATION 2024, 1:100, costs ×1; break-even on net, valid points only)

R̂ MAE × MEAN20 (× persistence). Full tables, including TRAIN, the 4/week
variant, the other leverages and ×2 costs, are in
`results/volatility_strategies/TABLAS_V1_V2.md`.

| B | Strategy | Ideal Sharpe (trades/wk) | A1r break-even | A3r break-even | A3r ×2 costs | Notes |
|---|---|---|---|---|---|---|
| 24 | V1 | invalid (1,6/wk) | — | — | — | below the frequency floor at every level |
| 24 | V2 | invalid (1,4/wk) | — | — | — | below the frequency floor |
| 24 | RF-M, coin | 12,78 (3,8) | **0,91 (0,77)** | 1,06 (0,90) | 0,83 (0,71) | Sharpe ≥ 1 needs ≤ ~0,7× |
| 24 | RF-L, coin | 14,36 (4,4) | — | — | — | invalid (< 3/wk) from 0,30–0,36× |
| 6 | V1 | −3,92 (4,4) | — | — | — | **loses even with an oracle range** (73 % SL exits) |
| 6 | V2 | 0,17 (5,5) | 0,05 (0,04) | 0,52 (0,40) | — (negative) | fragile; TRAIN: 1,60 and 0,83× |
| 6 | RF-M, coin | 14,56 (10,1) | **0,54 (0,41)** | 0,66 (0,51) | 0,52 (0,40) | TRAIN: 0,68× / 0,77× |
| 6 | RF-L, coin | 14,48 (12,2) | **0,61 (0,47)** | 0,68 (0,52) | 0,57 (0,44) | TRAIN: 0,70× / 0,77× |

**Leverage:**
- The 1:10 / 1:50 / 1:100 scenarios change Sharpe and break-evens by at most
  0,01–0,02. Exposure scales as 1×, 5× and 10× the balance (margin 10 %).
- No margin-call or ruin events at the reported points.
- **The leverage scenario does not change any conclusion; it changes the
  scale of profit and loss.**
- Figures: `apalancamiento_B24_*.png`, `apalancamiento_B6_*.png`.

## What this says

1. **V1, as specified (OCO stop entries), is refuted.**
   - It loses even with an exact range forecast: oracle Sharpe −3,9 on 2024 and
     −2,1 on TRAIN at 6 h, and it is invalid at 24 h.
   - Mechanism: with both legs resting, the leg on the side touched *first*
     fills. When the first extreme is reached before the other, price reverses
     into the SL (73 % SL exits). An OCO breakout is therefore an implicit
     first-touch bet — direction information again.
   - The 2023 sensitivity grid (k_tp 0,3–0,45, k_trail 0,25–0,5) is negative
     everywhere, so no multiplier rescues it.
2. **V2 is marginal.**
   - With an oracle range, Sharpe is 0,17 on 2024 (1,60 on TRAIN) at 6 h; it
     breaks even only at A3r ≤ 0,52× on 2024 (0,83× on TRAIN) and dies at ×2 costs.
   - At 24 h it is below the frequency floor.
   - The 2023 sensitivity grid (m 1,5–3, b 0,6–1,0) is ≤ 0.
3. **The "~0,6× range makes it profitable regardless of direction" hint is
   confirmed for 6 h blocks and refuted for RF-L at 24 h,** now without
   optimistic noise:
   - with a random direction, RF-M and RF-L at 6 h break even at about **0,54–0,61×
     MEAN20 on 2024** (0,68–0,70× on TRAIN) under A1r, and 0,66–0,68× under A3r;
   - at 24 h, RF-M needs ≤ 0,91× (A1r) but reaches Sharpe ≥ 1 only at ≤ 0,7×;
   - RF-L at 24 h falls below 3 trades/week before it breaks even.
   - The edge comes from placing TP and SL from the predicted extremes, not
     from direction.
4. **A3r is less strict than A1r near MAE ≈ naive.** Its noise hides a partial
   truth component behind an MAE close to the naive's. As before, equal MAE does
   not mean equal information. **Quote A1r** as the requirement, and A3r as a
   less strict reading.
5. **Where real predictors sit.** The cheap HAR predictor reaches 0,94–0,98× on Û
   and 0,90–0,91× on D̂ ([PREDICTORS_BASELINES.md](PREDICTORS_BASELINES.md)).
   That is short of the roughly 0,55–0,7× needed by the only direction-free
   engine that works here (random-direction RF at 6 h).

## Figures (PNG + PDF; three panels: MAE of Û, of D̂ and of R̂ with × MEAN20; Naïve MEAN20 and persistence lines)

- B = 6, VALIDACIÓN 2024:
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_sharpe_B6_validacion_V2.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_net_B6_validacion_V2.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_maxdd_B6_validacion_V2.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_sharpe_B6_validacion_V1.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_net_B6_validacion_V1.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_maxdd_B6_validacion_V1.png
- B = 24, VALIDACIÓN 2024 (all levels invalid: the figures say so):
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_sharpe_B24_validacion_V1.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/barrido_volatilidad_sharpe_B24_validacion_V2.png
- Leverage scenarios:
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/apalancamiento_B6_validacion.png
  - https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/volatility_strategies/apalancamiento_B24_validacion.png
- TRAIN copies use `entrenamiento`; the full list is in `results/volatility_strategies/MANIFEST.json`.

## Limits

- **Constructions, not models:** the error families are constructions; real
  errors must be scored through the sealed adapter.
- **Pilot-driven change:** V1 parameters are a priori, but its geometry was
  corrected once after a pilot showed the TP beyond the predicted extreme by
  construction (contract §16). V2 is unchanged.
- **Compounding:** compounding at 10× notional makes net profit explode for
  oracle-like cells, so figures plot the capital multiple on a log scale and
  Sharpe is primary.
- **Assumptions:** costs, swap and the broker minimum are assumed.
- **Clock:** the vendor bar clock is still under repair.
- **Data:** one market path; 2024 is DEVELOPMENT.
