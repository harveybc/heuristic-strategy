# Range-first driven only by predicted prices: Sharpe and profit versus price MAE

**DEVELOPMENT · estudio oráculo/ruido, no es un resultado de modelo.** Owner
redesign of 2026-10-19. The strategy now consumes **only** a predicted hourly
price path, with no separate classifier and no separate high/low forecast. The
x-axes are the error of that one input, as in the doctoral-proposal figures.
VALIDATION 2024 is primary; TRAIN 2020-2023 is the secondary copy. TEST
(2025+) stays closed: the loader refuses it at read time.

## Design

- **Input:** for each block of B hours (B = 24, business day aligned to the
  rollover; also B = 6), the strategy receives predicted closes for h = 1..B,
  i.e. one direct multi-output price path. The truth for h is the close of the
  bar labelled t0 + h − 1 h. An absent bar stays absent; nothing is filled.
- **Derivation (only from the predicted path):**
  - predicted high = max of the path, predicted low = min;
  - which comes first = order of the argmax and argmin (first occurrence);
  - a flat or empty path means no trade.
  - Code: `app/short_horizon/strategies.path_to_range`.
- **Execution:** RF-M (market entry + bracket) and RF-L (limit at the predicted
  first extreme + bracket), with the same parameters, costs (×1 and ×2, ASSUMED_NOT_BROKER_VERIFIED),
  calendar and intrabar rule as `e922be5`. Intrabar rule: open first, stop
  before TP when both are touched, no TP on a limit-fill bar.
- **Noise:** i.i.d. Gaussian σ (USD/EUR) on every predicted price of the path,
  with its own seeded stream per seed (42–46). σ ∈ {0, 0,0001, 0,0002, 0,0003,
  0,0005, 0,00075, 0,001, 0,0015, 0,002, 0,003, 0,004, 0,005, 0,0075, 0,01,
  0,015, 0,02, 0,03, 0,04, 0,06}. The grid was extended once, from 0,02 to
  0,06, so that RF-M clearly reaches Sharpe ≤ 0.
- **Achieved error:** MAE over every predicted price of the path, per horizon
  and as a mean, on the same rows as the backtest. The **naive is persistence**
  (price at t for every h) on the same rows.
  - B = 24, VALIDATION 2024 naive MAE: h6 0,00103, h12 (= B/2) 0,00164,
    h24 (= B) 0,00302, mean over h = 1..24 0,00153.
  - B = 6, VALIDATION 2024 naive MAE: h3 (= B/2) 0,00076, h6 (= B) 0,00143, mean 0,00091.
- **Persistence strategy:** feeding the persistence path into the strategy
  gives a flat path, hence no trades and 0 USD. It is drawn as the dotted
  reference line.
- **Compute:** 1,528 cell-split rows (both splits per cell), 67.5 backtest
  CPU-seconds in four `crispdm-run -q -m 2G -t 120s` chunks, plus bounded
  report runs.

## Figures (PNG; a PDF with the same name sits next to each)

Primary, business day (B = 24), VALIDACIÓN 2024:
- Sharpe: https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_sharpe_B24_validacion.png
- Beneficio neto: https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_net_B24_validacion.png
- Operaciones por corrida: https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_trades_B24_validacion.png

B = 6, VALIDACIÓN 2024:
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_sharpe_B6_validacion.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_net_B6_validacion.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_trades_B6_validacion.png

Secondary copies, ENTRENAMIENTO 2020-2023:
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_sharpe_B24_entrenamiento.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_net_B24_entrenamiento.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_trades_B24_entrenamiento.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_sharpe_B6_entrenamiento.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_net_B6_entrenamiento.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path/barrido_trayectoria_trades_B6_entrenamiento.png

**Layout.** Every figure has three stacked x-axes, one per error the strategy
consumes:
- B = 24: MAE at h = 6 (primary), MAE at h = 24, and mean MAE over h = 1..24.
- B = 6: MAE at h = 3, MAE at h = 6, and the mean.
- Each tick shows the value and, in parentheses, the multiple of the naive
  MAE at that same horizon.
- Vertical dashed Naïve lines mark the equal-MAE point at each horizon and are
  labelled with their MAE.

**Style elements.** As in the reference: mean of 5 seeds as a line (RF-M solid,
RF-L dashed), ±1 SD band plus thin per-seed lines, the ideal value annotated,
the first observed sign change shaded and marked, and a text box with period,
origins, capital, sizing, trades per run, intrabar rule and costs. The x-axes
use a square-root scale so that the region around the naive stays readable.

**Multiple axes are valid here.** I.i.d. noise moves every horizon's MAE
together. Measured: every axis increases strictly with σ (largest decrease
along σ = 0), and the largest relative deviation between horizons is 2.8 %
(B = 24 TRAIN, h24 vs h6; 2.3 % on VALIDATION). See
`results/price_path/verificacion_ejes_monotonos.json`. No fallback to a
single axis was needed. No other forecast or tool is consumed, so no other error axis exists.

## Break-even with the frequency rule applied (owner rule, applied retroactively)

**Una configuración que no cumple la frecuencia mínima de operación no se reporta como resultado.**

The rule:
- A seed is valid with at least 3 trades per business week, counted over weeks
  with at least one eligible block. A stricter 4-per-week variant is reported in
  the tables.
- At most 4 orders per business day (variant: 2). In this first price-path run,
  B = 24 has at most 1 per day and B = 6 at most 4 by construction of the block grid.
- A noise level needs at least 3 of 5 valid seeds.

How it is applied:
- Means, the ±1 SD band, the first sign change and every break-even use valid
  seeds and valid levels only. Reading stops at the first invalid level (the
  "frontier").
- Invalid points are drawn hollow and grey.
- This removes RF-L's abstention tail entirely: RF-L becomes invalid **before**
  it ever changes sign.

The figures above were regenerated with the rule (same filenames).

MAE in USD/EUR (× naive at that horizon); i.i.d. noise, earlier cost model. These
are upper bounds (see "Por qué el Sharpe en MAE = naive era alto" below).

| B | Split | Strategy | Costs | Frontier (σ) | Mean break-even: MAE h_primary | MAE h_B | MAE mean | All seeds: MAE h_primary |
|---|---|---|---|---|---|---|---|---|
| 24 | VAL 2024 | RF-M | ×1 | 0,04 | 0,0156 (15,22× h6) | 0,0153 (5,05× h24) | 0,0153 (9,98×) | 0,0119 (11,63×) |
| 24 | VAL 2024 | RF-M | ×2 | 0,04 | 0,0075 (7,27×) | 0,0073 (2,41×) | 0,0073 (4,77×) | 0,0063 (6,09×) |
| 24 | VAL 2024 | RF-L | ×1 / ×2 | 0,001 | invalid before any sign change | — | — | — |
| 24 | TRAIN | RF-M | ×1 / ×2 | 0,04 | — / 0,0135 (9,80×) | — / 0,0139 (3,22×) | — / 0,0136 (6,23×) | 0,0212 / 0,0113 |
| 24 | TRAIN | RF-L | ×1 / ×2 | 0,0015 | invalid before any sign change | — | — | — |
| 6 | VAL 2024 | RF-M | ×1 | always valid | 0,0058 (7,63× h3) | 0,0057 (4,00× h6) | 0,0058 (6,33×) | 0,0051 (6,69×) |
| 6 | VAL 2024 | RF-M | ×2 | always valid | 0,0023 (2,97×) | 0,0022 (1,56×) | 0,0022 (2,46×) | 0,0019 (2,51×) |
| 6 | VAL 2024 | RF-L | ×1 / ×2 | 0,003 | invalid before any sign change | — | — | — |
| 6 | TRAIN | RF-M | ×1 / ×2 | always valid | 0,0113 (10,40×) / 0,0046 (4,22×) | 0,0112 (5,91×) / 0,0045 (2,40×) | 0,0112 (8,86×) / 0,0046 (3,60×) | 0,0105 / 0,0040 |
| 6 | TRAIN | RF-L | ×1 / ×2 | 0,004 | invalid before any sign change | — | — | — |

The per-level tables now include "órdenes/semana (media, min seed)" and
"máx órdenes/día", and print "inválida" where a level fails the rule.

## Per-level table (excerpt; full tables in `results/price_path/TABLAS_POR_NIVEL.md` and `tabla_por_nivel_de_ruido.csv`)

B = 24, VALIDACIÓN 2024, means of 5 seeds:

| σ | MAE h6 (×naïve) | MAE h24 (×naïve) | MAE medio (×naïve) | RF-M ×1 Sharpe / neto / oper. | RF-L ×1 Sharpe / neto / oper. | RF-M ×2 neto | RF-L ×2 neto |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 12,04 / 20.292 / 221 | 13,50 / 26.975 / 253 | 16.337 | 22.210 |
| 0,0005 | 0,00041 (0,40) | 0,00040 (0,13) | 0,00040 (0,26) | 13,23 / 21.340 / 228 | 13,92 / 28.197 / 225 | 17.357 | 23.573 |
| 0,001 | 0,00081 (0,79) | 0,00080 (0,26) | 0,00080 (0,52) | 12,39 / 17.055 / 241 | 11,07 / 19.225 / 141 | 13.819 | 16.300 |
| 0,002 | 0,00163 (1,59) | 0,00159 (0,53) | 0,00159 (1,04) | 9,23 / 8.875 / 252 | 6,08 / 6.119 / 49 | 6.608 | 5.212 |
| 0,003 | 0,00244 (2,38) | 0,00239 (0,79) | 0,00239 (1,56) | 7,24 / 5.217 / 256 | 3,74 / 2.242 / 20 | 3.473 | 1.933 |
| 0,005 | 0,00407 (3,96) | 0,00398 (1,32) | 0,00398 (2,60) | 4,54 / 2.006 / 257 | 1,70 / 439 / 5 | 992 | 389 |
| 0,01 | 0,00814 (7,93) | 0,00795 (2,63) | 0,00796 (5,20) | 1,90 / 390 / 258 | 1,34 / 149 / 2 | −106 | 138 |
| 0,02 | 0,01628 (15,85) | 0,01591 (5,26) | 0,01592 (10,40) | −0,15 / −18 / 258 | 0,45 / 20 / 0 | −227 | 18 |
| 0,03 | 0,02443 (23,78) | 0,02386 (7,89) | 0,02387 (15,60) | −0,70 / −36 / 212 | 0,20 / 3 / 0 | −163 | 2 |

The full tables also give MAE at h = 1, 2, 3 and 12, max drawdown, and the TRAIN and B = 6 cases.

**Frequency rule:** the RF-L rows from σ 0,001 (B = 24) are **invalid**: fewer than
3 trades per week on at least 3 seeds. They are kept above only as the record of
the first run. The regenerated `TABLAS_POR_NIVEL.md` prints them as "inválida".

## Honest reading — what these curves can and cannot say

1. **Additive Gaussian noise around the truth is optimistic compared with a real
   model.** The realised price stays inside the error distribution of every
   prediction. That is why the strategy still earns money at MAE many times the
   naive's (2,6× naive h24 at σ 0,01 on 2024). A real forecaster's errors are
   neither centred on each realised price nor independent across horizons.
   These break-evens are **upper bounds on tolerable error, not targets.** The
   model-like blend frontier in [REQUIRED_ACCURACY_FRONTIER.md](REQUIRED_ACCURACY_FRONTIER.md)
   is stricter.
2. **The max/min of a noisy path is a biased estimate of the extreme.**
   - Noise pushes the path maximum up and the minimum down.
   - At small σ this *helps*: closes underestimate the true high/low, and a
     little outward bias moves the targets toward them. Sharpe and profit
     therefore first **rise** (e.g. B = 24 RF-M 12,0 → 13,2 at σ 0,0005).
     This is not a reason to prefer noisy forecasts.
   - At larger σ, RF-L's limits sit beyond the true extremes and stop filling:
     253 trades ideal, 49 at σ 0,002, 0–2 from σ 0,01 (see the trades figures).
   - RF-L therefore decays toward 0 instead of going negative. That is
     abstention, not robustness.
3. **Risk-based sizing.** With very wide predicted ranges, the 1 %-risk sizing
   rounds RF-M positions down to zero (24 h TRAIN: 1 016 trades at σ 0,015,
   70 at 0,06). Profit shrinks toward 0 for that mechanical reason, which is why
   RF-M ×1 TRAIN B = 24 never turns negative on the grid.
4. **No hidden second input.** First touch is derived from the same path, so its
   error is entirely inside the price MAE shown. No separate error axis is needed.
5. **Limits.**
   - One market path per split; seeds vary only the noise.
   - 2024 is DEVELOPMENT, not a holdout.
   - Costs are assumed; there is no rollover spread widening.
   - Hourly OHLC conventions are used for intrabar events.
   - The vendor bar clock is still under repair.
   - Partial business-day blocks (12 h at the start and end of a session) have
     shorter paths; MAE counts only existing bars.

## Por qué el Sharpe en MAE = naive era alto (realism pass, owner 2026-10-19)

**Owner question:** isn't a Sharpe of about 11 at MAE = naive suspiciously high?

**Answer:** yes. The cause is **the noise model, not the costs.**
- i.i.d. Gaussian noise centred on each true price keeps the truth inside every
  prediction. At MAE = naive it still carries far more information than the
  naive, which produces no trades at all.
- With model-like errors, the same strategy breaks even at about **0,6–0,7× the
  naive MAE** instead of 5–15×.
- The realistic execution assumptions barely change the ideal Sharpe: they are
  *cheaper* per round trip than the earlier flat 4,7 pips, but add latency,
  trade-through and rollover spreads.
- Still DEVELOPMENT, VALIDATION 2024 primary, TRAIN copy, 5 seeds, ×1 and ×2,
  TEST closed. This is not a model result.

### Model-like error families (all on the predicted price path h = 1..B)

| Family | Prediction for horizon h | Grid |
|---|---|---|
| i.i.d. (cota superior optimista) | P(t+h) + σ·z_h | σ = 0 … 0,06 |
| A1 mezcla hacia persistencia | P(t) + w·(P(t+h) − P(t)) + ε_h, ε ~ N(0; 0,5 pip) | w = 1 … 0; **w = 0 is the naive** |
| A2 paseo aleatorio | P(t+h) + s·Σ_(j≤h) z_j (error grows like √h, autocorrelated along the path) | s = 0 … 0,005 |
| A3 = A1 + A2 (realista, para citar) | P(t) + w·(P(t+h) − P(t)) + s·Σ z_j, with s = 0,0002·(1 − w) | w = 1 … 0, then w = 0 with s = 0,0004, 0,0008, 0,0016 |

### Realistic execution (applied to every curve; all ASSUMED_NOT_BROKER_VERIFIED)

| Item | Assumption |
|---|---|
| Spread by hour since the daily rollover (rollover = segment open + 24 k on the configured calendar) | 0–1 h: 3,0 pips · 1–3 h: 1,0 · 3–9 h: 0,6 · **9–21 h (core): 0,3** · 21–23 h: 0,6 · 23–24 h: 1,5; first hour after the weekly open: 3,0 |
| Charged | half spread on every fill, at that fill's bar |
| Slippage | 0,2 pip on market fills (entries and time exits), 0,5 pip on stops, none on limit fills |
| Commission | 3,5 USD/lot/side |
| Limit orders (entry and TP) | fill intrabar only if price trades **through** by ≥ 1 pip; an open beyond the limit fills at the open |
| Latency | one bar: decision on the bar closed at t0, market entry at the next bar's open, and limits active from that bar |
| Swap | long −7,0 USD/lot/night (paid), short +2,5 (received); **×3 on the third rollover after the weekly open**. This is the Wednesday convention, configured as an index; no day name is read. |
| Intrabar and gaps | stop-first kept; forced flat 12 h before close (no exposure across closures) |
| Orders per day | strategy-side cap ≤ 4 per business day (variant ≤ 2), enforced at decision time |

The "×1 previous costs" reference curve (2 pip spread, 1 pip slippage per fill,
7 USD/lot round trip, 10 USD/lot/rollover, no latency, touch fills) is drawn in
every figure.

### Figures (PNG + PDF; three panels = MAE at h_p, at h_B and the mean)

In these figures the stacked shared axes of the earlier i.i.d. figures are
replaced by three panels. The reason: under A1/A2/A3 the mapping from one
horizon's MAE to another differs by family (A2 grows like √h), so one shared
secondary axis would be wrong. The x range covers A1–A3; the i.i.d. tails
continue in the i.i.d. figures above.

B = 24, VALIDACIÓN 2024:
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_sharpe_B24_validacion_RFM.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_net_B24_validacion_RFM.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_trades_B24_validacion_RFM.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_sharpe_B24_validacion_RFL.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_net_B24_validacion_RFL.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_trades_B24_validacion_RFL.png

B = 6, VALIDACIÓN 2024 (same names with `B6`); TRAIN copies with `entrenamiento`
in place of `validacion`. 24 figures in total, listed in
`results/price_path_realism/MANIFEST_REALISMO.json`.

### Comparison (VALIDACIÓN 2024; break-even on net profit, valid points only, ≥ 3 trades/week)

Regenerated on 2026-10-19 after the business rules (SL/TP effective on every order, orders with symbolic levels not sent; stop-out at ruin) were applied to this grid. Changes against the first version: at 24 h the realistic break-evens moved by at most 0,02 (RF-M A3 0,71 → 0,69×; RF-L 0,73 → 0,72×; A1 0,61 → 0,60×) and the ideal Sharpe 13,07 → 12,93 (RF-M); at 6 h by up to 0,05 (RF-M A3 0,63 → 0,58×; RF-L A3 0,56 → 0,52×), stricter because more far-away levels are skipped. The i.i.d. curves now become invalid before their earlier 15× break-even: their far-away take-profits are symbolic and are no longer sent. Sizing here is still the earlier 1 %-risk reference arm; the owner sizing is used in the V1/V2 study.

| B | Strategy | Case | Ideal Sharpe | Break-even ×naïve h_p | ×naïve h_B | ×naïve mean | All seeds ×naïve h_p | Validity frontier ×naïve h_p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 24 | RF-M | (i) previous costs / i.i.d. | 11,91 | — | — | — | — | 2,38 |
| 24 | RF-M | (ii) realistic / i.i.d. | 12,93 | — | — | — | — | 2,38 |
| 24 | RF-M | **(iii) realistic / A3** | 12,93 | 0,69 | 0,66 | 0,68 | 0,64 | 3,26 |
| 24 | RF-M | realistic / A1 | 12,90 | 0,60 | 0,60 | 0,60 | 0,59 | 0,90 |
| 24 | RF-M | (iii) realistic ×2 / A3 | 12,39 | 0,64 | 0,61 | 0,63 | 0,61 | 0,87 |
| 24 | RF-L | (i) previous costs / i.i.d. | 13,06 | — | — | — | — | 0,79 |
| 24 | RF-L | (ii) realistic / i.i.d. | 13,92 | — | — | — | — | 0,59 |
| 24 | RF-L | **(iii) realistic / A3** | 13,92 | 0,72 | 0,69 | 0,70 | 0,69 | 3,26 |
| 24 | RF-L | realistic / A1 | 13,90 | 0,60 | 0,59 | 0,60 | 0,59 | 1,00 |
| 24 | RF-L | (iii) realistic ×2 / A3 | 13,60 | 0,63 | 0,60 | 0,61 | 0,61 | 3,26 |
| 6 | RF-M | (i) previous costs / i.i.d. | 7,10 | — | — | — | — | 3,18 |
| 6 | RF-M | (ii) realistic / i.i.d. | 10,58 | — | — | — | — | 3,18 |
| 6 | RF-M | **(iii) realistic / A3** | 10,58 | 0,58 | 0,57 | 0,57 | 0,55 | — |
| 6 | RF-M | realistic / A1 | 10,95 | 0,53 | 0,53 | 0,53 | 0,52 | 0,80 |
| 6 | RF-M | (iii) realistic ×2 / A3 | 9,74 | 0,53 | 0,51 | 0,52 | 0,51 | 0,86 |
| 6 | RF-M | realistic / A3, ≤ 2 orders/day | 10,75 | 0,58 | 0,57 | 0,58 | 0,55 | — |
| 6 | RF-L | (i) previous costs / i.i.d. | 7,35 | — | — | — | — | 2,12 |
| 6 | RF-L | (ii) realistic / i.i.d. | 11,69 | — | — | — | — | 2,12 |
| 6 | RF-L | **(iii) realistic / A3** | 11,69 | 0,52 | 0,51 | 0,52 | 0,50 | — |
| 6 | RF-L | realistic / A1 | 12,14 | 0,48 | 0,48 | 0,48 | 0,48 | 0,90 |
| 6 | RF-L | (iii) realistic ×2 / A3 | 10,70 | 0,46 | 0,44 | 0,45 | 0,43 | 0,97 |
| 6 | RF-L | realistic / A3, ≤ 2 orders/day | 11,99 | 0,53 | 0,52 | 0,52 | 0,50 | — |

How to read the table:
- A2 alone (a random-walk error centred on the truth) never crosses zero while
  valid. Like i.i.d., it keeps the truth inside the forecast; only the blend
  toward persistence removes the information.
- With the stricter **4 trades/week** rule, RF-M B = 24 under A3 becomes invalid
  at 0,65× (×1) and 0,44× (×2), before it would cross zero. So no break-even is
  reported there.
- RF-L B = 24 keeps 0,72×.
- The full table, with both frequency variants and TRAIN, is
  `results/price_path_realism/TABLA_COMPARATIVA_REALISMO.md`; per-level values
  are in `tabla_realismo_por_nivel.csv`.

**Where the pure naive sits (A1, w = 0: persistence + 0,5-pip ε).** Achieved MAE
is 1,00× naive.
- RF-M: 0 trades at ×1 and ×2, for both B.
- RF-L B = 24: 0,2 trades in the year (−2,50 USD).
- It is invalid under the frequency rule, as expected: a forecast with the
  naive's information cannot even meet the trading frequency.

**Requirement to quote (realistic costs, A3, ≥ 3 trades/week, 2024).** The
price-path forecaster must reach about **0,7× the naive MAE** at business-day
horizons (B = 24: 0,69× at h6, 0,66× at h24, after the SL/TP rule) and about
**0,52–0,58×** for 6-hour blocks. That means at least a 30–48 % error reduction
over persistence. Even
then, the curves are oracle constructions: a real model's errors have their own
structure and must be backtested directly.

### Compute

- Grid `pathr`: 3,186 cells, both splits each (6,372 rows), 443,0 CPU-seconds
  in 14 chunks of `crispdm-run -q -m 2G -t 120s`.
- Reports ran in a few further bounded calls.
- The earlier i.i.d. figures were re-rendered with the frequency rule from their
  retained cells; their week counts come from this run.
