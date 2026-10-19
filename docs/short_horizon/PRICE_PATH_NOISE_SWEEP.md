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

## Break-even (mean of 5 seeds; "all seeds" = first level where the worst seed reaches zero)

MAE in USD/EUR, with the multiple of the naive MAE at that horizon in brackets.
Read on net profit by linear interpolation between grid points. "—" means the
mean never turns negative on the grid; the strategy stops trading instead (see below).

| B | Split | Strategy | Costs | Mean break-even: MAE h_primary | MAE h_B | MAE mean | All seeds: MAE h_primary |
|---|---|---|---|---|---|---|---|
| 24 | VAL 2024 | RF-M | ×1 | **0,0156 (15,2× h6)** | 0,0153 (5,1× h24) | 0,0153 (10,0×) | 0,0119 (11,6×) |
| 24 | VAL 2024 | RF-M | ×2 | 0,0075 (7,3× h6) | 0,0073 (2,4× h24) | 0,0073 (4,8×) | 0,0063 (6,1×) |
| 24 | VAL 2024 | RF-L | ×1 | — (0 trades from MAE ≈ 0,012) | — | — | 0,0122 (11,9×) |
| 24 | VAL 2024 | RF-L | ×2 | — | — | — | 0,0040 (3,9×) |
| 24 | TRAIN | RF-M | ×1 | — (+4 USD, 70 trades at σ 0,06) | — | — | 0,0212 (15,4×) |
| 24 | TRAIN | RF-M | ×2 | 0,0135 (9,8× h6) | 0,0139 (3,2× h24) | 0,0137 (6,2×) | 0,0113 (8,2×) |
| 24 | TRAIN | RF-L | ×1 / ×2 | — (2 trades at σ 0,06) | — | — | 0,0106 / 0,0094 |
| 6 | VAL 2024 | RF-M | ×1 | **0,0058 (7,6× h3)** | 0,0057 (4,0× h6) | 0,0058 (6,3×) | 0,0051 (6,7×) |
| 6 | VAL 2024 | RF-M | ×2 | 0,0023 (3,0× h3) | 0,0022 (1,6× h6) | 0,0023 (2,5×) | 0,0019 (2,5×) |
| 6 | VAL 2024 | RF-L | ×1 | 0,0115 (15,0× h3) | 0,0113 (7,9× h6) | 0,0114 (12,5×) | 0,0069 (9,0×) |
| 6 | VAL 2024 | RF-L | ×2 | 0,0054 (7,1× h3) | 0,0053 (3,7× h6) | 0,0054 (5,9×) | 0,0037 (4,8×) |
| 6 | TRAIN | RF-M | ×1 | 0,0113 (10,4× h3) | 0,0112 (5,9× h6) | 0,0113 (8,9×) | 0,0105 (9,7×) |
| 6 | TRAIN | RF-M | ×2 | 0,0046 (4,2× h3) | 0,0045 (2,4× h6) | 0,0046 (3,6×) | 0,0040 (3,7×) |
| 6 | TRAIN | RF-L | ×1 | 0,0199 (18,3× h3) | 0,0197 (10,4× h6) | 0,0198 (15,6×) | 0,0149 (13,7×) |
| 6 | TRAIN | RF-L | ×2 | 0,0104 (9,5× h3) | 0,0103 (5,4× h6) | 0,0103 (8,1×) | 0,0079 (7,3×) |

The Sharpe zero crossing falls at the same place within 0,0004 (`break_even_trayectoria.json`).
At ×1, the ideal path gives Sharpe 12,0 (RF-M) / 13,5 (RF-L) and 20.292 / 26.975 USD
at B = 24 on 2024. At B = 6 it gives 15.003 / 17.679 USD.

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
