# Sealed-prediction adapter and cheap causal baseline predictors

**DEVELOPMENT · TRAIN 2020-2023 and VALIDATION 2024 · TEST closed · no GPU, no deep model.**

The question: how far do cheap, causal volatility-style predictors get toward the
roughly 0,7× naive requirement found in [PRICE_PATH_NOISE_SWEEP.md](PRICE_PATH_NOISE_SWEEP.md)?

## Adapter (`app/short_horizon/adapter.py`)

```bash
PYTHONPATH=. python -m app.short_horizon.adapter predictions.csv --data <lake>/market_data/forex/g10/eurusd/1h.parquet \
  --block-hours 24 --split VALIDATION_2024 --strategy RF-M --cost-mult 1 --out <dir>
```

**Input.** CSV or parquet, one row per block, keyed by `block_start` (t0) and
`made_at` (when the forecast was issued). It holds either
- a price path `p_h1 … p_hB`, or
- excursions `u_hat`, `d_hat` from the decision price, plus `p_low_first` or an
  `order` label.

**Sealing.** The file is sealed (sha256 in `SEAL.json`) before it is read, and
the seal is checked again after loading.

**Refused** (typed `PredictionError`):
- t0 outside the split;
- duplicated keys;
- `made_at` after t0;
- a key that is not an eligible block start;
- non-finite values;
- missing columns.

Blocks without a prediction are abstentions.

**Output.**
- MAE per horizon and × the persistence naive (path), and U/D MAE × NAIVE_MEAN20.
- Direction accuracy over blocks where a call is made, and log-loss.
- The realistic backtest (profile of `f093e6e`, ≤ 4 orders/day inside the
  strategy, stop-out at zero equity) and the frequency rule (≥ 3 trades/week, not ruined).

**Ruin.** A run whose marked-to-market equity reaches zero is flagged `ruined`,
has no Sharpe or Sortino, and is invalid. Realistic runs stop sending orders once
realized equity is at or below zero. The same flag is derived for older tables
from `max_drawdown_pct ≥ 100`. These fixes appear in this commit; break-evens
were unaffected.

## Predictors (`app/short_horizon/predictors.py`, `tools/short_horizon_baselines.py`)

**Causality.** Every prediction for block k uses only blocks completed before
k's start. Refits are expanding, at the first block of each week (weekly) or
business day (daily), and use only rows before that block.

| Input | Predictor |
|---|---|
| Excursions Û, D̂ | PERSIST (previous block) · MEAN20 · EWMA (half-life chosen on TRAIN only, grid 2–55 blocks: 13) · HAR (ridge-free least squares on past 1-day / 1-week / 1-month mean U and D, plus business-day and position-in-day dummies) |
| Order (which extreme first) | BASE (expanding base rate) · LOGIT (L2 logistic on lagged returns over 1 block / 1 day / 1 week, last order label, mean of last 3 labels, last U−D, calendar dummies) |
| Path | persistence (r for every h) · path **reconstructed** from (Û, D̂, order) |

**Reconstructed path shape:** r → first extreme at h₁ = round(B/3) → second
extreme at h₂ = round(2B/3) → flat to B, piecewise linear. Its max, min and order
equal (r + Û, r − D̂, order) exactly (tested), so it drives the strategy exactly
like the excursion forecast; it exists only so that a path MAE can be measured.

**Combinations:** 4 excursion × 2 order × 2 refit cadences, plus the persistence
path, for B = 24 and B = 6, both splits, RF-M and RF-L, costs ×1 and ×2: 272
scores. Fitting and scoring took 29,6 CPU-seconds. All 34 prediction files were
sealed before scoring (`results/predictors/predictores_sellos.json`).

## Results (VALIDACIÓN 2024, costs ×1; full table `results/predictors/PREDICTORES_TABLA.md`, CSV `predictores_resultados.csv`)

| B | Best of each kind | MAE U / D × MEAN20 | Order accuracy (log-loss vs base) | Path MAE × persistence (h_p / h_B / mean) | RF-M net · Sharpe · trades/wk | RF-L |
|---|---|---|---|---|---|---|
| 24 | HAR + LOGIT, daily | 0,98 / 0,91 | 0,560 (0,6923 vs 0,6934) | 2,32 / 1,36 / 1,60 | −548 · −0,52 · 4,9 | −1.689 · −1,66 · 2,0 (invalid) |
| 24 | EWMA + LOGIT, daily | 0,99 / 0,98 | 0,560 | 2,16 / 1,24 / 1,47 | −341 · −0,29 · 4,9 | −893 · −0,76 · 2,1 (invalid) |
| 24 | persistence path | 1,45 / 1,46 | no call | 1,00 / 1,00 / 1,00 | 0 trades (invalid) | 0 trades (invalid) |
| 6 | HAR + LOGIT, daily | 0,94 / 0,90 | 0,504 (0,6918 vs 0,6933) | 1,00 / 1,35 / 1,55 | −6.951 · −3,95 · 15,1 | −4.222 · −2,97 · 7,4 |

What the table shows:
- **Excursions:** the best cheap volatility predictor (HAR) cuts U/D error by
  only 2–10 % against MEAN20. The realistic requirement is roughly 30–45 %.
- **Order:** log-loss is 0,692 against a base rate of 0,693 — a coin. The
  logistic's 0,56 accuracy on 2024 is not mirrored on TRAIN (0,466).
- **Path:** as a path forecast, (Û, D̂, order) is 1,2–2,4× worse than persistence.
  Persistence is the strong level baseline at hourly scale.
- **Strategies:** every RF-M/RF-L configuration loses at ×1 on 2024. RF-L 24 h
  also fails the frequency rule.

Markers on the realism figures place each predictor (weekly refit plus the
persistence path) at its achieved path MAE: filled = valid, hollow = invalid.
They all sit right of the A3 break-even, at negative Sharpe:
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_sharpe_B24_validacion_RFM.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_net_B24_validacion_RFM.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_sharpe_B24_validacion_RFL.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_sharpe_B6_validacion_RFM.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/price_path_realism/barrido_trayectoria_realista_sharpe_B6_validacion_RFL.png

**Conclusion.** These strategies need a level/direction forecast that cheap
predictors cannot supply: at hourly EURUSD the price level is close to a
martingale, and persistence is the MSE-optimal level forecast. Volatility is
predictable but, used through RF-M/RF-L, it is not enough. This motivates the
direction-free designs (V1/V2) that follow.

## Limits

- **Construction:** fixed hyper-parameters (MEAN20 window, ridge-free least
  squares, logistic L2 = 1); only the EWMA half-life was chosen, on TRAIN.
- **TRAIN early weeks:** predictions on TRAIN start after a warm-up, so early
  weeks abstain.
- **Development data:** 2024 is DEVELOPMENT; one market path; costs ASSUMED_NOT_BROKER_VERIFIED.
- **Ruin handling:** stop-out at zero equity is a simplification of broker
  margin rules (no margin call before zero).
