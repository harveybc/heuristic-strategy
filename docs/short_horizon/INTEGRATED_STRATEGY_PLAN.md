# Integrated strategy plan — one strategy, every area that helps (PLAN ONLY)

Status: plan, DEVELOPMENT. No deep model is trained by this document. TEST (2025+)
stays closed. Reinforcement learning is out of scope here (another lane).
Owner instructions of 2026-10-19 (pivot, integrated pipeline, structural
novelty, business rules) are folded in.

## 0. Why the pivot — what the evidence says

- **Direction is close to unpredictable.** At hourly EURUSD the price *level*
  is close to a martingale. Persistence is the MSE-optimal level forecast, and
  every cheap predictor of the level path was 1,2–2,4× *worse* than persistence
  ([PREDICTORS_BASELINES.md](PREDICTORS_BASELINES.md)).
- **Direction classifiers score like a coin.** The logistic first-touch
  classifier's log-loss matched the base rate (0,692 vs 0,693).
- **The old strategy needs what cannot be had.** Range-first with a price-path
  input needs a path MAE of about 0,5–0,7× persistence
  ([PRICE_PATH_NOISE_SWEEP.md](PRICE_PATH_NOISE_SWEEP.md)), out of reach of
  volatility-style predictors.
- **Volatility and multi-scale trend strength are persistent and predictable.**
  HAR-style range forecasts already beat MEAN20 by 2–10 % without any learning.
- **Consequence.** The next strategies must **not need direction**. The study of
  the two direction-free engines (V1/V2) is in
  [VOLATILITY_STRATEGIES.md](VOLATILITY_STRATEGIES.md).

**What the V1/V2 study changed in this plan (2024 numbers are DEVELOPMENT):**
- **V1 OCO breakout loses even with an oracle range.** Its stop entries are an
  implicit first-touch bet. It is therefore not used as the expansion engine.
  The expansion mode now needs either a direction signal (module 5, only if it
  shows skill) or a different, non-OCO structure, to be designed and tested
  with the same oracle → noise method before inclusion.
- **V2 reversion is marginal:** oracle Sharpe 0,17 on 2024 and 1,60 on TRAIN at
  6 h, and below the frequency floor at 24 h.
- **The only direction-free engine with a positive requirement here** places
  TP/SL from the predicted extremes with a random side: range-first at 6 h.
  It needs a range forecast of about **0,55–0,7× MEAN20 MAE**.
- **The volatility forecaster is therefore the decisive module.** HAR is at
  ~0,9–0,98×. Module (a) is the first to improve, and the regime and
  meta-labeling modules are judged by whether they move the strategy toward
  that requirement.

The old strategy, kept only as a comparator, is structurally different. It was:
direction + an SL at a configurable multiple, a TP inset from the predicted
long extreme, and an early close when short-horizon predictions put the SL
before the TP. All of that rests on price-level forecasts treated here as infeasible.

## 1. Business rules (hard, apply to every order)

- **Toda orden lleva SL y TP calculados y efectivos desde la entrada; el cierre anticipado del agente es adicional, nunca sustituto.**
  - Both levels are native broker-side orders set at entry.
  - The SL may be moved (trailing) but never removed.
  - Distances run from the broker minimum up to 1,5 × the causal block sigma.
  - Exits by TP, SL, early close and time are reported, and a TP that is almost
    never hit is flagged (STRATEGY_CONTRACT §14, `OrderRules`).
- **Volume in integer units:**
  `units = floor(percentual_order_size · rel_volume · current_balance · leverage / price)`.
  - `rel_volume` = 0,10 (bounded ≤ 0,20; tuned later on the inner year only).
  - `percentual_order_size` ∈ (0, 1] is chosen per order by the risk engine.
  - Margin basis, broker minimum 1 unit (DOCUMENTED_GENERAL_NOT_ACCOUNT_VERIFIED),
    margin feasibility enforced.
  - Leverage scenarios 1:10 / 1:50 / 1:100 (default) are reported for every main
    result (STRATEGY_CONTRACT §15, `Sizing`).
- **Frequency:** ≥ 3 trades per business week (variant 4) and ≤ 4 orders per
  business day (variant 2), enforced inside the strategy. A configuration that
  fails is not reported as a result.
- **Ruin:** equity ≤ 0 means stop-out, no Sharpe, invalid.
- **Data:** TRAIN 2020–2022 and inner validation 2023 decide; 2024 confirms once;
  TEST closed. Costs are ASSUMED_NOT_BROKER_VERIFIED, using the realistic
  profile of `f093e6e`.

## 2. The integrated strategy: a fixed, documented decision pipeline per block

Each step has a **neutral fallback**, so the pipeline always runs.

| # | Module (ML area) | Output used | Neutral fallback |
|---|---|---|---|
| 1 | Regime (unsupervised, causal filtering) | mode ∈ {expansion → V1, contraction → V2, unclear → no trade} | fixed rule on predicted range / recent range (V1 if ≥ 1,10, V2 if ≤ 0,90, else no trade) |
| 2 | Volatility forecaster | Û, D̂ (and σ̂) for levels, filters and sizing | MEAN20 of past excursions |
| 3 | Scheduled-event module (causal effect, publication time ≤ t0) | blackout, or adjusted thresholds/mode around releases | events ignored |
| 4 | Anomaly score (self-supervised) | veto, or a lower `percentual_order_size` | no veto, full size |
| 5 | Direction classifier (optional) | asymmetric levels (tilt) only if it shows skill on 2023 | symmetric levels |
| 6 | Meta-labeling classifier | take/skip and `percentual_order_size` from P(profitable after costs) | take every proposal, size 1,0 |
| 7 | Risk engine | sizing, early exits, throttles | fixed size 1,0, no early exits beyond time/forced flat |

### Profit engine (direction-agnostic; harvests predictable volatility)

- **Expansion → V1 OCO breakout:**
  - buy-stop at r + k_entry·Û and sell-stop at r − k_entry·D̂; the first fill
    cancels the other;
  - broker-side TP = fill ± k_tp·(that side's excursion) and SL = fill ∓ k_sl·(excursion);
  - the SL is **moved** as a volatility-scaled trailing stop (k_trail), so winners
    run toward the TP.
  - Fixed a priori: k_entry 0,5, k_tp 0,4, k_sl 0,5, k_trail 0,35. k_entry + k_tp < 1
    keeps the TP inside the predicted extreme, so it is effective.
- **Contraction → V2 range reversion:**
  - resting OCO limits at a causal anchor (mean of the previous B closes; never
    centred) ± b·σ̂, with σ̂ = R̂/2;
  - TP at the anchor, SL at the band ∓ m·σ̂, time stop at block end;
  - b 0,8 and m 2 fixed a priori.
- **Event windows:** blackout or an event-specific mode, decided by module 3.
- **Meta-labeling (López de Prado):** the rule proposes a trade; a classifier
  estimates P(trade profitable after costs) from state features at t0 and decides
  take/skip and size.
  - Its metric vs naive: AUC and log-loss vs the base rate of profitable proposals.
  - Strategy Sharpe vs that metric uses the same oracle → realistic-noise method.

### Risk engine (separate; uses no price forecast)

- **Volatility targeting:** `percentual_order_size = clip(target_risk / σ̂, floor, 1)`.
  The volatility forecast pays twice: in the levels and in the sizing.
- **State-based early exits** (in addition to the broker SL/TP, never instead):
  - regime flips against the mode → close;
  - realized-volatility shock above σ̂·q (anomaly module) → close or cut;
  - time stop and the forced-flat rules.
- **Drawdown throttle:** size scaled down after an equity drawdown (CPPI-like
  floor), a daily loss limit and a weekly cap; frequency and daily-cap rules kept.
- **Reported separately** for the profit engine and the risk engine: net profit,
  Sharpe, **max drawdown, Sortino, worst week, tail loss (5 % daily CVaR)**.

## 3. Features (all causal; availability time ≤ t0)

- **Multi-scale volatility:** realized range over 6 h, 1 day, 1 week and 1 month
  (HAR-style), the ratio of the predicted range to recent range, and Parkinson /
  Garman–Klass estimators from completed bars.
- **Multi-scale trend strength:** normalized moving-average slopes (fast/slow),
  Kaufman efficiency ratio, and the absolute return over 1 day and 1 week
  divided by volatility.
- **Calendar:** position of the block in the business day and day in the week,
  both from the configured calendar (never weekday names). Event flags come
  from the economic-calendar contract (publication time).
- **Not allowed:** centred windows, wavelets, smoothers (Kalman smoothers, HP
  filters, two-sided filters), or any feature whose availability time is after t0.

## 4. Modules — method, metric vs its own naive, oracle → realistic error

| Module | First version (cheap) | Later (M5PHET providers) | Own metric vs naive | Requirement study |
|---|---|---|---|---|
| (a) Volatility | HAR / EWMA (exists: `app/short_horizon/predictors.py`) | TabICLv2, Chronos-2 | MAE of Û, D̂, R̂ × MEAN20 and × persistence | V1/V2 break-even table |
| (b) Regime | HMM (Gaussian, 2–3 states) and GMM/k-means on volatility + trend features; fit on TRAIN only; **forward filtering** probabilities (no smoothing) | same | log-likelihood vs single state; state persistence; mode hit rate vs the fixed-rule fallback | Sharpe vs mode accuracy (oracle mode = ex-post realized expansion/contraction, then flipped with probability p) |
| (c) Events | effect of the event type on block range: matched comparison of event vs non-event blocks with the same calendar position and recent volatility; publication-time contract | causal forests | MAE of range around events vs the volatility module alone; ATE with intervals | Sharpe with blackout vs without |
| (d) Anomaly | isolation forest / autoencoder reconstruction error on causal features, fit on TRAIN | self-supervised encoders | AUC for "block range > q · predicted" vs a constant score | Sharpe vs veto precision |
| (e) Direction | logistic on lagged returns (exists) | any | log-loss / AUC vs base rate | include only if it beats the base rate on 2023 |
| Meta-label | gradient-boosted classifier on state features at t0 | — | AUC / log-loss vs the base rate of profitable proposals | Sharpe vs AUC curve (oracle labels degraded) |

## 5. Ablation and inclusion rule (decided without 2024)

1. **Fit and decide without 2024.**
   - Fit every module on TRAIN **2020–2022**.
   - Choose inclusion and parameters on the **inner validation 2023**.
   - The split is chronological and **purged**: training labels end before the
     2023 blocks start, with an embargo of one block length.
2. **Ablation on identical blocks, costs and rules:**
   - **full pipeline vs the pipeline with module X set to its neutral fallback**;
   - a module is **included** if its marginal Sharpe on 2023 is > 0 and its own
     metric beats its naive.
3. **Confirm once.** The frozen integrated strategy is evaluated **once on
   VALIDATION 2024** as development evidence.
   - Report each module's marginal contribution on 2023 (the decision) and 2024
     (the confirmation).
   - Report each module's own metric vs its naive.
4. **Refit cadences** are a separate, declared comparison: static, weekly
   warm-start and daily.
5. **Parameter choices.** Every module parameter is fixed a priori or chosen on
   2023 only. `rel_volume` and `percentual_order_size` bounds go to a later
   DEAP/sweep on 2023 only.

## 6. Reuse (do not rebuild)

- **This repository (`app/short_horizon/`):**
  - calendar, block grid and causal labels;
  - the realistic event simulator (OCO, trailing SL, trade-through, latency, swap);
  - business rules (`OrderRules`, `Sizing`), frequency/ruin validity, the sealed
    prediction adapter, and the V1/V2 builders.
- **predictor (I6-DMO):**
  - the multi-output training/serving infrastructure and seals for later learned
    volatility models;
  - the audited **prequential weekly runner** for weekly warm-start refits.
- **M5PHET provider contract:** TabICLv2 / Chronos-2 as volatility and anomaly
  providers through the same sealed-prediction adapter.
- **feature-eng / preprocessor causal transforms:** multi-scale features with
  availability times. Their causality audits already exist and are reused.
- **Economic-calendar contract** (publication times) for module (c).

## 7. Leakage checks (each a test before scores)

- **No forward-dependent features.** Perturbing bars after t0 changes no feature,
  label of an earlier block, prediction or decision at t0 (pattern of
  `test_labels_are_causal_and_local`).
- **No smoothing.** No centred/two-sided filter or wavelet in any feature path;
  regime probabilities are forward-filtered (a test compares filtering with the
  smoother on a truncated series).
- **Availability ≤ t0** for every input, including macro releases (publication
  vintages).
- **Sealed predictions** with `made_at ≤ t0`, refused otherwise (adapter).
- **Purging and embargo** between 2020–2022, 2023 and 2024; no 2024 row in any
  fit, threshold or selection.
- **Inclusion decisions recorded before 2024 is scored.** Each decision is recorded
  with its digest before the 2024 run.

## 8. Deliverables (in order)

1. **V1/V2 study** ([VOLATILITY_STRATEGIES.md](VOLATILITY_STRATEGIES.md)): the
   required volatility-predictor MAE for each engine. Done in this branch:
   V1 refuted, V2 marginal, random-direction range-first at 6 h needs about
   0,55–0,7× MEAN20.
2. **Risk engine** (vol-targeting, early exits, throttles) on the V1/V2 rule
   fallback: contribution vs fixed size.
3. **Regime module** (HMM forward filter): the oracle → realistic curve for mode
   accuracy, then the fitted module.
4. **Meta-labeling:** profitable-proposal labels on 2020–2022, AUC curve, take/skip
   and sizing.
5. **Events module:** effect estimates on 2020–2022, blackout decision on 2023.
6. **Anomaly module.**
7. **Direction tilt:** only if skill on 2023.
8. **Integrated pipeline:** frozen on 2023, one confirmation on 2024; figures of
   Sharpe/profit/max drawdown vs each module's metric, in the doctoral style.
