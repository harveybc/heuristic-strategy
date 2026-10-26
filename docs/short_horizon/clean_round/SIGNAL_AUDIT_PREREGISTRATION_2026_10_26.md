# Signal audit: pre-registration (2026-10-26, written and committed before any result is computed)

Question: is there ANY out-of-sample predictive signal in the recipe's inputs for the targets the strategy needs?
Strategy-free: no strategy layer, no thresholds chosen on the scored block, no trading replay of a policy. TEST
(≥ 2025-01-01) stays closed. Tool: `tools/expert_guided_signal_audit.py`; evidence:
`docs/short_horizon/clean_round/evidence/signal_audit/SIGNAL_AUDIT.{md,json}`.

## Data

- Inputs: the recipe's 121 columns (fold recipes `CONSUMER_RECIPE_wf2022.json` / `_wf2023.json`, fixed fit-block scaler
  constants, loaded exactly as E3-M loads them), at the decision row and at 4 declared lags of observed rows: 0, 2, 6,
  12, 24 (window slots −1, −2, −4, −7, 0 of the 13 × stride-2 window) → 605 features. Rows: eligible flat decision rows
  (the C2 outcome table's `eligible`) with a complete window.
- Labels: the engine-exact C2 entry outcome tables `OUTCOMES_ENTRY_wf2022.npz` / `_wf2023.npz` (digests checked) plus
  one extra entry table for the symmetric 1σ/1σ cell, built with the same engine function (`outcomes.entry_table`,
  replay contract) because that cell is not in the C2 grid. Availability: a label is used in a period only when its
  resolution bar + 1 h ≤ the period end and it is not censored (the C2 rule).

## Targets (23 per fold)

- (a) P(TP before SL), per side (long, short), cells (TP σ, SL σ): (1.0, 1.0) symmetric (extra table), (1.5, 1.5)
  symmetric (C2 reference cell), (1.5, 2.5) and (2.0, 2.5) (the region C2 chose). 8 targets.
- (b) expected net in σ (engine net including every cost), same 4 cells × 2 sides, regression. 8 targets.
- (c) sign of the forward log return over 6, 12, 24 observed bars (CLOSE[b+h] vs CLOSE[b]), available when bar b+h's
  end ≤ the period end. 3 targets.
- (d) early-close target, strategy-free: a position opened at an eligible row b0 at cell (1.5, 2.5) and still open k
  observed bars later (k ∈ {3, 8}) is a decision row b = b0 + k; label = 1 when its final engine outcome is SL or a
  negative forced flat ("SL first from here"). Per side → 4 targets. Features at row b only (no order geometry; C2G
  geometry is a separate question).

## Folds and blocks

wf2022: fit TRAIN 2020-01 → 2021-07, INNER_A 2021-07 → 2022-01 (model selection and calibration only), INNER_B = 2022
(scored once). wf2023: fit 2020-01 → 2022-07, INNER_A 2022-07 → 2023-01, INNER_B = 2023. OBS_2024: one reported-only
read per fold model, never decides. "Pooled" = the two INNER_B years concatenated.

## Models

- (i) already-trained heads: the INNER_B reported-only metrics recorded in the E3M.json of WF_V13, WF_C2_13 and WF_C2G_13
  units (per seed); reported for context with their seed range; they have no bootstrap CI and do not enter the verdict.
- (ii) sklearn HistGradientBoosting (classifier / regressor): max_leaf_nodes 15, min_samples_leaf 200,
  learning_rate 0.05, l2_regularization 1.0, early_stopping off, random_state 20261026, max_iter ∈ {50, 100, 200, 400}
  chosen by INNER_A log-loss (MSE for regression); classifier probabilities Platt-calibrated on INNER_A.
- (iii) logistic regression (L2, C ∈ {0.001, 0.01, 0.1, 1} by INNER_A log-loss) / ridge (α ∈ {1, 10, 100, 1000} by
  INNER_A MSE), features standardized with fit-block (TRAIN) statistics; logistic probabilities Platt-calibrated on INNER_A.
- (iv) naive: TRAIN base rate (classification) / TRAIN mean and zero (regression).

## Metrics on INNER_B, each with a 95 % CI

Stationary block bootstrap over calendar days (mean block 5 days, 2 000 draws, seed 20261026), resampling all rows of
the drawn days.

- Classification: AUC; Brier skill = 1 − Brier / Brier(naive TRAIN base rate); log-loss skill likewise.
- Regression: Spearman ρ; R² vs the TRAIN mean and vs zero.
- Economic translation:
  - (a) the cost-inclusive break-even win rate p* = −L / (W − L), with W = mean engine net σ of TP-first outcomes and
    L = mean engine net σ of the other outcomes, both on TRAIN; compared with the precision (TP-first rate) in the
    top 10 % and top 20 % scored rows of INNER_B; the realized mean net σ there is reported too.
  - (b) the realized mean engine net σ (costs included) in the top 10 % / 20 % predicted rows.
  - (c) long the top 10 % / 20 % of P(up), short the bottom 10 % / 20 %; realized mean of side·(CLOSE[b+h] − CLOSE[b])
    / σ minus the replay contract's real round-trip cost in σ (`row_cost_sigma` without the allowance); the
    "precision" is the correct-sign share, and the break-even is that the realized net after cost is > 0.
  - (d) p* = the SL-first precision at which closing at the decision row equals holding: from TRAIN,
    Δ = net(hold, engine) − value(close now), with close-now value = side·(OPEN[b+1] − OPEN[b0+1]) / σ[b0+1] − the
    round-trip cost in σ (approximation from mid prices, declared); p* = mean(Δ | y=0) / (mean(Δ | y=0) − mean(Δ | y=1));
    compared with the top 10 % / 20 % precision; the realized mean Δ there is reported (< 0 means closing pays).
- Feature-sign stability (descriptive, never decides): for (a) cell (1.5, 1.5) long and short and (c) 12 bars, the
  Spearman correlation of each current-slot feature with the target per calendar year 2020, 2021, 2022 (rows of the
  wf2023 fit + inner blocks plus the wf2022 decision year; descriptive only) for the 15 features with the largest
  |coefficient| in the wf2023 logistic model; sign flips counted.

## Decision rule (verbatim)

"SIGNAL_PRESENT for a target only if the lower 95 % CI of its INNER_B AUC is > 0.52 (or its Spearman lower CI > 0.02)
in BOTH folds AND the top-decile precision exceeds the cost-inclusive break-even win rate in both folds; otherwise
NO_USABLE_SIGNAL. If NO target is SIGNAL_PRESENT, further strategy-layer tuning is suspended and work moves to inputs
(C3 invariance across years/regimes, data)."

Operationalization: a target is SIGNAL_PRESENT when at least one of models (ii)/(iii) satisfies both conditions in both
folds (the same model in both folds). For (b) and (c) the economic condition is the realized top-decile mean net after
cost > 0; for (a) and (d) it is top-decile precision > p*. Every target, model, fold and the pooled read are reported,
including the failures.
