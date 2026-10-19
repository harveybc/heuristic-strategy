# Learned causal pivot detector on the REAL feature arms (DEVELOPMENT)

**Question.** The ZigZag study found a ≈21-Sharpe gap between the hindsight expert (Depth 3 /
Deviation 36.3 pips / Backstep 2) and its causally confirmed version. A smoke probe (MLP, no window,
substitute price features) was inconclusive. Does a real detector, fed Musashi's 19-feature selection
on the two input arms, recover any of that gap?

**Answer (DEVELOPMENT, one instrument, 3 seeds, 2024 looked at once).** No. All three models are at
chance on the pivot targets (AUC 0.50–0.55 on 2023 and 2024), their calibrated log-loss is worse than
the base rate, and their signal precision (0.21–0.24) sits at the per-class base rate (0.20). The
trivial reversal strategy they drive is at or below zero Sharpe; the operating point chosen on 2023 for
m1 and m2 is "emit nothing" (Sharpe 0, not valid under the ≥ 3 orders/week rule). The oracle → noise
sanity says a detector on these targets needs precision ≳ 0.6–0.7 before the TP = 2·σ̂ strategy turns
positive and ≈ 0.9 to reach Sharpe 6–7. The gap stays open on the detection side.

Results: `results/pivot_detector/` (tables `detector_real.csv`, `detector_real_summary.csv`,
`oracle_noise.csv`, `training_curves.csv`; sealed calibrated predictions
`detector_real_predictions.csv.gz` + `DETECTOR_REAL_SEAL.json`; figures `zigzag_brecha_detector.png`,
`oraculo_ruido_precision.png`, `curvas_entrenamiento.png`). Code: `tools/short_horizon_pivot_detector.py`
(coordinator stages) and `tools/short_horizon_pivot_detector_train.py` (GPU stage).

## 1. Data

| Item | Value |
|---|---|
| Lake file | EUR/USD H1 `market_data/forex/g10/eurusd/1h.parquet`, sha256 `c6628375…` (loaded 2019-12-01 → 2025-01-01 exclusive; TEST refused by the loader) |
| Targets | hindsight ZigZag pivots (`app/short_horizon/zigzag.py`, sha256 `b30788e6…`, 6 371 pivots), y ∈ {none, valley within (t, t+2], peak within (t, t+2]}, k = 2 bars; `labels_h1.npz` sha256 `d7cf32ba…` |
| Features | Musashi's 19-feature selection, manifest `2ff34fa2…` (predictor `docs/i6n/direct_multi_output_static/INTERFACE_DATA.json` @ `00c27fed`; sealed in `EXPERIMENT.mg2m.json` seal `416c1156…`) |
| Arm A | sealed MG2m population `population-tf2210` (digest `ba01f87a…`, re-verified with the real `common_population verify` against the retained INPUT.npz `ef22ddb5…`): 13 slots × 2 open-market hours, `offsets_hours`, `closure_crossed`; FIT-only shared z-score of the population (20 717 scaler rows) |
| Arm B | raw 48-bar hourly window of the same 19 features from INPUT.npz, rows with ts ≤ origin only; same scaler |
| Origins | 21 837 population origins (FIT ∪ INNER ∪ DEVELOPMENT); 175 dropped (arm-B window contains a non-finite raw value), 20 without a lake bar; **21 642 kept**: TRAIN 2020–2022 = 14 423, INNER 2023 = 2 565, CONFIRM 2024 = 4 654 |
| Purge | origins whose (t, t+2] label window crosses a split cut: 0 (no 2-h-grid origin falls in the two hours before a 1 January cut) |
| Class counts | TRAIN none/valley/peak = 8 676 / 2 874 / 2 873; 2023 = 1 611 / 475 / 479; 2024 = 2 833 / 915 / 906 |

The labels are hindsight by construction (the final repainted ZigZag over 2019-12..2024-12) and are
used only as targets; every model input is causal (slot ≤ t, bar ≤ t).

## 2. Models and training

| Model | Input | Architecture |
|---|---|---|
| m1 | arm A | the I6-DMO network wired exactly as `tools.i6dmo.model.build_dmo` (per-feature causal Conv1D branches via the registered `dmo_branch`, weightless `sequence_concat` fusion, `DynamicPhysicalTimeEncoder`, `dmo_core`), softmax 3-class head instead of the regression heads; 2 443 parameters |
| m2 | arm B | GRU(32) → Dense(16, relu) → softmax(3); 5 667 parameters |
| m3 | arm B flattened (48 × 19 = 912) | `HistGradientBoostingClassifier`, lr 0.05, 31 leaves, up to 400 iterations |

Recipe for m1/m2: AdamW lr 1e-3, weight decay 1e-4, batch 256, class-weighted cross-entropy
(inverse frequency, normalized), max 200 epochs, seeds 6102026 + {0, 1, 2}
(`keras.utils.set_random_seed` before each build, as in `build_dmo`). Post-hoc temperature scaling of
the probabilities on INNER 2023 (log grid 0.25–8). The operating threshold τ ∈ {0.4, 0.5, 0.6, 0.7}
is chosen on 2023 by the mean Sharpe over seeds (valid points first); 2024 is reported once at that τ.

**Early stopping (owner's corrected formula).** TRAIN term = the class-weighted running loss that
training itself produces at the end of each epoch (no re-evaluation of TRAIN in inference mode);
INNER term = the same class-weighted log-loss on INNER 2023 (purged) at the end of each epoch;
monitor = 0.5·(TRAIN + INNER); patience 10, min_delta 1e-8, best weights restored. The paired-mean form
mirrors Musashi's `PairedMonitor` (§3). As cheap diagnostics, **not** part of the monitor, each epoch
also records on INNER only: the reversal strategy's Sharpe and trades/week (τ = 0.5, TP 2·σ̂),
plain log-loss and AUC. m3 is monitored every 10 boosting iterations with the same two loss terms
(the TRAIN term from its staged probabilities, since boosting exposes no running loss).
**2024 is a single confirmation: never evaluated during training, never used for stopping or
selection.** If the owner later wants 2024 inside the monitor, 2024 becomes development data and a
forward/paper period becomes the holdout.

Per-epoch curves (`training_curves.csv`, `curvas_entrenamiento.png`): for m1/m2 the TRAIN running
loss falls slowly (0.85 → 0.80) while the INNER weighted loss rises from its first epoch (1.10 → 1.12);
the monitor's best epoch is 13–34 and patience stops it at 23–44 epochs. The INNER Sharpe diagnostic
wanders between −1.3 and +1.4 with no relation to the loss. For m3 the TRAIN term collapses (1.04 →
0.21) while INNER rises (1.09 → 1.28): the loss mean never stops improving, so all three m3 seeds run
to 400 iterations. This is a documented weakness of the mean-of-losses monitor for a memorizing
learner: with a fixed capacity it ends at the most over-fitted iteration.

## 3. Musashi's current advances: what was reused (file @ commit)

Read-only. Reused exactly as recorded; nothing attributed to him was invented.

| Item | Where he recorded it | Reused here as |
|---|---|---|
| Paired early-stopping monitor: score = −0.5·(train + validation), patience 10, min_delta 1e-8, best checkpoint restored, `generalization_gap` recorded per evaluation | agent-multi `agent_plugins/paired_early_stopping.py` @ `e119e8ad` (`PairedMonitor`, `SETTINGS`); `docs/work_plan/46_PAIRED_SAC_EARLY_STOP.md` @ `e119e8ad` | the mean-of-two-terms monitor with the same patience / min_delta / restore semantics and the per-epoch gap column (`loss_gap`); the owner's correction fixes the two terms to the running TRAIN loss and the INNER loss |
| Evaluation in isolated inference mode, "never call learn", policy hash unchanged by evaluation | same files | the INNER term and the diagnostics come from one `model.predict` pass per epoch; no training during evaluation |
| Observation window: 13 observations at stride 2 open-market hours, exact selected support | `docs/work_plan/45_SELECTED_RL_COMPARISON.md` @ `ab265fad`; `docs/work_plan/43_STATIC_RL_CURATED_COMPARISON.md` @ `ab265fad` (line 15) | arm A is the same 13 × 2 h market-time window (the sealed MG2m population) |
| Selected 19-feature manifest `2ff34fa2…` as the ordered input columns | predictor `docs/i6n/direct_multi_output_static/INTERFACE_DATA.json` @ `00c27fed`; agent-multi `44_CURATED_RL_INPUT_INTERFACE.md` @ `ab265fad` (`feature_columns`) | both arms use exactly these 19 columns in manifest order (asserted against the population scaler) |
| Normalization: scaling uses only strictly preceding rows (`rolling_zscore` over [step − window, step)) | gym-fx `preprocessor_plugins/feature_window_preprocessor.py` @ `e1d0558` (branch `codex/rl-static-inputs-20261018`) | **not reproducible on the sealed population**: its z-score is the FIT-only shared scaler (fit rows 2020-01..2023-11, sealed); arm B reuses that scaler for comparability. Documented difference, not a reuse |
| No BatchNorm, dropout, augmentation or extra weight decay in his experiment; they are "separate future ablations, never combined" | `46_PAIRED_SAC_EARLY_STOP.md` @ `e119e8ad`; `43_STATIC_RL_CURATED_COMPARISON.md` @ `ab265fad` (line 23) | none added here either; the only regularizer is the I6-DMO recipe's AdamW weight decay 1e-4 (predictor `MODEL_MATCHED.md`) |
| Activity reporting against the no-trade baseline ("a no-trade policy can attain zero") | `46_PAIRED_SAC_EARLY_STOP.md` @ `e119e8ad` | trades/week and the ≥ 3/week validity flag are reported next to every Sharpe; the m1/m2 operating points are explicitly the no-trade zero |
| Sizing: rel_volume 0.10 of the balance; leverage as an account scenario | `45_SELECTED_RL_COMPARISON.md` @ `ab265fad` (balance budget) | the strategy adapter's `Sizing(rel_volume=0.10, leverage=100)` (the ZigZag study's setting; his initial campaign uses 50) |

His reported training Sharpe 300–400 with poor validation is not in these documents; the
anti-overfitting work he has recorded is the paired monitor above. No other technique of his exists
to reuse.

## 4. Evaluation through the sealed strategy

`tools/short_horizon_zigzag_study.Ctx.run`: trivial reversal at each signal (valley → long, peak →
short), one position at a time, entry next open, TP = k_tp·σ̂ with k_tp = 2 (the confirmed version's
choice on 2020–2022), SL = 2·σ̂, realistic costs, order rules (minimum distance, effectiveness
≤ 1.5 × causal range), ≤ 4 orders/day, validity ≥ 3 orders/week, rel_volume 0.10 at 1:100. One
signal per 2 bars.

Mean over 3 seeds; `valid` = all seeds valid.

| Model | τ (chosen on 2023) | Period | AUC valley / peak | Log-loss (calibrated) / base rate | Precision / recall at τ | Signals | Sharpe | Orders/wk | Valid |
|---|---:|---|---|---|---|---:|---:|---:|---|
| m1 | 0.7 | 2023 | 0.541 / 0.546 | 1.093 / 0.919 | – / 0.000 | 0 | 0.00 | 0.0 | no |
| m1 | 0.7 | 2024 (once) | 0.521 / 0.532 | 1.095 / 0.941 | – / 0.000 | 0 | 0.00 | 0.0 | no |
| m2 | 0.7 | 2023 | 0.517 / 0.504 | 1.097 / 0.919 | – / 0.000 | 0 | 0.00 | 0.0 | no |
| m2 | 0.7 | 2024 (once) | 0.510 / 0.501 | 1.104 / 0.941 | – / 0.000 | 0 | 0.00 | 0.0 | no |
| m3 | 0.4 | 2023 | 0.538 / 0.535 | 0.966 / 0.919 | 0.236 / 0.061 | 246 | **−0.71** | 7.8 | yes |
| m3 | 0.4 | 2024 (once) | 0.545 / 0.522 | 0.986 / 0.941 | 0.222 / 0.052 | 427 | **−1.99** | 7.5 | yes |

The m1/m2 choice is degenerate: every τ that trades is negative, so the no-trade zero wins. The
lowest threshold that trades (τ = 0.4): m1 Sharpe −0.53 (2023) / −0.65 (2024) at 3.2–3.8 orders/week,
precision 0.21–0.23; m2 −0.63 / −0.22 at 1.5–1.8 orders/week, precision 0.17–0.20. Per-seed and
per-threshold rows are in `detector_real.csv`. Temperatures on INNER hit the grid ceiling (8.0) for
three of the nine fits: calibration flattens those models toward the base rate because they carry
no information. Per-class base rate of valley/peak origins: 0.20 (TRAIN), 0.19 (2023), 0.20 (2024).

**Placement on the gap figure** (`zigzag_brecha_detector.png`, values from `confirmed.csv`,
`detector.csv` and `detector_real_summary.csv`):

| Period | (a) hindsight expert | (b) confirmed | (c) smoke probe | (d) m1 | (e) m2 | (f) m3 |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 (train) | 17.09 | −1.03 | −0.11 | 0.00 | 0.00 | −0.72 |
| 2023 | 19.48 | −2.14 | −0.35 | 0.00 | 0.00 | −0.71 |
| 2024 (once) | 18.91 | −2.16 | −0.61 | 0.00 | 0.00 | −1.99 |

Nothing moves toward (a). The real-feature detectors are not better than the substitute-feature
probe; the probe's AUC 0.66 came from price-path features the manifest mostly lacks (only
`px.close_loc`, `px.log_co`, `px.rv5`, `rg.atr_ratio` are price-derived; the rest are daily
macro / cross-asset returns, constant across the 2-h slots of a day).

## 5. Oracle → noise sanity: the precision the strategy needs

True 2-bar labels on the origins (recall 1), diluted with random false signals (random direction)
to a target precision; 3 seeds; TP = 2·σ̂, SL = 2·σ̂ (`oracle_noise.csv`, `oraculo_ruido_precision.png`).
The study's hindsight next-pivot TP is undefined for origin-based (pre-pivot) signals, so only the
causal TP is shown.

| Precision | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.45 | base rate (0.37 / 0.39) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sharpe 2023 | 7.74 | 5.93 | 4.75 | 2.23 | 1.20 | −0.42 | −1.28 | 0.00 (no trades after the rules) |
| Sharpe 2024 | 9.74 | 7.26 | 4.23 | 2.49 | −0.18 | −1.30 | −0.48 | −0.41 |

Reading: even perfect 2-bar timing with a causal 2·σ̂ TP yields Sharpe 8–10, not the expert's 19 (the
other half is the next-pivot level, as the timing-noise study showed); the strategy turns positive
around precision 0.6–0.7 and needs ≈ 0.9 for Sharpe 6–7. The real detectors sit at 0.21–0.24, i.e. at
the per-class base rate, far left of the break-even.

## 6. Compute receipt

| Item | Value |
|---|---|
| GPU role | option (1): the secondary worker's RTX 4090 Laptop GPU (16 GB). One `nvidia-smi` read on each ssh host before choosing: the secondary showed 0 compute processes; the preferred host (5090 + 5070 Ti) was never used |
| Runtime | TF 2.21.0 / Keras 3.13.2 / scikit-learn 1.8.0 (the role's pinned python), staged predictor tree `c34b1e48` for `tools.i6dmo`; the strategy code, `trading_contracts` and the lake file shipped to a scratch directory |
| Admission | `crispdm-run -m 3333M -t 7200s` = 1.25 × the measured peak of a one-epoch measurement run (2 795 114 496 B cgroup peak, 10.3 s wall) |
| Full run | cgroup peak **2 923 282 432 B** (under the cap), wall **313.8 s**: m1 27.3 / 14.5 / 15.9 s (44 / 23 / 25 epochs), m2 11.3 / 13.4 / 14.1 s (29 / 35 / 35 epochs), m3 71.5 / 71.3 / 71.2 s (400 iterations each) |
| Seeds | 6102026, 6102027, 6102028 |
| Coordinator CPU stages | labels 4 s; sanity 16 s; evaluate ≈ 2 min; all under `crispdm-run -m 3G` |
| Exact revisions | heuristic-strategy branch `satoshi/pivot-detector-20261019` from `ec7944f`; predictor tree `c34b1e48c17d83d0f2eaa08cfc8daed4f1afb861`; Musashi's branches as cited in §3 |
| Sealed outputs | GPU `SEAL.json` (sha256 of every `pred_*.npz` and `RUN.json`), verified before evaluation; `DETECTOR_REAL_SEAL.json` carries the calibrated predictions' sha256, the population / input / labels / design digests, temperatures and operating thresholds |

No campaign root, timer, deploy or TEST row was touched; nothing ran on the 5090.

## 7. Pending (declared, not run)

Continual-learning arms are not part of this run and are to be compared later as declared cadences:
(i) weekly warm-start refit of the detector through the audited prequential runner
(`tools/i6dmo/prequential_test.py`), (ii) test-time-training / online-update variants.

## 8. Limits

- One instrument, 2-bar targets of one ZigZag parameterization; 2024 looked at once. Costs
  ASSUMED_NOT_BROKER_VERIFIED; the bar clock is under repair (both as in the ZigZag study).
- The sealed population carries only origins with all six common heads valid, so the detector sees
  ≈ 36 origins/week, not every bar; the oracle in §5 is computed on the same origins and is the right
  ceiling for these models, not for a per-bar detector.
- Arm B uses the population's FIT-only scaler, not Musashi's causal rolling z-score (not reproducible
  on the sealed population without changing its identity).
- The loss-mean monitor does not stop a memorizing learner (m3 ran to max_iter); a strategy-level
  TRAIN term from the same pass is not available for boosting.
- Temperature calibration on 2023 is also the threshold-selection year (one development year, as the
  study did); 2024 confirms once.
- The hindsight labels near the 2022/2023 and 2023/2024 cuts depend on bars after the cut (repainting);
  that affects targets only, not inputs.
