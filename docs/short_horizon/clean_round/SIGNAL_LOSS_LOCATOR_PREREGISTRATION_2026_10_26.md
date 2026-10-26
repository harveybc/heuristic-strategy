# Signal-loss locator: pre-registration (2026-10-26)

Committed before any S1b–S6 or diagnostic number is computed. Supersedes the framing of
`SIGNAL_AUDIT_PREREGISTRATION_2026_10_26.md` (@99a71ef).

## 0. Why a locator, and what was already seen

- **Correction from the coordinator.** Four times "the inputs have no signal" was concluded, and each time the real
  cause was an internal defect: no feature selection, no input encoding, a continuous output used for classes, and a
  constant SL target with absolute strategy genes.
- **What this audit does.** It does not decide anything about the market. It LOCATES the stage of OUR pipeline where
  signal is lost. Each verdict is per stage: SIGNAL_PRESENT or ABSENT_AT_THIS_STAGE.
- **Clause withdrawn.** The "NO_USABLE_SIGNAL → suspend strategy tuning, move to inputs" clause of the earlier
  pre-registration is withdrawn.
- **Declared, not blind.** The earlier audit (tool `tools/expert_guided_signal_audit.py`, @a132022) was already run to
  completion before this correction arrived. Its HistGradientBoosting / logistic numbers on the decision-bar features
  plus lags 0–24 rows (here: stage S1) were SEEN before this document was written: overall AT_LEAST_ONE_SIGNAL_PRESENT,
  with 9 targets passing (TP-before-SL targets and the SL-first k = 8 target). S1 is therefore NOT a blind test.
- **Blind.** Everything else below (S1b, S2–S6 and the diagnostics) is computed after this commit.

## 1. Same rows, targets and folds for every stage

- **Folds.** wf2022 (fit 2020-01 → 2021-07, INNER_A 2021 H2, INNER_B 2022) and wf2023 (fit → 2022-07, INNER_A
  2022 H2, INNER_B 2023). OBS_2024 is a reported-only read. TEST is closed.
- **Rows.** The eligible flat decision rows of the C2 outcome table with complete windows and available labels (C2
  availability rule), identical across stages.
- **Targets (12).**
  - (a) TP before SL at the C2 reference cell (1.5σ, 1.5σ), long and short — the target of the C2 binary opener heads;
  - (b) engine net σ at cells (1.5, 1.5), (1.5, 2.5) and (2.0, 2.5), long and short — the C2 regression opener
    outputs;
  - (d) SL-first k ∈ {3, 8} bars after entry at cell (1.5, 2.5), long and short — evaluated against the C2 closers.

## 2. Stages

| stage | what scores the rows | applies to |
| --- | --- | --- |
| S1 | HistGradientBoosting on the decision-bar recipe features + lags of 2, 6, 12, 24 observed rows (= the earlier audit's model (ii); seen) | a, b, d |
| S1b | S1 features + causal rolling means over the last 72 and 168 observed rows and rolling std over 24 and 168 | a, b, d |
| S2 | HistGradientBoosting on the flattened recipe WINDOW tensor exactly as the network sees it (13 slots × stride 2 × 121, the fold's fixed constants) — a diagnostic, never a model | a, b, d |
| S3 | our trained network heads, per seed, before calibration (logit margin / raw regression) — WF_C2_13 and WF_C2G_13 openers; WF_C2_13 closers (zero state, as trained); WF_V13 zz_hold_v13 / both_hold_v13 openers (trained on ZigZag labels) scored against target (a) | a (C2, C2G, V13), b (C2, C2G), d (C2 closers) |
| S4 | the same heads after their Platt calibration (AUC and rank metrics are invariant to a monotone map by construction; Brier skill differs) | as S3 |
| S5 | the seed ensemble (mean of the 5 seeds' calibrated outputs) | as S3 |
| S6 | the strategy's opened trades on INNER_B (the trading-detail trade lists of the seed-ensemble primary strategy): precision of the opened trades against target (a) at their decision rows, side by side; compared with S5's and S1's precision in their top-k rows, k = the number of opened trades of that side | a (C2, C2G, V13) |

HistGradientBoosting settings and INNER_A selection/calibration as in the earlier pre-registration. C2G closers are
not evaluated (their outputs depend on the order's geometry); that is declared, not hidden.

## 3. Metrics and stage-to-stage drops

- On INNER_B per fold: AUC (a, d) or Spearman (b), Brier skill, and top-10 % precision (a, d) or top-10 % realized net
  σ (b).
- Stationary block-bootstrap 95 % CIs over days (block 5, 2 000 draws, seed 20261026).
- Stage drops are computed **paired** on the same bootstrap draws (the same days) for consecutive stages:
  - S1 → S1b, S1 → S2, S2 → S3 (the mean of the per-seed metrics), S3 → S5, and S5 → S6 (precision);
  - S6 is also compared with S1 at the same k.
- **SUSPECT_INTERNAL_DEFECT** is the label of a stage whose drop CI lies entirely below 0. A stage is SIGNAL_PRESENT
  when its AUC lower CI is > 0.52 (Spearman lower CI > 0.02), and ABSENT_AT_THIS_STAGE otherwise.
- None of these is a statement about the market.

## 4. Internal diagnostics (reported; they locate, they never decide)

1. **Training dynamics.**
   - Best epoch, epochs run and stop reason of every head of every seed of WF_C2_13, WF_C2G_13 and WF_V13 (both
     folds), and the share with best epoch ≤ 3.
   - Per-epoch learning curves (train vs INNER_A loss) are NOT stored by E3-M (only epochs / best epoch / stop
     reason). They are declared UNMEASURED: obtaining them needs a re-fit, which is out of scope here.
2. **Capacity.** Parameter count per head vs the TRAIN rows it was fitted on (ledger FIT events).
3. **Capacity vs drift.** S1 HistGradientBoosting AUC on TRAIN (in-sample) vs INNER_B.
4. **Grid bounds.** For the C2 / C2G / V13 opened trades, the TP and SL distances in σ_entry, rounded to the nearest
   grid value, and the share sitting on the grid maximum (TP 2.0σ, SL 2.5σ) or minimum.
5. **Daily cap: first vs best signal.** On each INNER_B day with an opened trade, the opened row is compared with the
   best-scored row of the same side on the same day. "Best" is measured twice: by S5 (the network's own calibrated
   score) and by S1. The comparison reports the target-(a) precision of each, with a paired bootstrap CI.
6. **Window vs horizons.**
   - The input window spans 26 h (13 slots × stride 2 observed rows).
   - For each target cell, report the resolution time in hours (median, p90), from the engine outcome table, entry
     fill → resolution bar.

## 5. Evidence

`docs/short_horizon/clean_round/evidence/signal_audit/SIGNAL_LOSS_LOCATOR.{md,json}`, next to the earlier
`SIGNAL_AUDIT.{md,json}`, which is kept as computed. Tool: `tools/expert_guided_signal_loss_locator.py`, with tests.
