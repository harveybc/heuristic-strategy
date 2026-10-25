# Stage WF_C2 — learning the consequences of our actions (spec `clean-expert-guided-a1-20261024-v9`, 2026-10-25)

Seed ensemble on the validation year of each walk-forward fold (INNER_B, scored once after every choice was frozen on INNER_A), against the two baselines on the same fold. OBS_2024 is one read and never decides. TEST closed. These are replays of a learned policy under the replay contract, not a profitability claim.

Pre-registered rule: PRE-REGISTERED 2026-10-25 before any WF_C2 result exists: HOLDS / SEPARABLE as already pre-registered (an arm HOLDS only if its seed-ensemble validation Sharpe minus one Lo standard error is > 0 in BOTH walk-forward folds (wf2022 and wf2023), otherwise DOES_NOT_HOLD; two arms are SEPARABLE on a window only when |Sharpe_a - Sharpe_b| > the pooled Lo standard error sqrt(SE_a^2 + SE_b^2), otherwise NOT_SEPARABLE); C2 is compared with BOTH baselines of the same fold (WF_<fold>_zz_hold and WF_V12_<fold>_zz_hold_v12) by the seed ENSEMBLES on INNER_B (never seed means); an ensemble with fewer than 2 trades/week on INNER_B is INVALID (owner's rule) whatever its Sharpe; 2024 one read, never decides; every verdict also obeys the sanity gate (sanity_gate.rule)

Early-close rule: PRE-REGISTERED 2026-10-25 before any WF_C2 result exists: C2 counts as fixing the early close only if its ensemble early-close value (engine counterfactual hold of every agent close, INNER_B, the arm's primary fitness) is > 0 in BOTH folds AND its closer recall of SL-first is >= 2x the baseline's (the WF_<fold>_zz_hold ensemble, same fold, same audit tool); a C2 strategy with no agent close on the calibration block is DEGENERATE_NO_EARLY_CLOSE and not selectable (require_early_close in the contract)

Fitness rule: PRE-REGISTERED 2026-10-25 before any WF_C2 result exists (owner's fitness variants F0-F6, no retraining: the same frozen specialists and seed ensembles, only the strategy point changes; activity = exactly 2.0 trades/week as a SELECTION criterion): per arm, the fitness variant is CHOSEN on the wf2022 decision year by the highest ensemble Sharpe - 1 Lo SE among variants whose chosen point is valid there out of sample (>= 2 trades/week); it is CONFIRMED on wf2023 when its ensemble Sharpe - 1 Lo SE > 0 there with >= 2 trades/week (the HOLDS rule on the confirmation fold), otherwise REFUTED; every variant is reported with its components, its chosen valid point and its full-archive view (best unconstrained Sharpe, best unconstrained point by the fitness with INNER_B metrics marked NOT SELECTABLE, best valid point, share valid)

## Arm `zz_feat_c2`

| fold | C2 Sharpe ± SE | trades/wk | DD % | valid | WF baseline ± SE | WF_V1 ± SE | vs WF | vs WF_V1 | 2024 (one read) |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| wf2022 | 0.06 ± 1.00 | 1.94 | 28.3 | False | 0.46 ± 1.00 | 1.05 ± 1.00 | REFUSED_FAILED_SANITY | REFUSED_FAILED_SANITY | -0.34 |
| wf2023 | -2.52 ± 1.01 | 1.48 | 38.9 | False | -1.51 ± 1.01 | -3.06 ± 1.01 | REFUSED_FAILED_SANITY | REFUSED_FAILED_SANITY | -0.44 |

- HOLDS (C2): **REFUSED_FAILED_SANITY**; baselines: WF REFUSED_FAILED_SANITY, WF_V1 REFUSED_FAILED_SANITY
- Early close: **REFUSED_FAILED_SANITY**
  - wf2022: value +4042 USD (CI -3875, +12343), recall 0.03 vs baseline 0.06, cut-winner share 0.46, AUC 0.52 → FAIL
  - wf2023: value +2331 USD (CI -1107, +6082), recall 0.04 vs baseline 0.10, cut-winner share 0.43, AUC 0.51 → FAIL
- Fitness variant: chosen on wf2022 = `None`, wf2023 → **REFUSED_FAILED_SANITY**

| fitness | wf2022 Sharpe ± SE | trades/wk | valid | wf2023 Sharpe ± SE | trades/wk | valid |
| --- | --- | ---: | --- | --- | ---: | --- |
| sharpe_lcb | 0.06 ± 1.00 | 1.94 | False | -2.52 ± 1.01 | 1.48 | False |
| sharpe_plus_return | 0.06 ± 1.00 | 1.94 | False | -1.96 ± 1.01 | 1.65 | False |
| sharpe_plus_return_plus_activity | 0.06 ± 1.00 | 1.94 | False | -1.96 ± 1.01 | 1.65 | False |
| trade_tstat | 0.06 ± 1.00 | 1.94 | False | -1.96 ± 1.01 | 1.65 | False |
| calmar | 0.06 ± 1.00 | 1.94 | False | -1.96 ± 1.01 | 1.65 | False |
| return_minus_dd | 0.06 ± 1.00 | 1.94 | False | -1.96 ± 1.01 | 1.65 | False |
| balanced | 0.06 ± 1.00 | 1.94 | False | -1.96 ± 1.01 | 1.65 | False |

## Sanity gate (FAILED_SANITY units are excluded from every verdict; never shown as results)

| unit | sanity | causes |
| --- | --- | --- |
| WF_C2_wf2022_zz_feat_c2 | **FAILED_SANITY** | {'below_2_trades_per_week': 2, 'one_side_over_90pct': 2} |
| WF_C2_wf2023_zz_feat_c2 | **FAILED_SANITY** | {'below_2_trades_per_week': 4, 'single_exit_reason': 2} |
| WF_wf2022_zz_hold | **PASS** | — |
| WF_wf2023_zz_hold | **FAILED_SANITY** | {'below_2_trades_per_week': 3, 'one_side_over_90pct': 1} |
| WF_V12_wf2022_zz_hold_v12 | **FAILED_SANITY** | {'below_2_trades_per_week': 1, 'one_side_over_90pct': 2} |
| WF_V12_wf2023_zz_hold_v12 | **FAILED_SANITY** | {'one_side_over_90pct': 4} |

