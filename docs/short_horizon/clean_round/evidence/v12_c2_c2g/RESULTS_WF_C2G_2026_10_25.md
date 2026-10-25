# Stage WF_C2G — closers that see their own order (spec `clean-expert-guided-a1-20261024-v9`, 2026-10-25)

C2 plus one change, the owner's extra arm: the closers alone also read the geometry of their own open order (signed distance to SL and TP in the order's sigma, hours held, side). Seed ensemble on the validation year (INNER_B, scored once after every choice was frozen on INNER_A) against the C2 arm and the WF baseline of the same fold. OBS_2024 one read, never decides. TEST closed. Replays of a learned policy under the replay contract, not a profitability claim.

Pre-registered rule: PRE-REGISTERED 2026-10-25 before any WF_C2G or WF_C2 result exists: HOLDS / SEPARABLE as already pre-registered (an arm HOLDS only if its seed-ensemble validation Sharpe minus one Lo standard error is > 0 in BOTH walk-forward folds (wf2022 and wf2023), otherwise DOES_NOT_HOLD; two arms are SEPARABLE on a window only when |Sharpe_a - Sharpe_b| > the pooled Lo standard error sqrt(SE_a^2 + SE_b^2), otherwise NOT_SEPARABLE); C2G is compared by the seed ENSEMBLES on INNER_B (never seed means) with the C2 arm of the same fold (WF_C2_<fold>_zz_feat_c2: the direct contrast, the only difference is the closers' order geometry) and with the WF baseline (WF_<fold>_zz_hold); an ensemble with fewer than 2 trades/week on INNER_B is INVALID (owner's rule) whatever its Sharpe; 2024 one read, never decides; every verdict also obeys the sanity gate (sanity_gate.rule)

Early-close rule: PRE-REGISTERED 2026-10-25 before any WF_C2G or WF_C2 result exists: C2G counts as fixing the early close only if its ensemble early-close value (engine counterfactual hold of every agent close, INNER_B, the arm's primary fitness) is > 0 in BOTH folds AND its closer recall of SL-first is >= 2x the baseline's (the WF_<fold>_zz_hold ensemble, same fold, same audit tool); the direct C2G vs C2 contrast (early-close value, recall, AUC, cut-winner share, and the same quantities by bucket of the order's distance to its SL) is reported per fold; a C2G strategy with no agent close on the calibration block is DEGENERATE_NO_EARLY_CLOSE and not selectable (require_early_close in the contract)

Fitness rule: PRE-REGISTERED 2026-10-25 before any WF_C2 result exists (owner's fitness variants F0-F6, no retraining: the same frozen specialists and seed ensembles, only the strategy point changes; activity = exactly 2.0 trades/week as a SELECTION criterion): per arm, the fitness variant is CHOSEN on the wf2022 decision year by the highest ensemble Sharpe - 1 Lo SE among variants whose chosen point is valid there out of sample (>= 2 trades/week); it is CONFIRMED on wf2023 when its ensemble Sharpe - 1 Lo SE > 0 there with >= 2 trades/week (the HOLDS rule on the confirmation fold), otherwise REFUTED; every variant is reported with its components, its chosen valid point and its full-archive view (best unconstrained Sharpe, best unconstrained point by the fitness with INNER_B metrics marked NOT SELECTABLE, best valid point, share valid)

## Arm `zz_feat_c2g`

| fold | C2G Sharpe ± SE | trades/wk | DD % | valid | C2 ± SE | WF baseline ± SE | vs C2 | vs WF | 2024 (one read) |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| wf2022 | 0.34 ± 1.00 | 1.64 | 29.8 | False | 0.06 ± 1.00 | 0.46 ± 1.00 | REFUSED_FAILED_SANITY | REFUSED_FAILED_SANITY | -2.35 |
| wf2023 | -2.02 ± 1.01 | 1.73 | 33.4 | False | -2.52 ± 1.01 | -1.51 ± 1.01 | REFUSED_FAILED_SANITY | REFUSED_FAILED_SANITY | 0.17 |

- HOLDS (C2G): **REFUSED_FAILED_SANITY**; C2 REFUSED_FAILED_SANITY; WF baseline REFUSED_FAILED_SANITY
- Early close (rule vs the WF baseline): **REFUSED_FAILED_SANITY**
  - wf2022 C2G vs C2: value +2060 vs +4042 USD, recall 0.04 vs 0.03, AUC 0.61 vs 0.52, cut-winner 0.47 vs 0.46 → MIXED
  - wf2023 C2G vs C2: value +618 vs +2331 USD, recall 0.07 vs 0.04, AUC 0.61 vs 0.51, cut-winner 0.47 vs 0.43 → MIXED
- Fitness variant: chosen on wf2022 = `None`, wf2023 → **REFUSED_FAILED_SANITY**

## Sanity gate (FAILED_SANITY units are excluded from every verdict; never shown as results)

| unit | sanity | causes |
| --- | --- | --- |
| WF_C2G_wf2022_zz_feat_c2g | **FAILED_SANITY** | {'below_2_trades_per_week': 3, 'one_side_over_90pct': 2} |
| WF_C2G_wf2023_zz_feat_c2g | **FAILED_SANITY** | {'below_2_trades_per_week': 2} |
| WF_wf2022_zz_hold | **PASS** | — |
| WF_wf2023_zz_hold | **FAILED_SANITY** | {'below_2_trades_per_week': 3, 'one_side_over_90pct': 1} |
| WF_C2_wf2022_zz_feat_c2 | **FAILED_SANITY** | {'below_2_trades_per_week': 2, 'one_side_over_90pct': 2} |
| WF_C2_wf2023_zz_feat_c2 | **FAILED_SANITY** | {'below_2_trades_per_week': 4, 'single_exit_reason': 2} |

