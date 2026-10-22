# Consumer evidence: selection v3 finalists and the matched scaling experiment

E3-M FULL specialists (open-L/open-S/close-L/close-S), head 16, units 2, 3 seeds (20261019/20/21), same full-pool store, dev3y arm (fit 2020-2022 / INNER_A 2023H1 choice / INNER_B 2023H2 report / 2024 read once). Metrics from the native simulator (SL/TP, sizing, costs, weekly rules). `valid` = the lane's activity rule (>= 3 orders/week on average, <= 4/day). Requested vs consumed columns: 0 missing / 0 extra in every arm; INNER_A / INNER_B / 2024 support hashes identical across the finalist arms (TRAIN differs only by window warm-up); the four scaling arms share every support hash (`--eligible-from 2020-02-01`). **CORRECTION (same day):** in that first run the option only entered the support hash, not the rows the agent trained and traded on (`observable()` ignored it), so the fixed arms were identical to the finalist core15 run and the 'shared support' claim did not hold for the model; the four scaling arms were invalidated and rerun with code @7896601 (observable/labelled honour the option); their table below is replaced when the rerun closes.

## Summary (seed means)

| arm | list | cols | scaling | INNER_A Sharpe mean (3 seeds) | all valid | TRAIN net mean USD | INNER_A net | INNER_B net | 2024 net | params FULL |
|---|---|---|---|---|---|---|---|---|---|---|
| final_v3_core15 | core15 | 15 | fixed_train | -1.14 | True | -5,567 | -2,732 | -4,507 | -5,641 | 2236 |
| final_v3_shortlist_strict | shortlist_strict | 125 | fixed_train | -1.13 | True | 3,623,585 | -2,862 | -3,220 | -1,966 | 11036 |
| final_v3_shortlist | shortlist | 173 | fixed_train | -1.96 | True | 4,896,156 | -3,774 | -2,208 | -2,718 | 14876 |
| final_v3_shortlist_strict_enc | shortlist_strict_enc | 137 | fixed_train | -1.38 | True | 8,165,798 | -2,507 | -3,058 | -2,845 | 11996 |

## Matched scaling experiment (RERUN with code @7896601; identical support hashes in all four arms: TRAIN 12f9ce87b1a454ec / INNER_A bddcd9eb9ad1bd88 / INNER_B 78b5834ee53e8707 / 2024 21f3dacd4915644a; rows eligible 14 372 / 2 461 / 2 436 / 4 871)

| arm | list | cols | scaling | INNER_A Sharpe mean (3 seeds) | all valid | TRAIN net mean USD | INNER_A net | INNER_B net | 2024 net | params FULL |
|---|---|---|---|---|---|---|---|---|---|---|
| scale_rolling_z | core15 | 15 | rolling_z | -0.86 | False | 3,431 | -2,249 | -3,822 | -3,832 | 2236 |
| scale_fixed_keep | core15 | 15 | fixed_train | -0.98 | True | 2,576 | -1,656 | -1,940 | -2,919 | 2236 |
| scale_fixed_diff | core15 | 15 | fixed_train | -0.98 | True | 2,576 | -1,656 | -1,940 | -2,919 | 2236 |
| scale_fixed_diff_rel | core15_plus_relatives | 24 | fixed_train | -0.97 | True | 14,347 | -2,445 | -5,665 | -5,797 | 2956 |

## Per seed and period

| arm | seed | period | net USD | trades | orders/week | max DD % | Sharpe | valid | tau |
|---|---|---|---|---|---|---|---|---|---|
| final_v3_core15 | 20261019 | TRAIN | -6,999 | 1587 | 10.1 | 85.9 | -0.47 | True | 0.365 |
| final_v3_core15 | 20261019 | INNER_A | -537 | 251 | 9.7 | 45.9 | 0.09 | True | 0.365 |
| final_v3_core15 | 20261019 | INNER_B | -5,634 | 285 | 10.6 | 59.3 | -2.96 | True | 0.365 |
| final_v3_core15 | 20261019 | OBS_2024 | -4,846 | 525 | 9.9 | 57.6 | -1.51 | True | 0.365 |
| final_v3_core15 | 20261020 | TRAIN | -7,406 | 1501 | 9.6 | 88.0 | -0.51 | True | 0.414 |
| final_v3_core15 | 20261020 | INNER_A | -3,841 | 245 | 9.4 | 51.9 | -1.53 | True | 0.414 |
| final_v3_core15 | 20261020 | INNER_B | -3,021 | 209 | 7.7 | 50.9 | -1.19 | True | 0.414 |
| final_v3_core15 | 20261020 | OBS_2024 | -5,561 | 458 | 8.6 | 63.3 | -2.0 | True | 0.414 |
| final_v3_core15 | 20261021 | TRAIN | -2,295 | 1273 | 8.1 | 86.7 | 0.13 | True | 0.383 |
| final_v3_core15 | 20261021 | INNER_A | -3,817 | 219 | 8.4 | 41.3 | -1.99 | True | 0.383 |
| final_v3_core15 | 20261021 | INNER_B | -4,866 | 213 | 7.9 | 52.0 | -2.69 | True | 0.383 |
| final_v3_core15 | 20261021 | OBS_2024 | -6,514 | 367 | 6.9 | 67.1 | -2.69 | True | 0.383 |
| final_v3_shortlist_strict | 20261019 | TRAIN | 1,285,072 | 864 | 5.5 | 35.9 | 3.39 | True | 0.496 |
| final_v3_shortlist_strict | 20261019 | INNER_A | 140 | 173 | 6.7 | 24.2 | 0.3 | True | 0.496 |
| final_v3_shortlist_strict | 20261019 | INNER_B | -4,293 | 134 | 5.0 | 53.5 | -2.51 | True | 0.496 |
| final_v3_shortlist_strict | 20261019 | OBS_2024 | -1,110 | 289 | 5.5 | 36.5 | -0.15 | True | 0.496 |
| final_v3_shortlist_strict | 20261020 | TRAIN | 2,191,330 | 1501 | 9.6 | 35.6 | 3.2 | True | 0.387 |
| final_v3_shortlist_strict | 20261020 | INNER_A | -6,054 | 272 | 10.5 | 61.1 | -2.7 | True | 0.387 |
| final_v3_shortlist_strict | 20261020 | INNER_B | -3,575 | 261 | 9.7 | 42.3 | -1.49 | True | 0.387 |
| final_v3_shortlist_strict | 20261020 | OBS_2024 | -5,082 | 526 | 9.9 | 61.4 | -1.37 | True | 0.387 |
| final_v3_shortlist_strict | 20261021 | TRAIN | 7,394,353 | 594 | 3.8 | 26.9 | 3.98 | True | 0.537 |
| final_v3_shortlist_strict | 20261021 | INNER_A | -2,672 | 106 | 4.1 | 41.7 | -0.99 | True | 0.537 |
| final_v3_shortlist_strict | 20261021 | INNER_B | -1,793 | 90 | 3.3 | 36.0 | -0.54 | True | 0.537 |
| final_v3_shortlist_strict | 20261021 | OBS_2024 | 294 | 184 | 3.5 | 28.6 | 0.25 | True | 0.537 |
| final_v3_shortlist | 20261019 | TRAIN | 8,427,410 | 725 | 4.6 | 25.8 | 4.44 | True | 0.52 |
| final_v3_shortlist | 20261019 | INNER_A | -4,894 | 104 | 4.0 | 48.9 | -2.49 | True | 0.52 |
| final_v3_shortlist | 20261019 | INNER_B | -1,131 | 108 | 4.0 | 34.5 | -0.42 | True | 0.52 |
| final_v3_shortlist | 20261019 | OBS_2024 | -785 | 249 | 4.7 | 31.3 | -0.04 | True | 0.52 |
| final_v3_shortlist | 20261020 | TRAIN | 1,795,379 | 535 | 3.4 | 20.9 | 4.07 | True |  |
| final_v3_shortlist | 20261020 | INNER_A | -3,278 | 83 | 3.2 | 36.0 | -2.45 | True |  |
| final_v3_shortlist | 20261020 | INNER_B | -1,225 | 76 | 2.8 | 35.3 | -1.01 | False |  |
| final_v3_shortlist | 20261020 | OBS_2024 | -2,758 | 168 | 3.2 | 31.4 | -1.41 | True |  |
| final_v3_shortlist | 20261021 | TRAIN | 4,465,679 | 1309 | 8.3 | 29.3 | 3.51 | True | 0.32 |
| final_v3_shortlist | 20261021 | INNER_A | -3,151 | 220 | 8.5 | 38.3 | -0.94 | True | 0.32 |
| final_v3_shortlist | 20261021 | INNER_B | -4,269 | 197 | 7.3 | 50.9 | -1.93 | True | 0.32 |
| final_v3_shortlist | 20261021 | OBS_2024 | -4,611 | 403 | 7.6 | 52.0 | -1.11 | True | 0.32 |
| final_v3_shortlist_strict_enc | 20261019 | TRAIN | 19,744,792 | 683 | 4.4 | 18.9 | 4.86 | True | 0.583 |
| final_v3_shortlist_strict_enc | 20261019 | INNER_A | -117 | 126 | 4.8 | 38.2 | 0.2 | True | 0.583 |
| final_v3_shortlist_strict_enc | 20261019 | INNER_B | -3,514 | 112 | 4.1 | 40.7 | -1.97 | True | 0.583 |
| final_v3_shortlist_strict_enc | 20261019 | OBS_2024 | -892 | 217 | 4.1 | 38.2 | -0.12 | True | 0.583 |
| final_v3_shortlist_strict_enc | 20261020 | TRAIN | 267,977 | 597 | 3.8 | 23.1 | 2.8 | True |  |
| final_v3_shortlist_strict_enc | 20261020 | INNER_A | -435 | 101 | 3.9 | 22.9 | -0.14 | True |  |
| final_v3_shortlist_strict_enc | 20261020 | INNER_B | -554 | 95 | 3.5 | 13.7 | -0.35 | True |  |
| final_v3_shortlist_strict_enc | 20261020 | OBS_2024 | -4,152 | 224 | 4.2 | 49.6 | -1.93 | True |  |
| final_v3_shortlist_strict_enc | 20261021 | TRAIN | 4,484,624 | 1495 | 9.5 | 37.9 | 3.58 | True | 0.468 |
| final_v3_shortlist_strict_enc | 20261021 | INNER_A | -6,967 | 294 | 11.3 | 73.6 | -4.2 | True | 0.468 |
| final_v3_shortlist_strict_enc | 20261021 | INNER_B | -5,107 | 238 | 8.8 | 58.7 | -2.37 | True | 0.468 |
| final_v3_shortlist_strict_enc | 20261021 | OBS_2024 | -3,490 | 529 | 10.0 | 51.5 | -0.78 | True | 0.468 |

## Reading

- Finalists (dev3y): no list beats core15 out of sample; the larger the list the larger the TRAIN memorization (net of millions on 10 k initial cash) with INNER_A/INNER_B/2024 negative. The selection delivered paired, measured candidates; this consumer (as configured) cannot turn them into out-of-sample advantage — the generalization gap is the problem to attack before any performance sweep.
- Matched scaling (rerun, identical support hashes): rolling-z −0.86 INNER_A (one seed invalid: 7 trades) vs fixed+keep −0.98 vs fixed+diff −0.98 (identical: the stationarity rule differenced nothing in this list, so keep/diff isolates nothing here) vs fixed+diff+explicit relatives −0.97. On the SAME rows and inputs the rolling z-score gives no measurable advantage over the owner's fixed TRAIN scale: the earlier apparent rolling edge (+0.86 / +0.43) came from unmatched supports and inputs, not from the normalization. Explicit relative columns add nothing here either. All arms are negative out of sample: the generalization gap, not the scaling, is the problem.
- Principal-contract (consolidated) arms and the generalization-gap diagnostic run on the peer worker; appended when they close.
- Pre-repair arms (selection v2 lists) are kept under `final_*` for the record and are not comparable (unpaired selection, wrong layer-2 file).
