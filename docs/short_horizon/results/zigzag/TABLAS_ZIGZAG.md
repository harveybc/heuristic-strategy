# ZigZag hindsight-expert study — tables (generated)

DEVELOPMENT. The hindsight ZigZag is NON-CAUSAL (repaints): it is the expert for demonstrations, never a result. Trivial reversal strategy on EUR/USD H1 2020-2024 (lake), rel_volume 0,10, 1:100, costs ×1, SL = 2·σ̂ (causal 24 h vol), ≥ 3 orders/week and ≤ 4/day.

## Pareto front (NSGA-II on 2020-2022, 16 individuals, 15 generations; 16 points, 16 accepted by the frequency rule)

| Depth | Deviation (pips) | Backstep | Pivots | Dec. Sharpe | Dec. max DD % | Dec. orders/wk | Val. 2023 Sharpe | Val. orders/wk | Frequency rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 29,7 | 3 | 4589 | 17,90 | 8,6 | 13,1 | 19,10 | 10,4 | ACCEPTED |
| 3 | 30,6 | 4 | 4507 | 17,68 | 8,6 | 13,0 | 18,45 | 10,6 | ACCEPTED |
| 3 | 28,9 | 4 | 4507 | 17,68 | 8,6 | 13,0 | 18,45 | 10,6 | ACCEPTED |
| 5 | 36,3 | 2 | 4314 | 17,42 | 6,8 | 12,2 | 18,92 | 10,1 | ACCEPTED |
| 5 | 47,3 | 1 | 4536 | 17,33 | 6,8 | 12,3 | 18,45 | 9,9 | ACCEPTED |
| 3 | 30,4 | 2 | 6371 | 17,09 | 6,8 | 14,8 | 19,48 | 13,0 | ACCEPTED |
| 3 | 36,3 | 2 | 6371 | 17,09 | 6,8 | 14,8 | 19,48 | 13,0 | ACCEPTED |
| 4 | 48,0 | 6 | 3077 | 16,91 | 5,9 | 9,3 | 17,74 | 7,6 | ACCEPTED |
| 3 | 31,3 | 6 | 3309 | 16,90 | 5,9 | 9,9 | 18,00 | 8,4 | ACCEPTED |
| 5 | 31,3 | 6 | 2866 | 16,46 | 5,9 | 8,5 | 17,32 | 7,2 | ACCEPTED |
| 5 | 30,6 | 6 | 2866 | 16,46 | 5,9 | 8,5 | 17,32 | 7,2 | ACCEPTED |
| 6 | 41,7 | 6 | 2654 | 15,87 | 5,9 | 7,8 | 16,87 | 6,7 | ACCEPTED |
| 6 | 39,6 | 6 | 2654 | 15,87 | 5,9 | 7,8 | 16,87 | 6,7 | ACCEPTED |
| 8 | 24,1 | 2 | 2818 | 15,50 | 5,0 | 8,2 | 17,04 | 7,2 | ACCEPTED |
| 8 | 36,5 | 3 | 2726 | 15,38 | 4,9 | 7,9 | 16,85 | 7,0 | ACCEPTED |
| 8 | 30,6 | 3 | 2726 | 15,38 | 4,9 | 7,9 | 16,85 | 7,0 | ACCEPTED |

Chosen (best 2023 Sharpe among accepted): Depth 3, Deviation 36,3 pips, Backstep 2. Confirmation 2024 (hindsight expert): Sharpe 18,91, max DD 3,2 %, 14,7 orders/wk, TP share 0,97. Leverage 1:10 / 1:50 Sharpe: 19,04 / 18,99.

## Hindsight (a) vs causal-confirmed (b): the confirmation-lag gap

| Params | Lag median / mean / p90 (bars) | k_tp | Period | (a) Sharpe | (a) orders/wk | (b) Sharpe | (b) orders/wk | (b) TP / SL / time | Gap (a)−(b) |
|---|---|---:|---|---:|---:|---:|---:|---|---:|
| D3_dev36.3_B2 (chosen) | 6 / 6,9 / 12 | 2,0 | DECISION_2020_2022 | 17,09 | 14,8 | -1,03 | 15,2 | 0,00 / 0,01 / 0,98 | 18,12 |
| D3_dev36.3_B2 (chosen) | 6 / 6,9 / 12 | 2,0 | VALIDATION_2023 | 19,48 | 13,0 | -2,14 | 13,3 | 0,01 / 0,01 / 0,99 | 21,62 |
| D3_dev36.3_B2 (chosen) | 6 / 6,9 / 12 | 2,0 | CONFIRM_2024 | 18,91 | 14,7 | -2,16 | 15,1 | 0,00 / 0,01 / 0,99 | 21,07 |
| D4_dev29.7_B3 | 8 / 9,8 / 17 | 2,0 | DECISION_2020_2022 | 17,90 | 13,1 | -1,45 | 13,4 | 0,00 / 0,02 / 0,98 | 19,36 |
| D4_dev29.7_B3 | 8 / 9,8 / 17 | 2,0 | VALIDATION_2023 | 19,10 | 10,4 | -3,35 | 10,8 | 0,01 / 0,02 / 0,98 | 22,45 |
| D4_dev29.7_B3 | 8 / 9,8 / 17 | 2,0 | CONFIRM_2024 | 19,25 | 12,8 | -2,57 | 13,0 | 0,01 / 0,02 / 0,97 | 21,81 |
| D3_dev30.6_B4 | 9 / 10,9 / 18 | 2,0 | DECISION_2020_2022 | 17,68 | 13,0 | -0,38 | 13,4 | 0,01 / 0,01 / 0,98 | 18,06 |
| D3_dev30.6_B4 | 9 / 10,9 / 18 | 2,0 | VALIDATION_2023 | 18,45 | 10,6 | -1,23 | 11,1 | 0,01 / 0,02 / 0,98 | 19,68 |
| D3_dev30.6_B4 | 9 / 10,9 / 18 | 2,0 | CONFIRM_2024 | 18,92 | 12,3 | -1,31 | 12,6 | 0,01 / 0,02 / 0,97 | 20,23 |

k_tp chosen on 2020-2022 per parameter set: D3_dev36.3_B2: 0.5 → -1,41, 1.0 → -1,12, 1.5 → -1,13, 2.0 → -1,03; D4_dev29.7_B3: 0.5 → -2,11, 1.0 → -1,70, 1.5 → -1,51, 2.0 → -1,45; D3_dev30.6_B4: 0.5 → -1,75, 1.0 → -0,72, 1.5 → -0,47, 2.0 → -0,38

## Timing-noise study (chosen params; hindsight TP; mean over seeds)

| Period | Delay d | Jitter ± | Timing error (bars) | Sharpe | Orders/wk | Valid |
|---|---:|---:|---:|---:|---:|---|
| CONFIRM_2024 | 0 | 0 | 0,00 | 18,91 | 14,7 | sí |
| CONFIRM_2024 | 0 | 1 | 0,64 | 13,46 | 10,3 | sí |
| CONFIRM_2024 | 0 | 2 | 1,09 | 10,90 | 8,8 | sí |
| CONFIRM_2024 | 0 | 4 | 1,79 | 7,71 | 7,7 | sí |
| CONFIRM_2024 | 1 | 0 | 1,00 | 15,24 | 15,3 | sí |
| CONFIRM_2024 | 1 | 1 | 1,00 | 16,45 | 13,9 | sí |
| CONFIRM_2024 | 1 | 2 | 1,34 | 13,13 | 10,5 | sí |
| CONFIRM_2024 | 1 | 4 | 1,95 | 9,44 | 8,2 | sí |
| CONFIRM_2024 | 2 | 0 | 2,00 | 13,26 | 13,4 | sí |
| CONFIRM_2024 | 2 | 1 | 2,00 | 13,00 | 13,2 | sí |
| CONFIRM_2024 | 2 | 2 | 2,01 | 13,86 | 12,1 | sí |
| CONFIRM_2024 | 2 | 4 | 2,41 | 10,50 | 8,5 | sí |
| CONFIRM_2024 | 3 | 0 | 3,00 | 11,21 | 11,3 | sí |
| CONFIRM_2024 | 3 | 1 | 3,01 | 11,72 | 11,0 | sí |
| CONFIRM_2024 | 3 | 2 | 3,01 | 11,78 | 11,0 | sí |
| CONFIRM_2024 | 3 | 4 | 3,13 | 11,28 | 9,2 | sí |
| CONFIRM_2024 | 5 | 0 | 5,00 | 8,90 | 9,3 | sí |
| CONFIRM_2024 | 5 | 1 | 5,00 | 8,77 | 8,9 | sí |
| CONFIRM_2024 | 5 | 2 | 4,99 | 8,66 | 8,8 | sí |
| CONFIRM_2024 | 5 | 4 | 5,03 | 8,21 | 9,1 | sí |
| CONFIRM_2024 | 8 | 0 | 8,00 | 5,77 | 8,9 | sí |
| CONFIRM_2024 | 8 | 1 | 8,00 | 7,05 | 8,2 | sí |
| CONFIRM_2024 | 8 | 2 | 7,99 | 7,64 | 7,9 | sí |
| CONFIRM_2024 | 8 | 4 | 7,99 | 7,37 | 7,5 | sí |
| CONFIRM_2024 | 12 | 0 | 12,00 | 5,93 | 8,8 | sí |
| CONFIRM_2024 | 12 | 1 | 12,01 | 7,19 | 8,2 | sí |
| CONFIRM_2024 | 12 | 2 | 12,01 | 7,22 | 8,2 | sí |
| CONFIRM_2024 | 12 | 4 | 11,99 | 7,32 | 7,8 | sí |
| VALIDATION_2023 | 0 | 0 | 0,00 | 19,48 | 13,0 | sí |
| VALIDATION_2023 | 0 | 1 | 0,64 | 13,42 | 8,8 | sí |
| VALIDATION_2023 | 0 | 2 | 1,09 | 10,50 | 7,5 | sí |
| VALIDATION_2023 | 0 | 4 | 1,79 | 8,60 | 6,9 | sí |
| VALIDATION_2023 | 1 | 0 | 1,00 | 16,32 | 13,6 | sí |
| VALIDATION_2023 | 1 | 1 | 1,00 | 16,94 | 12,2 | sí |
| VALIDATION_2023 | 1 | 2 | 1,34 | 12,78 | 9,3 | sí |
| VALIDATION_2023 | 1 | 4 | 1,95 | 9,43 | 7,1 | sí |
| VALIDATION_2023 | 2 | 0 | 2,00 | 14,49 | 11,8 | sí |
| VALIDATION_2023 | 2 | 1 | 2,00 | 14,15 | 11,5 | sí |
| VALIDATION_2023 | 2 | 2 | 2,01 | 14,42 | 10,7 | sí |
| VALIDATION_2023 | 2 | 4 | 2,41 | 9,50 | 7,6 | sí |
| VALIDATION_2023 | 3 | 0 | 3,00 | 11,79 | 9,8 | sí |
| VALIDATION_2023 | 3 | 1 | 3,01 | 12,45 | 9,6 | sí |
| VALIDATION_2023 | 3 | 2 | 3,01 | 11,99 | 9,8 | sí |
| VALIDATION_2023 | 3 | 4 | 3,13 | 11,10 | 8,0 | sí |
| VALIDATION_2023 | 5 | 0 | 5,00 | 9,50 | 8,4 | sí |
| VALIDATION_2023 | 5 | 1 | 5,00 | 9,52 | 7,9 | sí |
| VALIDATION_2023 | 5 | 2 | 4,99 | 9,34 | 7,9 | sí |
| VALIDATION_2023 | 5 | 4 | 5,03 | 9,68 | 8,1 | sí |
| VALIDATION_2023 | 8 | 0 | 8,00 | 7,84 | 7,8 | sí |
| VALIDATION_2023 | 8 | 1 | 8,00 | 7,58 | 7,4 | sí |
| VALIDATION_2023 | 8 | 2 | 7,99 | 7,26 | 7,1 | sí |
| VALIDATION_2023 | 8 | 4 | 7,99 | 7,59 | 6,9 | sí |
| VALIDATION_2023 | 12 | 0 | 12,00 | 6,55 | 7,5 | sí |
| VALIDATION_2023 | 12 | 1 | 12,01 | 7,79 | 7,2 | sí |
| VALIDATION_2023 | 12 | 2 | 12,01 | 7,54 | 6,9 | sí |
| VALIDATION_2023 | 12 | 4 | 11,99 | 7,61 | 6,6 | sí |

## Learned causal pivot detector (MLP 64-32, class-weighted CE, 30 epochs; trained 2020-2022, chosen 2023, confirmed 2024)

Features: 19 causal lake features (SUBSTITUTE_CAUSAL_LAKE_FEATURES_NOT_MUSASHI_MANIFEST). Device NVIDIA GeForce RTX 4070 Laptop GPU, peak VRAM 35 MiB, training wall 5,7 s. Predictions sealed sha256 `79fddac0094801df…`.

| k | Threshold | Period | Signals | Precision | AUC valley / peak | Log-loss (base rate) | Sharpe | Orders/wk | Valid |
|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2 | 0,4 | DECISION_2020_2022 | 7791 | 0,311 | 0,688 / 0,691 | 1,039 (0,948) | -4,17 | 19,7 | sí |
| 2 | 0,4 | VALIDATION_2023 | 2322 | 0,287 | 0,664 / 0,664 | 1,073 (0,924) | -5,10 | 19,2 | sí |
| 2 | 0,4 | CONFIRM_2024 | 2582 | 0,302 | 0,671 / 0,673 | 1,061 (0,941) | -6,95 | 19,4 | sí |
| 2 | 0,5 | DECISION_2020_2022 | 4950 | 0,350 | 0,688 / 0,691 | 1,039 (0,948) | -2,29 | 17,8 | sí |
| 2 | 0,5 | VALIDATION_2023 | 1515 | 0,305 | 0,664 / 0,664 | 1,073 (0,924) | -4,24 | 17,6 | sí |
| 2 | 0,5 | CONFIRM_2024 | 1573 | 0,352 | 0,671 / 0,673 | 1,061 (0,941) | -3,72 | 17,3 | sí |
| 2 | 0,6 | DECISION_2020_2022 | 1767 | 0,425 | 0,688 / 0,691 | 1,039 (0,948) | -0,11 | 9,6 | sí |
| 2 | 0,6 | VALIDATION_2023 | 580 | 0,360 | 0,664 / 0,664 | 1,073 (0,924) | -0,35 | 9,5 | sí |
| 2 | 0,6 | CONFIRM_2024 | 539 | 0,397 | 0,671 / 0,673 | 1,061 (0,941) | -0,61 | 8,7 | sí |
| 2 | 0,7 | DECISION_2020_2022 | 252 | 0,504 | 0,688 / 0,691 | 1,039 (0,948) | -0,33 | 1,9 | no |
| 2 | 0,7 | VALIDATION_2023 | 97 | 0,381 | 0,664 / 0,664 | 1,073 (0,924) | -0,71 | 1,9 | no |
| 2 | 0,7 | CONFIRM_2024 | 79 | 0,532 | 0,671 / 0,673 | 1,061 (0,941) | 0,85 | 1,7 | no |
| 4 | 0,4 | DECISION_2020_2022 | 2883 | 0,466 | 0,621 / 0,626 | 1,053 (1,098) | -1,01 | 15,1 | sí |
| 4 | 0,4 | VALIDATION_2023 | 867 | 0,373 | 0,571 / 0,564 | 1,099 (1,097) | -4,44 | 14,2 | sí |
| 4 | 0,4 | CONFIRM_2024 | 943 | 0,391 | 0,566 / 0,564 | 1,100 (1,098) | -1,97 | 14,9 | sí |
| 4 | 0,5 | DECISION_2020_2022 | 501 | 0,527 | 0,621 / 0,626 | 1,053 (1,098) | 0,41 | 3,1 | sí |
| 4 | 0,5 | VALIDATION_2023 | 174 | 0,385 | 0,571 / 0,564 | 1,099 (1,097) | -0,73 | 2,8 | no |
| 4 | 0,5 | CONFIRM_2024 | 137 | 0,482 | 0,566 / 0,564 | 1,100 (1,098) | -0,16 | 2,5 | no |
| 4 | 0,6 | DECISION_2020_2022 | 26 | 0,538 | 0,621 / 0,626 | 1,053 (1,098) | -0,34 | 1,4 | no |
| 4 | 0,6 | VALIDATION_2023 | 13 | 0,385 | 0,571 / 0,564 | 1,099 (1,097) | 0,23 | 1,1 | no |
| 4 | 0,6 | CONFIRM_2024 | 2 | 1,000 | 0,566 / 0,564 | 1,100 (1,098) | -0,89 | 1,0 | no |
| 4 | 0,7 | DECISION_2020_2022 | 2 | 0,000 | 0,621 / 0,626 | 1,053 (1,098) | -1,00 | 1,0 | no |
| 4 | 0,7 | VALIDATION_2023 | 0 | — | 0,571 / 0,564 | 1,099 (1,097) | 0,00 | 0,0 | no |
| 4 | 0,7 | CONFIRM_2024 | 0 | — | 0,566 / 0,564 | 1,100 (1,098) | 0,00 | 0,0 | no |

## Demonstration dataset

`demonstrations/expert_trades.csv` — 3791 expert trades 2020-2024, sha256 `352ddf6a55300864e11abe790b8bad396742abe4fd4d2aaef1012127ba544f76`, label **HINDSIGHT_NON_CAUSAL_EXPERT**, params {'depth': 3, 'deviation_pips': 36.3, 'backstep': 2}, data sha256 `c6628375b2872f16…`.

