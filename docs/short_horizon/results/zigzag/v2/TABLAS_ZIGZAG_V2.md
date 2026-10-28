# ZigZag NSGA-II v2 — tables (generated)

DEVELOPMENT. 2020-2022 decision; 2023 JOINT development (it chooses the expert); 2024 PREVIOUSLY OBSERVED development (read in v1; read once more here per chosen point, see READ_LEDGER.jsonl). pip = 0,0001; 1 pip = 10 MetaTrader points. Executable path: decision after the completed bar, fill at the next open, no entry in the first 12 h after a session reopens.

## Convergence (hypervolume of the first front per generation; converged = < 1 % change over the last 20 generations after ≥ 80)

| Optimization | Seed | Generations | Stop | HV gen 0 | HV final | Δ last 20 gens | Evaluations | Simulations |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| A1 | 1 | 80 | CONVERGED | 2.327,2 | 2.327,3 | 0,000 % | 5184 | 2014 |
| A1 | 2 | 80 | CONVERGED | 2.326,9 | 2.327,3 | 0,000 % | 5184 | 2699 |
| A1 | 3 | 80 | CONVERGED | 2.326,9 | 2.327,3 | 0,000 % | 5184 | 2683 |
| A2 | 1 | 80 | CONVERGED | 2.473,0 | 2.477,9 | 0,000 % | 5184 | 2218 |
| A2 | 2 | 80 | CONVERGED | 2.473,0 | 2.477,9 | 0,000 % | 5184 | 2106 |
| A2 | 3 | 80 | CONVERGED | 2.473,0 | 2.477,9 | 0,000 % | 5184 | 1950 |
| A4 | 1 | 80 | CONVERGED | 2.596,9 | 2.597,2 | 0,000 % | 5184 | 1556 |
| A4 | 2 | 80 | CONVERGED | 2.596,9 | 2.597,2 | 0,000 % | 5184 | 858 |
| A4 | 3 | 80 | CONVERGED | 2.596,9 | 2.597,2 | 0,000 % | 5184 | 1566 |
| B4 | 1 | 80 | CONVERGED | 536.961,8 | 700.782,6 | 0,000 % | 5184 | 3969 |
| B4 | 2 | 80 | CONVERGED | 299.604,5 | 646.671,9 | 0,000 % | 5184 | 3016 |
| B4 | 3 | 80 | CONVERGED | 182.715,6 | 646.671,9 | 0,000 % | 5184 | 3362 |
| A4_wide | 1 | 80 | CONVERGED | 2.596,9 | 2.635,9 | 0,000 % | 5184 | 2307 |
| A4_wide | 2 | 80 | CONVERGED | 2.596,9 | 2.635,9 | 0,000 % | 5184 | 2170 |
| A4_wide | 3 | 80 | CONVERGED | 2.596,9 | 2.635,9 | 0,000 % | 5184 | 1958 |
| A2_wide | 1 | 80 | CONVERGED | 2.473,0 | 2.477,9 | 0,000 % | 5184 | 2755 |
| A2_wide | 2 | 80 | CONVERGED | 2.473,0 | 2.477,9 | 0,000 % | 5184 | 2307 |
| A2_wide | 3 | 80 | CONVERGED | 2.473,0 | 2.477,9 | 0,000 % | 5184 | 2417 |
| B4_wide | 1 | 80 | CONVERGED | 341.335,8 | 702.977,8 | 0,000 % | 5184 | 3149 |
| B4_wide | 2 | 80 | CONVERGED | 326.456,2 | 705.158,7 | 0,007 % | 5184 | 4470 |
| B4_wide | 3 | 80 | CONVERGED | 520.983,3 | 647.121,6 | 0,007 % | 5184 | 3186 |

## Front A1 (union of 3 seeds, nondominated on 2020-2022; 5 points; bound hits: 0)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | 2023 Sharpe | 2023 orders/wk | 2023 valid | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|
| D 6 / Dev 1,1 pips (11 pts) / B 5 | 84 (1,1–241,5) | 13,99 | 5,56 | 4,2 | 13,97 | 3,8 | sí | — |
| D 5 / Dev 1,2 pips (12 pts) / B 4 | 5 (1,2–211,8) | 13,89 | 5,55 | 4,5 | 13,99 | 4,0 | sí | — |
| D 6 / Dev 19,6 pips (196 pts) / B 3 | 6 (19,6–122,6) | 13,74 | 4,25 | 4,4 | 14,19 | 4,0 | sí | — |
| D 5 / Dev 13,7 pips (137 pts) / B 3 | 5 (13,7–201,5) | 13,66 | 3,82 | 4,6 | 14,18 | 4,2 | sí | — |
| D 3 / Dev 1,5 pips (15 pts) / B 2 | 41 (1,5–203,2) | 11,99 | 2,90 | 4,9 | 13,28 | 4,7 | sí | — |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 6 / Dev 19,6 pips (196 pts) / B 3** — 2020-2022 Sharpe 13,74, 2023 Sharpe 14,19; 2024 (previously observed development, single read): Sharpe 14,48, max DD 3,21 %, 4,6 orders/wk, valid sí, 1:10 / 1:50 Sharpe 14,53 / 14,51.

## Front A2 (union of 3 seeds, nondominated on 2020-2022; 2 points; bound hits: 0)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | 2023 Sharpe | 2023 orders/wk | 2023 valid | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|
| D 5 / Dev 1,1 pips (11 pts) / B 3 | 5 (1,1–222,6) | 15,77 | 6,16 | 8,2 | 16,98 | 7,3 | sí | — |
| D 4 / Dev 1,0 pips (10 pts) / B 3 | 121 (1,0–299,5) | 15,67 | 3,82 | 8,8 | 16,62 | 7,5 | sí | — |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 5 / Dev 1,1 pips (11 pts) / B 3** — 2020-2022 Sharpe 15,77, 2023 Sharpe 16,98; 2024 (previously observed development, single read): Sharpe 17,24, max DD 3,21 %, 8,3 orders/wk, valid sí, 1:10 / 1:50 Sharpe 17,33 / 17,29.

## Front A4 (union of 3 seeds, nondominated on 2020-2022; 3 points; bound hits: 2)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | 2023 Sharpe | 2023 orders/wk | 2023 valid | (b) k_tp | (b) Sharpe 2020-22 | (b) lag median | (b) Sharpe 2023 | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D 4 / Dev 6,3 pips (63 pts) / B 1 | 6 (6,3–209,1) | 17,25 | 6,76 | 14,5 | 18,06 | 12,6 | sí | 2,0 | -1,96 | 4,0 | -1,04 | backstep=lower |
| D 5 / Dev 1,5 pips (15 pts) / B 2 | 4 (1,5–3,0) | 16,83 | 6,76 | 12,1 | 17,87 | 9,9 | sí | 1,0 | -1,40 | 6,0 | -2,51 | — |
| D 2 / Dev 1,0 pips (10 pts) / B 1 | 86 (1,0–294,5) | 15,61 | 4,55 | 18,4 | 17,54 | 17,3 | sí | 2,0 | -3,73 | 3,0 | -2,92 | depth=lower, backstep=lower |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 4 / Dev 6,3 pips (63 pts) / B 1** — 2020-2022 Sharpe 17,25, 2023 Sharpe 18,06; 2024 (previously observed development, single read): Sharpe 18,62, max DD 3,21 %, 14,6 orders/wk, valid sí, 1:10 / 1:50 Sharpe 18,79 / 18,72.

## Front B4 (union of 3 seeds, nondominated on 2020-2022; 11 points; bound hits: 2)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | Dec. lag median | 2023 Sharpe | 2023 orders/wk | 2023 valid | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|---|
| D 15 / Dev 7,6 pips (76 pts) / B 14 / k_tp 0,42 | 2 (7,6–266,1) | 0,61 | 33,58 | 3,8 | 33,0 | -2,20 | 3,2 | sí | — |
| D 3 / Dev 1,0 pips (10 pts) / B 2 / k_tp 2,51 | 29 (1,0–239,2) | 0,36 | 41,11 | 3,4 | 5,0 | -1,43 | 1,0 | no | — |
| D 3 / Dev 1,1 pips (11 pts) / B 2 / k_tp 2,51 | 6 (1,1–212,2) | 0,32 | 40,98 | 3,4 | 5,0 | -1,08 | 0,9 | no | — |
| D 3 / Dev 1,0 pips (10 pts) / B 2 / k_tp 2,51 | 6 (1,0–233,4) | 0,26 | 40,97 | 3,3 | 5,0 | -1,08 | 0,8 | no | — |
| D 3 / Dev 1,2 pips (12 pts) / B 2 / k_tp 2,50 | 6 (1,2–38,2) | 0,24 | 40,37 | 3,7 | 5,0 | -0,09 | 1,5 | no | — |
| D 3 / Dev 2,8 pips (28 pts) / B 2 / k_tp 2,50 | 4 (2,8–233,4) | 0,24 | 40,37 | 3,7 | 5,0 | -0,30 | 1,5 | no | — |
| D 3 / Dev 1,1 pips (11 pts) / B 2 / k_tp 2,51 | 6 (1,1–233,4) | 0,21 | 40,28 | 3,5 | 5,0 | -0,88 | 1,1 | no | — |
| D 4 / Dev 16,0 pips (160 pts) / B 3 / k_tp 2,50 | 4 (16,0–210,7) | -0,04 | 37,23 | 3,0 | 8,0 | -0,64 | 1,2 | no | — |
| D 4 / Dev 17,4 pips (174 pts) / B 3 / k_tp 2,50 | 5 (17,4–247,6) | -0,05 | 37,23 | 3,0 | 8,0 | -0,64 | 1,2 | no | — |
| D 3 / Dev 3,6 pips (36 pts) / B 1 / k_tp 2,52 | 55 (3,6–251,3) | -0,29 | 41,82 | 3,1 | 3,0 | -0,66 | 0,8 | no | backstep=lower |
| D 3 / Dev 6,3 pips (63 pts) / B 1 / k_tp 2,52 | 6 (6,3–254,2) | -0,50 | 40,98 | 3,0 | 3,0 | -0,44 | 0,8 | no | backstep=lower |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 15 / Dev 7,6 pips (76 pts) / B 14 / k_tp 0,42** — 2020-2022 Sharpe 0,61, 2023 Sharpe -2,20; 2024 (previously observed development, single read): Sharpe -1,38, max DD 53,62 %, 4,0 orders/wk, valid sí.

## Front A4_wide (union of 3 seeds, nondominated on 2020-2022; 4 points; bound hits: 1)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | 2023 Sharpe | 2023 orders/wk | 2023 valid | (b) k_tp | (b) Sharpe 2020-22 | (b) lag median | (b) Sharpe 2023 | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D 3 / Dev 40,7 pips (407 pts) / B 0 | 6 (40,7–493,7) | 17,67 | 7,21 | 16,3 | 18,23 | 14,6 | sí | 1,5 | -1,70 | 2,0 | -1,65 | backstep=lower |
| D 4 / Dev 1,0 pips (10 pts) / B 1 | 5 (1,0–34,4) | 17,25 | 6,76 | 14,5 | 18,06 | 12,6 | sí | 2,0 | -1,96 | 4,0 | -1,04 | — |
| D 5 / Dev 2,3 pips (23 pts) / B 2 | 5 (2,3–336,4) | 16,83 | 6,76 | 12,1 | 17,87 | 9,9 | sí | 1,0 | -1,40 | 6,0 | -2,51 | — |
| D 2 / Dev 0,5 pips (5 pts) / B 1 | 102 (0,5–588,7) | 15,61 | 4,55 | 18,4 | 17,54 | 17,3 | sí | 2,0 | -3,73 | 3,0 | -2,92 | — |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 3 / Dev 40,7 pips (407 pts) / B 0** — 2020-2022 Sharpe 17,67, 2023 Sharpe 18,23; 2024 (previously observed development, single read): Sharpe 18,16, max DD 2,35 %, 16,4 orders/wk, valid sí, 1:10 / 1:50 Sharpe 18,34 / 18,26.

## Front A2_wide (union of 3 seeds, nondominated on 2020-2022; 2 points; bound hits: 0)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | 2023 Sharpe | 2023 orders/wk | 2023 valid | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|
| D 5 / Dev 11,7 pips (117 pts) / B 3 | 6 (11,7–529,6) | 15,77 | 6,16 | 8,2 | 16,98 | 7,3 | sí | — |
| D 4 / Dev 0,5 pips (5 pts) / B 3 | 119 (0,5–598,8) | 15,67 | 3,82 | 8,8 | 16,62 | 7,5 | sí | — |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 5 / Dev 11,7 pips (117 pts) / B 3** — 2020-2022 Sharpe 15,77, 2023 Sharpe 16,98; 2024 (previously observed development, single read): Sharpe 17,24, max DD 3,21 %, 8,3 orders/wk, valid sí, 1:10 / 1:50 Sharpe 17,33 / 17,29.

## Front B4_wide (union of 3 seeds, nondominated on 2020-2022; 20 points; bound hits: 6)

| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk | Dec. lag median | 2023 Sharpe | 2023 orders/wk | 2023 valid | On bound (all tied sets) |
|---|---|---|---|---|---|---|---|---|---|
| D 15 / Dev 15,6 pips (156 pts) / B 14 / k_tp 0,43 | 2 (15,6–18,9) | 0,65 | 33,45 | 3,8 | 33,0 | -2,45 | 3,2 | sí | — |
| D 20 / Dev 214,6 pips (2.146 pts) / B 5 / k_tp 1,12 | 1 (214,6–214,6) | 0,43 | 55,69 | 3,4 | 20,0 | 0,52 | 3,1 | sí | — |
| D 22 / Dev 197,6 pips (1.976 pts) / B 5 / k_tp 1,11 | 2 (197,6–198,1) | 0,42 | 52,56 | 3,0 | 21,0 | 0,21 | 2,7 | no | — |
| D 20 / Dev 197,6 pips (1.976 pts) / B 5 / k_tp 1,11 | 1 (197,6–197,6) | 0,42 | 54,84 | 3,4 | 20,0 | 0,47 | 3,1 | sí | — |
| D 3 / Dev 0,6 pips (6 pts) / B 2 / k_tp 2,51 | 5 (0,6–434,3) | 0,36 | 41,11 | 3,4 | 5,0 | -1,43 | 1,0 | no | — |
| D 3 / Dev 1,1 pips (11 pts) / B 2 / k_tp 2,51 | 4 (1,1–62,5) | 0,32 | 40,98 | 3,4 | 5,0 | -1,08 | 0,9 | no | — |
| D 3 / Dev 0,8 pips (8 pts) / B 2 / k_tp 2,51 | 5 (0,8–312,5) | 0,26 | 40,97 | 3,3 | 5,0 | -1,08 | 0,8 | no | — |
| D 3 / Dev 0,7 pips (7 pts) / B 2 / k_tp 2,50 | 5 (0,7–304,3) | 0,24 | 40,37 | 3,7 | 5,0 | -0,09 | 1,5 | no | — |
| D 3 / Dev 0,7 pips (7 pts) / B 2 / k_tp 2,50 | 4 (0,7–525,4) | 0,24 | 40,37 | 3,7 | 5,0 | -0,30 | 1,5 | no | — |
| D 3 / Dev 57,7 pips (577 pts) / B 2 / k_tp 2,51 | 2 (57,7–250,0) | 0,21 | 40,28 | 3,5 | 5,0 | -0,88 | 1,1 | no | — |
| D 4 / Dev 0,5 pips (5 pts) / B 3 / k_tp 2,50 | 4 (0,5–454,4) | -0,04 | 37,23 | 3,0 | 8,0 | -0,64 | 1,2 | no | — |
| D 4 / Dev 312,5 pips (3.125 pts) / B 3 / k_tp 2,49 | 1 (312,5–312,5) | -0,04 | 37,06 | 3,2 | 8,0 | -0,24 | 1,3 | no | — |
| D 3 / Dev 0,5 pips (5 pts) / B 1 / k_tp 2,52 | 3 (0,5–2,3) | -0,29 | 41,82 | 3,1 | 3,0 | -0,66 | 0,8 | no | — |
| D 3 / Dev 0,5 pips (5 pts) / B 1 / k_tp 2,52 | 5 (0,5–184,1) | -0,50 | 40,98 | 3,0 | 3,0 | -0,44 | 0,8 | no | — |
| D 3 / Dev 0,6 pips (6 pts) / B 0 / k_tp 2,47 | 3 (0,6–47,6) | -1,11 | 74,58 | 4,7 | 2,0 | -0,52 | 3,0 | sí | backstep=lower |
| D 3 / Dev 0,8 pips (8 pts) / B 0 / k_tp 2,48 | 2 (0,8–1,2) | -1,13 | 73,35 | 4,5 | 2,0 | -0,72 | 2,6 | no | backstep=lower |
| D 3 / Dev 0,7 pips (7 pts) / B 0 / k_tp 2,52 | 5 (0,7–384,0) | -1,24 | 66,62 | 3,2 | 2,0 | -0,14 | 0,8 | no | backstep=lower |
| D 2 / Dev 0,5 pips (5 pts) / B 0 / k_tp 2,52 | 25 (0,5–496,7) | -1,81 | 79,77 | 3,8 | 1,0 | -0,77 | 1,0 | no | backstep=lower |
| D 2 / Dev 0,7 pips (7 pts) / B 0 / k_tp 2,52 | 5 (0,7–95,9) | -1,94 | 78,35 | 3,7 | 1,0 | -0,70 | 0,9 | no | backstep=lower |
| D 2 / Dev 195,2 pips (1.952 pts) / B 0 / k_tp 2,53 | 2 (195,2–232,8) | -2,01 | 78,17 | 3,0 | 1,0 | -1,42 | 0,8 | no | backstep=lower |

Chosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **D 20 / Dev 214,6 pips (2.146 pts) / B 5 / k_tp 1,12** — 2020-2022 Sharpe 0,43, 2023 Sharpe 0,52; 2024 (previously observed development, single read): Sharpe -1,01, max DD 48,84 %, 3,7 orders/wk, valid sí.

## Fill-path cost: decision after the completed bar, fill at b + 1 vs the pinned v1 path (b + 2)

| Case | Period | k_tp | Sharpe b+1 (v2 rules) | Sharpe b+2 (same rules) | Sharpe pinned v1 path | Net b+1 | Net b+2 | Net v1 path | Trades b+1 / v1 | Refused (reopen) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| B4 chosen | DECISION_2020_2022 | 1,12 | 0,43 | 0,34 | 0,37 | 3.056 | 1.512 | 1.940 | 516 / 545 | 48 |
| B4 chosen | VALIDATION_2023 | 1,12 | 0,52 | 0,58 | 0,55 | 1.395 | 1.710 | 1.609 | 153 / 166 | 20 |
| A4 chosen, its (b) | DECISION_2020_2022 | 1,50 | -1,70 | -1,64 | -1,44 | -9.760 | -9.588 | -9.542 | 2615 / 2334 | 479 |
| A4 chosen, its (b) | VALIDATION_2023 | 1,50 | -1,65 | -1,56 | -1,79 | -6.300 | -5.736 | -6.241 | 789 / 698 | 137 |
| v1 expert, its (b) | DECISION_2020_2022 | 2,00 | -1,38 | -0,90 | -0,89 | -9.671 | -8.781 | -8.882 | 2516 / 2392 | 347 |
| v1 expert, its (b) | VALIDATION_2023 | 2,00 | -3,29 | -3,47 | -3,31 | -8.865 | -8.796 | -8.754 | 736 / 717 | 110 |

## Confirmation lag, v1 definition vs v2 (v1 parameters D3/36,3/B2, 3899 pivots 2020-2022)

| Definition | Median | Mean | p90 |
|---|---:|---:|---:|
| v1_definition_next_final_pivot_plus_backstep | 6,0 | 6,79 | 11,0 |
| v2_knowable_first_opposite_candidate_plus_backstep | 5,0 | 6,07 | 10,0 |

## Demonstrations v2

`demonstrations_v2/expert_trades.csv` — 4185 trades, sha256 `38125b0b368a29b425b4f4ac192fe380b343894fbceedfe17c1e9b85c6564a24`, label **HINDSIGHT_NON_CAUSAL_EXPERT**, params {'depth': 3, 'deviation_pips': 40.7, 'backstep': 0} (407 points); supersedes v1 sha256 352ddf6a55300864e11abe790b8bad396742abe4fd4d2aaef1012127ba544f76 (kept).

