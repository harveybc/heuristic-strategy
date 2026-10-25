# Early-close audit, baseline (walk-forward validation years; A arms on INNER_A, biased by D1, reference only)

PRE-REGISTERED 2026-10-25 before any audit number was read: a selectable strategy must close early (agent closes > 0 on the calibration block), otherwise DEGENERATE_NO_EARLY_CLOSE and not selectable (owner's business rule); the early-close value is a REQUIRED reported metric for every arm; a closer whose early-close value 95 % CI is entirely < 0 is HARMFUL_EARLY_CLOSE (it cuts winners): a defect to fix, not a reason to remove early closing

Columns: trades; early-close share (agent closes + reversals / exits); SAVED / CUT_WINNER / NEUTRAL shares of the agent closes; early-close value = sum(actual net - engine counterfactual hold net) USD [bootstrap 95 % CI over trades]; the same in sigma; Sharpe actual / no-early-close policy (same frozen policy, voluntary exits -> HOLD, engine replay); closer exit-probability AUC vs SL_FIRST / vs CLOSE_BETTER on every positioned observable bar; precision / recall of the fired closes vs SL_FIRST; long / short entries; long / short net USD; flags. Every replay reproduces its run exactly (asserted).

| unit | policy | period | trades | early close | SAVED / CUT / NEUTRAL % | value USD [95 % CI] | value σ | Sharpe actual / no early close | AUC SL-first / close-better | precision / recall | long / short entries | long / short net | flags |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| A_both_hold | ensemble | INNER_A | 73 | 71 % | 71 / 29 / 0 | 2932 [649, 5176] | 14.3 | 0.19 / -3.00 | 0.69 / 0.67 | 0.73 / 0.24 | 37 / 36 | 251 / -169 | — |
| A_both_hold | seed 20261019 | INNER_A | 79 | 85 % | 54 / 46 / 0 | -61 [-2786, 2749] | 1.5 | 2.86 / 2.24 | 0.59 / 0.59 | 0.54 / 0.22 | 37 / 42 | 1685 / 252 | — |
| A_both_hold | seed 20261020 | INNER_A | 54 | 80 % | 65 / 35 / 0 | 1311 [-738, 3213] | 6.5 | 0.02 / -1.43 | 0.65 / 0.63 | 0.67 / 0.25 | 21 / 33 | -184 / 167 | — |
| A_both_hold | seed 20261021 | INNER_A | 79 | 81 % | 67 / 31 / 2 | 3472 [680, 6109] | 16.8 | 1.55 / -1.46 | 0.58 / 0.57 | 0.69 / 0.27 | 49 / 30 | 256 / 668 | — |
| A_both_hold | seed 20261022 | INNER_A | 78 | 68 % | 57 / 42 / 2 | 711 [-1747, 3229] | 3.9 | 0.64 / 0.24 | 0.69 / 0.69 | 0.58 / 0.22 | 37 / 41 | 431 / -3 | — |
| A_both_hold | seed 20261023 | INNER_A | 59 | 75 % | 66 / 34 / 0 | 1307 [-581, 3240] | 6.6 | -0.65 / -1.79 | 0.75 / 0.74 | 0.68 / 0.29 | 33 / 26 | -403 / 23 | — |
| A_zz_hold | ensemble | INNER_A | 90 | 79 % | 41 / 55 / 4 | -370 [-4052, 3402] | -0.0 | 0.96 / -0.17 | 0.70 / 0.71 | 0.44 / 0.37 | 43 / 47 | 491 / 157 | — |
| A_zz_hold | seed 20261019 | INNER_A | 92 | 84 % | 36 / 61 / 3 | -2273 [-6174, 1457] | -9.8 | 0.42 / 0.63 | 0.74 / 0.70 | 0.40 / 0.44 | 47 / 45 | -572 / 792 | — |
| A_zz_hold | seed 20261020 | INNER_A | 69 | 83 % | 33 / 63 / 4 | -2485 [-5932, 1066] | -10.4 | 2.51 / 0.57 | 0.59 / 0.57 | 0.37 / 0.24 | 25 / 44 | 814 / 767 | — |
| A_zz_hold | seed 20261021 | INNER_A | 100 | 79 % | 51 / 46 / 4 | 2274 [-1819, 6277] | 11.6 | 0.48 / -1.51 | 0.59 / 0.56 | 0.53 / 0.19 | 54 / 46 | 636 / -328 | — |
| WF_wf2022_both_hold | ensemble | INNER_B | 124 | 52 % | 56 / 44 / 0 | -905 [-5080, 3344] | -2.1 | 0.60 / 0.68 | 0.64 / 0.62 | 0.56 / 0.10 | 75 / 49 | -1123 / 2273 | — |
| WF_wf2022_both_hold | seed 20261019 | INNER_B | 173 | 69 % | 52 / 46 / 2 | -3535 [-8654, 1838] | -9.0 | 0.92 / 0.89 | 0.60 / 0.59 | 0.53 / 0.17 | 98 / 75 | 2108 / -239 | — |
| WF_wf2022_both_hold | seed 20261020 | INNER_B | 146 | 62 % | 54 / 44 / 1 | 70 [-4334, 4629] | -0.7 | 0.04 / -0.08 | 0.60 / 0.59 | 0.52 / 0.17 | 79 / 67 | -1705 / 1580 | — |
| WF_wf2022_both_hold | seed 20261021 | INNER_B | 120 | 32 % | 47 / 53 / 0 | -784 [-3322, 1837] | -3.6 | -0.31 / 0.17 | 0.47 / 0.49 | 0.47 / 0.03 | 59 / 61 | -1412 / 242 | — |
| WF_wf2022_both_hold | seed 20261022 | INNER_B | 112 | 0 % | — / — / — | 0 [—, —] | 0.0 | -0.65 / -0.65 | 0.55 / 0.54 | — / 0.00 | 60 / 52 | -801 / -1190 | DEGENERATE_NO_EARLY_CLOSE |
| WF_wf2022_both_hold | seed 20261023 | INNER_B | 153 | 61 % | 62 / 38 / 0 | 1275 [-2853, 5287] | 7.0 | -0.55 / -0.63 | 0.58 / 0.57 | 0.61 / 0.09 | 80 / 73 | -81 / -1388 | — |
| WF_wf2022_both_hold_shorter | ensemble | INNER_B | 147 | 49 % | 50 / 47 / 3 | -2783 [-7081, 1435] | -7.0 | 0.55 / 0.80 | 0.51 / 0.50 | 0.54 / 0.08 | 89 / 58 | 614 / 442 | — |
| WF_wf2022_both_hold_shorter | seed 20261019 | INNER_B | 189 | 71 % | 57 / 40 / 4 | -717 [-5416, 3955] | -0.8 | 0.06 / -0.42 | 0.55 / 0.53 | 0.58 / 0.18 | 117 / 72 | -750 / 651 | — |
| WF_wf2022_both_hold_shorter | seed 20261020 | INNER_B | 176 | 70 % | 56 / 41 / 3 | -635 [-5899, 4810] | 2.0 | 0.53 / 0.72 | 0.65 / 0.61 | 0.59 / 0.20 | 94 / 82 | -362 / 1260 | — |
| WF_wf2022_both_hold_shorter | seed 20261021 | INNER_B | 202 | 59 % | 54 / 44 / 3 | 573 [-4789, 6124] | 0.4 | 0.80 / 0.33 | 0.49 / 0.50 | 0.54 / 0.10 | 131 / 71 | 1940 / -130 | — |
| WF_wf2022_both_hold_shorter | seed 20261022 | INNER_B | 190 | 61 % | 49 / 49 / 3 | -3486 [-7705, 685] | -13.8 | -1.47 / 0.12 | 0.46 / 0.44 | 0.54 / 0.09 | 97 / 93 | -1322 / -1966 | — |
| WF_wf2022_both_hold_shorter | seed 20261023 | INNER_B | 158 | 33 % | 48 / 52 / 0 | -2021 [-5818, 1966] | -4.7 | -0.19 / 0.06 | 0.47 / 0.40 | 0.46 / 0.04 | 85 / 73 | -1556 / 796 | — |
| WF_wf2022_zz_hold | ensemble | INNER_B | 126 | 60 % | 41 / 57 / 1 | -838 [-6290, 4792] | -5.3 | 0.46 / 0.40 | 0.61 / 0.56 | 0.45 / 0.06 | 50 / 76 | 261 / 662 | — |
| WF_wf2022_zz_hold | seed 20261019 | INNER_B | 155 | 69 % | 39 / 59 / 2 | 32 [-5882, 6310] | -0.2 | -0.12 / 0.51 | 0.56 / 0.54 | 0.48 / 0.09 | 89 / 66 | -1012 / 358 | — |
| WF_wf2022_zz_hold | seed 20261020 | INNER_B | 177 | 82 % | 39 / 59 / 2 | -992 [-8317, 6354] | -6.2 | -0.92 / -0.58 | 0.54 / 0.51 | 0.44 / 0.12 | 103 / 74 | -2262 / 178 | — |
| WF_wf2022_zz_hold | seed 20261021 | INNER_B | 128 | 70 % | 43 / 56 / 1 | 2073 [-5723, 10533] | 4.4 | 1.33 / -0.63 | 0.58 / 0.50 | 0.46 / 0.11 | 43 / 85 | 521 / 2819 | — |
| WF_wf2022_zz_hold | seed 20261022 | INNER_B | 112 | 35 % | 49 / 49 / 3 | 858 [-2693, 4359] | 5.6 | -0.83 / -0.35 | 0.59 / 0.61 | 0.51 / 0.02 | 49 / 63 | -419 / -2430 | — |
| WF_wf2022_zz_hold | seed 20261023 | INNER_B | 159 | 69 % | 42 / 55 / 3 | -126 [-7592, 7310] | -5.5 | 0.61 / 0.71 | 0.62 / 0.56 | 0.49 / 0.11 | 55 / 104 | -35 / 1401 | — |
| WF_wf2023_both_hold | ensemble | INNER_B | 107 | 56 % | 63 / 37 / 0 | 105 [-2461, 2576] | 1.4 | -0.71 / -0.12 | 0.65 / 0.65 | 0.62 / 0.11 | 73 / 34 | -1487 / 309 | — |
| WF_wf2023_both_hold | seed 20261019 | INNER_B | 105 | 76 % | 62 / 35 / 2 | 1000 [-2319, 4317] | 5.5 | -0.00 / -0.52 | 0.62 / 0.63 | 0.59 / 0.27 | 68 / 37 | -403 / 322 | — |
| WF_wf2023_both_hold | seed 20261020 | INNER_B | 156 | 37 % | 53 / 47 / 0 | 835 [-1497, 3360] | 1.9 | -0.48 / -0.56 | 0.57 / 0.54 | 0.50 / 0.04 | 64 / 92 | -1518 / 336 | — |
| WF_wf2023_both_hold | seed 20261021 | INNER_B | 90 | 81 % | 70 / 26 / 4 | 3735 [976, 6268] | 19.2 | -1.29 / -2.20 | 0.55 / 0.57 | 0.68 / 0.17 | 40 / 50 | -1103 / -262 | — |
| WF_wf2023_both_hold | seed 20261022 | INNER_B | 109 | 76 % | 69 / 29 / 2 | 3142 [382, 6032] | 15.6 | -0.94 / -1.65 | 0.57 / 0.58 | 0.67 / 0.18 | 88 / 21 | -1339 / 70 | — |
| WF_wf2023_both_hold | seed 20261023 | INNER_B | 112 | 61 % | 47 / 49 / 4 | -1690 [-4290, 1132] | -7.9 | -1.64 / -0.57 | 0.59 / 0.60 | 0.49 / 0.09 | 62 / 50 | -1987 / -353 | — |
| WF_wf2023_both_hold_shorter | ensemble | INNER_B | 132 | 57 % | 47 / 51 / 3 | -1638 [-4309, 931] | -6.6 | -0.35 / 0.65 | 0.60 / 0.58 | 0.52 / 0.11 | 58 / 74 | -864 / 199 | — |
| WF_wf2023_both_hold_shorter | seed 20261019 | INNER_B | 197 | 75 % | 52 / 45 / 3 | 177 [-4100, 4208] | 1.0 | 0.74 / 0.61 | 0.54 / 0.54 | 0.56 / 0.18 | 78 / 119 | -522 / 1633 | — |
| WF_wf2023_both_hold_shorter | seed 20261020 | INNER_B | 91 | 40 % | 58 / 42 / 0 | 408 [-1660, 2381] | 1.1 | -0.06 / -0.24 | 0.57 / 0.58 | 0.58 / 0.06 | 35 / 56 | -336 / 116 | — |
| WF_wf2023_both_hold_shorter | seed 20261021 | INNER_B | 125 | 79 % | 61 / 35 / 4 | 2114 [-1066, 5338] | 10.4 | -0.86 / -1.43 | 0.50 / 0.51 | 0.62 / 0.19 | 86 / 39 | -2121 / 1124 | — |
| WF_wf2023_both_hold_shorter | seed 20261022 | INNER_B | 117 | 26 % | 71 / 29 / 0 | 1135 [-447, 2516] | 6.2 | -0.79 / -1.09 | 0.55 / 0.57 | 0.71 / 0.04 | 65 / 52 | -987 / -536 | — |
| WF_wf2023_both_hold_shorter | seed 20261023 | INNER_B | 115 | 58 % | 60 / 40 / 0 | 282 [-2250, 2851] | 0.8 | -2.11 / -1.12 | 0.59 / 0.57 | 0.57 / 0.12 | 33 / 82 | -1802 / -995 | — |
| WF_wf2023_zz_hold | ensemble | INNER_B | 104 | 77 % | 40 / 57 / 2 | -2817 [-6071, 548] | -13.5 | -1.51 / -0.32 | 0.56 / 0.52 | 0.49 / 0.10 | 50 / 54 | -2497 / -75 | — |
| WF_wf2023_zz_hold | seed 20261019 | INNER_B | 143 | 83 % | 40 / 59 / 1 | -3511 [-7610, 779] | -15.0 | -1.73 / -0.08 | 0.63 / 0.56 | 0.48 / 0.17 | 79 / 64 | -2575 / -24 | — |
| WF_wf2023_zz_hold | seed 20261020 | INNER_B | 158 | 89 % | 46 / 51 / 3 | 1907 [-2608, 6559] | 4.1 | -2.73 / -0.92 | 0.57 / 0.52 | 0.50 / 0.18 | 49 / 109 | -1859 / -1801 | — |
| WF_wf2023_zz_hold | seed 20261021 | INNER_B | 99 | 65 % | 45 / 48 / 6 | 554 [-2672, 3904] | 3.1 | -1.11 / -1.48 | 0.57 / 0.53 | 0.56 / 0.06 | 90 / 9 | -1841 / -76 | — |
| WF_wf2023_zz_hold | seed 20261022 | INNER_B | 95 | 66 % | 33 / 65 / 2 | -3803 [-6991, -354] | -20.0 | -0.29 / 0.70 | 0.53 / 0.47 | 0.43 / 0.08 | 26 / 69 | -1109 / 277 | HARMFUL_EARLY_CLOSE |
| WF_wf2023_zz_hold | seed 20261023 | INNER_B | 95 | 82 % | 41 / 56 / 3 | -2306 [-6001, 1599] | -11.7 | -1.59 / -0.17 | 0.66 / 0.60 | 0.51 / 0.16 | 23 / 72 | -654 / -1512 | — |

Reproduction of every run: {"A_both_hold": true, "A_zz_hold": true, "WF_wf2022_both_hold": true, "WF_wf2022_both_hold_shorter": true, "WF_wf2022_zz_hold": true, "WF_wf2023_both_hold": true, "WF_wf2023_both_hold_shorter": true, "WF_wf2023_zz_hold": true}
