# E1 · Clonación de comportamiento + DAgger-lite desde el experto ZigZag v2 (generado)

DESARROLLO. Mismos periodos, datos y reglas de negocio que E0, ahora en el entorno barra a barra (`app/short_horizon/expert_guided/env.py`): acción {mantener, largo, corto, cerrar} + múltiplos σ de TP y SL (siempre fijados; recortados a la banda efectiva), decisión tras la barra completa y llenado en la apertura siguiente, cerrar-y-revertir = 1 cierre + 1 entrada, ≤ 4 entradas por día hábil, sin entradas 12 h tras reapertura, plano forzado 12 h antes del cierre, unidades = ⌊1,0 × 0,10 × saldo × 100 / precio⌋, costos RealisticProfile. Ronda 0 = BC sobre la trayectoria propia del experto (demostraciones); rondas 1–3 = la política actúa y el oráculo retrospectivo (`LegOracle`) reetiqueta cada estado visitado. Política final = ronda 3 (fijada de antemano). 2024 leído una vez tras congelar.

Paridad del entorno: el experto reproducido en este entorno da el mismo Sharpe que el simulador por bloques del estudio v2 (INNER_A 18,58 / INNER_B 20,24 / 2024 18,07 en ambos).

![e1_sharpe_por_ronda.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_sharpe_por_ronda.png) — [e1_sharpe_por_ronda.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_sharpe_por_ronda.png)
![e1_imitacion_por_ronda.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_imitacion_por_ronda.png) — [e1_imitacion_por_ronda.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_imitacion_por_ronda.png)
![e1_curvas_entrenamiento.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_curvas_entrenamiento.png) — [e1_curvas_entrenamiento.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_curvas_entrenamiento.png)

## Compuerta pre-registrada

- I1 (log-loss de imitación en sus propios estados INNER_B < previa TRAIN suavizada): **no** [False, False, False]
- I2 (Sharpe INNER_B media de semillas > confirmado y > aleatoria previa): **no** (-4,00 vs -3,64 y -3,27); supera no operar: no
- Decisión: **STOP_AT_E1**

## TRAIN 2020–2022

| Política | Neto USD | Sharpe | Sortino | DD máx % | Entradas / cierres | Cierres TP / SL / agente / reversión / plano forzado | Acierto % | Bruto USD | Costos USD (spread / desl. / com. / swap) | Entradas/sem | Rechazos por tope diario | Válidas |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| Experto (demostraciones) | 1,06e+29 | 18,23 | 20.395,82 | 7,21 | 2.480 / 2.480 | 2.411 / 0 / 0 / 0 / 69 | 98,8 | 1,12e+29 | 5,69e+27 (2,26e+27 / 7,58e+26 / 2,54e+27 / 1,35e+26) | 15,80 | 733 | 1/1 |
| Oráculo de tramo (reetiquetador) | 4,81e+17 | 11,81 | 669,04 | 6,76 | 521 / 521 | 484 / 0 / 0 / 0 / 37 | 99,8 | 4,87e+17 | 6,78e+15 (1,76e+15 / 9,16e+14 / 3,11e+15 / 9,93e+14) | 3,32 | 0 | 1/1 |
| Experto confirmado | -9.685,17 | -1,55 | -2,19 | 97,61 | 2.536 / 2.536 | 87 / 4 / 0 / 2.344 / 101 | 32,9 | 2.183,91 | 11.869,08 (3.918,98 / 2.772,68 / 4.931,67 / 245,75) | 16,15 | 751 | 1/1 |
| BC ronda 0 (media 3 semillas) | 62.893,02 | 1,08 | 1,69 | 56,61 | 657 / 657 | 439 / 86 / 0 / 0 / 132 | 71,7 | 77.769,89 | 14.876,86 (3.623,63 / 2.602,46 / 6.174,33 / 2.476,45) | 4,19 | 5 | 3/3 |
| DAgger ronda 3 (media 3 semillas) | 4,12e+08 | 6,37 | 18,63 | 19,12 | 2.323 / 2.323 | 303 / 0 / 1.822 / 148 / 50 | 46,9 | 5,43e+08 | 1,31e+08 (4,18e+07 / 3,10e+07 / 5,72e+07 / 1.299.750,56) | 14,80 | 361 | 3/3 |
|   DAgger ronda 3, semilla 20261019 | 7,77e+07 | 5,57 | 14,77 | 19,67 | 2.429 / 2.429 | 271 / 0 / 1.965 / 149 / 44 | 46,2 | 1,05e+08 | 2,70e+07 (8.685.360,04 / 6.436.434,68 / 1,17e+07 / 174.040,26) | 15,47 | 426 | 1/1 |
|   DAgger ronda 3, semilla 20261020 | 9,23e+08 | 7,18 | 22,62 | 19,49 | 2.357 / 2.357 | 349 / 0 / 1.803 / 158 / 47 | 47,9 | 1,22e+09 | 2,94e+08 (9,48e+07 / 6,92e+07 / 1,28e+08 / 2.172.529,06) | 15,01 | 422 | 1/1 |
|   DAgger ronda 3, semilla 20261021 | 2,35e+08 | 6,35 | 18,50 | 18,22 | 2.184 / 2.184 | 290 / 1 / 1.698 / 136 / 59 | 46,5 | 3,08e+08 | 7,25e+07 (2,19e+07 / 1,72e+07 / 3,18e+07 / 1.552.682,36) | 13,91 | 235 | 1/1 |
| Aleatoria previa TRAIN (media 3 semillas) | -9.482,00 | -1,29 | -1,69 | 96,68 | 1.706 / 1.706 | 61 / 22 / 0 / 1.494 / 129 | 45,7 | -1.148,27 | 8.333,73 (2.775,91 / 1.823,28 / 3.218,02 / 516,52) | 10,86 | 140 | 3/3 |
| Sin operar | 0,00 | 0,00 | 0,00 | 0,00 | 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0,0 | 0,00 | 0,00 (0,00 / 0,00 / 0,00 / 0,00) | 0,00 | 0 | 0/1 |

## INNER_A 2023-S1

| Política | Neto USD | Sharpe | Sortino | DD máx % | Entradas / cierres | Cierres TP / SL / agente / reversión / plano forzado | Acierto % | Bruto USD | Costos USD (spread / desl. / com. / swap) | Entradas/sem | Rechazos por tope diario | Válidas |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| Experto (demostraciones) | 6,90e+07 | 18,58 | 20.636,78 | 5,31 | 406 / 406 | 396 / 0 / 0 / 0 / 10 | 97,5 | 7,39e+07 | 4.898.444,67 (1.879.278,82 / 659.927,27 / 2.255.559,29 / 103.679,28) | 15,62 | 124 | 1/1 |
| Oráculo de tramo (reetiquetador) | 2.624.319,26 | 13,09 | — | 5,31 | 94 / 94 | 88 / 0 / 0 / 0 / 6 | 100,0 | 2.687.951,24 | 63.631,98 (16.923,57 / 9.523,85 / 30.668,27 / 6.516,30) | 3,62 | 0 | 1/1 |
| Experto confirmado | -2.740,02 | -0,80 | -1,06 | 39,66 | 412 / 412 | 12 / 1 / 0 / 384 / 15 | 32,5 | 2.947,71 | 5.687,73 (1.933,66 / 1.296,00 / 2.297,14 / 160,93) | 15,85 | 124 | 1/1 |
| BC ronda 0 (media 3 semillas) | -3.068,11 | -0,93 | -1,26 | 57,50 | 91 / 91 | 53 / 12 / 0 / 1 / 25 | 64,2 | -1.637,00 | 1.431,11 (313,59 / 247,24 / 532,78 / 337,50) | 3,51 | 0 | 3/3 |
| DAgger ronda 3 (media 3 semillas) | -5.192,68 | -3,30 | -4,26 | 57,94 | 399 / 399 | 26 / 0 / 337 / 26 / 10 | 29,6 | -991,83 | 4.200,86 (1.310,72 / 1.024,19 / 1.849,12 / 16,83) | 15,33 | 87 | 3/3 |
|   DAgger ronda 3, semilla 20261019 | -3.060,41 | -1,41 | -2,27 | 41,35 | 413 / 413 | 24 / 0 / 350 / 30 / 9 | 32,2 | 2.302,33 | 5.362,74 (1.684,05 / 1.312,59 / 2.364,07 / 2,03) | 15,88 | 101 | 1/1 |
|   DAgger ronda 3, semilla 20261020 | -6.596,20 | -4,46 | -5,58 | 70,19 | 409 / 409 | 25 / 0 / 349 / 26 / 9 | 28,6 | -2.910,34 | 3.685,87 (1.186,21 / 889,13 / 1.600,41 / 10,11) | 15,73 | 102 | 1/1 |
|   DAgger ronda 3, semilla 20261021 | -5.921,43 | -4,03 | -4,94 | 62,27 | 374 / 374 | 29 / 0 / 313 / 21 / 11 | 28,1 | -2.367,48 | 3.553,96 (1.061,90 / 870,83 / 1.582,88 / 38,36) | 14,38 | 59 | 1/1 |
| Aleatoria previa TRAIN (media 3 semillas) | 352,08 | 0,31 | 0,53 | 43,92 | 283 / 283 | 11 / 3 / 0 / 247 / 23 | 48,8 | 5.106,89 | 4.754,81 (1.603,25 / 1.055,17 / 1.863,56 / 232,83) | 10,90 | 20 | 3/3 |
| Sin operar | 0,00 | 0,00 | 0,00 | 0,00 | 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0,0 | 0,00 | 0,00 (0,00 / 0,00 / 0,00 / 0,00) | 0,00 | 0 | 0/1 |

## INNER_B 2023-S2

| Política | Neto USD | Sharpe | Sortino | DD máx % | Entradas / cierres | Cierres TP / SL / agente / reversión / plano forzado | Acierto % | Bruto USD | Costos USD (spread / desl. / com. / swap) | Entradas/sem | Rechazos por tope diario | Válidas |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| Experto (demostraciones) | 2,92e+07 | 20,24 | 21.739,63 | 2,86 | 421 / 421 | 410 / 0 / 0 / 0 / 11 | 98,8 | 3,15e+07 | 2.319.067,53 (946.953,29 / 297.635,34 / 1.016.220,43 / 58.258,46) | 15,59 | 126 | 1/1 |
| Oráculo de tramo (reetiquetador) | 586.090,98 | 12,43 | — | 2,86 | 73 / 73 | 70 / 0 / 0 / 0 / 3 | 100,0 | 599.596,95 | 13.505,97 (3.809,51 / 1.996,26 / 6.631,83 / 1.068,38) | 2,70 | 0 | 0/1 |
| Experto confirmado | -6.023,38 | -3,64 | -4,42 | 62,95 | 430 / 430 | 12 / 0 / 0 / 404 / 14 | 30,5 | -1.706,08 | 4.317,30 (1.468,55 / 1.005,13 / 1.783,02 / 60,61) | 15,93 | 128 | 1/1 |
| BC ronda 0 (media 3 semillas) | -3.739,82 | -1,44 | -1,83 | 45,97 | 95 / 95 | 55 / 14 / 0 / 0 / 25 | 62,4 | -2.400,89 | 1.338,93 (323,99 / 247,92 / 526,19 / 240,83) | 3,51 | 0 | 3/3 |
| DAgger ronda 3 (media 3 semillas) | -5.102,87 | -4,00 | -4,79 | 55,25 | 389 / 389 | 24 / 1 / 327 / 27 / 10 | 30,5 | -1.147,85 | 3.955,02 (1.187,49 / 984,68 / 1.772,59 / 10,27) | 14,40 | 84 | 3/3 |
|   DAgger ronda 3, semilla 20261019 | -6.111,35 | -5,34 | -6,10 | 63,38 | 407 / 407 | 19 / 0 / 352 / 27 / 9 | 29,5 | -2.249,53 | 3.861,81 (1.180,04 / 962,41 / 1.722,87 / -3,51) | 15,07 | 94 | 1/1 |
|   DAgger ronda 3, semilla 20261020 | -5.219,07 | -3,88 | -4,86 | 58,46 | 397 / 397 | 28 / 1 / 327 / 29 / 12 | 30,2 | -1.612,09 | 3.606,98 (1.090,08 / 893,20 / 1.612,32 / 11,39) | 14,70 | 96 | 1/1 |
|   DAgger ronda 3, semilla 20261021 | -3.978,20 | -2,77 | -3,41 | 43,90 | 362 / 362 | 25 / 2 / 302 / 24 / 9 | 31,8 | 418,07 | 4.396,27 (1.292,36 / 1.098,42 / 1.982,57 / 22,92) | 13,41 | 63 | 1/1 |
| Aleatoria previa TRAIN (media 3 semillas) | -5.937,99 | -3,27 | -3,78 | 63,91 | 271 / 271 | 7 / 4 / 0 / 239 / 22 | 43,1 | -2.468,78 | 3.469,22 (1.165,71 / 752,65 / 1.324,10 / 226,75) | 10,05 | 21 | 3/3 |
| Sin operar | 0,00 | 0,00 | 0,00 | 0,00 | 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0,0 | 0,00 | 0,00 (0,00 / 0,00 / 0,00 / 0,00) | 0,00 | 0 | 0/1 |

## 2024 (observado, 1 lectura)

| Política | Neto USD | Sharpe | Sortino | DD máx % | Entradas / cierres | Cierres TP / SL / agente / reversión / plano forzado | Acierto % | Bruto USD | Costos USD (spread / desl. / com. / swap) | Entradas/sem | Rechazos por tope diario | Válidas |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| Experto (demostraciones) | 4,39e+09 | 18,07 | 10.039,81 | 2,35 | 852 / 852 | 830 / 0 / 0 / 0 / 22 | 96,7 | 4,71e+09 | 3,13e+08 (1,13e+08 / 4,45e+07 / 1,46e+08 / 9.958.900,04) | 16,08 | 237 | 1/1 |
| Oráculo de tramo (reetiquetador) | 2,21e+07 | 12,36 | 526,64 | 2,45 | 175 / 175 | 161 / 0 / 0 / 0 / 14 | 100,0 | 2,27e+07 | 601.035,22 (175.460,59 / 94.432,76 / 290.571,13 / 40.570,75) | 3,30 | 0 | 1/1 |
| Experto confirmado | -8.262,08 | -3,83 | -4,51 | 84,71 | 848 / 848 | 20 / 2 / 0 / 790 / 36 | 30,0 | 993,74 | 9.255,82 (3.185,49 / 2.140,94 / 3.790,80 / 138,58) | 16,00 | 256 | 1/1 |
| DAgger ronda 3 (media 3 semillas) | -8.093,15 | -5,21 | -6,08 | 82,06 | 809 / 809 | 44 / 2 / 708 / 40 / 15 | 27,6 | -2.297,81 | 5.795,34 (1.820,22 / 1.407,70 / 2.523,79 / 43,64) | 15,26 | 168 | 3/3 |
|   DAgger ronda 3, semilla 20261019 | -7.982,18 | -4,99 | -5,88 | 80,83 | 860 / 860 | 42 / 1 / 771 / 34 / 12 | 27,9 | -1.617,41 | 6.364,77 (2.031,69 / 1.537,64 / 2.751,24 / 44,21) | 16,23 | 192 | 1/1 |
|   DAgger ronda 3, semilla 20261020 | -8.196,08 | -5,58 | -6,27 | 82,69 | 809 / 809 | 47 / 1 / 710 / 32 / 19 | 26,7 | -2.303,92 | 5.892,16 (1.828,25 / 1.440,71 / 2.589,33 / 33,86) | 15,26 | 202 | 1/1 |
|   DAgger ronda 3, semilla 20261021 | -8.101,17 | -5,07 | -6,09 | 82,65 | 758 / 758 | 44 / 3 / 642 / 54 / 15 | 28,2 | -2.972,09 | 5.129,09 (1.600,72 / 1.244,75 / 2.230,79 / 52,83) | 14,30 | 110 | 1/1 |
| Aleatoria previa TRAIN (media 3 semillas) | -5.957,19 | -1,64 | -2,10 | 64,86 | 565 / 565 | 23 / 10 / 0 / 489 / 44 | 43,6 | 151,25 | 6.108,44 (2.097,67 / 1.351,23 / 2.381,79 / 277,74) | 10,65 | 41 | 3/3 |
| Sin operar | 0,00 | 0,00 | 0,00 | 0,00 | 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0,0 | 0,00 | 0,00 (0,00 / 0,00 / 0,00 / 0,00) | 0,00 | 0 | 0/1 |

## Entrenamiento e imitación por ronda

| Semilla | Ronda | Filas TRAIN | Épocas | Mejor | Causa | Pérdida TRAIN | Pérdida INNER_A | Brecha | log-loss − previa INNER_B | Acierto INNER_B | Etiquetas INNER_B (M/L/C/Cerrar) | Predichas |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 20261019 | BC | 18406 | 45 | 35 | patience | 0,2464 | 0,6490 | 0,4026 | 6,486 | 0,550 | [1926, 85, 56, 1011] | [2826, 122, 130, 0] |
| 20261019 | DAgger 1 | 36812 | 29 | 19 | patience | 0,5281 | 0,8309 | 0,3028 | 0,346 | 0,600 | [2176, 56, 77, 769] | [2258, 271, 269, 280] |
| 20261019 | DAgger 2 | 55218 | 28 | 18 | patience | 0,5771 | 0,9862 | 0,4092 | 0,182 | 0,625 | [2478, 44, 60, 496] | [1975, 386, 372, 345] |
| 20261019 | DAgger 3 | 73624 | 26 | 16 | patience | 0,5833 | 1,0071 | 0,4237 | 0,090 | 0,739 | [2589, 47, 54, 388] | [2205, 268, 253, 352] |
| 20261020 | BC | 18406 | 38 | 28 | patience | 0,2947 | 0,6116 | 0,3168 | 6,245 | 0,553 | [1916, 82, 54, 1026] | [2834, 121, 123, 0] |
| 20261020 | DAgger 1 | 36812 | 41 | 31 | patience | 0,4270 | 0,8409 | 0,4139 | 0,504 | 0,587 | [2180, 58, 77, 763] | [2242, 282, 260, 294] |
| 20261020 | DAgger 2 | 55218 | 27 | 17 | patience | 0,5745 | 0,9337 | 0,3592 | 0,202 | 0,620 | [2470, 58, 71, 479] | [1991, 372, 371, 344] |
| 20261020 | DAgger 3 | 73624 | 23 | 13 | patience | 0,6069 | 0,9858 | 0,3788 | 0,074 | 0,725 | [2538, 50, 71, 419] | [2246, 272, 233, 327] |
| 20261021 | BC | 18406 | 50 | 40 | patience | 0,2173 | 0,6828 | 0,4655 | 6,674 | 0,548 | [1909, 71, 66, 1032] | [2840, 112, 126, 0] |
| 20261021 | DAgger 1 | 36812 | 43 | 33 | patience | 0,4212 | 0,8971 | 0,4759 | 0,547 | 0,571 | [2133, 71, 72, 802] | [2244, 303, 242, 289] |
| 20261021 | DAgger 2 | 55218 | 36 | 26 | patience | 0,5067 | 1,0379 | 0,5312 | 0,213 | 0,614 | [2450, 49, 66, 513] | [1999, 396, 344, 339] |
| 20261021 | DAgger 3 | 73624 | 22 | 12 | patience | 0,6219 | 0,9965 | 0,3746 | 0,043 | 0,756 | [2570, 43, 88, 377] | [2346, 262, 168, 302] |
