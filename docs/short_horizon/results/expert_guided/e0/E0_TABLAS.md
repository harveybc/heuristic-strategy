# E0 · Imitabilidad del experto ZigZag v2 desde las 15 entradas del núcleo admitido (generado)

DESARROLLO. TRAIN 2020–2022 ajusta; INNER_A (2023-S1) monitorea, calibra la temperatura y elige el umbral; INNER_B (2023-S2) se puntúa después de congelar; 2024 es desarrollo ya observado, leído una sola vez después de congelar. TEST 2025 cerrado. Fuente: raíz admitida c80847bb (UTC inicio de barra, disponible +1 h); entradas: CSV preparados de Musashi (contrato 37e06192, selección 553fc047), 13 observaciones paso 2, z-score causal 256.

Ejecución de toda política: decisión tras la barra completa, llenado en la apertura siguiente, SL = 2σ, TP = 1,5σ (causal) o pivote siguiente (solo el experto retrospectivo), unidades = ⌊1,0 × 0,10 × saldo × 100 / precio⌋, sin entradas 12 h tras reapertura, plano 12 h antes del cierre, ≤ 4 órdenes por día hábil, costos RealisticProfile (spread por hora, deslizamiento, comisión, swap). Sharpe = diario, √365, días calendario incluidos los cierres. Órdenes/semana sobre TODAS las semanas ISO con filas elegibles (más estricto que el estudio v2). Cada operación = 1 entrada + 1 cierre; cierres por TP / SL / tiempo (plano forzado o siguiente señal).

![e0_brecha_imitabilidad.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_brecha_imitabilidad.png) — [e0_brecha_imitabilidad.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_brecha_imitabilidad.png)
![e0_curvas_entrenamiento.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_curvas_entrenamiento.png) — [e0_curvas_entrenamiento.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_curvas_entrenamiento.png)
![e0_prauc_exceso.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_prauc_exceso.png) — [e0_prauc_exceso.png](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_prauc_exceso.png)

## Compuerta pre-registrada

- G1 (log-loss calibrada < previa TRAIN en INNER_B): **sí** (por semilla [True, True, True], media -67,1 milinats)
- G2 (PR-AUC − prevalencia, IC 95 % bootstrap semanal > 0, ambas direcciones): **sí** ([True, True, True])
- Decisión: **CONTINUE_TO_E1**

## Negociación · TRAIN 2020–2022

| Política | Neto USD | Sharpe | Sortino | DD máx % | Operaciones | Cierres TP / SL / tiempo | Acierto % | Costos USD (spread / desl. / com. / swap) | Órdenes/sem | Válidas |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Experto retrospectivo (TP = pivote siguiente) | 1,05e+29 | 18,23 | 20.390,78 | 7,21 | 2.479 | 2.410 / 0 / 69 | 98,8 | 5,60e+27 (2,22e+27 / 7,47e+26 / 2,50e+27 / 1,33e+26) | 16,10 | 1/1 |
| Experto retrospectivo, TP causal 1,5σ | 1,82e+22 | 16,32 | 577,61 | 7,21 | 2.527 | 159 / 0 / 2.368 | 85,6 | 1,52e+21 (5,28e+20 / 3,43e+20 / 6,12e+20 / 3,91e+19) | 16,41 | 1/1 |
| Experto confirmado (reactivo) | -9.724,77 | -1,60 | -2,25 | 97,85 | 2.536 | 87 / 4 / 2.445 | 33,0 | 11.496,33 (3.870,01 / 2.659,75 / 4.730,93 / 235,64) | 16,47 | 1/1 |
| Estudiante del experto retrospectivo (media 3 semillas) | 5,65e+08 | 4,79 | 9,36 | 36,59 | 991 | 119 / 20 / 852 | 67,2 | 6,34e+07 (2,07e+07 / 1,23e+07 / 2,26e+07 / 7.894.370,62) | 6,44 | 3/3 |
| Estudiante del experto confirmado (media 3 semillas) | -9.491,88 | -1,35 | -1,76 | 96,78 | 1.344 | 77 / 26 / 1.242 | 40,3 | 5.816,12 (1.758,41 / 1.293,83 / 2.289,11 / 474,78) | 8,73 | 3/3 |
| Aleatoria con la previa de TRAIN (media 3 semillas) | -9.771,20 | -2,02 | -2,59 | 98,30 | 2.481 | 32 / 10 / 2.439 | 43,7 | 13.388,25 (4.745,23 / 3.054,34 / 5.363,98 / 224,70) | 16,11 | 3/3 |
| Sin operar | 0 | 0 | 0 | 0 | 0 | — | — | 0 | 0 | 0/1 |

## Negociación · INNER_A 2023-S1

| Política | Neto USD | Sharpe | Sortino | DD máx % | Operaciones | Cierres TP / SL / tiempo | Acierto % | Costos USD (spread / desl. / com. / swap) | Órdenes/sem | Válidas |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Experto retrospectivo (TP = pivote siguiente) | 6,90e+07 | 18,58 | 20.636,78 | 5,31 | 406 | 396 / 0 / 10 | 97,5 | 4.898.444,67 (1.879.278,82 / 659.927,27 / 2.255.559,29 / 103.679,28) | 15,62 | 1/1 |
| Experto retrospectivo, TP causal 1,5σ | 5.903.232,00 | 15,78 | 428,97 | 5,31 | 414 | 21 / 0 / 393 | 82,6 | 616.308,85 (204.512,59 / 141.374,60 / 255.356,12 / 15.065,53) | 15,92 | 1/1 |
| Experto confirmado (reactivo) | -3.129,01 | -0,98 | -1,27 | 39,25 | 412 | 12 / 1 / 399 | 32,3 | 5.536,24 (1.909,74 / 1.252,31 / 2.219,74 / 154,46) | 15,85 | 1/1 |
| Estudiante del experto retrospectivo (media 3 semillas) | -815,95 | -0,10 | -0,12 | 32,05 | 97 | 12 / 3 / 82 | 46,2 | 1.658,75 (475,68 / 344,35 / 624,84 / 213,88) | 3,74 | 3/3 |
| Estudiante del experto confirmado (media 3 semillas) | 1.027,06 | 0,57 | 0,86 | 34,70 | 159 | 15 / 4 / 140 | 45,0 | 2.923,16 (841,10 / 646,84 / 1.163,43 / 271,79) | 6,12 | 3/3 |
| Aleatoria con la previa de TRAIN (media 3 semillas) | -5.886,84 | -3,16 | -3,74 | 70,13 | 425 | 4 / 2 / 419 | 42,0 | 4.927,39 (1.800,63 / 1.103,31 / 1.931,80 / 91,66) | 16,35 | 3/3 |
| Sin operar | 0 | 0 | 0 | 0 | 0 | — | — | 0 | 0 | 0/1 |

## Negociación · INNER_B 2023-S2

| Política | Neto USD | Sharpe | Sortino | DD máx % | Operaciones | Cierres TP / SL / tiempo | Acierto % | Costos USD (spread / desl. / com. / swap) | Órdenes/sem | Válidas |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Experto retrospectivo (TP = pivote siguiente) | 2,92e+07 | 20,24 | 21.739,63 | 2,86 | 421 | 410 / 0 / 11 | 98,8 | 2.319.067,53 (946.953,29 / 297.635,34 / 1.016.220,43 / 58.258,46) | 16,19 | 1/1 |
| Experto retrospectivo, TP causal 1,5σ | 3.771.156,38 | 17,83 | 317,62 | 3,42 | 431 | 23 / 0 / 408 | 85,8 | 458.698,01 (165.544,68 / 100.363,91 / 181.293,32 / 11.496,11) | 16,58 | 1/1 |
| Experto confirmado (reactivo) | -5.934,28 | -3,58 | -4,36 | 62,22 | 430 | 12 / 0 / 418 | 31,2 | 4.376,13 (1.497,16 / 1.015,80 / 1.801,98 / 61,19) | 16,54 | 1/1 |
| Estudiante del experto retrospectivo (media 3 semillas) | -745,46 | -0,23 | -0,26 | 36,36 | 96 | 9 / 5 / 82 | 52,1 | 1.609,05 (435,15 / 347,49 / 609,20 / 217,22) | 3,71 | 3/3 |
| Estudiante del experto confirmado (media 3 semillas) | -4.060,58 | -1,74 | -2,17 | 48,91 | 163 | 13 / 5 / 146 | 42,3 | 1.916,96 (540,03 / 434,57 / 773,07 / 169,30) | 6,28 | 3/3 |
| Aleatoria con la previa de TRAIN (media 3 semillas) | -4.424,98 | -2,02 | -2,70 | 48,00 | 428 | 8 / 1 / 419 | 43,1 | 5.378,15 (1.883,41 / 1.226,79 / 2.161,21 / 106,74) | 16,47 | 3/3 |
| Sin operar | 0 | 0 | 0 | 0 | 0 | — | — | 0 | 0 | 0/1 |

## Negociación · 2024 (observado, 1 lectura)

| Política | Neto USD | Sharpe | Sortino | DD máx % | Operaciones | Cierres TP / SL / tiempo | Acierto % | Costos USD (spread / desl. / com. / swap) | Órdenes/sem | Válidas |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Experto retrospectivo (TP = pivote siguiente) | 4,39e+09 | 18,07 | 10.039,81 | 2,35 | 852 | 830 / 0 / 22 | 96,7 | 3,13e+08 (1,13e+08 / 4,45e+07 / 1,46e+08 / 9.958.900,04) | 16,08 | 1/1 |
| Experto retrospectivo, TP causal 1,5σ | 8,58e+07 | 15,99 | 279,75 | 3,81 | 864 | 44 / 0 / 820 | 82,5 | 9.995.548,01 (3.285.153,80 / 2.316.555,35 / 4.138.829,81 / 255.009,05) | 16,30 | 1/1 |
| Experto confirmado (reactivo) | -8.306,31 | -3,88 | -4,55 | 85,22 | 848 | 20 / 2 / 826 | 29,7 | 9.487,61 (3.369,26 / 2.157,94 / 3.820,95 / 139,46) | 16,00 | 1/1 |
| Estudiante del experto retrospectivo (media 3 semillas) | 3.601,71 | 0,94 | 1,42 | 29,13 | 201 | 25 / 7 / 168 | 51,9 | 4.420,97 (1.328,71 / 945,33 / 1.708,66 / 438,26) | 3,79 | 3/3 |
| Estudiante del experto confirmado (media 3 semillas) | -3.467,10 | -0,74 | -0,97 | 59,96 | 325 | 23 / 10 / 292 | 43,8 | 5.750,37 (1.588,25 / 1.313,70 / 2.323,43 / 524,99) | 6,14 | 3/3 |
| Aleatoria con la previa de TRAIN (media 3 semillas) | -7.956,84 | -3,60 | -4,36 | 82,81 | 838 | 16 / 3 / 819 | 40,6 | 7.606,75 (2.750,52 / 1.711,71 / 3.015,66 / 128,87) | 15,81 | 3/3 |
| Sin operar | 0 | 0 | 0 | 0 | 0 | — | — | 0 | 0 | 0/1 |

## Clasificación (probabilidades calibradas con temperatura en INNER_A)

| Objetivo | Semilla | Periodo | log-loss − previa (milinats) | PR-AUC largo (prev.) | PR-AUC corto (prev.) | IC95 exceso largo | IC95 exceso corto | Precisión/recall largo @umbral | Precisión/recall corto @umbral |
|---|---|---|---:|---|---|---|---|---|---|
| retrospectivo | 20261019 | TRAIN | -297,7 | 0,824 (0,113) | 0,813 (0,113) | — | — | 0,966 / 0,296 | 0,970 / 0,293 |
| retrospectivo | 20261019 | INNER_A | -59,8 | 0,283 (0,111) | 0,256 (0,109) | — | — | 0,370 / 0,062 | 0,302 / 0,060 |
| retrospectivo | 20261019 | INNER_B | -65,3 | 0,289 (0,115) | 0,249 (0,115) | [0,139; 0,220] | [0,110; 0,159] | 0,423 / 0,079 | 0,294 / 0,054 |
| retrospectivo | 20261019 | OBS_2024 | -60,0 | 0,279 (0,115) | 0,248 (0,115) | [0,141; 0,191] | [0,114; 0,157] | 0,369 / 0,073 | 0,327 / 0,064 |
| retrospectivo | 20261020 | TRAIN | -297,6 | 0,816 (0,113) | 0,834 (0,113) | — | — | 0,987 / 0,281 | 0,974 / 0,296 |
| retrospectivo | 20261020 | INNER_A | -68,3 | 0,267 (0,111) | 0,243 (0,109) | — | — | 0,318 / 0,051 | 0,309 / 0,063 |
| retrospectivo | 20261020 | INNER_B | -73,4 | 0,287 (0,115) | 0,243 (0,115) | [0,142; 0,214] | [0,102; 0,162] | 0,432 / 0,068 | 0,256 / 0,036 |
| retrospectivo | 20261020 | OBS_2024 | -59,3 | 0,270 (0,115) | 0,247 (0,115) | [0,133; 0,183] | [0,113; 0,154] | 0,394 / 0,070 | 0,351 / 0,061 |
| retrospectivo | 20261021 | TRAIN | -310,7 | 0,867 (0,113) | 0,842 (0,113) | — | — | 0,985 / 0,326 | 0,981 / 0,308 |
| retrospectivo | 20261021 | INNER_A | -63,0 | 0,268 (0,111) | 0,240 (0,109) | — | — | 0,354 / 0,062 | 0,294 / 0,056 |
| retrospectivo | 20261021 | INNER_B | -62,7 | 0,280 (0,115) | 0,248 (0,115) | [0,137; 0,198] | [0,103; 0,171] | 0,400 / 0,072 | 0,245 / 0,046 |
| retrospectivo | 20261021 | OBS_2024 | -64,0 | 0,270 (0,115) | 0,254 (0,115) | [0,132; 0,182] | [0,122; 0,161] | 0,337 / 0,054 | 0,327 / 0,057 |
| confirmado | 20261019 | TRAIN | -417,1 | 0,897 (0,113) | 0,895 (0,113) | — | — | 0,989 / 0,270 | 0,994 / 0,316 |
| confirmado | 20261019 | INNER_A | -208,3 | 0,488 (0,109) | 0,530 (0,109) | — | — | 0,651 / 0,104 | 0,732 / 0,153 |
| confirmado | 20261019 | INNER_B | -206,9 | 0,560 (0,114) | 0,507 (0,115) | [0,388; 0,504] | [0,334; 0,444] | 0,786 / 0,159 | 0,688 / 0,157 |
| confirmado | 20261019 | OBS_2024 | -199,6 | 0,475 (0,114) | 0,502 (0,113) | [0,318; 0,405] | [0,348; 0,433] | 0,713 / 0,130 | 0,718 / 0,144 |
| confirmado | 20261020 | TRAIN | -434,2 | 0,920 (0,113) | 0,934 (0,113) | — | — | 0,995 / 0,391 | 0,994 / 0,408 |
| confirmado | 20261020 | INNER_A | -208,7 | 0,498 (0,109) | 0,522 (0,109) | — | — | 0,746 / 0,187 | 0,667 / 0,201 |
| confirmado | 20261020 | INNER_B | -206,2 | 0,540 (0,114) | 0,503 (0,115) | [0,371; 0,478] | [0,343; 0,432] | 0,671 / 0,177 | 0,662 / 0,181 |
| confirmado | 20261020 | OBS_2024 | -184,9 | 0,468 (0,114) | 0,491 (0,113) | [0,317; 0,395] | [0,340; 0,417] | 0,596 / 0,146 | 0,621 / 0,173 |
| confirmado | 20261021 | TRAIN | -414,7 | 0,897 (0,113) | 0,899 (0,113) | — | — | 0,941 / 0,551 | 0,946 / 0,551 |
| confirmado | 20261021 | INNER_A | -204,2 | 0,485 (0,109) | 0,509 (0,109) | — | — | 0,573 / 0,265 | 0,634 / 0,291 |
| confirmado | 20261021 | INNER_B | -218,7 | 0,581 (0,114) | 0,511 (0,115) | [0,409; 0,530] | [0,339; 0,455] | 0,701 / 0,296 | 0,588 / 0,249 |
| confirmado | 20261021 | OBS_2024 | -198,3 | 0,480 (0,114) | 0,485 (0,113) | [0,327; 0,406] | [0,328; 0,418] | 0,590 / 0,273 | 0,599 / 0,264 |

## Entrenamiento (regla del dueño)

| Objetivo | Semilla | Épocas | Mejor época | Causa | Pérdida TRAIN | Pérdida INNER_A | Brecha INNER−TRAIN | Temperatura | Tasa de señal elegida (INNER_A) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| hindsight | 20261019 | 51 | 41 | patience | 0,3036 | 0,6842 | 0,3807 | 1,76 | 4 % |
| hindsight | 20261020 | 48 | 38 | patience | 0,3027 | 0,6925 | 0,3899 | 1,84 | 4 % |
| hindsight | 20261021 | 55 | 45 | patience | 0,2762 | 0,7128 | 0,4366 | 1,94 | 4 % |
| confirmed | 20261019 | 49 | 39 | patience | 0,2154 | 0,5161 | 0,3007 | 1,64 | 4 % |
| confirmed | 20261020 | 53 | 43 | patience | 0,1845 | 0,5359 | 0,3514 | 1,79 | 6 % |
| confirmed | 20261021 | 49 | 39 | patience | 0,2145 | 0,5244 | 0,3100 | 1,67 | 10 % |

## Población y exclusiones (por motivo; las máscaras se solapan)

| Motivo | TRAIN | INNER_A | INNER_B | 2024 |
|---|---:|---:|---:|---:|
| outside_periods | 0 | 0 | 0 | 0 |
| not_joined | 0 | 0 | 0 | 0 |
| window_or_scale_warmup | 256 | 0 | 0 | 0 |
| entry_gate_closed | 3913 | 650 | 650 | 1350 |
| no_causal_sigma | 109 | 0 | 0 | 0 |
| label_censored | 0 | 0 | 0 | 3 |
| label_support_crosses_period_end | 3 | 6 | 8 | 0 |
| **elegibles** | 14562 | 2461 | 2436 | 4871 |
