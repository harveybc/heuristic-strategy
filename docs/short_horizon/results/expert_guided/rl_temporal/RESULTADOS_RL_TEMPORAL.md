# RL guiado por el experto ZigZag con el tronco temporal mínimo: resultados finales

DESARROLLO, nada operable. TRAIN 2020–2022; validación = 2023 (VAL A = S1, VAL B = S2); 2024 = desarrollo ya observado, leído una sola vez después de congelar; TEST 2025 cerrado. Contrato: SL y TP efectivos en cada orden, unidades enteras con rel_volume 0,10 y apalancamiento 1:100, ≤ 4 entradas/día, ≥ 3 órdenes/semana, costos realistas (spread, slippage, comisión, swap) ASUMIDOS, no verificados con el broker. Capital 10 000 USD por periodo.

Cómo leer la ganancia: se capitaliza (el tamaño crece con el balance a 1:100), así que una política ganadora en TRAIN (3 años) produce cifras astronómicas, y TRAIN (3 años) no es comparable en USD con VAL A/B (6 meses cada una) ni con 2024 (1 año). Para comparar TRAIN contra validación, use el Sharpe (anualizado), el drawdown y las órdenes por semana; la ganancia solo indica el signo y el orden de magnitud dentro de un mismo periodo.

Modelo elegido (regla fijada antes de leer la corrida B): **T2_core15**, variables **core15**, **u = 2** unidades por componente.

Evidencia de la elección (pérdida INNER_A, media de semillas): T4_v4: 0.6423 (327 parámetros); T2_core15: 0.5517 (325 parámetros)

E1-T: arquitectura TemporalPolicyNet (temporal.py), 388 parámetros; compuerta: STOP_AT_E1.

E2-T: SAC (SB3), extractor temporal + net_arch [2]; presupuesto 100000 transiciones; 3 semillas por brazo.

## 1. Métricas por semilla y periodo

### E1-T BC+DAgger r3

| Política | Ganancia neta USD | Sharpe | Máx. DD % | Órdenes | Órdenes/sem | Acierto % | Profit factor | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN 2020–22 · semilla 1 | -9 806 | -2.09 | 98.9 | 2426 | 15.5 | 26.3 | 0.84 | sí |
| TRAIN 2020–22 · semilla 2 | -9 752 | -1.80 | 98.7 | 2340 | 14.9 | 29.6 | 0.91 | sí |
| TRAIN 2020–22 · semilla 3 | -9 694 | -1.88 | 98.1 | 1772 | 11.3 | 26.7 | 0.87 | sí |
| VAL A 2023S1 · semilla 1 | -5 577 | -2.44 | 68.6 | 415 | 16.0 | 26.7 | 0.80 | sí |
| VAL A 2023S1 · semilla 2 | -551 | 0.14 | 46.6 | 384 | 14.8 | 31.5 | 0.98 | sí |
| VAL A 2023S1 · semilla 3 | -2 816 | -0.86 | 43.4 | 295 | 11.3 | 28.8 | 0.87 | sí |
| VAL B 2023S2 · semilla 1 | -5 125 | -2.93 | 52.9 | 412 | 15.3 | 25.0 | 0.77 | sí |
| VAL B 2023S2 · semilla 2 | -5 998 | -4.03 | 62.1 | 388 | 14.4 | 28.4 | 0.71 | sí |
| VAL B 2023S2 · semilla 3 | -4 520 | -2.42 | 50.1 | 302 | 11.2 | 25.8 | 0.77 | sí |
| 2024 (una lectura) · semilla 1 | -7 616 | -3.06 | 79.6 | 822 | 15.5 | 26.2 | 0.72 | sí |
| 2024 (una lectura) · semilla 2 | -7 927 | -3.37 | 82.0 | 769 | 14.5 | 27.3 | 0.63 | sí |
| 2024 (una lectura) · semilla 3 | -5 798 | -2.10 | 63.1 | 612 | 11.5 | 26.6 | 0.78 | sí |

### E2-T SAC

| Política | Ganancia neta USD | Sharpe | Máx. DD % | Órdenes | Órdenes/sem | Acierto % | Profit factor | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN 2020–22 · semilla 1 | -8 186 | -1.04 | 84.9 | 901 | 5.7 | 45.4 | 0.80 | sí |
| TRAIN 2020–22 · semilla 2 | -8 494 | -0.60 | 93.3 | 617 | 3.9 | 28.2 | 0.87 | sí |
| TRAIN 2020–22 · semilla 3 | -7 917 | -0.38 | 86.9 | 506 | 3.2 | 52.2 | 0.88 | sí |
| VAL A 2023S1 · semilla 1 | 4 294 | 1.87 | 28.1 | 148 | 5.7 | 56.8 | 1.33 | sí |
| VAL A 2023S1 · semilla 2 | 545 | 0.49 | 37.2 | 95 | 3.7 | 30.5 | 1.03 | sí |
| VAL A 2023S1 · semilla 3 | -2 861 | -0.61 | 47.7 | 82 | 3.2 | 52.4 | 0.85 | sí |
| VAL B 2023S2 · semilla 1 | -185 | 0.09 | 25.4 | 138 | 5.1 | 47.1 | 0.98 | sí |
| VAL B 2023S2 · semilla 2 | -3 579 | -1.16 | 58.0 | 98 | 3.6 | 26.5 | 0.73 | sí |
| VAL B 2023S2 · semilla 3 | 71 | 0.33 | 34.8 | 86 | 3.2 | 58.1 | 1.00 | sí |
| 2024 (una lectura) · semilla 1 | -3 908 | -1.39 | 44.9 | 286 | 5.4 | 40.9 | 0.76 | sí |
| 2024 (una lectura) · semilla 2 | -6 971 | -2.13 | 71.5 | 217 | 4.1 | 27.2 | 0.70 | sí |
| 2024 (una lectura) · semilla 3 | 820 | 0.40 | 42.3 | 169 | 3.2 | 53.8 | 1.03 | sí |

### E2-T SAC+demos

| Política | Ganancia neta USD | Sharpe | Máx. DD % | Órdenes | Órdenes/sem | Acierto % | Profit factor | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN 2020–22 · semilla 1 | -7 456 | -0.33 | 90.3 | 783 | 5.0 | 52.7 | 0.79 | sí |
| TRAIN 2020–22 · semilla 2 | -7 922 | -0.37 | 86.3 | 506 | 3.2 | 66.4 | 0.87 | sí |
| TRAIN 2020–22 · semilla 3 | -8 632 | -0.62 | 94.0 | 609 | 3.9 | 29.2 | 0.87 | sí |
| VAL A 2023S1 · semilla 1 | 1 325 | 0.72 | 30.5 | 130 | 5.0 | 56.2 | 1.08 | sí |
| VAL A 2023S1 · semilla 2 | -3 994 | -1.12 | 57.6 | 85 | 3.3 | 65.9 | 0.78 | sí |
| VAL A 2023S1 · semilla 3 | 71 | 0.36 | 37.0 | 97 | 3.7 | 30.9 | 1.00 | sí |
| VAL B 2023S2 · semilla 1 | -5 490 | -2.49 | 66.8 | 122 | 4.5 | 52.5 | 0.55 | sí |
| VAL B 2023S2 · semilla 2 | -1 136 | -0.07 | 38.7 | 81 | 3.0 | 65.4 | 0.93 | sí |
| VAL B 2023S2 · semilla 3 | -2 986 | -0.87 | 55.0 | 94 | 3.5 | 30.9 | 0.77 | sí |
| 2024 (una lectura) · semilla 1 | -6 659 | -1.94 | 72.3 | 261 | 4.9 | 45.2 | 0.70 | sí |
| 2024 (una lectura) · semilla 2 | 26 | 0.25 | 45.3 | 176 | 3.3 | 67.6 | 1.00 | sí |
| 2024 (una lectura) · semilla 3 | -6 689 | -1.93 | 68.9 | 211 | 4.0 | 27.5 | 0.71 | sí |

## 2. TRAIN contra validación (media de semillas)

| Política | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Δ Sharpe (VAL media − TRAIN) | Ganancia TRAIN | Ganancia VAL A | Ganancia VAL B | DD % TRAIN | DD % VAL B | Órdenes/sem TRAIN | Órdenes/sem VAL B | Sharpe 2024 | Ganancia 2024 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-T BC+DAgger r3 | -1.92 | -1.06 | -3.13 | -0.17 | -9 751 | -2 981 | -5 214 | 98.6 | 55.0 | 13.9 | 13.6 | -2.84 | -7 114 |
| E2-T SAC | -0.67 | 0.58 | -0.25 | 0.84 | -8 199 | 659 | -1 231 | 88.4 | 39.4 | 4.3 | 4.0 | -1.04 | -3 353 |
| E2-T SAC+demos | -0.44 | -0.02 | -1.14 | -0.14 | -8 003 | -866 | -3 204 | 90.2 | 53.5 | 4.0 | 3.7 | -1.21 | -4 441 |
| Experto representable en el espacio de acción | 16.68 | 16.58 | 18.33 | 0.77 | 119 157 282 067 276 325 793 562 624 | 22 713 745 | 13 918 292 | 7.2 | 2.9 | 16.0 | 15.8 | 16.44 | 719 203 635 |
| Experto retrospectivo (techo, no causal) | 18.23 | 18.58 | 20.24 | 1.18 | 106 255 143 377 556 799 266 909 847 552 | 68 983 696 | 29 216 938 | 7.2 | 2.9 | 15.8 | 15.6 | 18.07 | 4 394 577 542 |
| Experto confirmado (causal) | -1.55 | -0.80 | -3.64 | -0.67 | -9 685 | -2 740 | -6 023 | 97.6 | 62.9 | 16.2 | 15.9 | -3.83 | -8 262 |
| Aleatoria con el prior de TRAIN | -1.29 | 0.31 | -3.27 | -0.19 | -9 482 | 352 | -5 938 | 96.7 | 63.9 | 10.9 | 10.0 | -1.64 | -5 957 |
| No operar | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0 |

## 3. Sensibilidad con el contrato de Musashi (≤ 1 entrada/día, 1:50), políticas congeladas

| Política | Ganancia neta USD | Sharpe | Máx. DD % | Órdenes | Órdenes/sem | Acierto % | Profit factor | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-T BC+DAgger r3 · TRAIN 2020–22 · semilla 20261019 | -2 539 | -0.32 | 40.8 | 755 | 4.8 | 41.3 | 0.94 | sí |
| E1-T BC+DAgger r3 · VAL A 2023S1 · semilla 20261019 | -158 | -0.04 | 14.2 | 125 | 4.8 | 42.4 | 0.98 | sí |
| E1-T BC+DAgger r3 · VAL B 2023S2 · semilla 20261019 | -243 | -0.18 | 12.2 | 127 | 4.7 | 44.1 | 0.96 | sí |
| E1-T BC+DAgger r3 · 2024 (una lectura) · semilla 20261019 | -1 342 | -0.74 | 20.2 | 258 | 4.9 | 40.3 | 0.88 | sí |
| E1-T BC+DAgger r3 · TRAIN 2020–22 · semilla 20261020 | 593 | 0.20 | 43.5 | 763 | 4.9 | 41.5 | 1.02 | sí |
| E1-T BC+DAgger r3 · VAL A 2023S1 · semilla 20261020 | 1 704 | 1.51 | 9.1 | 129 | 5.0 | 47.3 | 1.20 | sí |
| E1-T BC+DAgger r3 · VAL B 2023S2 · semilla 20261020 | 280 | 0.40 | 12.1 | 129 | 4.8 | 45.7 | 1.04 | sí |
| E1-T BC+DAgger r3 · 2024 (una lectura) · semilla 20261020 | -2 952 | -1.99 | 34.2 | 258 | 4.9 | 37.2 | 0.72 | sí |
| E1-T BC+DAgger r3 · TRAIN 2020–22 · semilla 20261021 | -4 490 | -0.82 | 55.9 | 580 | 3.7 | 39.1 | 0.88 | sí |
| E1-T BC+DAgger r3 · VAL A 2023S1 · semilla 20261021 | -2 225 | -2.42 | 23.1 | 100 | 3.8 | 33.0 | 0.68 | sí |
| E1-T BC+DAgger r3 · VAL B 2023S2 · semilla 20261021 | 1 628 | 1.65 | 10.8 | 104 | 3.9 | 49.0 | 1.26 | sí |
| E1-T BC+DAgger r3 · 2024 (una lectura) · semilla 20261021 | -979 | -0.60 | 22.2 | 207 | 3.9 | 37.7 | 0.89 | sí |
| SAC_20261019 · TRAIN 2020–22 | -4 431 | -1.01 | 47.2 | 738 | 4.7 | 40.7 | 0.79 | sí |
| SAC_20261019 · VAL A 2023S1 | 1 284 | 1.55 | 9.1 | 125 | 4.8 | 49.6 | 1.36 | sí |
| SAC_20261019 · VAL B 2023S2 | -62 | 0.01 | 15.7 | 121 | 4.5 | 41.3 | 0.99 | sí |
| SAC_20261019 · 2024 (una lectura) | -431 | -0.27 | 12.6 | 245 | 4.6 | 35.9 | 0.93 | sí |
| SAC_20261020 · TRAIN 2020–22 | -4 689 | -0.54 | 64.6 | 498 | 3.2 | 29.9 | 0.88 | sí |
| SAC_20261020 · VAL A 2023S1 | 296 | 0.34 | 19.8 | 85 | 3.3 | 29.4 | 1.04 | sí |
| SAC_20261020 · VAL B 2023S2 | -670 | -0.38 | 25.0 | 84 | 3.1 | 33.3 | 0.90 | sí |
| SAC_20261020 · 2024 (una lectura) | -3 982 | -2.15 | 40.6 | 185 | 3.5 | 27.6 | 0.66 | sí |
| SAC_20261021 · TRAIN 2020–22 | -4 567 | -0.44 | 58.2 | 465 | 3.0 | 50.5 | 0.89 | NO |
| SAC_20261021 · VAL A 2023S1 | -1 242 | -0.60 | 26.3 | 79 | 3.0 | 53.2 | 0.87 | sí |
| SAC_20261021 · VAL B 2023S2 | -722 | -0.37 | 20.6 | 77 | 2.9 | 53.2 | 0.91 | NO |
| SAC_20261021 · 2024 (una lectura) | 1 474 | 0.71 | 21.9 | 156 | 2.9 | 53.8 | 1.12 | NO |
| SAC_DEMO_20261019 · TRAIN 2020–22 | -3 785 | -0.37 | 61.4 | 439 | 2.8 | 36.2 | 0.84 | NO |
| SAC_DEMO_20261019 · VAL A 2023S1 | 502 | 0.48 | 16.2 | 72 | 2.8 | 44.4 | 1.11 | NO |
| SAC_DEMO_20261019 · VAL B 2023S2 | -2 133 | -1.72 | 33.8 | 74 | 2.7 | 44.6 | 0.61 | NO |
| SAC_DEMO_20261019 · 2024 (una lectura) | -3 416 | -1.77 | 38.6 | 147 | 2.8 | 27.2 | 0.59 | NO |
| SAC_DEMO_20261020 · TRAIN 2020–22 | -5 615 | -0.68 | 61.3 | 437 | 2.8 | 64.1 | 0.83 | NO |
| SAC_DEMO_20261020 · VAL A 2023S1 | -1 688 | -0.95 | 31.2 | 81 | 3.1 | 66.7 | 0.82 | sí |
| SAC_DEMO_20261020 · VAL B 2023S2 | -595 | -0.26 | 21.6 | 78 | 2.9 | 61.5 | 0.92 | NO |
| SAC_DEMO_20261020 · 2024 (una lectura) | 654 | 0.40 | 19.9 | 158 | 3.0 | 67.7 | 1.06 | NO |
| SAC_DEMO_20261021 · TRAIN 2020–22 | -5 520 | -0.72 | 68.3 | 498 | 3.2 | 29.7 | 0.85 | sí |
| SAC_DEMO_20261021 · VAL A 2023S1 | -288 | -0.02 | 21.1 | 86 | 3.3 | 30.2 | 0.97 | sí |
| SAC_DEMO_20261021 · VAL B 2023S2 | -205 | 0.00 | 25.0 | 81 | 3.0 | 34.6 | 0.97 | sí |
| SAC_DEMO_20261021 · 2024 (una lectura) | -3 512 | -1.82 | 37.2 | 175 | 3.3 | 28.6 | 0.70 | sí |

## 4. Parada temprana de E2-T

| Corrida | Parada | Transiciones del mejor checkpoint | Puntuación |
| --- | --- | --- | --- |
| SAC_20261019 | patience | 50000 | 0.41 |
| SAC_20261020 | patience | 30000 | -0.05 |
| SAC_20261021 | patience | 50000 | -0.50 |
| SAC_DEMO_20261019 | budget | 60000 | 0.19 |
| SAC_DEMO_20261020 | budget | 55000 | -0.75 |
| SAC_DEMO_20261021 | budget | 80000 | -0.13 |

