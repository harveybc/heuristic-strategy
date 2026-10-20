# E3-M: especialistas modulares guiados por el experto ZigZag (cabeza 16, tronco temporal mínimo)

DESARROLLO, nada operable. Filas, rondas (BC + 3 DAgger), semillas, tronco y etiquetas idénticos en las cuatro variantes; elección en VAL A; VAL B reportado; 2024 leído una sola vez para la variante elegida. Cerradores con etiqueta de salida anticipada (SL antes que TP en retrospectiva) en TODAS las variantes, incluida JOINT, para no confundir modularización con definición de etiqueta. Árbitro de dirección elegido en VAL A entre probabilidad (umbral por tasa), ganancia esperada y ganancia/riesgo. 4090, 29 min.

Parámetros por módulo: {"JOINT": {"joint": 610}, "TASK": {"opener": 593, "closer": 542}, "DIR": {"long": 593, "short": 593}, "FULL": {"open_l": 576, "open_s": 576, "close_l": 542, "close_s": 542}}. Posiciones de exploración ronda 0: {'TRAIN': 13035, 'INNER_A': 2185}.

## Tabla por variante, semilla y periodo

| Variante | Semilla | Árbitro | Periodo | Sharpe | Ganancia USD | Máx. DD % | Órdenes | Órdenes/sem | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| JOINT | 1 | argmax | TRAIN 2020–22 | -1.07 | -8 959 | 94 | 2110 | 13.4 | sí |
| JOINT | 1 | argmax | VAL A 2023S1 | -4.83 | -7 500 | 77 | 361 | 13.9 | sí |
| JOINT | 1 | argmax | VAL B 2023S2 | -3.06 | -5 508 | 59 | 354 | 13.1 | sí |
| JOINT | 2 | argmax | TRAIN 2020–22 | -2.46 | -9 790 | 98 | 2147 | 13.7 | sí |
| JOINT | 2 | argmax | VAL A 2023S1 | -3.08 | -5 998 | 64 | 367 | 14.1 | sí |
| JOINT | 2 | argmax | VAL B 2023S2 | -4.58 | -5 688 | 59 | 374 | 13.9 | sí |
| JOINT | 3 | argmax | TRAIN 2020–22 | -2.05 | -9 447 | 96 | 1605 | 10.2 | sí |
| JOINT | 3 | argmax | VAL A 2023S1 | -1.59 | -3 311 | 40 | 269 | 10.3 | sí |
| JOINT | 3 | argmax | VAL B 2023S2 | -1.11 | -2 182 | 24 | 269 | 10.0 | sí |
| TASK | 1 | prob (tasa 0.18) | TRAIN 2020–22 | 0.35 | 536 | 85 | 1785 | 11.4 | sí |
| TASK | 1 | prob (tasa 0.18) | VAL A 2023S1 | -1.15 | -3 331 | 50 | 294 | 11.3 | sí |
| TASK | 1 | prob (tasa 0.18) | VAL B 2023S2 | -1.99 | -4 794 | 58 | 292 | 10.8 | sí |
| TASK | 2 | prob (tasa 0.02) | TRAIN 2020–22 | 0.00 | 0 | 0 | 0 | 0.0 | NO |
| TASK | 2 | prob (tasa 0.02) | VAL A 2023S1 | 0.00 | 0 | 0 | 0 | 0.0 | NO |
| TASK | 2 | prob (tasa 0.02) | VAL B 2023S2 | 0.00 | 0 | 0 | 0 | 0.0 | NO |
| TASK | 3 | prob (tasa 0.03) | TRAIN 2020–22 | -1.06 | -8 879 | 94 | 1416 | 9.0 | sí |
| TASK | 3 | prob (tasa 0.03) | VAL A 2023S1 | 0.63 | 1 003 | 30 | 256 | 9.8 | sí |
| TASK | 3 | prob (tasa 0.03) | VAL B 2023S2 | -0.92 | -2 236 | 43 | 209 | 7.7 | sí |
| DIR | 1 | prob (tasa 0.18) | TRAIN 2020–22 | -0.75 | -7 400 | 81 | 586 | 3.7 | sí |
| DIR | 1 | prob (tasa 0.18) | VAL A 2023S1 | -0.14 | -652 | 27 | 89 | 3.4 | sí |
| DIR | 1 | prob (tasa 0.18) | VAL B 2023S2 | -0.30 | -760 | 16 | 96 | 3.6 | sí |
| DIR | 2 | prob (tasa 0.15) | TRAIN 2020–22 | -0.18 | -5 263 | 80 | 2163 | 13.8 | sí |
| DIR | 2 | prob (tasa 0.15) | VAL A 2023S1 | -2.22 | -4 493 | 61 | 390 | 15.0 | sí |
| DIR | 2 | prob (tasa 0.15) | VAL B 2023S2 | -2.26 | -4 944 | 56 | 374 | 13.9 | sí |
| DIR | 3 | prob (tasa 0.04) | TRAIN 2020–22 | -0.18 | -5 427 | 88 | 2040 | 13.0 | sí |
| DIR | 3 | prob (tasa 0.04) | VAL A 2023S1 | -2.35 | -5 286 | 57 | 346 | 13.3 | sí |
| DIR | 3 | prob (tasa 0.04) | VAL B 2023S2 | -3.55 | -5 529 | 61 | 321 | 11.9 | sí |
| FULL | 1 | prob (tasa 0.02) | TRAIN 2020–22 | -0.11 | -4 317 | 76 | 917 | 5.8 | sí |
| FULL | 1 | prob (tasa 0.02) | VAL A 2023S1 | 1.00 | 1 918 | 27 | 160 | 6.2 | sí |
| FULL | 1 | prob (tasa 0.02) | VAL B 2023S2 | -3.61 | -5 573 | 62 | 144 | 5.3 | sí |
| FULL | 1 | prob (tasa 0.02) | 2024 (una lectura) | -0.54 | -2 220 | 43 | 297 | 5.6 | sí |
| FULL | 2 | prob (tasa 0.12) | TRAIN 2020–22 | -0.69 | -6 694 | 72 | 551 | 3.5 | sí |
| FULL | 2 | prob (tasa 0.12) | VAL A 2023S1 | 0.42 | 453 | 26 | 79 | 3.0 | sí |
| FULL | 2 | prob (tasa 0.12) | VAL B 2023S2 | 0.04 | -344 | 23 | 92 | 3.4 | sí |
| FULL | 2 | prob (tasa 0.12) | 2024 (una lectura) | -0.05 | -460 | 26 | 165 | 3.1 | sí |
| FULL | 3 | prob (tasa 0.08) | TRAIN 2020–22 | 0.64 | 7 414 | 63 | 573 | 3.6 | sí |
| FULL | 3 | prob (tasa 0.08) | VAL A 2023S1 | 1.17 | 1 820 | 14 | 93 | 3.6 | sí |
| FULL | 3 | prob (tasa 0.08) | VAL B 2023S2 | -0.84 | -1 396 | 23 | 88 | 3.3 | sí |
| FULL | 3 | prob (tasa 0.08) | 2024 (una lectura) | 0.40 | 782 | 21 | 200 | 3.8 | sí |

## Medias de semillas

| Variante | Modelos | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Sharpe 2024 | DD % VAL B | Órdenes/sem VAL B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JOINT | 1 | -1.86 | -3.17 | -2.92 | nan | 47 | 12.3 |
| TASK | 2 | -0.24 | -0.17 | -0.97 | nan | 34 | 6.2 |
| DIR | 2 | -0.37 | -1.57 | -2.04 | nan | 44 | 9.8 |
| FULL | 4 | -0.05 | 0.86 | -1.47 | -0.06 | 36 | 4.0 |

## Referencias (misma contabilidad)

| Política | TRAIN | VAL A | VAL B | 2024 |
| --- | --- | --- | --- | --- |
| Experto retrospectivo (techo, no causal) | 18.23 | 18.58 | 20.24 | 18.07 |
| Experto confirmado (causal) | -1.55 | -0.80 | -3.64 | -3.83 |
| No operar | 0.00 | 0.00 | 0.00 | 0.00 |

Decisión: **FULL** (highest seed-mean INNER_A Sharpe among all-valid; ties -> fewer models).

## Lectura

1. La modularización ayuda frente a la política única: JOINT pierde en todo (VAL A −3,2, VAL B −2,9) y FULL es la mejor en VAL A (+0,86) y la menos mala en VAL B (−1,47), con drawdowns menores y 3–6 órdenes/semana.
2. Fuera de la muestra de selección no hay ventaja: VAL B negativo en 2 de 3 semillas; 2024 media −0,06 (−0,54 / −0,05 / +0,40). Es el resultado más cercano a cero de toda la escalera (E2-T SAC: −1,04 en 2024), no una ganancia.
3. TASK: una semilla no opera (umbral elegido en VAL A deja pasar cero órdenes): la regla de frecuencia la invalida, como debe ser.
4. El límite sigue siendo la señal del experto en estas entradas, no la arquitectura ni la modularización.
