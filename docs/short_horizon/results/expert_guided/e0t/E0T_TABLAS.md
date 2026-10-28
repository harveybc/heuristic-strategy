# E0-T: imitabilidad del experto ZigZag v2 con el tronco temporal mínimo (corrida 1)

Estado: DESARROLLO. Nada es operable. 2023 = desarrollo conjunto; 2024 = desarrollo ya observado, leído una
vez y solo para el brazo elegido. TEST 2025 cerrado.

- Código: `tools/expert_guided_e0t.py`, `app/short_horizon/expert_guided/temporal.py` (commit `9fad910`).
- Diagramas: `arquitectura_E0T_u2.png` y los demás PNG de esta carpeta.
- Resultado completo: `run/E0T.json`; ledger de ensayos: `run/TRIAL_LEDGER.jsonl`.

## Diseño

| Elemento | Valor |
| --- | --- |
| Filas | Idénticas para todos los brazos (elegibles con ambos conjuntos de variables): TRAIN 14.429, INNER_A 2.461, INNER_B 2.436, 2024 4.871; 133 filas de TRAIN solo existen con las 15 (calentamiento del fractal w200) |
| Entrada | Ventana de 13 slots, paso 2, z-score causal sobre 256 filas, sin aplanar (salvo el comparador MLP) |
| Entrenamiento | AdamW lr 1e-3, wd 1e-4, dropout 0,1, lote 256, sin BatchNorm, CE sin pesos, máximo 200 épocas, parada por 0,5·(TRAIN + INNER_A) con paciencia 10 y min_delta 1e-8, mejores pesos restaurados; temperatura y umbral elegidos en INNER_A |
| Semillas | 20261019, 20261020, 20261021 |
| Cómputo | CPU del coordinador, un trabajo, tope 2 GiB, 1.076 s |

## Objetivo retrospectivo (anticipatorio, primario): pérdidas por semilla

| Brazo | Parámetros | Épocas (mejor) | Pérdida TRAIN | INNER_A | INNER_B | INNER_B − prior |
| --- | --- | --- | --- | --- | --- | --- |
| T2_v4 | 105 | 52(42) / 200(198) / 200(200) | 0,689 / 0,662 / 0,666 | 0,676 / 0,654 / 0,657 | 0,697 / 0,671 / 0,677 | −0,002 / −0,029 / −0,025 |
| T4_v4 | 327 | 163(153) / 200(199) / 157(147) | 0,648 / 0,647 / 0,646 | 0,644 / 0,642 / 0,642 | 0,664 / 0,663 / 0,663 | −0,034 / −0,035 / −0,035 |
| T8_v4 | 1.131 | 118(108) / 194(184) / 171(161) | 0,630 / 0,627 / 0,625 | 0,649 / 0,649 / 0,657 | 0,673 / 0,668 / 0,671 | −0,025 / −0,030 / −0,027 |
| T2_core15 | 325 | 200(198) / 200(194) / 200(198) | 0,559 / 0,552 / 0,551 | 0,564 / 0,563 / 0,559 | 0,580 / 0,580 / 0,574 | −0,118 / −0,118 / −0,124 |
| MLP_v4_flat (comparador) | 15.235 | 94(84) / 80(70) / 84(74) | 0,452 / 0,468 / 0,459 | 0,735 / 0,730 / 0,732 | 0,747 / 0,744 / 0,739 | **+0,050 / +0,046 / +0,041** |
| MLP 195 de E0 original (otras filas) | 33.539 | 51 / 48 / 55 | 0,393 / 0,393 / 0,380 | 0,619 / 0,610 / 0,615 | — | −0,065 / −0,073 / −0,063 |

La última fila viene de `e0/run/E0.json`: población de 14.562 filas de TRAIN, referencia histórica, no
estrictamente comparable fila a fila.

## Trading ejecutado (umbral elegido en INNER_A, SL/TP, unidades y frecuencia del negocio): Sharpe por semilla

| Brazo | TRAIN | INNER_A | INNER_B | 2024 (una lectura) |
| --- | --- | --- | --- | --- |
| T2_v4 (elegido por la regla v1) | −0,46 / −0,39 / −1,45 | 0,23 / −0,80 / −1,70 | −2,17 / −0,59 / 0,85 | −1,99 / 0,10 / −0,38 |
| T4_v4 | −0,30 / −1,19 / 0,09 | −0,53 / 0,26 / −0,79 | −1,05 / −1,67 / −0,14 | no leído |
| T8_v4 | 0,95 / 1,17 / 0,90 | −1,58 / −0,64 / −0,96 | −0,09 / 0,92 / −0,50 | no leído |
| T2_core15 | 1,11 / −0,47 / 0,16 | 0,72 / 0,38 / −0,27 | −1,62 / 0,45 / −1,83 | no leído |
| MLP_v4_flat | 6,50 / 4,85 / 7,88 | 0,12 / 1,73 / −0,02 | −2,33 / −0,99 / −0,84 | no leído |
| Experto retrospectivo (techo, no causal) | 18,13 | 18,58 | 20,24 | 18,07 |
| Experto confirmado (causal) | −1,57 | −0,98 | −3,58 | −3,88 |

Todos los puntos cumplen la frecuencia (≥ 3 órdenes/semana, ≤ 4/día).

## Regla de capacidad y compuerta

- **Regla v1 (pre-registrada):** eligió T2_v4. Rechazó T4_v4, que es mejor en INNER_A (0,642 frente a 0,662),
  porque su brecha pasó de −0,010 a −0,005.
- **Defecto de la regla:** una brecha negativa no es sobreajuste. Enmienda v2, declarada después de esta
  corrida: se compara solo la parte positiva de la brecha. Se aplica en la corrida B; esta corrida no se
  reinterpreta.
- **Compuerta en INNER_B:** G1 y G2 pasan en todos los brazos temporales; la MLP aplanada falla G1. Decisión
  formal: CONTINUE_TO_E1.

## Lectura

1. **La generalización mejoró.** Los troncos temporales tienen una brecha INNER_A − TRAIN entre −0,01 y +0,03.
   La MLP aplanada tiene +0,27 a +0,28 y queda peor que el prior en INNER_B.
2. **Las variables importan más que la capacidad.** Con el mismo tronco u = 2 y las mismas filas, las 15 de
   Musashi dan −0,12 frente al prior y las 4 v4 entre −0,002 y −0,029.
3. **u = 8 empieza a sobreajustar** (brecha +0,02 a +0,03).
4. **Límite de presupuesto alcanzado:** T2_v4 (2 de 3 semillas), T4_v4 (1 de 3) y T2_core15 (3 de 3)
   terminaron en 200 épocas con la mejor época en 194–200. Esos brazos se subestiman y se repiten en la
   corrida B con 1.000 épocas.
5. **Trading:** sin ganancia consistente en ningún brazo. Una mejor probabilidad todavía no se convierte en
   una estrategia rentable con la política de umbral.
