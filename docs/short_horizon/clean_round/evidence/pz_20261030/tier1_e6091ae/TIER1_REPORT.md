# PZ3: nivel 1 de la matriz, PZ canónica, salidas perfectas, full_f4 (código e6091ae)

**Corrida.** Es la misma cola de nivel 1 de la cadena formal: 39 celdas base, 9 S_ y 9 M_, que suman 135 registros con las semillas 0, 1 y
2. Genes PZ con sha256 `36e3e235…`. Corrió localmente bajo crispdm entre las 10:58 y las 13:32Z del 2026-10-30. Para leer los resultados
se usan `tools/strategy_pz_tier_report.py summary` y `diverge`; la segunda se aplicó a las 39 celdas base con semilla 0.

**Veredicto pre-registrado (§3).** 135 de 135 registros dan FAIL, así que **PZ no pasa la matriz**. Por eso la regla de decisión del
§5 ya no puede adoptar PZ con el punto canónico.

## Testigos (forma gobernante)

| testigo | PASS | FAIL | otro | valor PZ (mediana) | control, oráculo de mantener (mediana) |
|---|---|---|---|---|---|
| W2 apertura con verdad > 0 | 0 | 135 | | 0.339 | 1.0 |
| W2 lado equivocado | 0 | 135 | | 0.304 | 0.0 |
| W3 NOP | 0 | 135 (28 con estado DEFECT del control) | | 0.232 | 0.037 |
| W7 mejor (h, lado) | 0 | 135 | | 0.376 | 1.0 |
| W15 sin apertura de alto riesgo | 0 | 135 | | 0.447 | 0.0 |
| W4 cierre temprano justificado | 5 | 130 (35 DEFECT) | | 0.737 | 1.0 |
| W4 salidas por SL | 103 | 32 | | 0.075 | 0.0 |
| W10 ventana de spread x5 | 1 | 25 | | | |
| W1 lado | 116 | 15 | 4 CONTROL_CONFLICT | | |
| W8 cierre en el cambio | 33 | 44 | | | |
| W8 dirección vieja tras el retraso | 74 | 3 | | | |
| W17 sin apertura en ráfaga | 2 | 7 | | | |
| E2 / E3 / E4 (por lado) | 0 | 135 | | | |
| E2' regret con destino SL (por lado) | 0 | 129 | 6 DEFECT del control | | |
| E1 / E1' venta | 110 / 125 | 25 / 10 | | | |
| W0, W5, W6, W9, W11, W12, W13, W14 (9 de 9), W16, WA, E5, E6 (9 de 9), E7, E8, EP, Z1 | todos | 0 | | | |

**Comportamiento**

- 11.2 operaciones por semana (mediana).
- Neto 0.50 σ por operación (mediana por registro).
- Mezcla de horizontes ejecutados (mediana): h3 0.58, h6 0.08, h12 0.13, h24 0.21.
- `levels_clipped` = 0.
- Salidas en las 39 celdas base con semilla 0 (32 231 operaciones): **EARLY 0.860, SL 0.104, TIME 0.036, TP 0.000**.

## Primera divergencia frente al oráculo de verdad (39 celdas base, semilla 0)

- **Apertura.** Hay 55 ventanas de divergencia en los 39 registros (un registro puede tener más de una). En 53 de ellas PZ abre: 23
  donde el oráculo no abre, 16 donde abre el otro lado y 14 donde abre el mismo lado con otro horizonte. En las 2 restantes PZ no abre
  y el oráculo sí.
- **Valor de la orden elegida.** En 47 de las 53 aperturas divergentes, la orden que PZ eligió tiene una verdad del motor ≤ 0.
- **Señal en esas filas.** PZ abre con `p_up` unánime (1 o 0 en los cuatro niveles) y con un score G/SL de 4 a 11, porque el TP queda
  lejos y el SL es estrecho. Aun así, la orden **mantenida hasta su salida natural** pierde, típicamente −0.07 a −0.16 σ.
- **Cierres.** Una mediana de 0.25 de los cierres tempranos tiene un valor de mantener restante > 0.02 σ. De esos, 5 985 los dispara
  **solo la regla urgente**, cuando el nivel fino se pone en contra, y 1 279 la regla HOLD.

## Mecanismo (la causa, no un ajuste)

1. **El TP no puede llenarse.** Con salidas perfectas y `a_tp = 1`, el TP de PZ es exactamente el extremo del próximo pivote del nivel h:
   la etiqueta fav mide hasta el pivote y el TP se redondea un tick hacia adentro. El motor exige que el precio *atraviese* el TP límite
   en `limit_through` = 1 pip para llenarlo. En el pivote el precio llega al TP pero no lo atraviesa, por construcción. Resultado: TP = 0
   de 32 231 salidas. La orden mantenida termina en SL o en TIME, y por eso la verdad del motor de la orden elegida es negativa. Esto
   explica W2, W3, W7 y W15: el oráculo actúa sobre esa misma verdad y abre otra cosa, o nada.
2. **El SL es estrecho.** El SL es `adv_q75` (con salidas perfectas, la MAE exacta hasta el pivote) más 1 pip, así que tras el pivote
   cualquier retroceso lo toca.
3. **Toda la ganancia de PZ viene del cierre temprano.** Hay un 86 % de EARLY y un Sharpe mediano de 18.5, sobre equity compuesta: no
   se usa como evidencia. La regla urgente del nivel fino cierra posiciones que todavía valían mantener (E3, E4 y W4).

Nada de esto se corrige en el punto canónico: el pre-registro lo fija. `a_tp` < 1 (la grilla de PZ5 incluye 0.75) y `p_urg` son genes
de PZ5. Ningún testigo cambió.
