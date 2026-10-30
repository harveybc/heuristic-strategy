# PZ3 ter: nivel 1 de la matriz con PZ v1.2 (segunda corrección declarada), canónica, salidas perfectas, full_f4 (código 5fa317e)

**Corrida.** Misma cola de nivel 1: 135 registros, semillas 0 a 2, `design_version` 1.2, q_sl 0.75. Corrió el 2026-10-30 entre las
16:58 y las 18:24Z, en local y en el host (raíz propia).

**Veredicto pre-registrado.** 135 de 135 registros dan FAIL, así que **PZ v1.2 no pasa la matriz**.

## Evolución por versión

Salvo donde se indica, las cifras son medianas, o PASS sobre 135 registros.

| | v1.0 | v1.1 | v1.2 |
|---|---|---|---|
| salidas TP / EARLY / SL / TIME (celdas base, semilla 0) | 0.000 / 0.860 / 0.104 / 0.036 | 0.659 / 0.128 / 0.116 / 0.098 | 0.730 / 0.083 / 0.060 / 0.126 |
| salidas SL de compra / de venta | (no medido) | 0 / 0.116 | 0 / 0.060 |
| W2 apertura con verdad > 0 | 0.339 (0 PASS) | 0.866 (18) | 0.879 (23) |
| W2 lado equivocado | 0.304 (0) | 0.214 (0) | 0.158 (0) |
| W7 | 0.376 (0) | 0.592 (3) | 0.658 (18) |
| W3 PASS | 0 | 93 | 121 |
| W4 cierre justificado: valor, PASS | 0.737, 5 | 0.593, 10 | 0.955, 58 |
| W4 salidas por SL, PASS | 103 | 104 | 121 |
| W1 PASS | 116 | 74 | 99 |
| W15 PASS | 0 | 18 | 23 |
| cierres tempranos con valor de mantener > 0.02 σ | 0.25 | 0.37 | **0.05** |
| operaciones por semana; σ por operación | 11.2; 0.50 | 6.6; 0.92 | 5.7; 1.02 |

## Fallas que quedan (forma gobernante)

| testigo | estado |
|---|---|
| W2 lado equivocado | 135 FAIL |
| EP_counts / E1' / E8 / Z1 de COMPRA, E1' de venta | 135 FAIL |
| E2' de compra | 135 DEFECT del control |
| W7 | 117 FAIL |
| E2' venta | 117 FAIL |
| W2 apertura con verdad > 0 | 112 FAIL |
| W15 | 112 FAIL |
| E4 venta | 112 FAIL |
| E1 venta | 106 FAIL |
| W4 justificado | 77 FAIL, todos DEFECT del control |
| W1 | 34 FAIL, 2 CONTROL_CONFLICT |
| W8 en el cambio | 34 FAIL |
| E2 venta | 20 FAIL |
| EP / E8 / Z1 de venta | 18 FAIL |
| W3 | 14 FAIL |
| W4 SL | 14 FAIL |
| W8 dirección vieja | 5 FAIL, 3 CONTROL_CONFLICT |
| W17 | 5 FAIL |
| W10 | 4 FAIL |
| E3 venta | 3 FAIL |

## Primera divergencia (39 celdas base, semilla 0)

- **Aperturas.** Hay 50 ventanas y en las 50 PZ abre: en 25 el oráculo abre el mismo lado con otro horizonte, en 23 el otro lado y en 2
  no abre nada.
- **Verdad de lo elegido.** Solo 8 de las 50 órdenes elegidas tienen verdad ≤ 0. Pero **en 45 de 50 la orden del oráculo tiene una verdad
  neta mayor** que la de PZ.
- **Cierres.** Los cierres injustificados bajan a una mediana de 0.05: 81 por la rama de TP inalcanzable y 1 por SL primero.

## Mecanismos que quedan (reportados, no corregidos; decide el coordinador)

1. **Métrica de ranking frente a métrica del testigo.**
   - §3 ordena por ganancia por unidad de riesgo, score = G/SL; con salidas perfectas, p_s = 1 y G/SL = (TP − costo)/SL.
   - W7 y W2 lado equivocado juzgan por la verdad neta (σ) de la mejor orden permitida. PZ elige de forma sistemática la orden de mayor
     cociente TP/SL, no la de mayor neto. Ejemplo, B_trans_down_pause_down_late s0: la compra tiene score 7.85 y neto 0.76; la venta,
     score 1.54 y neto 0.89.
   - Es la elección de diseño del propietario ("ganancia por riesgo"), no un error de geometría. Contra estos testigos sellados es una
     discrepancia de objetivo.
2. **Episodios de salida vacíos con salidas perfectas.**
   - Con salidas perfectas el SL de PZ cubre la MAE real hasta el pivote propio, así que ninguna orden de su envolvente está destinada a SL
     (0 SL de compra en 16 991 operaciones). Los EP_counts de compra quedan en 0 de 30 en los 135 registros, y E1', E8, Z1 y E2' de compra
     quedan vacíos.
   - Las ventas tienen algunos episodios destinados a SL: el spread de horas posteriores puede superar al de la hora de la decisión
     (0.060 de SL en ventas).
   - Esto es una vacuidad del instrumento EXIT_SPEC para una estrategia cuyo SL ideal es retrospectivamente exacto. EP_counts "nunca se
     excusa", así que se informa como mecanismo.
