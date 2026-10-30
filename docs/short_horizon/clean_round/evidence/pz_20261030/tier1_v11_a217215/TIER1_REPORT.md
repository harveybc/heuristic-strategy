# PZ3 bis: nivel 1 de la matriz con PZ v1.1 (corrección de diseño declarada), canónica, salidas perfectas, full_f4 (código a217215)

**Corrida.** Es la misma cola de nivel 1: 135 registros, semillas 0 a 2, genes con sha `36e3e235…` y `design_version` 1.1. Corrió entre
las 14:20 y las 15:46Z del 2026-10-30: dos tercios en local y un tercio en el host, en una raíz propia.

**Veredicto pre-registrado.** 135 de 135 registros dan FAIL, así que **PZ v1.1 no pasa la matriz**.

## Qué cambió frente a v1.0

| | v1.0 | v1.1 |
|---|---|---|
| salidas TP / EARLY / SL / TIME (celdas base, semilla 0) | 0.000 / 0.860 / 0.104 / 0.036 | 0.659 / 0.128 / 0.116 (todas en cortos) / 0.098 |
| W2 apertura con verdad > 0 (mediana) | 0.339 | 0.866 (18 de 135 PASS) |
| W3 NOP | 0 / 135 PASS | 93 / 135 PASS |
| W7 mejor (h, lado) | 0.376 | 0.592 (3 de 135 PASS) |
| W15 | 0 / 135 | 18 / 135 PASS |
| operaciones por semana; neto σ por operación (medianas) | 11.2; 0.50 | 6.6; 0.92 |

## Fallas que quedan (forma gobernante)

| testigo | estado |
|---|---|
| W2 lado equivocado | 135 FAIL (mediana 0.214; control 0.0) |
| W7 | 132 FAIL |
| W4 cierre temprano justificado | 125 FAIL (111 con DEFECT del control) |
| W2 apertura con verdad > 0 | 117 FAIL |
| W15 | 117 FAIL |
| W1 lado | 52 FAIL, 9 CONTROL_CONFLICT |
| W3 | 42 FAIL |
| W8 en el cambio | 34 FAIL |
| W4 salidas por SL | 31 FAIL |
| W10 | 14 FAIL |
| W8 dirección vieja | 11 FAIL, 3 CONTROL_CONFLICT |
| W17 | 5 FAIL |
| EP_counts / E1' / E8 / Z1 de COMPRA | 135 FAIL |
| E2' de compra | 135 DEFECT del control |
| E4 compra / venta | 106 / 123 FAIL |
| E1 venta | 97 FAIL |
| E2' venta | 116 FAIL |
| E1' venta | 62 FAIL |
| E2 venta | 44 FAIL |
| E3 venta | 22 FAIL |
| E8 / EP / Z1 de venta | 18 FAIL |
| E5 | 7 / 5 FAIL |

## Primera divergencia frente al oráculo (39 celdas base, semilla 0)

- **Aperturas.** Hay 52 ventanas. En 48 PZ abre: en 27 el oráculo abre el otro lado, en 17 el mismo lado con otro horizonte y en 4 no abre
  nada. En las 4 restantes PZ no abre y el oráculo sí.
- **Valor de la orden elegida.** Solo 11 de las 48 órdenes elegidas tienen verdad ≤ 0. El resto son positivas, pero no las mejores.
- **Cierres.** Una mediana de 0.37 de los cierres tempranos tiene valor de mantener > 0.02 σ: 500 por la regla HOLD y 466 por la urgente.

## Mecanismos que quedan, reportados y NO corregidos

La corrección de diseño la decide el coordinador.

1. **La p de §3 y §4 es la dirección del PRÓXIMO pivote, no P(TP antes que SL) de la orden de cada lado.**
   - Cada orden apunta a su propio pivote. La compra apunta al próximo pico aunque antes venga un valle, y su SL (adv_q75 + colchón)
     cubre la MAE de ese valle.
   - Con salidas perfectas, esa compra llega a su TP con certeza, pero recibe p = p_up = 0 y G = −SL. PZ la descarta y abre el otro
     lado, con verdad positiva pero menor. Eso explica W2 lado equivocado, W7 y W1.
   - La regla HOLD usa la misma p' y cierra una posición cuyo pivote propio todavía llega cuando el próximo pivote del nivel es contrario.
     Eso explica W4 y E4.
2. **El SL de los cortos no incluye el spread del lado de salida.**
   - El stop de un corto se dispara en el ASK (`short_trigger_on_ask`), pero adv se mide sobre los máximos BID. Resultado: en 19 831
     operaciones hay 0 SL de compra y 11.6 % de SL de venta.
   - Con salidas perfectas, ningún episodio de COMPRA está destinado a SL: los EP_counts de compra tienen 0 de 30 en 135 de 135 y E1',
     E8, Z1 y E2' de compra quedan vacíos. Las ventas sí tienen esos episodios, por el spread.
   - Es la misma clase de defecto que la corrección (1) de v1.1, pero del lado del SL.
