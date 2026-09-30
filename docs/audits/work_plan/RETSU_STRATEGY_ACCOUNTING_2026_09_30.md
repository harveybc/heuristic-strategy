# Convención de caja del sucesor (2026-09-30)

Padre conservado: `781022a4702244d866cf86bb69343673340e9eff`. La evidencia `docs/audits/evidence/STRATEGY_MICRO_20260930/` no se reescribió. Las cifras de ese commit se citan como LEGACY. Lo medido aquí está en `docs/audits/evidence/STRATEGY_ACCOUNTING_20260930/`.

## Convenio

La moneda de la cuenta es la moneda cotizada. El nocional es unidades por precio. El margen es el nocional dividido por el apalancamiento. Fracción de margen 0.05, apalancamiento 100, capital 10000. Quinientos de margen permiten hasta 50000 de nocional antes de costos: 50000 unidades si el precio es 1 y 25000 si el precio es 2. El mínimo de 10000 no eleva el margen. Si el tamaño permitido queda por debajo del mínimo, la orden es Margin y no hay posición.

El broker no usa el parámetro `margin` de CommInfo como fracción. Ese parámetro, con `commtype` vacío, cambia la comisión a un monto fijo por unidad. El sucesor deja `margin=None`, fija `commtype=COMM_PERC`, `percabs=True`, `stocklike=True` y `leverage=100`, y pone `shortcash=False`. El largo y el corto depositan `abs(unidades) * precio / 100`. El efectivo que un corto recibiría por el nocional no se acredita y no es beneficio. Al cierre se libera el colateral.

Un solo libro registra spread, slippage, comisión y swap. Spread 2 pips y slippage 1 pip, pip 0.00001, van dentro del precio del fill (`(2+1)*0.00001/2` por lado) y no se vuelven a restar de la caja. La comisión es por lado: `abs(tamaño) * 0.00007 * precio`. No es una tarifa fija de 7 unidades ni un round-trip de 7. El 0.00007 es `commission_per_lot / 100000`, el mismo cociente del legacy. El swap es 10 por lote de 100000 unidades por 24 horas, en las dos direcciones. El reloj es el delta de timestamps UTC desde el fill de entrada hasta el fill de cierre, no un conteo de barras. Cada tramo se debita de la caja antes de la decisión que lee ese saldo.

TP 0.9, SL 2.0, variante E, decisión solo al cierre y fill de mercado en la apertura siguiente. `stop()` llama a `close()` y eso no fabrica un fill ni liquida la posición pendiente.

## Cuatro brazos

Mismo soporte que el legacy. Orígenes 2019-05-01 00:00, 04:00, 08:00 y 09:00 UTC. MAE macro: media no ponderada de doce horizontes. Naive pareado, macro 0.009208333333333327, igual en los cuatro. MAE ideal 0 en cada horizonte. `persistence/persistence` iguala al naive. MAE igual no es la misma decisión: los brazos planos no abren y los brazos con familia larga ideal sí.

| Brazo | Cierres | PnL realizado | Caja | Equity | Pendiente | MAE macro | MAE corto | MAE largo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal/ideal | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0 | 0 | 0 | 0 |
| persistence/ideal | 3 | 1965.8373773999754 | 11965.837377399974 | 11965.837377399974 | 0 | 0.005312500000000002 | 0.010625000000000004 | 0 |
| ideal/persistence | 0 | 0 | 10000 | 10000 | 0 | 0.0038958333333333254 | 0 | 0.007791666666666651 |
| persistence/persistence | 0 | 0 | 10000 | 10000 | 0 | 0.009208333333333327 | 0.010625000000000004 | 0.007791666666666651 |

Hueco de caja en plano: 0 en ideal/ideal y en los dos brazos sin trades. En persistence/ideal, -1.5916157281026244e-12. Hueco de equity 0 en los cuatro. El PnL realizado ya resta comisión y swap una sola vez. Spread y slippage quedan dentro del fill.

A las 09:00 los dos brazos con trades están largos 63035 unidades, no 1e6. La mezcla 0.6*0.99+0.4*1.003 sigue bajo el stop 0.9998 y la familia corta pide el cierre en ideal/ideal. En persistence/ideal la petición a esa hora es ninguna y el stop cierra a las 10:00. Predicciones planas de la familia larga no abren.

## Fills frente al legacy

LEGACY operó con caja 2000000 y tamaño 1e6 porque el broker simulado exigía el nocional completo. Mismos relojes, mismo lado y mismo precio en los seis fills de ideal/ideal y de persistence/ideal. Cambia el tamaño.

ideal/ideal, tamaños del sucesor: 49999, -49999, -56328, 56328, 63035, -63035. El primer fill sigue en 2019-05-01 01:00 a 1.000015. La petición era 50000; el slippage deja el precio en 1.000015 y el guardia recorta a 49999, con margen usado 499.99749985 frente a un presupuesto de 500. Los tamaños posteriores suben porque el presupuesto es el 5% de la equity corriente, no 500 congelados: 56328 y 63035. persistence/ideal repite esos tamaños y cierra el tercero a las 11:00 a 0.989985, igual que el reloj LEGACY, con tamaño -63035. Los dos brazos planos tienen 0 fills, como el legacy.

El primer trade cerrado de ideal/ideal tiene comisión 7.10485790000007 (los dos lados) y swap 0.41665833333333335. Ese swap coincide con 2/24 de un lote parcial de 49999, el tramo desde el fill de las 01:00 hasta el fill de las 03:00. `duration_bars` sigue en 3 y no es el reloj del swap. Un fixture con barras a las 00:00, 01:00 y 05:00 debitó 4 horas, no una barra.

Sonda `broker_session_end` 2019-05-01 01:00, capital 10000: fill 1.000015, tamaño 49999, cierres 0, realizado 0, pendiente 49999, caja 9496.50251765105, equity 10045.74903250104, hueco de equity 0. `stop()` no obtiene fill. LEGACY en esa sonda tenía tamaño 1e6, realizado 0, equity 2000914.99895 y caja 999914.9989499999. La observación LEGACY a caja 10000 (un corto, PnL 396.3219999999965) no se repite subiendo el capital.

## Estados y harness

En los brazos con trades quedaron Accepted, Submitted, Completed e in_flight. Margin, Rejected y Cancelled se ejercieron en fixtures del mismo pytest: sin posición falsa y sin cierre falso. Una orden cancelada no completa. Un rechazo deja la posición en cero.

`app/factorial_harness.py` describe 18 celdas: corto variable con largo ideal fijo, y al revés; persistencia, ruido con MAE equivalente e ideal. Semillas pareadas 42, 43 y 44. Escalas solo en DEV. `execute_manifest` levanta `B0_NOT_STARTED` y no corre celdas. Holdout y presupuesto B0 quedan fuera. El techo histórico de 3600 s de CPU de las 241 celdas no es presupuesto de B0. B0_NOT_STARTED. No se corrió mercado, broker real, Postgres, Metabase, puertos ni GPUs.

## Gasto medido

`crispdm-run -m 2G -t 120s` con `/usr/bin/time` sobre los tres pytest: 41 pasaron, 0 fallaron. Pared 2.99 s. Usuario 2.75 s. Sistema 0.23 s. CPU 2.98 s.
