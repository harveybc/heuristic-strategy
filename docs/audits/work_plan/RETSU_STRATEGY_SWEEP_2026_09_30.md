# Ejecutor sintético del barrido (2026-09-30)

Padre conservado: `e7966f33fca9946d6b2b9e45e28bc5f8ba20f6d8`. No se reescribió `781022a` ni la evidencia `STRATEGY_MICRO_20260930` ni `STRATEGY_ACCOUNTING_20260930`. Lo medido aquí está en `docs/audits/evidence/STRATEGY_SWEEP_20260930/`.

Esto es un fixture sintético con el plugin real y la contabilidad reconciliada. No es una utilidad financiera. No se afirma que el corto importe más que el largo. Una persistencia en el largo que impide entrar en este fixture no demuestra que el corto carezca de utilidad. Persistencia real y ruido con MAE parecido siguen siendo controles distintos. No se eligieron parámetros por PnL. TP 0.9, SL 2.0 y la variante E no se rediseñaron. El capital sigue en 10000. B0 no arrancó. El techo histórico de 3600 s de CPU de las 241 celdas no es presupuesto.

`app/factorial_harness.py` sigue siendo el manifiesto que no corre: fija la otra familia en ideal, `execute_manifest` sigue rechazando y `family_input` sigue renormalizando cada vector por su propio MAE. Esas son limitaciones anteriores a esta ejecución, no resultados de una campaña. El ejecutor de este commit es `app/sweep_executor.py`.

## Protocolo declarado antes de medir

Dos orientaciones: corto variable con largo fijo, y largo variable con corto fijo. El contraste principal fija la otra familia en persistencia real. El ideal es un control adicional, no el único fijo. Persistencia e ideal no tienen réplicas: semilla e intensidad nulas. El ruido sí lleva semillas pareadas 42, 43 y 44, las mismas entre configuraciones. Las intensidades son 0.5, 1.0 y 2.0. No son las 21 razones del diseño histórico.

Doce horizontes, 1-6 y 24, 48, 72, 96, 120, 144. Es el protocolo de horizontes completo de este fixture, no un subconjunto. El soporte común son los cuatro orígenes del microensayo: 2019-05-01 00:00, 04:00, 08:00 y 09:00 UTC. El naive se calcula sobre esas mismas filas en cada celda corrida. El macro naive es 0.009208333333333327 en las 44.

La escala sale de DEV, fuera de esa población. Son 120 orígenes sintéticos. Los timestamps consumidos van de 2019-04-20 00:00 UTC a 2019-04-30 23:00 UTC, 264 marcas, todas anteriores al primer origen puntuado y a 2019-05-16 00:00 UTC. Cambiar un cierre puntuado, incluido el de la barra reservada, no cambia la escala. La escala por horizonte, media del valor absoluto del residual, es:

| Horizonte | Escala DEV |
| --- | ---: |
| 1 | 0.008514093220515212 |
| 2 | 0.012747456374839048 |
| 3 | 0.014942414375694558 |
| 4 | 0.0170617596344584 |
| 5 | 0.01865704776636292 |
| 6 | 0.02227804648287624 |
| 24 | 0.020372727584010018 |
| 48 | 0.026722742253317826 |
| 72 | 0.01580462259312427 |
| 96 | 0.010524143577760369 |
| 120 | 0.023107867095661615 |
| 144 | 0.024217870760695353 |

El ruido entra solo en la predicción de la familia variable. La base es el precio futuro de esa hora. El objetivo de esa hora es intensidad por escala DEV. Sigma es objetivo dividido por 0.7978845608028654, la media absoluta de una normal estándar. El choque es sigma por un normal estándar. No se divide el vector por su MAE muestral. La semilla de la configuración no basta: el sorteo mezcla semilla, origen UTC, familia y horizonte, así que el mismo 42 no repite el vector en otro origen. La intensidad no entra en la semilla: el mismo sorteo se escala. Persistencia e ideal no sortean.

El manifiesto declara 44 celdas: 4 combinaciones de orientación y fijo, por 2 baselines deterministas, más 4 por 3 intensidades por 3 semillas. La regla de corte, escrita antes del PnL, usa solo la pared de la celda sonda. Presupuesto interno 80 s, por debajo del tope de 120 s del lanzador. Si 44 por esa pared cupiera, se corría el protocolo declarado. Si no, una rebanada de una semilla, luego cinco celdas, luego solo la sonda. 3600 s no entra en esa cuenta.

## Gasto medido

Lanzador, con `/usr/bin/time` dentro del tope de memoria:

`crispdm-run -m 2G -t 120s -n strategy-sweep-20260930 -- /usr/bin/time -f 'WALL %e USER %U SYS %S' env PYTHONPATH=./:<trading-contracts>/src python -m pytest tests/unit_tests/test_strategy_sweep_20260930.py -q`

6 pasaron, 0 fallaron. Pared 6.87 s. Usuario 6.53 s. Sistema 0.33 s. CPU 6.86 s. Pytest imprimió 6.46 s.

La celda sonda, `short_var_long_fix_persistence_noise_i1p0_s42`, midió pared 0.1404030870180577 s y CPU 0.14040417000000005 s. Por 44 cabe en 80 s, así que la rebanada fue `full`. No quedó ninguna de las 44 solo declarada. Un calentamiento de dos barras, que no es celda, midió pared 0.06565224300720729 s y CPU 0.06564445100000027 s. La suma de los 44 relojes de celda es pared 4.370826597994892 s y CPU 4.3701428270000005 s. `execute_declared`, calentamiento incluido, midió pared 4.447364746010862 s y CPU 4.446620553 s.

## MAE logrado frente al objetivo

El MAE de los doce horizontes y el MAE de los horizontes que sí tienen objetivo no son el mismo agregado. El primero mezcla la familia fija. El segundo es el que se compara con el objetivo. En la sonda la familia variable es el corto a intensidad 1 y semilla 42, y el largo fijo es persistencia. El largo no tiene objetivo de ruido: su MAE logrado es el naive y la diferencia queda nula a propósito, no escondida.

| Horizonte | Logrado | Objetivo | Diferencia | Naive | Escala DEV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.004578939976795093 | 0.008514093220515212 | -0.003935153243720119 | 0.006000000000000005 | 0.008514093220515212 |
| 2 | 0.016423022912856133 | 0.012747456374839048 | 0.0036755665380170856 | 0.016750000000000015 | 0.012747456374839048 |
| 3 | 0.012681086343665882 | 0.014942414375694558 | -0.0022613280320286765 | 0.013750000000000012 | 0.014942414375694558 |
| 4 | 0.02246312811152862 | 0.0170617596344584 | 0.00540136847707022 | 0.01050000000000001 | 0.0170617596344584 |
| 5 | 0.027399164957719574 | 0.01865704776636292 | 0.008742117191356653 | 0.00874999999999998 | 0.01865704776636292 |
| 6 | 0.014687310754259497 | 0.02227804648287624 | -0.0075907357286167446 | 0.008000000000000007 | 0.02227804648287624 |
| 24 | 0.006749999999999978 | nulo | nulo | 0.006749999999999978 | 0.020372727584010018 |
| 48 | 0.013000000000000012 | nulo | nulo | 0.013000000000000012 | 0.026722742253317826 |
| 72 | 0.007500000000000007 | nulo | nulo | 0.007500000000000007 | 0.01580462259312427 |
| 96 | 0.00649999999999995 | nulo | nulo | 0.00649999999999995 | 0.010524143577760369 |
| 120 | 0.006749999999999978 | nulo | nulo | 0.006749999999999978 | 0.023107867095661615 |
| 144 | 0.006249999999999978 | nulo | nulo | 0.006249999999999978 | 0.024217870760695353 |

En esa sonda el MAE de los doce es 0.012081887754735392. El MAE de los seis horizontes con objetivo es 0.016372108842804132. El objetivo de esos seis es 0.015700136309124396. No coinciden. El naive de los doce es 0.009208333333333327. El horizonte 2 queda cerca del naive y no por eso el ruido es la persistencia: la predicción sigue siendo el futuro más el choque. Cada celda guarda el mismo desglose en `RESULTS.json`.

## Celdas corridas

44 declaradas, 44 corridas, 0 solo declaradas. Capital 10000, fracción de margen 0.05 de la equity corriente, apalancamiento 100, `shortcash` falso. La comisión es por lado, `abs(tamaño) * 0.00007 * precio`. El swap usa el delta de timestamps UTC. Spread y slippage van dentro del fill y no se vuelven a restar. El hueco de equity más grande en valor absoluto fue 1.8189894035458565e-12. Donde el libro quedó plano, el hueco de caja no nulo fue -1.5916157281026244e-12 en las dos celdas de persistencia corta con largo ideal, y 8.640199666842818e-12 en las dos de largo ruidoso a intensidad 0.5 y semilla 44 que sí cerraron. El resto de huecos de equity y de caja plana midió 0. `stop()` no fabricó fill en ninguna celda.

El primer recorte, cuando hay entrada, pide 50000 y admite 49999. El fill mide tamaño 49999 y precio 1.000015. No hay posición de 1e6. Las celdas que cierran tres veces y dejan el largo en ideal repiten ese libro; el ruido solo en el corto no cambió esos tres cierres en este fixture. Eso es la regla de entrada, que lee el largo, no una medida de que el corto no sirva. Cuando el largo es persistencia y el corto es persistencia o ideal, no hay entradas: mismo hecho ya visto en la contabilidad sintética anterior, y tampoco dice nada sobre la utilidad del corto.

Varias celdas con el largo ruidoso se quedan abiertas. `stop()` no les añade un fill. Quedan 49999 unidades, caja 9464.836484317724 y equity 10164.07999916772, con un solo fill, salvo tres libros distintos: pendiente -56328 con un cierre y PnL 1490.9485137666582; pendiente 54952 con un cierre y PnL 990.5768547333245; y el cierre plano de PnL -510.53812236667136. Esos números están en la tabla. No se interpretan como ventaja de una familia.

`MAE12` es la media de los doce. `MAE obj` y `Objetivo` son la media de los horizontes que tienen objetivo. Nulo significa que la celda no tiene objetivo de ruido. Naive macro, en todas: 0.009208333333333327.

| Celda | Cierres | PnL | Caja | Equity | Pendiente | MAE12 | MAE obj | Objetivo | Comisión | Swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_var_long_fix_persistence_persistence_ina_sna | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.009208333333333327 | nulo | nulo | 0.0 | 0.0 |
| short_var_long_fix_persistence_ideal_ina_sna | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.0038958333333333254 | 0.0 | 0.0 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i0p5_s42 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.00798886054403436 | 0.00818605442140207 | 0.007850068154562198 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i0p5_s43 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.00794036666605688 | 0.008089066665447111 | 0.007850068154562198 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i0p5_s44 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.007752693567199491 | 0.007713720467732332 | 0.007850068154562198 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i1p0_s42 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.012081887754735392 | 0.016372108842804132 | 0.015700136309124396 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i1p0_s43 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.011984899998780442 | 0.016178133330894236 | 0.015700136309124396 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i1p0_s44 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.011609553801065693 | 0.015427440935464734 | 0.015700136309124396 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i2p0_s42 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.020267942176137466 | 0.032744217685608284 | 0.03140027261824879 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i2p0_s43 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.02007396666422757 | 0.032356266661788494 | 0.03140027261824879 | 0.0 | 0.0 |
| short_var_long_fix_persistence_noise_i2p0_s44 | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.019323274268798037 | 0.030854881870929423 | 0.03140027261824879 | 0.0 | 0.0 |
| short_var_long_fix_ideal_persistence_ina_sna | 3 | 1965.8373773999754 | 11965.837377399974 | 11965.837377399974 | 0.0 | 0.005312500000000002 | 0.0 | 0.0 | 23.8504126 | 1.4113499999999999 |
| short_var_long_fix_ideal_ideal_ina_sna | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.0 | 0.0 | 0.0 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i0p5_s42 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.004093027210701035 | 0.004093027210701035 | 0.003925034077281099 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i0p5_s43 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.004044533332723556 | 0.004044533332723556 | 0.003925034077281099 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i0p5_s44 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.003856860233866166 | 0.003856860233866166 | 0.003925034077281099 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i1p0_s42 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.008186054421402066 | 0.008186054421402066 | 0.007850068154562198 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i1p0_s43 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.008089066665447118 | 0.008089066665447118 | 0.007850068154562198 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i1p0_s44 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.007713720467732367 | 0.007713720467732367 | 0.007850068154562198 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i2p0_s42 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.016372108842804142 | 0.016372108842804142 | 0.015700136309124396 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i2p0_s43 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.016178133330894247 | 0.016178133330894247 | 0.015700136309124396 | 23.898949549999998 | 1.1487041666666666 |
| short_var_long_fix_ideal_noise_i2p0_s44 | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.015427440935464711 | 0.015427440935464711 | 0.015700136309124396 | 23.898949549999998 | 1.1487041666666666 |
| long_var_short_fix_persistence_persistence_ina_sna | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.009208333333333327 | nulo | nulo | 0.0 | 0.0 |
| long_var_short_fix_persistence_ideal_ina_sna | 3 | 1965.8373773999754 | 11965.837377399974 | 11965.837377399974 | 0.0 | 0.005312500000000002 | 0.0 | 0.0 | 23.8504126 | 1.4113499999999999 |
| long_var_short_fix_persistence_noise_i0p5_s42 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.009794247099664345 | 0.008963494199328686 | 0.010062497822047455 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_persistence_noise_i0p5_s43 | 1 | 1490.9485137666582 | 10877.654002911042 | 12352.594233711037 | -56328.0 | 0.011044638132485846 | 0.011464276264971688 | 0.010062497822047455 | 11.126617955599999 | 35.15225833333333 |
| long_var_short_fix_persistence_noise_i0p5_s44 | 1 | -510.53812236667136 | 9489.461877633337 | 9489.461877633337 | 0.0 | 0.01097513109585635 | 0.011325262191712695 | 0.010062497822047455 | 6.964860699999999 | 2.0832916666666668 |
| long_var_short_fix_persistence_noise_i1p0_s42 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.014275994199328705 | 0.017926988398657404 | 0.02012499564409491 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_persistence_noise_i1p0_s43 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.016776776264971666 | 0.02292855252994333 | 0.02012499564409491 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_persistence_noise_i1p0_s44 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.01663776219171272 | 0.022650524383425436 | 0.02012499564409491 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_persistence_noise_i2p0_s42 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.023239488398657405 | 0.0358539767973148 | 0.04024999128818982 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_persistence_noise_i2p0_s43 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.028241052529943345 | 0.04585710505988669 | 0.04024999128818982 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_persistence_noise_i2p0_s44 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.027963024383425433 | 0.045301048766850864 | 0.04024999128818982 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_ideal_persistence_ina_sna | 0 | 0.0 | 10000.0 | 10000.0 | 0.0 | 0.0038958333333333254 | 0.0 | 0.0 | 0.0 | 0.0 |
| long_var_short_fix_ideal_ideal_ina_sna | 3 | 2659.4364862832954 | 12659.436486283295 | 12659.436486283295 | 0.0 | 0.0 | 0.0 | 0.0 | 23.898949549999998 | 1.1487041666666666 |
| long_var_short_fix_ideal_noise_i0p5_s42 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.004481747099664343 | 0.004481747099664343 | 0.005031248911023727 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_ideal_noise_i0p5_s43 | 1 | 1490.9485137666582 | 10877.654002911042 | 12352.594233711037 | -56328.0 | 0.005732138132485844 | 0.005732138132485844 | 0.005031248911023727 | 11.126617955599999 | 35.15225833333333 |
| long_var_short_fix_ideal_noise_i0p5_s44 | 1 | -510.53812236667136 | 9489.461877633337 | 9489.461877633337 | 0.0 | 0.005662631095856348 | 0.005662631095856348 | 0.005031248911023727 | 6.964860699999999 | 2.0832916666666668 |
| long_var_short_fix_ideal_noise_i1p0_s42 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.008963494199328702 | 0.008963494199328702 | 0.010062497822047455 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_ideal_noise_i1p0_s43 | 1 | 990.5768547333245 | 10404.230714233727 | 11172.742677033722 | 54952.0 | 0.011464276264971665 | 0.011464276264971665 | 0.010062497822047455 | 10.9165562996 | 33.804516666666665 |
| long_var_short_fix_ideal_noise_i1p0_s44 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.011325262191712718 | 0.011325262191712718 | 0.010062497822047455 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_ideal_noise_i2p0_s42 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.0179269883986574 | 0.0179269883986574 | 0.02012499564409491 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_ideal_noise_i2p0_s43 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.022928552529943344 | 0.022928552529943344 | 0.02012499564409491 | 3.4999824989499997 | 31.666033333333335 |
| long_var_short_fix_ideal_noise_i2p0_s44 | 0 | 0.0 | 9464.836484317724 | 10164.07999916772 | 49999.0 | 0.022650524383425432 | 0.022650524383425432 | 0.02012499564409491 | 3.4999824989499997 | 31.666033333333335 |

## Lo que no se corrió

Dentro de las 44, nada. Fuera de ese protocolo: B0, las 21 razones, las 241 celdas retenidas, la ventana desde 2019-05-16 00:00 UTC, mercado real, GPU, base de datos y broker real. El calentamiento de dos barras no es una celda puntuada. Las sondas de rechazo por margen, de recorte y de `stop()` con posición pendiente corrieron en el mismo pytest y no están repetidas como celdas del manifiesto. El rechazo deja la posición en cero y no completa. El recorte iguala la posición al tamaño admitido, no al pedido. `stop()` con posición pendiente no fabrica fill.
