# Ronda limpia guiada por el experto: especificación finita de la corrida (A1, 2026-10-24)

Especificación ejecutable: `CLEAN_ROUND_SPEC.json` (identidad `clean-expert-guided-a1-20261024`), consumida por
`tools/expert_guided_clean_round.py` (`check` / `emit` / `decide`). Esta página la explica; manda el JSON.

## 1. Lo que está fijo y de dónde viene

| fijo | valor | fuente |
| --- | --- | --- |
| representación compartida | receta `recipe_5a8f2155318ba7c7` (sha `93ac66dc…`): 121 columnas de la lista estricta v5 en su orden, 24 transformadas aguas arriba y 0 operaciones del consumidor, constantes TRAIN fijas (ajuste anterior a 2023-07-01), ventana 13 slots / stride 2 en filas observadas con la barra actual incluida, 25 grupos conceptuales → 31 extractores (3 puntuales, 28 conv causal), modelo 2/2/16 | `CONSUMER_RECIPE.json`; aceptada por Musashi (`MUSASHI_SHARED_RECIPE_REREVIEW_AND_NEXT_LANES_2026_10_24.md`) |
| store | `full_pool_s_a1` (`cd0eb8d2…`): productores aprendidos ajustados sólo en el alcance del consumidor | corrección A1 |
| regularización | fuerte: dropout 0.3, wd 1e-2 | T2 v5, regla congelada (medida para E3-M, no establecida como óptima para SAC) |
| contrato de negocio | 1 entrada/día, apalancamiento 1:50, SL y TP en toda orden, llenado en la apertura siguiente, cierre forzoso 12 h antes del cierre semanal y retardo de 12 h tras la reapertura, costos del simulador; actividad ≥ 2 órdenes/semana como promedio del periodo (validez, no obligación) | receta → `business` (misma función que usa E3-M) |
| partición | consolidated: ajuste 2020-01 → 2023-07, bloque interno 2023 S2, 2024 una lectura tras congelar, TEST cerrado | spec |
| observación | sólo mercado: sin realimentación del agente, sin etiquetas del experto, sin z-score móvil ni recortes | `--agent-state none`, receta |

El lanzador rechaza cualquier desvío: digest o identidad de la receta, store, modelo, regularización o partición
distintos a la receta; flags que la contradigan (`--feature-list`, `--groups`, `--window`, `--stride`, `--eligible-from`,
rolling); brazos que intenten anular campos de la receta. La prueba de no reajuste es la sonda: el cargador por
receta reproduce la sonda de referencia exactamente (`check --probe-reference`, tensor `f087d1a7…`).

## 2. Lo que varía: sólo tratamientos del lado del entrenamiento

Las demostraciones del ZigZag y las etiquetas `opportunity` / `both` son **tratamientos de entrenamiento**: nunca
entran en la observación ni en la elegibilidad en tiempo real, y no son realimentación humana. Diseño en
`EXPERT_SIGNAL_PER_AGENT.md` rev 2.1 (horizonte favorable compartido, disponibilidad por confirmación del pivote,
filas censuradas descartadas, tabla de verdad fija de `both`).

| etapa | brazo | etiquetas | cerrador | demostradores extra | estado del agente |
| --- | --- | --- | --- | --- | --- |
| A | `zz_hold` (control) | ZigZag heredado | hold_through | — | none |
| A | `zz_hold_shorter` | ZigZag | hold_through | depth 2 + reversión 10 pips | none |
| A | `opp_hold` | opportunity | hold_through | — | none |
| A | `opp_optimum` | opportunity | optimum (contraste declarado) | — | none |
| A | `both_hold` | both (OR, tabla fija) | hold_through | — | none |
| A | `both_hold_shorter` | both | hold_through | depth 2 + rev10 | none |
| A | `opp_hold_shorter` | opportunity | hold_through | depth 2 + rev10 | none |
| B | `winner_aux` | las del ganador | las del ganador | las del ganador | none; + cabezas auxiliares de oportunidad / ganancia adicional (sólo entrenamiento) |
| B | `winner_minimal_state` | las del ganador | las del ganador | las del ganador | minimal (posición + puede-entrar; control declarado) |

Tres semillas (20261019/20/21), 3 rondas DAgger, árbitro elegido en el bloque interno, 1 000 épocas máximo con paciencia
10: todo como en T2/T3 v5. Parámetros por módulo: 5 460 (5 528 con cabezas auxiliares).

## 3. Decisión (congelada antes de leer)

Etapa A: mayor Sharpe medio de las tres semillas en INNER_A entre brazos con las tres semillas válidas; empates →
menos parámetros, luego el control `zz_hold`. Para cada brazo se reportan TRAIN y 2024 (una lectura, nunca para
decidir), brecha TRAIN−INNER_A, dispersión entre semillas, DD, órdenes, aciertos, costos y actividad, incluidos los
resultados negativos. La etapa B hereda el ganador. Ninguna cifra de este experimento es una medida de rentabilidad.

## 4. Puerta de lanzamiento y lectura de vuelta

- El paquete fijado está en el host del 4090 (`expert-guided-clean-a1-20261024`: código, receta y su digest, store,
  barras admitidas, sel15, demostradores, spec, sonda de referencia; `PINS.sha256` verificado allí). Nada corre solo.
- La cola se emite **sólo** cuando exista `LAUNCH_EXPERT_GUIDED.json` en esa raíz, que escribo únicamente con la
  palabra explícita del propietario. Cada unidad vuelve a comprobar el digest de la receta y la puerta antes de
  correr; marcas de estado reanudables; sin bajar un tope de memoria rechazado.
- Lectura de vuelta: `E3M.json` de cada brazo (con `recipe.identity` y `recipe.sha256` registrados), `SCALER.json`,
  sonda, 12 checkpoints por brazo, recibo de ejecución en el formato compartido (`tools/receipted_run.py`), decisión
  `decide` sobre los recibos verificados; publicación al almacén con lectura de vuelta como punto aparte.
