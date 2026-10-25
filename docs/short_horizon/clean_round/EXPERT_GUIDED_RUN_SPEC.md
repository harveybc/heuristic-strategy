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

## 5. Etapa A' (multi-resolución; pre-registrada 2026-10-24, aún no emitida)

**Revisión 2026-10-24 (noche):** A' corre sobre la **receta v2**, el mismo soporte que A, para que sus brazos se comparen
**directamente** con los siete de A sin puente. La v3 de Musashi (sin `px.er_1`) se usará después en un **par puente**
(control + ganador repetidos sobre v3) que mide por separado el efecto del soporte recuperado. El texto siguiente
queda como se escribió, con esa sustitución (v3 → v2 en A'; el puente pasa al par v3).

Entradas que faltan: la receta v3 de Musashi (v2 sin `px.er_1`, identidad y sonda nuevas) y `CHOSEN_THRESHOLDS.json`
del DEAP (`evidence/multires/deap/`). Cuando existan, `CLEAN_ROUND_SPEC.json` recibe la etapa `A_prime` apuntando a la
v3 y el lanzador la valida como a la A (misma receta, modelo, regularización, partición; sólo tratamientos).

| brazo | etiquetas | umbrales | pesos en la pérdida | papel |
| --- | --- | --- | --- | --- |
| `zz_hold_v3` | ZigZag heredado, hold_through | — | no | **puente**: mismo control que en A, sobre el soporte de la v3; la diferencia A→A' del control mide el efecto del soporte recuperado, no del tratamiento |
| `multires_default` | multires | por defecto (0.5 / 0.5 / 0.25σ / 0.1 / 0.9) | no | tratamiento base |
| `multires_deap` | multires | `CHOSEN_THRESHOLDS.json` | no | umbrales optimizados (objetivo hindsight declarado) |
| `multires_deap_weighted` | multires | `CHOSEN_THRESHOLDS.json` | sí (`--sample-weights`) | NOP graduado y negativos con su peso |

Decisión congelada: igual que en A (Sharpe medio INNER_A de tres semillas válidas; empates → menos parámetros, luego el
puente). Comparaciones A↔A' sólo a través del puente; nunca brazo de A contra brazo de A' directamente. 2024 una lectura
por brazo, nunca para decidir; TEST cerrado. La etapa B hereda al ganador de A o de A' según el puente lo justifique.

## 6. Resultado de A y A' y cadena hacia B (2026-10-25)

A y A' terminaron (`RESULTS_A_APRIME_2026_10_25.md`): `both_hold_shorter` 1.56 ± 1.31 ≥ `both_hold` 1.48 ± 1.16 > control
1.13 ± 0.97; los tres brazos `multires` (etiquetas de clase) negativos en INNER_A. Decisiones del líder técnico, sin
lanzar nada nuevo fuera de la palabra ya dada: (1) cinco semillas para los dos brazos `both` antes de que B herede
(regla en el spec v4, `seed_extension`); (2) B añade `winner_multires_aux`: la aptitud multi-resolución como objetivo
auxiliar de regresión sobre las etiquetas heredadas, en vez de como etiqueta de clase (que dejó al estudiante sin
ejemplos: NOP 86 %); (3) cadena en el host: 4 unidades de semillas → `decide` 5 semillas (`DECISION_A_5SEED.json`) →
`emit --stage B --winner-from` → cola B (3 brazos × 3 semillas), todo a 10G, una unidad a la vez.

## 7. Walk-forward dentro del desarrollo y ensamble de semillas (spec v5, pre-registrado 2026-10-25 antes de leer)

Orden del propietario (2026-10-25): "arreglar primero el instrumento de medida"; ejecutar y orquestar todo, automatizado y
secuencial, sin vigilancia. Qué corre, en una sola cadena en el host del 4090 (`tools/clean_round_chain/run_chain_wf.sh`):

1. Repetición de `B_winner_multires_aux` (falló por `MULTIRES_SPEC_NEEDS_PARAMS`: el miembro óptimo de la escalera sin los
   parámetros registrados; corregido en la llamada de E3-M). `B_winner_minimal_state` **no es ejecutable bajo la receta**
   (E3-M rechaza `agent_state != none` con `--recipe`): queda registrado, no se corre.
2. **Ensamble de semillas** (`tools/expert_guided_seed_ensemble.py`) para cada brazo terminado de A / A' / B: media de las
   probabilidades de las N semillas por módulo, un solo replay; árbitro = el más frecuente entre las semillas (empate →
   `prob`), tasa mediana, τ re-derivado en INNER_A. Cada semilla se vuelve a replicar (debe reproducir su E3M.json).
3. **Errores estándar** (`app/short_horizon/expert_guided/stats.py`): SE de Lo (2002) del Sharpe anualizado y bootstrap
   estacionario (bloque 5 días, 2 000 remuestreos, semilla fija) → informe `RESULTS_B_AND_ENSEMBLE_2026_10_25.md`.
4. **Walk-forward**: pliegues `wf2022` (ajuste 2020-01→2022-01, interno 2021 S2, validación **2022**) y `wf2023` (ajuste
   2020-01→2023-01, interno 2022 S2, validación **2023**), brazos `zz_hold`, `both_hold`, `both_hold_shorter`, 5 semillas,
   10G. Receta **derivada por pliegue** (`tools/expert_guided_wf_recipe.py`): sólo cambian las constantes del escalador
   (re-ajustadas en el bloque de ajuste del pliegue) y la partición; identidad propia, `derived_from` declarado; el
   lanzador rechaza cualquier otra divergencia (`WF_FOLD_*`). Testigo del cargador en `recipes_wf/WF_RECIPE_<fold>_WITNESS.json`.
5. Ensambles sobre los pliegues (INNER_B = año de validación, puntuado tras congelar; 2024 una lectura) → informe
   `RESULTS_WALK_FORWARD_2026_10_25.md`.

**Reglas pre-registradas (verbatim, `stats.HOLDS_RULE` / `SEPARABLE_RULE`):** un brazo SOSTIENE sólo si su Sharpe de
ensamble en el año de validación menos un SE de Lo es > 0 en AMBOS pliegues; dos brazos son SEPARABLES en una ventana sólo
si |Δ Sharpe| > SE agrupado √(SE_a² + SE_b²), si no NOT_SEPARABLE. Nada se elige en INNER_B ni en 2024; el walk-forward es
diagnóstico y no libera nada. Recogida en el coordinador: `tools/clean_round_chain/collect.sh` (sin commit automático).
