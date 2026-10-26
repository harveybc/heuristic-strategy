# C3: arreglos de etiquetas y objetivos del registro de auditoría (carril F1, 2026-10-26)

Contratos sellados: `contracts/ACTION_CONTRACT_C3.json` (tronco `u`) y `contracts/ACTION_CONTRACT_C3_TCN.json`
(tronco `residual_tcn`); la arquitectura, la variante de ejecución (`full_f3`, `exec_709f9b1fa3512486`) y la puerta
"mejor del día" viajan DENTRO de la identidad. Código: `app/short_horizon/expert_guided/c3.py` (sobre `c2.py` / `outcomes.py`),
`tools/expert_guided_e3m.py` (`--action-contract <C3>`, `--trunk`, `--exec-variant`), `learn.py` (balance de pérdida),
`tools/expert_guided_clean_round.py` (rechazo de unidades anidadas con store consolidado). Pruebas: `tests/test_expert_guided_c3.py`.

| defecto | arreglo (archivo:función) | prueba |
| --- | --- | --- |
| A2-2 / A2-1 / A4-3 etiquetas que enseñan a perder, SL constante | abridores entrenados en la TABLA DE RESULTADOS del motor: cabeza de 3 clases {TP_FIRST, SL_FIRST, TIME} en la celda de referencia (1σ/1σ) + neto previsto por celda; **sin cabeza SL** (el SL es la constante declarada de la celda); `c3.check_target_std` rechaza (`C3_CONSTANT_TARGET`) todo objetivo de regresión con std < 0.05 en TRAIN | `test_constant_regression_target_is_refused…`, `test_each_mutation_is_refused[C3_SL_HEAD_FORBIDDEN]` |
| A2-3 TIME contado como "no TP" | TIME es clase propia; calibración por temperatura (preserva el argmax) | `test_temperature_scaling_recovers…`, `[C3_CONTRACT_OPENER_CLASSES]` |
| A2-9 / A4-10 rejilla fuera del tope efectivo | rejilla 0.5–2.0σ (máximo ≤ p1 del tope, 2.09σ) extendida bajo 0.75; celdas fuera de la banda por barra [mínimo, tope] no seleccionables (`exec_variants.effective_cap / min_multiple`) | `test_grid_is_under_the_effective_cap…`, `test_cells_outside_the_per_bar_effective_band…` |
| A4-5 / A2-8 probabilidad equivocada en la ganancia esperada; actividad forzada | regla EV = P(TP)·TP − P(SL)·SL + P(TIME)·E[neto\|TIME] − costo > margen por lado, con la celda PERMITIDA de mayor neto previsto (TP ≥ SL, o neto > costo + 0.1σ); "no operar" es válido; la actividad es sólo criterio de validez | `test_ev_rule_known_answer…`, `test_reward_risk_floor…`, `test_policy_opens_the_side…` |
| A2-4 / A2-6 / A4-6 cerradores ciegos, exploración con TP/SL fijos, claves contradictorias, fuga del experto en C2G, un solo umbral | cerradores con la geometría causal de su orden (C2G) entrenados **sin el experto**: ronda 0 = exploración con TP/SL de la rejilla (`prior_random_cells`); duplicados por (fila, lado, geometría) fusionados y los empates eliminados (`dedup_closer`); umbrales de cierre POR LADO mapeados en las visitas de su lado (`gene_map`); tras congelar, el cerrador debe pasar `exec_variants.early_close_ok` (≥ 5 % de salidas propias y cota inferior ≥ 0 del PnL diario pareado con / sin cierre) o se desactiva y se marca `early_close_disabled` | `test_round0_exploration_never_calls_the_expert…`, `test_contradictory_closer_keys…`, `test_closers_use_their_own_side_threshold…`, `test_early_close_benefit_lower_bound…`, `test_early_close_rule_uses_share_and_lower_bound` |
| A2-5 DAgger contradice la ronda 0; tope diario en las etiquetas | las rondas reetiquetan a los cerradores con la MISMA regla del motor; los abridores se etiquetan en toda fila plana elegible con cuenta limpia (`outcomes.entry_rows`): el tope diario vive sólo en el entorno | `[C3_CONTRACT_LABEL_RULES]` |
| A2-10 / A2-11 filas con puerta cerrada; soporte del TP | filas con puerta cerrada fuera de los abridores (cuenta limpia); sin objetivo ZigZag (no aplica) | — |
| A2-12 balance de pérdida | `learn.fit(reg_norm="masked", reg_weight)`: Huber normalizado por filas enmascaradas, pesos declarados (`LOSS_BALANCE`) | `test_loss_balance_kwargs_default…` |
| A4-2 primera señal del día | `exec_variants.BestOfDayGate` sobre la política PUNTUADA por EV (observa 3 decisiones, declarado) | `test_policy_call_returns_a_best_of_day_gate…` |
| A4-1 / A4-4 / A4-9 / A4-12 | variante de ejecución `full_f3` por defecto (reversión = una decisión, tamaño por riesgo 1 %, corte 24 h antes del cierre, disparo de cortos en el ask); `legacy` como control declarado (`--exec-variant legacy`) | `test_full_f3_variant_identity…` |
| A5 (protocolo v11) | parada temprana en `sc['early_stop']`, calibración y mapas de genes en `sc['calibration']`, búsqueda / fitness / prueba de cierre en `sc['selection']`; el bloque de decisión se lee una vez tras congelar y se anota en `READ_LEDGER.jsonl`; por búsqueda final: Sharpe deflactado y PBO (`search_overfit`); el lanzador rechaza `NESTED_FOLD_STORE_NOT_FOLD_LOCAL` | `test_blocks_fold_to_inner_a…`, `test_launcher_refuses_a_nested_fold_unit…`, `test_search_overfit_fields…` |
| arquitectura en la identidad | `architecture` {trunco, fusion_units, core_blocks, readout} sellado; `C3_ARCHITECTURE_MISMATCH` si la corrida difiere | `test_two_sealed_variants…`, `test_architecture_mismatch…` |

## Identidades selladas

| contrato | identidad | tronco | variante de ejecución |
| --- | --- | --- | --- |
| `ACTION_CONTRACT_C3.json` (**por defecto**) | `action_4752a9d261379258` | `residual_tcn` (TCN: conv causal por grupo, SiLU, fusión 1×1 de 8, dos bloques residuales d1,d2, flatten) | `full_f3` `exec_709f9b1fa3512486` |
| `ACTION_CONTRACT_C3_U.json` (control declarado) | `action_88ecc1867959d02c` | `u` (conv1d por grupo, el diseño u) | `full_f3` `exec_709f9b1fa3512486` |

Los dos difieren SÓLO en `architecture` (y por tanto en la identidad). Decisiones de codificación A1-11 declaradas en ambos (`encoding_decisions`): la receta de entrada debe traer `encoding_decisions_applied` (receta derivada por `tools/expert_guided_c3_recipe.py derive`: GMM one-hot, ventanas de 2 h de los relojes de eventos, `hours_to` eliminadas) o la corrida se rechaza.

## Salidas por cabeza (bloque `head_outputs`, verificado por el validador; para la reconciliación con F9)

| cabeza | codificación | rango exacto | calibración | cómo la consume la estrategia |
| --- | --- | --- | --- | --- |
| `open_l.decision` | softmax_ce | 3 probabilities in [0, 1], sum 1 | temperature scaling on INNER_A (argmax-preserving) | EV = P(TP)*TP_cell - P(SL)*SL_cell + P(TIME)*E[net|TIME] - cost (probabilities at the reference cell); OPEN_LONG iff EV > open_margin_l (quantile gene) and the gate places it |
| `open_s.decision` | softmax_ce | 3 probabilities in [0, 1], sum 1 | temperature scaling on INNER_A (argmax-preserving) | as open_l for OPEN_SHORT with open_margin_s |
| `open_l.continuous` | bounded_sigmoid ×25 | each in [-4, 4] sigma (predicted engine net per unit / sigma) | none (Huber-fitted regression; order-preserving) | cell choice only: argmax over the ALLOWED cells (TP >= SL, or net > cost + rr_margin; inside the per-bar band); never thresholded into an action by itself |
| `open_s.continuous` | bounded_sigmoid ×25 | each in [-4, 4] sigma | none | as open_l |
| `nop.decision` | softmax_ce | P(NOP) in [0, 1] | Platt on INNER_A | veto: no open when P(NOP) > nop_veto (quantile gene); kept only if it improves the selection-block lower bound |
| `close_l.decision` | softmax_ce | P(SL first) in [0, 1] | Platt on INNER_A | CLOSE when P(SL first) > close_p_l (per-side quantile gene on long positioned visits) |
| `close_s.decision` | softmax_ce | P(SL first) in [0, 1] | Platt on INNER_A | CLOSE when P(SL first) > close_p_s |
| `close_l.continuous` | bounded_sigmoid ×1 | [-4, 4] sigma (predicted remaining value of HOLD) | none | CLOSE when remaining < -close_margin_l (per-side quantile gene) |
| `close_s.continuous` | bounded_sigmoid ×1 | [-4, 4] sigma | none | CLOSE when remaining < -close_margin_s |

- puerta: the EV of the chosen open is the SCORE of exec_variants.BestOfDayGate (observe 1 signal per business day, then the first EV >= the day's running best)
- interruptor de cierre temprano: exec_variants.early_close_ok(agent close share >= 0.05, value lower bound >= 0) on the selection block; on failure the closers are not consulted (flagged)
- entradas de los cerradores: recipe window + the order's own geometry (dist to SL / TP in sigma_entry, hours held / 24, side); openers / NOP: recipe window only

## Controles (DIAGNOSTIC, wf2022, 1 semilla, CPU, `full_f3`)

Se llena con `PIPELINE_CONTROLS_C3` al terminar la cola (`queue_c3.sh`): trampa y sintético × {u, residual_tcn}, contra C2 v1.3 y el oráculo.
