# Petición de auditoría a Musashi: la estrategia simulada con salidas ideales y simuladas (2026-10-26)

Rama `satoshi/expert-guided-rl-20261019` (commits en §6). Documento auditado:
`STRATEGY_IDEAL_OUTPUT_SIMULATION_2026_10_26.md` (mismo directorio). Nada de lo que sigue toca el bloque TEST
(filas ≥ 2025-01-01): las verdades, las simulaciones y las puntuaciones leen sólo los bloques de calibración,
selección y decisión de los folds `wf2022` y `wf2023`.

## 1. Qué se pide auditar (afirmaciones, cada una con lo que la falsaría)

| # | afirmación | dónde está la evidencia | qué la falsaría |
| --- | --- | --- | --- |
| A1 | las tablas de resultados son **exactas por celda** (25 celdas × 2 lados; sin impacto de mercado): la verdad reproduce, celda a celda, lo que el motor real produce al colocar esa orden en esa fila | `tools/strategy_output_simulation.py::build_truth` (usa `outcomes.entry_table` y `hold_value` de la lane F1), chequeo `truth_matches_engine` en cada `RUN_*.json` (`correctness`) | una fila cuyo neto por celda en la verdad difiera del neto del motor al replicar la orden |
| A2 | las cabezas simuladas tienen la calidad que se declara: la AUC / Spearman / MAE **lograda** por bloque (`simulation.achieved`) coincide con el nivel pedido dentro de la tolerancia de muestreo | `RUN_*.json` → `per_seed[].simulation.achieved`, tabla de rangos de salida en §3 del documento | una corrida con AUC lograda a más de 0.03 del nivel en el bloque de calibración |
| A3 | la calibración Platt se ajusta **sólo** en el bloque de calibración; selección y decisión nunca entran en la calibración ni en la elección de genes | `fit_closer_cal`, `simulate_table` (parámetro `cal_rows`), `run_config` (freeze sobre `SEL_BLOCK`, puntuación sobre `DEC_BLOCK`) | cualquier lectura de `DEC_BLOCK` antes de la puntuación final |
| A4 | el oráculo (nivel 1.00) gana y el azar (0.50) no, en **todos** los escenarios; la calidad de equilibrio por escenario está en `BREAK_EVEN.json` y es la raíz de la interpolación lineal de la expectativa pooled | `evidence/strategy_sim/BREAK_EVEN.json`, `SWEEP_POOLED.csv`, figuras `*_vs_level_*.png` | un escenario donde el oráculo tenga expectativa ≤ 0 o el azar > 0 con IC que excluya 0 |
| A5 | la puerta del contrato (observar 1 señal) **no cumple** la actividad ≥ 2/semana fuera de bloque ni con el oráculo; la puerta `perbar` sí | `RUN_*_contract_*.json` frente a `RUN_*_perbar_*.json` (`decision.metrics.orders_per_week`, `valid`) | una corrida `contract` con IC90 inferior de actividad ≥ 2/semana en decisión |
| A6 | la fitness compuesta del propietario (F2) elige puntos con expectativa fuera de bloque no peor que F0 en la mayoría de celdas fold × semilla, y sus coeficientes son 1/mediana\|término\| sobre el frente válido | `FITNESS_ROWS.csv`, `FITNESS_SUMMARY.json`, §7 "fitness" del documento | una mayoría de celdas donde F2 pierda frente a F0 en expectativa de decisión |
| A7 | las fórmulas TP/SL y volumen por dirección (§8) son las implementadas y **todos** los coeficientes son genes optimizados por NSGA-II; los archivos listan cada punto evaluado | `app/short_horizon/expert_guided/strategy_full.py` (`levels`, `volume`, `gene_bounds`), `RUN_*_<familia>_*.json` → `genes_bounds`, `formulas`; archivos `ledger/ARCHIVE_*.json.gz` | un coeficiente fijado a mano fuera de los límites declarados, o un punto elegido ausente del archivo |
| A8 | la batería de intervenciones corre sobre el bloque de decisión con los genes congelados en selección y los veredictos TOLERANTE provienen de IC bootstrap que excluyen 1 | `interventions/INT_*.json`, tabla de intervenciones en §7 | un veredicto TOLERANTE cuyo IC incluya 1, o una re-optimización dentro de la batería |
| A9 | los testigos de cada hoja sintética coinciden con su etiqueta (signo de la deriva, razón de sd, densidad fina monótona) | `hier/STAGE_A_<familia>_leaves.json`, `tests/test_strategy_hierarchy.py` | una hoja con signo de deriva contrario o densidad fina no ordenada |
| A10 | la región robusta de la Etapa A cumple la regla del 5 % global y por régimen L1, y la Etapa B se optimizó **dentro** de la caja | `hier/STAGE_A_<familia>.json` (`robust_region`), `RUN_stageB_*.json` → `genes_bounds` ⊆ caja | un gen de la Etapa B fuera de su caja |
| A11 | `STRATEGY_PARAMS_FROZEN_v1.json` se deriva de las corridas de la Etapa B por la regla de banda declarada y su sha256 coincide con el cuerpo | `contracts/STRATEGY_PARAMS_FROZEN_v1.json` (`sha256_body`), §11 del documento | una mediana que no reproduzca `freeze_params` sobre los `RUN_stageB_*.json`, o un sha distinto |
| A12 | ningún entregable contiene trabajo diferido (las palabras que la orden prohíbe) ni nombres de host o rutas personales | `grep` de §5 | una coincidencia |

## 2. Cómo re-ejecutar (CPU, sin GPU)

```bash
export CUDA_VISIBLE_DEVICES=""
ADM=<admitted>; CSV=<sel15>; ROOT=<directorio de trabajo>
# verdades (una por fold y variante de motor: base, cost2, cost3, spike, cutoff12)
python tools/strategy_output_simulation.py truth --fold wf2022 --admitted $ADM --csv-dir $CSV --variant base --out $ROOT/truth
# una corrida (familia C3, puerta perbar, escalera completa, 3 semillas)
python tools/strategy_output_simulation.py run --fold wf2022 --admitted $ADM --csv-dir $CSV --truth-dir $ROOT/truth \
   --variant base --gate perbar --levels 0.5,0.55,0.6,0.62,0.65,0.7,0.75,0.8,0.9,1.0 --seeds 0,1,2 --out $ROOT/runs
# capa paramétrica completa (familias cell_param, param_risk, param_param)
python tools/strategy_output_simulation.py run ... --family param_param --out $ROOT/runs
# batería de intervenciones sobre una corrida congelada
python tools/strategy_intervention.py --run $ROOT/runs/RUN_wf2022_base_perbar_q0.70.json --truth-dir $ROOT/truth \
   --admitted $ADM --csv-dir $CSV --draws 6 --out $ROOT/interventions
# catálogo + Etapa A + Etapa B
tools/clean_round_chain/run_strategy_two_stage.sh $ROOT <checkout>
# reporte (tablas, figuras, contrato congelado con sha)
python tools/strategy_output_simulation.py report --runs $ROOT/runs --interventions $ROOT/interventions --hier $ROOT/hier \
   --contracts docs/short_horizon/clean_round/contracts --docs docs/short_horizon/clean_round/evidence/strategy_sim \
   --out docs/short_horizon/clean_round/evidence/strategy_sim
python -m pytest tests/test_strategy_output_simulation.py tests/test_strategy_full.py tests/test_strategy_intervention.py \
   tests/test_strategy_hierarchy.py -q
```

Los drivers de los barridos (`tools/clean_round_chain/run_strategy_sim_sweeps.sh`, `run_strategy_sim_full.sh`,
`run_strategy_sim_interventions.sh`, `run_strategy_two_stage.sh`) listan exactamente qué corrió y con qué semillas.

<!-- ARTIFACTS -->
