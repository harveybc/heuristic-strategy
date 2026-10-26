# Plan de selección copiado del de Musashi (agent-multi `tools/rl_selection_run.py` @e18d00d4, feature_selector @2fea401), 2026-10-26

Orden del propietario: reutilizar los datos existentes del lago y copiar el plan de Musashi; la única diferencia es el
evaluador (mis especialistas en vez de su PPO). Columna izquierda: lo que hace Musashi (leído en su código). Derecha: mi
copia.

| paso | Musashi (RL, PPO nativo) | mi línea (especialistas C3) |
| --- | --- | --- |
| datos | store consumidor existente `cd0eb8d2…` (121 columnas de la receta v2, 31 ramas; TRAIN 21 221 / INNER_A 2 805 orígenes válidos), barras admitidas, calendario de cierres, manifiesto fijado | **los mismos archivos**, sin reconstruir nada |
| alcance | `development`; TRAIN 2020-01-01→2023-07-01 para ajustar, INNER_A 2023-07-01→2024-01-01 como bloque de selección; TEST cerrado; declarado "el pool de candidatos usó 2023; INNER_A es desarrollo, no confirmación independiente" | idéntico, misma declaración |
| etapas previas | artefactos existentes de la metodología causal (calidad, perfiles, causal, relaciones, candidatos, codificación, extractores por grupo), autenticados por el puente de artefactos, no recalculados | **los mismos artefactos de la selección v5**, autenticados igual |
| restricciones | `agent_state none`, escalado `fixed_train`, sin recorte, sin realimentación, sin z-score móvil | idénticas |
| modelo evaluador | PPO con extractor TCN por grupos (una conv causal por grupo, fusión 1×1 a 8, dos bloques residuales d1,d2), cabezas categóricas separadas; costos `configured` | mis cinco especialistas (contrato C3) sobre **el mismo tronco TCN** (paridad exacta probada); costos `configured` |
| semillas | 7, 19, 43 | 7, 19, 43 |
| controles previos (entrada completa, 3 semillas cada uno) | (tcn, configured), (tcn, zero_cost = sólo diagnóstico), (cnn superficial, configured); cada control re-ejecutado bajo ambos modos de costo; sólo tcn/configured alimenta la búsqueda | (tcn, configured), (tcn, zero_cost diagnóstico), (conv1d-u, configured); misma regla |
| búsqueda | `feature_selector.search.FloatingBackward`: eliminación hacia atrás con mejor subconjunto por cardinalidad, reinclusión condicional sólo si mejora esa cardinalidad, estados visitados, empates deterministas, presupuesto duro 128 subconjuntos distintos × 3 semillas; ajuste **fresco** por subconjunto; grupo vacío → rama eliminada y fusión recalculada; bloques seno/coseno coherentes; escalador reajustado sólo en TRAIN; hash de soporte pareado; 2 intentos máximo; resultados inmutables y autenticados por bytes; bloqueo exclusivo; una búsqueda interrumpida se reporta BUDGET, nunca "completa" | **el mismo selector, los mismos parámetros**; única diferencia: el evaluador registrado `expert_guided_model` |
| puntuación | score de desarrollo por semilla (nativo del simulador), con actividad y brecha como restricciones; un Sharpe indefinido o actividad insuficiente no puede ganar | expectativa neta por trade en σ con IC (bootstrap por días) en INNER_A + Sharpe, actividad ≥ 2/sem, cierre temprano > 0, brecha TRAIN–INNER; mismo veto |
| reporte por subconjunto y semilla | parámetros, órdenes / dirección / aciertos, cierres tempranos, costos, profit, Sharpe, drawdown, brecha TRAIN–validación | idéntico |
| después | confirmación aparte; ninguna liberación global; el núcleo fijo dentro de la búsqueda (sin ajustar arquitectura a la vez) | idéntico; la confirmación usa el protocolo anidado (F2) sólo entonces |

Lo que queda **fuera** por esta orden: la lista de 19 (retirada), los stores por fold para la selección (parados; se
reutiliza el store existente), cualquier experimento con lista reducida antes de que el SBFS la produzca.

Secuencia: F1 (C3) → controles de entrada completa (3 familias × 3 semillas) → SBFS 128 × 3 → informe → confirmación.
