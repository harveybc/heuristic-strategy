# Plan de trabajo de Satoshi: estrategias basadas en ML (forecasting, clasificación, causal, no supervisado)

Estado: VIVO. Se actualiza en cada retorno de carril o cada decisión del propietario.
Separado del plan de Musashi (RL directo en gym-fx/agent-multi). Fecha de esta revisión: 2026-10-19.
Roles: Propietario (decide negocio), Satoshi (lidera este plan), Musashi (RL y reparación de datos), Retsu (revisión independiente).

## 0. Reglas permanentes (no se negocian)

| Regla | Detalle |
| --- | --- |
| Instrumento y datos | EURUSD H1. TRAIN 2020–2023, INNER/decisión 2023, VALIDATION 2024 = confirmación única. TEST 2025 CERRADO. Reloj de barras en reparación (Musashi). |
| Toda orden | Lleva `units` enteras, lado, entrada, **SL y TP calculados y efectivos** desde la entrada (nunca simbólicos). El cierre anticipado del agente es adicional, nunca sustituto. |
| Tamaño | `units = floor(percentual_order_size · rel_volume · balance_actual · apalancamiento / precio)`; `rel_volume` = 0,10 (parámetro optimizable después, nunca en 2024); `percentual_order_size` ∈ (0,1] lo decide la estrategia por riesgo; mínimo OANDA 1 unidad; margen libre verificado; apalancamiento declarado 1:10/1:50/1:100. |
| Frecuencia | Válido solo con ≥ 3 órdenes/semana (variante 4) y ≤ 4/día (variante 2), impuesto dentro de la estrategia. Un punto que no opera no es resultado. |
| Gráficas | Sharpe y ganancia contra el MAE (y MAE/naive) de **cada** predicción consumida, un eje por error; estilo de la propuesta doctoral; puntos inválidos huecos; siempre link de GitHub y local. |
| Rigor | Nada por debajo de un presupuesto real se presenta como resultado: declarar modelo, entradas reales/sustitutas, ventana, épocas, semillas, rangos, generaciones y choques con límites. Paridad en gym-fx para estrategias. |
| Regularización | Además de la parada temprana (que solo cubre el exceso de entrenamiento): por defecto AdamW con weight decay desacoplado (L2) 1e-4 + dropout bajo 0,1 tras la fusión/núcleo y antes de la cabeza; SIN BatchNorm (desarmonía dropout+BN por desplazamiento de varianza, Li et al. 2019; estadísticas frágiles ante cambio de régimen). Ablación declarada {dropout 0/0,1/0,2} × {wd 1e-4/1e-3}, decidida en bloques internos de 2023. Árboles: capacidad acotada (profundidad ≤4, hojas ≥200, L2). Decisión del propietario delegada a Satoshi, 2026-10-19. |
| Parada temprana | Promedio de la métrica de entrenamiento de la época (la que produce el entrenamiento) y la de validación interna (2023), paciencia 10, 1e-8, mejores pesos. |
| Aditividad | Nunca dañar lo existente hecho para predicción: todo cambio en selección, features o estrategias entra como plugin nuevo + config JSON (arquitectura expandible de cada repo), registro/append-only, con prueba de que lo viejo queda byte-idéntico. Orden del propietario 2026-10-19. |
| Causalidad | `available_time ≤ t`; prohibidos filtros centrados (STL, wavelets, loess) como entrada o target. Toda selección decide en 2023 y confirma una vez en 2024. |
| Cierre | Cada cierre lleva tabla generada desde artefactos (error+escala, naive mismas filas, skill, literatura, comparabilidad); `NO_NEW_MEASUREMENT` si no hubo medición. |
| GPU | 5090 = Musashi. Satoshi: 4090 si libre → 4070 de omega si cabe → 5070 Ti de gamma si hay memoria. Admisión `crispdm-run`, tope 1,25× pico medido, un trabajo pesado por host. |
| Operación | Sin monitoreo periódico; un retorno por orden; nada de secretos ni nombres de máquina en Git. |

## 1. Objetivo

Encontrar una estrategia operable en EURUSD H1 cuyo cruce por cero quede demostrado con el método oráculo → error realista → modelo real, bajo las reglas de la sección 0, y entregar al carril de RL las señales/demostraciones y los límites medidos.

## 2. Conocimiento establecido (con evidencia)

| # | Veredicto | Evidencia |
| --- | --- | --- |
| K1 | Predecir el nivel de precio horario no supera a la persistencia (martingala en la práctica); dirección ≈ moneda en cinco familias de modelos. | `TREND_PLUS_VOL_TP_SL.md`, `ROUND_CLOSURE_2026_10_19.md`, predictor I6-DMO MG2/MG12 cerrados. |
| K2 | La volatilidad sí se predice (HAR 15–17 % mejor que el naive en rango diario; GBM 23 % en 6 h), pero por lado solo 2–14 %, y el canal de volatilidad solo no cruza cero ni con oráculo. | `CALENDAR_RANGE_EXPERIMENT.md`, `TREND_PLUS_VOL_TP_SL.md`. |
| K3 | Estrategias de rango con precio: necesitan error por lado ≤ 0,67× naive (24 h) / 0,53–0,6× (6 h); alcanzable 0,86–0,98×. V1 ruptura refutada; V2 reversión marginal. | `VOLATILITY_STRATEGIES.md`, `PRICE_PATH_NOISE_SWEEP.md`. |
| K4 | Eventos macro: impulso real de volatilidad (NFP 3,3×, Fed 4,4×, t½ 0,3–0,8 h), tendencia solo vía sorpresa y demasiado pequeña a resolución horaria; conjunto seleccionado con FDR vacío. Calendario = filtro de riesgo. | causal-inference `satoshi/event-impact-20261019` @a4cb9ea; `CAUSAL_EVENT_STRATEGY.md`. |
| K5-nota | Corrección (Musashi F02/F03): la "confirmación 2024" del experto ZigZag es desarrollo ya observado, no holdout; su paridad gym-fx fue hold-to-reversal, no paridad de negocio. Mi afirmación de que gym-fx carece de SL/TP debe verificarse contra la revisión fijada del adaptador. |
| K5 | Experto ZigZag (Depth 3 / Dev 36,3 / Backstep 2, NSGA-II preliminar): Sharpe 17–19; versión confirmada causalmente −1 … −2,2; brecha ≈ 21 (mitad sincronización, mitad nivel del siguiente pivote). Paridad gym-fx del experto: neto +2,7 %, drawdown igual. | `ZIGZAG_EXPERT_STUDY.md`, `results/zigzag/`. |
| K11 | A2 CERRADO 2026-10-19: detector de pivotes sobre las 15 variables reales de Musashi (contrato reproducido: 13 obs stride 2, z-score causal 256; 4090). AUC 0,66 (mejora desde 0,52 del manifiesto viejo) pero calibración peor que la base, precisión 0,34–0,37 < base 0,38–0,39; Sharpe m1 −2,1/−3,5, m2 −1,5/−1,2, m3 +0,07/−2,0 (memoriza en TRAIN). Equilibrio requiere precisión 0,6–0,7. E0 de imitabilidad ≈ 0 también con sus 15. | `satoshi/pivot-detector-20261019` @052dda5, `PIVOT_DETECTOR_EXPERIMENT.md` §9 |
| K12 | A3 v2 SELLADO PROVISIONAL 2026-10-19: TIMING_PROXY_SELECTION sha 4fd6710e… (pendiente de decisión independiente de Musashi), 7 de 89: RAW_BAR_LOG_RETURN, cal.hod_sin, cal.session_europe, px.atr_ratio_14_120, px.close_loc_2, px.logret_24, variogram_hurst_w96. E0 en filas idénticas vs las 15: PR-AUC 0,31–0,37 vs base 0,19 en ambos; log-loss igual o mejor con 7 (único significativo: m1 2024 a favor de las 7); TODOS pierden en la estrategia trivial (Sharpe −0,1…−2,7 < no operar), DSR 0. Utilidad de clasificación para el proxy: sí; utilidad de estrategia/RL: no demostrada. Hallazgos: adaptador llena en b+2 (latencia contra k=2), regla 12 h no-reopen ausente en `signal_blocks`, lago 2023 con huecos (INNER_A). A05/A12 parciales (join raíz reparada; copia física independiente). | predictor `satoshi/timing-selection-20261019` @917a52f6; causal-inference @a379a67; heuristic-strategy `satoshi/timing-selection-eval-20261019` @fe3b80a; feature_selector @4f92033 |
| K13 | A1 CERRADO 2026-10-19 (@abbfaa6): NSGA-II v2 completo (pop 64 × 80 gen × 3 semillas, 21 cadenas convergidas por hipervolumen, cotas ampliadas una vez). Experto retrospectivo: óptimo fijado por el tope diario (tope 1: D6/19,6/B3 Sharpe 13,7–14,5; tope 2: D5/11,7/B3 15,8–17,2; tope 4: D3/40,7/B0 17,7/18,2/18,2). Deviation casi irrelevante en la semántica MT; Backstep 0 = piso estructural. Versión confirmada (b) de esos expertos: Sharpe −3,7…−1,4. Mejor ZigZag confirmado (B): D20/214,6/B5, Sharpe +0,43/+0,52 en 2020–22/2023 pero −1,01 en 2024 (DD 49 %, retardo 20 barras): sin borde causal. Fill b+1 ejecutable (aditivo) vs b+2: ≤0,1 Sharpe. Confirmación v1 sobreestimaba el retardo 1 barra. Demostraciones v2 sha 38125b0b… (v1 352ddf6a conservado). Invalida las etapas de A3 dependientes del target (F08) → rerun v3 lanzado. | heuristic-strategy `satoshi/short-horizon-strategy-20261018` @abbfaa6 |
| K6 | Detector de pivotes sobre el manifiesto viejo de 19 (14 diarias) ≈ azar; precisión de equilibrio 0,6–0,7; sincronización perfecta con TP causal da 8–10, no 19. **No aplica a la selección actual de Musashi (15 horarias).** | `satoshi/pivot-detector-20261019` `PIVOT_DETECTOR_EXPERIMENT.md`. |
| K7 | Ruido gaussiano alrededor de la verdad es optimista; abstención gratuita y llenado en el extremo exacto son artefactos; el error del rango total cancela errores por lado; un óptimo sobre un límite de parámetro es artefacto. | `LEARNINGS_CARRIED_FORWARD.md` 11–18. |
| K8 | Datos: calendario 2020–2024 y EVZ no están en el lago; el CSV 2011–2021 es export de Investing.com (solo investigación, reloj UTC−5 fijo hasta 2018-01 y seudo-DST después); fuente automatizable recomendada FXMacroData + agendas oficiales. | financial-data `satoshi/calendar-sources-20261019` @6d99bc44b. |
| K10 | Fractales causales (Retsu) para EURUSD H1 — CERRADO 2026-10-19: plugin feature-eng `causal_fractal` (bit-idéntico al módulo de Retsu), almacén fuera del lago `/var/tmp/features-candidates/eurusd_h1_causal_fractal/` (TRAIN 24.097 / VAL 6.243 filas, sin TEST), ventanas 200/96/48; manifiesto sha 906578e1…; 21 pruebas (perturbación del futuro, replay de fila real). Redundancia baja con volatilidad (|r| ≤ 0,13; spectrum_width vs rv24 0,23–0,28); fractal_dimension_graph = −hurst_rs (redundante); hurst_rs sesgado arriba vs variograma. Candidatas listas para A3; ningún valor predictivo medido. | feature-eng `satoshi/causal-fractal-eurusd-20261019` @6c028aa |
| K9 | Herramientas de selección: entrada pública `dataset_selection` (feature_selector); motores en predictor `tools/` y causal-inference `provider/`; un target nuevo requiere código en los dueños. Selección actual de Musashi para RL: 15 variables horarias (identidad 553fc047…). | `feature-selection-tooling-map` (memoria), `SELECTED_CORE_RL_2026_10_19.md`. |

## 3. Carriles activos (2026-10-19)

| ID | Carril | Repo/rama | Entregable esperado |
| --- | --- | --- | --- |

## 4. Pendientes (ordenados; se tachan al cerrarse)

1. [ ] Integrar A1–A4 en este plan y en `LEARNINGS_CARRIED_FORWARD.md`; actualizar K5/K6 con los números reales.
2. [ ] Si A1 cambia el hash de las demostraciones: re-sellar `expert_trades.csv`, avisar a Musashi (la escalera RL reinicia en E0).
3. [ ] Decidir con el propietario: ¿2024 entra al monitor de parada (pasa a desarrollo y se necesita periodo de papel) o sigue como confirmación única? (hoy: confirmación única).
4. [ ] Selección con target de sincronización: correr de punta a punta sobre inventario + pool de precio multiescala + fractales; manifiesto sellado reutilizable por gym-fx.
5. [ ] Experimento de cadencia de reajuste como brazos declarados (estático, semanal warm-start con el runner prequential auditado, diario, por tick / test-time training) sobre el mejor detector.
6. [ ] Estrategia integrada (`INTEGRATED_STRATEGY_PLAN.md`): motor de ganancia (régimen → modo, meta-etiquetado) y motor de riesgo (tamaño por volatilidad, cierres por estado, freno por drawdown); solo si algún módulo muestra skill en 2023.
7. [ ] Calendario: pedir a Musashi (ya solicitado) FXMacroData 2020–2024 con consenso punto-en-tiempo; al llegar, repetir `CALENDAR_RANGE_EXPERIMENT` sin cambios y el estudio causal con reloj reparado.
8. [ ] gym-fx: antes de cualquier entrenamiento RL con experto, los 8 cambios de la nota de diseño (SL/TP en el espacio de acción impuestos por el entorno, spread por hora, swap, calendario de aplanado, decisiones repetidas, unidades, frecuencia). Dueño: Musashi; Satoshi entrega la lista.
9. [ ] Datos de minutos alrededor de anuncios (cambia K4): solo si el propietario lo autoriza como adquisición.
10. [ ] Broker: verificar costos, spread por hora, swap, `minimumTradeSize`/`marginRate` reales desde la cuenta (configuración privada); hoy todo es ASSUMED_NOT_BROKER_VERIFIED.
11. [ ] DEAP sobre `VolTrendStrategy`/estrategia integrada con datos reales solo cuando exista un módulo con skill; nunca sobre 2024.
12. [ ] Evaluación de TabICLv2 / Chronos-2 como proveedores M5PHET para rango y primer toque con contexto causal purgado (ver `jev-alternatives-evaluation`), condicionada a 4.
13. [ ] Reporte de paridad gym-fx para cada estrategia nueva (regla permanente).
14a. [x] 2026-10-19 Solicitud de auditoría del PLAN de selección publicada para Musashi (predictor `satoshi/rl-from-expert-note-20261019`, `AUDIT_REQUEST_TIMING_SELECTION_PLAN_2026_10_19.md`); el sello de A3 queda PRELIMINARY_PENDING_PLAN_AUDIT hasta incorporar sus hallazgos.
14b. [ ] 2026-10-19 Auditoría de Musashi al plan: REVISE_BEFORE_FINAL_SEAL (predictor `codex/data-coverage-repair-20261018` @ed5b1acc, `MUSASHI_TIMING_SELECTION_PLAN_AUDIT_2026_10_19.md`). Incorporar F01–F09 y demostrar A01–A12 antes del sello final: ledger de soporte de etiquetas y censura (F01), 2024 = desarrollo ya observado, no confirmación prístina (F02), paridad de negocio con el adaptador gym-fx fijado y fills ejecutables (F03), conteo 88 y perfiles de fase 1 de familias nuevas con join en raíz reparada (F04), T1 anticipatorio primario / T2 comparador / T3 definido (F05), ledger de pruebas e incertidumbre consciente de la selección (F06), etapa causal = asociación predictiva condicional con test CI mixto (F07), DAG de invalidación (F08), manifiesto TIMING_PROXY_SELECTION (F09). Terminología: aprendizaje con demostraciones de un experto retrospectivo, no RLHF.
14. [x] 2026-10-19 Solicitud de auditoría de RESULTADOS publicada: predictor `satoshi/timing-selection-20261019` `docs/timing_selection/eurusd_h1_v2/AUDIT_REQUEST_TIMING_SELECTION_FOR_MUSASHI.md`. Pendientes derivados: 14g [en curso] rerun de TIMING_PROXY_SELECTION v3 con el experto v2 (etapas dependientes del target, ruta ejecutable b+1) y solicitud de auditoría v3. 14c join con raíz reparada (A05), 14d copia física independiente en el worker secundario (A12), 14e corregir latencia b+2 y guarda 12 h en el adaptador (aditivo) antes de cualquier estrategia sobre pivotes, 14f decisión independiente de Musashi.
14-old. [x] Al cerrar A3: redactar SOLICITUD DE AUDITORÍA para Musashi (qué se seleccionó, target del experto, etapas corridas, decisión 2023 / confirmación 2024, limitaciones, qué atacar primero) y publicarla junto al manifiesto. Orden del propietario 2026-10-19.

## 5. Decisiones del propietario registradas

| Fecha | Decisión |
| --- | --- |
| 2026-10-18 | PDT eliminado para el negocio; estrategia corto/largo a 72 h obsoleta; Satoshi = estrategias ML, Musashi = RL. |
| 2026-10-19 | Primero volatilidad simulable (negativo), luego ZigZag como experto para RL con guía; SL+TP obligatorios y efectivos; unidades por `rel_volume`×`percentual_order_size`; apalancamiento 1:10/50/100; parada temprana training+validación; GPU 4090→4070→5070 Ti; calendario como inferencia causal (paciente/droga/efecto; selección de eventos). |
| 2026-10-19 | Plan vivo obligatorio; actualizarlo en cada avance. |
| 2026-10-19 | La selección nueva es para RL guiado por el experto ZigZag, no para predicción; al terminarla, solicitud de auditoría a Musashi. |
| 2026-10-19 | Sin entregas parciales: la selección se termina completa y a calidad plena (todas las etapas, PCMCI real, brazos neuronales, tablas completas) antes de avisar; el humano del "RL asistido" son los trades del ZigZag. |

## 6. Cómo se actualiza

Cada retorno de carril: (1) mover el carril de §3 a §2 con su veredicto y link; (2) tachar o añadir pendientes en §4; (3) registrar decisiones nuevas en §5; (4) commit en la rama de este plan y link al propietario. Nada se borra: lo superado se marca SUPERSEDED con fecha.
