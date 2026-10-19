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
| K5 | Experto ZigZag (Depth 3 / Dev 36,3 / Backstep 2, NSGA-II preliminar): Sharpe 17–19; versión confirmada causalmente −1 … −2,2; brecha ≈ 21 (mitad sincronización, mitad nivel del siguiente pivote). Paridad gym-fx del experto: neto +2,7 %, drawdown igual. | `ZIGZAG_EXPERT_STUDY.md`, `results/zigzag/`. |
| K11 | A2 CERRADO 2026-10-19: detector de pivotes sobre las 15 variables reales de Musashi (contrato reproducido: 13 obs stride 2, z-score causal 256; 4090). AUC 0,66 (mejora desde 0,52 del manifiesto viejo) pero calibración peor que la base, precisión 0,34–0,37 < base 0,38–0,39; Sharpe m1 −2,1/−3,5, m2 −1,5/−1,2, m3 +0,07/−2,0 (memoriza en TRAIN). Equilibrio requiere precisión 0,6–0,7. E0 de imitabilidad ≈ 0 también con sus 15. | `satoshi/pivot-detector-20261019` @052dda5, `PIVOT_DETECTOR_EXPERIMENT.md` §9 |
| K6 | Detector de pivotes sobre el manifiesto viejo de 19 (14 diarias) ≈ azar; precisión de equilibrio 0,6–0,7; sincronización perfecta con TP causal da 8–10, no 19. **No aplica a la selección actual de Musashi (15 horarias).** | `satoshi/pivot-detector-20261019` `PIVOT_DETECTOR_EXPERIMENT.md`. |
| K7 | Ruido gaussiano alrededor de la verdad es optimista; abstención gratuita y llenado en el extremo exacto son artefactos; el error del rango total cancela errores por lado; un óptimo sobre un límite de parámetro es artefacto. | `LEARNINGS_CARRIED_FORWARD.md` 11–18. |
| K8 | Datos: calendario 2020–2024 y EVZ no están en el lago; el CSV 2011–2021 es export de Investing.com (solo investigación, reloj UTC−5 fijo hasta 2018-01 y seudo-DST después); fuente automatizable recomendada FXMacroData + agendas oficiales. | financial-data `satoshi/calendar-sources-20261019` @6d99bc44b. |
| K10 | Fractales causales (Retsu) para EURUSD H1 — CERRADO 2026-10-19: plugin feature-eng `causal_fractal` (bit-idéntico al módulo de Retsu), almacén fuera del lago `/var/tmp/features-candidates/eurusd_h1_causal_fractal/` (TRAIN 24.097 / VAL 6.243 filas, sin TEST), ventanas 200/96/48; manifiesto sha 906578e1…; 21 pruebas (perturbación del futuro, replay de fila real). Redundancia baja con volatilidad (|r| ≤ 0,13; spectrum_width vs rv24 0,23–0,28); fractal_dimension_graph = −hurst_rs (redundante); hurst_rs sesgado arriba vs variograma. Candidatas listas para A3; ningún valor predictivo medido. | feature-eng `satoshi/causal-fractal-eurusd-20261019` @6c028aa |
| K9 | Herramientas de selección: entrada pública `dataset_selection` (feature_selector); motores en predictor `tools/` y causal-inference `provider/`; un target nuevo requiere código en los dueños. Selección actual de Musashi para RL: 15 variables horarias (identidad 553fc047…). | `feature-selection-tooling-map` (memoria), `SELECTED_CORE_RL_2026_10_19.md`. |

## 3. Carriles activos (2026-10-19)

| ID | Carril | Repo/rama | Entregable esperado |
| --- | --- | --- | --- |
| A1 | NSGA-II ZigZag completo (pop 64 × ≥80 gen × 3 semillas, genes log, Depth 2–200, Dev 1–300 pips, Backstep<Depth, hipervolumen, frentes retrospectivo y confirmado, sensibilidad tope diario) | heuristic-strategy `satoshi/short-horizon-strategy-20261018` | frente, convergencia, nuevas demostraciones selladas |
| A3 | Selección de características PARA RL GUIADO POR EXPERTO (no para predicción de precio): target `timing_pivot` derivado de la señal ZigZag optimizada (sus picos y valles = tick de cierre de una orden y apertura de la siguiente) + adaptador causal + evaluador predictivo (detector de pivotes + Sharpe de la estrategia trivial). El código vive en los repos dueños de los motores (predictor `tools/`, causal-inference `provider/`) y se fija desde `dataset_selection`; eso es ubicación de código, no objetivo. | predictor / causal-inference / dataset_selection | mapa del tooling, manifiesto sellado, tabla causal, E0 nuevo vs 15 de Musashi; al cerrar → solicitud de auditoría a Musashi |

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
14. [ ] Al cerrar A3: redactar SOLICITUD DE AUDITORÍA para Musashi (qué se seleccionó, target del experto, etapas corridas, decisión 2023 / confirmación 2024, limitaciones, qué atacar primero) y publicarla junto al manifiesto. Orden del propietario 2026-10-19.

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
