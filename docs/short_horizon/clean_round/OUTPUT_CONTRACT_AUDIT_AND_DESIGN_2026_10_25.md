# Contrato de salidas del agente E3-M: auditoría medida y diseño v1 (Satoshi, 2026-10-25)

Para el propietario y Musashi. Las salidas exigen el mismo rigor que las entradas: cada decisión categórica con su
codificación categórica, cada parámetro continuo con su cabeza continua acotada, una capa superior explícita y
optimizable, y **ninguna elección sobre el bloque con el que se decide o se reporta**. Esta página audita lo que hay
(leyendo el código, `tools/expert_guided_e3m.py`, y midiendo sus registros) y propone el contrato v1. TEST cerrado.

## 1. Lo que hay hoy

| pieza | implementación | ¿correcto? |
| --- | --- | --- |
| abrir largo / abrir corto | dos clasificadores binarios independientes (`open_l`, `open_s`), softmax + CE con pesos 1/√frecuencia | la **codificación** es correcta (categórica); su **comparación** no (§2-D3) |
| cerrar largo / cerrar corto | dos binarios (`close_l`, `close_s`), EXIT = "el SL llegaría antes que el TP" | correcto |
| TP / SL | cabezas de regresión continuas en σ del propio abridor, **acotadas dentro de la red** con lo + (hi − lo)·sigmoid(z) a [0.25, 3] / [0.5, 3], pérdida Huber **enmascarada a las filas cuya etiqueta es una APERTURA de ese lado** (`labels_for` → `is_open`) | correcto (ver la retractación de D5) |
| **NOP** | **no existe como especialista**: es el residuo "ningún abridor supera τ" | **defecto** frente al diseño ordenado (NOP es el más importante) |
| capa superior (estrategia) | uno de 3 árbitros fijos (`prob` con τ de una rejilla de 9 tasas, `expected_gain`, `gain_over_risk`), **elegido por semilla con el Sharpe de INNER_A** | **defecto grave** (§2-D1, D4) |
| cerrar-y-revertir | si el cerrador sale y el árbitro elige el lado opuesto, revierte | correcto |

No hay ninguna salida continua usada para escoger entre clases: esa falla es de otra línea, no de ésta.

## 2. Defectos, medidos

- **D1 — elección sobre el bloque de decisión (grave).** Por semilla se prueban 12 configuraciones de árbitro/τ en
  INNER_A y se congela la de mayor Sharpe; ese mismo Sharpe de INNER_A decide entre brazos. Medido sobre los 30
  pares brazo×semilla de A/A' (`evidence/output_audit/ARBITER_SELECTION_OPTIMISM.json`): el elegido supera a la
  mediana de los candidatos válidos en **+1.08 de Sharpe en promedio** (rango hasta +2.6). Es del tamaño de todas las
  diferencias entre brazos de hoy. Las decisiones de A/A' a 5 semillas quedan como **sesgadas al alza en el mismo
  sentido para todos los brazos**, no como medidas. El walk-forward en curso **no** tiene este defecto: allí τ se
  elige en el semestre interno y el año de validación (INNER_B) se puntúa después de congelar.
- **D2 — sin especialista NOP.** "No operar" es lo que queda cuando τ no se supera; nadie aprende *cuándo no operar*.
- **D3 — probabilidades no calibradas comparadas por argmax.** `open_l` y `open_s` se entrenan por separado, con su
  propio balance de clases; `argmax(pL, pS)` compara dos escalas distintas. Ninguna calibración (temperatura,
  isotónica) ni métrica de calibración (Brier, ECE) se mide hoy.
- **D4 — la actividad es un hiperparámetro elegido, no un comportamiento aprendido.** Entre los 12 candidatos de una
  misma semilla las órdenes/semana van de **0.0 a 4.2**; en **9 de 30** semillas existe un candidato con **cero
  órdenes**. Que todos los brazos reporten 2–4 trades/semana es efecto de elegir τ entre los válidos. La capa superior
  no está optimizada: es una búsqueda en rejilla sobre el bloque evaluado.
- **D5 — RETRACTADO (2026-10-25, al implementar).** Afirmé que el entorno recortaba TP/SL en vez de acotarlos en la
  red. Es falso: `TemporalPolicyNet` ya devuelve lo + (hi − lo)·sigmoid(z) con las cotas del entorno, y la pérdida
  Huber ya está enmascarada a las filas de apertura de ese lado. Lo que sí es cierto y se reporta ahora: el objetivo
  de SL de los demostradores ZigZag es la constante 2σ, así que la cabeza de SL aprende casi una constante
  (`tp_sl_calibration_block.sl_target_std` en E3M.json).

## 3. Contrato de salidas v1 (propuesta)

**Especialistas** (cinco, cada uno con su etiqueta y su métrica; todas las etiquetas del lado del entrenamiento):

| especialista | salida | codificación | pérdida | métricas por cabeza |
| --- | --- | --- | --- | --- |
| `open_l`, `open_s` | P(abrir es correcto) | binaria, softmax 2 clases | CE con pesos declarados | AUC, Brier, ECE, curva de tasa |
| `nop` | P(no hay nada que ganar o el trade es riesgoso) | binaria | CE | ídem; etiqueta = NOP duro/graduado de la escalera multi-resolución (su mejor uso medido) |
| `close_l`, `close_s` | P(salir ahora) | binaria | CE | ídem |
| TP, SL (por abridor) | múltiplos de σ | continua acotada en la red: `lo + (hi − lo)·sigmoid(z)` | Huber | MAE en σ frente a la ganancia/adversa realizada, cobertura de cotas |

**Calibración**: temperatura por especialista ajustada en un bloque de **calibración** (el último tramo del ajuste,
nunca el bloque de decisión ni el reportado); Brier y ECE antes y después.

**Capa superior (la estrategia), explícita y optimizable**: entradas = probabilidades calibradas (pL, pS, pNOP, pClose)
+ TP/SL; parámetros = umbral de apertura, veto NOP (pNOP > v ⇒ no abrir), margen de ganancia esperada
(p·TP − (1−p)·SL − costo > m), permiso de reversión. Se optimiza con NSGA-II (Sharpe↑, DD↓; ≥ 2 trades/semana como
restricción de validez, sin tope semanal) **en el bloque de calibración**, se congela, y se puntúa **una vez** en el
bloque de decisión. En el walk-forward: calibración = semestre interno, decisión = año siguiente.

**Contrato con identidad**: `ACTION_CONTRACT_v1.json` (espacio de acciones, cabezas, codificación, cotas, calibración,
capa superior, bloques), con digest, igual que la receta de entradas; propuesto como **contrato compartido con la
línea SAC de Musashi**, para que las dos políticas se comparen con las mismas acciones y el mismo contrato de replay.

**Pruebas antes de entrenar**: codificaciones y cotas (unitarias), calibración en datos sintéticos con respuesta
conocida, la capa superior refusa elegir sobre el bloque de decisión (prueba de mutación), aditividad (los modos
actuales byte-idénticos).

## 4. Qué cambia y qué no

- El walk-forward que corre ahora sigue: es la medición honesta del diseño actual y será la línea base del v1.
- Las decisiones de A/A' se reetiquetan como **sesgadas por D1**; ninguna se usa como evidencia de mejora.
- Siguiente: implementar v1 (código + pruebas, CPU), correrlo en el walk-forward contra la línea base, decidir por la
  regla pre-registrada (sostiene si Sharpe − 1 EE > 0 en ambos años; separable si |Δ| > EE agrupado).

## 5. Implementación v1 (2026-10-25)

Contrato `contracts/ACTION_CONTRACT_v1.json` (identidad `action_1e8767f7877eb757`), módulo
`app/short_horizon/expert_guided/action_contract.py`, modo E3-M `--action-contract` (+ `--strategy-fitness`), ensamble
v1 en `tools/expert_guided_seed_ensemble.py`, etapa `WF_V1` en el lanzador y el informe `wf-v1`. Lo que cambió respecto
al diseño de §3, por las adendas del propietario:

- **Calibración Platt** (a·z + b sobre el margen del logit), no sólo temperatura: los pesos 1/√frecuencia desplazan el
  prior y una temperatura no lo quita. Brier, ECE (15 bins de igual masa) y tabla de fiabilidad antes y después en
  INNER_A; en INNER_B sólo se reportan.
- **Usos de INNER_A declarados**: parada temprana, calibración, optimización de la estrategia y la estrategia de
  visita de DAgger. INNER_B no toca ninguno; prueba de mutación: envenenar INNER_B deja idénticos las mejores épocas,
  los mapas Platt, las estrategias de visita y la congelada.
- **NOP coherente con los abridores**: si el brazo abre en la fila, NOP = 0; NOP = 1 sólo donde el brazo no abre y la
  escalera dice nada / graduado / negativo / conflicto (con su peso). Tabla de coherencia y prueba de información
  (AUC de pNOP frente a 1 − máx(pL, pS), Δ log-loss) en INNER_A. El veto NOP queda sólo si mejora estrictamente la
  cota inferior del bloque de calibración; si no, se desactiva y se reporta.
- **Estrategia**: pocos parámetros acotados (umbral de apertura, margen de ganancia, umbral de cierre, reversión
  booleana, veto NOP). Búsqueda NSGA-II (Sharpe ↑, DD ↓); el punto congelado lo elige una **variante de fitness
  declarada** entre los puntos válidos del frente: F0 `sharpe_lcb` (por defecto), F1 `sharpe_plus_return`,
  F2 `sharpe_plus_return_plus_activity`, F3 `trade_tstat`, F4 `calmar`. Validez: ≥ 2.5 trades/semana en INNER_A, al
  menos un trade y Sharpe definido (si no, fitness −∞, nunca 0). Costo = COST_SIGMA + costo real del contrato de replay
  convertido a σ.
- **DAgger**: las rondas 1–3 visitan con la estrategia congelada en INNER_A tras la ronda anterior.
- **Ensamble**: media de probabilidades calibradas y TP/SL, **una** recalibración Platt sobre el ensamble en INNER_A y
  la estrategia re-optimizada en INNER_A; nunca la estrategia de una semilla. La decisión usa el ensamble.
- **Etapa WF_V1** pre-registrada en el spec v6: `both_hold_v1` y `zz_hold_v1`, 5 semillas, ambos pliegues, línea base
  = el mismo brazo del WF; todas las variantes de fitness evaluadas sobre los mismos ensambles; la variante se elige en
  el año de decisión de wf2022 y se confirma o refuta en wf2023; < 2 trades/semana en INNER_B = INVALID.
