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
| TP / SL | cabezas de regresión continuas en σ del propio abridor; el entorno las **recorta** a [0.25, 3] / [0.5, 3] | aceptable; recorte en vez de acotado diferenciable (§2-D5) |
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
- **D5 — TP/SL recortados por el entorno** en vez de acotados en la red (gradiente nulo fuera del rango; la cabeza
  no sabe que su salida se recorta).

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
