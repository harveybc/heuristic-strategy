# El ZigZag multi-resolución como retroalimentación experta: diseño formal, arquitecturas, métricas y testigos (2026-10-26)

Para el propietario y para la auditoría de Musashi. Todo lo que aquí se afirma tiene un archivo, una función y un
testigo ejecutable (`tools/expert_feedback_witnesses.py`, `tests/test_expert_feedback_witnesses.py`, 9 pruebas, CPU,
sin datos de mercado) o un contador en el código de producción (§7). Lo que no está probado se declara en §8. Las
mejoras propias de este documento frente al diseño del responsable están marcadas **[mejora propuesta]**.

**Resumen.** El ZigZag multi-resolución no vuelve a ser el maestro que se imita (la auditoría demostró que imitar sus
pivotes enseña a perder: TP 0.43σ contra SL constante 2σ, 30 % de positivos con pérdida incluso con clarividencia). Pasa
a ser lo que es en RLHF un anotador humano: una fuente de **preferencias por tick** sobre las acciones {abrir largo,
abrir corto, mantener/NOP, cerrar}, con una **fuerza** π(t). La **realidad** —la tabla de resultados exacta del motor,
que para nuestro tamaño (sin impacto de mercado) da el contrafactual de **toda** acción en **todo** tick— decide qué
es correcto; la preferencia decide **qué ticks importan y cuánto**. Los cinco especialistas y el modelo de recompensa
comparten el tronco TCN de Musashi (una conv causal por grupo → fusión 1×1 a 8 → dos bloques residuales, campo
receptivo = ventana) con la codificación por tipo de dato de la selección v5 aplicada. Las métricas de conducta se
calculan en cada tick (conjunto deseado D(t), indeseado U(t), acción del estudiante, Q de todas las acciones, arrepentimiento
exacto) y se publican en un libro por tick para auditoría. Siete testigos sintéticos prueban las definiciones:
agreement por encima del azar con control negativo barajado, identidad del arrepentimiento exacto, disponibilidad
(perturbar el futuro no cambia etiquetas), ausencia de retrospectiva en el modelo de recompensa, cobertura total de
tipos de tick y reproducción exacta del agregado del motor desde el libro. Un hallazgo honesto del propio testigo: la
obediencia del estudiante a las preferencias de **no operar** es la más débil (0.21–0.27 en el juguete), y la fuerza de
preferencia hindsight **no** se traduce monótonamente en conducta causal; por eso el NOP es especialista propio y el
objetivo de la política es el arrepentimiento real, no la imitación.

---

## §1. Objetos: la escalera, las cantidades por tick, D(t), U(t), π(t)

**Escalera.** R = {opt (Depth 3, Dev 40.7, Back 0: el óptimo registrado), rev20, rev15, rev10, rev8, rev5} (umbrales de
reversión en pips). En producción: `multires.LADDER`, pivotes por `multires.resolution_pivots` / `app.short_horizon.zigzag`.
En el testigo: `expert_feedback_witnesses.reversal_pivots` (múltiplos de σ media: 4, 3, 2.5, 2, 1.5, 1).

**Tramo (leg) en hindsight.** Para la resolución r, el tramo que contiene el tick t es [s_r(t), e_r(t)) entre dos
pivotes consecutivos, con niveles L_s (pivote de inicio) y L_e (pivote de fin), |leg| = |L_e − L_s|.

**Disponibilidad (sólo entrenamiento, nunca observación).** Una cantidad del tramo es usable en el tick t sólo si el
pivote de **fin** está **confirmado** y su barra de confirmación + 1 h ≤ fin del periodo:
`multires.Tables.open_available`, `close_available` (producción); `available()` (testigo). Testigo W4.

**Cantidades por tick, lado d ∈ {+1, −1} y resolución r** (producción `multires.leg_tables`; testigo `leg_tables`):

| cantidad | fórmula | rango | significado |
| --- | --- | --- | --- |
| aptitud de apertura | f_open^r(t,d) = d·(L_e − open_{t+1}) / \|leg\| | [−1, 1] | 1 = abrir en el inicio óptimo; < 0 = abrir **contra** el tramo (ejecución en la apertura siguiente, como el motor) |
| aptitud de cierre | f_close^r(t,d) = d·(close_t − L_s) / \|leg\| | ≤ 1.5 (extremo intrabarra) | fracción del tramo ya capturada por una posición abierta en L_s |
| restante | g^r(t,d) = d·(L_e − close_t) / σ_t | σ causal | lo que queda hasta el cierre óptimo, en σ del tick |
| MAE | excursión adversa máxima entre open_{t+1} y e_r, / \|leg\| | ≥ 0 | riesgo del trade dentro del tramo |
| leg/σ | \|leg\| / σ_t | ≥ 0 | si el tramo paga la volatilidad |
| votos | n_L(t) = #{r : f_open^r(t,+1) ≥ θ_f}, n_S análogo | 0..\|R\| | consenso entre resoluciones |
| f* | f*(t,d) = max_r f_open^r(t,d) | [−1, 1] | la mejor lectura del lado d |

σ_t es la volatilidad **causal** del dataset (`Dataset.sigma`, sólo barras ≤ t).

**Umbrales** (`multires.Thresholds`, valores por defecto; en C3 se re-optimizan sólo en el bloque de selección):
θ_f = 0.5, θ_mae = 0.5, θ_vol = 0.25σ, θ_close = 0.1, take = 0.9, min_votes = 1.

**Conjuntos deseado D(t) e indeseado U(t) y fuerza π(t), tick plano** (`multires.origin_label` → `kind`; testigo
`expert_tick`):

| caso (en orden) | condición | D(t) | U(t) | π(t) |
| --- | --- | --- | --- | --- |
| NOP-duro | n_L = n_S = 0 | {HOLD} | {OPEN_L, OPEN_S} | 1 |
| conflicto | n_L ≥ 1 y n_S ≥ 1 | {HOLD} | ∅ | clip(1 − f*, 0, 1) |
| negativo | un lado d, f*(t,d) < 0 | {HOLD} | {OPEN_d} | min(1, \|f*\|) |
| NOP-graduado | f* < θ_f, o MAE/leg > θ_mae, o leg/σ < θ_vol (en el óptimo) | {HOLD} | {OPEN_d} | clip(1 − f*, 0, 1) |
| abrir | lo demás | {OPEN_d} | {OPEN_−d, HOLD} | (n_d / \|R\|) · f* |

**Tick posicionado, lado d** (`multires.close_label`; testigo `expert_close`), con rem_frac = 1 − f_close^opt:

| condición | D(t) | U(t) | π(t) |
| --- | --- | --- | --- |
| rem_frac < θ_close **o** f_close ≥ take | {CLOSE} | {HOLD} | clip(\|rem_frac − θ_close\| / θ_close, 0, 1) |
| en otro caso | {HOLD} | {CLOSE} | ídem |

D(t) ∩ U(t) = ∅ y D(t) ≠ ∅ en todo tick (prueba `test_desired_and_undesired_sets_are_disjoint_everywhere`).

**Ejemplo numérico (tramo de juguete, 10 barras, valle 100.0 en t=0, pico 104.0 en t=9, σ = 1, apertura = cierre
anterior; `worked_example()`):**

| t | open_{t+1} | f_open | f_close | restante (σ) | MAE/leg | abrir deseado | cerrar deseado |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 100.0 | 1.000 | 0.000 | 4.0 | 0.000 | sí | no |
| 1 | 100.5 | 0.875 | 0.125 | 3.5 | 0.000 | sí | no |
| 2 | 101.2 | 0.700 | 0.300 | 2.8 | 0.075 | sí | no |
| 3 | 100.9 | 0.775 | 0.225 | 3.1 | 0.000 | sí | no |
| 4 | 101.8 | 0.550 | 0.450 | 2.2 | 0.000 | sí | no |
| 5 | 102.6 | 0.350 | 0.650 | 1.4 | 0.000 | **no** (f* < θ_f → NOP-graduado) | no |
| 6 | 103.1 | 0.225 | 0.775 | 0.9 | 0.075 | no | no |
| 7 | 102.8 | 0.300 | 0.700 | 1.2 | 0.000 | no | no |
| 8 | 103.6 | 0.100 | 0.900 | 0.4 | 0.000 | no | **sí** (f_close ≥ take) |

Lectura: la apertura es deseada en la primera mitad del tramo y pasa a NOP-graduado cuando queda menos de la mitad;
el cierre es deseado al capturar el 90 % o cuando queda < 10 %. Contra el tramo (un corto aquí) f_open < 0 → negativo.

## §2. El ancla de realidad: tablas de resultados exactas y arrepentimiento por tick

Producción: `outcomes.entry_table` (por fila plana, lado y celda (TP, SL) de la rejilla `CELLS` = {0.5, 1, 1.5, 2}σ ×
{0.75, 1.5, 2.5}σ: neto en σ tras costos, `hit` ∈ {TP, SL, TIME}, MAE, MFE, duración, barra de resolución) y
`outcomes.exit_table` (por posición visitada: valor de mantener frente a cerrar ahora, SL-primero). Las dos se calculan
**con el propio motor** (`BarEnv.step`: llenado en la apertura siguiente, costos, SL antes que TP dentro de la barra,
cierre forzoso), 0 de 160 discrepancias contra corridas independientes (auditoría A6). Testigo: `q_open`, `q_table`,
`q_positioned` (motor de juguete con las mismas reglas).

Definición. Para el tick t y la acción a:

- plano: Q(t, OPEN_d) = max_{celda} neto_σ(t, d, celda); Q(t, HOLD) = 0;
- posicionado: Q(t, HOLD) = neto si se mantiene hasta TP/SL/forzoso con los niveles de la orden; Q(t, CLOSE) = neto si
  se sale en open_{t+1}.

**Arrepentimiento exacto** de cualquier política: ρ(t) = max_a Q(t,a) − Q(t, a_política(t)) ≥ 0. Sin impacto de
mercado es exacto, no estimado (peldaño 2 de Pearl calculado, `CAUSAL_LADDER_DESIGN` §2). Testigo W3: la política
argmax-Q tiene ρ(t) = 0 en **todos** los ticks (máximo medido 0.0 en 1 469 ticks); una política aleatoria tiene ρ medio
0.75σ.

**Acuerdo experto–realidad.** Para cada tick, 1[D(t) ∩ argmax_a Q(t,a) ≠ ∅]. Su proporción se **reporta** por tipo de
tick (§5) y entra en los pesos (§3c). No se exige que sea alta: el experto es una preferencia; la realidad juzga.

## §3. Formalización "RLHF"

(a) **Datos de preferencia.** En cada tick con D(t) y U(t) no vacíos, los pares (a⁺ ≻ a⁻) para a⁺ ∈ D(t), a⁻ ∈ U(t)
con fuerza π(t). Modelo Bradley–Terry: P(a⁺ ≻ a⁻ | x_t) = σ(r_φ(x_t, a⁺) − r_φ(x_t, a⁻)).

(b) **Modelo de recompensa** r_φ(x_t, a): tronco TCN (§4) ⊕ one-hot de la acción → escalar; pérdida
L_pref = −Σ_t π(t) Σ_{a⁺,a⁻} log σ(r_φ(x_t,a⁺) − r_φ(x_t,a⁻)). Sus entradas son **sólo** la observación x_t y la
acción (testigo W5: comprobación estática de nombres contra `HINDSIGHT_NAMES` y perturbación del futuro → entradas
idénticas). Las preferencias son hindsight y se usan sólo como objetivo de entrenamiento.

(c) **Objetivo de la política = realidad.** Minimizar J = Σ_t w_t · ρ(t), con
w_t = 1 + λ · π(t) · 1[D(t) ∩ argmax_a Q(t,a) ≠ ∅]. El experto decide **qué ticks pesan más** (cuando la realidad le da
la razón); la realidad decide **qué acción es correcta**. En C3 esto se instancia como: abridores entrenados sobre la
tabla de resultados (3 clases {TP-primero, SL-primero, TIME} + neto esperado por celda), cerradores sobre {SL-primero,
valor restante}, y los pesos por fila w_t en las pérdidas (`learn.fit(sample_weight=…)`, ya existente).

(d) **Moldeado en el optimizador de estrategia.** Fitness = expectativa neta por trade agrupada (primario) + μ · acuerdo
con el experto (secundario, reportado); μ y los umbrales se eligen **sólo** en el bloque de selección (protocolo v11).

(e) **Especialista NOP** entrenado con NOP-duro (peso 1) y NOP-graduado (peso 1 − f*): el agente aprende *cuándo no
operar*, lo más importante según el propietario. Se conserva sólo si su veto sube la cota inferior en calibración.

(f) **Currículo grueso→fino** sobre la escalera (opción declarada): primero preferencias de las resoluciones gruesas
(alta calidad, pocas), luego las finas.

(g) **DAgger con correcciones ponderadas por preferencia** sólo donde la realidad confirma (w_t > 1); la regla de
re-etiquetado es la misma tabla de resultados que la ronda 0 (defecto A2-5 corregido en C3).

**Lo que NO se hace** (y por qué, con el id del defecto): no se imitan barras pivote como etiqueta de entrada (A2-2);
no hay salida continua para elegir clases (auditoría de codificación, E); no hay cabeza de SL con objetivo constante
(A2-1/A4-3: el SL sale de la celda elegida); no se fuerza actividad con VE negativo (A2-8); nada hindsight entra en la
observación (W5).

**[mejora propuesta]** Las preferencias de la escalera se pueden usar además como *reward shaping* denso en el
regret: ρ̃(t) = ρ(t) + ν · 1[a_t ∈ U(t)] · π(t), con ν pequeño y elegido en selección; se reporta pero no es parte del
diseño base.

## §4. Arquitecturas exactas

**Tronco compartido** (`temporal.TemporalTrunk(trunk="residual_tcn")`, paridad exacta con `GroupedTCNExtractor` de
Musashi @e18d00d4, prueba `tests/test_expert_guided_residual_tcn.py`):

1. ramas por grupo de la receta: **una** conv causal por grupo (kernel 3 para grupos de serie, 1×1 para puntuales),
   unidades por grupo según `representation.resolved.units`, SiLU;
2. fusión causal 1×1 a `fusion_units` = 8;
3. dos bloques residuales causales `silu(x + conv2(silu(conv1(x))))`, convs kernel 3, dilataciones 1 y 2 → campo
   receptivo 1 + 2·(1+2)·2 = 13 slots = la ventana (13 slots, stride 2 filas);
4. lectura: aplanar los 13 slots (8 × 13 = 104) — `readout=flatten`;
5. dropout tras ramas y fusión (declarado), AdamW con wd de la receta.

**Cabezas (contrato C3, `ACTION_CONTRACT_C3_TCN.json`, identidad `action_088ee4fdc709f96b`; C3-u conv1d
`action_302289e2b317966e`):**

| cabeza | entrada | codificación de salida | pérdida | objetivo (de dónde sale) | bloque de uso | nunca toca |
| --- | --- | --- | --- | --- | --- | --- |
| open_L, open_S | tronco (estado = ceros) | softmax 3 clases {TP-primero, SL-primero, TIME} por celda de referencia + regresión E[neto] sobre las 12 celdas (conserva orden) | CE ponderada (w_t) + Huber enmascarada a filas con celda definida | `outcomes.entry_table` (motor) | ajuste; parada temprana en INNER_ES; calibración afín en INNER_A | INNER_B |
| nop | tronco | softmax 2 clases | CE con pesos NOP-duro 1 / graduado 1 − f* | `multires.origin_label` kinds | ídem | INNER_B |
| close_L, close_S | tronco ⊕ geometría causal de la propia orden: dist_SL, dist_TP en σ_entrada, horas, lado (C2G sin retrospectiva: posiciones del experto excluidas, exploración con la rejilla del estudiante) | softmax 2 {SL-primero, no} + regresión valor restante | CE + Huber | `outcomes.exit_table` | ídem | INNER_B |
| modelo de recompensa r_φ | tronco ⊕ one-hot acción | escalar | Bradley–Terry (§3b) | pares D/U con π | ajuste; selección de μ en INNER_SEL | INNER_B |
| capa superior (estrategia) | probabilidades calibradas + E[neto] + veto NOP | reglas paramétricas (umbrales por lado como cuantiles del bloque de calibración, margen de VE con costo, umbral de cierre por lado, reversión) | NSGA-II (expectativa ↑, DD ↓) con dominancia por restricciones (≥ 2 trades/sem con IC 90 %, lado ≤ 90 %, cierre temprano requerido) | — | INNER_SEL | INNER_B |

**Codificación por tipo de dato, aplicada** (defecto A1-11: las decisiones de `ENCODING.csv` no se aplicaban):

| tipo | decisión v5 | aplicación |
| --- | --- | --- |
| series de precio/nivel | diff / log decididos por fold en TRAIN (`fold_local_producers.redecide`) | `KEEP_BASE_ENCODING` para price_level y range_relative |
| ángulos (fase de Hilbert, hora, día) | sin/cos en bloques coherentes (salen y entran juntos del SBFS) | `hod → USE_sessions` (bloques de sesión), calendario sin/cos |
| régimen GMM | **one-hot** (`regime_gmm → USE_onehot`) en lugar de p0–p2 crudas | aplicar en la receta nueva |
| relojes de eventos | ventana 2 h (`event_clock → USE_window2h`) en lugar de hours_since/hours_to | aplicar en la receta nueva |
| colas pesadas (curtosis > 20: RAW_ELAPSED_HOURS, level@diff, ret_1) | transformación robusta declarada en TRAIN (log1p con desplazamiento o bandera) | A1-6 |
| constantes (px.er_1) | rechazadas (`STD_ZERO_COLUMN`) | A1-3 |
| duplicados R² ≥ 0.99 | se conserva un representante, registrado | A1-4 |

**Parámetros por módulo (abridor, receta de 121 columnas / 31 grupos; `evidence/residual_tcn/PARAM_COUNTS.json`):**
tronco u (actual) 1 382; **residual_tcn aplanado 3 916**; residual_tcn último slot 2 380. Cinco especialistas + modelo
de recompensa ≈ 6 × 3.9 k ≈ 23 k parámetros con TCN. El SBFS reduce los grupos; con menos grupos la fusión y las
cabezas se recalculan (poda de rama cuando un grupo se vacía).

## §5. Métricas, con fórmulas

**Entrenamiento.**
- CE por cabeza: L_CE = −Σ_t w_t Σ_k y_{t,k} log p_{t,k} / Σ_t w_t, pesos de clase 1/√freq normalizados (existente).
- Huber enmascarada: L_H = Σ_{t∈M} w_t · huber_δ(ŷ_t − y_t) / |M| (normalizada por filas enmascaradas: A2-12).
- Preferencia: L_pref (§3b).
- Arrepentimiento ponderado: J = Σ_t w_t ρ(t) (se reporta en ajuste y calibración; es el criterio de parada
  temprana **[mejora propuesta]** frente a la CE sola).

**Conducta en cada tick** (libro `LEDGER.csv`: tick, tipo, D(t), U(t), π(t), acción del estudiante, Q(t,·), ρ(t),
acuerdo experto–realidad):
- Acuerdo ponderado: A = Σ_t π(t) 1[a_t ∈ D(t)] / Σ_t π(t). Azar: el mismo A con acciones uniformes (20 réplicas).
- Desobediencia: Ū = Σ_t π(t) 1[a_t ∈ U(t)] / Σ_t π(t).
- Curvas de conducta: P(a_t ∈ D(t) | bin de π) sobre ticks de apertura; P(open | bin de f*) y P(close | bin de
  restante) con Spearman e IC; **se reportan como instrumento, no como requisito de monotonía** (ver W1, §8).
- Confusión por **tipo de tick** (`tick_type`, exactamente uno por tick, prioridad): forced_flat_window, nop_hard,
  at_pivot, pre_pivot (≤ 2 barras antes del pivote de fin), post_pivot (≤ 2 tras un pivote), mid_leg. Para cada
  tipo: acuerdo, desobediencia, ρ medio, Q medio de la acción tomada.
- Calibración del modelo de recompensa: r_φ(x_t,a) frente a Q(t,a) realizado (Spearman, fiabilidad por bins).
- Acuerdo experto–realidad por tipo (§2).

**Evaluación (protocolo v11).** Primario: expectativa neta por trade en σ con IC bootstrap por días, agrupada entre
folds; secundarios: Sharpe ± EE de Lo, DD, actividad (IC 90 %), lados, valor del cierre temprano (contrafactual de
mantener), salidas por razón. Elecciones en INNER_SEL; INNER_B una vez tras congelar; libro de lecturas por bloque.

## §6. Testigos (pruebas ejecutables, `tools/expert_feedback_witnesses.py`; resultados con semilla 0)

| id | afirmación | cómo | resultado medido | falsaría |
| --- | --- | --- | --- | --- |
| W1 | con una señal plantada, el entrenamiento por preferencias ponderadas produce un estudiante que obedece a las preferencias por encima del azar, y el instrumento expone dónde desobedece | mercado sintético AR(1) con κ = 0.35; escalera; D/U/π; regresión softmax ponderada por w_t (§3c) en 60 % de los ticks; evaluación en el 40 % restante | A = 0.487 frente a azar 0.335; obediencia en ticks de apertura 0.62; **en ticks HOLD/NOP 0.27** (semillas 1, 2: 0.50/0.67/0.23 y 0.48/0.64/0.21) | A ≤ azar, o curvas idénticas entre tipos |
| W2 | control negativo: preferencias barajadas → el estudiante no puede seguirlas | mismo flujo con D/U/π permutados | A = 0.351 ≈ azar 0.337; obediencia en aperturas 0.00–0.13 | A ≫ azar con preferencias barajadas (habría fuga) |
| W3 | el arrepentimiento exacto es exacto | política argmax-Q contra aleatoria, 1 469 ticks | ρ máximo del oráculo **0.0**; ρ medio aleatorio 0.746σ, mínimo 0.0 | ρ > 0 para el oráculo |
| W4 | disponibilidad: perturbar todo bar ≥ corte no cambia ninguna etiqueta disponible antes del corte | 848 ticks disponibles, bars ≥ 900 reescalados y desplazados | **0 diferencias** en D, U, π, kind | ≥ 1 diferencia |
| W5 | el modelo de recompensa no ve retrospectiva | nombres de entrada ∩ HINDSIGHT_NAMES = ∅; perturbar X y precios futuros → entradas iguales | ambas verdaderas | cualquier nombre hindsight o cambio de entrada |
| W6 | cada tick tiene exactamente un tipo y un D(t) no vacío | 1 469 ticks | 1 469 clasificados (forced_flat 11, at_pivot 68, pre 136, post 123, mid_leg 1 131, nop_hard 0), 0 sin D | tick sin tipo o con dos |
| W7 | el libro por tick reproduce el agregado del motor exactamente | replay secuencial con una política de umbral; suma del libro frente al motor | iguales a 1e-9 (311 trades; 19 cierres propios en libro = 19 en trades) | diferencia > 0 |

**Hallazgo del testigo W1 (no esperado y declarado):** la curva P(a ∈ D | bin de π) **no es monótona** (el bin de
π más alto, dominado por NOP-duro, tiene la menor obediencia: 0.30–0.54). La fuerza de una preferencia hindsight no
se traduce linealmente en conducta de un estudiante causal: el estudiante abre donde la señal dice dirección aunque el
experto diga "no hay nada que ganar a ninguna escala". Esto es precisamente lo que el instrumento debe exponer, y es la
justificación de (i) el especialista NOP entrenado aparte, (ii) el objetivo de arrepentimiento real en vez de
imitación, (iii) el veto NOP en la capa superior evaluado por su valor, no por su obediencia.

## §7. Lista de auditoría para Musashi

| afirmación | archivo:función | prueba | qué la falsaría |
| --- | --- | --- | --- |
| f_open, f_close, restante, MAE, leg/σ como en §1 | `multires.leg_tables`; testigo `leg_tables` | `test_expert_guided_multires.py` (producción), `test_worked_example_matches_the_design_table` | un valor fuera de rango o un signo invertido contra el tramo |
| D/U/π por caso | `multires.origin_label`, `close_label`; testigo `expert_tick`, `expert_close` | `test_desired_and_undesired_sets_are_disjoint_everywhere` | D ∩ U ≠ ∅ o D = ∅ |
| disponibilidad sólo con pivote de fin confirmado | `multires.Tables.open_available/close_available`; testigo `available` | W4 | etiqueta cambia al perturbar el futuro |
| tablas de resultados exactas al motor | `outcomes.entry_table/exit_table` (llaman `BarEnv.step`) | A6: 0/160 discrepancias; W3 | una celda cuyo neto difiera de una corrida independiente |
| arrepentimiento exacto | testigo `regret`, `simulate`; producción: libro por tick (C3) | W3, W7 | ρ(oráculo) > 0 |
| pesos w_t = 1 + λπ·confirmado | testigo `preference_dataset`; producción `learn.fit(sample_weight)` | W1/W2 | pesos que cambien sin cambiar π o la confirmación |
| modelo de recompensa sin retrospectiva | testigo `reward_model_inputs`; producción: entradas = tensor de la receta ⊕ one-hot | W5 | cualquier campo hindsight en la entrada |
| tronco TCN con paridad | `temporal.TemporalTrunk(trunk="residual_tcn")` | `test_expert_guided_residual_tcn.py` (salidas iguales a 1e-6 a la clase de Musashi) | salida distinta con los mismos pesos |
| sin SL constante, sin salida continua para clases | `action_contract` (auditoría de codificación, `STD_ZERO`-target refusal) | pruebas de mutación de C3 (F1) | una cabeza SL con objetivo de std < 0.05 aceptada |
| bloques: nada se elige en INNER_B | protocolo v11 (`data.py` contratos anidados, `read_ledger`) | `test_expert_guided_protocol_v11.py` (mutación de INNER_B → mismos parámetros) | un parámetro que cambie al envenenar INNER_B |
| libro por tick reproduce el motor | testigo `ledger_aggregate` | W7 | suma del libro ≠ neto del motor |

## §8. Lo no probado y los riesgos

1. **Acuerdo experto–realidad real**: medido sólo en el juguete; en datos reales se reporta por tipo de tick y puede ser
   bajo (el experto hindsight y la tabla con costos pueden discrepar en mitad del tramo). Si es bajo, los pesos w_t casi
   no se activan y el método degenera al puro arrepentimiento: aceptable, pero el ZigZag aportaría poco.
2. **Umbrales y μ elegidos en el bloque de selección** (6 meses): riesgo de sobreajuste de la capa superior (A4-8/A5-2);
   mitigado por bloques anidados, PBO y Sharpe deflactado, pero no eliminado.
3. **Redes pequeñas**: 3.9 k parámetros por módulo; la auditoría A6 mostró que las cabezas sobreajustan ~107 distractores
   y que una logística sobre el mismo tensor las superaba; el SBFS y la línea base logística junto a cada cabeza son la
   respuesta, no este documento.
4. **La fuerza π hindsight no es monótona en conducta** (W1): no se debe usar π como métrica de éxito; sólo como peso y
   como instrumento de exposición.
5. **W1/W2 usan una regresión softmax, no el TCN**: prueban las definiciones y el flujo de pesos, no la capacidad del
   tronco; la puerta con el TCN real es el control sintético a 5 semillas de la cadena (A6, pendiente con C3).
6. **TIME como clase**: en el motor real ~50 % de las entradas a 1.5/1.5 terminan en cierre forzoso (A2-3); la cabeza de
   3 clases lo separa, pero el calendario podría dominar la predicción de TIME; se exige la línea base de calendario como
   nulo.
7. **Costo computacional**: cinco especialistas + modelo de recompensa por semilla y por subconjunto del SBFS; el
   presupuesto 128 × 3 es el declarado y puede terminar en BUDGET.

## §9. La estrategia completa, para la auditoría cruzada

Autocontenido. Nombres de parámetros tal como están en `app/short_horizon/expert_guided/c2.py` (contrato C2/C3),
`action_contract.py` y `exec_variants.py`. Lo que no está en archivo se declara como no probado (§9d).

### (a) Salidas exactas del modelo y cómo se usan, decisión por decisión

**Cabezas** (contrato C3, módulos `open_l`, `open_s`, `nop`, `close_l`, `close_s`; `c2.modules`):

| cabeza | salida | codificación | rango | calibración |
| --- | --- | --- | --- | --- |
| `open_d.hit` (d ∈ {l, s}) | P(TP primero), P(SL primero), P(TIME) en la celda de referencia `REF_CELL` = (1.5σ, 1.5σ) | softmax 3 clases | simplex | Platt binaria por clase relevante (`platt_binary`, bloque `INNER_A`, 15 bins ECE) |
| `open_d.value:net_<celda>` | neto esperado en σ tras costos para cada celda (TP, SL) ∈ {0.5, 1, 1.5, 2} × {0.75, 1.5, 2.5} (12 valores) | regresión acotada en la red (`reg_bounds`: lo + (hi − lo)·sigmoid) | [−SL − costo, TP − costo] por celda, recortado al tope efectivo por barra (`exec_variants.effective_cap`) | sin calibración (regresión); se reporta Spearman frente al neto realizado |
| `nop.NOP` | P(no hay nada que ganar o el trade es riesgoso) | softmax 2 clases | [0, 1] | Platt |
| `close_d.SL_FIRST` | P(el SL llega antes que el TP desde aquí), con geometría causal de la propia orden | softmax 2 clases | [0, 1] | Platt |
| `close_d.value:remaining_sigma` | valor de mantener frente a cerrar ahora, en σ | regresión acotada | declarado en el contrato | — |

**Genes de la estrategia** (`c2.GENES_Q13`, todos como tasas/cuantiles del bloque de calibración, mapeados una vez y
congelados; `Q_BOUNDS_13`): `open_rate_l`, `open_rate_s` ∈ [0.002, 0.30]; `close_rem_rate` ∈ [0, 0.60];
`close_p_rate` ∈ [0, 0.60]; `allow_reverse` ∈ [0, 1] (≥ 0.5 = sí); `nop_veto_rate` ∈ [0, 0.50]. Mapa
(`gene_map`): `open_margin_l` = cuantil_{1 − open_rate_l} del mejor neto previsto LONG sobre las celdas en las filas planas
de calibración; `open_margin_s` análogo; `close_margin` = −cuantil_{close_rem_rate} del valor restante previsto en las
visitas posicionadas de calibración; `close_p` = cuantil_{1 − close_p_rate} de P(SL primero) calibrada; `nop_veto` =
cuantil_{1 − nop_veto_rate} de pNOP calibrada. Restricciones (dominancia de Deb): actividad ≥ 2.0 trades/semana con la
**cota inferior del IC 90 %** en calibración (`activity_interval`), lado ≤ `SIDE_CAP` = 0.90, `require_early_close`
(`early_close_ok(share ≥ 0.05, value_lcb ≥ 0)`); prueba de factibilidad (`feasibility_probe`) → `STRATEGY_SPACE_INFEASIBLE`.

**Tick plano, en orden** (`c2` regla `rule`, `ContractPolicy`):
1. Veto NOP: si pNOP_cal(t) > `nop_veto` → HOLD (si el veto se conservó: sólo cuando sube la cota inferior en calibración).
2. Por lado d: celda* = argmax_celda `open_d.value:net_<celda>`; neto*_d = su valor.
3. Candidato: el lado con mayor neto*_d si neto*_d > `open_margin_d`; si ninguno supera su margen → HOLD. (El neto
   previsto ya incluye costo; no se fuerza actividad: "sin trade" es válido.)
4. Puerta mejor-del-día (`exec_variants.BestOfDayGate`, `observe_bars` = 3, regla de secretaria causal): los primeros 3
   ticks elegibles del día de negocio sólo se observan; después se coloca la primera señal cuyo puntaje ≥ el mejor del
   día **y** ≥ `threshold`. Con `observe_bars` = 0 degenera a "la primera que cruza" (lo que la auditoría A4-2 midió).
5. Orden: lado d, fill en open_{t+1}, TP y SL = los de la celda* en σ_{t+1} (sólo barras ≤ t), recortados al tope
   efectivo (`clip_multiples`); tamaño por riesgo (`sizing_mode` = "risk", `risk_pct` = 0.01:
   unidades = ⌊risk_pct · balance / (distancia_SL · valor_pip)⌋, acotado por `rel_volume` ∈ [0.10, 0.20] y apalancamiento
   ≤ 1:50, enteras). Tope diario del contrato de replay (1 entrada/día) aplicado por el entorno (`DAILY_CAP`), nunca en
   las etiquetas.
6. Corte del viernes: `entry_cutoff_hours_before_flat` = 24 → una entrada cuyo fill quede a < 24 h del cierre forzoso se
   rechaza (`VARIANT_CUTOFF`); el calendario original sigue en vigor.

**Tick posicionado, lado d** (parada óptima):
1. CLOSE si `close_d.value:remaining_sigma`(t) < −`close_margin` **o** P(SL primero)_cal(t) > `close_p`; si no, HOLD.
2. Reversión: si CLOSE y `allow_reverse` ≥ 0.5 y la regla de apertura elige el lado opuesto en el mismo tick → se cierra y
   se abre el contrario con su celda*; con `reverse_counts_as_one_entry` = True la reversión cuenta como **una**
   decisión de entrada frente al tope diario (A4-1).
3. El entorno aplica SL/TP/cierre forzoso en cada barra con orden conservador (SL antes que TP dentro de la barra;
   `short_trigger_on_ask` = True desplaza el disparo del stop corto al ask).

Variante de ejecución declarada para C3: `full_f3` = `exec_709f9b1fa3512486` (riesgo 1 %, reversión = una decisión,
corte 24 h, disparos ask); control `legacy` = `exec_5d80473b21dc61a0`. La identidad del contrato C3 sella arquitectura
(`trunk`, `fusion_units`, `core_blocks`, `readout`), celdas, genes, mapa, calibración y variante de ejecución.

### (b) Cómo se prueba la estrategia con datos ideales

1. **Motor** (`evidence/exec_variants/ENGINE_CHECK.md`, barras admitidas 2021-01→2021-04, 1 entrada/día, 1:50, TEST
   nunca leído): política con **visión perfecta** de la dirección a 12 h, TP/SL 1σ/1σ: primera-del-día **+7 536 USD,
   Sharpe 9.37, DD 4.57 %** (47 trades, 21 % de entradas en la hora 0); con `BestOfDayGate(observe 3)` **+8 178 USD,
   Sharpe 14.82, DD 1.41 %**, 0 % de entradas en la hora 0; con riesgo 1 % Sharpe 9.49 / DD 2.24 % (neto menor porque el
   1 % dimensiona por debajo de la exposición fija 5×); corte 24 h: salidas TIME 7 → 3. **Moneda al aire** (5
   semillas): −151 / −83 / −120 / −69 USD según variante ≈ los costos. Sin trade = 0 exacto. Conclusión: el motor, los
   costos y el tamaño no están rotos (también auditoría A4, `synth.py`: +7.5–9.8 k USD con visión perfecta).
2. **Tubería completa con característica trampa** (A6, `PIPELINE_CONTROLS_2026_10_26.md`, rama
   `satoshi/pipeline-controls-20261026` @5b049e8; evidencia compacta `evidence/pipeline_controls_DIAGNOSTIC/`): mismo
   comando E3-M que WF_V13 / C2_13, fold wf2022, 1 semilla, 2022 congelado: v1.3 **Sharpe 8.33**, PF 4.3, 42/58 %
   compra/venta, 68 cierres propios, AUC de cabezas 0.80/0.88; C2 **9.67**, PF 11.6, AUC 0.84; oráculo 14.0. Ruido
   graduado: σ = 0.5 → 9.5/9.0; σ = 1 → 7.5/7.4; σ = 2 → 4.1/2.5 (oráculo 12.0/10.5/6.7).
3. **Mercado sintético con señal moderada plantada** (GARCH + deriva AR(1), κ = 0.1, media vida 12 h; la columna sola
   predice con AUC 0.63; oráculo Sharpe 2.58): v1.3 **0.48** (FALLA), C2 **−0.85** (FALLA), C2 sin distractores 0.49
   (FALLA). Dónde se pierde: etiquetas de pivote (la fuga perfecta las predice sólo a AUC 0.60), cabezas que sobreajustan
   ~107 distractores (logística 0.73 frente a red 0.55 sobre el mismo tensor), cerradores anti-predictivos fuera de
   muestra (AUC 0.43), búsqueda de estrategia con Sharpe 4.1 en calibración con ruido puro. **C3 debe pasar este
   control (5 semillas, tronco TCN) como puerta antes de cualquier selección**; hoy no lo ha corrido.
4. **Controles negativos**: etiquetas barajadas globalmente −0.29 / −0.01 y entradas de ruido 0.62 / −0.49 (≈ 0, sin
   fuga). El barajado intra-día dio ganancia por diseño del control (etiquetas con horizonte ≈ 1 día casi idénticas en el
   mismo día), no por fuga; se reemplaza por barajado global/semanal.
5. **Corrección del trading** (`evidence/trading_detail/`, @4cc2913): 15/15 chequeos PASAN en todas las unidades,
   semillas y periodos (SL y TP en toda orden y en el lado correcto, tamaño y `rel_volume`, apalancamiento ≤ 1:50,
   unidades enteras, ≤ 1 entrada/día, sin solapes, fill en la apertura siguiente, precio de salida coherente con la razón,
   ventana operativa, cierre antes del forzoso, costos > 0, protección re-derivada barra a barra, neto = bruto − costos);
   cada replay reproduce su corrida exactamente.

### (c) Dónde viven los resultados y cómo re-ejecutarlos

| resultado | ruta (rama `satoshi/expert-guided-rl-20261019` salvo indicación) | commit | re-ejecución |
| --- | --- | --- | --- |
| chequeo del motor (visión perfecta / moneda) | `docs/short_horizon/clean_round/evidence/exec_variants/{ENGINE_CHECK.md,ENGINE_CHECK.json,README.md}` | 36b55e8 | `tools/exec_variants_check.py` (CPU, barras admitidas 2021-01→04) |
| controles de tubería (trampa, ruido, sintético, negativos) | `docs/short_horizon/clean_round/PIPELINE_CONTROLS_2026_10_26.md`, `evidence/pipeline_controls_DIAGNOSTIC/` (rama `satoshi/pipeline-controls-20261026`) | 5b049e8 | `tools/pipeline_controls.py build / run-e3m / oracle / probe-heads / learner-check / ablate-strategy / summarize` (CPU 3G, un fold) |
| detalle de trading y 15 chequeos | `docs/short_horizon/clean_round/evidence/trading_detail/{SUMMARY.md,TRADING_DETAIL_V13.md,TRADING_DETAIL_C2_C2G_13.md,trades/*.csv.gz}` | 4cc2913 | `tools/expert_guided_trading_report.py` sobre los directorios de unidad del paquete |
| auditoría de cierre temprano | `evidence/early_close/EARLY_CLOSE_AUDIT_BASELINE.{md,json}` | 664821d | `tools/expert_guided_early_close_audit.py` |
| registro de defectos (A1, A2, A4, A5, A6) | `ADVERSARIAL_AUDIT_REGISTER_2026_10_26.md` | 3772893 | scripts de los auditores en sus scratch (no versionados; medidas citadas en el registro) |
| C3 (contratos, código) | `contracts/ACTION_CONTRACT_C3.json` (`action_302289e2b317966e`), `ACTION_CONTRACT_C3_TCN.json` (`action_088ee4fdc709f96b`) | 682539f | `tools/expert_guided_e3m.py run --recipe … --action-contract …` |
| tronco TCN con paridad | `app/short_horizon/expert_guided/temporal.py`, `tests/test_expert_guided_residual_tcn.py`, `evidence/residual_tcn/PARAM_COUNTS.json` | 62f6530 | `pytest tests/test_expert_guided_residual_tcn.py` |
| testigos de este diseño | `tools/expert_feedback_witnesses.py`, `tests/test_expert_feedback_witnesses.py` | este commit | `python tools/expert_feedback_witnesses.py` (CPU, segundos) |

Los directorios de unidad con checkpoints, `E3M.json`, `PROBE.json`, `TRIAL_LEDGER.jsonl` están en el paquete del host
del 4090 (`<root>/out/<unidad>/`) y una copia parcial en el coordinador (`<root>/out_from_host/out/`); no se versionan.

### (d) Lo que NO está probado para C3 (nada más que lo que hay en archivo)

- C3 **no ha corrido** ni la trampa, ni el ruido graduado, ni el mercado sintético, ni los negativos; su primera prueba
  con datos reales fue un humo de plomería. Hasta que pase el control sintético con 5 semillas y el tronco TCN, **no**
  hay evidencia de que C3 recupere una señal moderada.
- El acuerdo experto–realidad, el libro por tick y las curvas por tipo (§5) sólo existen en el testigo sintético; en
  producción se emiten con la primera corrida de C3.
- La puerta mejor-del-día, el tamaño por riesgo y el corte de 24 h están probados en el motor con visión perfecta y
  moneda al aire, **no** con una política aprendida: su efecto sobre C3 es una hipótesis medible.
- La selección SBFS con el evaluador real no ha empezado; las 121 columnas de la receta v2 son el candidato, no una
  selección.
- Todo resultado anterior a C3 (línea base WF, V13, C2, C2G) es negativo en los años congelados y, por el control
  sintético, fue medido **por debajo del suelo de la tubería**: no prueba ausencia de señal ni presencia.

## §10. La jerarquía del ZigZag como predictores de corto y largo plazo

Modelo mental del propietario: la estrategia original usaba un **predictor de largo plazo** para la dirección y un
**predictor de corto plazo** para cerrar temprano cuando el SL iba a llegar antes que el TP, tolerante a eventos
imprevistos. La escalera se mapea exactamente a eso.

### 10.1 Mapa

| papel | resoluciones | cantidad por tick (hindsight, sólo entrenamiento; disponibilidad por confirmación) | función en el testigo |
| --- | --- | --- | --- |
| **oráculo de largo plazo** | gruesas: opt, rev20 | dirección del tramo grueso (signo de f_open^opt); **espacio restante** g_coarse(t,d) = max_{r∈gruesas} d·(L_e^r − close_t)/σ_t (en σ, negativo contra el tramo); **tiempo al pivote** τ_coarse = e_opt(t) − t barras | `coarse_oracle` |
| **oráculo de corto plazo** | finas: rev10, rev8, rev5 | **P(pivote fino CONTRA d en ≤ h barras)** = 1[un tramo fino a favor de d termina en ≤ h]; **tiempo al próximo pivote fino** τ_fine = min_r (e_r − t); espacio fino restante; **pivote fino a favor reciente** = 1[un tramo fino a favor de d empezó hace ≤ h] (señal de timing de entrada) | `fine_oracle` (h = 6) |

Lecturas: un pivote fino contra la posición mientras el espacio grueso se debilita es la **alarma "SL antes que TP"**;
un pivote fino a favor al inicio de un tramo grueso es la **señal de timing**.

### 10.2 Cabezas como estimadores causales de esas cantidades

| cabeza | estima | codificación | objetivo (hindsight, disponible por confirmación) |
| --- | --- | --- | --- |
| largo plazo, por lado | g_coarse(t,d) y P(dirección gruesa = d) | regresión acotada + softmax 2 | `coarse_oracle` → en producción `multires.leg_tables["remaining"]` de opt/rev20 |
| corto plazo, por lado | P(pivote fino contra d en ≤ h) y τ_fine | softmax 2 + regresión | `fine_oracle` → en producción tramos de rev10/rev8/rev5 (`leg_end`, `leg_start`) |
| NOP | ninguna resolución ofrece espacio ≥ TP con costo | softmax 2 | NOP-duro/graduado (§1) |
| realidad | Q(t,a) y arrepentimiento | tablas de resultados del motor | §2; el objetivo de la política sigue siendo el arrepentimiento ponderado (§3c) |

Estas cabezas se añaden a las de C3 (§4) como objetivos auxiliares de regresión/clasificación sobre el mismo tronco TCN;
no sustituyen a las cabezas de resultado (TP/SL/TIME, neto por celda), que siguen siendo las que deciden.

### 10.3 Consumo en la estrategia (regla explícita)

- **Abrir** lado d cuando: ĝ_coarse(t,d) ≥ TP_d + costo (el espacio grueso paga el TP) **y** P̂(pivote fino contra d en ≤ h)
  es bajo (timing: no hay contrapivote inminente) **y** el veto NOP está apagado (P̂_NOP ≤ `nop_veto`); orden con el
  TP/SL de la celda elegida, tamaño por riesgo, puerta mejor-del-día (§9a).
- **Cerrar temprano** cuando la **alarma corta** dispara **y** (ĝ_coarse(t,d) < θ **o** P̂(SL primero) > `close_p`);
  mantener en otro caso; SL/TP del motor siempre en vigor.
- **Revertir** sólo cuando la dirección gruesa cambia **con un pivote fino confirmado** (y `allow_reverse`).

### 10.4 Tolerancia a eventos imprevistos: por qué

La alarma corta **no es un pronóstico**: en ejecución se calcula sobre el **camino realizado** —un retroceso confirmado
de `alarm_thr`·σ desde el extremo favorable alcanzado desde la entrada (`causal_alarm`, sólo barras ≤ t), que es
exactamente la confirmación causal de un pivote fino. Un evento imprevisto (salto, cambio de régimen, spread) no
necesita haber sido predicho: en cuanto el camino se mueve contra la posición, la alarma dispara y, si además el
espacio grueso ya no paga o la operación está bajo el agua por `loss_close`·σ, se cierra antes del SL. El SL del
motor sigue siendo la última defensa. La tolerancia es por tanto **reactiva y acotada por la velocidad del evento**:
sólo puede mejorar al SL si el movimiento adverso es más lento que la distancia al SL por barra (medido en 10.5).

### 10.5 Batería de intervenciones (testigos W8–W10, `W8_W10`, bloque de decisión [1800, 3000) del mercado sintético)

Choques inyectados en el **mismo camino** y comparados con contrafactuales de la **misma** estrategia (con/sin cierre
temprano, con/sin alarma, con/sin veto NOP): (i) seis movimientos adversos de 3σ **condicionados a posición** (empiezan
2 barras tras cada una de las primeras seis entradas), rápidos (3 barras, 1σ/barra) y lentos (6 barras, 0.5σ/barra);
(ii) cambio de régimen tendencia → rango (deriva plantada eliminada en [2100, 2500)) → tendencia; (iii) spread ×6 en
[2700, 2800); (iv) 60 ticks sin observación en [2200, 2260) (precios presentes, características ausentes).

Neto en la ventana de choque (σ, suma de los trades que tocan las ventanas), cuatro semillas:

| semilla | choque | alarmas | cierres propios | con cierre temprano | sin cierre temprano | sin alarma (sólo espacio) | ratio pérdida (con/sin) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | rápido 3 barras | 13 | 7 | −0.73 | −2.17 | −0.16 | 0.34 |
| 1 | rápido | 12 | 7 | −11.09 | −11.18 | −11.18 | 0.99 |
| 2 | rápido | 12 | 11 | −3.30 | −3.15 | −3.57 | 1.05 |
| 3 | rápido | 13 | 9 | −6.52 | −6.15 | −6.15 | 1.06 |
| 0 | lento 6 barras | 9 | 6 | +2.39 | +3.83 | +3.34 | n/d (ventana positiva) |
| 1 | lento | 10 | 8 | −1.83 | −4.20 | −4.20 | **0.44** |
| 2 | lento | 13 | 11 | −0.54 | −3.15 | −3.57 | **0.17** |
| 3 | lento | 14 | 9 | −3.20 | −2.17 | −2.17 | 1.47 |

Agrupado sobre las cuatro semillas: choque lento, pérdida en ventana −3.18σ con cierre temprano frente a −5.69σ sin
él (**ratio 0.56**); choque rápido −21.6 frente a −22.65 (ratio 0.95: sin tolerancia posible, el SL de 2σ llega antes
de que el agente pueda actuar en la apertura siguiente). Neto total del bloque de decisión: el cierre temprano **reduce**
el neto en tres de cuatro semillas de choque rápido (5.9 vs 9.5; 2.6 vs 15.2; −3.0 vs −0.6) y lo mejora en una (0.7
vs −0.7): en un mercado con tendencia plantada el cerrador también corta ganadores. Regla de la prueba: la alarma
**debe** disparar bajo los choques en toda semilla (si no, el testigo falla) y el ratio agrupado del choque lento debe ser
< 1; el ratio por semilla se reporta, no se exige.

Cambio de régimen: con el veto NOP causal (fuerza de tendencia 24 barras < 0.5σ-unidades), trades en el rango 4/6/6/3
frente a 10/12/13/6 sin veto (semillas 0–3); neto en el rango +2.38/−4.41/−5.98/+0.60 frente a −0.23/−4.30/−14.54/−0.17.
Spread ×6: neto de la ventana −1.17/0.00/+1.70 frente a −1.05/0.00/+1.95 sin choque (el costo por tick entra en el
motor; menos trades no se fuerzan). Sin observación: **0 entradas** durante el hueco en todas las semillas y las
protecciones (SL/TP/forzoso/cierre propio) siguen corriendo.

### 10.6 Lo que esta batería prueba y lo que no

Prueba: que las definiciones de los dos oráculos son computables y consistentes con los tramos (prueba
`test_hierarchy_oracles_are_consistent_with_the_leg_tables`); que la alarma corta reacciona al camino realizado bajo
eventos no previstos (dispara en 12–14 ocasiones por semilla); que el cierre temprano limita la pérdida de ventana para
choques más lentos que el SL por barra (ratio agrupado 0.56) y no la empeora para choques rápidos (0.95); que el veto
NOP reduce la actividad en rango; que sin observación no se abre nada y las protecciones siguen. No prueba: que las
cabezas aprendidas estimen bien g_coarse / P(contrapivote) (eso lo mide el libro por tick de C3 con datos reales,
§5); ni que el cerrador mejore el neto total (en el juguete lo reduce en tendencia: §8.4 y la auditoría de cierre
temprano); ni la tolerancia a saltos de apertura (gaps) mayores que el SL, que ninguna regla causal puede evitar.
