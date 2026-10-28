# Propuesta: diseño de la estrategia por mecanismos para el carril guiado por ZigZag multi-resolución

Fecha: 2026-10-28 (America/Bogota). Autor: Satoshi. Revisor solicitado: Musashi.
Estado: **PROPUESTA DE DISEÑO PARA AUDITORÍA INDEPENDIENTE.** Las partes marcadas MEDIDO existen y tienen evidencia; las partes marcadas PROPUESTO no están implementadas. Ningún resultado de este documento es una aptitud ni una ganancia.

Este documento es la contraparte, para mi carril, de la propuesta 112 de Musashi (agent-multi @4a6871b3). Sigue el contrato de arquitectura del `README.md` de heuristic-strategy (capas L1–L6, tres vistas, método guiado por pruebas, STR-01…08) y recoge mis propios hallazgos de la auditoría a la 112 (agent-multi `satoshi/audit-musashi-mechanism-synthesis-20261028` @3326f898). Donde el diseño se aparta del de Musashi, se declara una **identidad de juego distinta**, con su propia solución y auditoría, como pide su §11.

## 0. En qué se diferencia este carril

| | Carril RL puro (Musashi) | Carril guiado por experto (este) |
|---|---|---|
| Señal de aprendizaje | Recompensa descubierta por interacción | Retroalimentación de un experto retrospectivo: el ZigZag multi-resolución, que define el óptimo por nivel y la pérdida por tick frente a ese óptimo |
| Semántica de los informes | Supuesta por el productor | **Definida exactamente por la etiqueta**: cada informe tiene una verdad retrospectiva calculable por repetición en el motor |
| Cerradores | Informe escalar por tick | Curvas de primer paso de mercado, evaluadas por el controlador en la posición; filtro sobre el historial (MEDIDO) |
| Dónde vive el experto | No hay | Sólo en el plano de entrenamiento y de evaluación sintética; nunca en una observación |
| Qué se optimiza primero | El mecanismo con reportadores congelados | Igual, pero con **pujas ideales** derivadas del experto y una escalera de calidad |

Todo lo demás (motor, contrato de negocio, repetición contrafactual, aritmética de Pareto y ε, matriz de escenarios) se comparte, con la tabla del §13.

## 1. Lo que ya existe y está medido (punto de partida)

Rama `satoshi/expert-guided-rl-20261019`. Nada de esto es el mecanismo propuesto; es la base que el mecanismo debe superar o reutilizar.

- **Estrategia C3Z** con compuertas de jerarquía escritas a mano sobre la escalera de ZigZag (rev800 … rev5 y experto de profundidad 3). Contrato C3 re-sellado con el cerrador de curvas (`action_b75e17689bc3cff5`).
- **Matriz de comportamiento** (MEDIDO): 869 celdas base × 3 semillas + gemelos sin costo y espejo exacto (4 345 registros); bloques de tendencia, transiciones a varias intensidades, volatilidad (paso, GARCH, rampa), estructura fina, huecos, spread, calendario y superposición de horizontes. Oráculo de verdad y O* (cierre óptimo) como controles de validez de cada testigo, con estados DEFECT / VALIDATED / SEALED_UNVALIDATED / REPLAY_CONFLICT y registro de cambios sellado.
- **Capa de salida** (MEDIDO): episodios con posición abierta en ambas direcciones (≥ 30 destinados al SL por lado y celda), P(t), P*, regret, A(t), L_avail(t), O*, cabeza de curvas de primer paso sólo de mercado (exacta a precisión de máquina con cabezas perfectas), cerrador con filtro de Kalman sobre su historial, escalera de calidad del cerrador 1.00/0.85/0.70 × φ ∈ {0, 0.587, 0.98}, con el cerrador bayesiano de información limitada como control (D7).
- **Resultados que condicionan el diseño** (MEDIDO):
  - con cabezas perfectas el cierre temprano es la salida principal (56–87 % de las salidas) y el error de salida < 0.001σ;
  - por tick y con ruido realista, sin filtro: 73–96 % de cierres destructivos; con filtro: 4–12 %, regret igual al control bayesiano a φ = 0.587;
  - ZigZag perfecto a calidad 0.70 inflaba el Sharpe (8.68 frente a 7.60 con escalera de calidad y 8.26 con ZigZag causal confirmado);
  - Sharpe solo oculta el colapso del neto (−24 % frente a −84 %) y premia la rotación por reentrada (cerrador 0.85 superó al perfecto en 23/24 celdas);
  - el DEAP sólo Sharpe con restricciones de testigos (corrida 4b, 87 mundos) está en ejecución; su punto se califica en la matriz completa y en semillas no vistas.

## 2. Objetivo de la propuesta

Reemplazar las compuertas de jerarquía escritas a mano (C3Z: espacio de tendencia, contrapivote, post-giro, umbral fino) por un **mecanismo de asignación** entre especialistas por nivel del ZigZag, con propiedades verificables, y comparar ambos en el mismo panel. Lo que debe demostrarse no es "funciona con señales ideales" sino: (i) el mecanismo alcanza el oráculo con pujas ideales dentro de una tolerancia declarada; (ii) degrada de forma monótona con la calidad; (iii) su punto optimizado pasa la matriz y la confirmación ciega; (iv) sus pujas son aprendibles por la red real (control positivo).

## 3. Jugadores, informes y recursos

### 3.1 Niveles y jugadores

- **Niveles:** 5 niveles de la escalera elegidos por medición en TRAIN (G0): los de duración mediana de tramo más cercana a 1, 2, 6, 12 y 24 horas. `σ_L` = volatilidad causal a la escala del tramo de L.
- **Jugadores (20 lógicos):** `{AB, AS, CB, CS} × {L1..L5}`. AB/AS proponen abrir; CB/CS proponen cerrar la posición de su lado. NOP (plano) y HOLD (posicionado) son alternativas de referencia con puntaje 0, no redes.
- **Elegibilidad por legalidad** (estado del motor, no de la red): plano ⇒ 10 abridores + NOP; largo ⇒ 5 CB + HOLD; corto ⇒ 5 CS + HOLD.

### 3.2 Informes (contrato de señal, versión guiada)

| Jugador | Campos | Verdad retrospectiva que lo define (plano de entrenamiento) |
|---|---|---|
| AB_L / AS_L | `b` puja en unidades de riesgo; `p_act`; `u` incertidumbre calibrada | `b* = neto realizable del tramo de L desde t con el perfil de L / riesgo en dinero al SL`, por repetición exacta en el motor; `p_act* = 1[b* > 0]` |
| CB_L / CS_L | **curvas** de primer paso de mercado hasta el horizonte H_L: excursión favorable y adversa desde ahora, mejor cierre hasta cada barra, costos y financiación por barra; `ŝ` desviación predictiva | Las curvas verdaderas del motor (MEDIDO: exactas a precisión de máquina con cabezas perfectas) |
| todos | `available_at ≤ t`, `valid`, identidad del productor, versión de la definición de incertidumbre | — |

Diferencia esencial con la 112: aquí "0.9 de `p_act`" **sí** tiene una semántica exacta, porque la etiqueta la define. La calibración se mide contra esa verdad (§8, MD-1).

### 3.3 Recursos

Un cupo de posición; margen `q·rel_volume·B` con `rel_volume = 0.10`; presupuesto de pérdida planificada; cuota de entradas por día; tiempo. El margen se libera al salir; la cuota no. La ganancia se acredita una vez al libro común. **Constantes de negocio (decididas por el propietario el 2026-10-28):** capital 10 000; `rel_volume` 0.10; **hasta 4 entradas por día** de negocio (el PDT ya no aplica); **sin entradas en las primeras 12 h** tras la reapertura semanal; **cierre forzado 12 h antes** del cierre semanal; **sin entradas en las 24 h anteriores al cierre forzado** (aceptado: evita abrir órdenes que el cierre forzado cortaría); apalancamiento 100 en este carril. Difieren de la 112 (apalancamiento 50, 1 entrada por día): el propietario fija 4 entradas por día para ambos carriles; el apalancamiento común queda por fijar. Estas constantes van a un `BUSINESS_CONTRACT.json` único (hallazgo F-11 de mi auditoría), no son genes.

## 4. Perfiles de orden por nivel (ningún precio lo estima la red)

Con dirección `s = ±1` y `V_L(t)` la volatilidad causal a la escala de L en unidades de precio:

```text
d_SL = a[s,L] · V_L(t)          SL = entrada_ejecutable − s · d_SL
d_TP = b[s,L] · V_L(t)          TP = entrada_ejecutable + s · d_TP
presupuesto_pérdida = ρ[L] · B · g(u)         g decreciente, g(0) = 1, g ≥ g_min
unidades = min( ⌊presupuesto_pérdida / (d_SL · valor_por_unidad)⌋ ,
                ⌊rel_volume · B · apalancamiento / precio⌋ )       entero, ≥ mínimo del bróker o rechazo
q = margen_requerido / (rel_volume · B) ∈ [0, 1]
```

- Un nivel grueso tiene SL más ancho y, con el mismo riesgo, **menos** unidades.
- `a, b, ρ` por nivel (15 coeficientes simétricos; 30 si se separan direcciones, con control espejo), `g_min` y la pendiente de `g` son genes con cotas selladas en `STRATEGY_PARAMS.json`.
- **Regla conservadora (propietario):** el óptimo de la búsqueda es la base; se adopta el punto más conservador cuya aptitud en el panel de selección sea ≥ 90 % del óptimo, y se confirma en el panel no visto. El 90 % es un parámetro declarado.
- SL y TP no se mueven después de abrir en la primera versión; el cierre temprano es la salida discrecional.

## 5. Mecanismo M(θ)

### 5.1 Puntaje y asignación

- **Abridores (plano):** `score_i = ŵ_i · b̂_i − λ_open · u_i`; gana el máximo si `score > r_open` (reserva). Empate: regla simétrica declarada (clave estable, acoplada bajo permutación de dirección). Se compara **primero dentro de cada dirección y luego entre direcciones**, como pidió el propietario; el resultado es el mismo que el argmax global, pero el registro muestra ambas comparaciones.
- **Cerradores (posicionado):** cada CB_L/CS_L evalúa sus curvas en las distancias restantes de la posición y produce `Â_L(t)` y `ŝ_L(t)`; el controlador filtra el historial por posición (Kalman con parámetros estimados en calibración; MEDIDO). `score_L = Â_L − z·ŝ_L − c[L_abierto, L]`; CLOSE gana si `max_L score_L ≥ −margen` durante `k` ticks ejecutables consecutivos. HOLD es la referencia. El SL/TP del bróker siguen siendo el respaldo.
- **Matriz entre escalas** `c[L_abierto, L]` (5×5, 0 como control): una reversión falsa de escala corta no fuerza el cierre de una posición larga; el cerrador de la escala de la posición no veta a los demás (hallazgo F-08).
- **Legalidad y protección** mandan sobre el puntaje. Una decisión no cierra y reabre; la reversión es una variante declarada (`allow_reverse`, gen).

### 5.2 Pesos Hedge (PROPUESTO, ablación)

`ŵ_i ∝ exp(η · crédito acumulado_i)` normalizado por rol, actualizado sólo en el bloque de calibración y congelado antes de la selección y de TEST. Garantía: arrepentimiento del árbitro frente al mejor especialista en retrospectiva acotado por `O(√(T·log K))` **bajo el supuesto de secuencia de pérdidas fija**; con el cupo único las acciones cambian la secuencia, así que la cota se declara como heurística y se mide (MD-3), no se afirma.

### 5.3 Crédito y pago

No hay transferencia de dinero: el "pago de segundo precio" es contable. Para cada decisión ganada por i:

```text
crédito_i = Φ(π) − Φ(π_i^ref, π_-i)                    (difference reward, repetición exacta; es la U_i del juego)
shapley_i (por grupo de rol, 5 miembros ⇒ 32 coaliciones, exacto por repetición)   (señal de aprendizaje)
```

La *difference reward* se anula con redundancia (hallazgo F-06), por eso la señal de aprendizaje es Shapley por grupo, y la utilidad del juego es `U_i`.

## 6. Juego, utilidad y propiedades

- **Biblioteca de políticas por jugador** (finita, sellada en `GAME_SPEC`): abridores `{umbral de puja, techo de incertidumbre, perfil permitido}`; cerradores `{z, k, filtro on/off, sensibilidad entre escalas}`; el miembro de referencia siempre se abstiene.
- **Utilidad:** `U_i(θ,π) = Φ_w(θ,π) − Φ_w(θ, π_i^ref, π_-i)`, con la referencia repetida con las demás políticas fijas (no acciones fijas). Es un **potencial exacto** y, con el manifiesto finito de escenarios y ruido indexado por evento, **exacto por escenario** (hallazgos F-01, F-04). Se declara como lo que es: un problema de equipo cuyos equilibrios son óptimos locales de Φ coordenada a coordenada (F-02).
- **Certificado:** `ε_exhaustivo` sobre desviaciones unilaterales de la biblioteca declarada (costo lineal, `Σ|Π_i|` repeticiones por escenario) sobre fixtures exactos y una submuestra estratificada sellada del manifiesto; `ε` global sólo para el perfil final (F-03, F-19).
- **Φ_w:** frontera de cuenta (neto, Sharpe diario con convención fija, −MDD). El propietario ordenó Sharpe como fitness; propongo mantener Sharpe como objetivo escalar **con una restricción de neto mínimo** relativa al punto sin optimizar y reportar siempre neto, operaciones y tenencia (F-09). Es una decisión del propietario.
- **Selección entre equilibrios:** se reportan todos los alcanzados por arranques múltiples; la selección robusta por peor equilibrio se justifica sólo porque el aprendiz de G4 hace mejora coordenada (entrenamiento por cabeza); si no, manda el máximo de Φ (F-02).

## 7. Plano de entrenamiento: etiquetas y retroalimentación (RLHF con experto)

Aclaración de nombre: "RLHF" aquí es aprendizaje con retroalimentación de un **experto retrospectivo**, no humana. Ninguna etiqueta entra como observación.

| Etiqueta por tick t y nivel L | Definición | Soporte / censura |
|---|---|---|
| `dir_L, room_L, mae_L, tau_L` | Tramo vigente de L: dirección, recorrido restante, excursión adversa máxima, barras al próximo pivote | Disponible sólo cuando el pivote siguiente está confirmado; purga temporal ≥ duración máxima de tramo de L |
| `b*_L(t)`, `p_act*_L(t)` | Puja ideal y su signo (§3.2) | Idem |
| `regret_entrada_L(t)` | Neto de entrar en t frente al mejor tick del mismo tramo | Idem |
| `curvas*(t)` | Curvas de primer paso verdaderas hasta H_L | Censura al corte forzado y a huecos no observables |
| `A*(t)`, `L_avail(t) = max(0, −A*)` | Ventaja de cerrar ahora frente a la mejor continuación ejecutable; pérdida por tick frente a la mejor salida aún disponible antes del SL | Idem; nunca usa recuperación tras el SL ni extremos pasados |

- **Objetivo de las cabezas:** media condicional (contracción `ρ²`) de cada etiqueta más una desviación predictiva; incertidumbre por ensemble pequeño (3–5 semillas) o MC dropout fuera de la recolección; calibración medida con ECE y Brier contra la verdad.
- **Retroalimentación del experto a la política:** regresión ponderada por ventaja con `−L_avail` (salidas) y `−regret_entrada` (entradas); el mecanismo decide, la red sólo informa. Esta parte es G4 y **no se ejecuta** antes de que el mecanismo pase G2–G3.
- **Aliasing:** si entradas observables idénticas exigen acciones distintas (dos posiciones, mismo mercado), la red no puede resolverlo; por eso los cerradores entregan curvas y el controlador evalúa la posición (F-05). Se mide la tasa de aliasing residual por nivel.

## 8. Requisitos, testigos y propiedades (se sellan antes de correr)

Se mantiene **toda** la matriz vigente (W0–W17, E0–E8, EP, Z1, D7) con sus controles de validez. Se agregan:

| ID | Propiedad | Testigo (umbrales con pujas ideales; con calidad < 1, relativo al control) |
|---|---|---|
| G-1 | El mecanismo alcanza al oráculo | Φ(mecanismo, pujas ideales) ≥ Φ(oráculo de verdad) − δ por familia; δ declarado |
| G-2 | Nivel correcto | En celdas de un solo horizonte, ≥ 0.80 de las aperturas usan el nivel cuyo timeframe coincide |
| G-3 | Perfil escala con el timeframe | `d_SL, d_TP` medianos crecen con L; unidades decrecen con riesgo fijo |
| G-4 | Superposición | En celdas largo+medio+corto el mecanismo opera más de un nivel y cada uno se comporta como en su celda pura |
| MD-1 | Calibración = veracidad aprendible | ECE por jugador ≤ umbral; inflar la puja de un jugador no mejora Φ (prueba de inflado sintético) |
| MD-2 | Racionalidad individual | Jugador con Shapley ≤ 0 en calibración no recibe capital en confirmación |
| MD-3 | Arrepentimiento acotado | Φ(mecanismo) ≥ mejor especialista individual en retrospectiva − cota declarada, en el panel no visto |
| MD-4 | Simetría | Espejo y canales intercambiados con expectativa **acotada por riesgo** (no pérdida espejo; F-14) |
| MD-5 | Cierre intacto | E y D7 por perfil; la puja de HOLD no fuerza cierres destructivos; entre escalas: reversión falsa corta no cierra, verdadera sí |
| MD-6 | Reserva útil | NOP gana cuando ninguna puja paga costos; actividad dentro del contrato |
| MD-7 | Dos posiciones, mismo mercado | Con informes ideales, cierra una y mantiene la otra (fixture bloqueante de mi auditoría) |
| MD-8 | Redundancia | 5 cerradores idénticos: `U_i = 0`, Shapley reparte; el mecanismo no cambia Φ |
| MD-9 | Identidad de potencial | Igualdad `ΔU_i = ΔΦ` por escenario con 3 jugadores y referencia que cambia el estado |
| MD-10 | No se adapta la prueba | Alterar resultados de confirmación y rehacer la preparación: parámetros e identidades sin cambio |

Controles obligatorios en cada comparación: oráculo de verdad, O*, reloj fijo seleccionado en TRAIN, salidas permutadas, canales intercambiados, salidas retrasadas 1/3/6 barras, ZigZag causal confirmado, C3Z (las compuertas a mano) y nivel único fijo.

## 9. Escenarios

La matriz existente más: posiciones sembradas a varias edades (0, 6, 24, 72 barras) y estados de P&L; reversión y luego recuperación (HOLD óptimo); volatilidad decreciente; intensidad de transiciones; spread ×5; huecos más allá del SL con y contra; corte y reapertura. Tres semillas de desarrollo (0–2) y tres de confirmación no vista (3–5) por celda base; gemelos sin costo y espejo. Las expectativas se escriben en `QUALIFICATION_SPEC.json` **antes** de correr (ya sellado para D3–D6; se extiende para G y MD).

## 10. Vistas (SEI)

- **Componente–conector (ejecución):** informes (reales o ideales) → validación de contrato → árbitro M(θ) con estado del motor → perfiles y tamaño → motor → libro → (fuera de línea) evaluador contrafactual → juego y búsqueda → θ. En línea nunca se consulta una rama futura.
- **Módulos:** contratos, árbitro, perfiles y tamaño en heuristic-strategy (`app/short_horizon/expert_guided/`); etiquetas ZigZag y cabezas en predictor/heuristic-strategy según el productor; motor pinned; herramientas de juego compartidas con agent-multi vía módulo común (no un segundo registro de plugins).
- **Asignación:** juegos exactos, oráculos y contrafactuales en CPU bajo admisión medida por mundo (hallazgo F-18); cabezas reales en GPU del carril, por orden del propietario.

## 11. Etapas, automatización y evidencia de salida

| Etapa | Contenido | Evidencia de salida |
|---|---|---|
| G0 | Medición nivel ↔ horizonte en TRAIN; `σ_L`; soporte de etiquetas por nivel | Tabla por nivel; niveles sin soporte excluidos con razón |
| G1 | `GAME_SPEC`, `EXPERT_CONTRACT` v2 (curvas), `POLICY_LIBRARY`, fixtures exactos: MD-7, MD-8, MD-9, AA/BB con tercera opción, libro espejo, "toda propuesta se ejecuta o se cuenta" | Soluciones conocidas reproducidas; equilibrios dominados detectados |
| G2 | Mecanismo con pujas ideales sobre la matriz; comparación pareada contra C3Z y el oráculo | G-1…G-4, MD-4…MD-7; testigos vigentes al 100 % efectivo con regla por celda |
| G3 | Juego: ascenso por coordenadas con arranques múltiples, `ε_exhaustivo` en fixtures y submuestra sellada; búsqueda externa de θ (DEAP/NSGA-II) con frontera de cuenta; escalera de calidad × φ; confirmación ciega en semillas 3–5 | Archivo de equilibrios con cobertura; Pareto con testigos de dominancia; punto conservador (90 %) adoptado |
| G4 | Control positivo aprendido (STR-07): cabezas reales sobre tarea plantada, ambas direcciones, CLOSE y HOLD, réplica con pesos guardados; calibración ECE | Aprendibilidad demostrada o hallazgo de aliasing |
| G5 | Datos reales: controles C3, pre-filtro causal, SBFS con el modelo real, entrenamiento final | Sólo tras auditoría de G0–G4; TEST cerrado |

Automatización: cadena secuencial con receipts duraderos, reanudación por identidad, fallos terminales explícitos, el resumen lee los archivos de veredicto, vigilantes **sin señales** (banderas STOP, `JOBS_OVERRIDE`; cualquier señal la ejecuta el propietario), memoria medida por mundo antes de admitir. Artefactos mínimos: `GAME_SPEC.json`, `EXPERT_CONTRACT.json`, `POLICY_LIBRARY.json`, `SCENARIO_MANIFEST.json`, `QUALIFICATION_SPEC.json`, config efectiva, tensores de pago, `COUNTERFACTUALS`, `EQUILIBRIA` con cobertura, `PARETO`, métricas por jugador, tiempos, informe.

## 12. Qué falsaría el diseño aun con pujas ideales

1. Que el mecanismo con pujas ideales quede por debajo de C3Z o del oráculo más allá de δ en alguna familia (G-1): entonces las compuertas a mano codifican algo que el mecanismo no captura y hay que decir qué.
2. Que falle MD-7 (dos posiciones): el contrato de curvas sería insuficiente.
3. Que el perfil que maximiza Φ tenga `ε_exhaustivo > 0`: error de implementación del potencial.
4. Asimetría espejo o entre permutaciones de jugadores.
5. Que la confirmación ciega no reproduzca el orden de los mecanismos del panel de búsqueda.
6. Que la escalera de calidad no sea monótona o que el valor del cierre temprano frente a esperar al SL se vuelva ≤ 0 a calidad 0.70.
7. Que la tasa de aliasing residual por nivel supere el umbral declarado: la tarea no sería aprendible aunque el mecanismo sea correcto.

## 13. Reutilización entre carriles

| Componente | Clase |
|---|---|
| Motor pinned, contrato de negocio único (pendiente del propietario), repetición contrafactual indexada por evento, aritmética de Pareto y ε, formato de informe y auditoría | SHARED_UNCHANGED |
| Contrato de señal v1 `[B,20,2]`; v2 con curvas para cerradores | SHARED_UNCHANGED (v2 propuesta a ambos) |
| Biblioteca de participación por rol, prueba de inflado, fixtures MD-7/8/9, matriz entre escalas | SHARED_UNCHANGED |
| PPO, recompensa descubierta, 20 expertos sin etiquetas | RL_ONLY |
| Etiquetas ZigZag por nivel, pujas ideales, escalera de calidad de cabezas, curvas verdaderas, Kalman del cerrador, regresión ponderada por regret | EXPERT_GUIDED_ONLY |
| Constantes de negocio, suficiencia de cerradores, calibración por productor, nivel ↔ horizonte, umbral δ de G-1 | NEEDS_SEPARATE_TEST |

## 14. Decisiones solicitadas

- **Al propietario:** contrato de negocio único; Sharpe solo o Sharpe con restricción de neto; regla conservadora del 90 %.
- **A Musashi (auditor):** las del §5 de la solicitud de auditoría adjunta, en particular: contrato de curvas frente a biblioteca condicionada a la posición; Shapley por grupo como señal de aprendizaje; validez de la cota Hedge con cupo único; definición de δ en G-1; adecuación del modelo de ruido AR(1) con φ medido en H6; legitimidad de las formas corregidas de testigos admitidas por el oráculo (W4 v3′, W8 v3, W13 v3, W7 v2) frente a la prohibición de adaptar la prueba a la estrategia.

## 15. Fuentes

- Wolpert y Tumer, *Optimal payoff functions for members of collectives* (difference rewards).
- Monderer y Shapley, *Potential games*.
- Freund y Schapire, *A decision-theoretic generalization of on-line learning* (Hedge).
- Shapley, *A value for n-person games*.
- `README.md` de heuristic-strategy, *Strategy architecture and design method*; propuesta 112 y mi veredicto @3326f898.
