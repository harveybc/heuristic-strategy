# Diseño: targets multi-resolución con especialistas por timeframe (variante opcional MRT)

Autor: Satoshi, a partir de la propuesta del propietario (2026-10-28). Para revisión de Musashi.
Estado: **diseño, nada implementado ni ejecutado.** Es una variante opcional y configurable. La configuración por defecto de la estrategia actual (C3Z, contrato C3) queda idéntica byte a byte.

Este documento sigue el contrato de arquitectura de `README.md` de heuristic-strategy, sección *Strategy architecture and design method* (capas L1–L6, vistas SEI, método guiado por pruebas, requisitos STR-01 a STR-08). Donde se aparta de ese contrato, lo dice.

## 1. La idea en una frase

Un operador manual mira varios timeframes. Cuando ve un patrón en uno, fija el SL, el TP y el volumen según la escala de ese timeframe y según su seguridad sobre el patrón. La variante MRT hace lo mismo:
- un especialista por resolución del target propone la acción y su incertidumbre;
- los parámetros de la orden salen de un **perfil por resolución**, optimizado fuera de la red;
- un árbitro elige entre resoluciones y direcciones con una métrica común.

La red no estima precios exactos de SL, TP ni volumen.

## 2. Qué cambia frente a C3Z

| Aspecto | C3Z (actual) | MRT (propuesta) |
|---|---|---|
| Resolución del target | Una: celdas TP/SL alrededor de un horizonte corto | Varias: un target por nivel del ZigZag multi-resolución |
| SL / TP | 25 celdas fijas (múltiplos de σ), elegidas por las cabezas de apertura | Perfil por nivel: `d_SL = k_SL[L] · σ_L`, `d_TP = k_TP[L] · σ_L`, coeficientes optimizados por DEAP |
| Volumen | Fijo por el contrato (riesgo 1 %) | Calculado por riesgo hasta el SL y escalado por la confianza (sección 6) |
| Incertidumbre | Canal de sd sólo en el cerrador | Canal de incertidumbre en cada especialista, calibrado |
| Arbitraje | Mejor celda y mejor lado en una sola resolución | Mejor (nivel, dirección) por utilidad neta ajustada por incertidumbre |
| ZigZag | Compuertas de jerarquía (espacio, dirección, contrapivote) | Además define los targets de cada especialista |

## 3. Correspondencia nivel ↔ timeframe

La escalera actual es rev800, rev400, rev200, rev100, rev60, rev20, rev15, experto de profundidad 3, rev10, rev8 y rev5.

- **MRT-0 (medición, antes de nada):** para cada nivel, medir sobre los datos reales de TRAIN la duración mediana del tramo, su amplitud mediana en σ y su dispersión.
- Se eligen 5 niveles cuya duración mediana se acerque a **1, 2, 6, 12 y 24 horas**. Esa correspondencia se mide, no se supone. Si dos niveles caen en el mismo timeframe, se conserva el de tramos más estables.
- `σ_L` es la volatilidad causal en la escala del nivel L: la desviación de los retornos agregados a la duración mediana del tramo de L, sobre una ventana pasada. Nunca usa información futura.

## 4. Targets ideales por nivel (plano de entrenamiento)

Para cada tick t y cada nivel L, el ZigZag en retrospectiva da:

| Target | Definición | Uso |
|---|---|---|
| `dir_L(t)` | BUY, SELL o NOP. Es la dirección del tramo vigente de L; es NOP si el recorrido restante neto de costos no paga el perfil de L | Clase del especialista de apertura |
| `room_L(t)` | Recorrido restante hasta el próximo pivote de L, en σ_L | Referencia del TP del perfil |
| `mae_L(t)` | Excursión adversa máxima antes de ese pivote, en σ_L | Referencia del SL del perfil |
| `tau_L(t)` | Barras hasta ese pivote | Horizonte de la orden |
| `regret_entrada_L(t)` | Neto de entrar en t frente a entrar en el mejor tick del mismo tramo | Calidad de la entrada (análogo al error de salida) |

Reglas del README que se aplican:
- Las etiquetas son retrospectivas. Viven sólo en el entrenamiento y nunca entran como observación.
- Llevan censura por confirmación y purga temporal de al menos la duración máxima de tramo del nivel.

## 5. Especialistas y su salida

- **Arquitectura:** un tronco temporal compartido (residual TCN, regla vigente) y una cabeza por nivel, o bien el nivel como parámetro de entrada de la cabeza, como propuso el propietario. Es una sola red multi-tarea, más barata que 5 modelos separados. La comparación entre ambas formas es un experimento declarado.
- **Salida de cada cabeza de apertura:** probabilidades categóricas {BUY, SELL, NOP} más una **incertidumbre** de esa decisión.
- **Incertidumbre, en orden de preferencia:**
  1. un ensemble pequeño (3–5 semillas): desacuerdo entre miembros;
  2. MC dropout en inferencia, nunca dentro de la recolección de PPO;
  3. una cabeza bayesiana. Es opcional; el propietario ya la usó para predicciones.
  
  La probabilidad softmax **no** es incertidumbre. La incertidumbre debe estar calibrada antes de usarla: diagramas de confiabilidad y ECE por nivel en validación interna.
- **Cierre:** sigue siendo por tick y con información de mercado sola, con el cerrador de trayectorias de primer paso ya construido. Se evalúa dentro del SL y TP del perfil de la orden. Para una orden de nivel L, los niveles más finos informan el fin del tramo, porque marcan sus contrapivotes.

## 6. Perfiles de orden por nivel (optimizables)

Por nivel L, con dirección `s = ±1`:

```text
d_SL = k_SL[L] · σ_L(t)          SL = entrada − s · d_SL
d_TP = k_TP[L] · σ_L(t)          TP = entrada + s · d_TP
unidades = min( presupuesto_margen , riesgo[L] · balance / (d_SL · valor_por_unidad) ) · g(confianza)
presupuesto_margen = q · rel_volume · balance,   rel_volume = 0.1,   q ∈ [0, 1]
```

- Un timeframe mayor tiene un SL más ancho, así que con el mismo riesgo usa **menos** unidades.
- `g(confianza)` reduce el volumen cuando la incertidumbre es alta: monótona, con `g ∈ [g_min, 1]`.
- Genes DEAP por nivel: `k_SL[L]`, `k_TP[L]`, `riesgo[L]`, `q[L]` y los parámetros de `g`, todos con límites declarados en `STRATEGY_PARAMS.json`. Siguen vigentes los mínimos del bróker, el calendario y los límites de exposición.
- **Valores conservadores (orden del propietario):** el óptimo del DEAP es la base, pero no se usa tal cual. Se adopta el punto más conservador (SL más estrecho en riesgo, volumen menor) cuya aptitud sea al menos el 90 % del óptimo en el panel de selección, y se verifica en el panel no visto. El 90 % es un parámetro declarado.

## 7. Arbitraje

En cada tick, primero por acción y luego entre direcciones, como describió el propietario:

1. **Dentro de cada dirección,** entre los niveles L se elige la propuesta con mejor utilidad ajustada por incertidumbre.
2. **Entre BUY y SELL,** se comparan los dos ganadores con la misma métrica. Hay una regla de empate simétrica declarada, sin preferencia por compras.
3. **NOP** gana si ninguna propuesta supera el umbral.

Métrica (variante principal):

```text
U(L, s) = E[neto | L, s] / riesgo_en_dinero_al_SL  −  λ · incertidumbre(L, s)
```

Es una utilidad comparable entre niveles porque está en unidades de riesgo, no una proporción TP/SL. Una proporción TP/SL enorme puede venir con una probabilidad minúscula de alcanzar el TP.

Variantes declaradas que se comparan contra la principal:
- (a) **mínima incertidumbre** entre las propuestas que superan un neto mínimo, la regla literal del propietario;
- (b) **cota inferior** `E[neto] − z · sd` (LCB);
- (c) un único nivel fijo, como control.

`λ` y `z` son genes.

## 8. Requisitos y testigos (se sellan antes de correr)

La matriz de comportamiento vigente se aplica entera. Se agregan estos testigos:

| ID | Requisito | Testigo |
|---|---|---|
| MRT-W1 | El nivel elegido corresponde al patrón presente | En celdas con un solo horizonte activo, ≥ 0.80 de las aperturas usan el nivel cuyo timeframe coincide con el del patrón |
| MRT-W2 | El perfil escala con el timeframe | `d_SL` y `d_TP` mediana crecen con el nivel, y las unidades decrecen con riesgo fijo |
| MRT-W3 | La incertidumbre sirve | El neto por operación es monótono decreciente en la incertidumbre (por quintiles), por nivel y por lado |
| MRT-W4 | El árbitro no favorece un lado | Espejo exacto y canales de dirección intercambiados con las mismas expectativas del panel de calificación |
| MRT-W5 | Superposición de horizontes | En celdas H1–H6 (largo + medio + corto), la estrategia opera más de un nivel y cada nivel se comporta como en su celda pura |
| MRT-W6 | El cierre sigue siendo el principal | Testigos de salida E y D7 sin cambios, por perfil |

El control de validez con oráculo y O* aplica igual que en C3Z: un testigo se corrige sólo donde el oráculo lo falla, con registro de cambios.

## 9. Orden de pruebas (contraste explícito, una cosa a la vez)

1. **MRT-0:** medición de niveles ↔ timeframes en TRAIN.
2. **MRT-1:** las 5 acciones actuales con un perfil configurable en lugar de las 25 celdas, en una sola resolución. Se valida con señales ideales en la matriz. Así se aísla el efecto de cuantizar el riesgo.
3. **MRT-2:** dos niveles (por ejemplo 1 h y 12 h) con árbitro; el cierre sigue revisándose en cada tick.
4. **MRT-3:** se agrega la incertidumbre y se mide su aporte por separado (con y sin `λ`).
5. **MRT-4:** los 5 niveles. Se optimizan los perfiles y el árbitro con DEAP: fitness sólo Sharpe por orden vigente, testigos como restricciones, panel no visto y comparación pareada contra C3Z.
6. **MRT-5:** prueba de aprendizaje real (README paso 6, STR-07). Las salidas de los especialistas se aprenden con el tronco real en una tarea plantada, ambas direcciones, CLOSE y HOLD. Que la estrategia funcione con señales ideales no prueba que la red aprenda esas señales.

Cada paso crea una identidad nueva y se compara contra el anterior con las mismas semillas y el mismo panel.

## 10. Costos y riesgos

- **Entrenamiento:** una red multi-tarea con 5 cabezas cuesta cerca de una vez y media una red de una cabeza, no cinco veces. El ensemble multiplica por su tamaño.
- **Riesgo de sobreajuste en el árbitro:** con 5 niveles × 2 lados × parámetros de perfil, los genes crecen a unos 40. Hacen falta un panel de selección separado y uno no visto, como en el README.
- **Etiquetas de niveles gruesos:** los tramos de 24 h dejan pocas muestras por año. Puede faltar soporte para el nivel más grueso; se mide en MRT-0 antes de comprometerlo.
- **Acoplamiento por reentrada:** ya medimos que la salida óptima de una orden no es la óptima de la estrategia completa. Con varios niveles esto crece. Se mide en la composición, no se asume.

## 11. Preguntas para Musashi

1. ¿Un tronco compartido con cabezas por nivel, o el nivel como parámetro de entrada de una sola cabeza?
2. ¿La métrica principal del árbitro (utilidad en unidades de riesgo menos `λ` · incertidumbre) o la regla literal de mínima incertidumbre?
3. ¿Ensemble o MC dropout como primera fuente de incertidumbre, dado que el carril RL usa PPO?
4. ¿Se acepta la regla del 90 % del óptimo para los valores conservadores?

## 12. Árbitro por diseño de mecanismos (variante, para el problema guiado por ZigZag)

Propuesta del propietario (2026-10-28): tratar a los especialistas como agentes que compiten por un recurso compartido, el balance y el cupo de posición, cada uno con su ganancia contable. Musashi prepara una propuesta propia de teoría de juegos para su carril RL. Esta sección es la versión para **mi** problema, que es distinto: aprendizaje guiado por un experto (ZigZag multi-resolución), con etiquetas retrospectivas por nivel y no recompensas descubiertas por exploración.

### 12.1 El juego

| Elemento | En nuestro sistema |
|---|---|
| Jugadores | Especialistas (nivel L × acción); los cerradores pujan por mantener o liberar el cupo de la orden abierta |
| Recurso | Cupo de posición único, margen `q · rel_volume · balance`, riesgo total acotado |
| Puja | `b = E[neto]/riesgo_al_SL` reportada por el especialista, más su incertidumbre |
| Asignación | El árbitro concede el cupo a la puja ganadora, o a nadie (NOP) si ninguna supera la reserva |
| Pago | Costo de oportunidad: la mejor puja desplazada (segundo precio), incluida la del cerrador de la orden abierta |
| Contabilidad | Libro por agente: neto realizado de sus órdenes menos su pago; es su señal de crédito |

### 12.2 Qué cambia por ser guiado por un experto

- **Los agentes no son estratégicos en el sentido clásico.** Una red entrenada con etiquetas no "miente" para ganar la subasta. Su falla equivalente es la **descalibración**: reportar confianza que no tiene. Por eso la veracidad (VCG o segundo precio) se traduce aquí en **calibración** medible. Penalizamos la descalibración con una regla de puntuación propia: log-loss o Brier sobre el resultado realizado de la puja. Es la versión aprendible de la compatibilidad de incentivos.
- **El ZigZag da el contrafactual exacto en el simulador.** Para cada tick sabemos qué neto habría logrado cada especialista si hubiera ganado. Sin impacto de mercado, `do(acción)` se evalúa por repetición exacta. Con eso:
  - el **valor de Shapley** de cada agente se calcula exactamente sobre coaliciones de niveles, sin aproximar; con 5 niveles son 32 coaliciones por panel;
  - el **arrepentimiento** del árbitro frente al mejor agente en retrospectiva se mide directamente, tick a tick.
- **La etiqueta del experto define una "puja ideal"** por nivel: el neto realizable del tramo de ese nivel desde t. La cabeza de cada especialista se entrena para pujar ese valor. El mecanismo se prueba primero con pujas ideales y con la escalera de calidad, igual que la estrategia actual.
- **Presupuesto por agente con pesos multiplicativos (Hedge):** el peso de cada especialista en la reserva se ajusta por su neto contable pasado, sólo en calibración y validación interna, con arrepentimiento acotado `O(√(T log K))`. Es la protección formal contra "seguir adivinando".

### 12.3 Propiedades que se exigen y se prueban (testigos, se sellan antes de correr)

| ID | Propiedad | Testigo |
|---|---|---|
| MD-1 | Calibración (compatibilidad de incentivos aprendible) | ECE de la puja ≤ umbral por nivel; la utilidad del árbitro no mejora si un agente infla su puja (prueba de inflado sintético) |
| MD-2 | Racionalidad individual | Ningún agente con Shapley ≤ 0 en calibración sigue recibiendo capital en confirmación |
| MD-3 | Arrepentimiento acotado | Neto del árbitro ≥ mejor agente individual en retrospectiva − cota declarada, en el panel no visto |
| MD-4 | Simetría | Espejo y canales intercambiados: mismas expectativas del panel de calificación |
| MD-5 | No destruye el cierre | Testigos E y D7: la puja del cerrador por mantener compite con las aperturas sin forzar cierres destructivos |
| MD-6 | Reserva útil | NOP gana cuando la mejor puja no paga costos; tasa de actividad dentro del contrato de negocio |

### 12.4 Orden

El mecanismo entra como variante del árbitro en MRT-2 y siguientes. Se compara en el mismo panel contra la utilidad principal, la mínima incertidumbre y el nivel fijo. No se adopta sin pasar MD-1 a MD-6 y el panel no visto.

### 12.5 Criterios con que auditaré la propuesta de Musashi

1. **Supuestos del modelo de juego.** ¿Los agentes son estratégicos o aprendidos? Si son aprendidos, ¿cómo se traduce la veracidad en algo medible?
2. **Propiedades declaradas.** Compatibilidad de incentivos, racionalidad individual, balance presupuestario, eficiencia. ¿Cuáles se demuestran y cuáles se suponen?
3. **Restricciones reales.** Cupo único de posición, SL/TP en el bróker, calendario, costos y financiación. ¿El mecanismo las respeta o las simplifica?
4. **Asignación de crédito.** ¿La recompensa por agente es contable de verdad? ¿Qué pasa con la reentrada, que acopla las decisiones?
5. **Cálculo.** Shapley exacto o aproximado, su costo y su varianza.
6. **Garantías de arrepentimiento.** ¿Bajo qué supuestos (adversarial, bandido, estacionario) valen? ¿Valen con el cupo único?
7. **Evidencia.** Controles pareados, semillas, panel no visto, controles negativos y testigos sellados antes de correr.
8. **Separación de carriles.** Qué es común a RL puro y a guiado por experto, y qué no.
