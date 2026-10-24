# ZigZag multi-resolución como fuente de ejemplos para los cinco agentes (para discusión con el propietario, 2026-10-24)

Pregunta del propietario: el ZigZag óptimo (Depth 3) deja muy pocos ejemplos para abrir-compra, abrir-venta,
cerrar-compra, cerrar-venta y, sobre todo, para NOP (el más importante: que el agente no tome riesgos estúpidos). Con
varias resoluciones decrecientes por debajo del óptimo tendríamos ejemplos variados, de distinto rendimiento, para
todos los modelos. ¿Cómo las usamos y con qué métricas entrenamos? Nada de esto entra en la observación: son
etiquetas, pesos y objetivos del lado del entrenamiento. Medidas en `evidence/multires/MULTIRES_PROFILE.{json,md}`
(`tools/expert_guided_multires_profile.py`; TRAIN 2020-01 → 2023-07, contrato 1/día–1:50, costo declarado 0.06σ).

## 1. Qué da cada resolución (medido, TRAIN)

La escalera natural es el ZigZag de reversión por umbral en pips: es monótona y continua (el Depth de MetaTrader
sólo da saltos: Depth 2 ≈ rev12). Orígenes elegibles (puerta de entrada abierta, soporte final): **17 128**.

| resolución | tramos/sem | tramo mediano (pips) | ganancia neta mediana del tramo (σ) | tramos ≥ 0.25σ (TP mínimo) | abrir L / S | "ninguna" | igual al óptimo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev40 | 8.4 | 72 | 1.24 | 100 % | 587 / 600 | 15 941 | 82 % |
| rev30 | 13.7 | 54 | 0.95 | 100 % | 971 / 974 | 15 183 | 83 % |
| rev20 | 23.8 | 39 | 0.69 | 98 % | 1 690 / 1 691 | 13 747 | 84 % |
| **3 (óptimo)** | 27.1 | 30 | 0.55 | 85 % | 1 935 / 1 923 | 13 270 | 100 % |
| rev15 | 32.0 | 33 | 0.57 | 93 % | 2 263 / 2 266 | 12 599 | 83 % |
| 2 (MetaTrader) | 38.1 | 24 | 0.43 | 74 % | 2 732 / 2 726 | 11 670 | 84 % |
| rev12 | 38.1 | 29 | 0.50 | 87 % | 2 700 / 2 702 | 11 726 | 81 % |
| rev10 | 42.7 | 26 | 0.46 | 82 % | 3 022 / 3 021 | 11 085 | 80 % |
| rev8 | 47.7 | 24 | 0.42 | 76 % | 3 390 / 3 389 | 10 349 | 77 % |
| rev6 | 52.0 | 22 | 0.39 | 71 % | 3 716 / 3 713 | 9 699 | 75 % |
| rev5 | 53.7 | 21 | 0.37 | 69 % | 3 847 / 3 844 | 9 437 | 74 % |

Lecturas:

- **Más resolución = más ejemplos de apertura pero peores tramos.** De 1 935 largos en el óptimo a 3 847 en rev5
  (el doble), con la ganancia neta mediana del tramo cayendo de 0.55σ a 0.37σ y la fracción de tramos que al menos
  pagan el TP mínimo de 85 % a 69 %. Ningún tramo mediano baja del costo en esta escalera; el **suelo razonable** es
  rev5–rev6 (por debajo, el tramo se acerca al spread; rev4/rev3 se pueden medir si quieres).
- **NOP mejora mucho con la escalera.** El óptimo etiqueta "ninguna" 13 270 orígenes (77 %), pero 4 901 de ellos
  son **trades intermedios** que una resolución más fina sí encuentra. Con las 12 resoluciones, los orígenes donde
  **ninguna** encuentra un trade son 8 369 (49 %): un NOP más honesto, "aquí no hay nada que ganar a ninguna escala",
  que es justo el que evita el riesgo estúpido. Conflictos largo-y-corto entre resoluciones: 113 (0.7 %).
- **Las resoluciones gruesas (rev20–rev40) dan pocos ejemplos pero de alta calidad** (≥ 0.69σ mediana, 98–100 %
  pagan el TP): sirven como ejemplos de **alta confianza**; rev40 cumple la regla de ≥ 2 órdenes/semana (2.9) aunque
  no la antigua de 3.
- **Los cerradores no reciben ejemplos de CERRAR de ninguna resolución por sí sola**: la trayectoria del propio
  demostrador termina en TP (hindsight), así que casi todas sus filas posicionadas son MANTENER. Los ejemplos de
  CERRAR vienen de (a) las posiciones de exploración (ronda 0, como hoy) y, con la escalera, de (b) **cruzar
  resoluciones**: una posición abierta por una resolución fina llega a su pivote cuando una gruesa todavía dice
  mantener; eso da ejemplos graduados de "cierra ahora / aguanta" que un solo ZigZag no tiene.
- **Imitabilidad causal**: la contraparte que actúa sólo al confirmarse el pivote pierde en todas las resoluciones
  (acierto 27–38 %). Las etiquetas son objetivos de entrenamiento, no una estrategia ejecutable.

## 2. Propuesta de métricas y etiquetas por agente (para discutir; nada está decidido)

Sea R = {r_1 (óptimo) … r_K (la más fina)} la escalera admitida, y para cada origen o y resolución r: la etiqueta
y_r(o) ∈ {L, S, ninguna} y la **calidad** q_r(o) = ganancia neta en σ del tramo que empieza en o a esa resolución.

| agente | ejemplos | etiqueta | métricas/pesos propuestos |
| --- | --- | --- | --- |
| abrir-compra / abrir-venta | orígenes con y_r = L (o S) en **alguna** r admitida (4 459 L / 4 413 S frente a 1 935 / 1 923 sólo con el óptimo) | 1 | **votos** n_L(o) = número de resoluciones que dicen L (0–K) como confianza; **calidad** máx_r q_r(o) como peso de la muestra o como objetivo de la cabeza de regresión (la "oportunidad" ya existente, pero medida a todas las escalas); descartar como ejemplo positivo los orígenes con un solo voto y q < 0.5σ ("trade de baja calidad"), o pasarlos a NOP graduado |
| NOP | orígenes con n_L = n_S = 0 en **todas** las resoluciones (8 369) | 1 | peso 1 cuando ninguna resolución encuentra trade; peso menor (p. ej. 0.5) cuando sólo la escala más fina encuentra uno de q < 0.5σ; **nunca** NOP donde una resolución gruesa encuentra un trade de alta calidad |
| cerrar-compra / cerrar-venta | filas posicionadas (demostraciones de cualquier r + exploración) | CERRAR si la ganancia **restante** hasta el próximo pivote favorable de la resolución más gruesa que aún paga ≥ θ_close es < θ_close, o la adversa alcanza el SL; MANTENER en otro caso | **votos de cierre** = fracción de resoluciones cuya ganancia restante < θ_close; la ganancia restante por resolución (vector) como objetivos auxiliares; esto generaliza `hold_gain` (hoy sólo con el horizonte del óptimo) |
| todos | — | — | **cobertura** (ejemplos por clase por resolución y en la unión), **balance** NOP / abrir, **acuerdo** entre resoluciones (84 % → 74 % al afinar), **calidad** (percentiles de q), **imitabilidad** causal por resolución (reportada, no decide) |

Dos formas de consumirlo en E3-M, ambas sólo en el entrenamiento:

1. **Consenso por filas** (ya implementado: `--extra-experts`): cada fila plana recibe una etiqueta por cada
   resolución válida en ella; más resoluciones = más filas con la misma observación y etiquetas que pueden
   diferir; el modelo aprende la mezcla. Barato y ya probado en T3; no distingue calidad.
2. **Etiqueta única ponderada** (propuesta nueva): una fila por origen con la etiqueta de consenso (L si n_L > n_S y
   q máx ≥ umbral; S simétrico; NOP si nadie), peso = f(votos, calidad); cerradores por la regla de la
   resolución más gruesa que aún paga. Requiere escribir `MultiResLabels` (misma disciplina de disponibilidad por
   confirmación del pivote que `SignalLabels`) y sus pruebas.

## 3. Lo que te pido decidir, Maestro

1. **Escalera**: ¿{óptimo, rev20, rev15, rev10, rev8, rev5} (seis escalas, suelo rev5) o quieres medir rev4/rev3?
2. **Pesos**: ¿iguales (consenso simple), por calidad (q en σ) o por rango de resolución (más peso a la gruesa)?
3. **NOP**: ¿NOP sólo cuando ninguna resolución encuentra trade, o también "NOP graduado" para trades de baja
   calidad en la escala más fina?
4. **Cerradores**: ¿regla "la resolución más gruesa que aún paga" (mantener mientras una estructura mayor pague) o
   votos de cierre como objetivo blando?
5. **Umbrales**: θ_open 0.5σ y θ_close 0.1σ como hasta ahora, o fijarlos por resolución.

Con tus respuestas escribo `MultiResLabels`, sus pruebas y el perfil de ejemplos por agente, y sólo entonces
proponemos la corrida (que sigue en espera de tu palabra: sin realimentación, sin z-score móvil, nada en la observación).
