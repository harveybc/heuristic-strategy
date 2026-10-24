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

## 4. Respuestas del propietario (2026-10-24) y propuesta concreta

Respuestas recibidas: (1) escalera {óptimo, rev20, rev15, rev10, rev8, rev5} **aceptada**; (2) la aptitud de cada punto
se mide **siempre respecto al óptimo**: para una apertura, la ganancia que captura si cierra en el cierre óptimo; para
un cierre, la ganancia de un trade abierto en la apertura óptima y cerrado en ese punto (antes o después); (3) NOP como
valor **optimizable** (DEAP/DOIN sólo si es estrictamente necesario), dependiente de factores medibles con el ZigZag
en cada punto frente a los puntos óptimos (volatilidad alta, trade riesgoso); (4) y (5) abiertas.

### 4.1 Aptitud respecto al óptimo, medida (TRAIN; `tools/expert_guided_fitness_vs_optimum.py` → `evidence/multires/FITNESS_VS_OPTIMUM.json`)

Definición medida (tramo óptimo = pivote del óptimo → siguiente pivote del óptimo, en la dirección del trade):

- **apertura** f_open(b) = (nivel del cierre óptimo − apertura en b+1) / (tramo óptimo completo), con signo del lado;
  1 = abrir exactamente en el pivote de inicio; < 1 entrada tardía; **indefinida cuando b cae en el tramo contrario
  del óptimo** (abrir contra la escala óptima: la medición la cuenta aparte).
- **cierre** f_close(b) = (cierre en b − nivel de la apertura óptima) / (tramo óptimo completo); 1 sólo en el extremo
  intrabarra del pivote (inalcanzable al cierre de barra), < 1 antes, cae después del pivote.

| resolución | aperturas etiquetadas | dentro del tramo óptimo | f_open p50 | f_open ≥ 0.5 | f_close en sus pivotes p10 / p50 | f_close ≤ 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 (óptimo) | 4 948 | 100 % | 0.77 | 85 % | 0.45 / 0.78 | 0.1 % |
| rev30 | 2 506 | 91 % | 0.74 | 83 % | 0.30 / 0.75 | 4.8 % |
| rev20 | 4 334 | 86 % | 0.72 | 79 % | 0.20 / 0.71 | 6.4 % |
| rev15 | 5 834 | 83 % | 0.71 | 76 % | 0.14 / 0.69 | 7.4 % |
| rev10 | 7 781 | 79 % | 0.69 | 74 % | 0.10 / 0.67 | 8.1 % |
| rev8 | 8 695 | 77 % | 0.68 | 72 % | 0.09 / 0.66 | 8.4 % |
| rev5 | 9 799 | 75 % | 0.67 | 71 % | 0.08 / 0.65 | 8.4 % |

Lecturas:

- Tu definición **ordena bien** las resoluciones: cuanto más fina, menor aptitud mediana de sus aperturas (0.77 → 0.67)
  y de sus cierres (0.78 → 0.65), y más cierres que terminarían en pérdida medidos contra la apertura óptima (0.1 % →
  8.4 %). Es la "variedad de rendimiento" que pedías, con una sola vara: el óptimo.
- Hasta **una cuarta parte** de las aperturas de rev5 (25 %) está **dentro del tramo contrario del óptimo**: son los
  trades intermedios contra la escala óptima. Con tu regla ("siempre respecto al óptimo") su aptitud es negativa por
  construcción; propongo medirla como −(pérdida hasta el fin del tramo óptimo contrario)/(tramo) para que entren como
  **ejemplos negativos graduados**, no como NOP ciego ni como positivos. Es la siguiente medición si apruebas la regla.
- Incluso el **óptimo** abre con aptitud mediana 0.77, no 1: la etiqueta hold_through marca todo el tramo, así que la
  mayoría de sus filas son entradas tardías. El agente de apertura hoy aprende "abre en cualquier punto del tramo";
  con f_open como peso o como objetivo auxiliar aprende **cuánto queda**.
- Cerrar al cierre de la barra del pivote óptimo captura la mediana del 78 % del tramo (el 100 % es el extremo
  intrabarra). θ_close no debería exigir más que eso.

### 4.2 Cómo lo tenía planeado yo, y en qué cambia

`hold_gain` (ganancia restante hasta el próximo pivote favorable del óptimo, en σ, acotada) ya es tu aptitud sin
normalizar y sólo para el óptimo; `opportunity` ≥ θ_open era su umbral. Lo que tu definición añade: (a) normalizar por
el tramo óptimo (0–1), comparable entre regímenes de volatilidad; (b) medirla **en cada punto de cada resolución**, con
lo que las resoluciones finas dejan de ser "otros expertos" y pasan a ser **generadores de puntos candidatos** que el
óptimo califica; (c) el cierre tiene su propia aptitud, que `hold_gain` no daba. Adopto tu definición.

### 4.3 NOP: propuesta (no decidida; "no hay que tomarlo a la ligera")

NOP debe ser el **complemento de la aptitud** y además castigar el riesgo, con todo medido desde los pivotes
(hindsight, sólo etiquetas) y la volatilidad causal σ (ya en el dataset). Para cada origen o:

| factor | medida | de dónde sale |
| --- | --- | --- |
| nadie encuentra trade | ninguna resolución de la escalera etiqueta L ni S | 8 369 orígenes (49 %), ya medido |
| aptitud máxima | f*(o) = máx sobre resoluciones y lados de f_open | §4.1 |
| riesgo del trade | excursión adversa hasta el cierre óptimo, en unidades del tramo óptimo (MAE/tramo), y en σ | pivotes + mínimos/máximos entre o y el pivote |
| volatilidad alta | tramo óptimo / σ causal pequeño (el tramo no paga la volatilidad) y σ sobre su percentil alto de TRAIN | σ causal del dataset, nunca σ futura |
| consenso | votos L / S de la escalera; conflicto L∧S (0.7 %) | perfil |

Etiqueta propuesta: **NOP duro** (peso 1) donde nadie encuentra trade; **NOP graduado** (peso = 1 − f*, acotado) donde
f* < θ_f **o** MAE/tramo > θ_mae **o** tramo/σ < θ_vol; apertura en otro caso con peso f*. Valores iniciales desde lo
medido: θ_f = 0.5 (lo cumplen 85 % de las aperturas del óptimo y 71 % de rev5), θ_mae = 0.5 (el trade pierde la mitad
del tramo antes de ganar), θ_vol = 0.25σ (el TP mínimo). **Optimización**: los tres umbrales (y θ_open/θ_close de §4.4)
son los genes; el objetivo es el mismo con el que se eligió el óptimo —NSGA-II sobre 2020-2022 con Sharpe ↑ y DD ↓ de
la **demostración** ejecutada bajo el contrato de replay (1 entrada/día, 1:50, ≥ 2 órdenes/semana como validez),
elección por Sharpe 2023, 2024 una lectura— y cada evaluación es un replay de etiquetas (segundos), así que **DEAP
en una máquina basta; DOIN no es necesario**. Límite declarado: ese objetivo es hindsight; la única prueba real es si
el estudiante generaliza en INNER_A, y eso se decide en la ronda, no en el DEAP.

### 4.4 Propuesta para 4 y 5

- **Cerradores (4)**: primera ronda con **dos brazos**: el control `hold_through` de hoy y una regla de aptitud:
  CERRAR cuando la ganancia restante hasta el cierre óptimo < θ_close·tramo **o** f_close ≥ 0.9 (ya capturó lo que
  el cierre de barra permite: mediana 0.78); MANTENER en otro caso; f_close como objetivo auxiliar de regresión en
  lugar de `hold_gain`. Los votos de cierre por resolución quedan como medida reportada, no como etiqueta, hasta ver
  si la regla simple basta.
- **Umbrales (5)**: θ_open = 0.5 en aptitud (sustituye a 0.5σ absoluto), θ_close = 0.1 del tramo (≈ 0.1σ hoy en el
  tramo mediano de 0.55σ); por resolución no hace falta porque la aptitud ya está normalizada por el óptimo.

### 4.5 Sobre "el profit del óptimo me parece bajo"

El óptimo registrado (Depth 3 / Deviation 40.7 pips / Backstep 0; frente A4_wide del estudio v2) se eligió con
**NSGA-II sobre 2020-2022 con objetivos Sharpe ↑ y drawdown máximo ↓**, no profit; punto elegido por mejor Sharpe 2023
entre los válidos (Sharpe hindsight 17.67 / 18.23; 2024 una lectura 18.16, DD 2.35 %, 16.4 órdenes/semana en el
camino ejecutable con tope 4/día). Deviation resultó inerte (6 conjuntos empatados 40.7–493.7 pips) y Depth quedó en la
cota inferior. Además, en el perfil de §1 la demostración corre bajo el **contrato de replay** (1 entrada/día, 1:50),
que recorta su actividad frente al estudio. Si quieres un óptimo con profit como objetivo, es una re-optimización bajo
el contrato de replay (misma herramienta, horas), y cambiaría la vara de §4.1: lo haría antes de medir aptitudes
definitivas, no después.

### 4.6 Lo que sigue (sin lanzar nada)

1. Tu palabra sobre §4.1 (regla para las aperturas contra el óptimo), §4.3 (NOP), §4.4 y §4.5 (re-optimizar o no).
2. Con ella: `MultiResLabels` (aptitud, NOP graduado, cerrador por aptitud; misma disponibilidad por confirmación
   del pivote que `SignalLabels`), pruebas, perfil de ejemplos por agente, y el DEAP de umbrales si lo apruebas.
3. Brazos de la ronda limpia actualizados en `CLEAN_ROUND_SPEC.json`; revisión de Musashi; lanzamiento sólo con tu
   palabra (puerta `LAUNCH_EXPERT_GUIDED.json`).
