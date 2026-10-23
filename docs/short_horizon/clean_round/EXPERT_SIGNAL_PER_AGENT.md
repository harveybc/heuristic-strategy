# La señal del experto, diseñada desde cada agente (para aprobación del propietario, 2026-10-22)

Pregunta del propietario: ¿qué señal de aprendizaje necesita **cada** especialista (abre-largo,
abre-corto, cierra-largo, cierra-corto, no operar) para aprender su tarea de forma óptima? Hasta hoy
los cuatro aprendían de una sola etiqueta —el pivote del ZigZag óptimo— que no está pensada desde
ninguno de ellos. Este documento define la señal por agente, con sus números en TRAIN (bloque de
ajuste 2020-01 → 2023-07, 19 119 filas planas con horizonte final), y propone cómo consumirla.
Nada de esto es una entrada del modelo: son etiquetas y objetivos del lado de entrenamiento,
purgados por su soporte real (una fila usa la señal sólo si su horizonte termina dentro de su periodo).

## 1. Qué recibe hoy cada agente y por qué no basta

| agente | etiqueta actual | defecto desde su punto de vista |
| --- | --- | --- |
| abre-largo / corto | LARGO en la barra del valle del ZigZag óptimo (CORTO en el pico); TP = lo que falta hasta el siguiente pivote; SL 2σ | sólo 11 % de las filas planas son entradas; las demás se etiquetan "no abrir" aunque desde ellas haya una entrada rentable (el 60 % de las entradas del ZigZag ni siquiera alcanzan 0.5σ netos antes del siguiente pico, y muchas filas no-ZigZag sí): el abridor aprende a reconocer el pivote exacto, no la oportunidad |
| cierra-largo / corto | CERRAR si el SL se toca antes que el TP (o termina negativo al cierre forzoso) | mira sólo los niveles de la orden, no el óptimo: no aprende a cerrar en el pico ni a aguantar un retroceso pequeño antes de un pico mayor; la etiqueta es rara (desequilibrada) |
| no operar | implícito: ningún abridor supera el umbral del árbitro (elegido en el bloque interno) | no tiene ejemplos propios: hereda los huecos del ZigZag |
| TP / SL (regresión) | TP = resto hasta el pivote; SL fijo 2σ | el SL no se deriva del camino real: no aprende cuánto hay que resistir para llegar al mejor cierre |

## 2. La señal por agente (horizonte = el pivote del ZigZag **óptimo**, Depth 3)

Decisión tras la barra completada b; ejecución en la apertura siguiente; todo en σ causal.

**Abridor de lado d.** Horizonte = el próximo pivote óptimo **favorable** (pico para largo, valle para
corto), acotado por 72 barras y el cierre forzoso. Señales: `oportunidad` = máxima excursión favorable
desde la apertura siguiente hasta ese pivote, neta de costo ("cuánto es lo máximo que podrá ganar
antes del óptimo"); `adversa_hasta_el_máximo` = retroceso máximo que sufre antes de ese máximo (el SL
que necesita para sobrevivir); `barras_hasta_el_máximo`. Etiqueta: ABRIR si oportunidad ≥ θ_open y
adversa ≤ SL máximo (2σ). El horizonte favorable es esencial: la etiqueta del ZigZag marca el valle,
y el óptimo de esa entrada es el pico siguiente; con "próximo pivote de cualquier tipo" la señal se
truncaría en el valle.

**No operar.** Ningún abridor activo. Con θ_open = 0.5σ: 47 % de las filas planas.

**Cerrador de lado d.** Horizonte = el próximo pivote óptimo de **cualquier** tipo. Señales:
`ganancia_adicional` = cuánto más podría ganar frente a cerrar ahora; `adversa_si_mantiene`;
`próximo_pivote_favorable` (si es adverso, el óptimo ya pasó). Etiqueta: CERRAR si el próximo pivote
es adverso, o ganancia adicional < θ_close, o la adversa alcanza el SL de la orden. En los picos
óptimos, la ganancia adicional del largo es ≤ 0.29σ (p95): el cerrador aprende a cerrar justo ahí
("cuánto más pudo haber ganado respecto al óptimo").

**Cabezas de regresión.** Abridor: oportunidad y adversa (de ahí salen TP y SL derivados del camino
real: TP = fracción declarada de la oportunidad, SL = adversa + margen). Cerrador: ganancia adicional.
Rangos acotados por los percentiles de TRAIN.

## 3. Números en TRAIN (etiquetas, sin modelo; `evidence/SIGNAL_DESIGN_PROFILE.json`)

| θ_open (σ) | abrir largo | abrir corto | ambos lados | NOP | P(etiqueta \| ZigZag largo) | P(ZigZag \| etiqueta largo) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 53 % | 52 % | 15 % | 17 % | 0.71 | 0.15 |
| **0.50** | **29 %** | **29 %** | 3 % | **47 %** | 0.41 | 0.16 |
| 0.75 | 18 % | 17 % | 1 % | 67 % | 0.24 | 0.15 |
| 1.00 | 11 % | 11 % | 0 % | 79 % | 0.15 | 0.16 |

Percentiles (largo): oportunidad p50 0.27 / p75 0.58 / p95 1.40 σ; adversa hasta el máximo p50 0.15 /
p95 0.65 σ; barras hasta el máximo p50 4 / p95 13. Cerrador con θ_close = 0.1σ y SL 2σ: 51 % CERRAR
(equilibrado); con 0.25σ, 61 %.

Lectura: θ_open = 0.5σ (≥ el TP efectivo mínimo 0.25σ + costo, con margen) da **2.6 veces más
entradas** que el ZigZag, casi sin conflicto entre lados (3 %), y la mitad de las filas como NOP con
ejemplos propios. Sólo el 16 % de esas entradas son pivotes del ZigZag: el resto son los trades
intermedios de menor ganancia que el propietario pidió cubrir. Los demostradores más cortos (Depth 2,
reversión 10 pips) son una segunda vía hacia lo mismo; las dos se contrastan en la etapa B.

## 4. Cómo se consume (propuesta; no implementado hasta la aprobación)

1. E3-M `--labels zigzag|opportunity|both`: `opportunity` sustituye la etiqueta de ronda 0 de los
   abridores por ABRIR/NO (regla §2) y la de los cerradores por la regla de óptimo; `both` las une
   (consenso) — el ZigZag mantiene la entrada exacta, la oportunidad añade las intermedias.
2. Rondas DAgger: el oráculo que etiqueta los estados visitados usa las mismas reglas (la oportunidad
   desde la barra visitada), así la etiqueta del cerrador en una posición abierta por el alumno es
   "¿queda óptimo por delante?" y no "¿pierde esta orden?".
3. Regresión: oportunidad/adversa (abridores), ganancia adicional (cerradores), máscara por objetivo.
4. Umbrales declarados antes de correr: θ_open 0.5σ, SL máx 2σ, θ_close 0.1σ; una sola variante
   (0.75σ) como control de densidad. Etapa B de la ronda limpia: {etiqueta ZigZag, oportunidad,
   ambas} × {sin / con demostradores cortos}.
5. Prueba de purga: ninguna fila usa un horizonte que termine fuera de su periodo; prueba de
   extremo a extremo con el CLI.

Decisión en el bloque interno 2023H2 por Sharpe medio con tres semillas válidas, como el resto de la ronda.
