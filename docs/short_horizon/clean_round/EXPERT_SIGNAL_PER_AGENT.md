# La señal del experto, diseñada desde cada agente (para aprobación del propietario; rev 2, 2026-10-22)

Pregunta del propietario: ¿qué señal de aprendizaje necesita **cada** especialista (abre-largo,
abre-corto, cierra-largo, cierra-corto, no operar) para aprender su tarea de forma óptima? Hasta hoy
los cuatro aprendían de una sola etiqueta —el pivote del ZigZag óptimo— que no está pensada desde
ninguno de ellos. Este documento define la señal por agente con sus números en TRAIN sobre los
**soportes del negocio** (bloque de ajuste 2020-01 → 2023-07; abridores sobre los orígenes planos
elegibles con puerta de entrada abierta, cerradores sobre las barras observables), y propone cómo
consumirla. Nada de esto es una entrada del modelo: son etiquetas y objetivos del lado de entrenamiento,
purgados por su soporte real. Rev 2 incorpora tres testigos del auditor (ver §6).

## 1. Qué recibe hoy cada agente y por qué no basta

| agente | etiqueta actual | defecto desde su punto de vista |
| --- | --- | --- |
| abre-largo / corto | LARGO en la barra del valle del ZigZag óptimo (CORTO en el pico); TP = lo que falta hasta el siguiente pivote; SL 2σ | sólo 13 % de los orígenes elegibles son entradas; las demás se etiquetan "no abrir" aunque desde ellas haya una entrada rentable: el abridor aprende a reconocer el pivote exacto, no la oportunidad |
| cierra-largo / corto | CERRAR si el SL se toca antes que el TP (o termina negativo al cierre forzoso) | mira sólo los niveles de la orden, no el óptimo: no aprende ni a cerrar en el pico ni a aguantar un retroceso pequeño antes de un pico mayor |
| no operar | implícito: ningún abridor supera el umbral del árbitro | no tiene ejemplos propios: hereda los huecos del ZigZag |
| TP / SL (regresión) | TP = resto hasta el pivote; SL fijo 2σ | el SL no se deriva del camino real |

## 2. La señal por agente (horizonte = el próximo pivote **favorable** del ZigZag óptimo, Depth 3)

Decisión tras la barra completada b; ejecución en la apertura siguiente; todo en σ causal
(`sigma[b+1]` usa barras ≤ b). Un solo horizonte por lado, **compartido por el abridor y el cerrador
de ese lado**: el próximo pico para un largo, el próximo valle para un corto, acotado por 72 barras,
el cierre forzoso de la semana y el fin de los datos. Nada más allá de la ventana se lee jamás —ni
siquiera el tipo de un pivote posterior— y cada señal lleva `truncated = 1` cuando la ventana
terminó por un tope y no por el pivote (5 % de las filas), para que el consumidor pueda censurarla.

**Abridor de lado d:** `oportunidad` = máxima excursión favorable desde la apertura siguiente hasta el
horizonte, neta de costo (0.06σ declarado; el costo ejecutado por spread, slippage, comisión y funding
se registra aparte en el simulador) — "cuánto es lo máximo que podrá ganar antes del óptimo"; es una cota
superior sobre HIGH/LOW, no un precio de salida ejecutable; `adversa_hasta_el_máximo` = retroceso máximo
antes de ese máximo (el SL que necesita para sobrevivir); `barras_hasta_el_máximo`. Etiqueta: ABRIR si
oportunidad ≥ θ_open y adversa ≤ SL máximo (2σ).

**No operar:** ningún abridor activo.

**Cerrador de lado d:** `ganancia_adicional` = cuánto más podría ganar frente a cerrar ahora, hasta el
mismo horizonte; `adversa_si_mantiene` = retroceso máximo hasta ese máximo adicional. Dos
contrafactuales **declarados**, que se contrastan en el bloque interno en lugar de imponer uno:
- `hold_through`: CERRAR si ganancia adicional < θ_close o la adversa alcanza el SL de la orden. Mantiene
  la posición a través de un retroceso sobrevivible cuando hay un extremo favorable mayor por delante.
  Es coherente con el abridor en la misma barra (un largo que vale la pena abrir vale la pena mantener).
- `optimum`: además CERRAR si el próximo pivote óptimo de **cualquier** tipo dentro de la ventana es
  adverso (el experto óptimo cierra en cada extremo favorable y reentra). Censurado (NaN) cuando ningún
  pivote cae dentro de la ventana. Reentrar está limitado por 1 entrada/día y los costos: por eso es una
  política distinta, no "el cierre óptimo demostrado".

**Dependencia del SL:** la etiqueta del cerrador depende del SL de la orden; sin realimentación, dos
observaciones de mercado idénticas con SL distintos pueden exigir etiquetas distintas. Se registra como
observabilidad parcial; el SL **no** se mete como entrada oculta.

**Cabezas de regresión:** oportunidad y adversa (abridores), ganancia adicional (cerradores), acotadas por
percentiles de TRAIN; se conserva el objetivo crudo y se reporta el porcentaje saturado por periodo.

## 3. Números en TRAIN sobre los soportes del negocio (`evidence/SIGNAL_DESIGN_PROFILE.json`)

Soportes: 21 809 barras TRAIN; 17 128 orígenes planos elegibles; 13 272 con los dos horizontes
finales dentro de TRAIN (3 856 excluidos por horizonte no final); cerradores 16 947 / 16 959 barras
observables; 0 NaN por objetivo; 4.7–5.1 % de ventanas truncadas.

| θ_open (σ) | abrir largo | abrir corto | ambos lados | NOP | P(etiqueta \| ZigZag largo) | P(ZigZag \| etiqueta largo) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 52 % | 50 % | 19 % | 17 % | 0.71 | 0.18 |
| **0.50** | **29 %** | **28 %** | 4 % | **47 %** | 0.41 | 0.19 |
| 0.75 | 17 % | 17 % | 1 % | 67 % | 0.24 | 0.18 |
| 1.00 | 11 % | 11 % | 0 % | 79 % | 0.15 | 0.18 |

Percentiles (largo): oportunidad p50 0.26 / p75 0.56 / p95 1.37 σ; adversa hasta el máximo p50 0.14 /
p95 0.67 σ; barras hasta el máximo p50 4 / p95 13; ganancia adicional del cerrador p50 0.34 σ.

| cerrador (SL 2σ) | θ_close 0.05 | 0.10 | 0.25 | 0.50 | CERRAR en los picos óptimos (largo, θ 0.1) |
| --- | --- | --- | --- | --- | --- |
| `hold_through` | 8 % | 17 % | 40 % | 66 % | 31 % |
| `optimum` | 56 % | 57 % | 67 % | 81 % | 100 % |

Lectura: θ_open = 0.5σ da 2.2 veces más entradas que el ZigZag con casi ningún conflicto entre lados
(4 %) y la mitad de las filas como NOP con ejemplos propios; sólo el 19 % de esas entradas son pivotes
del ZigZag: el resto son los trades intermedios de menor ganancia que el propietario pidió cubrir. Los
dos cerradores difieren de verdad: en los picos óptimos `optimum` siempre cierra y `hold_through` sólo
cuando el pico siguiente no compensa el retroceso (mediana del retroceso 0.29σ, de la ganancia siguiente
0.20σ). Cuál enseña mejor se decide en el bloque interno, no aquí. Estas proporciones describen etiquetas;
no demuestran que el modelo las aprenda.

## 4. Cómo se consume (propuesta; no implementado hasta la aprobación)

1. E3-M `--labels zigzag|opportunity|both`. `opportunity` sustituye la etiqueta de ronda 0 de los abridores
   por ABRIR/NO (regla §2) y la de los cerradores por la regla elegida (`--closer-rule`). `both` es un OR
   con tabla de verdad fija: lado = LARGO si ZigZag dice LARGO o la oportunidad dice LARGO (ídem CORTO);
   si los dos lados quedan activos a la vez manda el ZigZag; NOP sólo cuando ninguno dice nada; máscara por
   tarea registrada. Ponderación 1:1 declarada; sin pesos a posteriori.
2. Rondas DAgger: el oráculo que etiqueta los estados visitados usa las mismas reglas desde la barra
   visitada (cerrador: "¿queda óptimo por delante / resiste el SL?", no "¿pierde esta orden?").
3. Regresión: oportunidad/adversa (abridores), ganancia adicional (cerradores), máscara por objetivo,
   objetivo crudo conservado y porcentaje saturado reportado.
4. Umbrales declarados antes de correr: θ_open 0.5σ, SL máx 2σ, θ_close 0.1σ; una variante (0.75σ) como
   control de densidad. Etapa B de la ronda limpia: {ZigZag, oportunidad, ambas} × {cerrador hold_through,
   optimum} × {sin / con demostradores cortos}.
5. Pruebas: ninguna fila usa un horizonte que termine fuera de su periodo; el tipo de un pivote fuera de la
   ventana nunca cambia una etiqueta (testigo 2 del auditor como regresión); extremo a extremo con el CLI.
6. Casos límite, con la política intrabar del ejecutor (sin inventar el orden de los ticks): fin de datos sin
   pivote siguiente → ventana truncada (`end`); horizonte de una barra; SL y TP tocados en la misma H1 → SL
   primero (conservador, como `exit_label`); el cierre forzoso tiene prioridad sobre todo.

## 5. Regla de actividad (propietario 2026-10-22: ≥ 2 órdenes/semana)

El código evalúa el **promedio de órdenes por semana del periodo** (`orders_per_week ≥ 2`), no un mínimo
cumplido en cada semana. Desde hoy cada corrida reporta además la distribución: `weeks_below_min_orders`
y su proporción. Es la semántica declarada; no se fuerzan operaciones para cumplirla.

## 6. Testigos del auditor (2026-10-22) y su cierre

| testigo | qué demostró | cierre |
| --- | --- | --- |
| 1 — dominio de LOG | cambiar un valor posterior a TRAIN cambiaba la receta (LOG → ASINHS) y los valores de TRAIN | receta decidida sólo en TRAIN; una fila posterior fuera del dominio pasa a NaN y se cuenta (predictor @048549f0, regresión PRE/POST); impacto medido en las stores `_s`: **0 columnas**; la store capa 1 reconstruida con la regla nueva es byte-idéntica (`8243f43f…`) |
| 2 — tipo de pivote fuera del horizonte | la etiqueta del cerrador usaba el tipo de un pivote posterior al cierre forzoso | nada más allá de la ventana se lee; `truncated` y motivo del fin (`pivot/cap/flat/end`) por fila; regla `optimum` censurada cuando no hay pivote dentro (regresión con su mismo camino) |
| 3 — abridor abre, cerrador cierra | "próximo pivote adverso ⇒ el óptimo pasó" no era válido | un horizonte favorable compartido por lado: `hold_through` coincide con el abridor; `optimum` queda como contrafactual declarado para contrastar |
