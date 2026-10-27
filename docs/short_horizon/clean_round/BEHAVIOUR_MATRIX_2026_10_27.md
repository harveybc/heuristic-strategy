# Matriz de comportamiento de la estrategia: variación sistemática del mercado con cabezas perfectas (2026-10-27)

Orden del propietario (2026-10-27): nada avanza (ni barridos, ni DEAP, ni dos etapas, ni reporte) hasta que la
estrategia, simulada sobre datos **sintéticos** que cubren **todas** las combinaciones de tendencia, volatilidad y las
demás variables del mercado durante una duración sostenida, se comporte **perfectamente** en cada situación con
cabezas perfectas. Este documento es la columna vertebral: la matriz de celdas (variables de mercado × comportamientos
del agente), con el comportamiento **esperado y prohibido escrito antes de correr**, el testigo que lo afirma y el
PASS/FAIL medido; cada celda que falla con cabezas perfectas es un **defecto de la estrategia** que se corrige y se
vuelve a correr, nunca una celda que se excluye.

Herramientas: `tools/strategy_behaviour_matrix.py` (`cells` / `run` / `report`), generadores en
`tools/strategy_hierarchy.py` (`synthetic_path_v2`, `mirror_bars`, `transition_witness`), parámetros de la estrategia
en `contracts/STRATEGY_PARAMS.json` (validados por `app/short_horizon/expert_guided/strategy_params.py`), pruebas
unitarias en `tests/test_strategy_behaviour_matrix.py`, `tests/test_strategy_params.py`, `tests/test_strategy_hierarchy.py`.
Resultado generado: `evidence/behaviour_matrix/BEHAVIOUR_MATRIX.json` y `BEHAVIOUR_MATRIX.md` (una fila por celda).

## 1. Qué es una celda

Una celda es un **mundo sintético sobre el calendario real** del fold `wf2022` (misma sesión, fines de semana y
rollovers que el mercado real), simulado durante el periodo `TRAIN` completo (18 meses de barras horarias) con la
**verdad exacta del motor** (tablas de resultados celda a celda), **cabezas perfectas** (AUC 1.0 en abridores, cerradores
y NOP; ρ = 1 en la regresión del neto), la **estrategia C3 real** con sus parámetros canónicos (`STRATEGY_PARAMS.json`,
genes de cuantil mapeados en el bloque de calibración de la propia celda) y el **motor real**. Cada celda base `B_<id>`
tiene dos gemelas: `S_<id>` (el mismo camino con todos los costos a cero y financiación simétrica) y `M_<id>` (el
**espejo exacto** en precio de `S_`: p' = 2·c₀ − p, máximos y mínimos intercambiados, exploración y ruido con los lados
intercambiados). Las gemelas existen porque el motor real es asimétrico por construcción: las barras son BID y los
cortos disparan en ASK = bid + spread (variante A4-12), y el swap es −7 largo / +2.5 corto por lote y rollover; sólo a
costo cero una estrategia simétrica puede producir un espejo exacto de sus operaciones. La celda real `B_` se atestigua
por sí misma; la diferencia `B_` − `S_` es la asimetría de costos declarada.

## 2. Variables de mercado y niveles (filas)

| variable | niveles |
| --- | --- |
| dirección × fuerza de la tendencia gruesa | alcista / bajista × débil (0.015 sd/h) / media (0.03 sd/h ≈ 13 %/año) / fuerte (0.06 sd/h); rango (reversión a la media) |
| transiciones | las 6 parejas ordenadas (alza→baja, baja→alza, alza→rango, rango→alza, baja→rango, rango→baja) + alza-pausa-alza y baja-pausa-baja; corte temprano / medio / tardío (25 / 50 / 75 % del bloque); 1–4 tramos por bloque en el hipercubo |
| nivel de volatilidad | 0.7× / 1.0× / 1.6× la sd horaria real |
| dinámica de la volatilidad | constante; escalón baja→alta; escalón alta→baja; GARCH persistencia 0.90; GARCH 0.97; rampa ascendente; rampa descendente |
| estructura fina (ZigZag fino) | amplitud 1.5 / 2.5 / 4 sd × densidad de pivotes ×0.5 / ×1 / ×2 |
| saltos y gaps | ninguno; 2σ y 4σ (de 24 h) a favor y en contra del tramo grueso, en la mitad del bloque |
| régimen de spread | normal; picos de rollover (×2); episodio ×5 de 24 h en la mitad del bloque |
| calendario | normal; corte de entrada 12 h antes del cierre del viernes; tramos de observación ausente (8 tramos de 6–24 h) |
| calidad de las cabezas | perfectas (esta matriz); la escalera de calidad se corre sobre la misma matriz después del 100 % PASS |

Muestreo: **columna vertebral** un-factor-a-la-vez alrededor de la línea base (tendencia bajista media, volatilidad
media constante, fina media, sin gap, spread y calendario normales): 61 celdas; **factoriales** tendencia × fina (63),
tendencia × nivel de vol × dinámica de vol (147), transiciones × amplitud fina (72), tendencia × gaps (35), tendencia ×
spread × calendario (63); **hipercubo latino** sobre todas las perillas, incluidos 1–4 tramos, persistencia GARCH,
pendiente de rampa, posiciones de corte, fuerza continua, tamaño y signo del gap, spread, calendario y perillas de
ejecución (64). Total **501 celdas base × {real, gemela a costo cero, espejo} = 1 503 celdas**; cada celda base con
**3 semillas**; gemelas y espejos con 1 semilla (el espejo es determinista). La especificación (lista de celdas con
esperado/prohibido) se **sella** por su sha256 antes de correr y se registra en `BEHAVIOUR_MATRIX.json`
(`spec_sha256`), junto al sha de `STRATEGY_PARAMS.json` y al commit del código de los testigos.

## 3. Comportamientos del agente (columnas) y testigos

| comportamiento | testigo | afirmación con cabezas perfectas |
| --- | --- | --- |
| OPEN_LONG / OPEN_SHORT | W1 | la cuota de cada lado **sigue la verdad**: \|cuota − cuota de oportunidades de la verdad\| ≤ 0.15, y la dirección del tramo es mayoría cuando la verdad está dominada por ella (≥ 0.60); en rango, ambos lados ≥ 0.25; en transiciones, por tramo |
| conocimiento | W2 | cada trade abierto tiene neto verdadero > 0 en la celda elegida (≥ 0.90); nunca el lado peor cuando el otro supera por > 0.10 σ (≤ 0.10) |
| TP/SL (celda) | W7 | la celda elegida es la mejor celda de la verdad (≥ 0.70) |
| NOP | W3 | aperturas en filas NOP de la verdad ≤ 0.05 |
| HOLD / EARLY_CLOSE | W4 | las posiciones destinadas al SL se cierran antes (salidas por SL ≤ 0.10); todo cierre temprano se justifica por P(SL primero) > umbral o por valor restante negativo (≥ 0.95); se reporta el valor medio del cierre temprano |
| REVERSE | W5 | toda reversión abre el lado que la verdad dice positivo (≥ 0.90); ≤ 1 por semana en tramos estáticos; cuota dentro de las ventanas de corte reportada |
| VOLUME | W6 | unidades = fórmula de riesgo (cuando el libro trae el balance) |
| actividad | W0 | ≥ 1 apertura por semana |
| transiciones | W8 | tras el retardo de confirmación (120 barras) la cuota de aperturas en la dirección vieja ≤ máx(0.40, oportunidad de la verdad + 0.10); tasa de cierres tempranos en la ventana de corte ≥ antes |
| gaps | W9 | el trade que cruza el gap pierde como máximo SL + gap + 0.5 σ |
| spread ×5 | W10 | dentro de la ventana sólo se abre con neto verdadero > 0 (costo incluido) |
| corte / ausencias | W11, W12 | el motor rechaza entradas en la ventana de corte (conteo); ninguna apertura decidida en una fila no observable |
| TP/SL con σ | W13 | la distancia TP en precio sigue a σ cuando la volatilidad cambia (razón segunda/primera mitad) |
| comparación de lados | WA | en cada tick donde ambos lados superan su margen, gana el de mayor EV (auditoría tick a tick: nunca "el primer lado gana") |
| espejo | W14 | la gemela espejo opera **exactamente** espejada: mismas barras de entrada y salida, lados invertidos, neto igual en σ (≤ 2 %) |

## 4. Defectos encontrados y corregidos (estrategia / generadores), con evidencia

| hallazgo | evidencia | corrección |
| --- | --- | --- |
| el generador sintético de tendencia usaba 0.35 sd/h: un camino de dos años colapsaba por e⁻⁴ y el motor rechazaba todos los niveles cortos (14 450 rechazos LEVELS); la hoja `LHS_22` tenía 18 visitas cortas contra 690 largas | reproducción de la hoja antes/después | deriva 0.03 sd/h (≈ 13 %/año) con niveles de fuerza; la hoja queda (607/443 visitas, 0 rechazos LEVELS) |
| `close_p` era un **cuantil** sobre la referencia de P(SL primero) del bloque de calibración; con cerradores perfectos la referencia está saturada ({1e-6, 1−1e-6}) y el umbral mapeado fue 1e-6: se cerraba toda posición corta con P(SL) = 2e-6 | `B_trans_up_down_early` s0: 49 cierres tempranos, 46 con valor restante > 0, valor medio −1.16 σ | `close_p_l/s` son umbrales **absolutos** de probabilidad calibrada (0.50 por defecto, límites [0.30, 0.95]) en C3, en el contrato C3 y en `STRATEGY_PARAMS.json`; después: 2 cierres, ambos con valor restante < 0, beneficio +0.21 σ |
| el veto NOP tenía el mismo defecto: cuantil sobre P(NOP) saturada → umbral 1.0 → nunca vetaba | 0 ticks vetados | `nop_veto` absoluto (0.50): 2 682 de 3 021 ticks vetados en la misma celda |
| filo de navaja de punto flotante: el margen de apertura es un **cuantil muestral** de la referencia de EV, la fila que lo define queda exactamente sobre él y una diferencia de 1e-7 (orden del ajuste Platt) decidió una entrada una barra distinta entre un mundo y su espejo | gemela y espejo 167/169 barras iguales | una apertura exige EV > margen + ε, ε = 1e-6 σ (`open_margin_eps` en `STRATEGY_PARAMS.json`): 155/155 barras iguales, discrepancia de neto 3e-14 σ |

## 5. Testigos sellados, control con el oráculo de la verdad y formas corregidas

Los testigos **no se adaptan a la estrategia**. Cada celda corre, además de la estrategia, un **oráculo de la verdad**:
la política que actúa sobre la verdad del motor sin estrategia alguna (por lado, la celda permitida con mayor neto
verdadero; abre el lado de mayor neto cuando es positivo; cerrado el lado cuando la verdad dice SL-primero o valor
restante negativo; reversión permitida). Los mismos testigos se evalúan sobre el oráculo **antes** de leer el
resultado de la estrategia: un testigo que el oráculo falla es un **testigo defectuoso** (WITNESS_DEFECT, automático
por celda), nunca un defecto de la estrategia. Cada fila del reporte lleva el veredicto bajo la **especificación
sellada original** (celdas `e07bbe73…`, testigos del commit del sellado) y bajo la **especificación re-sellada v2**
(celdas + definiciones de testigos + commit, sha propio), más el veredicto **efectivo** (forma sellada salvo en las
celdas donde el oráculo falla esa forma, entonces la forma corregida). Formas corregidas y su evidencia (del oráculo,
no de la estrategia): W1 en transiciones (el oráculo abre en la dirección del tramo sólo en la medida de las
oportunidades de la verdad: 0.52 largos en un tramo alcista con 0.47 de oportunidad), W5 (el oráculo revierte en
contrapivotes finos positivos fuera de las ventanas de corte: 3 reversiones, 0 en ventana, 3/3 justificadas), W8 (el
oráculo abre en la dirección vieja tras el retardo 0.41–0.44 con oportunidades 0.34–0.41, por encima del 0.40 fijo). La
relajación de W8 hecha tras el 0.405 de la estrategia fue **revertida** por adaptar la prueba al resultado; la forma
corregida descansa sólo en el oráculo. W4 (cierres tempranos con valor restante ≤ 0) **permanece sellado**: el oráculo
lo cumple (8–11 cierres por gemela, todos con valor restante ≤ 0); su forma corregida se calcula y reporta pero nunca
se admite. Antes del sellado se corrigieron dos defectos de instrumentación (fila de decisión desplazada una barra;
espejo aplicado dos veces), registrados en el `witness_changelog` del JSON. El sha sellado original cubre sólo la lista
de celdas; la lista de celdas cambió después del sellado al corregir el espejo (`legs_gen`), por lo que el reporte
muestra tres sha: el sellado original, el de las celdas actuales y el de la especificación v2.

## 5b. Semillas, ZigZag simétrico y corridas invalidadas

En las corridas v1 a v4 la semilla del camino sintético salía sólo del identificador de la celda y las cabezas
perfectas no añaden ruido: las **tres semillas de cada celda eran el mismo mundo**, y su acuerdo entre semillas **no es
evidencia** de robustez. Desde v5 la semilla de la corrida varía el camino. En v5 la escalera conservaba sólo los
pivotes hallados en el camino y en su espejo, una escalera que producción nunca calcula; desde v6 los dos ZigZag de
producción (reversión en pips y MetaTrader experto Depth 3 / Deviation 40.7 / Backstep 0) deciden el primer pivote con
una regla **simétrica bajo negación del precio** (`oracle.reversal_pivots`, `zigzag.select_pivots`; las reglas
anteriores quedan como `*_legacy` sólo para medir el cambio), con la prueba `tests/test_zigzag_mirror_symmetry.py`
(zigzag(K − camino) = espejo(zigzag(camino)) exactamente, 12 caminos × 10 umbrales y 12 caminos del experto). La misma
implementación sirve a la matriz, a las etiquetas y a producción. Impacto en los datos reales (folds `wf2022`/`wf2023`,
2020-01-02 → 2024-12-31): **un pivote por fold** (el primero del ZigZag experto, 2020-01-02 00:00); los diez niveles de
reversión, idénticos.

## 6. Resultado (generado)

<!-- BEHAVIOUR_MATRIX_FRAGMENT -->
