# La estrategia simulada con salidas ideales y simuladas del modelo: parámetros optimizados, barridos de ruido y catálogo de escenarios (2026-10-26)

Para el propietario y para la auditoría cruzada de Musashi. Responde a la orden del 2026-10-26: antes de pedir una
auditoría y antes de cualquier selección flotante, la estrategia debe estar **completamente simulada y justificada con
salidas ideales y simuladas de las cabezas**, con sus **parámetros optimizados sobre salidas sintéticas** de las
características que esperamos de los modelos, con **barridos de ruido** que muestren desde qué calidad de salida
empieza a pagar y cómo de robusta es, **en todas las condiciones**. Todo lo que sigue corre por el **mismo código**
que los experimentos reales (estrategia C3, motor, optimizador, chequeos); lo único simulado son las salidas de las
cabezas. Nada aquí es un resultado de trading del mercado real: es la medición del **instrumento** (la estrategia) con
una entrada de calidad conocida.

Herramienta: `tools/strategy_output_simulation.py` (+ `_report.py`, 17 pruebas en `tests/test_strategy_output_simulation.py`);
driver: `tools/clean_round_chain/run_strategy_sim_sweeps.sh`; evidencia: `evidence/strategy_sim/` (CSV, JSON, PNG,
libros por trade `ledger/*.csv.gz`, humo en `smoke/`).

## 1. Qué se simula y qué no

| capa | real o simulada | detalle |
| --- | --- | --- |
| barras, calendario, cierres, σ causal | **real** (barras admitidas EURUSD H1, 2020-01 → 2023-12; TEST ≥ 2025 jamás) | folds `wf2022` (ajuste 2020-01→2021-07, calibración = selección 2021 S2, decisión **2022**) y `wf2023` (ajuste →2022-07, calibración 2022 S2, decisión **2023**) |
| verdad de cada acción | **real, exacta del motor**: `outcomes.entry_table` (toda fila elegible × lado × 25 celdas TP/SL del contrato C3: neto en σ con costos, cuál barrera llega primero {TP, SL, TIME}, MAE, MFE); `outcomes.hold_value` (mantener vs cerrar ahora en cada barra posicionada) | sin impacto de mercado, el contrafactual de cada acción en cada tick es exacto |
| salidas de las cabezas | **simuladas con calidad declarada** (§2) | P(TP primero), P(SL primero), P(TIME) por lado; neto previsto por celda; P(NOP); P(SL primero) y valor restante de los cerradores |
| calibración | **real**: Platt por cabeza binaria **sólo en el bloque de calibración** (`action_contract.fit_platt`) | las probabilidades que consume la estrategia son las calibradas |
| estrategia | **real**: `c3.PolicyC3` (veto NOP → EV por lado sobre la rejilla con la regla de riesgo/beneficio y la banda efectiva por barra → margen por lado → puerta del mejor del día → orden con TP/SL de la celda; en posición: P(SL primero) y restante → umbral por lado → reversión) con el variante de ejecución `full_f3` (tamaño por riesgo 1 %, reversión = una decisión, corte 24 h antes del cierre del viernes, disparo de stops cortos al ask) | identidad del contrato `action_652cbdd91d03961d`; ejecución `exec_709f9b1fa3512486` |
| optimización de los genes | **real**: `c3.freeze` → `action_contract.freeze_strategy` (NSGA-II, dominancia por restricciones de Deb con actividad ≥ 2/sem por cota inferior IC 90 % y lado ≤ 90 %, archivo completo, F0–F6, cierre temprano exigido, prueba de beneficio del cierre temprano) **sólo sobre el bloque de selección** | presupuesto declarado 24 × 12 sin escalada (`optimizer.final_escalated` = `final`, declarado para el barrido) |
| evaluación | **real**: una réplica en el bloque de decisión por configuración y semilla, con los 15 chequeos de corrección (`expert_guided_trading_report.correctness`, conscientes del variante de ejecución), campos de cordura, valor del cierre temprano (con − sin), expectativa neta por trade con IC bootstrap por días, acierto por lado, desglose por trimestre y régimen | el bloque de decisión se lee **una vez** por configuración y semilla y queda en el libro por trade |

## 2. Cómo se simula una cabeza de calidad conocida

Para cada tick t (fila observable), lado d y cabeza:

- **Binarias** (TP primero, SL primero, NOP, SL primero del cerrador): latente s = β·y' + ε, con y' ∈ {−1, +1} la verdad
  del motor y ε ~ N(0, 1). Con esta codificación AUC = Φ(β·√2), luego **β = Φ⁻¹(AUC)/√2**; AUC 1.0 → β = 12 (separable),
  AUC 0.5 → β = 0 (azar), AUC < 0.5 → β < 0 (cabeza invertida). La probabilidad que consume la estrategia es Platt(s)
  ajustada en el bloque de calibración; TIME = 1 − P(TP) − P(SL), renormalizado (aproximación declarada de la cabeza
  de 3 clases).
- **Regresiones** (neto por celda, valor restante): r = x + σ_n·ε con **σ_n = sd(x)·√(1/ρ² − 1)** para una correlación
  ρ; ruido compartido por fila (50 %) + ruido por celda; se reporta la Spearman lograda, el MAE y **MAE/MAE naive**
  (naive = la media del bloque de calibración), como exige la regla de barridos de ruido del propietario.
- **Escalera de calidad**: AUC = ρ ∈ {0.50, 0.52, 0.55, 0.58, 0.62, 0.66, 0.70, 0.75, 0.80, 0.90, 1.00} (base) y
  {0.50, 0.62, 0.70, 0.80, 1.00} (escenarios). La calidad lograda se mide en el bloque de calibración y en el de
  decisión y se publica junto a la pedida (pruebas: AUC pedida lograda ± 0.01; ρ ± 0.03).
- **Realismo de forma**: desbalance de clases y cuota de TIME reales (vienen de la verdad), asimetría por lado,
  correlación serial del error dentro del día (variante `block`, ρ_día 0.5), caída de calidad a mitad del bloque de
  decisión (variante `regime`), deriva de calibración (logits × 0.8 / × 1.2), retardo de una fila (`lag1`), saturación,
  cabeza constante, cabeza invertida.
- **Cerradores**: `SimCloser.value(r, bar, pos)` simula sobre la verdad exacta de **esa** orden abierta
  (`hold_value`), con caché por (barra, lado, barra de entrada, TP, SL); las referencias de calibración vienen de visitas
  de exploración (política aleatoria con TP/SL de la rejilla, prior declarado 0.85/0.05/0.05/0.05), nunca del experto.
- **Determinismo**: ruido por clave (cabeza, fila, lado) y semilla; tres semillas de ruido (0, 1, 2) por configuración.

## 3. Rangos de salida de cada cabeza y cómo los consume la estrategia (para configurar las cabezas)

| cabeza | salida | rango | cómo la consume la estrategia (`c3.PolicyC3`) | qué pasa si… |
| --- | --- | --- | --- | --- |
| `open_l`, `open_s`: 3 clases | p3 = (P(TP), P(SL), P(TIME)) por lado en la celda de referencia (1σ/1σ) | símplex; Platt / temperatura en calibración | EV = P(TP)·TP − P(SL)·SL + P(TIME)·E[neto\|TIME] − costo, evaluada en la celda elegida por el neto previsto; abre si EV > margen del lado (cuantil congelado de la distribución de EV del bloque de calibración) | **saturada** (0.01/0.99): la EV salta entre dos valores → el margen por cuantil degenera en todo-o-nada (escenario `sat`); **constante**: EV constante → o abre siempre (actividad por tope) o nunca (`STRATEGY_SPACE_INFEASIBLE`); **invertida** (AUC < 0.5): abre contra la realidad (escenario `inv_open`) |
| `open_l`, `open_s`: regresión | neto previsto por celda (25) | [−4, 4] σ (`NET_BOUND`) | elige la celda con mayor neto previsto entre las **permitidas**: TP ≥ SL siempre; TP < SL sólo si neto > costo + 0.10σ; dentro de la banda efectiva por barra [mínimo, tope]; también fija el margen de apertura (referencia EV) | **constante** (escenario `const_net`): la regla RR decide sola la celda (la de mayor TP permitido) y la EV depende sólo de p3; **escala**: si predice en otra unidad que σ, la regla "neto > costo + margen" cambia de sentido — por eso el objetivo es el neto del motor en σ |
| `nop` | P(no operar) | [0, 1], Platt | veto: no abre si P(NOP) > umbral (cuantil congelado); el veto sólo se conserva si mejora la cota inferior en calibración | **saturada**: veto todo-o-nada; **invertida**: veta los mejores ticks (medido en `inv_open`? no: el NOP invertido se mide dentro de `nop_only` a calidad < 0.5 sólo en pruebas unitarias; en el catálogo el NOP entra a la calidad del nivel) |
| `close_l`, `close_s`: binaria | P(SL primero desde aquí) de **la orden abierta** (geometría causal de la orden como entrada) | [0, 1], Platt | cierra si P > umbral del lado (cuantil congelado sobre las visitas posicionadas de ese lado) | **saturada**: cierra todo o nada; **invertida**: mantiene las perdedoras y cierra las ganadoras (la auditoría de cierre lo detecta: valor del cierre < 0 → se desactiva por la prueba de beneficio) |
| `close_l`, `close_s`: regresión | valor restante de mantener (σ) | [−4, 4] σ | cierra si restante < −margen del lado | **constante**: el umbral por cuantil no dispara (sin información); la prueba de beneficio desactiva el cierre temprano |
| capa superior | 8 genes en cuantiles del bloque de calibración (`open_rate_l/s` ∈ [0.002, 0.5], `close_rem_rate_l/s`, `close_p_rate_l/s` ∈ [0, 0.6], `allow_reverse`, `nop_veto_rate` ∈ [0, 0.5]) | — | el mapa cuantil → umbral absoluto se congela en calibración y se usa sin cambios en decisión | la sensibilidad medida ±0.1 por gen está en §6 |

Consecuencia para la configuración de las cabezas: las cabezas binarias deben entregar **probabilidades calibradas** (la
estrategia usa cuantiles de su distribución y una EV monetaria), las regresiones deben estar **en σ del motor** y
acotadas, y ninguna cabeza puede ser constante (una cabeza constante o saturada no rompe el motor pero destruye la
información que la estrategia necesita, y el barrido lo cuantifica).

## 4. Catálogo de escenarios (lo que se corrió; la tabla de cobertura medida está en §7)

| escenario | qué cambia | dónde |
| --- | --- | --- |
| `base` | todas las cabezas a la calidad del nivel; **dos puertas**: `contract` (observa 1 señal del día antes de colocar, como el contrato C3) y `perbar` (coloca la primera señal válida y la EV paga el costo **de la barra**: spread de la hora + slippage + comisión en σ) | escalera completa, dos folds |
| `block` | ruido con componente compartida dentro del día (ρ_día 0.5) | cabezas |
| `regime` | la calidad cae 0.10 **desde la mitad** del bloque de decisión (cambio de régimen a mitad de bloque) | cabezas |
| `onesided` / `asym` | abridores cortos al azar / 0.10 por debajo del nivel (asimetría por lado) | cabezas |
| `opener_only` / `closer_only` / `nop_only` | sólo esa familia informativa; las demás al azar | cabezas |
| `cal08` / `cal12` | deriva de calibración: logits calibrados × 0.8 (subconfianza) / × 1.2 (sobreconfianza) | cabezas |
| `lag1` | señal retrasada una fila observable | cabezas |
| `missing10` / `missing30` | 10 % / 30 % de filas observables no observables (camino HOLD-sin-observación) | observación |
| `cost2` / `cost3` / `spike` | costos × 2 / × 3 (verdad y réplica) / spread del rollover y apertura semanal × 2 | motor (verdad propia) |
| `cap2` / `cap4` | tope diario 2 / 4 entradas (variante declarada) | motor |
| `cutoff12` | corte de entrada 12 h antes del cierre del viernes (contrato: 24 h; verdad propia) | motor |
| `risk05` / `risk2` | riesgo 0.5 % / 2 % por trade | motor |
| `inv_open` / `const_net` / `sat` | abridores invertidos (AUC 1 − nivel) / neto constante / probabilidades saturadas, a calidad 0.8 | forma de las cabezas |
| trimestres por régimen | cada bloque de decisión partido en trimestres etiquetados por terciles de volatilidad realizada y de eficiencia de tendencia \|Σr\|/Σ\|r\| sobre los trimestres 2021–2023 (tendencia / rango, alta / baja volatilidad) | evaluación de la estrategia congelada, sin nueva elección |
| oráculo y azar | niveles 1.00 y 0.50 en todos los escenarios: techo y suelo en cada figura | — |

## 5. Qué fitness usar (regla del propietario)

En cada configuración y semilla, `freeze_strategy` optimiza **una vez** con NSGA-II (Sharpe ↑, DD ↓, dominancia por
restricciones) y después **cada variante de fitness elige su punto** en el frente válido: F0 Sharpe − EE (defecto),
F1 Sharpe + retorno anualizado, **F2 Sharpe + retorno + órdenes/día (la compuesta del propietario)**, F3 t por trade,
F4 Calmar, F5 retorno − DD, F6 balanceada; los coeficientes de las compuestas son **c_k = 1 / mediana\|término_k\|** sobre
el frente válido de F0 en el bloque de calibración (`term_scales`, con el respaldo declarado por debajo de 1e-6). El
punto elegido por cada fitness se puntúa **una vez** en el bloque de decisión con el motor real. El reporte (§7) da, por
escenario y calidad, qué fitness obtiene la mejor expectativa / Sharpe / neto fuera de bloque y cuántas celdas
fold × semilla gana cada una (estabilidad): esa tabla es la recomendación de fitness para producción, con su evidencia.

## 6. Ventana de entrenamiento para la selección flotante: tabla de costo medida

Tiempos **medidos** por semilla (4 módulos + rondas DAgger, GPU 4090, tronco u, 121 columnas), tomados de los registros
de las corridas reales (`out/<unidad>.log`, campo `s` acumulado por semilla; filas de ajuste del libro `FIT`):

| ventana de ajuste | filas de ajuste (abridores) | segundos por semilla (cada semilla) | minutos por ajuste |
| --- | ---: | --- | ---: |
| 1.5 años (2020-01 → 2021-07, `wf2022`) | 7 182 | 347, 345, 294, 298, 295 | **5.3** |
| 2.5 años (2020-01 → 2022-07, `wf2023`) | 11 907 | 451, 401, 368, 373, 420 | **6.7** |
| 3.5 años (2020-01 → 2023-07, consolidado) | 16 958 | 654, 625, 560 | **10.2** |

El costo crece menos que lineal con las filas (la parada temprana corta antes con más datos). Para la búsqueda flotante
(hasta 128 subconjuntos × 3 semillas = 384 ajustes) eso es **34 h (1.5 años) / 43 h (2.5 años) / 65 h (3.5 años)** de
GPU; con el tronco TCN (≈ 3× parámetros) el factor medido en el humo de C3 es ≈ 1.3–1.5×.

Ruido del score (lo que de verdad decide): el score de la selección se mide en el bloque de **selección**, no en la
ventana de ajuste; su error estándar es sd_trade/√N. Medido en las simulaciones (§7, `sd_per_trade` y N por bloque):
un bloque de 6 meses da N ≈ 50–75 trades a 2–2.7/sem y sd ≈ 0.42–0.45σ → **EE ≈ 0.05–0.06σ**; un bloque de 12 meses da
N ≈ 90–140 → **EE ≈ 0.04σ**. Una diferencia entre dos subconjuntos sólo es distinguible si supera ≈ 1.4 × EE
(≈ 0.08σ con 6 meses, ≈ 0.06σ con 12). La ventana de ajuste larga cuesta el doble de GPU por ajuste y no reduce este
ruido; sólo un bloque de selección más largo lo reduce. Esa es la tabla; la decisión es del propietario.

## 7. Resultados de los barridos (medidos; tablas generadas por `tools/strategy_output_simulation_report.py`)

<!-- SWEEP_FRAGMENT -->

## 8. TP/SL y volumen por dirección: fórmulas explícitas, familias comparadas y coeficientes optimizados

Capa paramétrica completa en `app/short_horizon/expert_guided/strategy_full.py` (`StrategyFull`, `PolicyFull`,
`rollout_with_volume`; 5 pruebas en `tests/test_strategy_full.py`). Corre sobre el mismo motor y las mismas tablas de
resultados que C3; la EV por celda, el veto NOP, los umbrales de cierre por lado y la puerta mejor-del-día son los de C3.
Notación: d ∈ {largo, corto}; σ_t la volatilidad de la hora (la del contrato de salida); q_d = P(TP primero | d),
r_d = P(SL primero | d) (abridores calibrados por Platt en el bloque de calibración); p_nop = P(NOP) de la cabeza de 3
clases; [lo_t, cap_t] la banda de distancias de la barra (la del contrato C3).

**Familia TP/SL (a) `cell` (C3):** (TP_d, SL_d) = argmax sobre las 25 celdas de la EV prevista
`EV_c = q_c · TP_c − (1 − q_c) · SL_c − costo` dentro de la banda; una celda con TP < SL sólo se admite si su EV supera
el costo + 0.10 σ (regla RR del contrato).

**Familia TP/SL (b) `param`:** `TP_d = σ_t · (a_d + b_d · q_d)`, `SL_d = σ_t · (c_d + e_d · r_d)`, ambas recortadas a
[lo_t, cap_t] y a los límites de TP/SL del entorno. Genes: a_d ∈ [0.25, 2.0], b_d ∈ [−1.5, 1.5], c_d ∈ [0.5, 2.5],
e_d ∈ [−1.5, 1.5] por dirección (8 coeficientes).

**Familia de volumen (1) `risk` (motor):** `v_d = min(1, riesgo · balance / (distancia_SL · unidades_plenas))`, con
riesgo 1 % (0.5 % y 2 % en los escenarios `risk05`/`risk2`).

**Familia de volumen (2) `param`:** `v_d = clip(α_d + β_d · q_d − γ_d · p_nop − δ_d · r_d, 0, 1)`; si `v_d < v_min` no se
coloca la orden. Genes: α_d ∈ [0, 1], β_d ∈ [−1, 1], γ_d ∈ [0, 1], δ_d ∈ [0, 1] por dirección y v_min ∈ [0, 0.5] (9
coeficientes). En ambas familias las unidades son las de la regla del propietario:
`unidades = ⌊v · f · balance · rel_volume · apalancamiento / precio⌋`, rel_volume 0.10, apalancamiento ≤ 1:50 (el motor
aplica `percentual_order_size = f · v`, que persiste hasta la siguiente entrada; corregido en el humo, §12).

**Ventana de observación:** `observe_bars` ∈ [0, 3] señales del día antes de colocar (0 = la primera señal válida,
puerta `perbar`; 1 = el contrato C3). Es un gen más.

**Optimización:** NSGA-II (DEAP) sobre todos los genes a la vez (8 cuantiles de C3 + observe_bars + coeficientes:
18 genes en `param_risk`, 18 en `cell_param`, 26 en `param_param`), población 48 × 24 generaciones, dominancia por
restricciones de Deb (actividad IC90 ≥ 2/semana, lado ≤ 90 %, cierre temprano obligatorio), fitness primario **la
compuesta del propietario** (Sharpe + retorno/balance_inicial + órdenes/día con c_k = 1/mediana|término_k|). Se optimiza
en el bloque de selección y se puntúa **una vez** en el de decisión; los archivos completos de cada punto evaluado
(`ledger/ARCHIVE_*.json.gz`, genes + fitness + validez) son los que cita la petición de auditoría. Las cuatro
combinaciones (`c3` = cell+risk, `cell_param`, `param_risk`, `param_param`) corren en `base`; `param_param` corre además
en `cost2`, `cap2`, `risk2`, `regime`, `onesided`, `missing10`. La tabla de familias (§7, "familias") compara la
expectativa, el Sharpe, el neto y la actividad fuera de bloque; la tabla de coeficientes da el valor optimizado por
nivel, fold y semilla con su dispersión (estabilidad), y la sensibilidad (±1 paso por gen) qué coeficientes mueven el
resultado.

## 9. Batería de intervenciones y contrafactuales (sólo bloque de decisión)

`tools/strategy_intervention.py` (5 pruebas). Toma los genes congelados en selección de cada corrida `base`/`perbar` y
**no** re-optimiza. Cada intervención se replica en el bloque de decisión con y sin cada mecanismo (ablaciones:
cierre temprano, veto NOP, puerta mejor-del-día, fórmula de volumen) y se reporta la **razón limitadora de pérdida**
= DD (o pérdida neta) con el mecanismo / sin el mecanismo, con IC bootstrap sobre extracciones × semillas; veredicto
TOLERANTE si el IC excluye 1 por arriba, NO_TOLERANTE si lo excluye por abajo, INCONCLUSO en otro caso.

| intervención | cómo se aplica (cabezas intactas) |
| --- | --- |
| choques 2σ / 3σ / 4σ contra la posición | en una barra al azar con posición abierta, el precio salta k·σ_t en contra (OHLC coherentes); 6 extracciones por semilla |
| cambio de régimen por empalme | trimestres reales 2021–2023 etiquetados por tercil de tendencia, empalmados tendencia → rango → tendencia con continuidad de precio; las tablas de resultados se recalculan con el motor sobre las barras empalmadas |
| spread × 5 durante 24 h | perfil de spread con ventana [t0, t0 + 24 h] × 5 |
| ausencias de 6–24 h | 8 tramos de observación ausente por extracción (camino HOLD-sin-observación) |
| ruido de las cabezas × 2 | latentes binarias a la mitad (ruido relativo × 2) y ruido de regresión doblado alrededor de su media, misma calibración |

La tabla de intervenciones (§7) está indexada por intervención × mecanismo, con la razón, su IC y el veredicto.

## 10. Jerarquía de escenarios y optimización en dos etapas

`tools/strategy_hierarchy.py` (5 pruebas). Niveles: **L1** régimen grueso (tendencia alcista / bajista / rango) ×
volatilidad (baja 0.7× / alta 1.6× la sd horaria real del bloque), sintético sobre el **calendario real** (misma
sesión, mismos fines de semana y rollovers) y real (los trimestres por terciles de §4); **L2** estructura fina dentro
del tramo (amplitud de oscilación 1.5 / 2.5 / 4 sd y densidad de pivotes ×0.5 / ×1 / ×2); **L3** eventos (los de §9:
choques, empalmes, spread, ausencias); **L4** calidad de las cabezas (las variantes de §4); **L5** ejecución (costos,
tope diario, corte, riesgo). Cada hoja del catálogo (`hier/CATALOG.json`) lleva la configuración del generador, el
comportamiento esperado y el **testigo** medido (signo de la deriva, razón de sd, densidad de pivotes finos por 100
barras, calidad lograda por bloque). Muestreo: **factorial completo** L1 × L2 (18 hojas) y L1 × L4 a la calidad de
operación 0.70 (36 hojas), más **hipercubo latino** de 24 hojas sobre nueve perillas (deriva, multiplicador de vol,
amplitud y densidad finas, calidad 0.55–0.90, costo ×1–3, tope 1–4, corte 12–24 h, riesgo 0.5–2 %): 78 hojas.
La verdad de cada hoja se construye con el motor real sobre su trayectoria (tablas exactas de resultados), y las
cabezas se simulan a la calidad de la hoja.

**Etapa A (sintética):** NSGA-II sobre los genes de la familia con dos objetivos: **media** de la compuesta del
propietario sobre las hojas y su **CVaR-10 %** (media del peor decil de hojas); los coeficientes c_k se recalibran con
la primera generación. Salida: frente de Pareto y **región robusta** = puntos a ≤ 5 % del mejor fitness medio global
**y** a ≤ 5 % del mejor en cada régimen L1 → caja [mín, máx] por gen (`hier/STAGE_A_<familia>.json`, archivo completo
`..._archive.json.gz`, testigos `..._leaves.json`).

**Etapa B (real):** la misma optimización en el bloque de selección real de cada fold con los genes acotados a la
caja de A (`--bounds-from`), puntuada una vez en el bloque de decisión, a calidades 0.62 / 0.70 / 0.80 y 3 semillas.
Se reporta (§7, "dos etapas") el desplazamiento de parámetros A → B y la ganancia o pérdida fuera de bloque de la
re-optimización frente al óptimo de A replicado en el mismo bloque de decisión, con IC bootstrap.

## 11. Parámetros congelados por banda de calidad (contrato sellado)

`contracts/STRATEGY_PARAMS_FROZEN_v1.json` (sha256 del cuerpo dentro del archivo y en §7). Para cada familia y cada
banda de AUC [0.50, 0.55), [0.55, 0.60), …, [0.90, 1.00] la mediana sobre semillas y folds de los genes de la Etapa B
(cuantiles C3, observe_bars y coeficientes) de las corridas cuya **AUC lograda en el bloque de calibración** cae en la
banda, recortada a la caja de A; una banda sin corrida toma el nivel más cercano y lo declara (`source`).

**Regla de banda (la que usa el evaluador SBFS sin modificar):** por subconjunto, se mide la AUC **fuera de fold** de
las dos cabezas abridoras TP-primero en el bloque de **calibración** (media de los dos lados), nunca en el de
decisión; esa AUC elige la banda y el evaluador aplica sus genes tal cual (los cuantiles se convierten en umbrales
absolutos sobre el propio bloque de calibración del subconjunto por el mapa de genes congelado; coeficientes y
observe_bars se usan directamente; el corte es el del contrato de ejecución, 24 h). El evaluador no optimiza nada.

## 12. Lo que los barridos expusieron y se corrigió (registro)

| hallazgo | corrección | dónde |
| --- | --- | --- |
| la puerta del contrato (observar 1 señal) deja la actividad fuera de bloque en 1.6–1.8/semana **incluso con el oráculo** (inválida por la restricción de 2/semana) | variante de puerta `perbar` (primera señal válida, costo de la barra en la EV); las dos puertas se reportan y se comparan | `PolicyC3PerBar`, `--gate` |
| los chequeos de corrección de tamaño y protección fallaban bajo la variante `full_f3` | chequeos conscientes de la variante (unidades por riesgo, desplazamiento ask del lado corto) | `tools/expert_guided_trading_report.py` |
| en la capa de volumen el tamaño se reiniciaba tras cada paso (83/160 chequeos fallidos en el humo) | `percentual_order_size` persiste hasta la siguiente entrada | `rollout_with_volume` |
| el empalme de régimen no tenía donantes (el año de decisión no tiene trimestres de rango) | donantes de todos los trimestres 2021–2023 | `quarter_tags(cx, None)` |
| el testigo de densidad fina no era monótono con el nivel L2 | amplitudes 1.5 / 2.5 / 4 sd y umbral de pivote 2 sd | `strategy_hierarchy.fine_pivot_count` |
| el sondeo de factibilidad 2^k del optimizador es inviable con 26 genes | el contrato FULL declara un espacio de genes no cuantil y salta el sondeo | `contract_full`, `freeze_full` |
| la ausencia de observación por bloque de un día no se distinguía del ruido | prueba a través de días | `tests/test_strategy_output_simulation.py` |
