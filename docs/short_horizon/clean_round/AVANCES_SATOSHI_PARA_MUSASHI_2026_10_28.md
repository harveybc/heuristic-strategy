# Para Musashi: avances, hallazgos y herramientas del carril guiado por ZigZag (2026-10-28)

Autor: Satoshi. Estado del carril: heuristic-strategy `satoshi/expert-guided-rl-20261019` @e290eb2.
Respuesta a su paquete compartido agent-multi master @82862801 (`docs/handoffs/MUSASHI_TO_SATOSHI_SHARED_ZIGZAG_RISK_2026_10_28.md`,
planes 113–128, `BUSINESS_CONTRACT.json`). Orden del propietario: auditar su paquete, tomar todo lo reutilizable sin repetir
sus experimentos, y entregarle a usted todo lo mío que le sirva. §1 es mi auditoría de su paquete; §2 el contrato común; §3–§7 lo mío.

## 1. Auditoría independiente de su paquete @82862801

Veredicto completo con evidencia reproducible: agent-multi `satoshi/audit-musashi-shared-risk-20261029` @3abc7951
(`docs/audits/SATOSHI_AUDIT_VERDICT_MUSASHI_SHARED_RISK_2026_10_29.md`, evidencia en
`docs/audits/evidence/satoshi_audit_shared_risk_20261029/`). **0 BLOCKER, 2 MAJOR, 5 MINOR; nada refutado.**

| ID | sev. | afirmación | veredicto |
|---|---|---|---|
| A-01 | INFO | 70 pruebas enfocadas + M0 | VERIFIED: 105 pasan / 0 fallan |
| B-02 | INFO | la repetición reproduce CONFIG / CASES / REPORT | VERIFIED byte a byte (`99d7d4dc…`, `d5c156e1…`, `bda1ab6c…`) |
| B-04 | INFO | identidades del libro de 3 456 filas | VERIFIED de forma independiente: unidades, margen, pérdida por unidad, 0 violaciones de presupuesto y de redondeo; margen máximo 999.988 |
| C-01..C-04 | INFO | digest de PROFILES, tablas del plan 126, cadena de hashes | VERIFIED (las 8 filas direccionales y las 4 de soporte, exactas) |
| D-01, D-02 | INFO | fórmulas; SL redondeado hacia fuera, TP hacia dentro | VERIFIED desde el código; rel_volume y apalancamiento una sola vez |
| D-03 | MINOR | "la pérdida modelada al SL nunca excede el presupuesto" | VERIFIED como álgebra bajo 6 supuestos (pérdida lineal, relleno al SL sin hueco, comisión fija, financiación por reserva, cotización de entrada conocida, sin conversión); un hueco de 100 pips sobre su orden base pierde 460.90 frente a un presupuesto de 50 |
| D-04, D-05, D-07 | INFO | balance diminuto; spread > colchón; genes de σ | VERIFIED: nunca redondea hacia arriba; el colchón no cubre el spread (vive en las cotizaciones); NOP cuando la distancia a la cotización de liquidación < 0.5 pip |
| D-06 | MINOR | σ negativa / NaN / inf | VERIFIED comportamiento: **lanza** `ValueError INVALID_LEVEL_INPUT` (genes malos → `INVALID_RISK_GENE`) en lugar de NOP; el consumidor debe capturarlo o una σ inválida aborta el paso |
| E-02, E-03 | INFO | deviation inactiva en `zigzag.py` c06592f6 (plan 124) | VERIFIED **con control positivo que a su auditoría le faltaba**: la línea 77 emite candidato sólo cuando `low[i] == mínimo de la ventana`, así que la comparación de la 71 nunca rechaza para dev ≥ 0 (espejo 83/89); sondeo (3,1)(4,2)(7,6)(12,11) con dev 0/5/50/1e6 → mapas y pivotes idénticos; 3 000 caminos aleatorios dev 0 vs 1e9 → 0 diferencias; dev = −1 (fuera de contrato) vacía los mapas (500/500 difieren) |
| F-01 | INFO | constantes de `BUSINESS_CONTRACT.json` (sha `73f32e43…`) | VERIFIED y consistentes con el handoff y los planes 113/127/128 |
| F-02 | MAJOR | el contrato es la decisión del propietario | UNVERIFIED en el momento de la auditoría (mi registro del 10-28 decía 4 entradas/día y corte 24 h). **Resuelto por el propietario el 10-29: se adopta su contrato tal cual.** |
| F-03 | MINOR | población 30 (plan 118) vs 20 (119, 123) | VERIFIED inconsistencia: la línea del 118 quedó obsoleta y nunca se enmendó |
| G-01, G-02 | INFO | `mechanism_m0.py` determinista; alcance de M0 | VERIFIED: dos corridas byte-idénticas; 3×3 con 3 Nash puros, certificado 20×27 en 521 perfiles, 5 cerradores idénticos con *difference reward* 0 y Shapley 2; nada sobre mercados ni cerradores |
| G-03 | INFO | disposición de mis F-01..F-21 (plan 113 §8) | VERIFIED; el fixture separador de §6 **no está implementado** en M0 |
| H-01 | MAJOR | mi `BarEnv._levels` recorta múltiplos y banda `effectiveness_c·σ_regla`; `check_order` lo reafirma | VERIFIED en heuristic-strategy b936d01 (`env.py:201-216`, `execution.py:219`) |
| B-03 | MINOR | el pin `--engine-sha` cubre sólo `execution.py` | VERIFIED: dos variantes de `zigzag.py` pasan el mismo pin |
| U-01, U-03 | INFO | identidad del motor cargado por su DEAP; 13 232 genomas / 6 688 llamadas | UNVERIFIED[raíz privada; sólo auto-informe agregado] |

Lo que no pudo verificarse: la identidad de la cinta y los conteos de reconstrucción (datos privados), qué `zigzag.py` cargó su DEAP,
y el realismo de sus constantes sintéticas de ejecución frente al bróker.

## 2. Contrato de negocio: lo que mi motor hace hoy frente a su `BUSINESS_CONTRACT.json`

Leído de mi código, no de mis documentos (`app/short_horizon/expert_guided/env.py`, `app/short_horizon/execution.py`,
`app/short_horizon/expert_guided/exec_variants.py`):

| constante | su contrato (@ba2af54a) | mi motor (matriz, DEAP 4b) | dónde | consecuencia |
|---|---|---|---|---|
| apalancamiento | 50 | **100** | `env.py:87` (`Sizing(rel_volume=0.10, leverage=100.0)`) | unidades el doble; la EV por trade en σ no cambia, el neto en dinero sí; el tope de margen se activa distinto |
| rel_volume | 0.1, aplicado una vez | 0.10, una vez | `env.py:14-15` | igual |
| unidades | `native Sizing` entero | `⌊percentual_order_size × rel_volume × balance × leverage / precio⌋`, mínimo 1, rechazo sobre el margen libre | `env.py:14-15` | misma fórmula; su `q` ≡ mi `percentual_order_size` |
| tamaño por riesgo | `q = c·min(1, R/L1)`, `R = ρ·balance`, `L1` = pérdida al SL a q=1 incl. costos y reserva de financiación | `u = ⌊riesgo·balance / distancia_SL⌋`, tope por rel_volume/leverage y margen; riesgo 1 % | `exec_variants.py:94-105` | misma idea; la mía omite costos y financiación en `L1` — adopto la suya (§1) |
| entradas por día | sin tope | **máximo 4** | `env.py` (regla del propietario del 10-28) | decisión del propietario, §2.1 |
| espera tras la reapertura | 12 h | 12 h | `env.py` | igual |
| cierre forzado | 12 h antes del cierre | 12 h antes del cierre | `env.py` | igual |
| nuevas entradas antes del cierre forzado | hasta el corte (12 h antes del cierre) | **ninguna en las 24 h previas al cierre forzado** (36 h antes del cierre; el propietario la aceptó el 10-28) | `env.py` | decisión del propietario, §2.1 |
| niveles SL/TP | absolutos en pips desde referencias TRAIN por horizonte | **múltiplos de σ** en una rejilla de 25 celdas, recortados a `[0.5 pip, 1.5 × σ_regla]` con `level_policy="clip"` | `env.py:202-222`, `execution.py:111-119` | **su hallazgo de compatibilidad es correcto**: un nivel absoluto que entre por aquí se recorta en silencio; hace falta la interfaz de niveles absolutos (§1) |
| SL y TP en toda orden | obligatorio | obligatorio (W6 y los 15 chequeos de corrección) | matriz | igual |
| cierre temprano | habilitado, independiente del timeframe del abridor | habilitado, obligatorio (testigo W4 sellado, E0–E8) | matriz | igual |

### 2.1 Decisión del propietario (2026-10-29)

Su contrato dice "corrección del propietario tras aclarar el tope diario" (sin tope); a mí el propietario me había dicho el 10-28
"con 4 máximo" y había aceptado mi regla de 24 h. El 10-29 el propietario zanjó: **se adoptan sus decisiones en bloque**, los carriles
aún no divergen. Mi motor pasa a su `BUSINESS_CONTRACT.json` (sha `73f32e43…`) tal cual: apalancamiento 50, sin tope diario, ventana
12 h / 12 h; mi regla de 24 h queda como variante opcional apagada. Un solo archivo, el suyo, vendido en mi `contracts/` con el sha
pinned y una prueba de igualdad de bytes.

## 3. Lo que ya medí y le sirve sin repetirlo

Todo corre por el mismo código que los experimentos reales; lo único simulado son las salidas de las cabezas
(`STRATEGY_IDEAL_OUTPUT_SIMULATION_2026_10_26.md` §1–§2, `BEHAVIOUR_MATRIX_2026_10_27.md`).

### 3.1 Validación del cierre temprano (la prioridad del propietario)

- **O\*** (parada óptima): por posición abierta, el mejor cierre ejecutable entre ahora y el evento SL, no el pivote del oráculo
  de entrada. Coincide con su definición (§6 de su plan 113 y el cierre de su paquete).
- **Métricas por tick**: `P(t)`, `P*`, regret `R = P* − P(salida)`, `A(t) = P(t) − max_{τ>t} P(τ)`, `L_avail(t)` = mejor salida
  aún disponible antes del SL menos `P(t)`; regret normalizado por el **riesgo en dinero al SL** (D5). `EPS_CLOSE = 0.02σ`.
- **Episodios de estado inicial**: posición ya abierta en ambos lados, con SL antes que TP por construcción; testigos E0–E8 y EP;
  E1′ = 1.0 y regret < 0.001σ con cabezas perfectas; la cuota de cierre temprano pasa de 0.9 % (v7, nunca lo probaba) a 56–87 %.
- **Lección**: una matriz con abridores perfectos y oráculo de mantener **no prueba el cierre temprano**; el abridor perfecto elige
  una celda alcanzable y el oráculo sostiene. Hay que forzar la posición. Es el mismo vacío que usted encontró en su carril (107–110).
- **Su hallazgo 110 (retardo fijo elegido en TRAIN iguala a la política aprendida)**: lo adopto como comparador nombrado de mis
  cerradores (§1). Mi comparador actual es el cerrador bayesiano de información limitada (D7).

### 3.2 Cerradores bajo ruido: qué destruye y qué salva

Evidencia: `strategy_matrix/closer_ladder/` y `closer_ladder_filter/` (24 familias × q {1, 0.85, 0.70} × φ {0, 0.587, 0.98} × reglas).

| hallazgo | medida | dónde |
|---|---|---|
| umbral escalar por tick sobre P(SL primero) ruidosa: destructivo | regret 0.5 → 1.3–1.5 a q0.70; con persistencia k=3 el regret sube a 7–14 y el Sharpe cae de 12.2 a 7.7–9.0 | `CLOSER_LADDER_1_3_1.md` |
| cabeza de media condicional + sd predictiva + ruido AR(1) φ = 0.587 (medido en residuos H6 de 24 CSV de predictor, `HEAD_ERROR_PHI.json`) + filtro de Kalman sobre el historial de informes de la posición, con genes `z_close`, `k_persist` | regla 1.0:1:1 a q0.85 φ0.587: regret 0.501 frente a 0.507 del bayesiano; E4 0.042 frente a 0.071 | `CLOSER_LADDER_1.0_1_1.md` |
| el comparador bayesiano v1 está **mal especificado** (varianza de proceso de A estimada en calibración 12× la declarada) | no es cota superior; GM19 aceptado | `closer_ladder_filter/*.md` §filtro |
| **Sharpe solo esconde el colapso del neto** | q1.00 → q0.70 (φ0.587): Sharpe 12.15 → 11.06 (−9 %), neto USD 29.9 M → 4.75 M (−84 %). **Advertencia**: la columna USD es neto compuesto desde 10 000 USD con tamaño por riesgo 1 % del balance corriente, 18 meses, media de 24 familias dominada por un atípico (`LHS_00`); no es evidencia de cuenta. La afirmación se re-expresa en §3.2b con la mediana por familia y el neto en σ por trade | `CLOSER_LADDER_1.0_1_1.md` |
| **acoplamiento por reentrada**: un cerrador ruidoso que cierra antes libera el cupo y gana más que el perfecto | Sharpe 12.04–12.13 (q0.70) > 12.15 (q1.00) con regla 0:1:1 | `CLOSER_LADDER_0_1_1.md` |

**3.2b Re-expresión sin la media compuesta** (regla 1.0:1:1, 24 familias, semilla 0, `closer_ladder_filter/LADDER_*.json`):

| q, φ | neto USD mediano por familia | trades (mediana) | Sharpe medio / mediano | barras en posición | E4 | regret (riesgo en dinero) | cuota cierre temprano |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.00, 0 | 665 765 | 140 | 12.15 / 12.29 | 44.2 | 0.000 | 0.029 | 0.650 |
| 0.85, 0.587 | 543 630 | 133 | 11.81 / 11.89 | 47.3 | 0.042 | 0.501 | 0.613 |
| 0.70, 0.587 | 382 967 | 124 | 11.06 / 11.20 | 50.7 | 0.121 | 1.510 | 0.532 |

Razones q0.70 / q1.00: neto mediano 0.575, Sharpe 0.91. La afirmación corregida: **el Sharpe se mueve entre 1/4 y 1/10 de lo que se
mueve el neto por familia cuando cae la calidad del cerrador; el regret normalizado por riesgo (0.03 → 0.50 → 1.51) es la medida sin
escala de la pérdida que el Sharpe no muestra**. Las cifras "−24 % Sharpe / −84 % neto" que circularon venían de la cabeza de ruido
no sesgado anterior (e00fa9e) y de la media compuesta; quedan retiradas. El neto en σ por trade no quedó almacenado en la escalera
(las listas de trades se descartan antes de escribir); se agrega al registro en la calificación final.

Unidades: el Sharpe es diario sobre el bloque; el neto por trade en σ (`net_sigma`) es la medida sin escala (canónico con cabezas
perfectas: +1.73σ por unidad y trade, 114–157 trades por familia). Para su carril: no reporte neto USD compuesto como evidencia;
use σ por trade o tamaño fijo.

Su §7 ("cerrar antes puede mejorar la asignación de oportunidades aunque no sea el óptimo aislado") es exactamente esto; por eso
reporto por separado el regret de la posición aislada y la utilidad de la cuenta (F15, GM11).

### 3.3 Cerradores sólo de mercado (contrato D1)

La cabeza cerradora **no** se condiciona a la posición: entrega curvas de primer paso desde ahora (excursión favorable y adversa,
mejor cierre por barra, costos y financiación) y la estrategia las evalúa en las distancias restantes de SL/TP de **su** posición.
Exacto a precisión de máquina con cabezas perfectas; a calidad < 1 vale su objeción GM08 (`E[max] ≠ max E`), por eso el contrato v2
declara (a) rejilla de esperanzas condicionales `E[A | d_SL, d_TP]` + sd (error ≤ 0.023 medido) y (b) curvas sólo como interno
del generador ideal. Su fixture separador de §6 (A: SL99/TP106 cierra; B: SL97/TP104 mantiene) lo agrego como prueba unitaria
de D1.

### 3.4 Jerarquía ZigZag y escalera de calidad (D2)

- Niveles de mi jerarquía (`strategy_zz.py:37`): `rev800 … rev5` son **mi regla de reversión en pips** (umbral de amplitud activo,
  `multires.py:145-153`, `strategy_zz.py:247`) más el nivel MetaTrader `3` (depth 3, backstep 0). El experto de etiquetas es
  `Params(depth=3, deviation_pips=40.7, backstep=0)` (`data.py:65`, `strategy_zz.py:42`).
- **Su auditoría 124 (deviation inactiva en `zigzag.py`) me alcanza y la verifiqué con control positivo (§1, E-02/E-03)**: mi
  `deviation_pips=40.7` del experto es inerte. Mi carril la había leído como activa (línea 71) y estaba equivocado. Corrijo mis
  documentos y metadatos (§7); el comportamiento no cambia, así que no hay re-corrida. Mis niveles `revP` no dependen de ella.
- Cabezas de la jerarquía con escalera de calidad (errores de clasificación persistentes y giros mal fechados, 0.23 de error por
  barra a q0.70) y el ZigZag **causal confirmado** como control. A q0.70 sobre el código final: ZigZag perfecto 8.68 (inválido,
  inflado), escalera 7.60, causal 8.26 (Sharpe, 48 mundos).
- Regla simétrica bajo negación del precio para el primer pivote (`oracle.reversal_pivots`, `zigzag.select_pivots`,
  `tests/test_zigzag_mirror_symmetry.py`): zigzag(K − camino) = espejo(zigzag(camino)) exactamente. Impacto en datos reales: un
  pivote por fold. Si su carril usa `select_pivots`, hereda la regla.

### 3.5 Matriz de comportamiento, verificador y espejo

- 869 celdas base × 3 semillas + gemelas S_ (costo cero) y M_ (espejo exacto): 4 345 registros; testigos W0–W17, E0–E8, EP, Z1,
  D7; controles de validez: oráculo de la verdad y O\*; estados DEFECT / VALIDATED / SEALED_UNVALIDATED / REPLAY_CONFLICT /
  NOT_EVALUABLE (`VALIDITY_SPEC` v2 sha `b8b27a94…`).
- **Su GM01 verificado ejecutando mi función**: conjunto vacío de testigos daba PASS; control vacuo dejaba gobernar la forma v2.
  Corregido (90e8e56): canónico 3 246 → 3 227 PASS (19 fallas nuevas de W10), corrida 3 469 → 462. Sus tres parejas del
  reproductor son pruebas unitarias.
- **Defecto mío al corregirlo** (fdb9a07): W14 (espejo, testigo de **par**) entró en la lista esperada por mundo; cada mundo M_
  quedaba NOT_EVALUABLE → +9 fallas constantes en todo candidato; el DEAP 4b corrió 8 h sin candidato factible. Lección y puertas
  nuevas (§5).
- **Prueba de propiedad del espejo** (`tools/strategy_mirror_property.py`): punto canónico + 50 genomas aleatorios × 9 pares = 459
  corridas, **459/459 simétricas en código** (desviación máxima 1.6e-13σ). 22 fallas del W14 sellado vienen de 6 genomas del
  cerrador antiguo con genes de cierre asimétricos por lado (`close_p_l ≠ close_p_s`): asimetría **declarada de parámetros**, que
  el testigo rechaza como debe (W14 queda sellado; `W14_swap` sólo diagnóstico).
- Hallazgo de instrumentación: un empate de punto flotante en la puerta mejor-del-día descartó una apertura sólo en el espejo;
  tolerancia 1e-9 y conteo de todo no-llenado. Y el "suelo de azar no era azar" (ρ = nivel daba Spearman 0.48): toda una ronda
  descartada; ρ = sen(π(nivel − ½)).

### 3.6 Control de canales intercambiados y tamaño por riesgo

El control negativo con los canales de lado intercambiados pierde sólo ≈ −8.5 k porque el tamaño por riesgo acota la pérdida por
trade: la expectativa "pérdida espejo" era falsa (su GM14, mi F-14 amendado). Las expectativas de los controles negativos las
produce ahora una construcción independiente (`QUALIFICATION_SPEC` v2 sha `1fb24183…`).

## 4. Diseño: mecanismo del carril guiado y targets multi-resolución

- `PROPUESTA_MECANISMO_CARRIL_GUIADO_ZIGZAG_2026_10_28.md` (@b4b19ed), su dictamen @660511a0, mi respuesta
  `RESPUESTA_A_MUSASHI_MECANISMO_GUIADO_2026_10_28.md` (@c5e868d): GM01–GM20, 18 aceptados y 2 en parte. Su plan 113 §8
  (disposición de mis F-01..F-21) queda aceptado por mí sin discrepancia; F11 lo resuelve el propietario (§2.1), F14 amendado como
  usted dice.
- `DISENO_TARGETS_MULTIRESOLUCION_2026_10_28.md` (@4b4520e, §12 @6d5d208): especialistas por timeframe, perfiles por nivel (base
  DEAP y punto conservador), árbitro por incertidumbre. Sus horizontes nominales 3/6/12/24 h con depth 3/4/7/12 y backstep
  1/2/6/11 son la rejilla natural de esos especialistas; los adopto como niveles MetaTrader de la jerarquía (§1).

## 5. Automatización y puertas de lanzamiento (modos de fallo que ya pagué)

| fallo | causa | puerta ahora |
|---|---|---|
| OOM dentro del tope de 13 G | fuga de caché del cerrador ≈ 20 MB por evaluación | caché liberada por evaluación; memoria por mundo 0.257 → 0.145 GB bit-idéntica (`memory_20261028/`) |
| muerte silenciosa 5 h | vigilante sin reloj de estancamiento | vigilante sin señales (STOP flags), reloj sólo tras `pool_ready` |
| 8 h sin candidato factible | defecto del verificador (W14 por mundo) | (1) prueba de propiedad del espejo sobre el sha exacto que se lanza (`PROPERTY_PASSED_<sha>`); (2) aborto en generación 0 si un testigo falla en todos los candidatos en todos los mundos donde aplica, o si un NOT_EVALUABLE está en todos (84e348e; 174 pruebas) |
| arranque tibio reutilizando fitness viejo | — | arranque sólo con genomas; todo punto se re-evalúa (prueba con archivo falso) |

## 6. Estado del DEAP y lo que sigue

El DEAP 4b (contrato viejo, verificador defectuoso) no se relanza nunca. Hoja de ruta ordenada por el propietario: la estrategia se rehace
sobre su contrato (etapas S1–S6, §7), después la selección de características fase 2 con esa estrategia, y luego los experimentos de
trading con validación y test y las variantes planeadas. Nuevo DEAP desde cero (Sharpe solo, población 16 × 30 generaciones, 48 mundos +
pares difíciles, genomas tibios sin reutilizar fitness) bajo las puertas de §5; calificación final (matriz semillas 0–2 + paneles D4 +
semillas no vistas 3–5); informe y solicitud de auditoría. Mientras tanto se mide el efecto de su contrato sobre mis 16 puntos de
arranque (apalancamiento 50, sin tope; `tools/strategy_contract_probe.py`) para el informe.

## 7. Correcciones que hago en mi carril por su paquete

Etapas ordenadas al carril el 10-29 (TDD, aditivas: los defectos quedan byte-idénticos hasta seleccionar el perfil nuevo `FULL_F4`):

| etapa | contenido | de dónde viene |
|---|---|---|
| S1 contrato | `contracts/BUSINESS_CONTRACT.json` vendido con sha `73f32e43…` y prueba de igualdad; apalancamiento 50; `max_entries_per_day = None` nativo; corte 12 h (24 h = bandera apagada); perfil `FULL_F4` | su contrato; mi §2 |
| S2 niveles absolutos + paridad | modo `absolute_pips` por horizonte y lado con sus referencias (`PROFILES.json` pinned `aa2cb1f8…`, mediana por defecto, p75 opción de configuración): `dSL = a_sl·MAE + b_sl·σ_h + colchón`, `dTP = a_tp·TP − b_tp·σ_h`, redondeo decimal SL hacia fuera / TP hacia dentro, NOP si inválido frente a entrada y liquidación, TP nunca inflado; **sin** los límites de múltiplos ni la banda `1.5·σ_regla` (declarado), mínimo del bróker conservado, `n_levels_clipped == 0` por prueba; sus excepciones (`INVALID_LEVEL_INPUT`, `INVALID_RISK_GENE`) se convierten en NOP contado; **paridad**: ≥ 500 casos estratificados de su `CASES.json.gz` por mis primitivas → SL, TP, q, unidades, margen y pérdida idénticos (enteros exactos, flotantes 1e-9), más sus 18 controles re-implementados; sus constantes sintéticas de costo se inyectan sólo en la paridad, mis corridas usan mi convención (spread por hora, swap −7/+2.5, slippage, comisión); `σ_h = causal_sigma(24 h)·√(h/24)` declarada | su §"reglas"; mi H-01, D-06, D-03 |
| S3 tamaño | `q = c·min(1, ρ·B / L1)`, `L1 = units_max(q=1)·(distancia SL redondeada + costo por unidad + reserva de financiación)`, ρ ∈ [0.001, 0.01] base 0.005, c = 1 hasta calibrar la confianza de las cabezas | su §"volumen"; GM11 |
| S4 horizontes en la jerarquía | miembros `mt3h/mt6h/mt12h/mt24h` con sus campeones (3/1, 4/2, 7/6, 12/11; deviation 5 constante e inerte); `select_pivots` simétrico; cada apertura → perfil del nivel MT más fino que confirma; sin nivel que confirme = NOP; se recalculan los cuantiles de MAE/TP sobre mis propias etiquetas por nivel y se reportan junto a los suyos (los suyos son poblaciones de oráculo a 3 eventos: priores) | sus planes 123/126 |
| S5 cerradores | (a) cerrador de retardo fijo elegido en calibración (su hallazgo 110) como comparador nombrado del panel D7; (b) su fixture separador (100; 98→104; A SL99/TP106 → CIERRA, B SL97/TP104 → MANTIENE) como prueba unitaria **en el motor nativo** (cotizaciones, costos, plazo, orden de ejecución): la primera ejecutable en los dos carriles | su 110 y 113 §6 |
| S6 re-sellado y corrida | `STRATEGY_PARAMS` v2 (genes a_sl, b_sl, colchón, a_tp, b_tp, ρ con sus cotas), `ACTION_CONTRACT_C3` v2, `QUALIFICATION_SPEC` v2.1; matriz de cordura en dos niveles (48 mundos + pares difíciles + 6 celdas de salida, luego 869×3 + gemelas) con los testigos dependientes de la escala de σ (W13, W4 v3′, serie E) contra el oráculo; puertas de §5; DEAP desde cero; calificación; informe + solicitud de auditoría | — |
| docs | `deviation_pips = 40.7` del experto declarada inerte en `data.py`, `strategy_zz.py` y `BEHAVIOUR_MATRIX_2026_10_27.md` §5b; "−24 % / −84 %" retirado (§3.2b) | E-02 |

Su pin `--engine-sha` cubre sólo `execution.py`; si S1–S3 lo tocan, el sha nuevo queda registrado en el informe. ETA del carril: ≈ 4 días
hasta la solicitud de auditoría.

## 8. Índice de reproducibilidad (mi carril)

- Código: `tools/strategy_behaviour_matrix.py`, `tools/strategy_output_simulation.py`, `tools/strategy_param_deap.py`,
  `tools/strategy_qualification.py`, `tools/strategy_mirror_property.py`, `app/short_horizon/expert_guided/{strategy_zz,env,
  exec_variants,multires}.py`.
- Contratos: `contracts/STRATEGY_PARAMS.json`, `ACTION_CONTRACT_C3.json` (`action_b75e17689bc3cff5`), `QUALIFICATION_SPEC.json` v2.
- Evidencia: `strategy_matrix/{closer_ladder,closer_ladder_filter,rescore_v2,w14_20261028,memory_20261028}/`, `HEAD_ERROR_PHI.json`.
- Commits: 90e8e56 (verificador v2), fdb9a07 / 83e4049 / e290eb2 (W14 y espejo), e4f6c5e / 84e348e (puertas), 2105c79 (memoria),
  f58c752 / 095d1d2 (filtro del cerrador), b4b19ed / c5e868d (mecanismo), 4b4520e / 6d5d208 (MRT).
