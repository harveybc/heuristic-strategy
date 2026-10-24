# Ronda limpia del aprendizaje guiado por ZigZag (diseño para aprobación del propietario)

Estado: **PREPARADA, NO LANZADA.** El runner (`tools/expert_guided_clean_round.py`) rechaza
emitir la cola mientras falte cualquiera de dos puertas: (1) la selección v4 sobre las stores
corregidas cerrada; (2) `RELEASE.json`, que sólo escribo cuando el propietario confirme que
datos, selección, codificación y extracción (incluida la última parte de Musashi) están
cerrados, con la regularización ya declarada desde el diagnóstico de brecha.

## 1. Inventario del camino de entrada (auditado en el código, 2026-10-22)

Lo que llega al modelo, en orden, y qué pasa con cada operación heredada:

| Operación | Dónde | En la ronda limpia | Razón / evidencia |
| --- | --- | --- | --- |
| z-score móvil (256 filas) | `data.causal_rolling_windows`, default de `load`/`build`/E3-M | **PROHIBIDO** (el runner rechaza `rolling_z`) | orden del propietario; además, sobre las mismas filas no da ventaja medible (−0.86 vs −0.98) |
| z-score fijo con constantes de TRAIN | `data.fixed_train_windows` | **SÍ, para toda columna** | orden del propietario; constantes en `SCALER.json`, recarga rechaza si no coinciden |
| Regla de estacionariedad (ADF/KPSS en TRAIN → diferenciar) | `data.stationarity_decisions` | sí (`diff`; `keep` prohibido) | evita niveles crudos como proxy de fecha (E3-M sobre el CSV heredado: TRAIN +27.7 k, fuera −0.76) |
| Transformación de cola (asinh / log) | productor de la store (`layer1_pool_h1.decide`) | sí, **sólo la decidida por la métrica**, ya aplicada en la store | corregida hoy: el asinh en unidades nativas era la identidad para magnitudes ~1e-3 (§2) |
| Recorte (clip) de entradas | — | **no existe** en el camino de mercado | verificado: ningún clip/winsor entre store y ventana |
| Relleno de NaN / interpolación | — | **no existe** | una ventana con un valor no finito se excluye y se cuenta |
| Estado del agente: PnL no realizado (recortado ±10σ), duración (≤5 d), horas a cierre (≤5 d), etc. | `env.state` | **NO entra** (`--agent-state none`; el runner rechaza `full`) | orden del propietario: sin realimentación; `minimal` = posición + puede_entrar (acotadas, sin recorte) sólo como control declarado; el simulador conserva todo su estado para imponer el negocio |
| Pesos de clase 1/√frecuencia | `e3m.fit_module` | sí, declarado | etiquetas muy desbalanceadas (pivotes raros); frecuencias registradas en el ledger |
| Límites de SL/TP (0.25–3σ / 0.5–3σ) y nivel efectivo del broker | `env`, etiquetas | sí | reglas de negocio (SL y TP efectivos en toda orden) |
| Codificaciones alternativas (one-hot, sesiones, rel24, rel480) | etapa `encoding` | **no adoptadas** | ganancias ≤ 0.02 nats, inconsistentes; el brazo consumidor con ellas no mejoró (−1.38 vs −1.13) |
| Columnas de control `raw.*` | selección (brazo P_raw) | **prohibidas** en listas consumidoras | son controles de la selección, no entradas |

## 2. Defectos de datos encontrados y corregidos HOY (antes de cualquier experimento)

1. **Regla de asimetría sin escala.** `asinh(x)` en unidades nativas no transforma valores del
   orden de 1e-3: 38 de 50 columnas positivas con |asimetría| > 2 quedaron sin tratar
   (rangos y volatilidades realizadas de EURUSD — incluida `RAW_LOG_RANGE` de core15, 3.09 → 0.10
   con log —, volatilidades de los cruces y BTC, energías wavelet, amplitudes de Hilbert, potencias
   espectrales). Corrección (predictor @46af988e, 6 pruebas): orden declarado ASINH (se mantiene
   donde ya funcionaba) → LOG (TRAIN estrictamente positivo, con control de dominio) → ASINHS
   `asinh(x/s)`, `s` = mediana TRAIN de los positivos (congelada); probabilidades/entropías de
   régimen nunca se transforman.
2. **La regla medía la asimetría del candidato sobre toda la serie** (incluía 2023–2024).
   Ahora sólo TRAIN.
3. **La variante `log1p` de la etapa de codificación era nula** por la misma causa de escala;
   retirada (la escala la decide la regla de preprocesamiento).
4. Stores reconstruidas como revisión nueva, sin sobrescribir la v3: capa 1 `8243f43f…`,
   capa 2 `53616616…`, catálogo 410 filas, pool fusionado `32edb424…`; perturbación futura
   superada en todas las familias. **Selección v4 relanzada desde cero** sobre ellas (identidad
   nueva; nada de la v3 se reutiliza).
5. **ZigZag: el parámetro Deviation no actúa** en la réplica fiel del ZigZag de MetaTrader (el
   candidato sólo se registra cuando la barra es el extremo de la ventana, distancia 0): los
   pivotes dependen de Depth y Backstep. La desviación "optimizada" de 40.7 pips nunca tuvo
   efecto. Los expertos múltiples varían la profundidad.
6. **(2026-10-23) El `MANIFEST.json` del pool fusionado `_s` salió sin la clave `features`** (el
   script de fusión escribió sólo `columns`); el cargador de E3-M y el verificador de la ronda la
   exigen, así que todo consumo de la store `_s` habría fallado en la primera carga. Reparado en la
   store (manifiesto `10658d60…`, parquet intacto `32edb424…`, 316 features = columnas menos
   `timestamp`, convención del pool v3) y en el script de fusión; el paquete T2 de Musashi lleva la
   instrucción de reparación porque no puedo escribir en su directorio. El `check` de la ronda pasa
   salvo la regularización pendiente (fuente: T2) y la puerta `RELEASE.json` del propietario.

## 3. Señal del experto (dirección del propietario 2026-10-22; todo sólo para entrenar, nunca como entrada)

**Óptimo registrado:** ZigZag Depth 3 / Backstep 0 (Deviation es inerte en la réplica de MetaTrader).
**Demostradores más cortos que el óptimo**, para etiquetar los trades intermedios de menor ganancia
por trade (TRAIN, `evidence/SIGNAL_PROFILE.json`):

| demostrador | piernas/semana | pierna mediana | réplica bajo el contrato 1/día–1:50 (TRAIN) | confirmada causal |
| --- | --- | --- | --- | --- |
| óptimo, profundidad 3 | 27.2 | 30 pips | válido, 4.8 órdenes/semana, aciertos 97 % | −5 804 USD |
| profundidad 2 | 38.3 | 24 pips | válido, 4.9/semana, aciertos 90 % | −5 823 USD |
| reversión 10 pips | 87.0 | 23 pips | válido, 4.9/semana, aciertos 96 % | −4 906 USD |

Profundidad 1 es degenerada (210 pivotes) y queda descartada por dato. La reversión clásica fija un
pivote cuando el precio retrocede el umbral desde el extremo y se confirma siempre en una barra
posterior (sin suponer el orden dentro de la barra). Multi-experto = cada fila plana recibe la
etiqueta de cada demostrador cuya etiqueta es final dentro de su periodo (consenso; purga por experto).

**Objetivos auxiliares retrospectivos** (cabezas acotadas, Huber, máscara por objetivo y purga por soporte):
- al abrir: la mayor excursión favorable de una entrada en la siguiente apertura **antes del próximo pivote óptimo**
  (≤ 72 barras, ≤ cierre forzoso), en σ causal, neta de costo — cuánto podía ganar como máximo;
- al cerrar: la mayor ganancia adicional desde el cierre actual hasta el mismo horizonte — cuánto más podía ganar
  frente a cerrar ahora.
Rango de la cabeza [−0.1, 3.0] σ: cubre ≥ 99 % de TRAIN (p99 2.1–2.2).

**Contrato de ejecución: el del replay de Musashi** (máximo 1 entrada/día, apalancamiento 1:50,
rel_volume 0.10, capital 10 000) en demostraciones, rondas DAgger, elección del árbitro y todo periodo
reportado — el mismo de la línea SAC, así las dos políticas son comparables. Con los mismos pesos E3-M,
ese contrato redujo los drawdowns de 2024 a la mitad o menos.

## 4. Modelo (pequeño, cada valor con evidencia)

- **Extractores**: 2 unidades por extractor; dos variantes: uno por característica (registrado) o
  **uno por familia de la selección** (`--groups`, grupos de `FEATURE_GROUPS_<lista>.json`), que comparte
  extractor entre entradas afines.
- **Núcleo**: ancho registrado, **2 capas** convolucionales causales → campo receptivo 31 ≥ ventana de 24.
- **Cabeza**: 16 (la única coordenada promovida por la escalera de capacidad).
- **Ventana**: 13 observaciones cada 2 barras (control heredado) vs **24 barras consecutivas** (la de la
  sonda temporal de la selección; no salta barras).
- **Regularización**: CANDIDATA fuerte (dropout 0.3, wd 1e-2) por la regla declarada sobre el diagnóstico de brecha de Musashi
  (brecha media 2.20 vs 3.13), con salvedades (no uniforme sin realimentación; medida antes de la corrección de escala).
  Se fija en la liberación.
- Conteo de parámetros por módulo impreso por `check` antes de entrenar.

## 5. Experimentos (finitos, decididos en el bloque interno 2023H2, contrato principal)

**Etapa A (8 brazos, sin realimentación, un experto):** {core15, estricta} × {extractor por
característica, por familia} × {ventana 13@2, 24@1}.
**Etapa B (4 brazos sobre el ganador de A):** + objetivos auxiliares; + demostradores más cortos {profundidad 2, reversión 10};
ambos; + estado mínimo (control). Regla: mayor Sharpe medio de INNER_A con las tres semillas válidas; empate → menos
parámetros; brecha TRAIN−INNER_A reportada siempre; 2024 se lee una vez dentro de cada corrida y la
regla nunca lo consulta. Listas e identidades: las de la selección v4.

## 6. Comandos

```bash
python tools/expert_guided_clean_round.py check  --spec docs/short_horizon/clean_round/CLEAN_ROUND_SPEC.json
python tools/expert_guided_clean_round.py emit   --spec … --stage A --out queue_A.sh          # rechazado con puertas cerradas
python tools/expert_guided_clean_round.py decide --spec … --stage A --results <out> --out DECISION_A.json
python tools/expert_guided_clean_round.py emit   --spec … --stage B --winner-from DECISION_A.json --out queue_B.sh
python tools/expert_guided_input_justification.py --feature-list … --catalog … --selection-dir … --store … --out …
python tools/expert_guided_demonstrators.py --admitted … --csv-dir … --out DEMONSTRATORS.json
```

## 8. 2026-10-23: instrucciones de Musashi integradas (sin lanzar nada)

| instrucción | estado |
| --- | --- |
| señal rev 2: preparar integración y pruebas sin entrenar | hecho: E3-M `--labels zigzag\|opportunity\|both` + `--closer-rule`, disponibilidad por confirmación del pivote, horizonte de una barra válido, perfil recalculado; `EXPERT_SIGNAL_PER_AGENT.md` rev 2.1 §7 |
| T2 con `--split-contract consolidated` explícito | hecho en mi copia de la cola y en `T2_UNITS_2026_10_23.json` (predictor, rama de respuesta): 8 unidades tipadas con comando efectivo, identidad y campos esperados; el resultado NO es comparable con el diagnóstico v3 (dev3y) |
| regla de regularización ejecutable | hecho: `CLEAN_ROUND_SPEC.json` lleva `candidates` {base, strong} y la regla no dominada con desempate declarado; `tools/expert_guided_clean_round.py regularize --results <T2 out> --out REGULARIZATION_DECISION.json [--apply]` la aplica (sólo brazos `agent_state=none`, soporte común verificado, `OBS_2024` jamás leído; `DEFAULT_BY_VALIDITY` nunca se aplica); pruebas con dominancia, desempate, invalidez y soporte distinto |
| manifiesto T3 con dependencia de T2 y reparto sin duplicados | hecho: `tools/expert_guided_t3_manifest.py emit` → `evidence/t3/T3_MANIFEST.json` (26 unidades: control + 25 grupos de la lista estricta v4; ids estables por orden del archivo de grupos; impares Musashi / pares + control Satoshi; variante por grupo: `pointwise` para calendario/eventos/COT/diarios/factores/fuerza y grupos de una columna, `per_feature` para el resto); `queue --decision REGULARIZATION_DECISION.json` se niega sin la decisión T2; extractor `pointwise` (kernel 1) implementado en el tronco con prueba de no mezcla temporal |
| no declarar el conjunto terminado antes de T3 y auditoría | hecho: `release_states` en el spec separa `data_features` (CANDIDATE_VERIFIED) de `representation` (PENDING_T3); `RELEASE.json` sigue sin escribirse |

## 9. 2026-10-23: T1 de Musashi anexado; verificador de T2

- **T1** (trabajo de Musashi, @4fc98f6d): perfiles TRAIN de core15 y la lista estricta, copiados y verificados
  por manifiesto en `evidence/t1/`; resumen por grupo de extractor y pares duplicados en `evidence/t1/T1_ANNEX.md`.
  Seis pares con |ρ| ≥ 0.99 cruzan grupos de extractor: T3 mide el tipo de extractor con esa información
  redundante presente. Nada se retira ni se veta.
- **Verificador** `verify-t2`: comprueba cada `E3M.json` contra el listado tipado; un bloque `labels` ausente del
  productor fijado se lee como ZigZag heredado, no como fallo; `regularize` aplica la misma lectura. El primer brazo
  sellado (`gap_core15_none_base`, sha `d07622dc…`, igual al recibo de Musashi) pasa.

## 10. 2026-10-23: T2 cerrado, regularización decidida, T3 v2 desde el perfil de cada grupo

- **T2** (trabajo de Musashi, 8/8, 0 fallos): los ocho resultados se autentican contra sus recibos
  (`verify-t2`): identidad de la configuración (sha256 del JSON con claves ordenadas), fuente E3-M fijada igual al blob
  de 186b131 (que no emite `labels`), comando efectivo e identidad tipada recalculada, regularización, estado, lista,
  columnas consumidas, soporte y el hash de cada archivo sellado (E3M, scaler, sonda, 12 checkpoints). Cierra C01
  (ausencia sólo con productor autenticado; null, tipos inválidos y productores desconocidos se rechazan) y C02
  (el mismo resultado bajo otra unidad se rechaza). `evidence/t2/T2_VERIFY.json`.
- **Regularización = base** (dropout 0.1, wd 1e-4) por la regla congelada: ambos ajustes no dominados (el fuerte sólo
  gana en dispersión entre semillas de la lista estricta), el desempate 1 no alcanza (diferencia 0.67 < dispersión
  agrupada 0.75), el 2 elige la menor brecha. `regularize` exige ahora la verificación y que cada brazo sea el archivo
  verificado. `evidence/t2/REGULARIZATION_DECISION.json`; aplicada al spec.
- **T3 v2** (`tools/expert_guided_t3_manifest.py`): el manifiesto anterior asignaba un tipo por nombre de familia; ahora
  cada grupo multi-columna prueba `pointwise` y `per_feature` contra el control `conv`, y cada grupo de una columna
  prueba `pointwise`; la expectativa de cada grupo se escribe ANTES de cualquier resultado a partir de su perfil T1
  (memoria: persistente / casi blanca / intermedia; redundancia interna: alta / media / baja) y sólo se reporta la tasa
  de acierto; la decisión por grupo es medida y congelada (Sharpe INNER_A más allá de la dispersión agrupada de
  semillas, brecha no peor más allá de su dispersión, mismo soporte, tres semillas válidas; si no, el control se queda);
  una unidad de combinación confirma la suma de las elecciones. 43 unidades (22 Satoshi, 21 Musashi), ventana 13/2 y
  modelo de la ronda (2/2/16) fijos. `evidence/t3/`.

## 11. 2026-10-23: T3 v2 cerrado (43 + C), representación decidida, receta de consumo compartida

- **43 unidades completas**, 22 mías (4090) y 21 de Musashi (5090 y 4070 tras un traslado por presión de memoria,
  sin cambios científicos); una unidad mía (36) se repitió completa tras un error de la GPU, con el intento caído
  conservado aparte. Los 44 `E3M.json` (43 + C) están en `evidence/t3/results/` con el mismo soporte y las mismas
  116 columnas; Musashi verificó mis 22 por su cuenta (`musashi_delivery/`).
- **Decisión por grupo** (`T3_DECISION.json`, regla congelada; mi ejecución y la de Musashi dan el **mismo hash**
  `edb331ef…`): `cot.eur` → un extractor por columna; `gmm/logret_1+rv_24+log_range` y `fracdiff/log_close` →
  puntual; los otros 22 grupos conservan la convolución causal. Las expectativas del perfil T1 acertaron 9 de 17:
  se reporta, no decide.
- **Unidad C** (combinación de las tres elecciones; archivo de grupos `a9d878cf…`, idéntico al que emite mi
  `combine`): entrenada por Musashi en la 5090; regla de combinación **SELECCIONADA** (Sharpe INNER_A 1.558 frente a
  0.780 del control, dispersión agrupada 0.725; brecha 1.128 frente a 1.195). Reproducción exacta de sondas (516 + 12
  checkpoints) y de las métricas de trading del control y de C en sus CPU originales (Musashi).
- **2024, leído una vez, no es un holdout**: control −0.760, C −0.125 de Sharpe medio; la única semilla positiva de C
  en 2024 es inválida por la regla de actividad (1.5 órdenes/semana) y no se presenta como ganadora. Nada se ajustó
  sobre 2024; TEST cerrado.
- **Receta de consumo compartida** `CONSUMER_RECIPE.json` (`tools/expert_guided_consumer_recipe.py`, derivada del
  scaler sellado de C, del archivo de grupos y de las decisiones; rechaza columnas, grupos o digests que no
  coincidan): columnas en orden con sus transformaciones, constantes TRAIN del escalador, grupos y tipo de extractor,
  modelo y regularización, contrato de negocio, particiones e identidades. Es una descripción para los dos
  consumidores, no una liberación.
- **Lo que sigue**: solicitud final de auditoría conjunta (Musashi) y la palabra del propietario para `RELEASE.json`;
  después, por separado, la ronda limpia guiada por el experto (mía) y el contraste SAC sin experto (Musashi) sobre
  esta misma receta.

## 12. 2026-10-24: corrección A1–A4 ejecutada sobre la cadena completa (v5)

- **Selección v5** con productores locales a cada fold (3 882 unidades, 0 errores; 3 540 recalculadas, 342 reutilizadas con
  procedencia): listas 15 / 121 / 165; estricta 116 → 121 (+15 / −10); 11 familias cambian de estado; las codificaciones
  ganadoras cambian y **no** se adoptan (contraste afectado, pendiente). Testigos reales de aislamiento del ajuste:
  `evidence/a1/CHECK_consumer.json`, `CHECK_sel_K3_f1.json` (parámetros y valores anteriores idénticos; el ajuste
  contaminado mueve 270 509 celdas).
- **T2 v5**: 4/4 autenticados con dos productores → regularización **fuerte**; la ganancia interna de la estricta v4 era
  contaminada y desaparece.
- **T3 v5**: 45 unidades (23 mías, 22 de Musashi) autenticadas por mí con cinco productores; mi `decide` y `combine` dan
  los **mismos digests** que los de Musashi (`7291c1aa…`, `a9f553a7…`): `inv.raw`, `inv.momentum`, `hilbert/log_close`
  → puntual; `px.close_loc` → por columna; 21 grupos conv; expectativas 7/18. **C** (entrenada por Musashi en la 5090):
  SELECCIONADA (INNER_A 1.392 vs 0.836, dispersión agrupada 0.713); replay exacto (12 checkpoints, trading). Lectura
  honesta: en 2024 (una lectura) C −1.011 frente a −0.572 del control; TRAIN 2.572 frente a 2.997.
- **Receta v2** `CONSUMER_RECIPE.json` (`recipe_5a8f2155…`) derivada de la cadena de recibos de la C nueva, con
  etapas aguas arriba / consumidor, ventana tipada y contrato de negocio; sonda de referencia
  `evidence/a1/PROBE_a1_reference.json`; la v1 queda como `CONSUMER_RECIPE_v1_contaminated.json`.
- **Testigos contra el cierre de Musashi** (su ítem 2): 18 casos sobre copias (manifiesto, grupo, soporte, pesos,
  escalador, resultado, replay viejo / incompleto / inexacto): todos rechazados con su código; mi primer arnés era
  inválido para los tres de replay (la copia carecía del estado de réplica y su finalizador salía antes), queda
  registrado y corregido. `evidence/a1/CLOSURE_ATTACK.json`, `closure_attack_witnesses.py`.

## 13. 2026-10-24: receta aceptada; lanzador ligado a la receta; lanzamiento EN ESPERA por diseño multi-resolución

- Musashi: `ACCEPT_SHARED_RECIPE_FOR_DEVELOPMENT` (predictor @1e507f14): receta `recipe_5a8f2155…` como base compartida de
  desarrollo (no óptima ni rentable por demostrar); paridad exacta de la sonda en su consumidor SAC; corrigió el
  exportador (replay ligado a la unidad y a los módulos; integrado como merge `8208dd6`); su receta resultante es idéntica.
- Lanzador reparado (`tools/expert_guided_clean_round.py`): cada brazo consume `--recipe`; se rechazan flags que la
  contradigan y cualquier desvío de store / modelo / regularización / partición; la puerta `RELEASE.json` del lanzador
  antiguo queda sustituida por `LAUNCH_EXPERT_GUIDED.json` (palabra del propietario para esta línea); `check` prueba
  que el cargador por receta reproduce la sonda de referencia (tensor `f087d1a7…`): sin reajuste ni transformación implícita.
- Spec v2 `CLEAN_ROUND_SPEC.json` (`clean-expert-guided-a1-20261024`) y `EXPERT_GUIDED_RUN_SPEC.md`: 7 brazos A + 2 B que
  varían sólo tratamientos de entrenamiento (etiquetas ZigZag / opportunity / both, cerrador, demostradores cortos,
  cabezas auxiliares, control de estado mínimo). Paquete fijado en el host del 4090 (`PINS.sha256` `59ff4552…`).
- **Propietario 2026-10-24: no lanzar todavía.** Antes hay que definir el ZigZag **multi-resolución** (varios periodos
  decrecientes por debajo del óptimo, la máxima resolución posible) y sus métricas, porque el óptimo deja muy pocos
  ejemplos para abrir-compra / abrir-venta / cerrar-compra / cerrar-venta y sobre todo para NOP. Diseño y medidas en
  `ZIGZAG_MULTIRESOLUTION_DESIGN.md` (siguiente).
