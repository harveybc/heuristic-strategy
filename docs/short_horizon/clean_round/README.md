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

## 3. Demostradores ZigZag (sólo entrenamiento, nunca entrada)

| profundidad | pivotes (serie admitida 2020–2024) | órdenes/semana, réplica retrospectiva en el bloque de ajuste 2020-01 → 2023-07 | réplica confirmada causal, mismo bloque |
| --- | --- | --- | --- |
| 3 (primario) | 7 094 | 15.9 | −9 783 USD, Sharpe −1.47 |
| 6 | 3 677 | 10.1 | −9 824 USD, Sharpe −1.43 |
| 12 | 1 961 | 5.2 | −9 216 USD, Sharpe −0.82 |

Los tres son válidos para el negocio en retrospectiva; ninguno sobrevive a la confirmación causal.
Toda la ventaja está en **anticipar** el pivote: es la tarea `opener3`, la que la selección encontró
con señal fuerte (0.44 vs 0.70 nats). Multi-experto = cada fila plana recibe la etiqueta de cada
demostrador cuyo soporte de etiqueta termina dentro de su periodo (consenso; purga por experto).

## 4. Modelo (pequeño, cada valor con evidencia)

- **Extractores**: 2 unidades por extractor; dos variantes: uno por característica (registrado) o
  **uno por familia de la selección** (`--groups`, grupos de `FEATURE_GROUPS_<lista>.json`), que comparte
  extractor entre entradas afines.
- **Núcleo**: ancho registrado, **2 capas** convolucionales causales → campo receptivo 31 ≥ ventana de 24.
- **Cabeza**: 16 (la única coordenada promovida por la escalera de capacidad).
- **Ventana**: 13 observaciones cada 2 barras (control heredado) vs **24 barras consecutivas** (la de la
  sonda temporal de la selección; no salta barras).
- **Regularización**: PENDIENTE del diagnóstico de brecha (regla declarada en la spec).
- Conteo de parámetros por módulo impreso por `check` antes de entrenar.

## 5. Experimentos (finitos, decididos en el bloque interno 2023H2, contrato principal)

**Etapa A (8 brazos, sin realimentación, un experto):** {core15, estricta} × {extractor por
característica, por familia} × {ventana 13@2, 24@1}.
**Etapa B (2 brazos sobre el ganador de A):** + expertos {6, 12} (multi-experto); + estado mínimo
(control). Regla: mayor Sharpe medio de INNER_A con las tres semillas válidas; empate → menos
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
