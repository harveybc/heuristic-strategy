# Respuestas a preguntas pendientes

## Pregunta 1 — Método de causal inference

No es un solo método, es un pipeline de 6 métodos complementarios implementado en el repo `causal-inference/causal_regime_analysis_v2.py`:

1. **PCMCI+** (Tigramite, `RobustParCorr`) — descubrimiento causal lag-específico con control de autocorrelación, tau_max=10 (40h), transformación no-paranormal para colas pesadas
2. **RPCMCI** (Tigramite) — grafos causales dependientes de régimen: descubre simultáneamente 3 asignaciones de régimen + estructura causal por régimen
3. **ICP** (Invariant Causal Prediction) — test de Wald para estabilidad de coeficientes entre regímenes
4. **CausalForestDML** (econml) — efectos de tratamiento heterogéneos por régimen, 5-fold cross-fitting, `bb_position` como tratamiento
5. **Transfer Entropy** — flujo direccional de información: feature→return vs return→feature
6. **Sensitivity Analysis** (DoWhy) — placebo + confounders aleatorios + estabilidad de subsets

Los 6 métodos producen un **score causal compuesto por feature**. Resultados clave:

- `bb_position` → **CORE** (score=5, invariante en todos los regímenes)
- `atr_ratio` → **CORE** (score=4, ATE robusto=+0.020)
- `ema_alignment` → **LEADING** (única feature con Transfer Entropy positiva)
- `adx`, `di_spread` → **NOISE** (se usaban en V1, eliminados en V2)

**La conexión con el CNN es INDIRECTA/offline**: el análisis causal produce rankings de features, esos rankings se usaron para rediseñar `plugin_regime_wfo.py` en heuristic-strategy. V1 usaba `adx/di_spread` (noise), V2 usa `bb_position/atr_ratio/ema_alignment` (causalmente validados). El CNN y el causal inference son pipelines separados en runtime — no hay dependencia directa.

---

## Pregunta 2 — Tipo de salida del CNN

**Direction** (sigmoid → P(up)). Específicamente:

- `predictor_plugin_direction_cnn.py`: `Dense(1, activation="sigmoid")`, loss=`BinaryCrossentropy`, signal_type=`"direction_long"`
- Arquitectura: Conv1D stack (causal padding, stride=2) → head Conv1D → BiLSTM → sigmoid
- Lo consume `plugin_direction_atr.py` en heuristic-strategy vía HTTP API de prediction_provider: `P(up) > threshold → buy`, `P(up) < (1-threshold) → sell`, con TP/SL basado en ATR

También existe variante **binary** (`predictor_plugin_binary_cnn.py`) con arquitectura idéntica pero signal_type=`"buy_entry"` — predice si el precio cruzará TP antes que SL.

---

## Cómo se conecta todo en la práctica

```
causal-inference (offline)           predictor (training/inference)    heuristic-strategy (trading)
───────────────────────             ────────────────────────          ──────────────────────────

PCMCI+/RPCMCI/ICP/DML/TE/DoWhy     CNN direction model               plugin_direction_atr
       │                            (sigmoid → P(up))           ───→  consume P(up) vía API
       │ scores causales                                              para señales entry/exit
       │
       └─ bb_position=5 ────────────────────────────────────→   plugin_regime_wfo
          atr_ratio=4                                           clasifica régimen usando
          ema_alignment=LEADING                                 features causalmente validadas
                                                                (reemplazó adx/di_spread de V1)
```

---

## Dato importante para el plan

Las estrategias que sobrevivieron Phase 5.5 (pure_mr, tsmom, dual_momentum) **NO usan nada de esto** — son puramente rule-based sin predictor. La oportunidad híbrida es precisamente conectarlas: usar el CNN direction como filtro de régimen sobre las reglas base, o usar `plugin_regime_wfo.py` (que ya consume los features causales) para filtrar cuándo activar/desactivar las estrategias base.

---

## Sobre la distribución de máquinas

De acuerdo con tu análisis. Omega NO debe estar en el critical path de training. Un detalle: Omega es la única con el conda env `tensorflow` completo con todas las dependencias del pipeline (feature-eng, preprocessor, predictor, prediction_provider, heuristic-strategy). Dragon y Gamma usan system Python 3.13.7 — asegúrate de que las dependencias estén instaladas allá antes de lanzar training.
