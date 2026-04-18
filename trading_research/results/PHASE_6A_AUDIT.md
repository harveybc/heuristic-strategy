> **HISTORICAL** — Phase-level document. For consolidated project closure, see [PROJECT_CLOSURE.md](PROJECT_CLOSURE.md).

# Phase 6.A — Strategy × Predictor Combination Audit

**Date:** 2026-04-17  
**Status:** COMPLETE  
**Machine:** Omega (read-only audit, no backtests run)

---

## Executive Summary

Phase 3.5–5.5 tested **only oracle-free, rule-based strategies** (pure MR, TSMOM, dual momentum) through the rigorous evaluation harness (rolling windows, transaction costs, worst-2Y, regime breakdown, OOS validation, benchmark comparison). **Zero combinations of heuristic-strategy plugins + ML predictors were ever evaluated with Phase 5.5 rigor.**

However, ML-based combinations (CNN direction, ANN, ensemble, NEAT) **were** tested through a separate WFO/GA framework — and **all produced negative returns over 14 years**. The oracle noise sweep establishes that profitability requires prediction F1 > 0.91, while the best CNN achieves F1 ≈ 0.45. This gap is structural, not parametric.

The two regime plugins (`plugin_regime_wfo` with V2 causal features, `plugin_regime_adaptive` with GMM) have **never been evaluated in any framework**, making them the only genuinely untested candidates.

**Recommendation for Phase 6.B:** 3 combinations worth testing. The rest can be skipped with high confidence.

---

## A.1 — Strategy Plugin Inventory

### 5 Strategy Plugins in `heuristic-strategy/app/plugins/`

| # | Plugin | File | Consumes | Prediction Type | V2 Causal Features | Phase 5.5 Tested? |
|---|--------|------|----------|----------------|--------------------|--------------------|
| 1 | `plugin_long_short_predictions` | `plugin_long_short_predictions.py` | CSV regression price paths | Regression (hourly + daily absolute prices) | NO | Never |
| 2 | `plugin_api_predictions` | `plugin_api_predictions.py` | Prediction Provider API | Binary (buy/sell entry + exit) | NO | Never |
| 3 | `plugin_direction_atr` | `plugin_direction_atr.py` | Prediction Provider API | Binary direction (P(up) → buy/sell) + ATR-based TP/SL | NO | Never (WFO only) |
| 4 | `plugin_regime_wfo` | `plugin_regime_wfo.py` | Prices only (computes features locally) | None (reactive threshold-based regime classification) | **YES** — bb_position, atr_ratio, ema_alignment | **Never in any framework** |
| 5 | `plugin_regime_adaptive` | `plugin_regime_adaptive.py` | Prices only (computes features locally) | None (GMM nearest-centroid regime classification) | **YES** — same 3 features via K=9 GMM centroids | **Never in any framework** |

### Plugin Details

**plugin_long_short_predictions:** Reads CSV with `Prediction_h_{i}` and `Prediction_d_{i}` columns (multi-horizon regression forecasts). Computes ideal profit/drawdown from daily predictions for both directions. Enters when ideal_profit_pips ≥ profit_threshold with higher RR. 7 exit variants (A–G). Requires pre-computed regression predictions.

**plugin_api_predictions:** Sends datetime + cost parameters to PP API. Receives `buy_entry_binary`, `sell_entry_binary`, `exit_binary`. Fixed pip-based TP/SL. Position sizing from `bars_remaining × confidence`.

**plugin_direction_atr:** Same as api_predictions but with ATR-adaptive TP/SL distances. Sends ATR-derived TP/SL to PP. The ATR makes TP/SL responsive to current volatility.

**plugin_regime_wfo (V2 — causal):** Resamples 1h → 4h bars. Classifies regime using causally-validated features:
- `bb_position` (CORE, causal score=5) — Bollinger Band position
- `atr_ratio` (CORE, causal score=4) — ATR relative to its own average
- `ema_alignment` (LEADING, positive Transfer Entropy) — EMA stack alignment
- Threshold-based classification → 6 regimes → buy_reversal / buy_meanrevert / buy_trend / flat
- Entry only on regime transitions, confirmed by RSI + Stochastic K

**plugin_regime_adaptive (V3 — GMM):** Same 3 causal features but classified via K=9 GMM centroids (hardcoded from 15yr EURUSD 4h). Nearest-centroid in scaled feature space. Confidence filter rejects ambiguous assignments. Same 6 regime mappings and entry/exit logic as V2.

### 3 Script-Level Strategies (NOT plugins)

| Strategy | Implemented In | Plugin Exists? |
|----------|---------------|----------------|
| Pure Mean Reversion (z-score) | `phase5_5_corrective_audit.py`, `eurusd_mr_audit.py`, `track_a_mr_deploy.py` | `eurusd_mr_strategy.py` in LTS only (EUR/USD specific) |
| TSMOM (Moskowitz 2012) | `track_c_academic.py`, `phase5_5_corrective_audit.py` | **NO** |
| Dual Momentum (Antonacci) | `track_c_academic.py`, `phase5_5_corrective_audit.py` | **NO** |

---

## A.2 — Predictor Plugin Inventory

### Active Regression Plugins (12)

| # | Plugin | Architecture | Output | Loss | Key Feature |
|---|--------|-------------|--------|------|-------------|
| 1 | `predictor_plugin_base` | RandomForest | Per-horizon RF ensemble | sklearn | Non-Keras baseline |
| 2 | `predictor_plugin_ann` | ANN (MLP) | Dense(1, linear) per horizon | Huber | Bayesian DenseFlipout heads |
| 3 | `predictor_plugin_cnn` | CNN | Conv1D → BiLSTM → Dense(1, linear) | Huber | Causal padding, stride=2 |
| 4 | `predictor_plugin_lstm` | Stacked LSTM | Multi-step time_horizon | Huber | Legacy-style with common imports |
| 5 | `predictor_plugin_tcn` | TCN | Dense(1, linear) per horizon | Configurable (MAE/Huber/Pearson/DTW) | Dilated causal Conv1D residual blocks |
| 6 | `predictor_plugin_tft` | TFT | Dense(1, linear) per horizon | MAE | GRN + GLU + LSTM encoder + MHA decoder |
| 7 | `predictor_plugin_transformer` | Transformer | Bayesian per-horizon heads | Huber | MHA(2 heads) + Conv1D + BiLSTM |
| 8 | `predictor_plugin_n_beats` | N-BEATS | Dense(1, linear) per horizon | Huber | 3 blocks × 4 FC layers, backcast/forecast |
| 9 | `predictor_plugin_prophet` | Prophet | yhat continuous | Prophet internal | Meta Prophet per-horizon |
| 10 | `predictor_plugin_mimo` | MIMO | Cross-attention decoder per horizon | MAE | Conv1D encoder + BiLSTM + horizon embeddings |
| 11 | `predictor_plugin_composite` | Composite CNN | Dense(16) → Dense(1) per horizon | Huber | Multi-branch (CLOSE, HF ticks, point features) |
| 12 | `stl_mimo_predictor` | STL-MIMO | 3 parallel MIMO branches | MAE | trend/seasonal/resid decomposition |

### Binary Classification Plugins (9)

All share: `Dense(1, sigmoid)`, `BinaryCrossentropy`, metrics: `[accuracy, AUC, BinaryF1Score]`, signal_types: `buy_entry, sell_entry, buy_exit, sell_exit`.

| # | Architecture | Trunk |
|---|-------------|-------|
| 1 | Binary ANN | Flatten → Dense stack |
| 2 | Binary CNN | Conv1D → BiLSTM |
| 3 | Binary LSTM | PE → MHA → AvgPool → BiLSTM×2 |
| 4 | Binary Transformer | MHA + Conv1D → BiLSTM |
| 5 | Binary TCN | Dilated causal Conv1D backbone |
| 6 | Binary TFT | GRN + LSTM encoder + MHA decoder |
| 7 | Binary N-BEATS | Block stack backcast/forecast |
| 8 | Binary Logistic | Dense(1, sigmoid) — no hidden layers |
| 9 | Binary MIMO | Conv1D + BiLSTM + GlobalAvgPool |

### Direction Classification Plugins (9)

Identical architectures to binary. Signal_types: `direction_long, direction_short`.

| # | Architecture | Trunk |
|---|-------------|-------|
| 1–9 | Same 9 architectures as binary | Same trunks |

### Trained Model Artifacts

| Location | Models |
|----------|--------|
| `predictor/predictor_model.keras` | Root-level regression model |
| `predictor/pretrained_model.keras` | Pretrained regression model |
| `examples/results/phase_1c_direction/` | CNN long, CNN short, ANN long, ANN short, Logistic long, Logistic short (.keras) |
| `examples/results/phase_1b_binary/` | Champion binary model + optimization output (.keras) |

### Training Performance (Direction Models — Phase 1c)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| CNN direction long | 62.5% | 29.2% | 90.9% | 0.442 | 0.785 |
| CNN direction short | 64.8% | 30.4% | 88.8% | 0.453 | 0.812 |
| ANN direction long | 54.3% | 25.9% | 96.8% | 0.409 | 0.848 |
| ANN direction short | 63.7% | 26.9% | 70.8% | 0.390 | 0.733 |

### Prediction Provider Serving Plugins (10)

| Plugin | Type | Purpose |
|--------|------|---------|
| `default_predictor` | Regression | Loads Keras model, MC uncertainty |
| `binary_predictor` | Binary | Wraps entry + exit binary predictors |
| `binary_entry_predictor` | Binary entry | buy_prob, sell_prob → threshold signals |
| `binary_exit_predictor` | Binary exit | keep_open probability |
| `direction_predictor` | Direction | 2 models (long + short) → buy/sell |
| `csv_direction_predictor` | Direction | Pre-computed CSV predictions |
| `binary_ideal_oracle` | Oracle | Look-ahead oracle with noise parameter |
| `direction_ideal_oracle` | Oracle | ATR-based look-ahead direction oracle |
| `noisy_ideal_predictor` | Noisy oracle | Regression + configurable Gaussian noise |
| `default_predictor_new` | — | Empty file |

---

## A.3 — Combinations Matrix

### Evaluation Frameworks

Two distinct frameworks exist:

| Framework | Metrics | Scripts | Used In |
|-----------|---------|---------|---------|
| **Phase 5.5 Harness** | Rolling worst-2Y Sharpe, regime robustness, transaction costs, OOS validation, benchmark comparison | `evaluation_harness.py`, `transaction_cost_model.py` | Phases 3.5–5.5 |
| **WFO/GA Pipeline** | Sharpe×√N fitness, fold profit, win rate, max DD | `walk_forward_optimizer.py` | Phases B–D |

### Full Combinations Matrix

**Rows:** Strategy implementations. **Columns:** Predictor options.

| Strategy | No Predictor | CNN Direction | CNN Binary | ANN Direction | Ensemble (CNN+ANN+Log) | NEAT ANN | Regime WFO filter | Regime Adaptive filter |
|----------|-------------|---------------|------------|---------------|----------------------|----------|-------------------|----------------------|
| **pure_mr** (script) | ✅ P5.5 | compat. but untested | incompatible | compat. but untested | compat. but untested | compat. but untested | **NOT TESTED** | **NOT TESTED** |
| **tsmom** (script) | ✅ P5.5 | compat. but untested | incompatible | compat. but untested | compat. but untested | compat. but untested | **NOT TESTED** | **NOT TESTED** |
| **dual_momentum** (script) | ✅ P5.5 | compat. but untested | incompatible | compat. but untested | compat. but untested | compat. but untested | **NOT TESTED** | **NOT TESTED** |
| **direction_atr** (plugin) | incompatible | ❌ WFO: -$19,454 | compat. but untested | ❌ WFO: -$9,281 | ❌ WFO: -$1,207 | ❌ WFO: -$931 | N/A | N/A |
| **api_predictions** (plugin) | incompatible | compat. via PP API | compat. via PP API | compat. via PP API | compat. via PP API | compat. via PP API | N/A | N/A |
| **long_short_predictions** (plugin) | incompatible | incompat. (needs regression) | incompat. (needs regression) | incompat. (needs regression) | incompat. (needs regression) | incompat. (needs regression) | N/A | N/A |
| **regime_wfo** (plugin) | ✅ self-contained | N/A (doesn't use preds) | N/A | N/A | N/A | N/A | — | N/A |
| **regime_adaptive** (plugin) | ✅ self-contained | N/A (doesn't use preds) | N/A | N/A | N/A | N/A | N/A | — |

**Legend:**
- ✅ P5.5 = tested with Phase 5.5 rigor
- ❌ WFO = tested with WFO pipeline, negative result
- **NOT TESTED** = compatible but never evaluated in either framework
- "compat. but untested" = technically combinable as hybrid (base strategy + predictor filter) but never tried
- N/A = structurally incompatible or not applicable

### The Critical Gap: Oracle Noise Ceiling

The oracle noise sweep from Phase A1 establishes:

| Oracle F1 | Oracle Accuracy | 14yr Profit | Sharpe |
|-----------|----------------|-------------|--------|
| 1.000 | 100% | +$759,385 | +0.439 |
| 0.909 | 95.1% | +$113,897 | +0.081 |
| **0.448** | **84.6%** | **-$8,730** | **-0.094** |

**Profitability threshold: F1 > ~0.91 (accuracy > ~95%)**

Current CNN direction models achieve F1 ≈ 0.44. This is **half** the required threshold. No amount of parameter tuning, threshold adjustment, or ensemble combination can close a 2× F1 gap. The WFO results confirm this — all predictor-based strategies are negative.

**This means: any combination where the predictor is the PRIMARY signal source (direction_atr, api_predictions) is structurally below the profitability floor.** The only viable predictor use is as a SECONDARY filter on an already-profitable base strategy.

---

## A.4 — Priority Classification of Untested Combinations

### HIGH Priority (Include in Phase 6.B)

| # | Combination | Rationale |
|---|------------|-----------|
| **H1** | `plugin_regime_wfo` (standalone on EUR/USD 4h) | Uses V2 causally-validated features (bb_position score=5, atr_ratio score=4, ema_alignment). The only strategy that directly operationalizes the causal inference pipeline results. **Never evaluated in any framework.** Self-contained — no predictor dependency, no F1 gap problem. |
| **H2** | `plugin_regime_adaptive` (standalone on EUR/USD 4h) | Same causal features via GMM centroids hardcoded from 15yr EURUSD. Different classification method (nearest-centroid vs thresholds). **Never evaluated in any framework.** Direct comparison to H1 reveals whether threshold vs. GMM matters. |
| **H3** | P3 cells + `regime_wfo` as meta-filter | Hybrid: run pure_mr / tsmom / dual_momentum as base, but only execute trades when `regime_wfo` classifies the current regime as favorable. Addresses P3's worst-2Y weakness by potentially filtering out bad-regime trades. Does NOT require the predictor to be profitable alone — only needs the regime filter to suppress losses in adverse regimes. |

### MEDIUM Priority (Consider for Phase 6.B if HIGH results are promising)

| # | Combination | Rationale |
|---|------------|-----------|
| M1 | `direction_atr` + CNN direction (re-eval with Phase 5.5 harness) | Already tested negative in WFO (-$19,454), but the WFO framework lacked rolling worst-2Y and regime decomposition. Could reveal that some regimes are profitable even if full-sample is negative. However, F1 gap makes positive full-sample result extremely unlikely. |
| M2 | P3 cells + CNN direction as trade filter | Hybrid: execute P3 trades only when CNN P(up) aligns with trade direction. Weaker than H3 because CNN F1=0.44 means the filter is barely better than random. |
| M3 | `regime_wfo` on USD/JPY (adapted thresholds) | regime_wfo was designed on EURUSD 4h. Adapting to USDJPY might provide a diversifier cell for the portfolio, reducing the 79.5% USD/JPY concentration problem from a different angle. |

### LOW Priority / SKIP

| # | Combination | Reason to Skip |
|---|------------|----------------|
| L1 | `api_predictions` + any predictor (Phase 5.5 harness) | Same as direction_atr but without ATR adaptation. Strictly worse than M1. |
| L2 | `long_short_predictions` + regression predictor | Requires CSV regression predictions, not connected to live pipeline. Structural dead end for deployment. |
| L3 | `direction_atr` + TCN/TFT/N-BEATS/Transformer direction | No evidence these architectures outperform CNN in this domain. All share the F1 < 0.91 structural limitation. Adding more architectures is exploration, not validation. |
| L4 | Binary predictor combinations (any strategy) | Binary signal (TP-before-SL) addresses a different question than direction. No trained binary models were ever connected to a strategy. Requires new plumbing. Low expected value given the F1 ceiling. |
| L5 | Legacy/archived predictor plugins | Superseded by refactored versions. No reason to test old architectures. |

---

## A.5 — P3 Implementation Status

### Clear Statement

**P3 was tested entirely with script-level NumPy/Pandas functions, NOT with heuristic-strategy Backtrader plugins or LTS strategy plugins.**

| P3 Cell | Script Function | heuristic-strategy Plugin? | LTS Plugin? |
|---------|----------------|---------------------------|-------------|
| EUR/USD daily pure_mr | `run_pure_mr()` in `phase5_5_corrective_audit.py` | NO | `eurusd_mr_strategy.py` exists (EUR/USD only) |
| USD/JPY daily tsmom | `run_tsmom()` in `phase5_5_corrective_audit.py` | NO | NO |
| USD/JPY daily dual_momentum | `run_dual_momentum()` in `phase5_5_corrective_audit.py` | NO | NO |

### Implications

1. **P3's Phase 5.5 evaluation is valid** — the evaluation harness (rolling windows, costs, OOS) is independent of whether strategies are plugins or functions. The strategy logic is identical.

2. **Deployment requires plugin porting.** To run P3 through the real pipeline (heuristic-strategy → lts → OANDA), USD/JPY TSMOM and USD/JPY dual_momentum must be ported to LTS strategy plugins. This is Phase 6.B work.

3. **Plugin vs. script discrepancy risk is LOW** for these strategies because they are purely rule-based (z-score, 12-month return sign, relative momentum rank). There is no ML model loading, no feature pipeline, no API dependency. The porting is mechanical.

---

## A.6 — Consolidated Recommendations for Phase 6.B

### Recommended Phase 6.B Scope: 3 High-Priority Combinations

| # | Combination | What's Needed | Expected Outcome |
|---|------------|---------------|------------------|
| **H1** | `plugin_regime_wfo` standalone (EUR/USD 4h) | Run through Phase 5.5 harness: rolling worst-2Y, regime breakdown, OOS split. Requires adapting harness for 4h data (currently daily/weekly). | Unknown — this is the most promising untested strategy. Causal validation suggests the features are genuinely predictive. May produce a new cell for the portfolio. |
| **H2** | `plugin_regime_adaptive` standalone (EUR/USD 4h) | Same as H1. Direct comparison reveals threshold vs. GMM classification difference. | Likely similar to H1 — same features, different classifier. |
| **H3** | P3 + `regime_wfo` meta-filter | Run P3 cells but suppress trades when regime_wfo classifies as unfavorable (regimes 2, 3, 4 = flat). Compute worst-2Y improvement. | Could improve P3's worst-2Y from -0.632 to something better by cutting losing periods. Risk: may also cut profitable trades and reduce Sharpe. |

### Plugin Porting Work Required

| # | Task | Effort | Blocking? |
|---|------|--------|-----------|
| 1 | Port `run_tsmom()` → LTS strategy plugin `usdjpy_tsmom_strategy.py` | Low (mechanical, rule-based) | Blocks P3 deployment only |
| 2 | Port `run_dual_momentum()` → LTS strategy plugin `usdjpy_dual_momentum_strategy.py` | Low (mechanical, rule-based) | Blocks P3 deployment only |
| 3 | Adapt Phase 5.5 evaluation harness for 4h data | Medium (new timeframe handling) | Blocks H1, H2 evaluation |
| 4 | Wire `plugin_regime_wfo` regime classification into P3 cells as filter | Medium (new hybrid logic) | Blocks H3 evaluation |

### What to Skip

- **All pure-predictor strategies** (direction_atr + CNN/ANN/NEAT/ensemble): structurally below F1 profitability threshold. WFO already confirmed negative over 14 years. Re-testing with Phase 5.5 harness will not change a -$19K → positive.
- **Binary predictor combinations**: no existing wiring, no evidence of edge, not connected to any strategy.
- **Regression predictor combinations**: only consumed by `long_short_predictions` via CSV, dead end for live deployment.
- **TCN/TFT/N-BEATS/Transformer variants**: no evidence they outperform CNN in this domain. Same F1 ceiling applies.

### Decision Framework for Phase 6.B Results

| H1/H2 Result | H3 Result | Action |
|--------------|-----------|--------|
| Positive SR, worst-2Y > -0.9 | Improves P3 worst-2Y | Add H1/H2 as new portfolio cell + deploy P3+filter hybrid |
| Positive SR, worst-2Y > -0.9 | No improvement | Add H1/H2 as new cell, deploy original P3 |
| Negative or worse than P3 | Improves P3 worst-2Y | Deploy P3+filter hybrid |
| Negative or worse than P3 | No improvement | **Confirm P3 as best candidate, proceed to deployment** |

In all cases, the terminal decision from Phase 5.5 (TERMINAL 1: Deploy) remains valid — Phase 6.B can only add evidence, not remove it.

---

## Honest Acknowledgments

1. **The WFO-based tests (Phases B–D) used a different framework.** It is theoretically possible that the Phase 5.5 harness would produce different conclusions for CNN-based strategies. However, the F1 gap (0.44 vs. required 0.91) is so large that a framework change cannot plausibly close it.

2. **plugin_regime_wfo and plugin_regime_adaptive have never been run.** The code exists, configs exist, but there is zero evidence of any execution. Their causal foundation is strong, but untested code may have bugs. Phase 6.B must treat them as unverified.

3. **The GMM centroids in plugin_regime_adaptive are hardcoded from 15yr EURUSD data.** This is in-sample fitting. The Phase 5.5 harness OOS split will test whether these centroids generalize.

4. **This audit found no hidden gems.** The most likely Phase 6.B outcome is confirmation that P3 is the best available candidate, with regime filtering as a potential incremental improvement. This is a valid and honest outcome.
