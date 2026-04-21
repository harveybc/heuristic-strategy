# F-2: Jansen "Machine Learning for Algorithmic Trading" — Framework Integration

**Date**: 2025-06-17 (original), **CORRECTED**: 2026-04-18  
**Scope**: Map Stefan Jansen's ML4T framework to Project 2 experimental design  
**Status**: Original written without direct book access. Corrected 2026-04-18 using actual book TOC (23 chapters + appendix). Chapter-level mappings now verified.

---

## 1. Jansen Framework Overview

Stefan Jansen's *Machine Learning for Algorithmic Trading* (2nd edition, 2020) provides a comprehensive framework across 23 chapters:

| Ch | Topic | P2 Mapping |
|----|-------|------------|
| 1 | ML for Trading overview | Context |
| 2 | Market & fundamental data | F-3 data catalog |
| 3 | Alternative data | F-3 (Google Trends, CoT) |
| 4 | **Financial feature engineering, alpha factors, Alphalens IC** | feature-eng, F-6 IC analysis |
| 5 | **Portfolio optimization, Sharpe ratio, HRP** | F-10 evaluation, Part III multi-asset |
| 6 | **ML process, purged/embargo CV, cross-validation** | F-5 pipeline spec, F-10 |
| 7 | **Linear models (Ridge, Lasso, ElasticNet)** | Path B baselines |
| 8 | **ML4T workflow, backtesting, deflated SR, PBO** | F-10, heuristic-strategy |
| 9 | **Time-series models (ARIMA, GARCH, cointegration, pairs trading)** | lts `arima_predictor.py` |
| 10 | **Bayesian ML, dynamic Sharpe ratio** | F-10 (Bayesian SR comparison) |
| 11 | **Random forests, long-short strategy** | Path B baselines |
| 12 | **Boosting (XGBoost, LightGBM, CatBoost), SHAP** | Path B baselines + interpretability |
| 13 | **Unsupervised (PCA, UMAP, DBSCAN, GMM, HRP)** | F-7, feature-eng |
| 14-16 | NLP / text data | Not in P2 scope |
| 17 | **Deep learning fundamentals** | predictor base architecture |
| 18 | **CNNs for time series** | predictor CNN plugins |
| 19 | **RNNs (LSTM, GRU)** | predictor LSTM plugins |
| 20 | **Autoencoders, conditional autoencoders** | feature-extractor (VAE/CVAE) |
| 21 | **GANs, TimeGAN for synthetic data** | synthetic-datagen, timeseries-gan |
| 22 | **Deep RL, OpenAI Gym trading agent** | gym-fx, rl-optimizer |
| 23 | Conclusions | Context |
| App | **WorldQuant formulaic alphas, TA-Lib factor library** | feature-eng |

### Chapters missed in original F-2 (now corrected)

- **Ch 9 (ARIMA/GARCH/cointegration)**: lts has `arima_predictor.py` — should be included as Path B baseline. Cointegration relevant if multi-asset scope expands.
- **Ch 10 (Bayesian ML)**: Bayesian Sharpe ratio comparison is superior to frequentist for strategy comparison. Should be added to F-10 evaluation.
- **Ch 12 (SHAP)**: Model interpretability via SHAP values should be part of F-10 diagnostics for neural models.
- **Ch 13 (UMAP, DBSCAN, HRP implementation)**: More complete than F-7 acknowledged. UMAP + DBSCAN implementations available in the book.
- **Ch 20 (Conditional autoencoders)**: Maps to feature-extractor repo (CVAE work, June 2025). This repo was not known during original F-7/F-8.
- **Ch 21 (TimeGAN)**: Maps directly to synthetic-datagen (`TimeGANGenerator`) and timeseries-gan (SC-VAE-GAN). Both repos exist — original F-8 said they didn't.
- **Ch 22 (Deep RL + OpenAI Gym)**: Maps directly to gym-fx (208 stars) + rl-optimizer (PPO). Both repos exist — original F-8/F-9 said gym-fx was missing.
- **Appendix (WorldQuant alphas)**: feature-eng has TA-Lib but no WorldQuant formulaic alphas. Low priority addition.

---

## 2. Mapping to Project 2 Parts

### 2.1 Data and Factor Research → F-3 (Data Catalog) + F-4 (Asset Selection)

| Jansen Concept | P2 Mapping | Status |
|---------------|------------|--------|
| Market data (OHLCV) | OANDA + HistData + yfinance (F-3 Category 1) | ✅ Covered |
| Fundamental data | FRED + Alpha Vantage Economic (F-3 Category 2) | ✅ Covered |
| Alternative data (sentiment, satellite) | Google Trends, CFTC CoT (F-3 Category 3) | ✅ Covered (marked speculative for sentiment) |
| Alpha factor universe | Technical indicators (feature-eng plugins) + macro features (F-4 Phase 2) | ✅ Partially covered — macro features not yet engineered |
| Data quality / survivorship bias | F-5 §2.3 data quality checks | ✅ Covered |

**Gap**: Jansen emphasizes **factor evaluation before model training** — computing Information Coefficient (IC), factor turnover, and factor decay for each alpha candidate. Our F-6 causal analysis serves a similar purpose but is methodologically different (causal vs correlational). Consider adding IC analysis to the feature selection pipeline.

### 2.2 Feature Engineering → feature-eng repo + F-6/F-7

| Jansen Concept | P2 Mapping | Status |
|---------------|------------|--------|
| Technical alpha factors | feature-eng `tech_indicator` plugin (12 features) | ✅ Exists |
| Factor combination / interaction | Not implemented — features used individually | ⚠️ Gap: no interaction terms |
| PCA / dimensionality reduction | feature-eng `regime_analysis.py` (PCA 6 components) | ✅ Exists |
| Clustering for regime detection | GMM K=9 (causal-inference + heuristic-strategy) | ✅ Exists |

### 2.3 Supervised Learning Models → predictor repo + Path B

| Jansen Concept | P2 Mapping | Status |
|---------------|------------|--------|
| Linear models (Ridge, Lasso, ElasticNet) | Not in predictor (uses neural models only) | ⚠️ Gap: no linear baselines |
| Tree models (Random Forest, Gradient Boosting, LightGBM) | Not in predictor | ⚠️ Gap: no tree baselines |
| Deep learning (RNN, CNN, Transformer) | predictor plugins: LSTM, CNN, Transformer, TFT, TCN, N-BEATS | ✅ Comprehensive |
| **Walk-forward cross-validation** | heuristic-strategy WFO + F-5 pipeline spec | ✅ Designed |
| **Purged / embargo cross-validation** | Not implemented | ⚠️ Gap (see §3.1) |

**Key Jansen principle adopted**: Walk-forward validation is the only valid approach for time-series strategies. Standard k-fold is invalid due to temporal dependence. This is already our design (F-5, F-10).

### 2.4 Unsupervised Learning → F-7

| Jansen Concept | P2 Mapping | Status |
|---------------|------------|--------|
| K-Means / hierarchical clustering | feature-eng + causal-inference | ✅ Exists |
| GMM | causal-inference + heuristic-strategy | ✅ Exists |
| PCA | feature-eng | ✅ Exists |
| Topic models (LDA) | Not applicable (no text data in scope) | N/A |
| Autoencoders | Designed but code removed (UL-3 opportunity) | ⚠️ Gap |

### 2.5 Reinforcement Learning → Path C

| Jansen Concept | P2 Mapping | Status |
|---------------|------------|--------|
| OpenAI Gym environment | causal-inference `PredictionEnv(gym.Env)` — legacy, needs `gymnasium` migration | ⚠️ Exists but broken |
| PPO / A2C agents | causal-inference has `stable-baselines3` dependency | ⚠️ Exists but untested |
| Reward shaping (Sharpe-based) | Not implemented | ❌ Gap |
| RL for portfolio allocation | Not in scope (single-asset focus) | N/A |

### 2.6 Backtesting and Evaluation → F-10

| Jansen Concept | P2 Mapping | Status |
|---------------|------------|--------|
| Vectorized backtest | heuristic-strategy + custom | ✅ Exists |
| Event-driven backtest (backtrader) | heuristic-strategy + lts (backtrader plugins) | ✅ Exists |
| **Deflated Sharpe Ratio** | F-10 §5.2 (specified) | ✅ Specified |
| **Probability of Backtest Overfitting (PBO)** | F-10 §5.2 (specified) | ✅ Specified |
| Multiple testing correction | F-10 §5.1 (Bonferroni within path) | ✅ Specified |
| Transaction cost modeling | F-10 §7 (EUR/USD: 1.5 pips RT) | ✅ Specified |

---

## 3. Key Methodological Additions from Jansen Framework

### 3.1 Purged Cross-Validation with Embargo (HIGH Priority)

**Concept**: In walk-forward validation, training and test sets must have a gap ("embargo") to prevent information leakage from overlapping labels. If the target is a 6-bar forward return, using bar $t$ in training and bar $t+1$ in validation creates leakage because their labels overlap.

**Current state**: Our F-5 window design uses annual steps (12+ months between train end and test start), which naturally provides embargo. But within each window, the train/validation split may not have an embargo.

**Recommendation**: Add a configurable embargo period between train and validation sets within each window:
```
Train: [window_start, train_end]
EMBARGO: [train_end, train_end + embargo_bars]  ← exclude these bars
Validation: [train_end + embargo_bars, val_end]
```
Where `embargo_bars` ≥ forward-return horizon (e.g., 6 bars for 6-bar target).

**Effort**: Small — modify window manifest generator.

### 3.2 Linear and Tree Model Baselines (MEDIUM Priority)

**Concept**: Jansen emphasizes starting with simple models (Ridge regression, Random Forest) before deep learning. If a simple model achieves similar performance, the complexity of neural networks is not justified.

**Current state**: The predictor repo only has neural network models (ANN, CNN, LSTM, Transformer, TFT, TCN, N-BEATS). No linear or tree baselines exist.

**Recommendation**: Add 2-3 baseline models for Path B:
1. **Ridge regression** — simplest possible linear model
2. **LightGBM** — strongest tree-based model for tabular data
3. Compare against all neural models per rolling window

**Implementation**: These don't need predictor plugins — can be standalone scripts using scikit-learn and lightgbm. Their predictions feed into heuristic-strategy's backtest the same way.

**Effort**: Medium — ~100 lines per model plus evaluation integration.

### 3.3 Alpha Factor IC Analysis (MEDIUM Priority)

**Concept**: Before training models, evaluate each feature's Information Coefficient (rank correlation with forward returns) and factor decay profile (IC at different horizons). This identifies which features have predictive value and at what timeframe.

**Current state**: F-6 provides causal evidence (PCMCI+, ICP, TE) but not IC analysis. The causal null result (no lagged links at 4h) suggests IC may also be near zero, but IC is a different lens (correlational, not causal).

**Recommendation**: Add IC analysis as a pre-training diagnostic:
```python
ic = spearmanr(feature_values, forward_returns).correlation
# Compute IC per rolling 1-year window → IC mean, IC std, IC IR (IC/std)
```

**Effort**: Small — ~50 lines of analysis code.

### 3.4 Hierarchical Risk Parity (LOW Priority — Multi-Asset Phase Only)

**Concept**: When combining multiple strategy signals or assets, use hierarchical risk parity (HRP) instead of mean-variance optimization. HRP is more stable and doesn't require covariance matrix inversion.

**Current state**: Not relevant until Part III when secondary assets (USD/JPY, SPY) are added.

**Recommendation**: Defer to Part III. When combining strategy signals across assets, use HRP for capital allocation.

---

## 4. Jansen-Informed Experiment Design Modifications

### 4.1 Path B (Supervised ML Rolling) — Modified Pipeline

Based on Jansen framework, the Path B pipeline (F-5 §6.2) should be augmented:

```
For each window w:
  1. feature-eng → features_w.csv
  2. IC analysis on features_w (diagnostic — does NOT gate training)
  3. preprocessor --normalize --fit_on train_w --embargo 6 bars
  4. BASELINE: Ridge/LightGBM on train_w → evaluate on val_w
  5. NEURAL: predictor --plugin {model} on train_w → evaluate on val_w
  6. Compare baseline vs neural: if IC(baseline) ≥ 0.95 × IC(neural), flag neural as unjustified
  7. Select best model per window
  8. Evaluate on test_w → record OOS metrics
  9. Log with full metadata
```

### 4.2 F-10 Additions

Two evaluation methods from Jansen not in our original F-10:

1. **Combinatorial Symmetric Cross-Validation (CSCV)** for PBO estimation — already referenced in F-10 §5.2 but implementation details needed
2. **Factor-model attribution** — decompose strategy returns into market exposure (beta), factor exposure (momentum, value, etc.), and alpha. Report alpha separately from factor loading.

---

## 5. Summary: What Jansen Adds to Project 2

| Addition | Priority | Part | Effort | Impact |
|----------|----------|------|--------|--------|
| Purged CV with embargo | HIGH | II, III | Small | Prevents information leakage |
| Linear/tree baselines | MEDIUM | II (Path B) | Medium | Justifies neural complexity |
| IC analysis | MEDIUM | II | Small | Feature quality diagnostic |
| Factor-model attribution | LOW | III | Medium | Decomposes alpha vs beta |
| HRP allocation | LOW | III+ | Medium | Multi-asset capital allocation |

---

## 6. Limitations of This Document

1. **Book TOC used, not full text**: Chapter-level mapping verified against actual TOC (23 chapters). Specific code examples and implementation details not cross-referenced.
2. **FX-specific adaptations needed**: Jansen's framework is primarily equity-focused. FX-specific considerations (24h market, no dividends, carry trade, central bank events) are not covered in the book and must be addressed by our own analysis (F-6, F-4).
3. **The book predates 2020**: Post-COVID market dynamics, the 2022-2024 rate-hike cycle, and recent advances in transformer architectures (which we already use via TFT) are not covered.
4. **Newly discovered repos resolve gaps**: The original F-2 noted missing linear baselines, autoencoders, synthetic data, and RL environments as gaps. feature-extractor (autoencoders), synthetic-datagen (TimeGAN + regime generators), timeseries-gan (SC-VAE-GAN), gym-fx (RL trading env), and rl-optimizer (PPO) all exist and partially fill these gaps.
