# F-7: Unsupervised Learning Opportunities for Project 2

**Date**: 2025-06-17  
**Scope**: Inventory of existing unsupervised methods, assessment of gaps, and identification of integration points for Project 2  
**Depends on**: F-1 (Project 1 lessons), F-6 (Causal inference), F-8 (Infrastructure audit)

---

## 1. Current Unsupervised Methods Inventory

### 1.1 What Exists and Works

| Method | Repo | File | Status | Purpose |
|--------|------|------|--------|---------|
| **GMM (K=9)** | causal-inference | `cluster_regime_analysis.py` | ✅ Trained | Regime discovery on 3 causal features (bb_position, atr_ratio, ema_alignment). BIC-optimal K=9. Centroids exported. |
| **GMM inference** | heuristic-strategy | `plugin_regime_adaptive.py` | ✅ Deployed | Live regime classification using hardcoded K=9 centroids. Maps 9 clusters → 6 tradeable regimes. |
| **K-Means baseline** | causal-inference | `cluster_regime_analysis.py` | ✅ Trained | Hard-clustering comparison to GMM. |
| **Hierarchical + PCA** | feature-eng | `regime_analysis.py` | ✅ Trained | Ward linkage on PCA(6) of 10+ technical features. K=3 and K=5 regime cuts. Transition matrices computed. |
| **PCA** | feature-eng | `regime_analysis.py` | ✅ Trained | 6-component PCA for noise reduction before clustering. |
| **HMM (regime-bootstrap)** | doin-plugins | `predictor/synthetic.py` | ⚠️ Partial | Pre-trained HMM (5 regimes) for synthetic data generation. Model file (.joblib) not included in repo. |
| **Threshold regime detector** | feature-eng | `regime_detector.py` | ✅ Deployed | V1/V2/V3 threshold-based regime classifier derived from clustering results. |
| **RPCMCI (regime-causal)** | causal-inference | `causal_regime_analysis_v2.py` | ❌ Failed | Joint regime + causal graph discovery. Numpy type bug prevented completion. |

### 1.2 What Does NOT Exist

| Method | Potential Value | Status |
|--------|----------------|--------|
| **Autoencoders** | Feature compression, representation learning | Code removed from feature-eng; test stubs remain. D1-D3 preprocessor splits designed for autoencoder but no implementation. |
| **SSA (Singular Spectrum Analysis)** | Trend/cycle/noise decomposition | `ssa.py` in feature-eng is misnamed — it's actually a Transformer predictor. No SSA code exists. |
| **Anomaly detection** | Identifying regime transitions, market stress | Nothing (no IsolationForest, LOF, OneClassSVM). |
| **Change-point detection** | Identifying structural breaks for retraining triggers | No ruptures/BOCPD/CUSUM code. |
| **UMAP / t-SNE** | Nonlinear dimensionality reduction, visualization | Not implemented. |
| **DBSCAN** | Density-based regime discovery (no K required) | Not implemented. |
| **Self-Organizing Maps** | Topology-preserving regime mapping | Not implemented. |
| **VAE (Variational Autoencoders)** | Generative regime modeling, synthetic data | Not implemented. |

---

## 2. Gap Analysis

### 2.1 The Autoencoder Gap

The preprocessor's D1-D3 splits (33%/8.3%/8.3%) were explicitly designed for autoencoder training/validation/testing. The feature-eng repo once had an `AutoencoderManager` class (referenced in tests) that was removed. This represents **abandoned infrastructure** — the architecture was designed for unsupervised feature compression but the implementation was dropped.

**Impact**: Without representation learning, the full 12+ feature space is passed directly to predictors. This may contribute to overfitting in rolling retraining (Path B), especially with small rolling windows.

### 2.2 The Change-Point Detection Gap

No repo implements any form of change-point detection. The F-6 document proposed "causal retrain trigger" (CI-5) using rolling PCMCI, but simpler statistical change-point methods (CUSUM, BOCPD, ruptures) could achieve similar triggering at much lower computational cost.

**Impact**: The current regime system (GMM K=9) classifies bars into regimes but does not detect *when* the regime model itself becomes stale. Change-point detection on the GMM log-likelihood or on feature distribution moments could signal when to re-fit the regime model.

### 2.3 The Regime Model Staleness Problem

The K=9 GMM was trained once on 15 years of 4h EURUSD data. Its centroids are hardcoded in `plugin_regime_adaptive.py`. There is no mechanism to:
1. Detect that the regime model is outdated
2. Re-fit the GMM on more recent data
3. Validate that K=9 is still optimal
4. Handle regime drift (gradual centroid movement)

This is the unsupervised analogue of the "no rolling retraining orchestrator" gap (G-1 in F-8).

---

## 3. Unsupervised Opportunities for Project 2

### 3.1 Opportunity UL-1: Online/Rolling GMM Re-Fitting (HIGH Priority)

**What**: Implement rolling GMM with periodic re-fitting. Every N months (or triggered by change-point detection), re-fit the K-component GMM on the most recent M years of data. Re-validate K via BIC.

**Why**: The static K=9 GMM from 2005-2020 data may not represent 2021+ market regimes. Adaptive strategies (Path A) need an adaptive regime model, not a static one.

**Implementation**:
1. Extract `cluster_regime_analysis.py` fitting logic into a reusable module
2. Add to rolling retraining orchestrator: before each window, re-fit GMM on training portion
3. Pass updated centroids to `regime_adaptive` plugin (currently hardcoded → make configurable)
4. Track: K_optimal, centroid drift, regime transition rates per window

**Effort**: Medium  
**Part**: II (Path A)

### 3.2 Opportunity UL-2: Change-Point Detection for Retraining Triggers (HIGH Priority)

**What**: Implement statistical change-point detection on feature distributions and model residuals to trigger retraining.

**Methods to consider**:
| Method | Library | Complexity | Type |
|--------|---------|-----------|------|
| CUSUM | Manual or `detecta` | Low | Cumulative sum monitoring |
| BOCPD (Bayesian Online CPD) | `bayesian_changepoint_detection` | Medium | Online Bayesian |
| `ruptures` (Pelt, BinSeg, Window) | `ruptures` | Low | Offline multi-changepoint |
| Distribution divergence (KL, Wasserstein) | `scipy` | Low | Rolling window comparison |

**Recommended approach**: Rolling Wasserstein distance on feature marginals (simple, interpretable, low compute). When $W_p(F_{t-w:t}, F_{t-2w:t-w}) > \tau$ for any core feature, trigger retraining.

**Why**: Cheaper than rolling PCMCI (CI-5 from F-6), and directly addresses the "when to retrain" question. Can be combined with CI-5 for a two-level trigger: fast statistical trigger (UL-2) + slow causal structure trigger (CI-5).

**Effort**: Small-Medium  
**Part**: II and III

### 3.3 Opportunity UL-3: Autoencoder Reconstruction (MEDIUM Priority)

**What**: Implement the autoencoder that was designed for (D1-D3 splits exist) but never completed. Use it for:
1. Feature compression (reduce 12+ features to 4-6 latent dimensions)
2. Anomaly detection (high reconstruction error = out-of-distribution regime)
3. Representation learning (latent space may capture regime structure)

**Implementation**:
1. Implement `AutoencoderManager` class in feature-eng (restore abandoned infrastructure)
2. Architecture: Input(12) → Dense(8, relu) → Dense(4, relu) → Dense(8, relu) → Output(12) — simple symmetric
3. Train on D1 (33%), validate on D2 (8.3%), test on D3 (8.3%)
4. Use encoder output as compressed feature set for predictor
5. Monitor reconstruction error as anomaly/regime-change signal

**Why**: Addresses dimensionality and overfitting for rolling retraining. Also provides an unsupervised anomaly signal. The preprocessor infrastructure (D1-D3 splits) is already designed for this.

**Effort**: Medium  
**Part**: III (Path B — compressed features for ML models)

### 3.4 Opportunity UL-4: DBSCAN for Regime Discovery (LOW Priority)

**What**: Replace or supplement K-optimal GMM with DBSCAN, which discovers cluster count from data density and identifies noise points.

**Why**: GMM requires pre-specifying K range and assumes Gaussian clusters. DBSCAN may find non-spherical regime shapes and explicitly labels outlier bars as noise (useful for "don't trade" signals). However, DBSCAN is sensitive to ε and min_samples — may not improve on GMM.

**Effort**: Small  
**Part**: II (quick experiment)

### 3.5 Opportunity UL-5: SSA for Trend/Cycle Decomposition (MEDIUM Priority)

**What**: Implement actual SSA (Singular Spectrum Analysis) to decompose price series into trend + oscillatory + noise components. The `ssa.py` plugin name is available but currently contains a Transformer — either rename the Transformer or create a true SSA plugin.

**Why**: SSA provides a clean decomposition without assuming periodicity (unlike FFT). The trend component could replace moving averages, and the oscillatory components could identify cyclic regime frequencies. The existing `fft` plugin in feature-eng provides Fourier decomposition but SSA handles non-stationary data better.

**Implementation**: Use `pyts.decomposition.SingularSpectrumAnalysis` or manual SVD-based implementation. Parameters: window_length L (typically N/3 to N/2), number of groups.

**Effort**: Medium  
**Part**: II or III (feature engineering improvement)

### 3.6 Opportunity UL-6: VAE for Synthetic Regime Generation (LOW Priority, Part V)

**What**: Use a Variational Autoencoder to learn a generative model of market regimes. Sample from the latent space to generate synthetic market conditions for strategy stress-testing.

**Why**: The doin-plugins HMM synthetic generator requires a pre-trained model (.joblib) that isn't in the repo. A VAE could replace it with a fully trainable, regime-aware generator. Also relevant to Part V (Synthetic Augmentation).

**Effort**: Large  
**Part**: V

### 3.7 Opportunity UL-7: Regime Transition Modeling (MEDIUM Priority)

**What**: Model regime transitions as a Markov chain. Estimate transition probabilities from the GMM regime sequence. Use the transition matrix to:
1. Predict upcoming regime (most-probable next state)
2. Weight strategy allocation by transition probability
3. Detect transition probability drift (staleness signal)

**Why**: The hierarchical clustering analysis in feature-eng already computes transition matrices (K=3 and K=5). The GMM K=9 regime sequence could also be modeled this way. Transition probabilities add a temporal dimension that static regime classification lacks.

**Implementation**: Already partially done in `regime_analysis.py`. Extend to rolling transition matrices.

**Effort**: Small  
**Part**: II (Path A)

---

## 4. Integration with Project 2 Parts

| Opportunity | Part II (Path A) | Part III (Path B) | Part IV (RL) | Part V (Synthetic) |
|-------------|-----------------|-------------------|-------------|-------------------|
| UL-1: Rolling GMM | ✅ Core need | ✅ Feature for ML | ✅ State component | — |
| UL-2: Change-point triggers | ✅ When to re-optimize | ✅ When to retrain | — | — |
| UL-3: Autoencoder | — | ✅ Feature compression | ✅ State compression | — |
| UL-4: DBSCAN | ✅ Quick comparison | — | — | — |
| UL-5: SSA decomposition | ✅ Better features | ✅ Better features | ✅ Better state | — |
| UL-6: VAE synthetic | — | — | — | ✅ Core need |
| UL-7: Transition modeling | ✅ Temporal regime info | ✅ Feature for ML | ✅ State component | — |

---

## 5. Relationship to F-6 (Causal Inference)

Several F-6 and F-7 opportunities are complementary:

| F-6 Opportunity | F-7 Complement | Synergy |
|----------------|---------------|---------|
| CI-1 (RPCMCI fix) | UL-1 (Rolling GMM) | RPCMCI jointly discovers regimes + causal structure. Rolling GMM provides the regime input that RPCMCI refines. |
| CI-4 (Causal feature filter) | UL-3 (Autoencoder) | Causal filter selects features, autoencoder compresses them. Pipeline: raw features → causal filter → autoencoder → predictor. |
| CI-5 (Causal retrain trigger) | UL-2 (Change-point trigger) | Two-level trigger system: fast statistical (UL-2, seconds) + slow causal (CI-5, hours). Use UL-2 for immediate response, CI-5 for structural validation. |

---

## 6. Recommended Priority Order

| Priority | Opportunity | When | Effort | Value |
|----------|-------------|------|--------|-------|
| 1 | UL-1: Rolling GMM | Part II start | Medium | HIGH — foundational for adaptive regime strategies |
| 2 | UL-2: Change-point triggers | Part II | Small-Medium | HIGH — answers "when to retrain" cheaply |
| 3 | UL-7: Transition modeling | Part II | Small | MEDIUM — adds temporal regime information |
| 4 | UL-3: Autoencoder | Part III | Medium | MEDIUM — reduces overfitting for ML rolling |
| 5 | UL-5: SSA decomposition | Part II/III | Medium | MEDIUM — better feature engineering |
| 6 | UL-4: DBSCAN | Part II | Small | LOW — quick experiment, likely marginal improvement |
| 7 | UL-6: VAE synthetic | Part V | Large | LOW until Part V — then becomes core |

---

## 7. Honest Caveats

1. **The existing GMM K=9 is the most valuable unsupervised asset**. It was derived from causally-validated features, uses BIC-optimal K, and is already deployed in a working strategy plugin. Most new unsupervised work should extend this, not replace it.

2. **Autoencoders were previously attempted and abandoned**. The reasons for removal are unknown. Before rebuilding, check git history for why `AutoencoderManager` was removed — it may have underperformed.

3. **More clusters ≠ better strategies**. The regime_adaptive plugin maps 9 clusters → 6 tradeable regimes. Adding more unsupervised structure (more regimes, more features, more methods) risks over-segmenting the market into regimes with insufficient data per regime for reliable strategy estimation.

4. **Change-point detection requires threshold tuning**. The Wasserstein distance threshold τ and the lookback window w are hyperparameters that must themselves be validated — introducing another layer of optimization.

5. **SSA implementation is straightforward but tuning window_length L is not**. L controls the trend/cycle separation frequency. Incorrect L can either over-smooth (losing signal) or under-smooth (retaining noise). L should be informed by the dominant cycle frequencies found by the existing FFT plugin.
