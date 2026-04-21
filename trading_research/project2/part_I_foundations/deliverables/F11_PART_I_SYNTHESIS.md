# F-11: Part I Foundations — Synthesis Document

**Date**: 2025-06-17 (original), **CORRECTED**: 2026-04-18  
**Scope**: Integrate all Part I deliverables into a coherent foundation for Project 2 execution  
**Gate**: User review of this document determines whether to proceed to Part II  
**Correction note**: 7 additional repos discovered on GitHub (gym-fx, synthetic-datagen, timeseries-gan, rl-optimizer, feature-extractor, trading-signal, agent-multi). G-2 resolved, G-3 downgraded. Total repos: 18 (16 active + 2 obsolete). Total plugins: 163.

---

## 1. Part I Deliverables Summary

| Task | Deliverable | Status | Key Outcome |
|------|------------|--------|-------------|
| **F-1** | [Project 1 Knowledge Consolidation](F1_PROJECT1_LESSONS_FOR_PROJECT2.md) | ✅ Complete | 6 constraints, 10 design guidances, 5 reopened items, 14 infrastructure reuse items, 7 pitfalls |
| **F-2** | [Jansen Framework Integration](F2_JANSEN_FRAMEWORK_INTEGRATION.md) | ✅ Complete (corrected) | 23-chapter mapping verified against actual TOC. 8 missed chapters added (ARIMA, Bayesian, SHAP, autoencoders, TimeGAN, RL). 5 methodological additions confirmed. |
| **F-3** | [Data Sources Catalog](F3_DATA_SOURCES_CATALOG.md) | ✅ Complete | 20+ sources across 6 categories. Zero-cost stack: OANDA + FRED + HistData + CoT + yfinance + Binance |
| **F-4** | [Asset and Timeframe Selection](F4_ASSET_TIMEFRAME_SELECTION.md) | ✅ Complete | Primary: EUR/USD @ 4h. Secondary: USD/JPY, SPY, BTC/USD. Data: 2005-2019 IS, 2020-2024 held-out |
| **F-5** | [Data Pipeline Specification](F5_DATA_PIPELINE_SPEC.md) | ✅ Complete | 5-stage pipeline. Anchored expanding windows. ~30 hours total compute across 3 machines |
| **F-6** | [Causal Inference Opportunities](F6_CAUSAL_INFERENCE_OPPORTUNITIES.md) | ✅ Complete | 7 opportunities (CI-1 to CI-7). Critical finding: no lagged causal links at 4h. RPCMCI needs fixing |
| **F-7** | [Unsupervised Learning Opportunities](F7_UNSUPERVISED_LEARNING_OPPORTUNITIES.md) | ✅ Complete | 7 opportunities (UL-1 to UL-7). Rolling GMM and change-point triggers highest priority |
| **F-8** | [Infrastructure Audit](F8_INFRASTRUCTURE_AUDIT_PROJECT2.md) | ✅ Complete (corrected) | 18 repos audited (was 11), 163 plugins (was 114). G-2 resolved (synthetic-datagen exists). G-3 downgraded (gym-fx + rl-optimizer exist). Only G-1 remains critical. |
| **F-9** | [gym-fx / NEAT Retrospective](F9_GYMFX_NEAT_RETROSPECTIVE.md) | ✅ Complete (corrected) | gym-fx EXISTS (208 stars, cloned). Two NEAT systems confirmed. Evolution chain: gym-fx → agent-multi → causal-inference → rl-optimizer → doin-core. |
| **F-10** | [Evaluation Framework](F10_EVALUATION_FRAMEWORK.md) | ✅ Complete | 4 primary + 10 secondary metrics. 7 kill criteria. 3 statistical tests. Pre-registered. |

---

## 2. Core Hypothesis and Evidence Assessment

### 2.1 The Hypothesis

> **Project 2 Hypothesis**: Periodically re-optimized (adaptive) strategies can outperform the static strategies tested in Project 1 on EUR/USD.

### 2.2 Evidence For

| Evidence | Source | Strength |
|----------|--------|----------|
| P1 static strategies failed on held-out data (SR = −0.065) | F-1 | STRONG — static demonstrably fails |
| 23 years with fixed parameters is unrealistic (D-10) | F-1 | STRONG — theoretical argument |
| Regime structure exists (GMM K=9 with distinct centroids) | F-7, F-6 | MEDIUM — regimes detectable but profitability unproven |
| WFO infrastructure already works for heuristic strategies | F-8, F-9 | MEDIUM — reduces implementation risk for Path A |
| Model save/load + warm-start exists in predictor | F-8 | MEDIUM — enables rolling retraining |
| Regime-adaptive plugin already outperforms base strategies in some periods | heuristic-strategy P1 results | WEAK — not consistent across all periods |

### 2.3 Evidence Against

| Evidence | Source | Strength |
|----------|--------|----------|
| **No lagged causal links found at 4h** — features don't predict returns | F-6 | STRONG — undermines all prediction-based approaches |
| All features scored 1/4 on sensitivity analysis robustness | F-6 | STRONG — causal claims are weak |
| +0.663 return autocorrelation is overlapping-window artifact | F-6 | MEDIUM — the most promising signal was spurious |
| EUR/USD momentum unprofitable after costs | F-6 (cross-asset audit) | MEDIUM — momentum path is challenging |
| No Hurst exponent significantly different from 0.5 | F-6 (cross-asset audit) | MEDIUM — no detectable serial dependence |
| NFP event-driven thesis not supported | F-6 (nfp_event_response_poc) | WEAK — one event type tested |
| Phase D NEAT showed all negative profit | F-9 | WEAK — specific to one configuration |

### 2.4 Honest Assessment

The evidence against is **stronger and more specific** than the evidence for. The causal analysis (F-6) found that:
- Features are contemporaneous with returns, not leading
- The only leading indicator (ema_alignment by TE) has trivially small effect
- Sensitivity analysis downgrades all features to WEAK

However:
- **The null result is at 4h only**. Other timeframes (daily, weekly) are untested. Macro features are untested. CI-2 (multi-timeframe PCMCI) may find lagged links that change the picture.
- **Adaptive ≠ predictive**. Path A (adaptive heuristic) doesn't require prediction — it re-optimizes strategy parameters to current market conditions. The regime structure (GMM K=9) is real even if its predictive value is uncertain.
- **The static baseline is negative**, so even a small adaptive advantage may achieve positive held-out SR.

**Risk rating**: The probability of all three paths failing is **moderate to high (40-60%)**. But the infrastructure exists, the cost is primarily compute time, and a well-documented negative result establishes what doesn't work — which has value for future research.

---

## 3. Critical Blockers for Part II

**Only ONE critical blocker remains** (was three). G-2 and G-3 are resolved/downgraded after discovering repos on GitHub.

| Blocker | Gap ID | Resolution | Effort | Owner |
|---------|--------|-----------|--------|-------|
| **No rolling retraining orchestrator** | G-1 | Build `rolling_orchestrator.py` — manage windows, dispatch training, log results | Large (2-3 days) | First Part II task |
| ~~No synthetic data pipeline~~ | ~~G-2~~ | **RESOLVED (2026-04-18)**: `synthetic-datagen` has VAE/GAN/TimeGAN + plugin arch + predictive utility evaluator. `timeseries-gan` has SC-VAE-GAN (23-feature). Minor cleanup: register 7 generators, add deps. | ~~N/A~~ Small | Part V start |
| ~~No RL trading environment~~ | ~~G-3~~ | **DOWNGRADED (2026-04-18)**: `gym-fx` has 14 gym envs + NEAT/DDQN agents (208 stars). `rl-optimizer` has PPO + PredictionEnv. Need: gym→gymnasium migration, TF1→TF2 modernization. | ~~Medium~~ Medium | Part IV start |

**For Part II specifically**: Only G-1 is blocking. G-2 and G-3 resolution reduces overall project risk.

---

## 4. Part II Execution Readiness

### 4.1 What's Ready

| Component | Status |
|-----------|--------|
| EUR/USD historical data (existing P1 data at 1h) | ✅ Available in `feature-eng/tests/data/` (2005-2020) |
| Feature engineering pipeline (12 technical features) | ✅ feature-eng plugins operational |
| Preprocessing pipeline (normalize, split) | ✅ preprocessor plugins operational |
| Heuristic WFO (Path A) | ✅ heuristic-strategy `walk_forward_optimizer.py` |
| Predictor models (Path B: CNN, LSTM, TFT, TCN, etc.) | ✅ predictor 28 model plugins |
| NEAT hyperparameter optimizer | ✅ predictor `neat_optimizer` |
| DEAP GA optimizer | ✅ predictor `default_optimizer` |
| Regime classification (GMM K=9) | ✅ heuristic-strategy `regime_adaptive` plugin |
| Evaluation metrics and kill criteria | ✅ F-10 specification |
| Experiment tracking format | ✅ F-5 §7 CSV format |
| 3 machines with GPUs | ✅ Omega/Dragon/Gamma confirmed reachable |
| Synthetic data generators (Part V) | ✅ synthetic-datagen + timeseries-gan repos (newly discovered) |
| RL trading environment (Part IV) | ⚠️ gym-fx exists (needs gymnasium migration) + rl-optimizer (PPO) |
| Autoencoder feature compression | ✅ feature-extractor repo (ANN/CNN/LSTM/VAE/CVAE) |
| Prediction target generation | ✅ trading-signal repo (h_1..h_6, d_1..d_6 targets) |

### 4.2 What Needs Building (Part II Sprint 0)

| Task | Description | Effort | Priority |
|------|-------------|--------|----------|
| `rolling_orchestrator.py` | Window management + pipeline dispatch | 2-3 days | **CRITICAL** |
| `download_data.py` | Acquire + validate 2005-2024 EUR/USD 1h | 0.5 day | HIGH |
| `resample_data.py` | 1h → 4h + daily | 0.5 day | HIGH |
| `window_manifest_generator.py` | Generate anchored expanding windows | 0.5 day | HIGH |
| Feature-eng entry_points fix (G-4) | `.get()` → `.select()` | 30 min | MEDIUM |
| Embargo implementation | Add to window splitting | 2 hours | MEDIUM |
| Ridge + LightGBM baselines (F-2) | Simple model scripts | 1 day | MEDIUM |
| IC analysis script (F-2) | Feature diagnostic | 2 hours | LOW |

**Total Sprint 0 estimate**: 5-7 days of focused work.

### 4.3 Part II Experiment Order

| Order | Experiment | Path | Machine | Purpose |
|-------|-----------|------|---------|---------|
| 1 | Static baseline replay | N/A | Omega | Reproduce P1 results with new pipeline (validation) |
| 2 | Adaptive heuristic WFO (regime_adaptive) | A | Omega | First adaptive experiment — uses existing WFO |
| 3 | Rolling ML (TFT, single model) | B | Dragon | First ML rolling retraining — TFT is strongest P1 model |
| 4 | Rolling ML (CNN, LSTM, TCN) | B | Dragon/Gamma | Expand to other architectures |
| 5 | NEAT-HPO vs DEAP-GA comparison | B (sub) | Dragon | Optimizer comparison |
| 6 | Ridge/LightGBM baselines | B | Omega | Simple model baselines |
| 7 | Causal feature filter (CI-4) | B | Dragon | Reduced feature set |
| 8 | Multi-timeframe (daily) | A+B | Gamma | If CI-2 finds lagged links |

---

## 5. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **All paths produce negative held-out SR** | 35-50% | HIGH — project fails to demonstrate adaptive advantage | Pre-registered kill criteria (F-10). Document as negative result. Pivot to different asset/timeframe. |
| **Rolling retraining overfits to recent window** | 30-40% | MEDIUM — adaptive worse than static | Embargo (F-2 §3.1), DSR/PBO checks (F-10 §5.2), cost ratio kill (K-3) |
| **Compute budget exceeded** | 20% | LOW — experiments take longer | Parallelize across 3 machines. Start with single model type, expand if promising. |
| **4h timeframe truly has no predictable structure** | 50% | HIGH — all prediction-based paths fail | CI-2 (multi-timeframe) as escape hatch. Path A doesn't require prediction. |
| **Infrastructure breaks during rolling** (version conflicts, memory errors) | 20% | MEDIUM — delays | Sprint 0 includes end-to-end pilot run (F-5 §10). Reduced from 30% — 7 more repos available reduces rebuild risk. |
| **Regime model staleness** | 20% | LOW — regime_adaptive underperforms | UL-1 (rolling GMM re-fitting) |

---

## 6. Decision Points for User

### 6.1 Proceed to Part II?

**Recommendation**: YES, with the following caveats:
- The evidence for the core hypothesis is weaker than the evidence against
- The strongest path is Path A (adaptive heuristic) because it doesn't require prediction and has the most infrastructure
- Path B (ML rolling) is the most informative experiment but has the highest failure risk
- Path C (RL) should be deferred until Paths A and B results are available

### 6.2 Open Decisions Needed

| Decision | Options | Recommendation |
|----------|---------|---------------|
| **Start with Path A or B?** | A first (lower risk, faster) or B first (more informative) | A first — validates pipeline with existing WFO before adding ML complexity |
| **Crypto in scope?** | Include BTC/USD or FX-only | FX-only for Part II. Add BTC if Paths A/B show promise. |
| **RPCMCI fix timing** | Now (Part I) or Part II | Part II Sprint 0 — it's a bug fix, not foundational research |
| **Data download timing** | Now or Part II start | Now if possible — downloading 20 years of 1h data takes time |
| **Daily timeframe priority** | Test before Part II experiments or during | Before Part II — CI-2 results may change the 4h default |

### 6.3 What a "No" Would Mean

If the user decides NOT to proceed to Part II:
- Part I deliverables remain as research documentation
- The causal null result (F-6) and infrastructure audit (F-8) have standalone value
- The 11 repos + 114 plugins are documented and auditable
- The evaluation framework (F-10) can be applied to any future project

---

## 7. Cross-Reference Matrix

| Part I Finding | Affects Part | How |
|---------------|-------------|-----|
| C-1: F1 ≥ 0.91 | II (Path B binary models) | Kill criterion K-4 |
| C-4: Worst 2Y SR > −0.9 | II, III, IV | Kill criterion K-2 |
| D-10: Static unrealistic | II (core motivation) | Justifies adaptive approach |
| No lagged causal links at 4h (F-6) | II, III | Caution for prediction-based paths |
| GMM K=9 regime structure (F-7) | II (Path A) | Regime-adaptive plugin ready |
| G-1: No rolling orchestrator | II (all paths) | Sprint 0 critical build |
| Two NEAT systems (F-9) | II, VI | Use System B (custom HPO), document System A as legacy |
| Zero-cost data stack (F-3) | II | OANDA + FRED + HistData + yfinance |
| EUR/USD @ 4h primary (F-4) | II | All initial experiments on this |
| 2020-2024 held-out (F-4) | II, III, IV | Final evaluation, touch once only |
| Kill criteria K-1 to K-7 (F-10) | II, III, IV | Pre-registered, no modification allowed post-registration |
| Purged CV with embargo (F-2) | II (Path B) | Add embargo to window splits |
| Linear/tree baselines (F-2) | II (Path B) | Ridge + LightGBM alongside neural models |

---

## 8. File Manifest

All Part I deliverables located at:
```
heuristic-strategy/trading_research/project2/part_I_foundations/deliverables/
├── F1_PROJECT1_LESSONS_FOR_PROJECT2.md
├── F2_JANSEN_FRAMEWORK_INTEGRATION.md
├── F3_DATA_SOURCES_CATALOG.md
├── F4_ASSET_TIMEFRAME_SELECTION.md
├── F5_DATA_PIPELINE_SPEC.md
├── F6_CAUSAL_INFERENCE_OPPORTUNITIES.md
├── F7_UNSUPERVISED_LEARNING_OPPORTUNITIES.md
├── F8_INFRASTRUCTURE_AUDIT_PROJECT2.md
├── F9_GYMFX_NEAT_RETROSPECTIVE.md
├── F10_EVALUATION_FRAMEWORK.md
└── F11_PART_I_SYNTHESIS.md        ← this document
```
