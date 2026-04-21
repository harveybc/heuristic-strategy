# F-8: Infrastructure Audit for Project 2

**Date**: 2025-06-17 (original), **CORRECTED**: 2026-04-18  
**Scope**: All 18 repos under `/home/harveybc/Documents/GitHub/` assessed for Project 2 readiness  
**Method**: Automated code audit via subagent — pyproject.toml, plugin directories, test suites, config systems, rolling/adaptive capabilities reviewed per repo  
**Correction note**: Original audit covered 11 repos. 7 additional repos (gym-fx, synthetic-datagen, timeseries-gan, rl-optimizer, feature-extractor, trading-signal, agent-multi) were discovered on GitHub, cloned, and audited on 2026-04-18. Gaps G-2 and G-3 are now **RESOLVED/DOWNGRADED**. 2 repos (doin-evaluator, doin-optimizer) confirmed obsolete (merged into doin-node).

---

## 1. Per-Repo Audit

### 1.1 feature-eng

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MINOR GAPS** |
| **Plugins** | 5 registered: `tech_indicator`, `oracle_labels`, `direction_labels`, `ssa`, `fft` |
| **Plugin loader** | `importlib.metadata.entry_points().get()` (Python 3.8+ compat) |
| **Python** | ≥3.8 (implicit) |
| **Key deps** | numpy, pandas, pandas_ta, keras |
| **Config** | JSON (defaults → file → CLI) |
| **I/O** | CSV in/out, DATE_TIME index + numeric columns |
| **Test files** | ~12 files, ~60% estimated coverage |
| **Rolling/adaptive** | **None**. Feature generation is stateless per run. No incremental or rolling feature computation. |
| **Gaps for P2** | (1) No rolling feature generation — must re-run full pipeline for each retraining window. (2) `entry_points().get()` deprecated in Python 3.12+, needs `.select()` migration. (3) No streaming/incremental mode for live feature updates. |

### 1.2 preprocessor

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Plugins** | 6 registered: `default`, `normalizer`, `unbiaser`, `trimmer`, `feature_selector`, `cleaner` + 3 unregistered |
| **Plugin loader** | Advanced `PluginLoader` with isolation; `importlib.metadata.entry_points()` |
| **Python** | ≥3.8 (implicit) |
| **Key deps** | numpy, pandas, scikit-learn |
| **Config** | JSON (defaults → file → CLI) |
| **I/O** | CSV in/out. D1–D6 dataset splitting (33%/8.3%/8.3%/33%/8.3%/8.3%) |
| **Test files** | ~15 files, ~55% estimated coverage |
| **Rolling/adaptive** | **Partial**. Rolling features supported (rolling_std, rolling_ema). Normalization params save/load for retraining support. |
| **Gaps for P2** | (1) No windowed re-normalization orchestration — params save/load exists but caller must manage window boundaries. (2) D1–D6 split is static; rolling WFO would need custom split logic. |

### 1.3 predictor

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MINOR GAPS** |
| **Plugins** | 40 total — 28 predictor (regression/binary/direction × ANN/CNN/LSTM/Transformer/N-BEATS/TFT/TCN/MIMO), 2 optimizer (DEAP GA, NEAT), 4 pipeline, 2 preprocessor, 4 target |
| **Plugin loader** | `importlib.metadata.entry_points()` |
| **Python** | **≥3.12** (explicit in pyproject.toml) |
| **Key deps** | tensorflow, keras, numpy, pandas, deap, neat-python |
| **Config** | JSON (defaults → file → CLI) |
| **I/O** | CSV in, .keras model files out. Sliding windows (window_size=256 default) |
| **Test files** | ~20 files, ~40% estimated coverage |
| **Rolling/adaptive** | **Partial**. Model save/load (.keras). Warm-start via `initial_epoch`. Sliding windows built-in. NO rolling retraining loop — single-pass train/evaluate. |
| **Base classes** | `BasePredictorPlugin` → `BaseKerasPredictor` → `BaseBayesianKerasPredictor`. TensorFlow/Keras only. |
| **Gaps for P2** | (1) No rolling retraining orchestrator. (2) Python ≥3.12 requirement may conflict with some deps. (3) Test coverage lowest among critical repos. (4) No early-stopping across rolling windows. (5) No concept of "retraining trigger" (time-based or performance-based). |

### 1.4 prediction_provider

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Plugins** | 18 total — 2 core (FastAPI), 5 endpoint, 1 feeder, 9 predictor, 1 pipeline |
| **Plugin loader** | `importlib.metadata.entry_points()` |
| **Python** | **≥3.12** |
| **Key deps** | fastapi, uvicorn, tensorflow, yfinance, sqlalchemy, pydantic, python-jose |
| **Config** | JSON + env vars |
| **I/O** | REST API (JSON), SQLite DB, CSV feeder |
| **Test files** | ~25 files, ~75% estimated coverage (best of all repos, includes security tests) |
| **Rolling/adaptive** | **Partial**. Model hot-loading supported (model_cache_size: 5). Feature replication at inference time. JWT auth, rate limiting. NO auto-retraining — models swapped externally. |
| **Gaps for P2** | (1) No auto-retrain trigger. (2) Model versioning is implicit (filesystem path), not tracked in DB. (3) No A/B testing framework for comparing model versions. |

### 1.5 heuristic-strategy

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Plugins** | 5 strategy: `ls_pred_strategy`, `api_predictions`, `direction_atr`, `regime_adaptive`, `regime_wfo` + 4 embedded predictor plugins (ANN/CNN/LSTM/Transformer) |
| **Plugin loader** | `importlib.metadata.entry_points()` group `heuristic_strategy.plugins` |
| **Python** | ≥3.8 (implicit) |
| **Key deps** | numpy, pandas, backtrader, deap, requests, scipy, h5py, keras/tensorflow |
| **Config** | JSON (defaults → file → CLI), merged via config_merger |
| **I/O** | CSV in (DATE_TIME + OHLC + predictions). Out: trades.csv, summary.csv, parameters.json, balance_plot.png |
| **Test files** | 10 files (7 unit + 3 integration), ~40-50% estimated coverage |
| **Rolling/adaptive** | **Yes**. Walk-Forward Optimizer with anchored expanding-window, yearly folds, GA optimization per training slice, OOS evaluation. `regime_adaptive` uses GMM clustering (K=9 → 6 tradeable regimes). `regime_wfo` uses causal features for regime classification. WFO scripts: run_wfo.py, deploy_wfo.sh, merge_wfo_results.py. |
| **Gaps for P2** | (1) WFO is heuristic-strategy-specific, not reusable across repos. (2) No ML model retraining inside WFO — only parameter re-optimization. (3) Acceptance/system tests empty. |

### 1.6 lts (Live Trading System)

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MINOR GAPS** |
| **Plugins** | 7 groups: AAA, core, pipeline, strategy (6 plugins), broker (4: default, backtrader, backtrader_sim, OANDA), portfolio + feeder/predictor |
| **Plugin loader** | `importlib.metadata.entry_points().select()` (Python 3.12+ compatible) |
| **Python** | ≥3.8 (implicit, uses asyncio/aiosqlite) |
| **Key deps** | pandas, tensorflow, backtrader, deap, fastapi, uvicorn, sqlalchemy, aiosqlite, httpx, oandapyV20, hypothesis |
| **Config** | JSON (defaults → file → CLI), supports remote config URLs |
| **I/O** | CSV in (DATE_TIME + OHLC). SQLite DB (Users, Portfolios, Assets, Orders, Positions). REST API (FastAPI + JWT + RBAC). |
| **Test files** | 16+ files (unit, integration, acceptance, system, security), ~60-70% estimated coverage |
| **Rolling/adaptive** | **No**. LTS is a live execution framework. `prediction_client.py` calls external Prediction Provider API. Pipeline architecture allows hot-swap but no retraining orchestration. |
| **Gaps for P2** | (1) No automated model refresh cycle. (2) Strategy plugin switching requires restart or manual config change. (3) No performance monitoring → retrain trigger loop. |

### 1.7 doin-core

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Plugins** | 3 abstract base classes: `OptimizationPlugin`, `InferencePlugin`, `SyntheticDataPlugin`. Entry point groups: `doin.optimization`, `doin.inference`, `doin.synthetic_data`. |
| **Plugin loader** | [loader.py](doin-core/src/doin_core/plugins/loader.py) — `load_optimization_plugin()`, `load_inference_plugin()`, `load_synthetic_data_plugin()`, `list_plugins()`. Type-validated. |
| **Python** | **≥3.10** (explicit) |
| **Key deps** | pydantic ≥2.0, cryptography ≥41.0 |
| **Config** | Plugin-specific `dict[str, Any]` passed to `configure()` |
| **I/O** | Pydantic models for all domain objects (Block, Optimae, Transaction, Task). Parameters as `dict[str, Any]`. |
| **Test files** | 20 files, ~80-90% estimated coverage (best test coverage) |
| **Rolling/adaptive** | **Yes by design**. `OptimizationPlugin.optimize()` receives `current_best_params` + `current_best_performance`, returns improved parameters. Iteratively called by node. Reputation tracking with EMA decay. |
| **Gaps for P2** | (1) No domain-specific logic — that lives in doin-plugins. (2) Consensus layer adds overhead not needed for single-machine P2 experiments. |

### 1.8 doin-node

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Modules** | blockchain/chain, network (transport, flooding, gossip, peer, discovery, sharding, sync), scheduling/gpu_scheduler, stats (experiment_tracker, olap_db), storage (chaindb, migrate), validation, dashboard |
| **Python** | **≥3.10** |
| **Key deps** | doin-core ≥0.1.0, aiohttp ≥3.9, aiosqlite ≥0.20, psutil ≥5.9 |
| **Config** | JSON config files. 31 example configs in `examples/` (quadratic, predictor, cross-machine). |
| **I/O** | CSV experiment tracker (50+ columns). SQLite OLAP DB (WAL mode, star-schema, auto-migrating). SQLite chain DB. |
| **Test files** | 15 files, ~75-85% estimated coverage |
| **Rolling/adaptive** | **Yes**. Island-model migration (share champion params between nodes). Continuous optimization loop. Staged optimization (incremental complexity). `_force_stage_advance` for network-synchronized stage transitions. Full experiment tracking per round (CSV + OLAP). |
| **Gaps for P2** | (1) Designed for distributed P2P — may be overkill for single-machine rolling retraining. (2) GPU scheduler may conflict with direct TF GPU allocation. (3) Need to separate "optimization orchestrator" capability from "blockchain/consensus" overhead. |

### 1.9 doin-plugins

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MINOR GAPS** |
| **Plugins** | 8 registered across 3 groups: optimization (simple_quadratic, predictor, binary_predictor), inference (same 3), synthetic_data (simple_quadratic, predictor) |
| **Python** | **≥3.10** |
| **Key deps** | doin-core ≥0.1.0, numpy ≥1.24. Runtime: predictor, preprocessor, tensorflow/keras, sklearn (loaded dynamically) |
| **I/O** | Quadratic: dict params + float metric. Predictor: CSV timeseries, base64-encoded .keras models, dict metrics. Synthetic: pandas DataFrame (typical_price), SHA-256 hashed. |
| **Test files** | 6 files, ~60-70% estimated coverage |
| **Rolling/adaptive** | **Yes**. `PredictorOptimizer` wraps DEAP GA with staged optimization, island-model migration callbacks (champion inject IN/broadcast OUT), generation-level hooks. Thread-safe. `BinaryPredictorOptimizer` subclass for binary classification. Each DOIN round = one retraining cycle. |
| **Gaps for P2** | (1) Tightly coupled to DOIN consensus — needs extraction for standalone rolling retraining. (2) Runtime deps not declared in pyproject.toml. (3) No standalone "run one optimization round" entry point — requires full DOIN node. (4) Synthetic data plugin loads pre-trained HMM (.joblib) — model not included in repo. |

### 1.10 causal-inference

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MAJOR GAPS** |
| **Plugins** | 6 plugins in 3 groups: optimizers (openrl, neat, neat_p2p), environments (prediction), agents (openrl_ppo, dummy_automation) |
| **Analysis scripts** | 9 standalone: causal_regime_analysis (v1/v2), cluster_regime_analysis, cross_asset_audit, cross_asset_comparison, momentum_analysis, nfp_event_response_poc, path_watcher, run_rpcmci_only |
| **Plugin loader** | `importlib.metadata.entry_points()` with `.select()` |
| **Python** | ≥3.6 (stated), effectively ≥3.8+ |
| **Key deps** | tensorflow-gpu, scikit-learn, neat-python, stable-baselines3, gym, tigramite, ortools, dowhy, econml, gcastle |
| **Config** | JSON (same pattern as other repos) |
| **I/O** | CSV timeseries in, JSON + CSV results out |
| **Test files** | 9 files (6 unit + 3 integration), ~30-40% estimated coverage |
| **Rolling/adaptive** | NEAT P2P supports iterative evolutionary optimization. Analysis scripts are offline/one-shot. |
| **Gaps for P2** | (1) Repo is a hybrid RL-optimizer + causal-analysis toolkit with inconsistent naming (`rl_optimizer` entry points). (2) Lowest test coverage. (3) `tensorflow-gpu` dep is legacy — modern TF auto-detects GPU. (4) `gym` is deprecated in favor of `gymnasium`. (5) Analysis scripts have hardcoded paths and are not pipeline-integrated. (6) No integration with preprocessor/feature-eng for feature-to-causality pipeline. |

### 1.11 gym-fx (CORRECTED 2026-04-18 — repo EXISTS on GitHub)

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MAJOR GAPS — needs modernization** |
| **Status** | Exists at `github.com/harveybc/gym-fx` (208 stars, 54 forks). Last commit: 2023-01-08. Now cloned locally. |
| **Plugins** | **None** — no entry_points, no plugin system. Standalone scripts. |
| **Environments** | 14 registered gym envs (`ForexTrainingSet-v0` through `ForexTrainingSet12-v0`, `ForexValidationSet-v0`). 6 env versions (`ForexEnv` through `ForexEnv6`). |
| **Agents** | NEAT (`agent_NEAT.py`, `agent_NEAT_p2p.py`), DDQN (`agent_DDQN.py`). No PPO/SAC. |
| **NEAT integration** | Extensive — 38 inputs → 3 outputs (buy/sell/nop), custom `AgentGenome` with learnable discount rate, `PopulationSyn` for P2P genome sync via REST API. |
| **Key deps** | `gym` (old API, NOT gymnasium), `neat-python`, `keras` (standalone, NOT tf.keras), `tensorflow` (TF1: `tf.Session`). |
| **Data format** | CSV, no headers, 16 columns (OHLC bid + volume + datetime fields + 6 indicators). Incompatible with current pipeline. |
| **Gaps for P2** | (1) Old `gym` API (4-tuple step, not 5-tuple gymnasium). (2) TF1/standalone Keras — incompatible. (3) No plugin architecture. (4) No header CSV format. (5) Stale 3+ years. **Valuable for concepts** (reward function design, P2P genome sync → doin-core, NEAT-as-agent pattern) but code needs full rewrite. |

### 1.12 synthetic-datagen (CORRECTED 2026-04-18 — repo EXISTS on GitHub)

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Status** | Active development. Last commit: 2026-02-18. Now cloned locally. |
| **Plugins** | 8 registered: 3 trainers (`vae_trainer`, `gan_trainer`, `vae_gan_trainer`), 1 generator (`typical_price_generator`), 3 evaluators (`distribution_evaluator`, `predictive_evaluator`, `augmentation_evaluator`), 1 optimizer (`ga_optimizer`). |
| **Unregistered code** | 7 additional generators NOT registered as entry_points: `BlockBootstrapGenerator`, `GrasyndaGenerator`, `RegimeBootstrapHybrid`, `RegimeConditional`, `RegimeGAN`, `RegimeHmmGarch`, `TimeGANGenerator`. |
| **Plugin loader** | `importlib.metadata.entry_points().select()` — identical pattern to predictor. |
| **Key deps** | numpy, pandas, tensorflow ≥2.14, scipy, scikit-learn, deap. Runtime (undeclared): hmmlearn, ruptures. |
| **Data format** | CSV with `DATE_TIME,typical_price` columns. 4h periodicity. Log-returns internally. Window size default 144 (24 days @ 4h). |
| **Evaluators** | Distribution (JSD, Wasserstein, ACF), Predictive utility (**thesis metric**: prepend synthetic → train → compare MAE delta), Augmentation (uses actual predictor repo as subprocess). |
| **Optimizer** | DEAP-based staged incremental GA (5 stages, tournament selection, resume support). |
| **Gaps for P2** | (1) 7 unregistered generators need entry_points. (2) hmmlearn/ruptures not in deps. (3) Best-performing generator (HMM+bootstrap hybrid) is unregistered. |

### 1.13 timeseries-gan (CORRECTED 2026-04-18 — repo EXISTS on GitHub)

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MINOR GAPS** |
| **Status** | Last commit: 2025-06-21. 371 commits. Now cloned locally. |
| **Plugins** | 6 registered: `default_feeder`, `vae_generator`/`default_generator`, `default_discriminator`, `default_evaluator`, `default_optimizer`, `gan_trainer`. |
| **Plugin loader** | `importlib.metadata.entry_points()` — same pattern. |
| **Architecture** | SC-VAE-GAN: BiLSTM Z-generator → pre-trained VAE decoder → 23 base features → Conv1D+BiLSTM discriminator. |
| **Key deps** | tensorflow, numpy, pandas, h5py, scipy, deap, pandas_ta, statsmodels. |
| **Data format** | CSV, 45 columns (DATE_TIME + 44 features: OHLC + 15 TIs + bid/ask spreads + S&P500 + VIX + 16 sub-periodicity ticks + datetime). 1h periodicity. |
| **Relationship to synthetic-datagen** | Predecessor. timeseries-gan = multi-feature (23 cols) SC-VAE-GAN. synthetic-datagen = simpler (1 col typical_price) with multiple model types (VAE, GAN, TimeGAN, regime-based). Not redundant — timeseries-gan is the only option for multi-feature OHLC generation. |
| **Gaps for P2** | (1) 707 MB repo (large model files). (2) 1h periodicity (P2 primary is 4h). (3) Older codebase. |

### 1.14 rl-optimizer (CORRECTED 2026-04-18 — repo EXISTS on GitHub)

| Attribute | Value |
|-----------|-------|
| **Readiness** | **MINOR GAPS** |
| **Status** | Last commit: 2025-01-13. Now cloned locally. |
| **Plugins** | 3 groups: optimizers (`openrl`, `neat`*, `neat_p2p`*), environments (`prediction`), agents (`openrl_ppo`, `dummy_automation`). *NEAT plugins registered but files MISSING from disk. |
| **Plugin loader** | `importlib.metadata.entry_points()` — same pattern. |
| **RL algorithms** | PPO (PyTorch actor-critic via OpenRL). No DQN/SAC. |
| **Environment** | `PredictionEnv(gym.Env)` — regression environment: obs=feature row, action=continuous prediction, reward=1/MAE. |
| **Key deps** | numpy, tensorflow-gpu, pandas, openrl, neat-python, stable-baselines3, gym, scikit-learn. |
| **Gaps for P2** | (1) 2 registered NEAT optimizer plugins missing from disk. (2) Uses old `gym` not `gymnasium`. (3) `PredictionEnv` is regression-only, no trading actions. |

### 1.15 feature-extractor

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Status** | Last commit: 2025-06-21. Active (CVAE branch). Now cloned locally. |
| **Plugins** | 14 registered: 7 encoders + 7 decoders (`ann`, `cnn`, `lstm`, `transformer`, `vae`, `vae_small`, `cnn_signed`). |
| **Plugin loader** | `importlib.metadata.entry_points()` — same pattern. |
| **Purpose** | Autoencoder-based dimensionality reduction. Trains encoder-decoder pairs (ANN, CNN, LSTM, Transformer, VAE, CVAE) on normalized data. |
| **Key deps** | tensorflow-gpu, numpy, pandas, h5py, scipy, keras-multi-head. |
| **Relationship to feature-eng** | Complementary, not competing. feature-eng creates features/labels (Phase 1). feature-extractor compresses features via autoencoders (Phase 3-4). Pipeline: preprocessor → feature-eng → feature-extractor → predictor. |
| **Gaps for P2** | (1) `tensorflow-gpu` dep is legacy. (2) No pyproject.toml entry points (uses setup.py). |

### 1.16 trading-signal

| Attribute | Value |
|-----------|-------|
| **Readiness** | **READY** |
| **Status** | Last commit: 2025-02-13. Stable. Now cloned locally. |
| **Plugins** | 2 registered: `default_plugin` (hourly+daily shifted targets), `ls` (OHLC + rolling std + daily aggregation for NEAT). |
| **Plugin loader** | `importlib.metadata.entry_points()` — same pattern. |
| **Purpose** | Generates shifted prediction targets (`Prediction_h_1..h_6`, `Prediction_d_1..d_6`) from CLOSE prices. |
| **Key deps** | pandas, numpy (minimal). |
| **Data format** | Input: CSV with DATE_TIME, OPEN, HIGH, LOW, CLOSE. Output: CSV with shifted prediction columns. |
| **Gaps for P2** | None significant. Lightweight utility. |

### 1.17 agent-multi (LEGACY)

| Attribute | Value |
|-----------|-------|
| **Readiness** | **LEGACY — superseded by rl-optimizer** |
| **Status** | Last commit: 2020-01-17. Only 4 commits. Abandoned. |
| **Plugins** | **None** — no entry_points, no plugin architecture. Standalone scripts. |
| **Purpose** | Multi-agent NEAT-evolved forex trading using gym-fx environments (ForexEnv4, ForexEnv7). |
| **Agents** | NEAT (`agent_NEAT.py`, `agent_multi.py`, `agent_NEAT_p2p.py`), DQN (`q_agent_dcn_ema1020close.py`). |
| **Relationship** | Direct predecessor to rl-optimizer + doin-node. The P2P genome migration concept (`PopulationSyn`) reappears in doin-core. |
| **Gaps for P2** | Obsolete. No plugin arch, old gym, old Keras. Historical reference only. |

### 1.18 Note on Obsolete Repos

| Repo | Status | Notes |
|------|--------|-------|
| `doin-evaluator` | **Obsolete** — merged into doin-node | User confirmed 2026-04-18 |
| `doin-optimizer` | **Obsolete** — merged into doin-node | User confirmed 2026-04-18 |

---

## 2. Cross-Cutting Concerns

### 2.1 Python Version Compatibility

| Constraint | Repos |
|------------|-------|
| **≥3.12** (strictest) | predictor, prediction_provider |
| **≥3.10** | doin-core, doin-node, doin-plugins |
| **≥3.8** (implicit) | feature-eng, preprocessor, heuristic-strategy, lts, causal-inference |

**Verdict**: All repos run on Python 3.12. The Omega machine uses Python 3.12.7 via conda `tensorflow` environment. No conflict — but causal-inference's `tensorflow-gpu` and `gym` packages need updating for 3.12 compatibility.

### 2.2 Data Format Compatibility

All repos share a common data contract:
- **Input**: CSV files with `DATE_TIME` column (datetime index) + numeric columns
- **Config**: JSON files with defaults → file → CLI override pattern
- **Model artifacts**: `.keras` files (predictor), `.joblib` files (doin-plugins synthetic)
- **Parameters**: JSON dictionaries

**Verdict**: **Compatible**. The CSV + JSON convention is universal. No format translation needed between repos.

### 2.3 Plugin Architecture Compatibility

All repos use `importlib.metadata.entry_points()` for plugin discovery. Two loader variants exist:
1. **Legacy**: `entry_points().get(group)` — feature-eng (deprecated in Python 3.12)
2. **Modern**: `entry_points(group=...)` or `.select(group=...)` — all others

**Verdict**: feature-eng needs `.get()` → `.select()` migration. All other repos are compatible.

### 2.4 Config Management

Uniform pattern across all repos:
```
defaults (config.py) → JSON file (--load_config) → CLI args
```
Implemented via `config_handler.py` + `config_merger.py` in each repo.

**Verdict**: **Compatible**. A rolling retraining orchestrator can generate JSON configs per window and pass them to any repo's CLI.

### 2.5 Logging & Observability

| Repo | Logging |
|------|---------|
| predictor | TensorFlow/Keras built-in (epoch logs), CSV metrics output |
| prediction_provider | Python logging + SQLite DB |
| heuristic-strategy | Print-based + CSV trade/summary output |
| lts | Python logging + SQLite DB + FastAPI dashboard |
| doin-node | CSV experiment tracker (50+ columns) + OLAP SQLite DB |
| doin-core | None (library) |
| Others | Print-based or minimal |

**Verdict**: **Inconsistent**. No unified logging framework. Project 2 rolling retraining will need a consistent experiment tracking layer. doin-node's experiment tracker is the most mature and could be extracted.

### 2.6 GPU Resource Management

| Mechanism | Repo |
|-----------|------|
| `TF_FORCE_GPU_ALLOW_GROWTH=true` | doin-node CLI |
| `TF_GPU_ALLOCATOR=cuda_malloc_async` | doin-node CLI |
| Keras `tf.config.experimental.set_memory_growth` | predictor (in some plugins) |
| None | feature-eng, preprocessor, heuristic-strategy |

**Verdict**: GPU memory management is ad-hoc. Multi-machine rollout (Omega/Dragon/Gamma) needs standardized GPU allocation. Risk of OOM if two retraining jobs land on the same GPU.

---

## 3. Capability Matrix for Project 2 Paths

| Capability | Path A (Adaptive Heuristic) | Path B (Supervised ML Rolling) | Path C (RL) | Part V (Synthetic) | Part VI (NEAT) |
|------------|----------------------------|-------------------------------|-------------|--------------------|--------------------|
| Feature generation | feature-eng ✅ | feature-eng ✅ | feature-eng ✅ | feature-eng ✅ | feature-eng ✅ |
| Preprocessing/normalization | preprocessor ✅ | preprocessor ✅ | preprocessor ✅ | preprocessor ✅ | preprocessor ✅ |
| Model training | N/A | predictor ✅ | causal-inference ⚠️ | predictor ✅ | predictor (NEAT optimizer) ✅ |
| Parameter optimization | heuristic-strategy (WFO) ✅ | predictor (DEAP GA) ✅ | causal-inference (NEAT) ⚠️ | N/A | predictor (NEAT) ✅ |
| Walk-forward orchestration | heuristic-strategy ✅ | **MISSING** ❌ | **MISSING** ❌ | N/A | **MISSING** ❌ |
| Live inference | prediction_provider ✅ | prediction_provider ✅ | **MISSING** ❌ | N/A | prediction_provider ⚠️ |
| Backtesting | heuristic-strategy (backtrader) ✅ | heuristic-strategy ✅ | **MISSING** ❌ | heuristic-strategy ✅ | heuristic-strategy ✅ |
| Experiment tracking | doin-node (CSV+OLAP) ✅ | doin-node ✅ | doin-node ✅ | doin-node ✅ | doin-node ✅ |
| Distributed optimization | doin-core + doin-node ✅ | doin-node ✅ | **MISSING** ❌ | N/A | doin-node ✅ |
| Synthetic data generation | N/A | N/A | N/A | synthetic-datagen + timeseries-gan ✅ | N/A |
| Causal feature selection | causal-inference ⚠️ | causal-inference ⚠️ | causal-inference ⚠️ | N/A | N/A |

Legend: ✅ = exists and usable, ⚠️ = exists but needs work, ❌ = does not exist

---

## 4. Gap Summary Table

| # | Gap Description | Affected Repo(s) | P2 Part(s) | Est. Size | Blocker? |
|---|----------------|-------------------|------------|-----------|----------|
| G-1 | **No rolling retraining orchestrator** — pieces exist (feature regen, norm save/load, model save/load, warm-start) but no coordinating loop that manages window boundaries, triggers retraining, and swaps models | Cross-repo | II, III, IV | **LARGE** | **YES** — critical for all ML-based rolling experiments |
| G-2 | ~~No synthetic data generation code~~ **RESOLVED (2026-04-18)**: `synthetic-datagen` repo exists with VAE/GAN/VAE-GAN/TimeGAN/regime generators + plugin architecture. `timeseries-gan` exists with SC-VAE-GAN (23-feature). **Remaining gap**: 7 generators unregistered as entry_points, hmmlearn/ruptures deps undeclared. | synthetic-datagen, timeseries-gan | V | ~~LARGE~~ SMALL | **NO** — code exists, needs minor cleanup |
| G-3 | ~~No RL environment for trading~~ **DOWNGRADED (2026-04-18)**: `gym-fx` repo exists (208 stars) with 14 gym envs + NEAT/DDQN agents. `rl-optimizer` exists with PPO + PredictionEnv. **Remaining gap**: both use old `gym` API (not gymnasium), TF1/standalone Keras in gym-fx. Need modernization, not rebuild. | gym-fx, rl-optimizer, agent-multi | IV, VI | ~~LARGE~~ MEDIUM | **NO** — code exists, needs API migration |
| G-4 | **feature-eng `entry_points().get()` deprecated** — will break on Python 3.12+ | feature-eng | All | SMALL | No — easy migration to `.select()` |
| G-5 | **Inconsistent logging/observability** — no unified experiment tracking across repos. Print-based logging in several repos | Cross-repo | All | MEDIUM | No — but limits reproducibility |
| G-6 | **causal-inference repo quality** — lowest test coverage (30-40%), legacy deps (tensorflow-gpu, gym), hardcoded paths in analysis scripts, inconsistent naming (rl_optimizer entry points) | causal-inference | III, IV, VI | MEDIUM | No — usable but needs cleanup |
| G-7 | **No performance monitoring → retrain trigger** — no repo implements "monitor live performance and trigger retraining when degraded" | lts, prediction_provider | II, III | MEDIUM | No for backtesting, **YES** for live deployment |
| G-8 | **GPU resource management ad-hoc** — no standardized GPU allocation across machines. Risk of OOM with concurrent training jobs | predictor, doin-node | All | SMALL | No — manageable with env vars |
| G-9 | **WFO only exists for heuristic strategies** — heuristic-strategy's walk_forward_optimizer.py is tightly coupled to GA parameter optimization, not ML model retraining | heuristic-strategy | II, III | MEDIUM | No — but needs generalization or parallel implementation for ML |
| G-10 | **predictor has lowest test coverage (~40%)** among critical repos — the most complex repo (40 plugins) is the least tested | predictor | II, III, V, VI | MEDIUM | No — but increases risk of regressions during P2 modifications |
| G-11 | **No model versioning / A/B framework** — prediction_provider supports hot-loading but doesn't track model versions in DB or support comparison testing | prediction_provider | II, III | SMALL | No |
| G-12 | **doin-plugins runtime deps undeclared** — predictor, preprocessor, tensorflow, sklearn loaded dynamically but not in pyproject.toml | doin-plugins | All (DOIN-based) | SMALL | No — works at runtime but blocks clean `pip install` |
| G-13 | **heuristic-strategy and causal-inference acceptance/system tests empty** — test scaffolding exists but no tests implemented | heuristic-strategy, causal-inference | II, IV | SMALL | No |

---

## 5. Recommendations for Project 2

### 5.1 Critical Path Items (Must-Build Before Experiments)

1. **Rolling Retraining Orchestrator (G-1)**: Build a lightweight orchestrator script that:
   - Defines rolling windows (anchored expanding or sliding)
   - For each window: calls feature-eng → preprocessor → predictor → evaluates on OOS
   - Manages model versioning (save each window's model)
   - Logs results in doin-node experiment tracker format
   - Consider extracting doin-node's experiment_tracker.py as a standalone module

2. **Synthetic Data Pipeline (G-2) — RESOLVED**: `synthetic-datagen` has VAE-GAN trainer + predictive utility evaluator + DEAP GA optimizer. `timeseries-gan` has SC-VAE-GAN for multi-feature generation. **Remaining work**: register 7 unregistered generators as entry_points, add hmmlearn/ruptures to deps, verify 4h compatibility (synthetic-datagen uses 4h, timeseries-gan uses 1h).

3. **RL Trading Environment (G-3) — DOWNGRADED**: `gym-fx` has 14 gym environments with NEAT + DDQN agents. `rl-optimizer` has PPO + PredictionEnv. **Remaining work**: migrate gym → gymnasium API, modernize TF1 → TF2/PyTorch in gym-fx, integrate gym-fx observation format with feature-eng pipeline output.

### 5.2 Quick Wins (Pre-Experiment Cleanup)

4. Fix feature-eng `entry_points().get()` → `.select()` (G-4) — 30 min
5. Add `TF_FORCE_GPU_ALLOW_GROWTH=true` to all training scripts (G-8) — 15 min
6. Declare doin-plugins runtime deps in pyproject.toml (G-12) — 15 min

### 5.3 Deferred (Address During Experiments)

7. Unified experiment tracking (G-5) — adopt doin-node's tracker incrementally
8. causal-inference cleanup (G-6) — as Part IV work begins
9. Model versioning (G-11) — as live deployment approaches
10. Test coverage improvements (G-10, G-13) — alongside code changes

---

## 6. Infrastructure Reuse Summary

| Existing Asset | Reusable For | Reuse Effort |
|---------------|-------------|--------------|
| heuristic-strategy WFO | Path A (adaptive heuristic) | Low — already works |
| predictor 40 plugins | Path B (ML rolling), Part V, Part VI | Low — training API stable |
| predictor DEAP GA optimizer | Path B hyperparameter tuning | Low — already integrated |
| predictor NEAT optimizer | Part VI (NEAT comparison) | Low — already integrated |
| preprocessor norm save/load | All rolling paths — preserve normalization across windows | Low — API exists |
| prediction_provider hot-loading | All paths — deploy retrained models | Low — API exists |
| doin-node experiment tracker | All paths — unified metrics logging | Medium — needs extraction from DOIN context |
| doin-plugins PredictorOptimizer | Distributed rolling retraining | Medium — needs DOIN consensus removal for standalone use |
| heuristic-strategy regime_adaptive | Path A — GMM-based regime detection | Low — plugin ready |
| causal-inference PCMCI scripts | Feature selection / regime analysis | Medium — scripts need parameterization |
| lts OANDA broker | Live deployment | Low — broker plugin ready |
| lts backtrader broker | Backtesting | Low — broker plugin ready |

---

## 7. Machine Inventory

| Machine | GPU | SSH Access | Conda Env | Status |
|---------|-----|-----------|-----------|--------|
| Omega (local) | RTX 4070 (12 GB) | N/A | `tensorflow` (Python 3.12.7) | ✅ Primary dev |
| Dragon | RTX 4090 (24 GB) | `192.168.1.235:62024` | tensorflow | ✅ Confirmed reachable |
| Gamma | RTX 5070 Ti (16 GB) | `192.168.0.106:62024` | tensorflow | ✅ Confirmed reachable |

**Total GPU memory**: 52 GB across 3 machines. Sufficient for parallel rolling retraining experiments (Path A on Omega, Path B on Dragon, etc.).

---

## Appendix: Raw Plugin Count by Repo

| Repo | Registered Plugins | Unregistered/Embedded | Total |
|------|-------------------|-----------------------|-------|
| feature-eng | 5 | 0 | 5 |
| preprocessor | 6 | 3 | 9 |
| predictor | 40 | 0 | 40 |
| prediction_provider | 18 | 0 | 18 |
| heuristic-strategy | 5 | 4 | 9 |
| lts | 14 | 2 | 16 |
| doin-core | 3 (ABCs) | 0 | 3 |
| doin-node | 0 | 0 | 0 |
| doin-plugins | 8 | 0 | 8 |
| causal-inference | 6 | 0 | 6 |
| synthetic-datagen | 8 | 7 | 15 |
| timeseries-gan | 6 | 0 | 6 |
| rl-optimizer | 5 | 0 | 5 |
| feature-extractor | 14 | 0 | 14 |
| trading-signal | 2 | 0 | 2 |
| gym-fx | 0 | 3 (standalone agents) | 3 |
| agent-multi | 0 | 4 (standalone agents) | 4 |
| **Total** | **140** | **23** | **163** |
