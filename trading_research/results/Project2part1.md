# Project 2 — Part I Foundations: Execution Work Plan

**For:** User's agent executing on Omega, Dragon, and Gamma
**Reference:** `project_2_master_plan.md` (approved Phase 0 master plan)
**Status:** Ready for execution
**Gate to Part II:** All Part I tasks completed and synthesized in `PART_I_FOUNDATIONS_SYNTHESIS.md`

---

## Overview

Part I executes 11 foundation tasks across 3 machines in parallel where possible, sequential where required by dependencies. No experiments, no backtests, no code for trading strategies. This is purely preparation: knowledge consolidation, research, analysis, documentation, and infrastructure audit.

**Machine roles for Part I:**

- **Omega (RTX 4070 12GB, full conda env):** Coordination, synthesis, infrastructure audit (most repos are here)
- **Dragon (RTX 4090 16GB, fastest):** Heavy analytical tasks, Jansen book review (Dragon can run local LLM for book summarization if needed)
- **Gamma (RTX 5070 Ti 12GB, fast):** Parallel research tasks, data sources exploration

**Parallelization strategy:** Tasks that can run independently run in parallel on separate machines. Tasks with dependencies wait for their prerequisites.

---

## Dependency Graph

```
WAVE 1 (all three machines in parallel, no dependencies):
├── Omega:  Task F-1 (Project 1 Knowledge Consolidation)
├── Dragon: Task F-2 (Jansen Integration Mapping)
└── Gamma:  Task F-3 (Data Sources Catalog)

WAVE 2 (depends on Wave 1 completion):
├── Omega:  Task F-8 (Infrastructure Audit)   [independent of F-1/F-2/F-3 but sequenced here]
├── Dragon: Task F-6 (Causal Inference Opportunities)  [depends on F-2 Jansen mapping]
└── Gamma:  Task F-7 (Unsupervised Opportunities)       [depends on F-2 Jansen mapping]

WAVE 3 (depends on Waves 1-2):
├── Omega:  Task F-9 (gym-fx / NEAT Retrospective)
├── Dragon: Task F-4 (Asset + Timeframe Selection)       [depends on F-1, F-2, F-3]
└── Gamma:  Task F-5 (Data Pipeline Specification)       [depends on F-3, F-4 partial]

WAVE 4 (depends on all prior):
└── Omega:  Task F-10 (Evaluation Framework Specification)

WAVE 5 (final synthesis):
└── Omega:  Task F-11 (Part I Foundations Synthesis)
```

Notes on wave structure:
- Wave 1 runs fully in parallel (3 machines, 3 independent tasks)
- Wave 2 begins when each machine's Wave 1 task completes; machines don't need to wait for others
- Wave 3 requires deliverables from Wave 1 (F-4 needs F-1, F-2, F-3 outputs); machines coordinate via shared filesystem
- Waves 4 and 5 are synthesis on Omega only

---

## Shared Conventions (All Tasks)

### File locations

All deliverables go to a shared project directory accessible to all three machines:
```
trading_research/project2/part_I_foundations/
├── deliverables/           # All markdown outputs
├── data/                   # Any data gathered (catalogs, references, etc.)
├── scripts/                # Any analysis scripts (no strategy code)
└── references/             # Saved web pages, cited sources, notes
```

Each machine works in its assigned subdirectory but reads others' outputs when dependencies require.

### Document conventions

Every deliverable document starts with:
```
# [Task Title]
**Part I Foundations Task:** F-N
**Machine:** [Omega/Dragon/Gamma]
**Status:** [Draft / Complete]
**Produced:** [date]
**Feeds into:** [list of downstream Part I tasks, plus master plan Part II/III/IV/V/VI references]
```

Documents use section numbering. Internal references use `see §N.N`. External references (web, Jansen book, code repos) are cited with full paths/URLs.

### When to stop and escalate

If any task reveals:
- A fundamental flaw in Part I methodology
- A dependency on data/capability that does not exist
- A contradiction with Project 1 conclusions that deserves reopening

Stop that task, document the issue in a file named `ESCALATION_[task_id].md`, and wait for user response before continuing. Do not silently work around issues.

---

## WAVE 1

### TASK F-1 (Omega): Project 1 Knowledge Consolidation

**Purpose:** Distill Project 1 closure document (`PROJECT_CLOSURE.md` on GitHub at harveybc/heuristic-strategy) into concrete, categorized action items for Project 2.

**Inputs:**
- `PROJECT_CLOSURE.md`
- `PROJECT_RESEARCH_SUMMARY.md`
- `PHASE_6C_SYNTHESIS_FINAL_v2.md`
- `PHASE_6E01_STRATEGY_AUDIT.md`
- All prior phase synthesis documents (PHASE_3_5 through PHASE_6E01)

**Steps:**
1. Read all Project 1 synthesis documents end-to-end. Take structured notes.
2. Build classification of every substantive finding into one of five categories:
   - **CONSTRAINT** — hard rule that carries to Project 2 (e.g., "F1 ≥ 0.91 required for predictor-primary hourly FX" is a constraint)
   - **DESIGN GUIDANCE** — strong heuristic for Project 2 design (e.g., "plugin-canonical metrics are source of truth, not script-canonical")
   - **REOPENED** — conclusion that Project 2 should reconsider from scratch (e.g., "asset universe selection")
   - **INFRASTRUCTURE REUSE** — working code/tools available for Project 2 (e.g., cost model, walk-forward evaluator)
   - **PITFALL / BUG** — issues encountered that Project 2 must avoid (e.g., double vol-scaling bug, script-plugin divergence)
3. For each item, specify:
   - Short description
   - Original source (which phase document)
   - Category (from above)
   - Action item for Project 2 (specific, concrete)
   - Which Part of Project 2 this affects (I, II, III, IV, V, VI)
4. Produce cross-reference table: for each Project 2 Part, which Project 1 findings apply.

**Deliverable:** `deliverables/F1_PROJECT1_LESSONS_FOR_PROJECT2.md`

**Specific things to capture (minimum checklist):**
- F1 gap finding (0.44 vs 0.91 required) — constraint for Path B
- Double vol-scaling bug and lessons on equity curve methodology
- Script-canonical vs plugin-canonical divergence — production code is source of truth
- Multiple held-out windows needed (not single) — evaluation framework input
- Deflated Sharpe Ratio — multiple-testing penalty
- Parameter stability as first-class gate
- Oracle-free discipline — no oracle cells in Project 2
- USD/JPY concentration as structural risk — don't repeat
- Event-driven strategy killed in Project 1 static — REOPENED for adaptive
- Worst-2Y of −0.9 to −1.0 is industry-calibrated threshold
- Extended history worst-window filter as first-class gate
- Mean-reversion strategies are oracle-blind (statistical-property based)
- Portfolio diversification rescues individually non-deployable cells

**Success criteria:** User reading the deliverable can answer "what do I carry forward from Project 1?" in 15 minutes.

---

### TASK F-2 (Dragon): Jansen Book Integration Mapping

**Purpose:** Systematic review of "Machine Learning for Algorithmic Trading" 2nd edition by Stefan Jansen. Map each chapter's techniques to Project 2 components.

**Inputs:**
- Jansen 2nd edition book (user has access)
- Master plan document for Part mapping

**Steps:**
1. Produce structured inventory of Jansen 2nd edition's chapter structure (TOC-level).
2. For each chapter, produce:
   - One-paragraph summary of chapter topic
   - Key techniques/methods presented
   - Applicability assessment to each Project 2 Part:
     - Part I (Foundations): relevance?
     - Part II (Path A — Adaptive Heuristic): relevance?
     - Part III (Path B — Supervised ML): relevance?
     - Part IV (Path C — RL): relevance?
     - Part V (Synthetic Augmentation): relevance?
     - Part VI (NEAT + Final): relevance?
   - For each applicable Part: priority (REQUIRED / RECOMMENDED / OPTIONAL / REFERENCE)
   - Specific sections/pages to re-read carefully when the relevant Part begins
3. Produce reverse index: for each Project 2 Part, list Jansen chapters ordered by priority.
4. Flag any Jansen techniques that contradict Project 1 findings, noting both views.

**Deliverable:** `deliverables/F2_JANSEN_METHODOLOGY_INTEGRATION.md`

**Specific chapters to give deep coverage (known-high-relevance):**
- Data sources and providers chapter (Part I foundation)
- Feature engineering chapter (Part I, III)
- Alpha factor research chapter (Path A, B)
- Strategy evaluation chapter (Evaluation framework)
- Workflow and backtesting chapter (Part I pipeline)
- Linear models chapter (Path B baseline)
- Time series models chapter (Path B)
- Deep learning for trading chapters (Path B advanced)
- Conditional autoencoders / factor models (Path B feature reduction)
- Reinforcement learning chapters (Path C primary reference)
- Generative models / TimeGAN chapter (Part V)
- Bayesian ML chapter (Path B uncertainty, if covered)

**Implementation notes for Dragon agent:**
- If book PDF available locally, use it. If only user has physical/digital access, produce structured template that user fills in with chapter contents, then Dragon agent completes the analysis.
- Can supplement with web searches for chapter reviews/summaries where direct access is limited
- Do NOT reproduce large portions of book content (copyright). Paraphrase, summarize, reference page numbers.

**Success criteria:** User can open the document when starting Part II and know exactly which Jansen chapters to re-read for that Part.

---

### TASK F-3 (Gamma): Data Sources Catalog

**Purpose:** Research and catalog candidate data sources for Project 2. Analysis only, no acquisition.

**Steps:**

1. **Category 1: FX price data**
   - OANDA historical API (current Project 1 source)
   - Dukascopy (free tick data, professional quality)
   - HistData.com (free 1m FX data)
   - TrueFX (free institutional tick data)
   - TrueData / IQFeed (paid institutional)
   - Interactive Brokers (if retail access sufficient)
   - For each: cost, quality, temporal resolution, historical depth, API access, licensing

2. **Category 2: Economic calendar data**
   - ForexFactory (free, widely used)
   - Investing.com (free tier + paid)
   - Econoday (paid professional)
   - TradingEconomics (freemium API)
   - FRED (Federal Reserve Economic Data — free, already used in Project 1)
   - ECB Statistical Data Warehouse (free)
   - For each: events covered, historical depth, API access, field structure (actual/forecast/previous)
   - Special attention: event-driven strategies require event timestamps and surprise calculations

3. **Category 3: Alternative data per Jansen**
   - NASDAQ Data Link (formerly Quandl) — many datasets
   - Sentiment data (StockTwits, Twitter via X API, news sentiment providers)
   - Google Trends
   - Social media data providers
   - Satellite imagery (Orbital Insight, RS Metrics) — likely out of budget but catalog
   - Credit card / transaction data — institutional only, catalog only
   - Web scraping possibilities (legal ones)

4. **Category 4: Crypto data**
   - Binance API (free, extensive)
   - Coinbase Pro API (free)
   - Kaiko (institutional)
   - CoinGecko / CoinMarketCap (free)
   - For each: coverage, rate limits, historical depth, tick availability

5. **Category 5: Equity indices (if in scope)**
   - Yahoo Finance (free, retail quality)
   - Alpha Vantage (free tier + paid)
   - Polygon.io (freemium + paid)
   - IEX Cloud
   - Potentially relevant if Part I decides to include S&P 500, NASDAQ indices

6. **Category 6: Synthetic data approaches**
   - TimeGAN (Jansen-recommended, covered in Part V)
   - Other GAN variants for time series
   - Block bootstrap methods
   - Reference only here, active evaluation in Part V

**Use web_search and web_fetch tools to verify current state of each provider.** Things change (APIs deprecated, pricing changed, providers merged). Do not rely on outdated assumptions.

**Deliverable:** `deliverables/F3_DATA_SOURCES_CATALOG.md`

**Structure per source entry:**
```
### [Provider Name]
- **URL:** [link]
- **Data types:** [FX / crypto / economic / sentiment / etc.]
- **Cost tier:** [free / freemium / paid / enterprise]
- **Historical coverage:** [depth]
- **API availability:** [yes/no, REST/WebSocket/etc.]
- **Update frequency:** [realtime / daily / etc.]
- **Quality assessment:** [known issues, reputation]
- **Project 2 relevance:** [HIGH / MEDIUM / LOW / SPECULATIVE]
- **Notes:** [anything else relevant]
```

**Success criteria:** User has a comprehensive catalog to choose from in Foundations-4 (asset + timeframe selection) and later.

---

## WAVE 2

### TASK F-8 (Omega): Infrastructure Audit

**Purpose:** Document current state of every relevant repo for Project 2 readiness. Identify gaps and required work.

**Inputs:**
- All project repos accessible via filesystem
- Project 1 infrastructure notes from F-1

**Steps for each repo:**

1. **`feature-eng`**
   - Current plugins available (tech indicators, SSA, FFT, oracle labels, etc.)
   - Interface definition
   - Test coverage
   - Gaps for Project 2 (new features needed? causal-derived? regime-derived?)
   - Estimated readiness: READY / MINOR GAPS / MAJOR GAPS

2. **`preprocessor`**
   - Current transformations (normalization, unbiasing, trimming, feature selection)
   - Rolling window support?
   - Data leakage safeguards?
   - Readiness assessment

3. **`predictor`**
   - Current model plugins (ANN, CNN, LSTM, Transformer, TFT, N-BEATS, TCN)
   - Training/validation/test split handling
   - Rolling retraining support (THIS IS CRITICAL for Project 2)
   - Hot-swap mechanism preparation
   - Readiness assessment

4. **`prediction_provider`**
   - Current REST API
   - Model loading mechanism (the hot-swap target)
   - Deployment patterns
   - Gaps for adaptive use
   - Readiness assessment

5. **`heuristic-strategy`**
   - Current plugins (MR, TSMOM, Dual Momentum, regime plugins, others)
   - Parameter optimization infrastructure (what you've used with NEAT)
   - Rolling parameter retraining support
   - Readiness assessment

6. **`lts` (live trading system)**
   - Current orchestration layer state (from Project 1 Phase 6.E.0.1)
   - DefaultPipeline and DefaultPortfolio status
   - Adaptive-retraining specific infrastructure needed
   - Readiness assessment

7. **`causal-inference`**
   - Current techniques implemented (DML, Causal Forests, Meta-Learning, transfer entropy, Granger, etc.)
   - Integration points with other repos
   - Gaps identified in master plan Task F-6
   - Readiness assessment

8. **`synthetic-datagen` / `timeseries-gan`**
   - Current TimeGAN or other synthetic generators
   - Validation tooling
   - Readiness for Part V (not immediate priority)

9. **`gym-fx`**
   - Current state (master plan notes: mostly obsolete except parameter control)
   - Reusable components
   - To be superseded by modern RL in Part IV? Decision documented here
   - Readiness assessment

10. **`doin-core`, `doin-node`, `doin-plugins`**
    - Decentralized optimization infrastructure
    - Applicability to Project 2 (distributed retraining? decentralized inference?)
    - Readiness assessment

**Deliverable:** `deliverables/F8_INFRASTRUCTURE_AUDIT_PROJECT2.md`

**Cross-cutting concerns to identify:**
- Version compatibility across repos (are all on same Python version, same tensorflow/torch version?)
- Data format compatibility (do all repos speak the same format?)
- Logging and monitoring (is there unified logging across repos?)
- Configuration management (how are configs shared across repos?)

**Gap summary table:** at end of document, produce table listing every identified gap with:
- Gap description
- Affected repo
- Which Project 2 Part requires this
- Estimated work size (small / medium / large — no time units)
- Blocker status (BLOCKER / IMPORTANT / NICE-TO-HAVE)

**Success criteria:** User knows what infrastructure work must happen before Part II can begin.

---

### TASK F-6 (Dragon): Causal Inference Opportunities Mapping

**Purpose:** Systematic identification of where causal inference contributes across Project 2. Based on F-2 Jansen mapping (causal inference techniques covered) and causal-inference repo capabilities.

**Prerequisites:** F-2 Jansen mapping complete (provides book's causal inference coverage)

**Steps:**

1. **Inventory of techniques** (from causal-inference repo + literature):
   - Double Machine Learning (DML)
   - Causal Forests
   - Meta-Learning causal effects
   - Transfer Entropy (directional feature importance)
   - Granger Causality
   - Instrumental Variables
   - Regression Discontinuity
   - Synthetic Control
   - Difference-in-Differences
   - Do-calculus / structural causal models
   - Uplift modeling
   - PC algorithm / FCI for causal graph discovery
   - Any additional from Jansen book or user's personal library

2. **For each technique, document:**
   - Core concept (paraphrase, not reproduce)
   - What question it answers
   - Input requirements (data, assumptions)
   - Output interpretation
   - Computational cost
   - Relevance to Project 2 components

3. **Application mapping:**

   **Part I (Foundations):**
   - Feature selection: which features causally drive returns vs coincidentally correlate?
   - Data quality: do causal tests reveal data corruption (missing confounders, leakage)?

   **Part II (Path A — Adaptive Heuristic):**
   - Parameter stability: is parameter X's effect on performance causal or spurious?
   - Event-driven validation: regression discontinuity on NFP release moments

   **Part III (Path B — Supervised ML):**
   - Causal feature engineering: transfer entropy for lag selection
   - Model interpretation: causal forests on model predictions to find where model works
   - Distributional shift detection: causal relationships changing = regime shift signal

   **Part IV (Path C — RL):**
   - Counterfactual reward: what would have happened with different action?
   - Off-policy evaluation: causal inference for policy comparison without deployment
   - State feature design: include causal features, not just correlational

   **Part V (Synthetic Augmentation):**
   - Validate synthetic data preserves causal structure, not just correlational

   **Part VI (NEAT + Final):**
   - Causal analysis of which paths succeeded or failed and why

4. **Produce priority matrix:**

   | Technique | Part I | Part II | Part III | Part IV | Part V | Part VI |
   |-----------|--------|---------|----------|---------|--------|---------|
   | DML       | MED    | HIGH    | HIGH     | MED     | LOW    | HIGH    |
   | ... etc.  |        |         |          |         |        |         |

5. **Specific experiment proposals per Part:**
   - For each Part, list 2-3 concrete causal inference experiments with expected insight

**Deliverable:** `deliverables/F6_CAUSAL_INFERENCE_OPPORTUNITIES.md`

**Note:** User mentioned having additional causal inference books that may be consulted during Parts II-VI. This document should flag which techniques might benefit from deeper study with those books before that Part begins.

**Success criteria:** When Part II begins, user opens this document and immediately sees "for Path A, here are the causal inference techniques to integrate, and here is why."

---

### TASK F-7 (Gamma): Unsupervised Learning Opportunities Mapping

**Purpose:** Parallel to F-6 but for unsupervised learning techniques.

**Prerequisites:** F-2 Jansen mapping complete

**Steps:**

1. **Inventory of techniques** (from Jansen + literature + repo capabilities):
   - K-means clustering
   - Hierarchical clustering (agglomerative)
   - Gaussian Mixture Models (GMM)
   - Hidden Markov Models (HMM)
   - DBSCAN / HDBSCAN
   - Spectral clustering
   - Autoencoders (vanilla, variational, denoising)
   - t-SNE / UMAP for visualization
   - Self-organizing maps
   - Independent Component Analysis (ICA)
   - Principal Component Analysis (PCA) — classical baseline
   - Change-point detection algorithms (CUSUM, Bayesian, PELT)
   - Time-series clustering (DTW-based, shape-based)
   - Topic models (LDA, if sentiment data used)

2. **For each technique:**
   - Core concept (paraphrase)
   - Applicable data types
   - Parameter sensitivity
   - Computational cost
   - Interpretability

3. **Application mapping per Part:**

   **Part I (Foundations):**
   - Regime identification for labeling historical data
   - Outlier detection in training data
   - Feature space visualization for sanity checks

   **Part II (Path A):**
   - Regime-conditional parameter sets (different params per regime)
   - Regime-based strategy switching

   **Part III (Path B):**
   - Regime label as input feature to supervised model
   - Autoencoder for feature compression before modeling
   - Anomaly detection for low-confidence prediction flagging

   **Part IV (Path C):**
   - Regime state as RL state feature
   - Autoencoder for high-dimensional feature compression
   - Clustering of agent behavior across training episodes

   **Part V (Synthetic):**
   - Evaluate synthetic data regime distribution vs real data regime distribution
   - Cluster-based stratified sampling for augmentation

4. **Specific experiment proposals:**
   - Per Part, 2-3 concrete unsupervised experiments

**Deliverable:** `deliverables/F7_UNSUPERVISED_OPPORTUNITIES.md`

**Success criteria:** Same as F-6 — per-Part actionable list of unsupervised techniques with rationale.

---

## WAVE 3

### TASK F-9 (Omega): gym-fx / NEAT Retrospective

**Purpose:** Document historical experiments, identify reasons for underperformance, determine salvageability for Part VI.

**Inputs:**
- gym-fx repo access
- User's historical notes / memory of experiments (may need to query user)
- Any preserved experiment logs

**Steps:**

1. **Historical experiment inventory:**
   - What assets were used
   - What timeframes
   - What features / data
   - What NEAT configurations
   - What reward functions
   - What results (Sharpe, drawdown, trade frequency if logged)

2. **Failure mode analysis:**
   - User's hypothesis: wrong asset, wrong timeframe, wrong data, wrong features
   - Validate each hypothesis against available evidence
   - Identify additional possible failure modes:
     - NEAT-specific limitations (small population, slow convergence)
     - Reward function issues
     - State representation insufficiency
     - Episode definition problems
     - Action space mismatch

3. **Salvage assessment:**
   - Which components of gym-fx codebase are well-designed and reusable?
   - Which should be discarded / rewritten?
   - Parameter control mechanism (user mentioned this was the best-functioning use) — document this specifically

4. **Decision input for Part IV (RL):**
   - Recommend: modern RL framework (stable-baselines3, RLlib, CleanRL, etc.)
   - Recommend: gym environment architecture (reuse gym-fx or build new)
   - Specific design choices informed by what failed before

5. **Part VI NEAT comparison planning:**
   - What would NEAT need to produce good comparison results?
   - With Part I data / features, is NEAT likely to work now?
   - Scope of NEAT revisit in Part VI

**Deliverable:** `deliverables/F9_GYMFX_NEAT_RETROSPECTIVE.md`

**If user input is needed** (i.e., details of experiments that aren't in code/logs):
- Produce structured questionnaire as appendix
- User fills in blanks
- Agent updates document

**Success criteria:** Part IV design is informed by specific lessons from NEAT work. Part VI has clear plan for revisiting NEAT.

---

### TASK F-4 (Dragon): Asset + Timeframe Selection

**Purpose:** Deliberate, documented decision about which assets and timeframes Project 2 targets. Not inherited from Project 1.

**Prerequisites:** F-1 (Project 1 lessons), F-2 (Jansen mapping), F-3 (data sources)

**Steps:**

1. **Define selection criteria** (in priority order):
   - Trade frequency support (≥1/week per cell given user's stated minimum)
   - Data availability (cost, quality, historical depth per F-3 catalog)
   - Retail-viable cost structure (spreads, commissions)
   - Adaptive approach fit (higher frequency = more retraining data)
   - Causal analysis feasibility (assets with documented exogenous drivers)
   - Diversification benefit across selected assets (avoid concentration)
   - Project 1 evidence incorporation (without binding heritage)

2. **Candidate universe:**

   **FX majors:**
   - EUR/USD, USD/JPY (Project 1 finalists)
   - GBP/USD (insufficient history was issue in Project 1, may now be OK)
   - USD/CHF, AUD/USD, USD/CAD, NZD/USD

   **FX crosses:**
   - EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF
   - Different dynamics vs major pairs

   **Crypto:**
   - BTC/USD, ETH/USD (Project 1 killed on static; may be viable adaptive)
   - Note: Project 1 Phase 4 had specific findings on crypto worth reviewing

   **Equity indices (if in scope):**
   - S&P 500 futures/CFD, NASDAQ futures/CFD
   - Dow, Russell 2000, sector ETFs

3. **Timeframe candidates (per user guidance):**
   - 5m (high frequency, high cost sensitivity)
   - 15m (moderate)
   - 1h (Project 1 EUR/USD MR killed here — reconsider under adaptive)
   - 4h (moderate)
   - Daily (optional benchmark)

4. **Per (asset × timeframe) combination:**
   - Score against each criterion
   - Red flags (liquidity, cost, data availability)
   - Green flags (evidence of viability, good data, low cost)
   - Preliminary ranking

5. **Selection decision:**
   - Recommended primary set (2-4 assets, 2-3 timeframes each = 4-12 cells)
   - Rationale for each inclusion
   - Explicit exclusions with rationale (e.g., "USD/JPY intentionally de-emphasized to avoid Project 1 concentration")
   - Secondary/tertiary candidates for expansion if primary set shows promising early results

6. **Event-driven revisit specific note:**
   - Project 1 event-driven killed on static backtest (Phase 4 Track C)
   - Adaptive reconsideration: which events + which timeframes might work now
   - Explicit choice: include event-driven candidate cell, or defer to Part II decision

**Deliverable:** `deliverables/F4_ASSET_TIMEFRAME_SELECTION.md`

**Success criteria:** Clear list of (asset, timeframe) cells with documented rationale. This feeds directly into F-5 Data Pipeline Specification.

---

### TASK F-5 (Gamma): Data Pipeline Specification

**Purpose:** Design the production-grade data pipeline feeding all three experimental paths.

**Prerequisites:** F-3 (data sources), F-4 (asset/timeframe selection)

**Steps:**

1. **Ingestion layer:**
   - Per selected (asset, timeframe) from F-4
   - Source(s) from F-3 catalog
   - Initial bulk download
   - Incremental update mechanism
   - Gap handling (weekends, holidays, exchange closures)
   - Data validation on ingestion (OHLC consistency, timestamp monotonicity, volume sanity)
   - Storage format (CSV vs Parquet vs database)

2. **Preprocessing layer:**
   - Bar validation rules
   - Missing data policy (forward-fill exogenous, interpolation rules)
   - Outlier detection (from F-7 unsupervised)
   - Normalization strategy (per-symbol, per-window, rolling)
   - Stationarity handling (log returns vs price levels, differencing)
   - Timezone normalization (everything in UTC)

3. **Feature engineering layer:**
   - Reuse feature-eng plugins (from F-8 audit): list specific plugins
   - Technical features (moving averages, oscillators, volatility measures, volume indicators)
   - Statistical features (realized vol, skewness, kurtosis, autocorrelation)
   - Transformation features (FFT components, SSA components, wavelets)
   - Causal-derived features (from F-6): what specific causal features
   - Regime features (from F-7): what specific regime features
   - Event features (from F-3 economic calendar): surprise, proximity to event, post-event behavior
   - Cross-asset features (other FX pairs' momentum, correlation state, etc.)

4. **Splits layer:**
   - Default: 4 years train, 1 year val, 1 year test (per user's standard)
   - Rolling schedule: parameterizable (weekly, monthly, criteria-based)
   - Split integrity: no future data in past splits
   - Per-cell splits (each asset × timeframe has its own split state)

5. **Storage architecture:**

   ```
   data/
   ├── raw/                  # Direct from source, immutable
   │   └── [asset]/[timeframe]/[year].parquet
   ├── preprocessed/         # After validation and normalization
   │   └── [asset]/[timeframe]/[year].parquet
   ├── features/             # After feature engineering
   │   └── [asset]/[timeframe]/[feature_set_version]/[year].parquet
   ├── splits/               # Train/val/test boundaries
   │   └── [split_config]/boundaries.json
   └── metadata/             # Provenance, versions, checksums
       └── ...
   ```

6. **Versioning:**
   - Feature set versions (so retraining experiments can reproduce results)
   - Preprocessing version
   - Data source version / snapshot date
   - Per Jansen emphasis on reproducibility

7. **Augmentation hooks (for Part V):**
   - Interface for synthetic data insertion
   - No actual augmentation in Part I; just design the pluggable point

**Deliverable:** `deliverables/F5_DATA_PIPELINE_SPEC.md`

**Structure:**
- Section 1: Architecture diagram (text-based)
- Section 2: Layer-by-layer specification
- Section 3: Interface contracts between layers
- Section 4: Quality assurance checks
- Section 5: Implementation checklist (what must be built before Part II)

**Success criteria:** Another agent could implement this pipeline from the specification without needing additional design decisions.

---

## WAVE 4

### TASK F-10 (Omega): Evaluation Framework Specification

**Purpose:** Formalize comparative evaluation framework for Parts II, III, IV.

**Prerequisites:** All prior Wave 1-3 tasks (this integrates them)

**Steps:**

1. **Metrics stack formalization:**
   - Primary: average test-Sharpe across rolling test windows
   - Deflated Sharpe Ratio: formula, assumptions, how to compute for adaptive setting
   - Per-window: Sharpe, max DD, return, volatility, trade count, hit rate
   - Aggregate: mean, median, percentiles, fraction positive
   - Adaptive-specific: retraining frequency, parameter stability CV, retraining effectiveness ratio
   - Cross-path: statistical tests for path comparison (paired tests given shared test periods)

2. **Rolling window specification:**
   - Train/val/test ratios (default 4/1/1 years, but parameterizable)
   - Roll-forward periods to test (daily, weekly, monthly, criteria-based)
   - Minimum number of test windows for meaningful evaluation (per path)
   - Handling of early/late windows where history is limited

3. **Gate specifications:**
   - Per-Path gates (already in master plan section 2.3)
   - Per-cell gates (if applicable for sub-cell evaluation)
   - Cross-Path comparison gates

4. **Baseline specifications:**
   - Each path has a "fixed-parameter" baseline for adaptive-vs-static comparison
   - Baseline: same strategy, same data, same splits, no retraining
   - Comparison: adaptive version wins if exceeds baseline by gate-specified margin

5. **Reporting templates:**
   - Per-Path results table template
   - Cross-Path comparison template
   - Per-cell detail template
   - Visualization recommendations (equity curves, drawdown profiles, parameter evolution plots)

6. **Statistical significance:**
   - Paired t-tests for comparing adaptive vs static
   - Bootstrap confidence intervals for Sharpe differences
   - Handling of dependent test windows (overlapping data in some rolling schemes)

**Deliverable:** `deliverables/F10_EVALUATION_FRAMEWORK.md`

**Implementation note:** Some evaluation infrastructure may already exist in `lts/evaluation_harness.py` (from Project 1). F-10 specifies; implementation gaps identified here feed into F-8 infrastructure audit conclusions.

**Success criteria:** Every experiment in Parts II-IV can be evaluated and compared using exactly this framework.

---

## WAVE 5

### TASK F-11 (Omega): Part I Foundations Synthesis

**Purpose:** Integrate all Wave 1-4 deliverables into single coherent starting point for Parts II-VI.

**Prerequisites:** F-1 through F-10 complete

**Steps:**

1. **Read all Part I deliverables.**

2. **Produce integrated synthesis document with sections:**

   **Executive Summary:**
   - Project 2 Part I completion status
   - Selected assets and timeframes (from F-4)
   - Data pipeline architecture (from F-5)
   - Evaluation framework (from F-10)
   - Infrastructure readiness (from F-8)
   - Key Project 1 lessons applied (from F-1)
   - Jansen integration active (from F-2)
   - Causal inference plan (from F-6)
   - Unsupervised learning plan (from F-7)
   - Part II ready to begin: yes/no + conditions

   **Part II Readiness Checklist:**
   - Data ingested for selected (asset, timeframe) cells
   - Features computed for selected cells
   - Splits defined
   - Evaluation framework functional
   - Causal inference integration point specified
   - Unsupervised integration point specified
   - Baseline specifications ready
   - Gates pre-registered

   **Part III Readiness Preview:**
   - Same as Part II but noting additional prerequisites for Path B (predictor setup, model selection framework)

   **Part IV Readiness Preview:**
   - Same for Path C (RL framework selection from F-9, env design)

   **Parts V-VI Readiness Preview:**
   - Lighter; these are downstream but note any Part I outputs they'll use

3. **Identify cross-cutting risks:**
   - Any Part I finding that may cause Part II/III/IV/V/VI design to change
   - Any infrastructure gap that is blocker

4. **Part II kick-off document:** At the end of synthesis, draft the skeleton of the Part II work plan based on what Foundations revealed. This isn't the full Part II plan (that's written after Part I approval) but a transition bridge showing logical flow.

**Deliverable:** `deliverables/F11_PART_I_FOUNDATIONS_SYNTHESIS.md`

**This is the Gate document for Part II.** User reviews this, either approves (→ Part II execution plan gets written) or requests revision of Part I deliverables before proceeding.

**Success criteria:** User can open this one document and answer "where does Project 2 stand, and what's next?" in 10 minutes.

---

## Cross-Wave Coordination

### Communication between machines

Machines do not need to communicate in real-time. The file system is the coordination mechanism:
- Each task has input dependencies clearly specified
- When a task's dependencies are present in `deliverables/`, the task can proceed
- Agents should check for prerequisite deliverables before starting a Wave 2/3 task

### If an agent finishes faster than expected

An agent that completes its Wave task ahead of others can:
- Quality-review its own output (tighten prose, check citations, validate claims)
- Check whether subsequent Wave tasks are eligible to start on its machine (if dependencies met)
- Begin the next eligible task without waiting for other machines to finish Wave

### If an agent gets stuck

If a task is blocked by:
- Missing data
- Missing repo access
- Ambiguous specification
- Tool/environment issue

Produce `ESCALATION_F[task_number].md` describing the blocker and stop. Do not attempt workarounds.

### Progress tracking

Each machine maintains a log file: `logs/[machine]_progress.log` with entries:
```
[timestamp] Started Task F-N
[timestamp] Task F-N: [specific sub-step]
[timestamp] Completed Task F-N, deliverable at deliverables/FN_[title].md
```

This enables user to check progress across machines at any time.

---

## Final Part I Deliverable Summary

When Part I is complete, deliverables directory contains:

```
deliverables/
├── F1_PROJECT1_LESSONS_FOR_PROJECT2.md
├── F2_JANSEN_METHODOLOGY_INTEGRATION.md
├── F3_DATA_SOURCES_CATALOG.md
├── F4_ASSET_TIMEFRAME_SELECTION.md
├── F5_DATA_PIPELINE_SPEC.md
├── F6_CAUSAL_INFERENCE_OPPORTUNITIES.md
├── F7_UNSUPERVISED_OPPORTUNITIES.md
├── F8_INFRASTRUCTURE_AUDIT_PROJECT2.md
├── F9_GYMFX_NEAT_RETROSPECTIVE.md
├── F10_EVALUATION_FRAMEWORK.md
└── F11_PART_I_FOUNDATIONS_SYNTHESIS.md
```

Plus any `ESCALATION_*.md` files if blockers arose.

User reviews F-11 synthesis as Part I gate. Approval triggers Part II work plan writing.

---

## Start Instructions

**Right now, the three machines can begin:**

- **Omega:** Begin Task F-1 (Project 1 Knowledge Consolidation)
- **Dragon:** Begin Task F-2 (Jansen Integration Mapping)
- **Gamma:** Begin Task F-3 (Data Sources Catalog)

Each machine reads its task section above, executes per the specification, produces its deliverable, and then checks whether Wave 2 tasks can begin on that machine based on dependency state.

When all Part I tasks complete and F-11 synthesis is ready, notify user for Part I gate review.

Execution begins at user's signal.