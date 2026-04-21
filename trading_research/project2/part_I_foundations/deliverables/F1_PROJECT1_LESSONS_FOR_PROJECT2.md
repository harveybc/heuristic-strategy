# Project 1 Knowledge Consolidation for Project 2
**Part I Foundations Task:** F-1  
**Machine:** Omega  
**Status:** Complete  
**Produced:** 2026-04-18  
**Feeds into:** F-4 (asset selection), F-5 (pipeline spec), F-8 (infra audit), F-10 (eval framework), F-11 (synthesis); Parts II–VI

---

## 1. Classification of Project 1 Findings

### 1.1 CONSTRAINTS — Hard rules that carry to Project 2

| ID | Finding | Source | Action for Project 2 | Affects |
|----|---------|--------|----------------------|---------|
| C-1 | **F1 ≥ 0.91 required for predictor-primary hourly FX at retail costs.** Oracle noise sweep showed strategies need prediction F1 > 0.91 to overcome retail spread (1.5 bps) + slippage. Best ML predictor achieved F1 ≈ 0.45. | Phase 6.A | Path B (supervised ML) must either close this gap or use prediction as a signal modifier, not primary driver. Consider: lower-cost assets, higher timeframes, or ensemble approaches to push F1 up. | Part III (Path B) |
| C-2 | **Plugin-canonical metrics are source of truth, not script-canonical.** Script-level pandas vectorized computations diverge from bar-by-bar plugin execution at held-out level (held-out SR +0.316 script vs −0.065 plugin). | Phase 6.E.0.1 | ALL Project 2 evaluation must run through production code path from the start. No script-only validation accepted as canonical. | Parts II–VI |
| C-3 | **Pre-registered kill criteria must be honored without post-hoc relaxation.** Terminal 3 was triggered because the project honored its held-out SR > 0 criterion. Post-hoc arguments ("−0.065 is close to zero") were explicitly rejected. | Phase 6.E.0.1, §6.3 | Define kill criteria for each Project 2 path before experiments run. Do not weaken them when results disappoint. | Parts II–VI |
| C-4 | **Worst-2Y Sharpe > −0.9 is industry-calibrated threshold.** Benchmarked against 6 managed-futures ETFs; 5 of 6 fail −0.5. The −0.9 threshold is calibrated to live professional products. | Phase 5 | Use −0.9 as worst-2Y gate for Project 2 portfolio evaluation. | Part II–VI eval |
| C-5 | **Retail FX spread costs are 2–10× institutional.** At 1.5 bps spread + slippage, many strategies that work at institutional cost break even or lose at retail. Cost breakeven ≥ 2× was a hard gate. | Phase 6.C.3 | Transaction cost model carries forward. Require cost breakeven ≥ 2× for any deployed strategy. Consider lower-cost venues or assets. | Parts II–VI |
| C-6 | **Parameter perturbation plateau ≥ 60% is a genuine robustness signal.** MR cell failed at 57% and this correctly predicted held-out fragility. | Phase 6.C.5 | Treat parameter perturbation as hard gate, not informational. ≥ 60% plateau required. | Part II (Path A), F-10 |

### 1.2 DESIGN GUIDANCE — Strong heuristics for Project 2 design

| ID | Finding | Source | Action for Project 2 | Affects |
|----|---------|--------|----------------------|---------|
| D-1 | **Extended-history worst-window filter as first-class gate, not afterthought.** Phase 3.5 was framed as "audit" — should have been a Phase 1.5 baseline gate. | Phase 3.5, Closure §4.3 | Apply worst-2Y filter from the earliest evaluation stage. No candidate advances without passing it. | F-10 (eval framework) |
| D-2 | **Multiple held-out windows, not single.** One 2024–2025 sample triggered the kill. Rolling held-out (train 2003–2018/test 2019–2020, train 2003–2019/test 2020–2021, etc.) gives distribution of outcomes. | Closure §4.3 | Design rolling held-out into evaluation framework. Report mean and worst held-out SR across windows. | F-10 (eval framework) |
| D-3 | **Implement production pipeline first, validate against backtest.** Project 1 built scripts first, plugins later, discovered divergence at the very end. | Closure §4.3 | For each Project 2 path: implement strategy as plugin first, validate plugin against any research script, not the other way around. | Parts II–IV |
| D-4 | **Portfolio diversification rescues individually non-deployable cells.** Cells with SR 0.18–0.41 individually combined into portfolio SR 0.45 due to low cross-correlation (ρ < 0.10). | Phase 5 | Design multi-cell portfolios from the start. Evaluate both individual cell viability and portfolio-level metrics. | Parts II–VI |
| D-5 | **Avoid single-asset concentration.** P3 had 79.4% in USD/JPY — effectively a single-asset bet. | Phase 5.5 | Enforce diversification constraint: no single asset > 50% portfolio weight. Select 3+ assets from F-4. | F-4 (asset selection) |
| D-6 | **Warm-up period handling materially changes max DD.** Including first 252 bars (when rolling estimates are unstable) adds 2–5pp to reported max DD. | Phase 6.D.2 | Document warm-up exclusion policy explicitly. Standardize: exclude first max(lookback) bars from all metric computations. | F-10 (eval framework) |
| D-7 | **Oracle-free discipline.** Every oracle-dependent cell failed extended-history stress. Oracle is useful for diagnostic search but not for strategy construction. | Phase 3.5, 5.5 | Project 2 uses no oracle signals in strategy construction. Oracle testing can be used for diagnostic purposes only (e.g., "does this feature space have theoretical edge?"). | Parts II–IV |
| D-8 | **Mean-reversion strategies are oracle-blind (statistical-property based).** EUR/USD and USD/JPY daily MR showed < 0.5% sign-flip across oracle noise spectrum. | Phase 1.3, 3.5 | MR strategies are valid candidates for Path A (adaptive heuristic) because they depend on statistical properties, not prediction. Periodic re-optimization of MR parameters is a natural fit. | Part II (Path A) |
| D-9 | **Deflated Sharpe Ratio needed for multiple-testing penalty.** Project 1 tested hundreds of cells without formal multiple-testing correction. | Closure §3.2 | Incorporate DSR (Harvey & Liu, 2015) into evaluation framework. Report both raw and deflated Sharpe for all candidates. | F-10 (eval framework) |
| D-10 | **Static parameters over 23 years is unrealistic.** All P1 strategies used fixed parameters from 2003–2026. Production strategies would re-optimize periodically (weekly/monthly). This was never tested. | User observation, post-closure | This is the core hypothesis of Project 2. Adaptive re-optimization is the primary experimental variable. | Parts II–IV (core) |

### 1.3 REOPENED — Conclusions Project 2 should reconsider from scratch

| ID | Finding | Source | Why Reopened | Affects |
|----|---------|--------|-------------|---------|
| R-1 | **Asset universe selection.** P1 started with 12 assets, ended on 2. Crypto (BTC, ETH) killed on static basis may be viable adaptive. Equities never explored. | Closure §9, Phase 4 | Adaptive strategies may perform differently. Crypto volatility may be advantageous for re-optimized parameters. | F-4 (asset selection) |
| R-2 | **Timeframe selection.** P1 used daily primarily. Hourly MR killed on static. 4h never tested properly (data too short). 5m/15m never considered. | Phase 2, 4 Track D | Higher-frequency data = more retraining opportunities for adaptive approaches. Hourly MR might work with periodic re-fitting. | F-4 (asset selection) |
| R-3 | **Event-driven strategies.** Phase 4 Track C tested academic event-driven on static basis — all killed. Adaptive event-driven (recalibrate around each event) was never tested. | Phase 4 Track C | Regression discontinuity around NFP/ECB events might work if strategy adapts to recent event patterns. | Part II (Path A) |
| R-4 | **Regime filter overlays.** Phase 6.B showed regime filters (WFO V2, GMM V3) worsened P3 on static basis. Adaptive regime filters that re-fit periodically may behave differently. | Phase 6.B | Regime detection with periodic recalibration is fundamentally different from fixed-threshold regime classification. | Part II (Path A) |
| R-5 | **ML predictor architectures.** P1 used CNN, ANN, ensemble, NEAT — all negative over 14 years on static training. With rolling retraining, these might overcome distribution shift. | Phase 6.A | This is Path B's core hypothesis. Rolling retraining may close or narrow the F1 gap. | Part III (Path B) |

### 1.4 INFRASTRUCTURE REUSE — Working code/tools available for Project 2

| ID | Component | Location | State | Reuse Notes |
|----|-----------|----------|-------|-------------|
| I-1 | **Transaction cost model** (FX, 16 assets) | `trading_research/transaction_cost_model.py` | Functional | Carry forward directly. May need extension for crypto/equity costs. |
| I-2 | **Evaluation harness** (Sharpe, rolling window, worst-2Y) | `trading_research/evaluation_harness.py` | Functional | Extend for rolling held-out windows (D-2), DSR (D-9). |
| I-3 | **Walk-forward evaluator** (52-quarter expanding window) | `phase6c_stress.py`, `phase6e01_plugin_stress.py` | Functional | Core reuse for Project 2. Adapt for monthly/weekly re-optimization windows. |
| I-4 | **Parameter perturbation harness** (grid, plateau %) | `phase6e01_plugin_stress.py` (Task 6.C.5) | Functional | Reuse directly. Now a hard gate (C-6). |
| I-5 | **Monte Carlo regime simulator** (3000 paths × 3 regimes) | `phase6c_stress.py` (Task 6.C.2) | Functional | Reuse for stress testing adaptive strategies. |
| I-6 | **Plugin-vs-script comparison framework** | `phase6e01_plugin_canonical.py` | Functional | Critical reuse — validates production code matches research code. |
| I-7 | **Feature engineering plugins** (tech indicators, SSA, FFT) | `feature-eng/` repo | Functional | Extend with causal-derived and regime features for P2. |
| I-8 | **Preprocessing pipeline** (normalizer, unbiaser, trimmer) | `preprocessor/` repo | Functional | Needs rolling-window support audit (F-8). |
| I-9 | **Predictor models** (ANN, CNN, LSTM, Transformer, TFT, N-BEATS, TCN) | `predictor/` repo | Functional | Key for Path B. Needs rolling retraining support (F-8). |
| I-10 | **Prediction Provider API** | `prediction_provider/` repo | Functional | Hot-swap target for adaptive model updates. |
| I-11 | **Strategy plugins** (MR, TSMOM, DM, regime WFO, regime GMM) | `heuristic-strategy/app/plugins/` | Functional | Path A candidates. Need parameter re-optimization hooks. |
| I-12 | **LTS orchestration** (DefaultPipeline, DefaultPortfolio) | `lts/` repo | Partial | Needs completion for adaptive retraining loop. |
| I-13 | **Extended data store** (12 assets, 20+ years, COT, macro) | `trading_research/feature_store/`, `extended_data/` | Functional | Expand with additional sources from F-3. |
| I-14 | **Causal inference tools** (DML, Causal Forests, Transfer Entropy) | `causal-inference/` repo | Experimental | Integration into main pipeline needed. |

### 1.5 PITFALLS / BUGS — Issues Project 2 must avoid

| ID | Issue | Source | Prevention in Project 2 |
|----|-------|--------|------------------------|
| P-1 | **Double vol-scaling bug.** Cells individually scaled to 10% vol, then portfolio re-scaled to 10%. Compounds drawdowns, inflated max DD from 18.9% to 24.6%. | Phase 6.D | Standardize: scale cells individually, compute portfolio at realized vol, then single portfolio-level scaling to target. Never double-scale. |
| P-2 | **Script-plugin divergence at held-out.** Full-period agreement ~90% but held-out diverged materially (DM 2024 at 58% agreement). | Phase 6.E.0.1 | Validate plugin against script per-period, not just full-period aggregate. Flag any period with < 80% agreement for investigation. |
| P-3 | **Oracle contamination in portfolio construction.** Phase 5 included oracle-dependent cells in "oracle-free" portfolio. Caught by Phase 5.5 audit. | Phase 5.5 | Maintain strict oracle/non-oracle separation. No oracle signals in any production strategy component. |
| P-4 | **Premature Terminal 1 declaration.** Declared deployment based on script-canonical metrics before plugin validation. | Phase 6.C → 6.E.0.1 | No terminal decision until plugin-canonical metrics are produced and pass all gates. |
| P-5 | **Soft interpretation of hard gates.** Parameter perturbation FAIL was characterized as "informational." It was a real signal. | Phase 6.C.5 | All gates are hard. A FAIL is a FAIL. Document reasoning if proceeding despite FAIL, but call it what it is. |
| P-6 | **Late pipeline integration.** E2E orchestration validated only in final phase. | Phase 6.E.0.1 | Integrate and validate production pipeline per-path from the first experiment, not as final gate. |
| P-7 | **Single held-out period fragility.** One 2-year window determined entire project outcome. | Phase 6.E.0.1 | Use rolling held-out windows. Never base terminal decision on a single held-out period. |

---

## 2. Cross-Reference: Project 1 Findings by Project 2 Part

### Part I (Foundations)
| Finding | Category | Relevance |
|---------|----------|-----------|
| D-1 Worst-2Y as first-class gate | Design | Bake into evaluation framework from start |
| D-2 Multiple held-out windows | Design | Design rolling held-out into F-10 |
| D-5 Avoid single-asset concentration | Design | Input to F-4 asset selection |
| D-6 Warm-up period standardization | Design | Specify in F-10 |
| D-9 Deflated Sharpe Ratio | Design | Include in F-10 |
| I-1 through I-14 | Infrastructure | Audit in F-8 |

### Part II (Path A — Adaptive Heuristic)
| Finding | Category | Relevance |
|---------|----------|-----------|
| D-10 Static parameters never re-optimized | Design | Core hypothesis — adaptive re-optimization |
| D-8 MR is oracle-blind | Design | MR is natural adaptive candidate |
| R-3 Event-driven reopened | Reopened | Adaptive event strategies viable |
| R-4 Regime filters reopened | Reopened | Adaptive regime detection viable |
| C-6 Parameter plateau ≥ 60% | Constraint | Hard gate for parameter stability |
| I-11 Strategy plugins | Infrastructure | Starting point for adaptive variants |

### Part III (Path B — Supervised ML)
| Finding | Category | Relevance |
|---------|----------|-----------|
| C-1 F1 ≥ 0.91 for predictor-primary | Constraint | Must close gap or change approach |
| R-5 ML architectures reopened | Reopened | Rolling retraining may help |
| D-3 Plugin-first implementation | Design | Train models as plugins from day 1 |
| I-9 Predictor models | Infrastructure | Starting models for Path B |
| I-10 Prediction Provider API | Infrastructure | Hot-swap infrastructure |

### Part IV (Path C — RL)
| Finding | Category | Relevance |
|---------|----------|-----------|
| C-2 Plugin-canonical metrics | Constraint | RL agent must be evaluated through production path |
| D-4 Portfolio diversification | Design | RL agent may manage portfolio, not single asset |
| D-10 Adaptive re-optimization | Design | RL is inherently adaptive — compare with Path A/B |

### Part V (Synthetic Augmentation)
| Finding | Category | Relevance |
|---------|----------|-----------|
| I-5 MC regime simulator | Infrastructure | Baseline synthetic data approach |
| D-7 Oracle-free discipline | Design | Synthetic data must not leak future information |

### Part VI (NEAT + Final)
| Finding | Category | Relevance |
|---------|----------|-----------|
| C-3 Pre-registered kill criteria | Constraint | Define NEAT comparison criteria before running |
| D-2 Multiple held-out windows | Design | NEAT evaluated on same windows as other paths |

---

## 3. Key Metrics Reference (Project 1 Final)

For quick reference when Project 2 needs to compare against Project 1 baseline:

| Metric | P1 Value (Plugin-Canonical) | Context |
|--------|---------------------------|---------|
| Best full-period Sharpe | 0.41 | P3 portfolio, static params, 2003–2026 |
| Best held-out Sharpe | −0.065 | 2024–2025, plugin-canonical |
| Best worst-2Y | −0.744 | P3 portfolio |
| Cost breakeven | >3× | Robust |
| Walk-forward positive | 61.5% | 32/52 quarters |
| Best individual cell SR | 0.41 (USD/JPY DM) | Static params |
| Portfolio maxDD | 20.18% | At 10% target vol |
| Strategy families tested | MR, TSMOM, DM, momentum, carry, VRS, regime | All static |
| Assets in final portfolio | EUR/USD, USD/JPY | 2 of 12 original |
| Total cells evaluated | 360+ | 12 assets × 4 TF × 5+ strategies |

**Project 2 adaptive strategies should aim to exceed P1 static baseline.** If adaptive strategies cannot beat SR = 0.41 full-period and SR > 0 held-out (the criteria P1 failed), the adaptive hypothesis is not supported.

---

## 4. Open Questions for Project 2

1. **How much does periodic re-optimization improve worst-2Y?** The core untested hypothesis.
2. **What re-optimization frequency is optimal?** Weekly, monthly, quarterly — depends on data frequency and strategy type.
3. **Does rolling retraining close the F1 gap?** Path B's central question (0.45 → 0.91 needed).
4. **Do adaptive regime filters add value?** Fixed regime filters failed in P1 (R-4). Adaptive ones are untested.
5. **Is the held-out failure from parameter staleness or structural weakness?** If staleness, adaptive fixes it. If structural, it doesn't.
