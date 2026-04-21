# F-10: Evaluation Framework Specification

**Date**: 2025-06-17  
**Scope**: Define metrics, statistical tests, kill criteria, and comparison methodology for all Project 2 experiments  
**Depends on**: F-1 (P1 lessons — constraints C-1 through C-6), F-4 (asset/timeframe), F-5 (pipeline spec), F-8 (infrastructure audit)

---

## 1. Design Principles

1. **Pre-registered**: All metrics, thresholds, and kill criteria defined before experiments begin. No post-hoc metric shopping.
2. **Multi-level**: Evaluate at per-window level (rolling performance), aggregate level (full OOS period), and held-out level (2020-2024 stress test).
3. **Cost-realistic**: All performance metrics computed after realistic transaction costs.
4. **Comparative**: Every adaptive strategy compared against its static baseline (P1 results) and a naive benchmark.
5. **Honest**: Report both successes and failures. A well-documented negative result is more valuable than a poorly documented positive one.

---

## 2. Metrics Hierarchy

### 2.1 Primary Metrics (Decision-Making)

| Metric | Formula | Level | Kill Threshold |
|--------|---------|-------|---------------|
| **Held-out Sharpe Ratio** | $SR = \frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{252}$ (annualized) | Held-out (2020-2024) | $SR_{held} > 0$ (must be positive) |
| **Worst 2-Year Rolling Sharpe** | $\min_{w} SR_{2Y}(w)$ over all 2-year windows in OOS | Aggregate OOS | $> -0.9$ (from C-4) |
| **Cost Ratio** | $\frac{\text{Gross PnL}}{\text{Total Costs}}$ | Aggregate OOS | $\geq 2.0$ (from C-5) |
| **Adaptive vs Static Delta** | $\Delta SR = SR_{adaptive} - SR_{static}$ | Held-out | $> 0$ (the core hypothesis) |

### 2.2 Secondary Metrics (Diagnostic)

| Metric | Formula | Level | Purpose |
|--------|---------|-------|---------|
| **Maximum Drawdown** | $MDD = \max_{t} \left(\frac{\text{peak}_t - \text{trough}_t}{\text{peak}_t}\right)$ | Per-window + aggregate | Risk assessment |
| **Calmar Ratio** | $\frac{\text{Annualized Return}}{|MDD|}$ | Aggregate OOS | Return-per-risk efficiency |
| **Win Rate** | $\frac{\text{Winning trades}}{\text{Total trades}}$ | Per-window | Stability diagnostic |
| **Profit Factor** | $\frac{\sum \text{winning PnL}}{|\sum \text{losing PnL}|}$ | Per-window + aggregate | Asymmetry of returns |
| **Number of Trades** | Count | Per-window | Activity level / cost exposure |
| **Time in Market** | $\frac{\text{Bars with position}}{\text{Total bars}}$ | Per-window | Capital efficiency |

### 2.3 ML-Specific Metrics (Path B only)

| Metric | Formula | Level | Purpose |
|--------|---------|-------|---------|
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Per-window (train/val/test) | Prediction accuracy |
| **Directional Accuracy** | $\frac{1}{n}\sum \mathbb{1}[\text{sign}(y_i) = \text{sign}(\hat{y}_i)]$ | Per-window | Signal quality |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Per-window (binary models) | Kill criterion: $F1 \geq 0.91$ (from C-1) |
| **Information Coefficient** | $\text{corr}(y, \hat{y})$ | Per-window | Predictive power |
| **Train-Test MAE Gap** | $MAE_{test} - MAE_{train}$ | Per-window | Overfitting diagnostic |

### 2.4 Adaptive-Specific Metrics (All Paths)

| Metric | Formula | Level | Purpose |
|--------|---------|-------|---------|
| **Parameter Stability** | $\text{std}(\theta_w) / \text{mean}(\theta_w)$ across windows $w$ | Cross-window | How much params change per retrain |
| **Regime Transition Rate** | $\frac{\text{Regime changes}}{\text{Total bars}}$ | Per-window | GMM regime switching frequency |
| **Retrain Improvement** | $\frac{SR_{w} - SR_{w-1}}{SR_{w-1}}$ | Sequential windows | Does retraining actually help? |
| **Window OOS Consistency** | $\frac{\text{Windows with } SR_{OOS} > 0}{\text{Total windows}}$ | Cross-window | From C-6: must be $\geq 60\%$ |
| **Retraining Compute Cost** | Wall-clock time per retrain cycle | Per-window | Practical viability |

---

## 3. Benchmarks

Every experiment must be compared against these baselines:

### 3.1 Naive Benchmarks

| Benchmark | Description | Purpose |
|-----------|-------------|---------|
| **Buy-and-Hold** | Long EUR/USD for entire period | Market exposure baseline |
| **Random Entry** | Random long/short with same trade frequency as strategy | Skill vs luck |
| **Zero (flat)** | No trades | Cost baseline (any strategy must beat zero after costs) |

### 3.2 Project 1 Baselines

| Benchmark | Source | Held-out SR |
|-----------|--------|------------|
| **Plugin-canonical static** | P1 Phase 7 | $-0.065$ (held-out) |
| **Best static WFO** | P1 Phase 6 | 61.5% positive windows |
| **P1 full-period static** | P1 | $SR = 0.41$ (full), $MDD = 20.18\%$ |

### 3.3 Cross-Path Comparison

Each Project 2 path is a benchmark for the others:

| Comparison | Question Answered |
|------------|------------------|
| Path A vs P1 static | Does adaptive heuristic optimization beat static? |
| Path B vs Path A | Does ML rolling retraining beat adaptive heuristics? |
| Path C vs Path B | Does RL beat supervised ML? |
| Any path vs Buy-and-Hold | Does any adaptive strategy beat passive? |
| NEAT-HPO vs DEAP-GA (Part VI) | Does NEAT hyperparameter search beat standard GA? |

---

## 4. Kill Criteria (Pre-Registered)

### 4.1 Per-Experiment Kill

An experiment is **killed** (stopped, not continued to next phase) if ANY of these trigger:

| ID | Criterion | Threshold | When Checked |
|----|-----------|-----------|-------------|
| K-1 | Held-out Sharpe | $SR_{held} \leq 0$ | After held-out evaluation |
| K-2 | Worst 2-year Sharpe | $\min SR_{2Y} \leq -0.9$ | After aggregate OOS |
| K-3 | Cost ratio | $CR < 2.0$ | After aggregate OOS |
| K-4 | Binary F1 (Path B binary models) | $F1 < 0.91$ | Per-window validation |
| K-5 | Window OOS consistency | $< 60\%$ windows with $SR > 0$ | After all windows |
| K-6 | Train-test MAE divergence | $\frac{MAE_{test}}{MAE_{train}} > 3.0$ in $> 50\%$ of windows | Cross-window |
| K-7 | No improvement over static | $\Delta SR_{adaptive - static} \leq 0$ AND $p_{bootstrap} > 0.10$ | After held-out |

### 4.2 Path-Level Kill

A **path** (A, B, or C) is abandoned if:
- Its best configuration triggers K-1 (negative held-out Sharpe)
- All model variants within the path trigger K-5 (inconsistent OOS)
- The path shows no advantage over the simplest benchmark (buy-and-hold or zero)

### 4.3 Project-Level Kill

**Project 2 is terminated** if:
- All three paths (A, B, C) are killed
- No adaptive configuration achieves $\Delta SR > 0$ over the P1 static baseline
- Evidence accumulates that EUR/USD at 4h is not profitably predictable with available features (reinforcing the F-6 null finding)

---

## 5. Statistical Testing

### 5.1 Sharpe Ratio Comparison

Comparing two Sharpe ratios requires accounting for autocorrelation and non-normality:

| Test | Method | When to Use |
|------|--------|------------|
| **Ledoit-Wolf (2008)** | HAC-adjusted Sharpe difference test | Primary test for $SR_{adaptive} - SR_{static} > 0$ |
| **Bootstrap (circular block)** | 10,000 resamples of daily returns, block length = 20 | Robustness check; handles non-normality |
| **Paired t-test on window SRs** | $t$-test on $(SR_{w,adaptive} - SR_{w,static})$ across windows $w$ | Simpler but assumes normality of SR differences |

**Significance level**: $\alpha = 0.05$ (one-sided; we're testing "adaptive > static").

**Multiple comparison correction**: Bonferroni within each path (correct for number of model variants tested). Do NOT correct across paths (they test different hypotheses).

### 5.2 Overfitting Detection

| Test | Method | Threshold |
|------|--------|-----------|
| **Deflated Sharpe Ratio (Bailey & López de Prado, 2014)** | Adjusts SR for number of trials, variance, skewness, kurtosis | $DSR > 0.95$ (95% probability the strategy is genuinely positive) |
| **Probability of Backtest Overfitting (PBO)** | Combinatorial cross-validation over all parameter sets | $PBO < 0.50$ (less than 50% chance of overfit) |
| **Train-OOS correlation** | $\text{corr}(SR_{train}, SR_{OOS})$ across windows | $> 0.3$ (positive transfer from train to OOS) |

### 5.3 Regime-Conditional Testing

Test whether strategy performance differs across regimes:

| Test | Method |
|------|--------|
| **Regime-conditional SR** | Compute SR per regime, test heterogeneity via Kruskal-Wallis |
| **Regime profitability** | % of trades profitable per regime, chi-squared test vs uniform |
| **Drawdown clustering** | Test whether drawdowns cluster in specific regimes |

---

## 6. Reporting Template

### 6.1 Per-Experiment Report

Each experiment (one path × one model × one feature set × one window design) produces:

```markdown
## Experiment: [path]_[model]_[features]_[design]

### Configuration
- Path: A/B/C
- Model: [specific model name]
- Features: Phase [0-4], [count] features
- Window design: [anchored/sliding/triggered]
- Windows: [count]

### Kill Criteria Check
| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| K-1 Held-out SR | [value] | > 0 | PASS/FAIL |
| K-2 Worst 2Y SR | [value] | > -0.9 | PASS/FAIL |
| K-3 Cost ratio | [value] | ≥ 2.0 | PASS/FAIL |
| K-5 Window consistency | [value]% | ≥ 60% | PASS/FAIL |
| K-7 Adaptive advantage | [ΔSR], p=[value] | > 0, p < 0.10 | PASS/FAIL |

### Per-Window Results
| Window | Train SR | Val SR | Test SR | MDD | Trades | Cost Ratio |
|--------|---------|--------|---------|-----|--------|------------|
| ... | ... | ... | ... | ... | ... | ... |

### Held-Out Results (2020-2024)
| Metric | Value |
|--------|-------|
| Sharpe Ratio | [value] |
| Max Drawdown | [value] |
| Calmar Ratio | [value] |
| Total Return | [value] |
| Win Rate | [value] |
| Profit Factor | [value] |

### Statistical Tests
| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Ledoit-Wolf SR diff | [value] | [value] | [significant/not] |
| Bootstrap ΔSR | [CI] | [value] | [significant/not] |
| Deflated SR | [value] | — | [genuine/suspect] |

### Verdict
[PASS: Advance to next phase / FAIL: Kill experiment / INCONCLUSIVE: Needs more data]
```

### 6.2 Cross-Path Comparison Report

After all paths are evaluated:

```markdown
## Cross-Path Comparison

| Metric | Path A (best) | Path B (best) | Path C (best) | P1 Static | Buy-and-Hold |
|--------|--------------|--------------|--------------|-----------|--------------|
| Held-out SR | | | | -0.065 | |
| Max DD | | | | 20.18% | |
| Cost Ratio | | | | | N/A |
| Window Consistency | | | | 61.5% | N/A |
| Compute per retrain | | | | N/A | N/A |

### Ranking
1. [Best path] — [key advantage]
2. [Second] — [key advantage]
3. [Third] — [key advantage]
4. P1 Static — [reference]

### Core Hypothesis Verdict
"Periodically re-optimized strategies outperform static" — [SUPPORTED / NOT SUPPORTED / PARTIALLY SUPPORTED]
Evidence: [summary]
```

---

## 7. Transaction Cost Model

### 7.1 EUR/USD

| Cost Component | Value | Source |
|----------------|-------|--------|
| Spread | 1.0 pip (0.0001) | OANDA typical for EUR/USD |
| Commission | 0 | OANDA spread-only model |
| Slippage | 0.5 pip (0.00005) | Conservative estimate for hourly entries |
| **Total round-trip** | 1.5 pips (0.00015) = **1.5 bps on notional** | |

### 7.2 Other Assets (when added)

| Asset | Round-trip Cost | Source |
|-------|----------------|--------|
| USD/JPY | 1.5 pips ≈ 1 bps | OANDA |
| SPY | $0.01/share ≈ 0.2 bps | Estimated commission + spread |
| BTC/USD | 0.1% ≈ 10 bps | Binance spot fee |

### 7.3 Cost Application

Costs are deducted per trade:
```
Net PnL = Gross PnL - (num_trades × cost_per_trade × position_size)
```

All reported metrics (SR, MDD, returns) are **NET of costs** unless explicitly labeled "gross."

---

## 8. Held-Out Protocol

### 8.1 Timing

The held-out evaluation (2020-2024) is performed **exactly once per experiment**, after all in-sample tuning is complete. There is no iteration on the held-out set.

### 8.2 Procedure

1. Select the best configuration from in-sample (2005-2019) based on aggregate validation SR
2. **Freeze** all parameters (no further tuning)
3. For adaptive strategies: apply the retrain schedule to 2020-2024 as if in real-time (retrain at each window boundary using only data available up to that point)
4. Evaluate on 2020-2024 OOS returns
5. Report all metrics from §2

### 8.3 Contamination Prevention

- No analysis of 2020-2024 data during Part I (foundations)
- No feature engineering decisions informed by 2020-2024 patterns
- No hyperparameter tuning on 2020-2024 performance
- If the held-out set is ever accidentally used for tuning, it must be disclosed and results marked as contaminated

---

## 9. Relationship to P1 Evaluation

| P1 Metric | P2 Equivalent | Change |
|-----------|--------------|--------|
| Full-period SR | Per-window OOS SR + Held-out SR | Split into rolling + held-out |
| Walk-forward % positive | Window OOS consistency (K-5) | Same concept, formalized |
| Max drawdown | Max drawdown (same) | Unchanged |
| Cost ratio ≥ 2× | Cost ratio ≥ 2.0 (K-3) | Same, formalized as kill criterion |
| F1 ≥ 0.91 | F1 ≥ 0.91 (K-4, Path B binary only) | Same, narrowed to binary models |

**Key addition in P2**: The $\Delta SR_{adaptive - static}$ test (K-7) — the core hypothesis test that did not exist in P1.

---

## 10. Implementation Notes

### 10.1 Libraries

| Function | Library | Notes |
|----------|---------|-------|
| Sharpe, MDD, Calmar | `quantstats` or manual | Prefer manual for transparency |
| Ledoit-Wolf SR test | Manual (see Ledoit-Wolf 2008 paper) | ~50 lines of code |
| Bootstrap | `arch` or manual | Circular block bootstrap |
| Deflated SR | Manual (see Bailey-López de Prado 2014) | ~30 lines |
| PBO | `pbo` package or manual (CSCV) | Combinatorial symmetric cross-validation |

### 10.2 Automation

The evaluation framework should be a single script:
```bash
python evaluate_experiment.py \
  --experiment_dir experiments/pathB_tft_phase1_anchored/ \
  --baseline_dir experiments/p1_static/ \
  --output report_pathB_tft.md
```

This generates the full per-experiment report from §6.1 automatically.
