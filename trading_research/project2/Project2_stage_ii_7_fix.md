# Project 2 — Stage II-7-fix: Validation Repair Plan

**Purpose:** Stage II-7 produced results with serious methodological issues that must be fixed before Part III commitment. This plan repairs four specific issues with extremely detailed, deterministic instructions.

**Critical issues identified:**

1. **Pilot evaluation bug:** PPO/SAC produced identical metrics for BTC (val Sharpe 2.427 to 4 decimals); PPO/DQN produced identical metrics for ETH (val Sharpe 1.651). This is statistically impossible by chance, indicates evaluation infrastructure bug.

2. **Held-out contamination:** Pilots used Train 2020-2022 / Val 2023 / Test 2024 — but per project framework, 2020-2025 is HO and must not be touched during pilot/training. Pilot test Sharpes are NOT legitimate held-out evidence.

3. **Causal finding fragility:** macd_hist τ=1 found on single year IS (2019), single feature, single lag. Inconsistent with Stage II-0.5 finding (RSI t-6 on 4h). May be regime-specific to 2019.

4. **Cross-asset finding suspicion:** Same feature (macd_hist) at same lag (τ=1) appearing in BOTH BTC 1h AND ETH 1h is suspicious. Could indicate common-factor artifact (crypto market beta) rather than asset-specific causal structure.

5. **Part III scope extrapolation:** Target Sharpe 1.5 has no evidence basis given Project 1 P3 was -0.065 and Part II-Redux best HO was +0.083. Aspirational, not evidence-based.

6. **NEAT advanced specification not honored:** User explicitly specified 5-stage advanced NEAT (neurogenesis, synaptogenesis, pruning, maduración, estabilización) as Part III capstone. Current PART_III_SCOPE_RECOMMENDATION relegates NEAT to "deferred if PPO/DQN saturate."

---

## 0. AGENT EXECUTION RULES (READ BEFORE EVERY TASK)

These rules are MANDATORY. No deviation, no interpretation, no shortcuts.

### Rule 0.1: Follow specifications literally

If this plan says "use period X to Y", use exactly X to Y. Do not extend, shorten, or substitute. If specification is unclear, HALT and produce `ESCALATION_clarification_needed.md`. Do not guess.

### Rule 0.2: No held-out contamination

Held-out period = 2020-01-01 to 2025-12-31. ABSOLUTE PROHIBITION:
- No pilot training touches 2020-2025 data
- No pilot validation touches 2020-2025 data
- No pilot evaluation touches 2020-2025 data
- No causal analysis IS touches 2020-2025 data

If a task seems to require touching 2020-2025 and is not labeled "FINAL HO EVALUATION", HALT and escalate.

### Rule 0.3: All scripts emit explicit JSON results

Every analysis script MUST write results to `deliverables/<task_id>_results.json` with explicit fields. Markdown deliverables are summaries — JSON is source of truth.

### Rule 0.4: Cross-algorithm sanity checks mandatory

When running multiple algorithms on same data, after completion:
- If any two algorithms produce identical Sharpe (to 3+ decimals), HALT immediately
- Produce `ESCALATION_identical_metrics_<asset>.md` with both result files
- Do not proceed until investigated

### Rule 0.5: Conda environment activation

Every SSH command starts with:
```
source /home/harveybc/anaconda3/etc/profile.d/conda.sh && conda activate tensorflow && <command>
```

### Rule 0.6: Progress logging

Every task action logged to `logs/stage_II-7-fix_progress.log`:
```
[ISO timestamp] [task] [action] [status]
```

### Rule 0.7: Each task has user gate

After completing task, produce deliverable, HALT, await user "proceed" before next task.

### Rule 0.8: When in doubt, escalate

If anything unclear, unexpected, or contradicts specifications, produce `ESCALATION_<reason>.md` and HALT. Do NOT make autonomous decisions.

---

## 1. TASK SEQUENCE

| Task | Purpose | Machine | Output |
|------|---------|---------|--------|
| **II-7-fix.1** | Investigate pilot evaluation bug | Omega | Bug root cause identified |
| **II-7-fix.2** | Validate causal findings temporal stability | Dragon | Sub-period analysis |
| **II-7-fix.3** | Validate causal findings cross-asset independence | Dragon | Common-factor test |
| **II-7-fix.4** | Re-run RL pilots with strict IS-only discipline | Dragon + Gamma + Omega | Bug-free pilot evidence |
| **II-7-fix.5** | Re-write Part III scope recommendation | Omega | Conservative evidence-based scope |

Sequential execution. Each task has explicit deliverable and gate.

---

## 2. TASK II-7-fix.1: Pilot Evaluation Bug Investigation

### 2.1 Objective

Identify root cause of identical metrics across PPO/SAC for BTC and PPO/DQN for ETH in Stage II-7.5 pilots.

### 2.2 Specific values to investigate

From `deliverables/pilot_results_II7.json`:

```
BTC PPO: val_return=1.5577, val_sharpe=2.427
BTC SAC: val_return=1.5577, val_sharpe=2.427
ETH PPO: val_return=0.9252, val_sharpe=1.651
ETH DQN: val_return=0.9252, val_sharpe=1.651
```

These identical values to 4 decimals are statistically impossible across different algorithms. Bug location is in pilot training/evaluation code.

### 2.3 Step-by-step procedure

**STEP 1: Locate pilot training script**

```bash
find /home/harveybc -name "*.py" -newer /tmp -path "*stage_ii7*" -o -name "*pilot*"
```

Expected file: `scripts/stage_ii7_algorithm_pilots.py` or similar. If not found, search wider:

```bash
grep -r "val_return.*val_sharpe" /home/harveybc/Documents/GitHub --include="*.py"
```

**STEP 2: Read entire pilot script and document evaluation flow**

Read the script. Document in `TASK_II-7-fix.1_BUG_INVESTIGATION.md` Section "Code Flow":
- How is each algorithm trained?
- How is validation evaluation performed?
- Does evaluation use the trained policy, or default/random/buy-hold?
- Are different algorithms evaluated on identical environment seeds?
- Is there caching, memoization, or fixed-result logic?

**STEP 3: Specific bug hypotheses to check (verify each)**

Check each hypothesis explicitly. For each, write to deliverable: "Hypothesis X: [statement]. Result: CONFIRMED / REJECTED. Evidence: [code line citation or test output]."

**Hypothesis A: Evaluation uses environment with fixed buy-and-hold strategy regardless of algorithm**

Look for code patterns like:
```python
# WRONG
def evaluate(env):
    actions = [1] * len(env)  # always long
    return run_episode(env, actions)
```

vs correct:
```python
# RIGHT
def evaluate(model, env):
    obs = env.reset()
    actions = [model.predict(obs) for _ in steps]
```

**Hypothesis B: Same model object reused across algorithm names**

Look for variable shadowing:
```python
# WRONG
model = PPO(env)
model.learn(timesteps=100000)
ppo_results = evaluate(model)
model = SAC(env)  # but if "model" reference was cached elsewhere
sac_results = evaluate(model)  # might evaluate cached PPO
```

**Hypothesis C: Evaluation seed identical, deterministic environment, same final state**

If env is deterministic and starting state identical, and both algorithms converge to similar policies (e.g., always-long), final equity could be identical.

Test: print final positions taken by each algorithm. If both always-long, root cause is environment/reward not differentiating.

**Hypothesis D: Results JSON written from cached/global variable**

Look at JSON write code:
```python
results = {}
for algo in [PPO, SAC, DQN]:
    train_and_eval(algo)
    results[algo.__name__] = global_metrics  # bug if global_metrics not updated per algo
```

**Hypothesis E: Pilot didn't actually train, evaluated only initial random policy**

Check if `model.learn(total_timesteps=100000)` actually executed or was skipped/short-circuited. Verify training logs exist with loss progression.

**STEP 4: Reproduction test**

Re-run BTC PPO pilot with verbose output. Note exact final policy (action distribution per state). Re-run BTC SAC pilot. Compare action distributions.

If action distributions identical → models converged to same policy (env doesn't differentiate)
If action distributions differ → bug in evaluation/results writing

**STEP 5: Document root cause**

In deliverable, write definitive section "Root Cause": one paragraph stating exactly what was wrong, with line citations.

If multiple bugs found, list all.

If no bug found and identical metrics are genuine (e.g., both algorithms converge to identical optimal policy on this dataset), write: "No bug found. Identical metrics result from [explanation]. This itself is suspicious because [reason]. Recommendation: [action]."

### 2.4 Deliverable II-7-fix.1

`TASK_II-7-fix.1_BUG_INVESTIGATION.md` with sections:

1. **Code Flow** — how pilots train and evaluate
2. **Hypothesis Tests** — A through E with explicit CONFIRMED/REJECTED + evidence
3. **Reproduction Test** — action distributions per algorithm
4. **Root Cause** — definitive statement
5. **Fix Required** — what code changes are needed for II-7-fix.4 re-runs

### 2.5 User gate

User reviews deliverable, confirms understanding, approves proceeding to next task.

---

## 3. TASK II-7-fix.2: Causal Temporal Stability Test

### 3.1 Objective

Determine whether macd_hist τ=1 causal link found in Stage II-7.2 (using IS=2019 only) is temporally stable or regime-specific.

### 3.2 Procedure (deterministic — execute exactly as specified)

**STEP 1: Define sub-periods**

For BTC 1h technical and ETH 1h technical (the two α configurations), test PCMCI+ on each of these IS sub-periods:

| Sub-period | Start | End | Bars (1h) approx |
|-----------|-------|-----|------------------|
| 2017-H2 | 2017-08-17 | 2017-12-31 | 3,300 |
| 2018-H1 | 2018-01-01 | 2018-06-30 | 4,344 |
| 2018-H2 | 2018-07-01 | 2018-12-31 | 4,392 |
| 2019-H1 | 2019-01-01 | 2019-06-30 | 4,344 |
| 2019-H2 | 2019-07-01 | 2019-12-31 | 4,392 |

5 sub-periods × 2 assets = 10 PCMCI+ runs.

For BTC, 2017 H2 may be sparse (Binance launched Aug 2017). If <2000 bars after dropna, skip and document.

**STEP 2: Use exactly these PCMCI+ parameters (do not change)**

```python
PCMCI_PARAMS = {
    "method": "PCMCI+",
    "independence_test": "ParCorr",
    "tau_max": 10,
    "pc_alpha": 0.01,
    "alpha_level": 0.05,
    "max_samples": 5000,
    "target_variable": "forward_return_6",
    "features": [
        "returns", "log_returns", "rsi", "macd_hist", "bb_pos",
        "volume_ratio", "ema_cross", "atr_norm", "obv_delta",
        "momentum_5", "momentum_20", "volatility_20"
    ],
}
```

These are IDENTICAL to Stage II-7.2 parameters. Do not modify.

**STEP 3: Per sub-period, record exactly these fields**

For each (asset, sub-period) PCMCI+ run, write to JSON:

```json
{
  "asset": "BTC",
  "sub_period": "2018-H1",
  "samples_used": <int>,
  "all_lagged_links": [
    {"feature": "<name>", "tau": <int>, "MCI": <float>, "p_value": <float>}
  ],
  "macd_hist_tau1_present": <bool>,
  "macd_hist_tau1_MCI": <float or null>,
  "macd_hist_tau1_p": <float or null>,
  "classification": "α | β | γ"
}
```

**STEP 4: Run all 10 (or fewer if 2017 H2 skipped) sub-period analyses**

Use Dragon (RTX 4090, fastest for PCMCI+). Sequential execution. Each run estimated 3-8 minutes.

**STEP 5: Stability classification**

After all runs complete, classify macd_hist τ=1 as:

- **TEMPORALLY STABLE** if present (p<0.05) in 4+ of 5 sub-periods AND MCI sign consistent
- **PARTIALLY STABLE** if present in 2-3 of 5 sub-periods
- **REGIME-SPECIFIC** if present in only 1 sub-period (likely 2019)
- **NOT STABLE** if present in 0 sub-periods (different from full IS finding)

Write classification per asset to deliverable.

### 3.3 Deliverable II-7-fix.2

`TASK_II-7-fix.2_TEMPORAL_STABILITY.md` with:

1. Table of 10 sub-period results
2. macd_hist τ=1 stability classification per asset
3. Other stable lagged links present in 3+ sub-periods (any feature, any lag)
4. Conclusion: which findings are robust, which are regime-specific

### 3.4 User gate

User reviews. If macd_hist τ=1 is REGIME-SPECIFIC or NOT STABLE, the original II-7.2 finding is weakened. User decides if II-7-fix.4 pilots proceed despite weakened causal evidence.

---

## 4. TASK II-7-fix.3: Cross-Asset Independence Test

### 4.1 Objective

Determine whether macd_hist τ=1 appearing in BOTH BTC 1h AND ETH 1h is independent asset-specific evidence, or is artifact of common crypto market factor.

### 4.2 Specific procedure

**STEP 1: Compute BTC-ETH return correlation**

For 1h returns 2017-2019 (IS only), compute Pearson correlation. If correlation > 0.7, the assets share substantial common factor.

```python
btc_returns = btc_1h_data["close"].pct_change()
eth_returns = eth_1h_data["close"].pct_change()
corr = btc_returns.corr(eth_returns)
```

Record correlation in deliverable.

**STEP 2: Compute residual ETH after removing BTC factor**

```python
import statsmodels.api as sm
# Align timestamps
common = pd.concat([btc_returns, eth_returns], axis=1, join="inner").dropna()
common.columns = ["btc_ret", "eth_ret"]
# Regress ETH on BTC
X = sm.add_constant(common["btc_ret"])
model = sm.OLS(common["eth_ret"], X).fit()
common["eth_residual"] = model.resid
```

ETH residual is ETH return component independent of BTC.

**STEP 3: Compute features on ETH residual**

Recompute the 12 technical features but on ETH residual price series (constructed by cumulative residual returns). Specifically:

```python
eth_residual_price = (1 + common["eth_residual"]).cumprod() * 100  # synthetic price
# Apply same feature engineering pipeline to this price series
```

**STEP 4: Run PCMCI+ on ETH-residual features → ETH-residual forward return**

Use exactly same parameters as II-7.2 (see Section 3.2 STEP 2 above).

**STEP 5: Compare results**

- Original ETH 1h: macd_hist τ=1 present (MCI=0.178)
- Residual ETH 1h: macd_hist τ=1 present? With what MCI?

If macd_hist τ=1 disappears or weakens substantially in residual ETH:
- **CONCLUSION: ETH α was BTC factor artifact, not independent ETH signal**

If macd_hist τ=1 persists in residual ETH with similar MCI:
- **CONCLUSION: ETH has independent macd_hist signal beyond crypto beta**

### 4.3 Deliverable II-7-fix.3

`TASK_II-7-fix.3_CROSS_ASSET_INDEPENDENCE.md` with:

1. BTC-ETH 1h returns correlation (number)
2. ETH residual regression results (R², coefficient on BTC)
3. PCMCI+ on residual ETH: full results
4. Comparison: original vs residual macd_hist τ=1 link
5. Conclusion: independent signal OR common-factor artifact

### 4.4 User gate

User reviews. If common-factor artifact, ETH 1h drops from Part III consideration. If independent, both assets remain.

---

## 5. TASK II-7-fix.4: Re-Run Pilots with Strict IS-Only Discipline

### 5.1 Objective

Re-execute pilot training/validation/evaluation using ONLY IS data (2017-2019 for BTC, 2017-2019 for ETH). Apply bug fix from II-7-fix.1.

### 5.2 STRICT TEMPORAL BOUNDARIES (mandatory)

```
PILOT_TRAIN_START   = "2017-08-17"
PILOT_TRAIN_END     = "2019-06-30"
PILOT_VAL_START     = "2019-07-01"
PILOT_VAL_END       = "2019-12-31"
HELD_OUT_BOUNDARY   = "2020-01-01"  # NEVER TOUCHED
```

For ETH, training start may be later (ETH/USDT Binance launch). Use earliest available data through 2019-06-30.

**ABSOLUTE PROHIBITION: No data from 2020-01-01 onward used in any way during pilots.**

If agent considers using 2020+ data for any reason, HALT and produce `ESCALATION_held_out_attempt.md`.

### 5.3 Apply bug fix from II-7-fix.1

Whatever fix was identified in Task II-7-fix.1, apply it to pilot script BEFORE re-running. If fix was code change, modify the script. If fix was structural (e.g., separate evaluation environment), restructure.

Document the exact change in `TASK_II-7-fix.4_PILOTS_REDONE.md` Section "Bug Fix Applied".

### 5.4 Pilot configuration (use exactly these specs)

```python
PILOT_CONFIGS = [
    {
        "config_id": "btc_1h_technical",
        "asset": "BTC",
        "timeframe": "1h",
        "feature_set": "technical",
        "data_path": "data/raw/binance/btcusdt_1h_2019_2025.parquet",
    },
    {
        "config_id": "eth_1h_technical",
        "asset": "ETH",
        "timeframe": "1h",
        "feature_set": "technical",
        "data_path": "data/raw/binance/ethusdt_1h_2019_2025.parquet",
    },
]

ALGORITHMS = [
    {"name": "PPO", "lib": "stable_baselines3.PPO", "policy": "MlpPolicy"},
    {"name": "SAC", "lib": "stable_baselines3.SAC", "policy": "MlpPolicy"},
    {"name": "DQN", "lib": "stable_baselines3.DQN", "policy": "MlpPolicy"},
]

PILOT_PARAMS = {
    "total_timesteps": 100_000,
    "reward_type": "log_return",
    "transaction_cost": 0.001,
    "max_drawdown_stop": 0.30,
    "seed": 42,
}
```

2 configs × 3 algorithms = 6 pilot runs.

Note: SAC requires continuous action space, DQN requires discrete. Configure env accordingly per algorithm:
- PPO: discrete (3-action)
- SAC: continuous (Box([-1,1]))
- DQN: discrete (3-action)

### 5.5 Per-pilot procedure (deterministic)

For each (config, algorithm) combination:

**STEP 1: Load data**

```python
df = pd.read_parquet(config["data_path"])
df = df[(df.index >= PILOT_TRAIN_START) & (df.index <= PILOT_VAL_END)]
# Verify no held-out contamination
assert df.index.max() < pd.Timestamp(HELD_OUT_BOUNDARY), "HELD-OUT CONTAMINATION DETECTED"
```

Add the assertion. If it fails, HALT.

**STEP 2: Split**

```python
train_df = df[(df.index >= PILOT_TRAIN_START) & (df.index <= PILOT_TRAIN_END)]
val_df = df[(df.index >= PILOT_VAL_START) & (df.index <= PILOT_VAL_END)]
```

**STEP 3: Build features**

Use existing feature engineering (12 technical features). Apply z-score normalization fit on train only.

**STEP 4: Build training env, train**

```python
train_env = TradingEnv(train_df, ...)
model = AlgorithmClass(policy, train_env, seed=42, verbose=1)
model.learn(total_timesteps=100_000)
```

**STEP 5: Build validation env, evaluate trained model**

```python
val_env = TradingEnv(val_df, ...)
obs, _ = val_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = val_env.step(action)
    done = terminated or truncated
val_metrics = val_env.get_metrics()
```

CRITICAL: Use `model.predict(obs)`, not random action, not always-long. Verify by logging first 10 actions per pilot.

**STEP 6: Compute metrics**

```python
val_return = val_env.get_total_return()
val_sharpe = val_env.get_sharpe_ratio()
val_max_dd = val_env.get_max_drawdown()
val_actions_distribution = val_env.get_action_counts()  # for sanity check
val_num_trades = val_env.get_num_trades()
```

**STEP 7: Save per-pilot result**

```python
result = {
    "config_id": config["config_id"],
    "algorithm": algo["name"],
    "train_period": [PILOT_TRAIN_START, PILOT_TRAIN_END],
    "val_period": [PILOT_VAL_START, PILOT_VAL_END],
    "val_return": val_return,
    "val_sharpe": val_sharpe,
    "val_max_dd": val_max_dd,
    "val_num_trades": val_num_trades,
    "val_actions_distribution": val_actions_distribution,
    "first_10_actions": [...],
    "training_loss_final": <from training log>,
}
json.dump(result, open(f"deliverables/pilot_{config_id}_{algo}.json", "w"))
```

### 5.6 MANDATORY post-execution sanity checks

After all 6 pilots complete:

**CHECK 1: No identical metrics across algorithms**

```python
metrics_by_config = defaultdict(list)
for result in all_results:
    metrics_by_config[result["config_id"]].append({
        "algo": result["algorithm"],
        "val_sharpe": result["val_sharpe"],
        "val_return": result["val_return"],
    })

for config, results_list in metrics_by_config.items():
    sharpes = [r["val_sharpe"] for r in results_list]
    if len(set([round(s, 3) for s in sharpes])) < len(sharpes):
        # Two or more algorithms produced identical Sharpe to 3 decimals
        # ESCALATION
        produce_escalation(...)
        sys.exit(1)
```

If escalation triggered, HALT immediately. Do not proceed to deliverable.

**CHECK 2: Action distributions differ across algorithms**

For each config, compare action distributions across algorithms:

```python
for config, results_list in metrics_by_config.items():
    distributions = [r["val_actions_distribution"] for r in results_list]
    if all(d == distributions[0] for d in distributions):
        # All algorithms produced identical action sequences
        # ESCALATION
        produce_escalation(...)
        sys.exit(1)
```

**CHECK 3: Training actually occurred**

Verify training loss decreased per algorithm. If final loss = initial loss, training was no-op.

### 5.7 Compute distribution

- **Dragon (RTX 4090):** SAC pilots (2: BTC, ETH) — replay buffer heavy
- **Gamma (RTX 5070 Ti):** PPO pilots (2: BTC, ETH) — on-policy, fast
- **Omega (RTX 4070):** DQN pilots (2: BTC, ETH) — moderate, plus orchestration

### 5.8 Deliverable II-7-fix.4

`TASK_II-7-fix.4_PILOTS_REDONE.md` with:

1. **Bug Fix Applied** — exact code change made
2. **Temporal Boundary Verification** — confirmation no 2020+ data touched (assertion result)
3. **Pilot Results Table** with columns: config, algorithm, train_period, val_period, val_return, val_sharpe, val_max_dd, val_num_trades
4. **Sanity Check Results** — Check 1, 2, 3 all PASS
5. **Action Distributions** — per pilot, distribution of actions taken in val
6. **Learnability Verdict per Pilot** — based on val Sharpe > 0 AND val_num_trades > 5

### 5.9 User gate

User reviews. Pilots either show learnability (some val Sharpe > 0, distinct algorithm behaviors) or null result (no learning).

---

## 6. TASK II-7-fix.5: Conservative Part III Scope Recommendation

### 6.1 Objective

Replace `PART_III_SCOPE_RECOMMENDATION.md` with version based on actual evidence (post II-7-fix.1 through II-7-fix.4), with conservative targets and explicit NEAT advanced placement.

### 6.2 Mandatory content sections

The new document MUST contain these sections in this order:

**Section 1: Evidence Summary**

- Causal evidence: from II-7-fix.2 stability + II-7-fix.3 independence
  - State explicitly: how many features have stable lagged links
  - State explicitly: which assets have independent (non-common-factor) signal
- Pilot evidence: from II-7-fix.4 redone pilots
  - State explicitly: which (config, algorithm) showed learnability on IS-only val
  - State explicitly: any pilot that failed sanity checks
- Bug fix evidence: from II-7-fix.1
  - State explicitly: bug found and fixed, OR no bug found

**Section 2: Targets (Evidence-Based, NOT Aspirational)**

Mandatory target table:

```
| Metric | Minimum Acceptable | Realistic Target | Aspirational |
|--------|-------------------|------------------|--------------|
| HO 2020-2025 Sharpe (DSR-adjusted) | > 0.0 with CI | > 0.20 | > 0.50 |
| HO Max Drawdown | < 50% | < 35% | < 25% |
| Win rate | > 48% | > 50% | > 53% |
| Cost ratio | > 2.0 | > 3.0 | > 4.0 |
```

Rationale per metric (mandatory): cite Project 1 P3 result (-0.065) and Part II-Redux best HO (+0.083). Targets must be modest improvements over these, not 20× extrapolations.

DO NOT use "Sharpe > 1.5" anywhere. Such targets are aspirational without evidence.

**Section 3: Configurations to Test**

Based on II-7-fix evidence:

- If BTC 1h temporal stability PASS and pilots learnable: include BTC 1h
- If ETH 1h independence test PASS and pilots learnable: include ETH 1h
- If ETH was common-factor artifact: drop ETH, include only BTC

State explicit reasoning per inclusion/exclusion.

**Section 4: Algorithms**

Based on II-7-fix.4 redone pilots, list algorithms in priority order:

- Primary: algorithm with best val Sharpe across both configs
- Secondary: second-best algorithm
- Tertiary: third (if any showed learnability)

Failed algorithms (val Sharpe ≤ 0) explicitly excluded with reasoning.

**Section 5: NEAT (CRITICAL — user-specified)**

Mandatory subsection structure:

```
### 5.1 NEAT Simple (Phase 2 of Part III)

NEAT simple-connected architecture as comparison baseline against PPO/SAC/DQN.
Specification: standard NEAT (Stanley & Miikkulainen 2002) connected directly
to feature inputs. Population size 150, generations 100, standard mutation rates.

### 5.2 NEAT Advanced (Part III CAPSTONE — user-specified design)

Per user specification, advanced NEAT is the CAPSTONE experiment of Part III,
not optional or conditional. It will be implemented and executed regardless
of PPO/SAC/DQN outcomes.

Specification:
- External feature extractor trained separately (architecture TBD: ANN/CNN/LSTM/VAE)
- Features extracted from training data, used as NEAT input
- 5-stage evolutionary cycle:
  1. Neurogenesis — add nodes
  2. Synaptogenesis — add connections
  3. Pruning — remove weak elements
  4. Maduración — parameter stabilization
  5. Estabilización — final convergence
- Early stopping with patience on validation set (per neat_optimizer in predictor repo)
- Experience replay buffer (modern RL technique applied to neuroevolution)
- Other modernizations as documented during Part III implementation

This is substantial implementation work and is the explicit deliverable
of Part III's final phase.
```

DO NOT relegate to "optional" or "if PPO/DQN saturate." User specified as mandatory capstone.

**Section 6: Held-Out Evaluation Protocol**

Strict rules:
- Held-out 2020-2025 touched ONCE per (config, algorithm) at final evaluation
- Apply F-10 kill criteria K-1 through K-7
- Compute Deflated Sharpe Ratio with N = total Part III experiments tested

**Section 7: Sequence of Execution**

Phase ordering:
1. Modern RL training (PPO, SAC, DQN) on chosen configs — fail fast, IS validation
2. Best modern RL: held-out evaluation
3. NEAT simple comparison
4. NEAT advanced capstone

Highest-risk-of-failure first to fail fast.

**Section 8: Compute Distribution**

Per machine, per algorithm, per phase. Heaviest on Dragon, medium on Gamma, light on Omega.

**Section 9: Decision Tree (mandatory)**

```
Phase 1 modern RL outcome:
├── At least 1 (config, algo) HO Sharpe DSR-adjusted > 0
│   └── Proceed to NEAT comparisons (Section 5)
│       Final synthesis recommends Part III as success
│
└── All HO Sharpes ≤ 0 or DSR fails
    └── Still execute NEAT advanced capstone (user specified mandatory)
        Final synthesis honest: modern RL also failed, NEAT result determines next step
```

### 6.3 Deliverable II-7-fix.5

`PART_III_SCOPE_RECOMMENDATION_v2.md` (replaces v1) with all mandatory sections above.

Original `PART_III_SCOPE_RECOMMENDATION.md` archived to `deliverables/superseded/PART_III_SCOPE_RECOMMENDATION_v1.md` with note "Superseded by v2 due to Stage II-7-fix evidence updates."

### 6.4 User gate

User reviews v2 scope recommendation. Approves Part III start, or requests further revisions.

---

## 7. ESCALATION PROTOCOL

If at any point any of these occur, HALT and produce escalation:

| Condition | File | Action |
|-----------|------|--------|
| Cannot find pilot script in II-7-fix.1 | `ESCALATION_pilot_script_not_found.md` | List all searches attempted |
| Bug investigation inconclusive | `ESCALATION_bug_inconclusive.md` | Document hypotheses tried |
| macd_hist τ=1 not stable in II-7-fix.2 | (proceed to deliverable) | Document, user decides |
| ETH common-factor artifact in II-7-fix.3 | (proceed to deliverable) | Document, user decides |
| Held-out contamination detected in II-7-fix.4 | `ESCALATION_held_out_contamination.md` | Halt all pilots, do not write results |
| Identical metrics in II-7-fix.4 sanity checks | `ESCALATION_identical_metrics_v2.md` | Halt, do not write deliverable |
| Any sanity check fails in II-7-fix.4 | `ESCALATION_sanity_check_failed.md` | Halt, document failure |

After producing escalation, await user decision. Do NOT make autonomous fix attempts.

---

## 8. FILE STRUCTURE (mandatory paths)

```
trading_research/project2/part_II_redux/stage_II-7-fix/
├── deliverables/
│   ├── TASK_II-7-fix.1_BUG_INVESTIGATION.md
│   ├── TASK_II-7-fix.2_TEMPORAL_STABILITY.md
│   ├── TASK_II-7-fix.3_CROSS_ASSET_INDEPENDENCE.md
│   ├── TASK_II-7-fix.4_PILOTS_REDONE.md
│   ├── PART_III_SCOPE_RECOMMENDATION_v2.md
│   ├── superseded/
│   │   └── PART_III_SCOPE_RECOMMENDATION_v1.md
│   ├── temporal_stability_results.json
│   ├── cross_asset_independence_results.json
│   ├── pilot_redone_results.json
│   └── pilot_<config>_<algo>.json (one per pilot)
├── scripts/
│   ├── investigate_pilot_bug.py
│   ├── temporal_stability.py
│   ├── cross_asset_independence.py
│   └── pilots_redone.py
├── logs/
│   └── stage_II-7-fix_progress.log
└── escalations/
    └── (ESCALATION_*.md files if any)
```

---

## 9. SEQUENCE SUMMARY (for agent reference)

```
START
   │
   ▼
TASK II-7-fix.1: Bug investigation (Omega)
   │ [user gate: review bug findings]
   ▼
TASK II-7-fix.2: Temporal stability (Dragon, 10 PCMCI+ runs)
   │ [user gate: review stability]
   ▼
TASK II-7-fix.3: Cross-asset independence (Dragon, 1 PCMCI+ run on residuals)
   │ [user gate: review independence]
   ▼
TASK II-7-fix.4: Pilots redone with bug fix + IS-only (Dragon + Gamma + Omega)
   │ [user gate: review pilot results]
   ▼
TASK II-7-fix.5: Part III scope v2 (Omega, document writing)
   │ [user gate: review and approve Part III start]
   ▼
END (user decides whether to proceed to Part III)
```

---

## 10. AGENT REMINDERS

1. Follow specifications LITERALLY. No interpretation.
2. If unclear, ESCALATE. Do not guess.
3. Held-out 2020-2025 is UNTOUCHABLE.
4. Cross-algorithm sanity checks are MANDATORY.
5. Each task has user gate.
6. All scripts emit JSON results.
7. Document everything.
8. Use SSH conda activation pattern always.
9. Write to specified file paths exactly.
10. When in doubt, halt and ask.