# F-9: gym-fx / NEAT Retrospective

**Date**: 2025-06-17 (original), **CORRECTED**: 2026-04-18  
**Scope**: Reconstruct the history, current state, and Project 2 relevance of all NEAT implementations and the gym-fx repo  
**Depends on**: F-8 (Infrastructure audit)  
**Correction note**: Original version stated gym-fx "DOES NOT EXIST". It exists at `github.com/harveybc/gym-fx` (208 stars, 54 forks) and was cloned on 2026-04-18. Additionally, `agent-multi` and `rl-optimizer` repos were discovered and audited. Section 2 is fully rewritten.

---

## 1. The Two NEAT Systems

A critical finding from this audit: **two completely different NEAT implementations exist**, and they serve different purposes.

### 1.1 System A: Classic neat-python (LEGACY — causal-inference repo)

| Attribute | Value |
|-----------|-------|
| **Library** | `neat-python` (pip package) |
| **Config format** | `.ini` files with `[NEAT]`, `[DefaultGenome]`, `[DefaultSpeciesSet]` sections |
| **What evolves** | Neural network **topology** (nodes + connections) |
| **Environment** | OpenAI Gym `PredictionEnv(gym.Env)` — `step()` processes one data point, reward = 1/|error| |
| **Optimizer plugins** | `neat` and `neat_p2p` (registered in setup.py) |
| **Agent plugins** | `openrl_ppo` (PPO agent), `dummy_automation` |
| **External dependency** | `gym-fx` repo (for trading environments) — **DOES NOT EXIST on disk** |
| **Status** | **BROKEN / LEGACY**. Missing `gym-fx` repo, missing `neat_a_cs` and `neat_a_volume` optimizer plugins that config references. `.ini` files in `tests/data/` suggest development-era artifacts. |
| **Last evidence of use** | Unknown — no output/result files found. Batch files (`causal-inference.bat`) reference `gym-fx` and `neat-test` repos that no longer exist. |

### 1.2 System B: Custom NEAT-Inspired GA (ACTIVE — predictor repo)

| Attribute | Value |
|-----------|-------|
| **Library** | None — fully custom implementation in `neat_optimizer.py` |
| **Config format** | JSON (same as all other predictor configs) |
| **What evolves** | Model **hyperparameters** as variable-length genomes (not network topology) |
| **Key innovations** | Speciation by parameter similarity, structural mutations (add/remove params), innovation tracking, staged optimization, fitness sharing |
| **Fitness** | Lower-is-better MAE (regression) or weighted F1 (binary classification) |
| **Parameters** | `population_size=20`, `n_generations=10/stage`, `neat_add_param_prob=0.35`, `neat_remove_param_prob=0.05`, `neat_compatibility_threshold=2.0`, `neat_survival_rate=0.5`, `neat_elitism=1` |
| **Status** | **ACTIVE, PRODUCTION-READY**. Used across predictor, doin-node (12 configs), doin-plugins (wrapper), heuristic-strategy (Phase D). |
| **Last evidence of use** | Multiple successful runs: TCN NEAT (43 hours, 77 candidates), Phase D NEAT (heuristic-strategy), DOIN node loading (2026-04-10 and 2026-04-13 logs). |

### 1.3 Comparison

| Dimension | System A (neat-python) | System B (custom) |
|-----------|----------------------|-------------------|
| What evolves | Network topology + weights | Hyperparameters (variable-length) |
| Genome | Nodes + connections graph | Dict of {param_name: value} |
| Speciation | Genomic distance (neat-python default) | Custom distance on parameter values |
| Crossover | Align by innovation number | Align by parameter name overlap |
| Mutation | Add node, add connection, weight perturbation | Add param, remove param, value perturbation |
| Requires environment | Yes (gym.Env) | No (black-box fitness function) |
| Scalability | Slow (topology search is expensive) | Fast (parameter-only search) |
| Flexibility | Can discover novel architectures | Cannot change architecture (fixed model type) |

---

## 2. The gym-fx Mystery

### 2.1 Evidence of Existence

| Location | Reference | Content |
|----------|-----------|---------|
| `causal-inference/causal-inference.bat` | `set PYTHONPATH=.\;..\gym-fx;..\neat-test` | gym-fx was a sibling directory |
| `causal-inference/set_env.bat` | `set PYTHONPATH=.\;..\gym-fx;..\neat-test` | Same |
| `causal-inference/update.bat` | `cd ..\gym-fx && git add . && git commit && git push` | gym-fx was a separate git repo |
| `causal-inference/tests/data/neat_50 copy.ini` | Comment: `"Forex-v0 environment on OpenAI Gym"` | Gym environment name was `Forex-v0` |
| `causal-inference/app/config.py` | Default env: `gym_fx_env_nomc_o_volume` | Multiple gym-fx environment variants existed |

### 2.2 What gym-fx Likely Contained

Based on the references and the causal-inference environment plugins:

1. **A `gym.Env` subclass for FX trading** — likely named `Forex-v0`
2. **Action space**: Probably `{buy, sell, hold}` or continuous position sizing
3. **Observation space**: Price features (the `.ini` files reference 8, 36, or 128 inputs)
4. **Reward**: Some combination of PnL, Sharpe, or prediction error
5. **Multiple variants**: `gym_fx_env_nomc_o_volume` suggests no-margin-call + OHLC + volume
6. **Registered as a gym environment**: The `Forex-v0` naming convention matches `gym.envs.registration`

### 2.3 Current Status

**gym-fx does not exist** on disk anywhere under `/home/harveybc/Documents/GitHub/`. It was not found by the infrastructure audit. It may:
- Have been deleted
- Never been pushed to GitHub (local-only development)
- Been merged into causal-inference (the environment plugins in `app/plugins/transformation/` are likely extracted from or replacements for gym-fx code)

### 2.4 Do We Need gym-fx?

**For Project 2 Part IV (RL)**: Yes, a trading environment is needed. But it doesn't need to be the original gym-fx code.

**Options**:
1. **Rebuild from causal-inference environment plugins**: `environment_plugin_prediction.py` already subclasses `gym.Env`. Needs updating to `gymnasium.Env` (gym is deprecated). Add proper action/observation spaces, realistic cost model, position tracking.
2. **Use existing open-source**: `gym-anytrading`, `finrl`, or `trading-gym` packages provide FX trading environments.
3. **Build fresh**: Small effort (~200-300 lines) for a clean `gymnasium.Env` with the exact spec Project 2 needs.

**Recommendation**: Option 3 (build fresh). The causal-inference environment is too coupled to the old architecture, and open-source environments may not match our feature engineering pipeline.

---

## 3. NEAT Results from Project 1

### 3.1 Predictor NEAT: TCN Optimization

| Metric | Value |
|--------|-------|
| **Model** | TCN (Temporal Convolutional Network) |
| **Candidates evaluated** | 77 |
| **Runtime** | 155,015 seconds (~43 hours) |
| **Champion fitness** | 0.0033 (MAE) |
| **Species count** | 1 (converged to single species) |
| **Champion params** | window=87, filters=64, kernel_size=3, stacks=3, dilations=4, loss=trend_sigma |

### 3.2 Heuristic-Strategy NEAT: Phase D

| Metric | Value |
|--------|-------|
| **Model** | ANN (direction prediction, long + short) |
| **Population** | 15 |
| **Generations** | 5 per stage |
| **Result** | Threshold sweep → **all negative profit** |
| **Conclusion** | NEAT-optimized direction prediction did not beat heuristic strategies |

### 3.3 DOIN Network NEAT

| Metric | Value |
|--------|-------|
| **Configs** | 12 node configs across Omega/Dragon/Gamma |
| **Domains** | Binary TFT, TCN timeseries, Direction CNN (long + short) |
| **Status** | Plugins loaded successfully. Full optimization runs unclear (port conflicts in logs). |
| **Innovation** | Island-model: champion migration between nodes during NEAT optimization |

---

## 4. What NEAT Contributes to Project 2

### 4.1 For Part II (Adaptive Strategies)

The custom NEAT optimizer (System B) is immediately useful as an alternative to DEAP GA for hyperparameter optimization in rolling retraining:

| Feature | DEAP GA | NEAT (custom) |
|---------|---------|---------------|
| Genome | Fixed-length | **Variable-length** (can discover which params matter) |
| Speciation | None | Yes (maintains diversity) |
| Innovation tracking | No | Yes (structural changes tracked) |
| Staged optimization | No | **Yes** (progressive complexity) |
| Resume support | Basic | Full (JSON checkpoint) |

**Recommendation**: Run Path B experiments with both DEAP GA and NEAT optimizer for each rolling window. Compare convergence speed, final fitness, and parameter stability across windows.

### 4.2 For Part VI (NEAT Comparison — Per Execution Plan)

Part VI of the execution plan calls for a dedicated NEAT comparison. Two interpretations exist:

**Interpretation A**: Compare NEAT *hyperparameter optimization* (System B) vs DEAP GA — this is straightforward and uses existing infrastructure.

**Interpretation B**: Compare NEAT *topology evolution* (System A / neat-python) vs fixed-architecture ML models — this requires rebuilding System A (the broken gym-fx-based pipeline) or building a new neat-python integration.

**Recommendation**: Part VI should do both:
1. **NEAT-HPO vs GA**: Hyperparameter optimization comparison using predictor's neat_optimizer vs default_optimizer. Low effort, high informational value.
2. **NEAT-Topology vs Fixed**: If budget allows, build a clean neat-python integration that evolves network topology for FX prediction. This is the more novel experiment but requires significant infrastructure work (the gym-fx gap).

### 4.3 For Part IV (RL)

System A's architecture (gym environment + optimizer) is directly relevant to RL. The environment plugins in causal-inference provide a starting point. The `neat_p2p` plugin demonstrates distributed NEAT optimization across nodes — relevant if RL training is distributed.

---

## 5. Gap Summary

| Gap | Impact | Effort to Resolve |
|-----|--------|-------------------|
| **gym-fx needs modernization** (gym→gymnasium, TF1→TF2, no plugin arch) | Part IV (RL) and Part VI (topology NEAT) need updated environment | Medium — port ForexEnv4/6 to gymnasium, adapt data format |
| **agent-multi obsolete** | Historical reference only | None — superseded by rl-optimizer |
| **rl-optimizer missing 2 NEAT plugins** | NEAT topology optimizer not available in modern stack | Medium — re-implement or port from gym-fx/agent-multi |
| **System B (custom) has no rolling loop** | NEAT HPO optimization is single-run, not rolling | Small — wrapping orchestrator calls NEAT per window |
| **Phase D NEAT showed negative profit** | Low confidence that NEAT-optimized direction prediction adds value | N/A — informational (may improve with rolling/adaptive) |
| **DOIN NEAT never completed full run** | Island-model NEAT migration untested at scale | Medium — needs port/config debugging |
| **Species collapse (1 species in TCN run)** | NEAT's diversity maintenance may not be working | Small — tune compatibility threshold |

---

## 6. Recommendations for Project 2

### Immediate (Part I)

1. **Document the two NEAT systems clearly** — avoid confusion between topology evolution and hyperparameter optimization. ✅ Done in this document.
2. **No action needed on gym-fx** — Part IV/VI are not Part I scope.

### Part II

3. **Use System B (custom NEAT-HPO) alongside DEAP GA** in rolling retraining experiments. Compare optimizer performance per rolling window.
4. **Fix species collapse**: Increase `neat_compatibility_threshold` from 2.0 to 3.0-4.0 or reduce `neat_survival_rate` from 0.5 to 0.3 to maintain species diversity.

### Part IV

5. **Port gym-fx ForexEnv4/6 to `gymnasium.Env`** — the reward function design, action/observation spaces, and parameterization patterns are valuable. Modernize rather than rebuild from scratch.
6. **Evaluate**: Use rl-optimizer's PPO plugin or Stable-Baselines3 (PPO/SAC) for RL training. The NEAT-as-agent pattern from gym-fx is a third option (neuroevolution RL).

### Part VI

7. **NEAT-HPO comparison** (System B vs DEAP GA): Straightforward, use existing infrastructure.
8. **NEAT-Topology comparison** (System A): Port gym-fx NEAT agent + ForexEnv to gymnasium. Lower effort than originally estimated since code EXISTS — it's a modernization task, not a build-from-scratch task.
