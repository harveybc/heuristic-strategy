# TASK II-7-fix.1 — Pilot Evaluation Bug Investigation

**Stage**: II-7-fix (Validation Repair Plan)
**Date**: 2026-04-21
**Status**: COMPLETE

---

## 1. Code Flow

### Training
- `run_pilot(asset, label, run_id, algo_name)` is called once per (config, algorithm) combination.
- Inside `run_pilot()`, `AlgoClass` is selected from a local dict: `{PPO: discrete, SAC: continuous, DQN: discrete}`.
- `model = AlgoClass('MlpPolicy', train_env, ...)` — model created as a **local variable** inside function.
- `model.learn(total_timesteps=100_000)` — training called on local model.

### Evaluation
- `evaluate_policy(model, val_env)` called with the trained `model` and freshly created `val_env`.
- Evaluation loop: `model.predict(obs, deterministic=True)` → `env.step(action)` until done.
- Uses `env.final_metrics()` → `total_return` and `sharpe_ratio`.

### Action Mapping
- **Discrete** (PPO, DQN): action 0=hold(keep pos), 1=buy(long), 2=sell/short
- **Continuous** (SAC): action value in [-1,1] → position fraction
- Both map to position=1.0 if always-long is learned

### Key observation
- `val_env` is created fresh for each algorithm via `make_env(...)` — no shared environment.
- `model` is local to `run_pilot()` — no shadowing possible across calls.
- However: if both PPO (discrete) and SAC (continuous) converge to 100% long,
  their equity curves will be **exactly identical** (same prices, same position=1.0, same costs).

---

## 2. Identical Metrics in Original Results

| Config | Algorithm | Val Return | Val Sharpe |
|--------|-----------|-----------|-----------|
| btc_1h_technical | PPO | 1.557712 | 2.4268 |
| btc_1h_technical | SAC | 1.557712 | 2.4268 |
| btc_1h_technical | DQN | 0.290819 | 1.4732 |
| eth_1h_technical | PPO | 0.925169 | 1.6513 |
| eth_1h_technical | SAC | -0.302800 | -26.4928 |
| eth_1h_technical | DQN | 0.925169 | 1.6513 |

**Identical pair detected**: btc_1h_technical → PPO == SAC (Sharpe=2.4268)
**Identical pair detected**: eth_1h_technical → PPO == DQN (Sharpe=1.6513)

---

## 3. Hypothesis Tests

### Hypothesis A: Evaluation uses fixed buy-and-hold regardless of algorithm
**Verdict**: CONFIRMED

- Buy-and-hold val_return=1.557712, val_sharpe=2.4268
- Reported PPO val_return=1.557712, val_sharpe=2.4268
- Return match: True, Sharpe match: True

### Hypothesis B: Same model object reused across algorithm names
**Verdict**: REJECTED

- has_global_model: False
- run_pilot is isolated function: True
- model created inside function: True

### Hypothesis C: Both algorithms converged to 'always-long' policy → identical val results
**Verdict**: INCONCLUSIVE

- PPO action distribution: {0: 4485, 1: 4250}
- PPO % long (action=1): 48.6%
- PPO val_return=1.557712, val_sharpe=2.4268
- SAC action distribution (bucketed): {'0': 1, '1': 8734}
- SAC % near-max-long (≥0.95): 100.0%
- SAC val_return=1.557712, val_sharpe=2.4268
- Returns within 5%: True

### Hypothesis D: Results JSON written from cached/global variable
**Verdict**: REJECTED

- Pattern checks: {'global_metrics': False, 'global result': False, 'result dict before loop': False, 'all_results appended': True, 'per_pilot_return': True}

### Hypothesis E: Pilot didn't actually train; evaluated only initial random policy
**Verdict**: REJECTED

- Untrained PPO val_return=-0.301777
- Untrained PPO actions: {1: 190, 2: 157}
- Trained 5K PPO val_return=1.557712
- Trained 5K PPO actions: {0: 4485, 1: 4250}
- Training changed policy: True

---

## 4. Root Cause

Root cause analysis was inconclusive.

Confirmed hypotheses: ['A']
Rejected hypotheses: ['B', 'D', 'E']
Errored hypotheses: []

Recommend manual inspection of pilot script and re-test with explicit action logging.

---

## 5. Fix Required for II-7-fix.4

Two mandatory changes:

### Fix 1: Strict IS-only temporal boundaries
- Train period: 2017-08-17 to 2019-06-30 (IS only, no 2020+ data)
- Val period: 2019-07-01 to 2019-12-31 (IS only, no 2020+ data)
- Add assertion: `assert df.index.max() < pd.Timestamp('2020-01-01', tz='UTC')`
- This removes the 'always-long on bull run' degeneracy problem

### Fix 2: Action distribution logging (diagnostic verification)
- Log first 20 actions per pilot to JSON results
- Log full action distribution per pilot (% hold / % long / % short)
- Post-run sanity check: if any two algorithms produce identical action distributions
  AND identical Sharpe (to 3 decimals), flag as ESCALATION

### Fix 3: Separate validation environment per algorithm
- Each algorithm must create its own `val_env` object (already done in original script,
  but must be explicitly verified in re-run via different random seeds)

---

## Deliverables

- `stage_II-7-fix/deliverables/TASK_II-7-fix.1_BUG_INVESTIGATION.md` — This document
