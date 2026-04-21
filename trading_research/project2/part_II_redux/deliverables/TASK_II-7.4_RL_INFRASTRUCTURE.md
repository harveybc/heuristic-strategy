# TASK II-7.4 — RL Infrastructure

**Stage**: II-7 (RL Configuration Reconnaissance)
**Date**: April 2026
**Status**: COMPLETE — SB3 env_checker: PASS

---

## Objective

Build a Gymnasium-compatible single-asset trading environment suitable for Stable-Baselines3 agents, configurable for the alpha configurations identified in II-7.2/7.3 (BTC 1h, ETH 1h, technical features).

---

## Environment Design

File: `infrastructure/rl/trading_env.py`

### Observation Space
- Flattened vector of `lookback × n_features` (default: 1 × 12 = 12)
- Features: returns, log_returns, rsi, macd_hist, bb_pos, volume_ratio, ema_cross, atr_norm, obv_delta, momentum_5, momentum_20, volatility_20
- dtype: float32, bounds: [-inf, +inf]

### Action Spaces
Two modes configurable via `action_space_type`:

| Mode | Type | Values |
|------|------|--------|
| `discrete` | `Discrete(3)` | 0=hold, 1=long, 2=short |
| `continuous` | `Box([-1,1])` | position fraction |

### Reward Functions
Three modes configurable via `reward_type`:

| Mode | Formula |
|------|---------|
| `log_return` | log(1 + strategy_return) |
| `sharpe_incremental` | μ/σ over rolling 60-bar window |
| `risk_adjusted` | log_return − 0.5 × drawdown |

### Episode Mechanics
- Transaction cost: 0.1% of trade value (configurable)
- Max drawdown stop: 30% (configurable, terminates episode)
- Episode length: full dataset slice or configurable `max_steps`
- Reset with optional random start for training variety

### Key Config Parameters

```python
DEFAULT_CONFIG = {
    "feature_cols": [...],       # 12 technical features
    "lookback": 1,               # observation window
    "action_space_type": "discrete",
    "reward_type": "log_return",
    "transaction_cost": 0.001,
    "max_drawdown_stop": 0.30,
    "reward_clip": (-1.0, 1.0),
}
```

---

## Verification Results

Script: `infrastructure/rl/verify_env.py`

```
Observation space: Box(-inf, inf, (12,), float32)
Action space:      Discrete(3)
Total bars:        2000

Running stable_baselines3 env_checker ...
  env_checker: PASS
  Episode 1: steps=481, equity=0.7005, sharpe=-32.721
  Episode 2: steps=451, equity=0.6999, sharpe=-34.766
  Episode 3: steps=518, equity=0.6985, sharpe=-29.734

Manual rollout: 1450 steps across 3 episodes — PASS
  Continuous action env_checker: PASS

All checks PASS — TradingEnv is SB3-compatible
```

Verified on: Omega (RTX4070), stable-baselines3 2.8.0, gymnasium 1.2.3, torch 2.11.0

Note: TensorFlow/protobuf library conflict warnings (`MessageFactory`, `cuDNN factory already registered`) appear at import time but do not affect SB3 functionality.

---

## Usage

```python
from infrastructure.rl.trading_env import TradingEnv

env = TradingEnv(df_features, prices_series, config={
    "action_space_type": "discrete",
    "reward_type": "log_return",
    "transaction_cost": 0.001,
})

from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Deliverables

- `infrastructure/rl/trading_env.py` — TradingEnv class
- `infrastructure/rl/verify_env.py` — SB3 env_checker verification script
- `deliverables/TASK_II-7.4_RL_INFRASTRUCTURE.md` — This document
