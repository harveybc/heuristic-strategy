"""
Stage II-7.4 — RL Infrastructure verification script
======================================================
Verifies the TradingEnv passes stable_baselines3 env_checker
and basic gymnasium compliance checks.

Run:
    python infrastructure/rl/verify_env.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from infrastructure.rl.trading_env import TradingEnv, DEFAULT_CONFIG


def make_synthetic_data(n_bars: int = 2000, n_features: int = 12, seed: int = 42):
    """Create synthetic feature/price data for env verification."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Geometric Brownian price
    log_returns = rng.normal(0, 0.002, n_bars)
    prices      = 10000.0 * np.exp(np.cumsum(log_returns))
    prices_s    = pd.Series(prices, index=idx, name="Close")

    # Random features
    feat_cols = DEFAULT_CONFIG["feature_cols"]
    feat_data = rng.standard_normal((n_bars, len(feat_cols)))
    features  = pd.DataFrame(feat_data, index=idx, columns=feat_cols)

    return features, prices_s


def verify_env(config_overrides=None):
    from stable_baselines3.common.env_checker import check_env

    features, prices = make_synthetic_data()
    cfg = {**(config_overrides or {})}
    env = TradingEnv(features, prices, cfg)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Total bars:        {env.n_bars}")

    print("\nRunning stable_baselines3 env_checker ...")
    check_env(env, warn=True, skip_render_check=True)
    print("  env_checker: PASS")

    # Manual rollout test
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, "Obs shape mismatch"

    total_steps = 0
    episodes    = 0
    while episodes < 3:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        assert isinstance(reward, float), "Reward not float"
        if terminated or truncated:
            metrics = env.final_metrics()
            print(f"  Episode {episodes+1}: steps={info['step']}, "
                  f"equity={info['equity']:.4f}, sharpe={metrics.get('sharpe_ratio', 0):.3f}")
            obs, info = env.reset()
            episodes += 1

    print(f"\nManual rollout: {total_steps} steps across 3 episodes — PASS")
    return True


def verify_continuous():
    """Also verify continuous action space."""
    features, prices = make_synthetic_data()
    cfg = {"action_space_type": "continuous"}
    env = TradingEnv(features, prices, cfg)
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True, skip_render_check=True)
    print("  Continuous action env_checker: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE II-7.4 — RL ENVIRONMENT VERIFICATION")
    print("=" * 60)

    ok_discrete   = verify_env({"action_space_type": "discrete"})
    ok_continuous = verify_continuous()

    print("\n" + "=" * 60)
    print("All checks PASS — TradingEnv is SB3-compatible")
    print("=" * 60)
