"""
Stage II-7.4 — RL Trading Environment
======================================
A Gymnasium-compatible trading environment for RL pilot training.

Supports configurable:
  - observation space (feature set)
  - action space (discrete 3-action: hold/buy/sell, or continuous position sizing)
  - reward function (log return, Sharpe-incremental, risk-adjusted)
  - transaction cost model
  - episode length / reset strategy

Usage:
    env = TradingEnv(df_features, df_prices, config)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ── Default config ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Feature columns in df_features to use as observations
    "feature_cols": [
        "returns", "log_returns", "rsi", "macd_hist", "bb_pos",
        "volume_ratio", "ema_cross", "atr_norm", "obv_delta",
        "momentum_5", "momentum_20", "volatility_20"
    ],
    # Lookback window stacked into observation vector
    "lookback": 1,
    # Action space: "discrete" (hold/buy/sell) or "continuous" (position fraction)
    "action_space_type": "discrete",
    # Reward type: "log_return", "sharpe_incremental", "risk_adjusted"
    "reward_type": "log_return",
    # Transaction cost as fraction of trade value
    "transaction_cost": 0.001,
    # Max steps per episode (None = full dataset)
    "max_steps": None,
    # Initial capital (normalised to 1.0 for dimensionless returns)
    "initial_capital": 1.0,
    # Clip rewards to this range (None = no clip)
    "reward_clip": (-1.0, 1.0),
    # Stop loss: terminate episode if equity drawdown exceeds this fraction
    "max_drawdown_stop": 0.30,
    # Sharpe window for sharpe_incremental reward
    "sharpe_window": 60,
}


# ── Trading Environment ────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Single-asset trading environment with configurable obs/action/reward.

    Parameters
    ----------
    df_features : pd.DataFrame
        Technical/auxiliary features, index aligned with df_prices.
        NaN rows are dropped at init.
    df_prices : pd.Series or pd.DataFrame
        Close prices, same index as df_features.
    config : dict
        Override DEFAULT_CONFIG keys as needed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.Series,
        config: dict | None = None,
    ):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        # Align and drop NaN
        feat_cols = cfg["feature_cols"]
        combined  = df_features[feat_cols].copy()
        combined["_close"] = df_prices.values if hasattr(df_prices, "values") else df_prices
        combined.dropna(inplace=True)

        self._features = combined[feat_cols].values.astype(np.float32)
        self._prices   = combined["_close"].values.astype(np.float64)
        self._n        = len(self._features)

        if self._n < 2:
            raise ValueError("Dataset too small after NaN drop.")

        self._lookback = cfg["lookback"]
        self._n_feat   = len(feat_cols)

        # Observation space: lookback × n_features (flattened)
        obs_dim = self._lookback * self._n_feat
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space
        if cfg["action_space_type"] == "discrete":
            # 0=hold, 1=buy (long), 2=sell/short
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous position fraction in [-1, 1]
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self._max_steps = cfg["max_steps"] or (self._n - self._lookback)
        self._tc        = cfg["transaction_cost"]
        self._reward_type = cfg["reward_type"]
        self._reward_clip = cfg["reward_clip"]
        self._max_dd    = cfg["max_drawdown_stop"]
        self._sharpe_window = cfg["sharpe_window"]

        self._step  = 0
        self._pos   = 0.0      # current position fraction: -1 short, 0 flat, +1 long
        self._equity     = cfg["initial_capital"]
        self._peak_equity = cfg["initial_capital"]
        self._equity_history: list[float] = []
        self._start_idx = self._lookback

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Optionally randomise start point for training variety
        if options and options.get("random_start", False):
            max_start = max(self._lookback, self._n - self._max_steps - 1)
            self._start_idx = self.np_random.integers(self._lookback, max_start + 1)
        else:
            self._start_idx = self._lookback

        self._step         = 0
        self._pos          = 0.0
        self._equity       = self.cfg["initial_capital"]
        self._peak_equity  = self.cfg["initial_capital"]
        self._equity_history = [self._equity]
        self._ret_buffer: list[float] = []

        obs  = self._get_obs()
        info = {"equity": self._equity, "position": self._pos}
        return obs, info

    def step(self, action):
        cur_idx  = self._start_idx + self._step
        next_idx = cur_idx + 1

        if next_idx >= self._n:
            # Reached end of data
            obs = self._get_obs()
            return obs, 0.0, True, False, self._info()

        # Map discrete action to target position
        if self.cfg["action_space_type"] == "discrete":
            target_pos = {0: self._pos, 1: 1.0, 2: -1.0}[int(action)]
        else:
            target_pos = float(np.clip(action[0], -1.0, 1.0))

        # Transaction cost on position change
        delta_pos = abs(target_pos - self._pos)
        tc_cost   = delta_pos * self._tc

        # Price return
        p0 = self._prices[cur_idx]
        p1 = self._prices[next_idx]
        price_ret = (p1 - p0) / p0 if p0 != 0 else 0.0

        # Strategy return
        strat_ret = self._pos * price_ret - tc_cost
        self._pos = target_pos

        # Update equity
        self._equity *= (1.0 + strat_ret)
        self._equity_history.append(self._equity)
        self._ret_buffer.append(strat_ret)
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Compute reward
        reward = self._compute_reward(strat_ret)
        if self._reward_clip:
            reward = float(np.clip(reward, *self._reward_clip))

        self._step += 1

        # Termination conditions
        drawdown = (self._peak_equity - self._equity) / self._peak_equity
        terminated = bool(drawdown > self._max_dd)
        truncated  = bool(self._step >= self._max_steps)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._info()

    def render(self):
        pass  # No rendering required for pilot experiments

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx = self._start_idx + self._step
        start = max(0, idx - self._lookback)
        chunk = self._features[start:idx]
        if len(chunk) < self._lookback:
            # Pad with first row
            pad   = np.tile(self._features[0], (self._lookback - len(chunk), 1))
            chunk = np.vstack([pad, chunk])
        return chunk.flatten().astype(np.float32)

    def _compute_reward(self, strat_ret: float) -> float:
        if self._reward_type == "log_return":
            return float(np.log1p(strat_ret)) if strat_ret > -1.0 else -1.0

        elif self._reward_type == "sharpe_incremental":
            buf = self._ret_buffer[-self._sharpe_window:]
            if len(buf) < 2:
                return 0.0
            mu  = np.mean(buf)
            std = np.std(buf)
            return float(mu / std) if std > 1e-9 else 0.0

        elif self._reward_type == "risk_adjusted":
            # Calmar-like: return penalised by drawdown
            drawdown = (self._peak_equity - self._equity) / self._peak_equity
            log_ret  = float(np.log1p(strat_ret)) if strat_ret > -1.0 else -1.0
            penalty  = drawdown * 0.5
            return log_ret - penalty

        else:
            raise ValueError(f"Unknown reward_type: {self._reward_type}")

    def _info(self) -> dict:
        drawdown = (self._peak_equity - self._equity) / self._peak_equity
        return {
            "equity":   self._equity,
            "position": self._pos,
            "drawdown": drawdown,
            "step":     self._step,
        }

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_bars(self) -> int:
        return self._n

    @property
    def equity_curve(self) -> list[float]:
        return self._equity_history.copy()

    def final_metrics(self) -> dict:
        """Compute episode-end performance metrics."""
        eq = np.array(self._equity_history)
        if len(eq) < 2:
            return {}
        rets = np.diff(eq) / eq[:-1]
        total_return = float(eq[-1] / eq[0] - 1.0)
        ann_factor   = 365 * 24  # hourly bars
        sharpe = 0.0
        if rets.std() > 1e-9:
            sharpe = float((rets.mean() / rets.std()) * np.sqrt(ann_factor))
        peak       = np.maximum.accumulate(eq)
        drawdowns  = (peak - eq) / peak
        max_dd     = float(drawdowns.max())
        calmar     = (total_return / max_dd) if max_dd > 1e-6 else 0.0
        return {
            "total_return":   round(total_return, 6),
            "sharpe_ratio":   round(sharpe,        4),
            "max_drawdown":   round(max_dd,         4),
            "calmar_ratio":   round(calmar,          4),
            "n_steps":        len(rets),
        }
