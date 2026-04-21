#!/usr/bin/env python3
"""
Stage II-7.5 — RL Algorithm Pilot Matrix
==========================================
Tests learnability of RL policies on alpha configurations:
  - Run 3: BTC 1h, technical features
  - Run 11: ETH 1h, technical features

For each config × algorithm:
  - Algorithms: PPO, SAC, DQN
  - IS period train: 2020-01-01 → 2022-12-31 (~26K bars)
  - Val split: 2023-01-01 → 2023-12-31 (~8760 bars)
  - Test split: 2024-01-01 → 2024-12-31 (~8784 bars)
  - Training: 100,000 timesteps
  - Verdict: LEARNABLE if val Sharpe > 0.0 AND val return > 0.0

Outputs: deliverables/pilot_results_II7.json
         deliverables/TASK_II-7.5_ALGORITHM_PILOTS.md
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infrastructure.rl.trading_env import TradingEnv, DEFAULT_CONFIG

warnings.filterwarnings("ignore")

DATA_BINANCE = ROOT / "data" / "raw" / "binance"
DELIVERABLES = ROOT / "deliverables"

# ── Splits ────────────────────────────────────────────────────────────────────
TRAIN_START = "2020-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

ALGORITHMS  = ["PPO", "SAC", "DQN"]
TIMESTEPS   = 100_000

ALPHA_CONFIGS = [
    {"run_id": 3,  "label": "btc_1h_technical", "asset": "btc"},
    {"run_id": 11, "label": "eth_1h_technical", "asset": "eth"},
]

# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["returns"]      = close.pct_change()
    out["log_returns"]  = np.log(close / close.shift(1))

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_pos"] = (close - sma20) / (std20.replace(0, np.nan) * 2)

    vol_ma = vol.rolling(20).mean()
    out["volume_ratio"] = vol / vol_ma.replace(0, np.nan)

    ema9  = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema_cross"] = (ema9 - ema21) / close.replace(0, np.nan)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_norm"] = atr / close.replace(0, np.nan)

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).cumsum()
    out["obv_delta"] = obv.diff()

    out["momentum_5"]  = close / close.shift(5)  - 1
    out["momentum_20"] = close / close.shift(20) - 1
    out["volatility_20"] = out["log_returns"].rolling(20).std()

    return out


def load_and_slice(asset: str, start: str, end: str):
    if asset == "btc":
        fp = DATA_BINANCE / "btcusdt_1h_2019_2025.parquet"
    else:
        fp = DATA_BINANCE / "ethusdt_1h_2019_2025.parquet"
    df = pd.read_parquet(fp)
    if "DateTime" in df.columns:
        df = df.set_index("DateTime")
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    df_feat = compute_features(df)
    df_feat.dropna(inplace=True)
    aligned = df.loc[df_feat.index]

    mask = (df_feat.index >= pd.Timestamp(start, tz="UTC")) & \
           (df_feat.index <= pd.Timestamp(end,   tz="UTC"))
    return df_feat.loc[mask], aligned["Close"].loc[mask]


# ── Rollout evaluation ────────────────────────────────────────────────────────

def evaluate_policy(model, env: TradingEnv, n_eval_episodes: int = 1) -> dict:
    all_returns = []
    all_sharpes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        m = env.final_metrics()
        all_returns.append(m.get("total_return", 0.0))
        all_sharpes.append(m.get("sharpe_ratio",  0.0))

    return {
        "mean_total_return": round(float(np.mean(all_returns)), 6),
        "mean_sharpe":       round(float(np.mean(all_sharpes)), 4),
        "n_episodes":        n_eval_episodes,
    }


# ── Pilot run ─────────────────────────────────────────────────────────────────

def make_env(asset, start, end, action_type="discrete"):
    feat, prices = load_and_slice(asset, start, end)
    cfg = {**DEFAULT_CONFIG, "action_space_type": action_type}
    return TradingEnv(feat, prices, cfg)


def run_pilot(asset: str, label: str, run_id: int, algo_name: str) -> dict:
    from stable_baselines3 import PPO, SAC, DQN

    algo_map = {"PPO": (PPO, "discrete"), "SAC": (SAC, "continuous"), "DQN": (DQN, "discrete")}
    AlgoClass, action_type = algo_map[algo_name]

    print(f"  [{algo_name}] building envs ...")
    train_env = make_env(asset, TRAIN_START, TRAIN_END, action_type)
    val_env   = make_env(asset, VAL_START,   VAL_END,   action_type)
    test_env  = make_env(asset, TEST_START,  TEST_END,  action_type)

    n_train = train_env.n_bars
    n_val   = val_env.n_bars
    n_test  = test_env.n_bars

    print(f"  [{algo_name}] train_bars={n_train}, val_bars={n_val}, test_bars={n_test}")

    # Build model with conservative hyperparams for learnability check
    model_kwargs = {}
    if algo_name in ("PPO",):
        model_kwargs = {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 64}
    elif algo_name == "SAC":
        model_kwargs = {"learning_rate": 3e-4, "batch_size": 256, "learning_starts": 1000}
    elif algo_name == "DQN":
        model_kwargs = {"learning_rate": 1e-4, "batch_size": 64, "learning_starts": 1000,
                        "exploration_fraction": 0.3}

    try:
        model = AlgoClass("MlpPolicy", train_env, verbose=0, **model_kwargs)
        print(f"  [{algo_name}] training {TIMESTEPS:,} timesteps ...")
        model.learn(total_timesteps=TIMESTEPS)

        print(f"  [{algo_name}] evaluating ...")
        val_metrics  = evaluate_policy(model, val_env)
        test_metrics = evaluate_policy(model, test_env)

        learnable = (val_metrics["mean_total_return"] > 0.0 and
                     val_metrics["mean_sharpe"] > 0.0)
        verdict   = "LEARNABLE" if learnable else "NOT_LEARNABLE"

        print(f"  [{algo_name}] val_ret={val_metrics['mean_total_return']:.4f}, "
              f"val_sharpe={val_metrics['mean_sharpe']:.3f} → {verdict}")

        return {
            "algo":        algo_name,
            "run_id":      run_id,
            "label":       label,
            "n_timesteps": TIMESTEPS,
            "n_train_bars": n_train,
            "n_val_bars":   n_val,
            "n_test_bars":  n_test,
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "verdict":      verdict,
            "error":        None,
        }

    except Exception as exc:
        print(f"  [{algo_name}] ERROR: {exc}")
        return {
            "algo": algo_name, "run_id": run_id, "label": label,
            "verdict": "ERROR", "error": str(exc),
            "val_metrics": {}, "test_metrics": {},
        }


# ── Markdown writer ────────────────────────────────────────────────────────────

def write_markdown(all_results: list[dict], out_path: Path):
    lines = [
        "# TASK II-7.5 — Algorithm Pilots",
        "",
        "**Stage**: II-7 (RL Configuration Reconnaissance)",
        f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "**Status**: COMPLETE",
        "",
        "---",
        "",
        "## Setup",
        "",
        f"- Configs: BTC 1h (Run 3), ETH 1h (Run 11) — both alpha/PROCEED_TO_RL from II-7.3",
        f"- Algorithms: {', '.join(ALGORITHMS)}",
        f"- Timesteps: {TIMESTEPS:,}",
        f"- Train: {TRAIN_START} → {TRAIN_END}",
        f"- Val: {VAL_START} → {VAL_END}",
        f"- Test: {TEST_START} → {TEST_END}",
        f"- Reward: log_return, transaction_cost=0.1%",
        f"- Pass threshold: val_return > 0 AND val_sharpe > 0",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Config | Algorithm | Val Return | Val Sharpe | Test Return | Test Sharpe | Verdict |",
        "|--------|-----------|-----------|-----------|------------|------------|---------|",
    ]

    for r in all_results:
        vm = r.get("val_metrics",  {})
        tm = r.get("test_metrics", {})
        vret  = vm.get("mean_total_return", float("nan"))
        vshar = vm.get("mean_sharpe",       float("nan"))
        tret  = tm.get("mean_total_return", float("nan"))
        tshar = tm.get("mean_sharpe",       float("nan"))
        lines.append(
            f"| {r['label']} | {r['algo']} | {vret:.4f} | {vshar:.3f} | {tret:.4f} | {tshar:.3f} | {r['verdict']} |"
        )

    n_learnable = sum(1 for r in all_results if r["verdict"] == "LEARNABLE")
    n_total     = len(all_results)

    lines += [
        "",
        f"**Learnable configurations: {n_learnable} / {n_total}**",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "The pilot matrix tests learnability only — 100K timesteps is insufficient for a production policy. "
        "A positive val_return and val_sharpe after 100K steps indicates the RL signal is strong enough "
        "to learn directional bias, which justifies full-scale training in Part III.",
        "",
        "Algorithms not achieving LEARNABLE status in this pilot are not necessarily unsuitable — "
        "they may require more timesteps, tuned hyperparameters, or reward shaping.",
        "",
        "## Deliverables",
        "",
        "- `deliverables/pilot_results_II7.json` — Full pilot metrics",
        "- `deliverables/TASK_II-7.5_ALGORITHM_PILOTS.md` — This document",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage II-7.5 RL Algorithm Pilots")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--algos", default=",".join(ALGORITHMS))
    parser.add_argument("--output_json", default=str(DELIVERABLES / "pilot_results_II7.json"))
    parser.add_argument("--output_md",   default=str(DELIVERABLES / "TASK_II-7.5_ALGORITHM_PILOTS.md"))
    args = parser.parse_args()

    global TIMESTEPS
    TIMESTEPS = args.timesteps
    algos = [a.strip() for a in args.algos.split(",")]

    print("=" * 60)
    print("STAGE II-7.5 — ALGORITHM PILOT MATRIX")
    print("=" * 60)

    all_results = []
    for cfg in ALPHA_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {cfg['label']} (Run {cfg['run_id']})")
        print(f"{'='*60}")
        for algo in algos:
            result = run_pilot(cfg["asset"], cfg["label"], cfg["run_id"], algo)
            all_results.append(result)

    # Save JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"results": all_results, "timestamp": datetime.utcnow().isoformat()},
                  f, indent=2, default=str)
    print(f"\nSaved JSON: {out_json}")

    write_markdown(all_results, Path(args.output_md))

    print("\n" + "=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['label']:25s} {r['algo']:5s} → {r['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
