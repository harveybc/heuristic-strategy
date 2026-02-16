#!/usr/bin/env python3
"""Cross-sensitivity sweep: vary hourly and daily noise independently.

Tests how strategy profit depends on prediction quality of each timeframe
separately. Runs a grid of (hourly_noise, daily_noise) combinations.

Results saved to CSV + SQLite OLAP cube.

Usage:
    cd heuristic-strategy
    STRATEGY_QUIET=1 PYTHONPATH=./ python examples/scripts/cross_sensitivity_sweep.py
"""

import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Quiet mode
os.environ["STRATEGY_QUIET"] = "1"

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import backtrader as bt

# Import strategy
sys.path.insert(0, str(REPO_ROOT))
from app.heuristic_strategy import HeuristicStrategy


def load_base_data(filepath: str) -> pd.DataFrame:
    """Load base OHLC dataset."""
    df = pd.read_csv(filepath)
    if "DATE_TIME" in df.columns:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        df.set_index("DATE_TIME", inplace=True)
    return df


def create_ideal_hourly_predictions(base_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create ideal (perfect foresight) hourly predictions."""
    blocks = []
    close = base_df["CLOSE"] if "CLOSE" in base_df.columns else base_df.iloc[:, -1]
    for i in range(len(close) - horizon):
        block = close.iloc[i + 1: i + 1 + horizon].values.flatten()
        blocks.append(block)
    cols = [f"Prediction_h_{j+1}" for j in range(horizon)]
    return pd.DataFrame(blocks, index=close.index[:-horizon], columns=cols)


def create_ideal_daily_predictions(base_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create ideal (perfect foresight) daily predictions (24h spacing)."""
    close = base_df["CLOSE"] if "CLOSE" in base_df.columns else base_df.iloc[:, -1]
    nrows = len(close)
    required = horizon * 24
    if nrows < required:
        return pd.DataFrame()
    blocks = []
    for i in range(nrows - required):
        block = []
        for d in range(1, horizon + 1):
            idx = i + d * 24
            if idx < nrows:
                block.append(close.iloc[idx])
        if block:
            blocks.append(block)
    cols = [f"Prediction_d_{j+1}" for j in range(len(blocks[0]))]
    return pd.DataFrame(blocks, index=close.index[:len(blocks)], columns=cols)


def add_noise(df: pd.DataFrame, std: float) -> pd.DataFrame:
    """Add zero-mean Gaussian noise with given std to predictions."""
    if df is None or df.empty or std == 0.0:
        return df.copy() if df is not None else df
    noise = np.random.normal(loc=0.0, scale=std, size=df.shape)
    noisy = df.copy()
    noisy[:] = noisy.values + noise
    return noisy


def run_single_backtest(base_data, hourly_preds, daily_preds, params: dict) -> dict:
    """Run backtest using backtrader with HeuristicStrategy."""
    # Merge predictions into single CSV for the strategy
    merged = pd.DataFrame()
    if hourly_preds is not None and not hourly_preds.empty:
        for col in hourly_preds.columns:
            merged[col] = hourly_preds[col]
    if daily_preds is not None and not daily_preds.empty:
        for col in daily_preds.columns:
            merged[col] = daily_preds[col]

    if merged.empty:
        return {"profit": 0, "num_trades": 0, "win_pct": 0, "max_dd": 0}

    merged.index.name = "DATE_TIME"
    temp_file = "/tmp/cross_sens_preds.csv"
    merged.reset_index().to_csv(temp_file, index=False)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        HeuristicStrategy,
        pred_file=temp_file,
        pip_cost=params.get("pip_cost", 0.00001),
        rel_volume=params.get("rel_volume", 0.02),
        min_order_volume=params.get("min_order_volume", 10000),
        max_order_volume=params.get("max_order_volume", 1000000),
        leverage=params.get("leverage", 1000),
        profit_threshold=params.get("profit_threshold", 5),
        min_drawdown_pips=params.get("min_drawdown_pips", 10),
        tp_multiplier=params.get("tp_multiplier", 0.9),
        sl_multiplier=params.get("sl_multiplier", 2.0),
        lower_rr_threshold=params.get("lower_rr_threshold", 0.5),
        upper_rr_threshold=params.get("upper_rr_threshold", 2.0),
        max_trades_per_5days=params.get("max_trades_per_5days", 20),
    )

    data_feed = bt.feeds.PandasData(dataname=base_data)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(10000.0)

    try:
        # Suppress matplotlib
        import matplotlib
        matplotlib.use("Agg")
        runresult = cerebro.run()
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return {"profit": 0, "num_trades": 0, "win_pct": 0, "max_dd": 0, "error": str(e)}

    final_value = cerebro.broker.getvalue()
    profit = final_value - 10000.0

    strat = runresult[0]
    trades = getattr(strat, "trades", [])
    num_trades = len(trades)
    win_pct = (sum(1 for t in trades if t["pnl"] > 0) / num_trades * 100) if num_trades > 0 else 0
    max_dd = max((t["max_dd"] for t in trades), default=0)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return {"profit": profit, "num_trades": num_trades, "win_pct": win_pct, "max_dd": max_dd}


def main():
    base_file = str(REPO_ROOT / "examples/data/phase_1/phase_1_base_d3.csv")
    params_file = str(REPO_ROOT / "examples/results/phase_2_1_daily/phase_2_1_ann_25200_1d_results_high_frequency_parameters.json")
    output_dir = REPO_ROOT / "examples/results/cross_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_horizon = 6  # Same as config

    # Use default strategy params (matching noise sensitivity experiments)
    strategy_params = {
        "pip_cost": 0.00001,
        "rel_volume": 0.02,
        "min_order_volume": 10000,
        "max_order_volume": 1000000,
        "leverage": 1000,
        "profit_threshold": 5,
        "min_drawdown_pips": 10,
        "tp_multiplier": 0.9,
        "sl_multiplier": 2.0,
        "lower_rr_threshold": 0.5,
        "upper_rr_threshold": 2.0,
        "max_trades_per_5days": 20,
    }

    # Load base data
    print("Loading base data...")
    base_data = load_base_data(base_file)

    # Generate ideal predictions
    print("Generating ideal predictions...")
    hourly_ideal = create_ideal_hourly_predictions(base_data, time_horizon)
    daily_ideal = create_ideal_daily_predictions(base_data, time_horizon)

    # Align
    common = base_data.index.intersection(hourly_ideal.index).intersection(daily_ideal.index)
    base_aligned = base_data.loc[common]
    hourly_ideal = hourly_ideal.loc[common]
    daily_ideal = daily_ideal.loc[common]
    print(f"Aligned: {len(common)} rows, hourly={hourly_ideal.shape[1]}h, daily={daily_ideal.shape[1]}d")

    # Noise levels
    noise_levels = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01]
    grid = list(product(noise_levels, noise_levels))
    print(f"Grid: {len(noise_levels)}×{len(noise_levels)} = {len(grid)} combinations")

    results = []
    t0 = time.time()
    for i, (h_noise, d_noise) in enumerate(grid):
        np.random.seed(42)  # Reproducible

        hourly_noisy = add_noise(hourly_ideal, h_noise)
        daily_noisy = add_noise(daily_ideal, d_noise)

        try:
            bt_result = run_single_backtest(base_aligned, hourly_noisy, daily_noisy, strategy_params)
        except Exception as e:
            print(f"  [{i+1}/{len(grid)}] h={h_noise:.4f} d={d_noise:.4f} ERROR: {e}")
            continue

        row = {
            "hourly_noise": h_noise,
            "daily_noise": d_noise,
            "profit": round(bt_result["profit"], 2),
            "num_trades": bt_result["num_trades"],
            "win_pct": round(bt_result["win_pct"], 1),
            "max_dd": round(bt_result["max_dd"], 2),
        }
        results.append(row)

        tag = "✅" if bt_result["profit"] > 0 else "❌"
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(grid) - i - 1)
        print(f"  [{i+1}/{len(grid)}] h={h_noise:.4f} d={d_noise:.4f} → ${bt_result['profit']:>10,.0f} trades={bt_result['num_trades']:>4} win={bt_result['win_pct']:.0f}% {tag}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    # Save CSV
    csv_path = output_dir / "cross_sensitivity_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to {csv_path}")

    # Save SQLite OLAP
    try:
        import sqlite3
        db_path = output_dir / "cross_sensitivity_olap.db"
        conn = sqlite3.connect(str(db_path))
        df = pd.DataFrame(results)
        df.to_sql("cross_sensitivity", conn, if_exists="replace", index=False)
        conn.execute("CREATE VIEW IF NOT EXISTS daily_marginal AS SELECT daily_noise, AVG(profit) as avg_profit, AVG(win_pct) as avg_win_pct, COUNT(*) as n FROM cross_sensitivity GROUP BY daily_noise ORDER BY daily_noise")
        conn.execute("CREATE VIEW IF NOT EXISTS hourly_marginal AS SELECT hourly_noise, AVG(profit) as avg_profit, AVG(win_pct) as avg_win_pct, COUNT(*) as n FROM cross_sensitivity GROUP BY hourly_noise ORDER BY hourly_noise")
        conn.commit()
        conn.close()
        print(f"OLAP cube: {db_path}")
    except Exception as e:
        print(f"SQLite error: {e}")

    # Summary
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*70}")
        print("DAILY noise marginal effect (avg over all hourly levels):")
        for dn, grp in df.groupby("daily_noise"):
            print(f"  d={dn:.4f}: avg_profit=${grp['profit'].mean():>10,.0f}  win={grp['win_pct'].mean():.0f}%  trades={grp['num_trades'].mean():.0f}")
        print(f"\nHOURLY noise marginal effect (avg over all daily levels):")
        for hn, grp in df.groupby("hourly_noise"):
            print(f"  h={hn:.4f}: avg_profit=${grp['profit'].mean():>10,.0f}  win={grp['win_pct'].mean():.0f}%  trades={grp['num_trades'].mean():.0f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
