#!/usr/bin/env python3
"""
Benchmark strategies for Stage II-2 baseline validation.

Computes buy-and-hold, random-entry, and flat (zero) benchmarks
across window manifest for F-10 §3 comparisons.

Output: benchmarks.csv with per-window and aggregate metrics.
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime


def _compute_sharpe(pnl_array, annualization=1.0):
    """Sharpe from PnL array."""
    if len(pnl_array) < 2:
        return 0.0
    mean_pnl = np.mean(pnl_array)
    std_pnl = np.std(pnl_array, ddof=1)
    if std_pnl == 0:
        return 0.0
    return (mean_pnl / std_pnl) * np.sqrt(annualization)


def _compute_max_drawdown(equity_curve):
    """Max drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _slice_window(df, start_str, end_str):
    """Slice DataFrame by date range."""
    mask = (df.index >= pd.Timestamp(start_str)) & (df.index <= pd.Timestamp(end_str))
    return df.loc[mask].copy()


def _load_data(path):
    """Load OHLCV CSV with datetime index."""
    df = pd.read_csv(path)
    date_col = None
    for candidate in ['DATE_TIME', 'Date', 'date', 'datetime', 'Datetime', 'DATE']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    return df


def buy_and_hold(data, initial_cash=10000.0, leverage=100):
    """
    Simple buy-and-hold benchmark.
    Enter long at first close, exit at last close.
    Returns dict with metrics.
    """
    if len(data) < 2:
        return {"sharpe": 0, "max_dd": 0, "total_return": 0, "num_trades": 1, "final_equity": initial_cash}

    closes = data['Close'].values
    # Position size: use leverage on initial cash
    position_value = initial_cash * leverage
    units = position_value / closes[0]

    # Per-bar PnL
    bar_returns = np.diff(closes)
    bar_pnl = bar_returns * units

    # Equity curve
    equity = np.zeros(len(closes))
    equity[0] = initial_cash
    for i in range(len(bar_pnl)):
        equity[i + 1] = equity[i] + bar_pnl[i]

    total_return = (equity[-1] - initial_cash) / initial_cash
    sharpe = _compute_sharpe(bar_pnl)
    max_dd = _compute_max_drawdown(equity)

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "num_trades": 1,
        "final_equity": equity[-1],
    }


def random_entry(data, num_simulations=100, avg_trade_freq=7,
                 initial_cash=10000.0, leverage=100, pip_cost=0.00001,
                 spread_pips=15.0, seed=42):
    """
    Random entry benchmark: same trade frequency as strategy, random direction.
    Returns distribution statistics over num_simulations.
    """
    if len(data) < 10:
        return {
            "sharpe_mean": 0, "sharpe_std": 0,
            "max_dd_mean": 0, "total_return_mean": 0,
            "num_trades_mean": 0, "final_equity_mean": initial_cash,
        }

    rng = np.random.RandomState(seed)
    closes = data['Close'].values
    n_bars = len(closes)

    # Expected number of trades (approx trade every avg_trade_freq bars)
    n_trades = max(1, n_bars // avg_trade_freq)
    spread_cost = spread_pips * pip_cost

    sharpes = []
    max_dds = []
    returns_list = []

    for sim in range(num_simulations):
        # Random entry points (no overlap)
        entry_bars = sorted(rng.choice(n_bars - 1, size=min(n_trades, n_bars - 1), replace=False))
        trade_pnls = []

        for bar_idx in entry_bars:
            direction = rng.choice([-1, 1])
            # Hold for random 1-20 bars
            hold = rng.randint(1, min(21, n_bars - bar_idx))
            exit_bar = bar_idx + hold

            entry_price = closes[bar_idx]
            exit_price = closes[exit_bar]
            position_value = initial_cash * 0.1 * leverage  # 10% of capital
            units = position_value / entry_price

            pnl = direction * (exit_price - entry_price) * units
            pnl -= spread_cost * units * 2  # entry + exit spread
            trade_pnls.append(pnl)

        pnl_arr = np.array(trade_pnls) if trade_pnls else np.array([0.0])
        sharpes.append(_compute_sharpe(pnl_arr))

        equity = [initial_cash]
        for p in pnl_arr:
            equity.append(equity[-1] + p)
        max_dds.append(_compute_max_drawdown(np.array(equity)))
        returns_list.append((equity[-1] - initial_cash) / initial_cash)

    return {
        "sharpe_mean": np.mean(sharpes),
        "sharpe_std": np.std(sharpes),
        "sharpe_p5": np.percentile(sharpes, 5),
        "sharpe_p95": np.percentile(sharpes, 95),
        "max_dd_mean": np.mean(max_dds),
        "total_return_mean": np.mean(returns_list),
        "num_trades_mean": n_trades,
        "final_equity_mean": initial_cash * (1 + np.mean(returns_list)),
        "num_simulations": num_simulations,
    }


def flat_benchmark(initial_cash=10000.0):
    """Zero (flat) benchmark — always out of market."""
    return {
        "sharpe": 0.0,
        "max_dd": 0.0,
        "total_return": 0.0,
        "num_trades": 0,
        "final_equity": initial_cash,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark strategies for Stage II-2")
    parser.add_argument("--manifest", required=True, help="Window manifest JSON")
    parser.add_argument("--data", required=True, help="OHLCV CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--embargo_bars", type=int, default=6)
    parser.add_argument("--random_sims", type=int, default=100)
    args = parser.parse_args()

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    data = _load_data(args.data)
    print(f"Loaded {len(data)} bars from {args.data}")

    windows = manifest["windows"]
    results = []

    for wdef in windows:
        wid = wdef["id"]
        test_data = _slice_window(data, wdef["test_start"], wdef["test_end"])

        if len(test_data) < 2:
            print(f"  Window {wid}: insufficient test data ({len(test_data)} bars), skipping")
            continue

        print(f"  Window {wid}: test {wdef['test_start']} → {wdef['test_end']} ({len(test_data)} bars)")

        # Buy-and-hold
        bh = buy_and_hold(test_data)
        results.append({
            "window_id": wid, "benchmark": "buy_and_hold",
            "test_start": wdef["test_start"], "test_end": wdef["test_end"],
            "sharpe": bh["sharpe"], "max_dd": bh["max_dd"],
            "total_return": bh["total_return"], "num_trades": bh["num_trades"],
            "final_equity": bh["final_equity"],
        })

        # Random entry
        re = random_entry(test_data, num_simulations=args.random_sims)
        results.append({
            "window_id": wid, "benchmark": "random_entry",
            "test_start": wdef["test_start"], "test_end": wdef["test_end"],
            "sharpe": re["sharpe_mean"], "max_dd": re["max_dd_mean"],
            "total_return": re["total_return_mean"],
            "num_trades": re["num_trades_mean"],
            "final_equity": re["final_equity_mean"],
            "sharpe_std": re.get("sharpe_std", 0),
            "sharpe_p5": re.get("sharpe_p5", 0),
            "sharpe_p95": re.get("sharpe_p95", 0),
        })

        # Flat
        fl = flat_benchmark()
        results.append({
            "window_id": wid, "benchmark": "flat",
            "test_start": wdef["test_start"], "test_end": wdef["test_end"],
            "sharpe": fl["sharpe"], "max_dd": fl["max_dd"],
            "total_return": fl["total_return"], "num_trades": fl["num_trades"],
            "final_equity": fl["final_equity"],
        })

    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nBenchmarks saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for bm in ["buy_and_hold", "random_entry", "flat"]:
        subset = df[df["benchmark"] == bm]
        if len(subset) == 0:
            continue
        mean_sr = subset["sharpe"].mean()
        mean_dd = subset["max_dd"].mean()
        mean_ret = subset["total_return"].mean()
        print(f"\n  {bm}:")
        print(f"    Mean Sharpe:     {mean_sr:.4f}")
        print(f"    Mean Max DD:     {mean_dd:.4f}")
        print(f"    Mean Return:     {mean_ret:.4f}")
        if bm == "random_entry" and "sharpe_std" in subset.columns:
            mean_sr_std = subset["sharpe_std"].mean()
            print(f"    Mean Sharpe SD:  {mean_sr_std:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
