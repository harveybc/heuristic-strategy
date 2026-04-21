#!/usr/bin/env python3
"""
Compute simple baselines for BTC/USD 4h: buy-and-hold, random, zero.

Reads a window manifest and BTC 4h data, evaluates each test window with:
  1. Buy-and-Hold: long from test_start to test_end
  2. Random: 50/50 coin-flip long/short each bar
  3. Zero: no trades (sanity check: Sharpe=0)

Outputs JSON + summary to deliverables/.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def compute_log_returns(prices):
    """Compute log returns from price series."""
    return np.diff(np.log(prices))


def sharpe_from_returns(returns, periods_per_year=2190):
    """Annualized Sharpe ratio. 4h bars → 6*365=2190 per year."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown_from_returns(returns):
    """Max drawdown from return series."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def run_baselines(manifest_path, data_path, output_path):
    # Load data
    df = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    results = {
        "experiment": "baselines",
        "asset": manifest.get("asset", "btcusd_4h"),
        "trigger": manifest.get("trigger", "yearly"),
        "generated": datetime.now().isoformat(),
        "windows": []
    }

    all_bh_sharpes = []
    all_random_sharpes = []

    for window in manifest["windows"]:
        wid = window["id"]
        test_data = df[window["test_start"]:window["test_end"]]

        if len(test_data) < 10:
            continue

        close = test_data['close'].values
        log_rets = compute_log_returns(close)

        # Buy-and-hold
        bh_sharpe = sharpe_from_returns(log_rets)
        bh_dd = max_drawdown_from_returns(log_rets)
        bh_return = float(close[-1] / close[0] - 1)

        # Random (average of 100 Monte Carlo trials)
        np.random.seed(42 + wid)
        random_sharpes = []
        for _ in range(100):
            signs = np.random.choice([-1, 1], size=len(log_rets))
            random_rets = log_rets * signs
            random_sharpes.append(sharpe_from_returns(random_rets))
        random_mean_sharpe = float(np.mean(random_sharpes))
        random_std_sharpe = float(np.std(random_sharpes))

        # Zero (no position)
        zero_sharpe = 0.0

        w_result = {
            "window_id": wid,
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "test_bars": len(test_data),
            "buy_and_hold": {
                "sharpe": round(bh_sharpe, 4),
                "max_dd": round(bh_dd, 4),
                "return_pct": round(bh_return * 100, 2),
            },
            "random": {
                "mean_sharpe": round(random_mean_sharpe, 4),
                "std_sharpe": round(random_std_sharpe, 4),
            },
            "zero": {
                "sharpe": zero_sharpe,
            },
        }
        results["windows"].append(w_result)
        all_bh_sharpes.append(bh_sharpe)
        all_random_sharpes.append(random_mean_sharpe)

        print(f"Window {wid}: BH Sharpe={bh_sharpe:.4f} (ret={bh_return*100:.1f}%), "
              f"Random Sharpe={random_mean_sharpe:.4f}±{random_std_sharpe:.4f}")

    # Aggregates
    results["summary"] = {
        "mean_bh_sharpe": round(float(np.mean(all_bh_sharpes)), 4) if all_bh_sharpes else 0,
        "mean_random_sharpe": round(float(np.mean(all_random_sharpes)), 4) if all_random_sharpes else 0,
        "zero_sharpe": 0.0,
        "num_windows": len(results["windows"]),
    }

    print(f"\nSummary: BH mean Sharpe={results['summary']['mean_bh_sharpe']:.4f}, "
          f"Random mean Sharpe={results['summary']['mean_random_sharpe']:.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute baselines")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_baselines(args.manifest, args.data, args.output)


if __name__ == "__main__":
    main()
