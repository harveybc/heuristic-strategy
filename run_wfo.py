#!/usr/bin/env python3
"""
Walk-Forward Optimization runner for heuristic-strategy.

Usage:
    python run_wfo.py [options]

Example:
    python run_wfo.py --population_size 30 --num_generations 20

Loads the full 15yr EURUSD dataset, runs anchored expanding-window WFO,
and reports only out-of-sample performance.
"""

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np

# Force line-buffered stdout so nohup logs flush per-line
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure app/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from app.data_handler import load_csv
from app.plugin_loader import load_plugin
from app.walk_forward_optimizer import run_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument("--base_dataset_file", type=str,
                        default="tests/data/eurusd_hour_2005_2020.csv",
                        help="Full OHLC dataset CSV")
    parser.add_argument("--plugin", type=str, default="regime_wfo",
                        help="Plugin name")
    parser.add_argument("--train_years", type=int, default=3,
                        help="Rolling training window size in years")
    parser.add_argument("--first_test_year", type=int, default=2009,
                        help="First test fold year")
    parser.add_argument("--last_test_year", type=int, default=2019,
                        help="Last test fold year")
    parser.add_argument("--population_size", type=int, default=30,
                        help="GA population per fold")
    parser.add_argument("--num_generations", type=int, default=20,
                        help="GA generations per fold")
    parser.add_argument("--min_trades", type=int, default=10,
                        help="Minimum trades for valid fitness in each fold")
    parser.add_argument("--save_results", type=str,
                        default="wfo_results.json",
                        help="Save WFO results to JSON")
    parser.add_argument("--forecast_evidence", type=str, default=None,
                        help="Declared forecast-vs-naive evidence (S09 gate)")
    parser.add_argument("--naive_gate_receipt", type=str, default="naive_gate_receipt_wfo.json")
    parser.add_argument("--save_trades", type=str,
                        default="wfo_oos_trades.csv",
                        help="Save OOS trades to CSV")
    args = parser.parse_args()

    # Suppress verbose inner strategy output
    os.environ["STRATEGY_QUIET"] = "1"

    # Naive gate at entry (S09): decided before any data is loaded or evaluated.
    from app.runner_naive_gate import RunnerGate, plugin_consumes_learned_predictions
    plugin_class, _ = load_plugin('heuristic_strategy.plugins', args.plugin)
    plugin = plugin_class()
    gate = RunnerGate("run_wfo", evidence_config=args.forecast_evidence,
                      receipt_path=args.naive_gate_receipt,
                      consumes_predictions=plugin_consumes_learned_predictions(plugin),
                      asset="EURUSD", prediction_source="API")
    if not gate.entry():
        print(f"SKIPPED_NOT_BETTER_THAN_NAIVE: walk-forward not run (receipt {args.naive_gate_receipt})")
        return

    # Load dataset
    print(f"Loading dataset: {args.base_dataset_file}")
    base_data = load_csv(args.base_dataset_file, headers=True)
    print(f"Loaded: {base_data.shape[0]} bars, "
          f"{base_data.index.min()} to {base_data.index.max()}")

    print(f"Plugin: {args.plugin} (gate entry: {gate.entry_decision['status']})")

    # Config stub (regime_wfo uses API mode — no CSV predictions needed)
    config = {
        "prediction_source": "API",
        "headers": True,
        "disable_multiprocessing": True,
    }

    # Run walk-forward optimization
    results = run_walk_forward(
        plugin=plugin,
        full_data=base_data,
        config=config,
        train_years=args.train_years,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        population_size=args.population_size,
        num_generations=args.num_generations,
        min_trades=args.min_trades,
        gate=gate,
    )

    # Save results
    if args.save_results:
        # Make JSON-serializable
        save_data = {
            "total_oos_profit": results["total_oos_profit"],
            "total_oos_trades": results["total_oos_trades"],
            "total_win_pct": results["total_win_pct"],
            "aggregate_sharpe": results["aggregate_sharpe"],
            "max_drawdown_usd": results["max_drawdown_usd"],
            "final_equity": results["final_equity"],
            "total_time_sec": results["total_time_sec"],
            "fold_results": [
                {k: v for k, v in f.items() if k != "equity_curve"}
                for f in results["fold_results"]
            ],
        }
        with open(args.save_results, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Results saved to {args.save_results}")

    if args.save_trades and results["all_oos_trades"]:
        pd.DataFrame(results["all_oos_trades"]).to_csv(
            args.save_trades, index=False
        )
        print(f"OOS trades saved to {args.save_trades}")

    # Print equity curve
    ec = results["equity_curve"]
    if len(ec) > 1:
        print(f"\nEquity curve: ${ec[0]:,.0f} → ${ec[-1]:,.0f} "
              f"(max DD: ${results['max_drawdown_usd']:,.0f})")


if __name__ == "__main__":
    main()
