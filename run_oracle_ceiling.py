#!/usr/bin/env python3
"""
Oracle Ceiling Test for direction_atr strategy.

Runs the direction_ideal_oracle (perfect future knowledge) through the
direction_atr strategy on the full 15yr EURUSD dataset, fold by fold.

This establishes the THEORETICAL MAXIMUM profit possible with realistic
trading costs. If the perfect oracle can't profit, no ML model will.

Then sweeps noise levels to find the minimum accuracy needed.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))
pp_root = os.path.join(os.path.dirname(__file__), '..', 'prediction_provider')
sys.path.insert(0, pp_root)

from app.data_handler import load_csv


class OfflineOraclePredictionSource:
    """
    Wraps DirectionIdealOracle to act like ApiPredictionSource,
    but calls the oracle directly (no HTTP server needed).
    """

    def __init__(self, oracle):
        self.oracle = oracle

    def get_entry_prediction(self, dt_hour, tp_pips=0, sl_pips=0,
                             spread_pips=0, commission_per_lot=0,
                             slippage_pips=0):
        try:
            result = self.oracle.predict_entry(
                dt_hour,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                spread_pips=spread_pips,
                commission_per_lot=commission_per_lot,
                slippage_pips=slippage_pips,
            )
            result["available"] = True
            return result
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_exit_prediction(self, dt_hour, direction="buy",
                            tp_price=0, sl_price=0):
        try:
            result = self.oracle.predict_exit(
                dt_hour,
                direction=direction,
                tp_price=tp_price,
                sl_price=sl_price,
            )
            result["available"] = True
            return result
        except Exception as e:
            return {"available": False, "error": str(e)}


def run_oracle_backtest(base_data, oracle, config, label=""):
    """Run direction_atr strategy with offline oracle on given data slice."""
    import backtrader as bt
    from app.plugins.plugin_direction_atr import Plugin as DirectionATRPlugin

    plugin = DirectionATRPlugin()

    # Monkey-patch the strategy class to use offline oracle instead of HTTP
    original_init = plugin.DirectionATRStrategy.__init__

    offline_source = OfflineOraclePredictionSource(oracle)

    def patched_init(self_strat, *args, **kwargs):
        # Skip ApiPredictionSource creation
        bt.Strategy.__init__(self_strat)
        self_strat._pred_source = offline_source
        self_strat.data0 = self_strat.datas[0]
        self_strat.initial_balance = self_strat.broker.getvalue()
        self_strat.trade_entry_dates = []
        self_strat.balance_history = []
        self_strat.date_history = []
        self_strat.trade_low = None
        self_strat.trade_high = None
        self_strat.trades = []
        self_strat.current_tp = None
        self_strat.current_sl = None
        self_strat.current_direction = None
        self_strat.order_direction = None
        self_strat.order_entry_price = None
        self_strat.trade_entry_bar = None
        self_strat.current_volume = None
        self_strat.atr = bt.indicators.ATR(
            self_strat.data0, period=self_strat.p.atr_period
        )

    plugin.DirectionATRStrategy.__init__ = patched_init

    # Run with default ATR params (no optimization — just oracle signal quality)
    candidate = [
        config.get("atr_period", 14),
        config.get("atr_tp_multiplier", 2.0),
        config.get("atr_sl_multiplier", 1.5),
    ]
    profit, stats = plugin.evaluate_candidate(
        candidate, base_data, None, None, config
    )
    trades = plugin.trades

    # Restore original init
    plugin.DirectionATRStrategy.__init__ = original_init

    return profit, stats, trades


def main():
    os.environ["STRATEGY_QUIET"] = "1"
    os.environ["PREDICTION_PROVIDER_QUIET"] = "1"

    # Import oracle
    from plugins_predictor.direction_ideal_oracle import DirectionIdealOracle

    data_file = "tests/data/eurusd_hour_2005_2020.csv"
    print(f"Loading dataset: {data_file}")
    base_data = load_csv(data_file, headers=True)
    print(f"Loaded: {base_data.shape[0]} bars, "
          f"{base_data.index.min()} to {base_data.index.max()}")

    # Strategy config (realistic costs)
    config = {
        "prediction_source": "API",
        "pp_api_url": "http://offline",
        "pp_timeout": 999,
        "headers": True,
        "disable_multiprocessing": True,
        "atr_period": 14,
        "atr_tp_multiplier": 2.0,
        "atr_sl_multiplier": 1.5,
        "spread_pips": 15.0,
        "commission_per_lot": 7.0,
        "slippage_pips": 5.0,
    }

    # ================================================================
    # PHASE A1: Oracle ceiling per year fold
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE A1: ORACLE CEILING TEST (Perfect Knowledge)")
    print(f"{'='*70}")
    print(f"ATR period={config['atr_period']}, "
          f"TP mult={config['atr_tp_multiplier']}, "
          f"SL mult={config['atr_sl_multiplier']}")
    print(f"Costs: spread={config['spread_pips']}pips, "
          f"comm={config['commission_per_lot']}/lot, "
          f"slip={config['slippage_pips']}pips")
    print(f"{'='*70}\n")

    fold_results = []
    all_trades = []

    for test_year in range(2006, 2020):
        year_mask = base_data.index.year == test_year
        year_data = base_data[year_mask].copy()
        if len(year_data) < 100:
            continue

        t0 = time.time()

        # Create oracle loaded with FULL data (needs future lookahead)
        oracle = DirectionIdealOracle({
            "csv_file": None,  # We'll load manually
            "atr_period": config["atr_period"],
            "noise_std": 0.0,
            "noise_seed": 42,
            "pip_cost": 0.00001,
            "prediction_horizon": 120,
            "friday_close_hour": 20,
        })
        # Load full data for oracle (needs ALL bars including future)
        oracle._data = base_data.copy()
        oracle._compute_atr()
        oracle._loaded = True
        oracle._rng = np.random.default_rng(42)

        profit, stats, trades = run_oracle_backtest(
            year_data, oracle, config, label=f"Y{test_year}"
        )
        elapsed = time.time() - t0

        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
        pnls = [t['pnl'] for t in trades]
        sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) if n_trades > 1 else 0

        metrics = oracle.get_metrics()

        fold_results.append({
            "year": test_year,
            "profit": profit,
            "trades": n_trades,
            "win_pct": win_pct,
            "sharpe": sharpe,
            "oracle_accuracy": metrics["accuracy"],
            "oracle_f1": metrics["f1"],
            "elapsed": elapsed,
        })
        for t in trades:
            t["year"] = test_year
        all_trades.extend(trades)

        sign = '+' if profit >= 0 else ''
        print(f"  {test_year}: {sign}${profit:,.2f} | "
              f"{n_trades} trades | Win {win_pct:.0f}% | "
              f"Sharpe {sharpe:.2f} | "
              f"Oracle acc={metrics['accuracy']:.1%} F1={metrics['f1']:.3f} | "
              f"{elapsed:.0f}s")

        oracle.reset_metrics()

    # Aggregate
    total_profit = sum(f["profit"] for f in fold_results)
    total_trades = sum(f["trades"] for f in fold_results)
    total_wins = sum(1 for t in all_trades if t['pnl'] > 0)
    total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0
    all_pnls = [t['pnl'] for t in all_trades]
    agg_sharpe = (np.mean(all_pnls) / (np.std(all_pnls) + 1e-10)) if len(all_pnls) > 1 else 0
    profitable_folds = sum(1 for f in fold_results if f["profit"] > 0)

    # Equity curve
    equity = 10000.0
    peak = equity
    max_dd = 0
    for t in all_trades:
        equity += t['pnl']
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    print(f"\n{'='*70}")
    print(f"ORACLE CEILING RESULTS (PERFECT KNOWLEDGE)")
    print(f"{'='*70}")
    print(f"Total Profit:     ${total_profit:,.2f}")
    print(f"Total Trades:     {total_trades}")
    print(f"Win Rate:         {total_win_pct:.1f}%")
    print(f"Aggregate Sharpe: {agg_sharpe:.3f}")
    print(f"Max Drawdown:     ${max_dd:,.2f}")
    print(f"Final Equity:     ${equity:,.2f}")
    print(f"Profitable Years: {profitable_folds}/{len(fold_results)}")
    print(f"{'='*70}\n")

    # Save results
    with open("oracle_ceiling_results.json", "w") as f:
        json.dump({
            "phase": "A1_oracle_ceiling",
            "total_profit": total_profit,
            "total_trades": total_trades,
            "total_win_pct": total_win_pct,
            "aggregate_sharpe": agg_sharpe,
            "max_drawdown_usd": max_dd,
            "final_equity": equity,
            "profitable_folds": profitable_folds,
            "total_folds": len(fold_results),
            "fold_results": fold_results,
            "config": {k: v for k, v in config.items()
                       if isinstance(v, (int, float, str, bool))},
        }, f, indent=2, default=str)
    print("Saved oracle_ceiling_results.json")

    if all_trades:
        pd.DataFrame(all_trades).to_csv("oracle_ceiling_trades.csv", index=False)
        print("Saved oracle_ceiling_trades.csv")

    # ================================================================
    # PHASE A2: Noise sweep — find minimum accuracy needed
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE A2: NOISE SWEEP (Degradation from perfect)")
    print(f"{'='*70}")
    print(f"Testing noise levels: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0")
    print(f"Using 2012-2016 as representative sample (5 years)")
    print(f"{'='*70}\n")

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    noise_results = []

    # Use a 5-year representative sample for speed
    sample_mask = (base_data.index.year >= 2012) & (base_data.index.year <= 2016)
    sample_data = base_data[sample_mask].copy()

    for noise_std in noise_levels:
        oracle = DirectionIdealOracle({
            "csv_file": None,
            "atr_period": config["atr_period"],
            "noise_std": noise_std,
            "noise_seed": 42,
            "pip_cost": 0.00001,
            "prediction_horizon": 120,
            "friday_close_hour": 20,
        })
        oracle._data = base_data.copy()
        oracle._compute_atr()
        oracle._loaded = True
        oracle._rng = np.random.default_rng(42)

        t0 = time.time()
        profit, stats, trades = run_oracle_backtest(
            sample_data, oracle, config, label=f"noise={noise_std}"
        )
        elapsed = time.time() - t0

        metrics = oracle.get_metrics()
        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_pct = (wins / n_trades * 100) if n_trades > 0 else 0
        pnls = [t['pnl'] for t in trades]
        sharpe = (np.mean(pnls) / (np.std(pnls) + 1e-10)) if n_trades > 1 else 0

        row = {
            "noise_std": noise_std,
            "oracle_accuracy": metrics["accuracy"],
            "oracle_f1": metrics["f1"],
            "oracle_precision": metrics["precision"],
            "oracle_recall": metrics["recall"],
            "profit": profit,
            "trades": n_trades,
            "win_pct": win_pct,
            "sharpe": sharpe,
            "elapsed": elapsed,
        }
        noise_results.append(row)

        sign = '+' if profit >= 0 else ''
        print(f"  noise={noise_std:.1f}: {sign}${profit:,.2f} | "
              f"{n_trades} trades | Win {win_pct:.0f}% | Sharpe {sharpe:.2f} | "
              f"Acc={metrics['accuracy']:.1%} F1={metrics['f1']:.3f} | "
              f"{elapsed:.0f}s")

        oracle.reset_metrics()

    # Find breakeven accuracy
    print(f"\n{'='*70}")
    print(f"NOISE SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'Noise':>6} {'Accuracy':>10} {'F1':>8} {'Profit':>12} "
          f"{'Trades':>8} {'Win%':>7} {'Sharpe':>8}")
    print("-" * 65)
    for r in noise_results:
        sign = '+' if r['profit'] >= 0 else ''
        print(f"{r['noise_std']:>6.1f} {r['oracle_accuracy']:>9.1%} "
              f"{r['oracle_f1']:>7.3f} {sign}${r['profit']:>10,.2f} "
              f"{r['trades']:>8} {r['win_pct']:>6.1f}% {r['sharpe']:>7.2f}")

    # Find breakeven point
    profitable = [(r['oracle_accuracy'], r['noise_std'])
                  for r in noise_results if r['profit'] > 0]
    if profitable:
        min_acc = min(profitable, key=lambda x: x[0])
        print(f"\nMinimum accuracy for profit: ~{min_acc[0]:.1%} "
              f"(noise_std={min_acc[1]:.1f})")
    else:
        print(f"\nNo noise level was profitable — costs too high or "
              f"strategy structure cannot profit even with perfect oracle.")

    # Save noise sweep
    with open("oracle_noise_sweep.json", "w") as f:
        json.dump(noise_results, f, indent=2, default=str)
    print("Saved oracle_noise_sweep.json")


if __name__ == "__main__":
    main()
