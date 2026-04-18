#!/usr/bin/env python3
"""
Phase 4 — Track A: Deploy Oracle-Independent Mean Reversion

Task A.1: Parameter-perturbation audit for EUR/USD and USD/JPY daily MR
Task A.2: Build Backtrader-compatible plugin (standalone, no predictor dependency)
Task A.3: Risk-calibrated position sizing
Task A.5: Monitoring protocol definition

(Task A.4 — OANDA demo deployment — requires manual setup and is documented, not auto-run.)
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import itertools
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import download_asset_data
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe, max_drawdown
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# MR strategy parameters (from audit_noise_budget.py get_strategy_positions)
BASE_PARAMS = {
    "lookback": 20,
    "z_entry": 1.5,
    "z_exit": 0.5,
}

CELLS = [
    {"asset": "EUR/USD", "timeframe": "daily"},
    {"asset": "USD/JPY", "timeframe": "daily"},
]


def run_mr_strategy(log_ret, close, lookback=20, z_entry=1.5, z_exit=0.5,
                    sl_multiple=3.0, tp_multiple=2.0, max_holding=30):
    """
    Pure mean-reversion strategy — NO oracle signal.
    Returns positions array and trade log.
    """
    n = len(log_ret)
    positions = np.zeros(n)
    trades = []
    cum_ret = np.cumsum(log_ret)
    entry_bar = -999
    entry_price = 0.0

    for i in range(lookback, n):
        window = cum_ret[max(0, i - lookback):i + 1]
        std = np.std(window)
        if std < 1e-12:
            continue
        z = (cum_ret[i] - np.mean(window)) / std

        # Existing position logic
        if positions[i - 1] != 0:
            bars_held = i - entry_bar
            # Check SL/TP/max holding
            pnl_since_entry = positions[i - 1] * (cum_ret[i] - cum_ret[entry_bar])
            atr = np.std(log_ret[max(0, i - lookback):i]) if i >= lookback else 0.01

            if pnl_since_entry < -sl_multiple * atr:
                # Stop loss hit
                positions[i] = 0
                trades.append({"entry": entry_bar, "exit": i, "direction": int(positions[i - 1]),
                                "pnl": pnl_since_entry, "reason": "SL"})
            elif pnl_since_entry > tp_multiple * atr:
                # Take profit hit
                positions[i] = 0
                trades.append({"entry": entry_bar, "exit": i, "direction": int(positions[i - 1]),
                                "pnl": pnl_since_entry, "reason": "TP"})
            elif bars_held >= max_holding:
                # Max holding period
                positions[i] = 0
                trades.append({"entry": entry_bar, "exit": i, "direction": int(positions[i - 1]),
                                "pnl": pnl_since_entry, "reason": "max_hold"})
            elif abs(z) < z_exit:
                # Z-score mean-reverted
                positions[i] = 0
                trades.append({"entry": entry_bar, "exit": i, "direction": int(positions[i - 1]),
                                "pnl": pnl_since_entry, "reason": "z_exit"})
            else:
                positions[i] = positions[i - 1]
        else:
            # No position — check entry
            if z > z_entry:
                positions[i] = -1  # Overbought → short
                entry_bar = i
                entry_price = close[i] if close is not None else 0
            elif z < -z_entry:
                positions[i] = 1  # Oversold → long
                entry_bar = i
                entry_price = close[i] if close is not None else 0

    return positions, trades


def evaluate_mr(asset, timeframe, lookback=20, z_entry=1.5, z_exit=0.5,
                sl_multiple=3.0, tp_multiple=2.0, max_holding=30):
    """Evaluate MR strategy with given parameters. Returns metrics dict."""
    df = download_asset_data(asset, timeframe)
    if df is None or len(df) < 200:
        return None

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    positions, trades = run_mr_strategy(
        log_ret, close, lookback=lookback, z_entry=z_entry, z_exit=z_exit,
        sl_multiple=sl_multiple, tp_multiple=tp_multiple, max_holding=max_holding
    )

    # Compute returns
    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])

    # B&H
    bh_sharpe = annualized_sharpe(log_ret[1:], ppy)
    oracle_sharpe = annualized_sharpe(net, ppy)
    edge_sharpe = oracle_sharpe - bh_sharpe

    # Rolling window
    rolling = rolling_window_evaluation(net, ppy)

    # Max drawdown
    equity = np.cumsum(net)
    equity_curve = np.exp(equity)
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        avg_hold = np.mean([t["exit"] - t["entry"] for t in trades])
        trades_per_year = n_trades / (len(log_ret) / ppy) if ppy > 0 else 0
    else:
        win_rate = avg_win = avg_loss = avg_hold = trades_per_year = 0

    return {
        "sharpe": round(oracle_sharpe, 4),
        "bh_sharpe": round(bh_sharpe, 4),
        "edge_sharpe": round(edge_sharpe, 4),
        "regime_robustness": rolling["regime_robustness"],
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4) if n_trades > 0 else None,
        "avg_win": round(avg_win, 6) if n_trades > 0 else None,
        "avg_loss": round(avg_loss, 6) if n_trades > 0 else None,
        "avg_hold_bars": round(avg_hold, 1) if n_trades > 0 else None,
        "trades_per_year": round(trades_per_year, 1),
        "n_bars": len(log_ret),
    }


# ============================================================
# TASK A.1 — Parameter perturbation audit
# ============================================================
def task_a1():
    print("=" * 70)
    print("TRACK A — TASK A.1: PARAMETER PERTURBATION AUDIT")
    print("=" * 70)

    perturbation_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    param_names = ["lookback", "z_entry", "z_exit"]
    base_values = [BASE_PARAMS["lookback"], BASE_PARAMS["z_entry"], BASE_PARAMS["z_exit"]]

    all_results = {}

    for cell in CELLS:
        asset = cell["asset"]
        tf = cell["timeframe"]
        cell_key = f"{asset}_{tf}"
        print(f"\n  {asset} / {tf}:")

        # Baseline
        base_result = evaluate_mr(asset, tf, **BASE_PARAMS)
        if base_result is None:
            print(f"    ⚠ No data, skipping")
            continue
        print(f"    Baseline: SR={base_result['sharpe']:+.3f}, Edge={base_result['edge_sharpe']:+.3f}, "
              f"Trades/yr={base_result['trades_per_year']:.0f}, WinRate={base_result.get('win_rate', 'N/A')}")

        # One-at-a-time perturbation
        sweep_results = {"baseline": base_result, "sweeps": {}}

        for p_idx, p_name in enumerate(param_names):
            sweep_results["sweeps"][p_name] = []
            print(f"\n    Sweeping {p_name} (base={base_values[p_idx]}):")

            for factor in perturbation_factors:
                params = dict(BASE_PARAMS)
                if p_name == "lookback":
                    params[p_name] = max(5, int(base_values[p_idx] * factor))
                else:
                    params[p_name] = round(base_values[p_idx] * factor, 3)

                result = evaluate_mr(asset, tf, **params)
                if result is None:
                    continue

                sweep_results["sweeps"][p_name].append({
                    "factor": factor,
                    "value": params[p_name],
                    **result,
                })
                print(f"      {p_name}={params[p_name]:>6}: SR={result['sharpe']:+.3f}, "
                      f"Edge={result['edge_sharpe']:+.3f}, Trades/yr={result['trades_per_year']:.0f}")

        # Full grid sweep (3×3×3 = 27 combos for plateau analysis)
        print(f"\n    Full grid sweep (lookback × z_entry × z_exit):")
        grid_lookbacks = [10, 15, 20, 25, 30]
        grid_z_entries = [1.0, 1.25, 1.5, 1.75, 2.0]
        grid_z_exits = [0.25, 0.5, 0.75]

        grid_results = []
        for lb, ze, zx in itertools.product(grid_lookbacks, grid_z_entries, grid_z_exits):
            if zx >= ze:
                continue  # z_exit must be < z_entry
            result = evaluate_mr(asset, tf, lookback=lb, z_entry=ze, z_exit=zx)
            if result is None:
                continue
            grid_results.append({
                "lookback": lb, "z_entry": ze, "z_exit": zx, **result
            })

        sweep_results["grid"] = grid_results

        # Analyze plateau
        sharpes = [r["sharpe"] for r in grid_results if r["sharpe"] is not None]
        if sharpes:
            sr_mean = np.mean(sharpes)
            sr_std = np.std(sharpes)
            sr_max = max(sharpes)
            sr_min = min(sharpes)
            pct_above_zero = np.mean([s > 0 for s in sharpes]) * 100
            pct_above_half_max = np.mean([s > sr_max * 0.5 for s in sharpes]) * 100

            sweep_results["plateau_analysis"] = {
                "n_combos": len(grid_results),
                "sharpe_mean": round(sr_mean, 4),
                "sharpe_std": round(sr_std, 4),
                "sharpe_max": round(sr_max, 4),
                "sharpe_min": round(sr_min, 4),
                "pct_positive": round(pct_above_zero, 1),
                "pct_above_half_max": round(pct_above_half_max, 1),
                "is_plateau": pct_above_half_max >= 50,  # >50% of space yields >50% of max
            }
            pa = sweep_results["plateau_analysis"]
            print(f"      Grid: {pa['n_combos']} combos, SR range [{pa['sharpe_min']:+.3f}, {pa['sharpe_max']:+.3f}], "
                  f"mean={pa['sharpe_mean']:+.3f}±{pa['sharpe_std']:.3f}")
            print(f"      Plateau: {pa['pct_above_half_max']:.0f}% of space ≥ 50% of max SR → "
                  f"{'VALID PLATEAU' if pa['is_plateau'] else '⚠ NARROW SPIKE'}")

        all_results[cell_key] = sweep_results

    return all_results


# ============================================================
# TASK A.3 — Risk-calibrated position sizing
# ============================================================
def task_a3(a1_results):
    print("\n" + "=" * 70)
    print("TRACK A — TASK A.3: RISK-CALIBRATED POSITION SIZING")
    print("=" * 70)

    sizing_results = {}
    ACCOUNT_SIZE = 10000  # USD demo account
    MAX_ACCOUNT_DD = 0.15  # 15% max account drawdown tolerance

    for cell_key, cell_data in a1_results.items():
        baseline = cell_data.get("baseline")
        if baseline is None:
            continue

        asset = cell_key.split("_")[0] + "/" + cell_key.split("_")[1]
        worst_window_dd = abs(baseline["max_drawdown"])  # historical worst

        # Size so that worst historical DD maps to MAX_ACCOUNT_DD
        if worst_window_dd > 0:
            leverage = MAX_ACCOUNT_DD / worst_window_dd
        else:
            leverage = 1.0

        # Clamp leverage
        leverage = min(leverage, 5.0)
        leverage = max(leverage, 0.1)

        risk_per_trade_pct = leverage * 1.0  # 1% base risk scaled by leverage
        position_size_usd = ACCOUNT_SIZE * leverage

        sizing = {
            "asset": asset,
            "account_size_usd": ACCOUNT_SIZE,
            "max_account_dd_target": MAX_ACCOUNT_DD,
            "historical_max_dd": round(worst_window_dd, 4),
            "worst_2y_sharpe": baseline["worst_window_sharpe"],
            "computed_leverage": round(leverage, 4),
            "position_size_usd": round(position_size_usd, 0),
            "risk_per_trade_pct": round(risk_per_trade_pct, 2),
            "expected_trades_per_year": baseline["trades_per_year"],
        }

        print(f"\n  {asset}:")
        print(f"    Historical max DD: {worst_window_dd:.1%}")
        print(f"    Worst 2Y Sharpe: {baseline['worst_window_sharpe']:+.3f}")
        print(f"    Leverage: {leverage:.2f}x")
        print(f"    Position size: ${position_size_usd:,.0f} per trade on ${ACCOUNT_SIZE:,} account")
        print(f"    Risk per trade: {risk_per_trade_pct:.1f}% of account")
        print(f"    Expected trades/year: {baseline['trades_per_year']:.0f}")

        sizing_results[cell_key] = sizing

    return sizing_results


# ============================================================
# TASK A.5 — Monitoring protocol
# ============================================================
def task_a5(a1_results):
    print("\n" + "=" * 70)
    print("TRACK A — TASK A.5: MONITORING PROTOCOL")
    print("=" * 70)

    protocols = {}
    for cell_key, cell_data in a1_results.items():
        baseline = cell_data.get("baseline")
        if baseline is None:
            continue

        asset = cell_key.split("_")[0] + "/" + cell_key.split("_")[1]

        # Compute alert thresholds
        worst_dd = abs(baseline["max_drawdown"])
        pause_dd = worst_dd * 1.5

        protocol = {
            "asset": asset,
            "review_schedule": {
                "weekly": "trades executed vs backtest distribution",
                "monthly": "Sharpe, drawdown, trade stats vs backtest",
            },
            "alert_criteria": {
                "max_drawdown_pause": round(pause_dd, 4),
                "consecutive_losing_trades_pause": 10,
                "slippage_threshold_bps": 3.0,  # 2× modeled for FX
                "slippage_breach_count": 10,
            },
            "backtest_reference": {
                "expected_sharpe": baseline["sharpe"],
                "expected_trades_per_year": baseline["trades_per_year"],
                "expected_win_rate": baseline.get("win_rate"),
                "expected_max_dd": baseline["max_drawdown"],
            },
        }
        protocols[cell_key] = protocol

        print(f"\n  {asset}:")
        print(f"    Pause if DD exceeds: {pause_dd:.1%} (1.5× historical {worst_dd:.1%})")
        print(f"    Pause if 10 consecutive losing trades")
        print(f"    Pause if slippage > 3 bps for 10+ trades")
        print(f"    Expected: SR={baseline['sharpe']:+.3f}, {baseline['trades_per_year']:.0f} trades/yr")

    return protocols


def main():
    print("=" * 70)
    print("PHASE 4 — TRACK A: ORACLE-INDEPENDENT MEAN REVERSION")
    print("=" * 70)

    # Task A.1
    a1_results = task_a1()

    # Task A.3
    a3_results = task_a3(a1_results)

    # Task A.5
    a5_results = task_a5(a1_results)

    # Aggregate
    output = {
        "track": "A",
        "cells": {},
    }

    for cell_key in a1_results:
        output["cells"][cell_key] = {
            "parameter_audit": {
                "baseline": a1_results[cell_key]["baseline"],
                "plateau_analysis": a1_results[cell_key].get("plateau_analysis"),
                "sweeps_summary": {
                    p_name: [{
                        "value": r["value"],
                        "sharpe": r["sharpe"],
                        "edge_sharpe": r["edge_sharpe"],
                    } for r in a1_results[cell_key]["sweeps"][p_name]]
                    for p_name in a1_results[cell_key]["sweeps"]
                },
                "grid_size": len(a1_results[cell_key].get("grid", [])),
            },
            "position_sizing": a3_results.get(cell_key),
            "monitoring_protocol": a5_results.get(cell_key),
        }

    # Decision
    all_plateau = all(
        a1_results[k].get("plateau_analysis", {}).get("is_plateau", False)
        for k in a1_results
    )
    output["decision"] = {
        "plateau_valid": all_plateau,
        "deployable": all_plateau,
        "notes": "Both cells show valid parameter plateau" if all_plateau
                 else "One or more cells have narrow spike — deployment NOT recommended",
    }

    print("\n" + "=" * 70)
    print("TRACK A DECISION")
    print("=" * 70)
    print(f"  Plateau valid: {all_plateau}")
    print(f"  Deployable: {all_plateau}")

    output_path = os.path.join(RESULTS_DIR, "phase4_track_a_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
