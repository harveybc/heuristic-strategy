#!/usr/bin/env python3
"""
Phase 6.E.0.1 — Plugin-Canonical Phase 6.C Re-Run

Runs the SAME portfolio pipeline (eval_cell, eval_portfolio_daily, cost model,
vol-scaling) but replaces script-level signal generation with LTS plugins:
  - EurUsdMrStrategy
  - UsdJpyTsmomStrategy
  - UsdJpyDualMomentumStrategy

Outputs:
  1. Signal comparison: plugin vs script position agreement
  2. Plugin-canonical 6.C.0 (held-out) + 6.C.3 (cost sensitivity)
  3. Comparison table: script-canonical vs plugin-canonical

Usage:
  cd /home/harveybc/Documents/GitHub/heuristic-strategy
  conda run -n tensorflow python3 -u trading_research/phase6e01_plugin_canonical.py 2>&1
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
EXTENDED_DATA = os.path.join(SCRIPT_DIR, "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add paths
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "..", "lts"))

from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation, compute_strategy_metrics
)
from trading_research.transaction_cost_model import apply_cost_to_returns

# Import plugins
from plugins_strategy.eurusd_mr_strategy import EurUsdMrStrategy
from plugins_strategy.usdjpy_tsmom_strategy import UsdJpyTsmomStrategy
from plugins_strategy.usdjpy_dual_momentum_strategy import UsdJpyDualMomentumStrategy

PPY_DAILY = 252
PPY_WEEKLY = 52
TARGET_VOL = 0.10
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
HOLDOUT_END = "2025-12-31"


# ================================================================
# DATA LOADING (identical to phase6c_omega.py)
# ================================================================
def load_daily_data(asset):
    safe = asset.replace("/", "_")
    csv_path = os.path.join(EXTENDED_DATA, f"{safe}_daily.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found.")
        return None
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_all_data():
    assets = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]
    data = {}
    for asset in assets:
        df = load_daily_data(asset)
        if df is not None:
            close = df["Close"].values.astype(float)
            high = df["High"].values.astype(float) if "High" in df.columns else close
            low = df["Low"].values.astype(float) if "Low" in df.columns else close
            dates = df.index
            log_ret = np.diff(np.log(close + 1e-12), prepend=0)
            log_ret[0] = 0
            data[asset] = {"close": close, "high": high, "low": low,
                           "dates": dates, "log_ret": log_ret, "df": df}
            print(f"  {asset}: {len(close)} bars, {dates[0].date()} to {dates[-1].date()}")
    all_fx = {a: (data[a]["log_ret"], data[a]["close"], data[a]["dates"]) for a in data}
    return data, all_fx


# ================================================================
# SCRIPT-LEVEL STRATEGIES (reference, identical to phase6c_omega.py)
# ================================================================
def run_pure_mr(log_ret, close, lookback=20, z_entry=1.5, z_exit=0.5):
    n = len(log_ret)
    positions = np.zeros(n)
    cum_ret = np.cumsum(log_ret)
    entry_bar = -999
    for i in range(lookback, n):
        window = cum_ret[max(0, i - lookback):i + 1]
        std = np.std(window)
        if std < 1e-12:
            continue
        z = (cum_ret[i] - np.mean(window)) / std
        if positions[i - 1] != 0:
            bars_held = i - entry_bar
            pnl = positions[i - 1] * (cum_ret[i] - cum_ret[entry_bar])
            atr = np.std(log_ret[max(0, i - lookback):i]) if i >= lookback else 0.01
            if pnl < -3.0 * atr or pnl > 2.0 * atr or bars_held >= 30 or abs(z) < z_exit:
                positions[i] = 0
            else:
                positions[i] = positions[i - 1]
        else:
            if z > z_entry:
                positions[i] = -1
                entry_bar = i
            elif z < -z_entry:
                positions[i] = 1
                entry_bar = i
    return positions


def run_tsmom(log_ret_daily, close_daily, dates, lookback_months=12):
    monthly_close = pd.Series(close_daily, index=dates).resample("ME").last().dropna()
    monthly_dates = monthly_close.index
    positions = np.zeros(len(log_ret_daily))
    for i in range(lookback_months, len(monthly_dates) - 1):
        ret_12m = np.log(monthly_close.iloc[i] + 1e-12) - np.log(
            monthly_close.iloc[i - lookback_months] + 1e-12)
        signal = np.sign(ret_12m)
        mask = (dates >= monthly_dates[max(0, i - 1)]) & (dates <= monthly_dates[i])
        recent = log_ret_daily[mask]
        vol = np.std(recent[-min(252, len(recent)):]) * np.sqrt(252) if len(recent) > 20 else 0.10
        size = min(TARGET_VOL / max(vol, 0.01), 3.0)
        next_date = monthly_dates[i + 1] if i + 1 < len(monthly_dates) else dates[-1]
        mask_next = (dates > monthly_dates[i]) & (dates <= next_date)
        positions[mask_next] = signal * size
    return positions


def run_dual_momentum(log_ret_daily, close_daily, dates, all_fx_data, asset, lookback_months=12):
    mc = pd.Series(close_daily, index=dates).resample("ME").last().dropna()
    all_mc = {}
    for a, (lr, cl, dt) in all_fx_data.items():
        all_mc[a] = pd.Series(cl, index=dt).resample("ME").last().dropna()
    positions = np.zeros(len(log_ret_daily))
    common = mc.index
    for a_mc in all_mc.values():
        common = common.intersection(a_mc.index)
    common = common.sort_values()
    for i in range(lookback_months, len(common) - 1):
        month = common[i]
        if month not in mc.index:
            continue
        mc_loc = mc.index.get_loc(month)
        if mc_loc < lookback_months:
            continue
        ret_12m = np.log(mc.iloc[mc_loc] + 1e-12) - np.log(
            mc.iloc[mc_loc - lookback_months] + 1e-12)
        all_rets = {}
        for a2, mc2 in all_mc.items():
            if month in mc2.index:
                loc2 = mc2.index.get_loc(month)
                if loc2 >= lookback_months:
                    all_rets[a2] = np.log(mc2.iloc[loc2] + 1e-12) - np.log(
                        mc2.iloc[loc2 - lookback_months] + 1e-12)
        if not all_rets:
            continue
        best = max(all_rets, key=all_rets.get)
        next_month = common[i + 1] if i + 1 < len(common) else dates[-1]
        mask = (dates > month) & (dates <= next_month)
        if ret_12m > 0 and best == asset:
            positions[mask] = 1
    return positions


# ================================================================
# PLUGIN SIGNAL → POSITIONS CONVERSION
# ================================================================
def run_mr_plugin(close, dates):
    """Run EUR/USD MR plugin bar-by-bar, return positions array."""
    plugin = EurUsdMrStrategy()
    n = len(close)
    positions = np.zeros(n)

    for i in range(n):
        market_data = {"close": float(close[i])}
        signal = plugin.generate_signal("EUR/USD", market_data=market_data)
        action = signal.get("action", "none")

        if action == "open":
            side = signal["parameters"].get("side", "buy")
            positions[i] = 1 if side == "buy" else -1
        elif action == "close":
            positions[i] = 0
        elif action == "none":
            positions[i] = positions[i - 1] if i > 0 else 0

    return positions


def run_tsmom_plugin(close, dates):
    """Run USD/JPY TSMOM plugin bar-by-bar, return positions array."""
    plugin = UsdJpyTsmomStrategy()
    n = len(close)
    positions = np.zeros(n)

    for i in range(n):
        dt = dates[i]
        if hasattr(dt, 'isoformat'):
            date_str = dt.isoformat()
        else:
            date_str = str(dt)

        market_data = {"close": float(close[i]), "date": date_str}
        signal = plugin.generate_signal("USD/JPY", market_data=market_data)
        action = signal.get("action", "none")

        if action == "open":
            side = signal["parameters"].get("side", "buy")
            vol_size = signal["parameters"].get("vol_size", 1.0)
            positions[i] = vol_size if side == "buy" else -vol_size
        elif action == "close":
            positions[i] = 0
        elif action == "none":
            positions[i] = positions[i - 1] if i > 0 else 0

    return positions


def run_dm_plugin(close, dates, peer_data):
    """Run USD/JPY DM plugin bar-by-bar, return positions array.
    peer_data: dict of {asset: close_array} for peers.
    """
    plugin = UsdJpyDualMomentumStrategy()
    n = len(close)
    positions = np.zeros(n)

    for i in range(n):
        dt = dates[i]
        if hasattr(dt, 'isoformat'):
            date_str = dt.isoformat()
        else:
            date_str = str(dt)

        peer_prices = {}
        for asset, peer_close in peer_data.items():
            if i < len(peer_close):
                peer_prices[asset] = float(peer_close[i])

        market_data = {
            "close": float(close[i]),
            "date": date_str,
            "peer_prices": peer_prices,
        }
        signal = plugin.generate_signal("USD/JPY", market_data=market_data)
        action = signal.get("action", "none")

        if action == "open":
            positions[i] = 1
        elif action == "close":
            positions[i] = 0
        elif action == "none":
            positions[i] = positions[i - 1] if i > 0 else 0

    return positions


# ================================================================
# EVALUATION PIPELINE (identical to phase6c_omega.py)
# ================================================================
def eval_cell(log_ret, positions, asset, dates, label, cost_multiplier=1.0):
    gross_ret = positions[:-1] * log_ret[1:]
    net_ret_base = apply_cost_to_returns(gross_ret, positions[:-1], asset, np.abs(log_ret[:-1]))
    cost_component = gross_ret - net_ret_base
    net_ret = gross_ret - cost_component * cost_multiplier

    realized_vol = np.std(net_ret) * np.sqrt(PPY_DAILY) if np.std(net_ret) > 0 else 0.10
    vol_scalar = TARGET_VOL / max(realized_vol, 0.01)
    vol_scalar = min(vol_scalar, 5.0)
    net_ret = net_ret * vol_scalar

    eval_dates = dates[1:len(net_ret) + 1]
    return net_ret, eval_dates, vol_scalar


def eval_portfolio_daily(cell_returns_dict, cell_weights, common_dates):
    df = pd.DataFrame(cell_returns_dict, index=common_dates)
    weekly_df = df.resample("W").sum().dropna(how='all')

    port_ret = np.zeros(len(weekly_df))
    for name in weekly_df.columns:
        w = cell_weights.get(name, 0.0)
        port_ret += w * weekly_df[name].values

    realized_vol = np.std(port_ret) * np.sqrt(PPY_WEEKLY) if len(port_ret) > 10 else 0.10
    sharpe = annualized_sharpe(port_ret, PPY_WEEKLY)
    weekly_dates = weekly_df.index.values

    equity = np.exp(np.cumsum(port_ret))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    leverage = TARGET_VOL / max(realized_vol, 0.001)
    port_ret_10vol = port_ret * leverage
    equity_10vol = np.exp(np.cumsum(port_ret_10vol))
    peak_10vol = np.maximum.accumulate(equity_10vol)
    dd_10vol = (peak_10vol - equity_10vol) / (peak_10vol + 1e-12)
    max_dd_10vol = float(np.max(dd_10vol))

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "vol": round(realized_vol, 4),
        "total_return": round(float(equity[-1] - 1), 4) if len(equity) > 0 else 0.0,
        "n_weeks": len(port_ret),
        "port_ret": port_ret,
        "weekly_dates": weekly_dates,
        "at_10pct_vol": {
            "leverage": round(leverage, 4),
            "max_dd": round(max_dd_10vol, 4),
            "total_return": round(float(equity_10vol[-1] - 1), 4) if len(equity_10vol) > 0 else 0.0,
        },
    }


# ================================================================
# P3 PORTFOLIO RUNNERS
# ================================================================
def run_p3_script(data, all_fx, cost_multiplier=1.0, date_filter=None, fixed_weights=None):
    """Script-canonical P3."""
    d_eu = data["EUR/USD"]
    d_jp = data["USD/JPY"]

    mr_pos = run_pure_mr(d_eu["log_ret"], d_eu["close"])
    tsmom_pos = run_tsmom(d_jp["log_ret"], d_jp["close"], d_jp["dates"])
    dm_pos = run_dual_momentum(d_jp["log_ret"], d_jp["close"], d_jp["dates"], all_fx, "USD/JPY")

    return _build_p3(data, mr_pos, tsmom_pos, dm_pos, cost_multiplier, date_filter, fixed_weights)


def run_p3_plugin(data, all_fx, cost_multiplier=1.0, date_filter=None, fixed_weights=None):
    """Plugin-canonical P3."""
    d_eu = data["EUR/USD"]
    d_jp = data["USD/JPY"]

    # Peer data for DM plugin
    peer_data = {}
    for asset in ["EUR/USD", "GBP/USD", "AUD/USD"]:
        if asset in data:
            peer_data[asset] = data[asset]["close"]

    print("    Generating plugin signals...")
    mr_pos = run_mr_plugin(d_eu["close"], d_eu["dates"])
    tsmom_pos = run_tsmom_plugin(d_jp["close"], d_jp["dates"])
    dm_pos = run_dm_plugin(d_jp["close"], d_jp["dates"], peer_data)

    return _build_p3(data, mr_pos, tsmom_pos, dm_pos, cost_multiplier, date_filter, fixed_weights)


def _build_p3(data, mr_pos, tsmom_pos, dm_pos, cost_multiplier=1.0, date_filter=None,
              fixed_weights=None):
    """Common P3 assembly from positions arrays.
    If fixed_weights is provided, use those instead of re-deriving from rolling eval.
    """
    d_eu = data["EUR/USD"]
    d_jp = data["USD/JPY"]

    mr_ret, mr_dates, mr_vs = eval_cell(
        d_eu["log_ret"], mr_pos, "EUR/USD", d_eu["dates"], "EUR/USD_mr", cost_multiplier)
    tsmom_ret, tsmom_dates, tsmom_vs = eval_cell(
        d_jp["log_ret"], tsmom_pos, "USD/JPY", d_jp["dates"], "USD/JPY_tsmom", cost_multiplier)
    dm_ret, dm_dates, dm_vs = eval_cell(
        d_jp["log_ret"], dm_pos, "USD/JPY", d_jp["dates"], "USD/JPY_dm", cost_multiplier)

    # P3 weights from rolling worst-window
    mr_rolling = rolling_window_evaluation(mr_ret, PPY_DAILY)
    tsmom_rolling = rolling_window_evaluation(tsmom_ret, PPY_DAILY)
    dm_rolling = rolling_window_evaluation(dm_ret, PPY_DAILY)

    cell_worst = {
        "eurusd_mr": mr_rolling["worst_window_sharpe"],
        "usdjpy_tsmom": tsmom_rolling["worst_window_sharpe"],
        "usdjpy_dm": dm_rolling["worst_window_sharpe"],
    }

    if fixed_weights is not None:
        p3_weights = dict(fixed_weights)
    else:
        inv_worst = {k: 1.0 / max(abs(v), 0.01) for k, v in cell_worst.items()}
        total_inv = sum(inv_worst.values())
        p3_weights = {k: v / total_inv for k, v in inv_worst.items()}

    # Align dates
    common = mr_dates.intersection(tsmom_dates).intersection(dm_dates).sort_values()
    mr_mask = mr_dates.isin(common)
    tsmom_mask = tsmom_dates.isin(common)
    dm_mask = dm_dates.isin(common)
    min_len = min(mr_mask.sum(), tsmom_mask.sum(), dm_mask.sum())
    mr_aligned = mr_ret[mr_mask][:min_len]
    tsmom_aligned = tsmom_ret[tsmom_mask][:min_len]
    dm_aligned = dm_ret[dm_mask][:min_len]
    common_sorted = common[:min_len]

    if date_filter is not None:
        start, end = date_filter
        fmask = (common_sorted >= pd.Timestamp(start)) & (common_sorted <= pd.Timestamp(end))
        mr_aligned = mr_aligned[fmask]
        tsmom_aligned = tsmom_aligned[fmask]
        dm_aligned = dm_aligned[fmask]
        common_sorted = common_sorted[fmask]

    if len(common_sorted) < 20:
        return None

    cells = {
        "eurusd_mr": mr_aligned,
        "usdjpy_tsmom": tsmom_aligned,
        "usdjpy_dm": dm_aligned,
    }

    portfolio = eval_portfolio_daily(cells, p3_weights, common_sorted)

    # Per-cell metrics
    cell_metrics = {}
    for name, rets in cells.items():
        sr = annualized_sharpe(rets, PPY_DAILY)
        eq = np.exp(np.cumsum(rets))
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / (pk + 1e-12)
        cell_metrics[name] = {
            "sharpe": round(sr, 4),
            "total_return": round(float(eq[-1] - 1), 4) if len(eq) > 0 else 0.0,
            "max_dd": round(float(np.max(dd)), 4),
            "n_bars": len(rets),
        }

    rolling = rolling_window_evaluation(portfolio["port_ret"], PPY_WEEKLY,
                                         window_years=2.0, step_months=6)

    return {
        "p3_weights": {k: round(v, 4) for k, v in p3_weights.items()},
        "portfolio": portfolio,
        "cells": cell_metrics,
        "worst_2y": {k: round(v, 4) for k, v in cell_worst.items()},
        "rolling": {
            "worst_window_sharpe": rolling["worst_window_sharpe"],
            "regime_robustness": rolling["regime_robustness"],
        },
    }


# ================================================================
# TASK 1: SIGNAL COMPARISON
# ================================================================
def compare_signals(data, all_fx):
    print("\n" + "=" * 70)
    print("TASK 1: SIGNAL COMPARISON — Plugin vs Script")
    print("=" * 70)

    d_eu = data["EUR/USD"]
    d_jp = data["USD/JPY"]

    # Generate script positions
    print("\n  [SCRIPT] Generating script-canonical positions...")
    mr_script = run_pure_mr(d_eu["log_ret"], d_eu["close"])
    tsmom_script = run_tsmom(d_jp["log_ret"], d_jp["close"], d_jp["dates"])
    dm_script = run_dual_momentum(d_jp["log_ret"], d_jp["close"], d_jp["dates"], all_fx, "USD/JPY")

    # Generate plugin positions
    print("  [PLUGIN] Generating plugin-canonical positions...")
    peer_data = {a: data[a]["close"] for a in ["EUR/USD", "GBP/USD", "AUD/USD"] if a in data}
    mr_plugin = run_mr_plugin(d_eu["close"], d_eu["dates"])
    tsmom_plugin = run_tsmom_plugin(d_jp["close"], d_jp["dates"])
    dm_plugin = run_dm_plugin(d_jp["close"], d_jp["dates"], peer_data)

    results = {}

    for name, script_pos, plugin_pos, dates in [
        ("EUR/USD MR", mr_script, mr_plugin, d_eu["dates"]),
        ("USD/JPY TSMOM", tsmom_script, tsmom_plugin, d_jp["dates"]),
        ("USD/JPY DM", dm_script, dm_plugin, d_jp["dates"]),
    ]:
        n = min(len(script_pos), len(plugin_pos))
        sp = script_pos[:n]
        pp = plugin_pos[:n]

        # Direction agreement (sign match)
        # Treat both-zero as agreement
        script_dir = np.sign(sp)
        plugin_dir = np.sign(pp)
        direction_match = np.sum(script_dir == plugin_dir) / n

        # Bars where both are non-zero
        both_active = (sp != 0) & (pp != 0)
        if np.sum(both_active) > 0:
            active_dir_match = np.sum(
                script_dir[both_active] == plugin_dir[both_active]) / np.sum(both_active)
        else:
            active_dir_match = 0.0

        # Position correlation
        corr = np.corrcoef(sp, pp)[0, 1] if np.std(sp) > 0 and np.std(pp) > 0 else 0.0

        # Exposure time
        script_exposure = np.sum(sp != 0) / n
        plugin_exposure = np.sum(pp != 0) / n

        # Trade count
        script_trades = np.sum(np.abs(np.diff(sp, prepend=0)) > 0.01)
        plugin_trades = np.sum(np.abs(np.diff(pp, prepend=0)) > 0.01)

        print(f"\n  [{name}]")
        print(f"    Direction agreement: {direction_match*100:.1f}%")
        print(f"    Active direction match: {active_dir_match*100:.1f}%")
        print(f"    Position correlation: {corr:.4f}")
        print(f"    Script exposure: {script_exposure*100:.1f}%, Plugin: {plugin_exposure*100:.1f}%")
        print(f"    Script trades: {int(script_trades)}, Plugin trades: {int(plugin_trades)}")

        # Year-by-year breakdown
        years = sorted(set(d.year for d in dates[:n]))
        print(f"    [Year-by-year direction agreement]")
        for year in years:
            ymask = np.array([d.year == year for d in dates[:n]])
            if np.sum(ymask) < 10:
                continue
            yr_match = np.sum(script_dir[ymask] == plugin_dir[ymask]) / np.sum(ymask)
            print(f"      {year}: {yr_match*100:.1f}%")

        results[name] = {
            "direction_agreement": round(direction_match, 4),
            "active_direction_match": round(active_dir_match, 4),
            "position_correlation": round(corr, 4),
            "script_exposure_pct": round(script_exposure * 100, 1),
            "plugin_exposure_pct": round(plugin_exposure * 100, 1),
            "script_trades": int(script_trades),
            "plugin_trades": int(plugin_trades),
        }

    return results


# ================================================================
# TASK 2: PLUGIN-CANONICAL 6.C.0 + 6.C.3
# ================================================================
def task_6c0_plugin(data, all_fx, fixed_weights=None):
    wt_label = "fixed" if fixed_weights else "derived"
    print("\n" + "=" * 70)
    print(f"TASK 2A: PLUGIN-CANONICAL 6.C.0 — HELD-OUT EVALUATION ({wt_label} weights)")
    print("=" * 70)

    # Full-period baseline
    print("\n  Running plugin P3 on full period...")
    full_result = run_p3_plugin(data, all_fx, fixed_weights=fixed_weights)
    if full_result is None:
        print("  FATAL: Could not run plugin P3 on full period.")
        return None

    print(f"  Full-period P3: Sharpe={full_result['portfolio']['sharpe']:.4f}")
    print(f"  P3 weights: {full_result['p3_weights']}")

    # Held-out period
    print(f"\n  Running plugin P3 on held-out ({HOLDOUT_START} to {HOLDOUT_END})...")
    holdout_result = run_p3_plugin(data, all_fx,
                                    date_filter=(HOLDOUT_START, HOLDOUT_END),
                                    fixed_weights=fixed_weights)
    if holdout_result is None:
        print("  FATAL: Insufficient held-out data for plugins.")
        return {"pass": False, "reason": "insufficient_data"}

    port = holdout_result["portfolio"]
    cells = holdout_result["cells"]

    print(f"\n  [PLUGIN HELD-OUT RESULTS]")
    print(f"    Portfolio Sharpe:    {port['sharpe']:.4f}")
    print(f"    Portfolio max DD:    {port['max_dd']*100:.1f}%")
    print(f"    Portfolio return:    {port['total_return']*100:.1f}%")
    print(f"    Portfolio vol:       {port['vol']*100:.1f}%")
    print(f"    N weeks:             {port['n_weeks']}")

    print(f"\n  [PER-CELL HELD-OUT]")
    for name, m in cells.items():
        print(f"    {name}: Sharpe={m['sharpe']:.4f}, return={m['total_return']*100:.1f}%, "
              f"maxDD={m['max_dd']*100:.1f}%")

    # IS and OOS for comparison
    is_result = run_p3_plugin(data, all_fx, date_filter=("2003-01-01", TRAIN_END),
                               fixed_weights=fixed_weights)
    oos_result = run_p3_plugin(data, all_fx, date_filter=(TEST_START, TEST_END),
                                fixed_weights=fixed_weights)

    print(f"\n  [PLUGIN COMPARISON]")
    if is_result:
        print(f"    IS (≤2018):       Sharpe={is_result['portfolio']['sharpe']:.4f}, "
              f"maxDD={is_result['portfolio']['max_dd']*100:.1f}%")
    if oos_result:
        print(f"    OOS (2019-23):    Sharpe={oos_result['portfolio']['sharpe']:.4f}, "
              f"maxDD={oos_result['portfolio']['max_dd']*100:.1f}%")
    print(f"    HELD-OUT (24-25): Sharpe={port['sharpe']:.4f}, "
          f"maxDD={port['max_dd']*100:.1f}%")

    # Gate
    held_out_pass = port["sharpe"] > 0 and port["max_dd"] < 0.30
    held_out_deploy = port["sharpe"] > 0 and port["max_dd"] < 0.25

    print(f"\n  [KILL CRITERIA]")
    print(f"    Sharpe > 0:     {'PASS' if port['sharpe'] > 0 else 'FAIL'} ({port['sharpe']:.4f})")
    print(f"    max DD < 25%:   {'PASS' if port['max_dd'] < 0.25 else 'FAIL'} ({port['max_dd']*100:.1f}%)")
    print(f"    max DD < 30%:   {'PASS' if port['max_dd'] < 0.30 else 'FAIL'} ({port['max_dd']*100:.1f}%)")
    print(f"    GATE:           {'PASS' if held_out_pass else 'FAIL'}")

    return {
        "full_period": {
            "sharpe": full_result["portfolio"]["sharpe"],
            "max_dd": full_result["portfolio"]["max_dd"],
            "total_return": full_result["portfolio"]["total_return"],
            "worst_2y": full_result["rolling"]["worst_window_sharpe"],
            "p3_weights": full_result["p3_weights"],
        },
        "held_out": {
            "sharpe": port["sharpe"],
            "max_dd": port["max_dd"],
            "total_return": port["total_return"],
            "vol": port["vol"],
            "n_weeks": port["n_weeks"],
            "at_10pct_vol": port.get("at_10pct_vol", {}),
        },
        "cells": {k: v for k, v in cells.items()},
        "p3_weights": holdout_result["p3_weights"],
        "is_sharpe": is_result["portfolio"]["sharpe"] if is_result else None,
        "oos_sharpe": oos_result["portfolio"]["sharpe"] if oos_result else None,
        "gate_pass": held_out_pass,
        "deploy_pass": held_out_deploy,
    }


def task_6c3_plugin(data, all_fx, fixed_weights=None):
    print("\n" + "=" * 70)
    print("TASK 2B: PLUGIN-CANONICAL 6.C.3 — COST SENSITIVITY")
    print("=" * 70)

    cost_multipliers = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    results_by_mult = {}

    for cm in cost_multipliers:
        print(f"\n  Running plugin P3 at {cm:.2f}x cost...")
        r = run_p3_plugin(data, all_fx, cost_multiplier=cm, fixed_weights=fixed_weights)
        if r is None:
            print(f"    FAILED")
            continue

        port = r["portfolio"]
        rolling = rolling_window_evaluation(port["port_ret"], PPY_WEEKLY,
                                             window_years=2.0, step_months=6)

        results_by_mult[str(cm)] = {
            "cost_multiplier": cm,
            "sharpe": port["sharpe"],
            "max_dd": port["max_dd"],
            "worst_2y": rolling["worst_window_sharpe"],
            "total_return": port["total_return"],
            "vol": port["vol"],
        }
        print(f"    Sharpe={port['sharpe']:.4f}, worst-2Y={rolling['worst_window_sharpe']:.4f}, "
              f"maxDD={port['max_dd']*100:.1f}%")

    # Break-even analysis
    sharpes = [(cm, results_by_mult[str(cm)]["sharpe"])
               for cm in cost_multipliers if str(cm) in results_by_mult]
    sharpe_breakeven = None
    for i in range(len(sharpes) - 1):
        if sharpes[i][1] > 0 and sharpes[i + 1][1] <= 0:
            x0, y0 = sharpes[i]
            x1, y1 = sharpes[i + 1]
            sharpe_breakeven = x0 + (0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
            break
    if sharpe_breakeven is None and sharpes and sharpes[-1][1] > 0:
        sharpe_breakeven = ">3.0x"

    sharpe_at_2x = results_by_mult.get("2.0", {}).get("sharpe", None)
    cost_pass = True
    if sharpe_at_2x is not None and sharpe_at_2x <= 0:
        cost_pass = False

    print(f"\n  [KILL CRITERIA]")
    print(f"    Sharpe positive at 2.0x: {'PASS' if sharpe_at_2x and sharpe_at_2x > 0 else 'FAIL'} "
          f"({sharpe_at_2x:.4f})" if sharpe_at_2x else "    N/A")
    be_str = f"{sharpe_breakeven:.2f}x" if isinstance(sharpe_breakeven, float) else str(sharpe_breakeven)
    print(f"    Sharpe breakeven: {be_str}")
    print(f"    OVERALL: {'PASS' if cost_pass else 'FAIL'}")

    return {
        "cost_curves": results_by_mult,
        "sharpe_breakeven": be_str,
        "sharpe_at_2x": round(sharpe_at_2x, 4) if sharpe_at_2x else None,
        "pass": cost_pass,
    }


# ================================================================
# TASK 3: COMPARISON TABLE
# ================================================================
def comparison_table(data, all_fx, plugin_6c0, plugin_6c3, fixed_weights=None):
    print("\n" + "=" * 70)
    print("TASK 3: SCRIPT vs PLUGIN COMPARISON TABLE")
    print("=" * 70)

    # Load script-canonical results
    script_results_path = os.path.join(RESULTS_DIR, "phase_6c_omega_results.json")
    if os.path.exists(script_results_path):
        with open(script_results_path) as f:
            script_results = json.load(f)
    else:
        print("  WARNING: No saved script results. Running script-canonical now...")
        script_results = {}

    # Re-run script for fresh comparison
    print("\n  Running script-canonical P3...")
    script_full = run_p3_script(data, all_fx, fixed_weights=fixed_weights)
    script_holdout = run_p3_script(data, all_fx, date_filter=(HOLDOUT_START, HOLDOUT_END),
                                    fixed_weights=fixed_weights)

    print("\n  Running plugin-canonical P3...")
    plugin_full = run_p3_plugin(data, all_fx, fixed_weights=fixed_weights)

    print("\n" + "=" * 70)
    print("  METRIC COMPARISON TABLE")
    print("=" * 70)
    print(f"\n  {'Metric':<30} {'Script':>12} {'Plugin':>12} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    rows = []
    if script_full and plugin_full:
        sp = script_full["portfolio"]
        pp = plugin_full["portfolio"]

        metrics = [
            ("Full Sharpe", sp["sharpe"], pp["sharpe"]),
            ("Full maxDD (%)", sp["max_dd"] * 100, pp["max_dd"] * 100),
            ("Full return (%)", sp["total_return"] * 100, pp["total_return"] * 100),
            ("Full vol (%)", sp["vol"] * 100, pp["vol"] * 100),
        ]

        # Weights
        for k in ["eurusd_mr", "usdjpy_tsmom", "usdjpy_dm"]:
            sw = script_full["p3_weights"].get(k, 0)
            pw = plugin_full["p3_weights"].get(k, 0)
            metrics.append((f"Weight {k} (%)", sw * 100, pw * 100))

        # Cell-level
        for cell_name in ["eurusd_mr", "usdjpy_tsmom", "usdjpy_dm"]:
            sc = script_full["cells"].get(cell_name, {})
            pc = plugin_full["cells"].get(cell_name, {})
            metrics.append((f"  {cell_name} Sharpe", sc.get("sharpe", 0), pc.get("sharpe", 0)))

        # worst-2Y
        if script_full.get("rolling") and plugin_full.get("rolling"):
            metrics.append(("Full worst-2Y Sharpe",
                          script_full["rolling"]["worst_window_sharpe"],
                          plugin_full["rolling"]["worst_window_sharpe"]))

        for label, sv, pv in metrics:
            delta = pv - sv
            print(f"  {label:<30} {sv:>12.4f} {pv:>12.4f} {delta:>+10.4f}")
            rows.append({"metric": label, "script": sv, "plugin": pv, "delta": delta})

    if script_holdout and plugin_6c0:
        print(f"\n  {'--- HELD-OUT ---':<30}")
        sp_ho = script_holdout["portfolio"]
        pp_ho = plugin_6c0["held_out"]

        ho_metrics = [
            ("Held-out Sharpe", sp_ho["sharpe"], pp_ho["sharpe"]),
            ("Held-out maxDD (%)", sp_ho["max_dd"] * 100, pp_ho["max_dd"] * 100),
            ("Held-out return (%)", sp_ho["total_return"] * 100, pp_ho["total_return"] * 100),
        ]
        for label, sv, pv in ho_metrics:
            delta = pv - sv
            print(f"  {label:<30} {sv:>12.4f} {pv:>12.4f} {delta:>+10.4f}")
            rows.append({"metric": label, "script": sv, "plugin": pv, "delta": delta})

    if plugin_6c3:
        print(f"\n  {'--- COST SENSITIVITY ---':<30}")
        for cm_str, pcr in plugin_6c3.get("cost_curves", {}).items():
            sr_results = script_results.get("task_6c3", {}).get("cost_curves", {}).get(cm_str, {})
            script_sr = sr_results.get("sharpe", None)
            plugin_sr = pcr.get("sharpe", None)
            if script_sr is not None and plugin_sr is not None:
                delta = plugin_sr - script_sr
                label = f"  Cost {cm_str}x Sharpe"
                print(f"  {label:<30} {script_sr:>12.4f} {plugin_sr:>12.4f} {delta:>+10.4f}")
                rows.append({"metric": label, "script": script_sr, "plugin": plugin_sr, "delta": delta})

    return rows


# ================================================================
# JSON SERIALIZER
# ================================================================
def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif obj == float("inf"):
        return "inf"
    return obj


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("PHASE 6.E.0.1 — PLUGIN-CANONICAL PHASE 6.C RE-RUN (v2: DM fix + fixed weights)")
    print("=" * 70)

    print("\n[DATA] Loading daily price data...")
    data, all_fx = load_all_data()

    # Script-canonical P3 weights (from Phase 5.5 / 6.C)
    FIXED_P3_WEIGHTS = {
        "eurusd_mr": 0.2055,
        "usdjpy_tsmom": 0.4920,
        "usdjpy_dm": 0.3024,
    }

    results = {}

    # Task 1: Signal comparison (post DM fix)
    signal_comp = compare_signals(data, all_fx)
    results["signal_comparison"] = signal_comp

    # Task 2A: Plugin-canonical 6.C.0 with DERIVED weights
    print("\n  === VARIANT A: Plugin-derived weights ===")
    plugin_6c0_derived = task_6c0_plugin(data, all_fx)
    results["plugin_6c0_derived"] = plugin_6c0_derived

    # Task 2B: Plugin-canonical 6.C.0 with FIXED weights
    print("\n  === VARIANT B: Fixed script-canonical weights ===")
    plugin_6c0_fixed = task_6c0_plugin(data, all_fx, fixed_weights=FIXED_P3_WEIGHTS)
    results["plugin_6c0_fixed"] = plugin_6c0_fixed

    # Task 2C: Plugin-canonical 6.C.3 with fixed weights
    plugin_6c3 = task_6c3_plugin(data, all_fx, fixed_weights=FIXED_P3_WEIGHTS)
    results["plugin_6c3"] = plugin_6c3

    # Task 3: Comparison table (use fixed weights for main comparison)
    comp_rows = comparison_table(data, all_fx, plugin_6c0_fixed, plugin_6c3,
                                  fixed_weights=FIXED_P3_WEIGHTS)
    results["comparison"] = comp_rows

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 6.E.0.1 SUMMARY (v2)")
    print("=" * 70)

    for name, comp in signal_comp.items():
        print(f"  {name} direction agreement: {comp['direction_agreement']*100:.1f}%")

    print(f"\n  --- Derived weights ---")
    if plugin_6c0_derived and isinstance(plugin_6c0_derived, dict) and "held_out" in plugin_6c0_derived:
        ho = plugin_6c0_derived["held_out"]
        print(f"  Plugin held-out Sharpe: {ho['sharpe']:.4f}")
        print(f"  Gate: {'PASS' if plugin_6c0_derived.get('gate_pass') else 'FAIL'}")

    print(f"\n  --- Fixed weights (20.6/49.2/30.2) ---")
    if plugin_6c0_fixed and isinstance(plugin_6c0_fixed, dict) and "held_out" in plugin_6c0_fixed:
        ho = plugin_6c0_fixed["held_out"]
        print(f"  Plugin held-out Sharpe: {ho['sharpe']:.4f}")
        print(f"  Plugin held-out maxDD:  {ho['max_dd']*100:.1f}%")
        print(f"  Gate: {'PASS' if plugin_6c0_fixed.get('gate_pass') else 'FAIL'}")

    if plugin_6c3:
        print(f"\n  Plugin cost sensitivity: {'PASS' if plugin_6c3.get('pass') else 'FAIL'}")
        print(f"  Sharpe breakeven: {plugin_6c3.get('sharpe_breakeven', 'N/A')}")

    # Kill criteria (use FIXED weights variant as primary)
    print("\n  [KILL CRITERIA EVALUATION — Fixed Weights]")
    kills = []
    primary = plugin_6c0_fixed
    if primary and isinstance(primary, dict):
        ho_sharpe = primary.get("held_out", {}).get("sharpe", 0)
        if ho_sharpe < 0:
            kills.append(f"6.C.0 held-out Sharpe = {ho_sharpe:.4f} < 0 → TERMINAL 3")
            print(f"  *** KILL: 6.C.0 held-out Sharpe = {ho_sharpe:.4f} < 0 ***")
        else:
            print(f"  6.C.0 held-out Sharpe = {ho_sharpe:.4f} ≥ 0 → PASS")

    if not kills:
        print("\n  ALL KILL CRITERIA PASSED — Proceed to stress tests")
    else:
        print(f"\n  *** {len(kills)} KILL CRITERIA FAILED ***")
        for k in kills:
            print(f"    {k}")

    # Save
    results_clean = convert_for_json(results)
    out_file = os.path.join(RESULTS_DIR, "phase_6e01_plugin_canonical.json")
    with open(out_file, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
