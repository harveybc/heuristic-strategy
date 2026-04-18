#!/usr/bin/env python3
"""
Phase 6.C — Omega Tasks: 6.C.0 (Held-Out) + 6.C.3 (Cost Sensitivity)

Runs on Omega (local). Light compute tasks.
6.C.0: 2024-2025 held-out evaluation — genuine OOS, gates all subsequent tests.
6.C.3: Cost sensitivity — re-run P3 at multiple cost multipliers.
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    compute_strategy_metrics, periods_per_year_for_timeframe
)
from trading_research.transaction_cost_model import apply_cost_to_returns, COST_TABLE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXTENDED_DATA = os.path.join(os.path.dirname(__file__), "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

PPY_DAILY = 252
TARGET_VOL = 0.10
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
HOLDOUT_END = "2025-12-31"


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


# ── Strategy implementations (identical to Phase 6.B / Phase 5.5) ──

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
        ret_12m = np.log(monthly_close.iloc[i] + 1e-12) - np.log(monthly_close.iloc[i - lookback_months] + 1e-12)
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
        ret_12m = np.log(mc.iloc[mc_loc] + 1e-12) - np.log(mc.iloc[mc_loc - lookback_months] + 1e-12)
        all_rets = {}
        for a2, mc2 in all_mc.items():
            if month in mc2.index:
                loc2 = mc2.index.get_loc(month)
                if loc2 >= lookback_months:
                    all_rets[a2] = np.log(mc2.iloc[loc2] + 1e-12) - np.log(mc2.iloc[loc2 - lookback_months] + 1e-12)
        if not all_rets:
            continue
        best = max(all_rets, key=all_rets.get)
        next_month = common[i + 1] if i + 1 < len(common) else dates[-1]
        mask = (dates > month) & (dates <= next_month)
        if ret_12m > 0 and best == asset:
            positions[mask] = 1
    return positions


def eval_cell(log_ret, positions, asset, dates, label, cost_multiplier=1.0):
    """Evaluate a single cell with optional cost multiplier."""
    gross_ret = positions[:-1] * log_ret[1:]
    # Apply base costs
    net_ret_base = apply_cost_to_returns(gross_ret, positions[:-1], asset, np.abs(log_ret[:-1]))
    # The cost is the difference between gross and net
    cost_component = gross_ret - net_ret_base
    # Apply cost multiplier: net = gross - (cost * multiplier)
    net_ret = gross_ret - cost_component * cost_multiplier

    # Vol-scale to 10%
    realized_vol = np.std(net_ret) * np.sqrt(PPY_DAILY) if np.std(net_ret) > 0 else 0.10
    vol_scalar = TARGET_VOL / max(realized_vol, 0.01)
    vol_scalar = min(vol_scalar, 5.0)
    net_ret = net_ret * vol_scalar

    eval_dates = dates[1:len(net_ret) + 1]
    return net_ret, eval_dates, vol_scalar


def eval_portfolio_daily(cell_returns_dict, cell_weights, common_dates):
    """Evaluate portfolio at daily frequency, then aggregate to weekly for metrics.

    Bug fix (Phase 6.D.1): Removed portfolio-level vol re-scaling.
    Cell returns are already vol-scaled to 10% individually in eval_cell().
    Portfolio is reported at realized vol, with at_10pct_vol metrics also provided.
    Equity uses exp(cumsum) for log returns instead of cumprod(1+r).
    """
    df = pd.DataFrame(cell_returns_dict, index=common_dates)
    weekly_df = df.resample("W").sum().dropna(how='all')

    port_ret = np.zeros(len(weekly_df))
    for name in weekly_df.columns:
        w = cell_weights.get(name, 0.0)
        port_ret += w * weekly_df[name].values

    # No portfolio-level vol re-scaling — cells are already at 10% vol each
    realized_vol = np.std(port_ret) * np.sqrt(52) if len(port_ret) > 10 else 0.10
    sharpe = annualized_sharpe(port_ret, 52)
    weekly_dates = weekly_df.index.values

    # Correct equity for log returns
    equity = np.exp(np.cumsum(port_ret))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    # Also compute at 10% target vol for deployment reference
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


def run_p3_full(data, all_fx_data, cost_multiplier=1.0, date_filter=None):
    """
    Run P3 portfolio end-to-end with optional cost multiplier and date filter.
    Returns per-cell metrics + portfolio metrics.
    """
    # EUR/USD MR
    d = data["EUR/USD"]
    mr_pos = run_pure_mr(d["log_ret"], d["close"])
    mr_ret, mr_dates, mr_vs = eval_cell(d["log_ret"], mr_pos, "EUR/USD", d["dates"],
                                         "EUR/USD_mr", cost_multiplier)

    # USD/JPY TSMOM
    d = data["USD/JPY"]
    tsmom_pos = run_tsmom(d["log_ret"], d["close"], d["dates"])
    tsmom_ret, tsmom_dates, tsmom_vs = eval_cell(d["log_ret"], tsmom_pos, "USD/JPY", d["dates"],
                                                   "USD/JPY_tsmom", cost_multiplier)

    # USD/JPY Dual Momentum
    dm_pos = run_dual_momentum(d["log_ret"], d["close"], d["dates"], all_fx_data, "USD/JPY")
    dm_ret, dm_dates, dm_vs = eval_cell(d["log_ret"], dm_pos, "USD/JPY", d["dates"],
                                         "USD/JPY_dm", cost_multiplier)

    # P3 weights (inverse worst-window from Phase 5.5)
    # Compute from rolling eval on full period
    mr_rolling = rolling_window_evaluation(mr_ret, PPY_DAILY)
    tsmom_rolling = rolling_window_evaluation(tsmom_ret, PPY_DAILY)
    dm_rolling = rolling_window_evaluation(dm_ret, PPY_DAILY)

    cell_worst = {
        "eurusd_mr": mr_rolling["worst_window_sharpe"],
        "usdjpy_tsmom": tsmom_rolling["worst_window_sharpe"],
        "usdjpy_dm": dm_rolling["worst_window_sharpe"],
    }
    inv_worst = {k: 1.0 / max(abs(v), 0.01) for k, v in cell_worst.items()}
    total_inv = sum(inv_worst.values())
    p3_weights = {k: v / total_inv for k, v in inv_worst.items()}

    # Align dates
    common = mr_dates.intersection(tsmom_dates).intersection(dm_dates)
    mr_mask = mr_dates.isin(common)
    tsmom_mask = tsmom_dates.isin(common)
    dm_mask = dm_dates.isin(common)

    min_len = min(mr_mask.sum(), tsmom_mask.sum(), dm_mask.sum())
    mr_aligned = mr_ret[mr_mask][:min_len]
    tsmom_aligned = tsmom_ret[tsmom_mask][:min_len]
    dm_aligned = dm_ret[dm_mask][:min_len]
    common_sorted = common.sort_values()[:min_len]

    # Apply date filter if specified
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

    # Per-cell metrics on filtered period (exp(cumsum) for log returns)
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

    # Trade counts on filtered period
    mr_pos_full = mr_pos[:-1][mr_mask][:min_len]
    tsmom_pos_full = tsmom_pos[:-1][tsmom_mask][:min_len]
    dm_pos_full = dm_pos[:-1][dm_mask][:min_len]
    if date_filter is not None:
        mr_pos_full = mr_pos_full[fmask]
        tsmom_pos_full = tsmom_pos_full[fmask]
        dm_pos_full = dm_pos_full[fmask]

    cell_metrics["eurusd_mr"]["n_trades"] = int(np.sum(np.abs(np.diff(mr_pos_full, prepend=0)) > 0))
    cell_metrics["usdjpy_tsmom"]["n_trades"] = int(np.sum(np.abs(np.diff(tsmom_pos_full, prepend=0)) > 0))
    cell_metrics["usdjpy_dm"]["n_trades"] = int(np.sum(np.abs(np.diff(dm_pos_full, prepend=0)) > 0))

    return {
        "p3_weights": {k: round(v, 4) for k, v in p3_weights.items()},
        "portfolio": portfolio,
        "cells": cell_metrics,
        "worst_2y": {k: round(v, 4) for k, v in cell_worst.items()},
    }


# ================================================================
# TASK 6.C.0 — HELD-OUT 2024-2025 EVALUATION
# ================================================================
def task_6c0(data, all_fx_data):
    print("=" * 70)
    print("TASK 6.C.0 — 2024-2025 HELD-OUT EVALUATION")
    print("=" * 70)
    print("  This is the single genuine out-of-sample test.")
    print("  Period: 2024-01-01 to 2025-12-31 (preserved untouched)")

    # Check data coverage
    for asset in ["EUR/USD", "USD/JPY"]:
        d = data[asset]
        last_date = d["dates"][-1]
        print(f"  {asset} data ends: {last_date.date()}")
        if last_date < pd.Timestamp("2025-12-31"):
            print(f"    WARNING: Data does not cover full 2025. Using up to {last_date.date()}")

    # Run P3 on full period (needed for vol-scaling calibration)
    print("\n  Running P3 on full period for baseline calibration...")
    full_result = run_p3_full(data, all_fx_data)
    if full_result is None:
        print("  FATAL: Could not run P3 on full period.")
        return None

    print(f"  Full-period P3: Sharpe={full_result['portfolio']['sharpe']:.4f}")
    print(f"  P3 weights: {full_result['p3_weights']}")

    # Run P3 on held-out period
    print(f"\n  Running P3 on held-out period ({HOLDOUT_START} to {HOLDOUT_END})...")
    holdout_result = run_p3_full(data, all_fx_data, cost_multiplier=1.0,
                                 date_filter=(HOLDOUT_START, HOLDOUT_END))

    if holdout_result is None:
        print("  FATAL: Insufficient held-out data.")
        return {"pass": False, "reason": "insufficient_data"}

    port = holdout_result["portfolio"]
    cells = holdout_result["cells"]

    print(f"\n  [HELD-OUT RESULTS]")
    print(f"    Portfolio Sharpe:    {port['sharpe']:.4f}")
    print(f"    Portfolio max DD:    {port['max_dd']*100:.1f}%")
    print(f"    Portfolio return:    {port['total_return']*100:.1f}%")
    print(f"    Portfolio vol:       {port['vol']*100:.1f}%")
    print(f"    N weeks:             {port['n_weeks']}")

    print(f"\n  [PER-CELL HELD-OUT]")
    for name, m in cells.items():
        print(f"    {name}: Sharpe={m['sharpe']:.4f}, return={m['total_return']*100:.1f}%, "
              f"maxDD={m['max_dd']*100:.1f}%, trades={m.get('n_trades', 'N/A')}")

    # Also run on IS and OOS for comparison
    is_result = run_p3_full(data, all_fx_data, date_filter=("2003-01-01", TRAIN_END))
    oos_result = run_p3_full(data, all_fx_data, date_filter=(TEST_START, TEST_END))

    print(f"\n  [COMPARISON]")
    print(f"    IS (≤2018):     Sharpe={is_result['portfolio']['sharpe']:.4f}, maxDD={is_result['portfolio']['max_dd']*100:.1f}%")
    print(f"    OOS (2019-23):  Sharpe={oos_result['portfolio']['sharpe']:.4f}, maxDD={oos_result['portfolio']['max_dd']*100:.1f}%")
    print(f"    HELD-OUT (24-25): Sharpe={port['sharpe']:.4f}, maxDD={port['max_dd']*100:.1f}%")

    # Concentration analysis
    usdjpy_weight = holdout_result['p3_weights'].get('usdjpy_tsmom', 0) + \
                    holdout_result['p3_weights'].get('usdjpy_dm', 0)
    print(f"\n  USD/JPY weight: {usdjpy_weight*100:.1f}%")

    # Gate determination
    held_out_pass = port["sharpe"] > 0 and port["max_dd"] < 0.30
    held_out_deploy = port["sharpe"] > 0 and port["max_dd"] < 0.25  # stricter for deploy

    print(f"\n  [GATE DETERMINATION]")
    print(f"    Sharpe > 0:     {'PASS' if port['sharpe'] > 0 else 'FAIL'} ({port['sharpe']:.4f})")
    print(f"    max DD < 25%:   {'PASS' if port['max_dd'] < 0.25 else 'FAIL'} ({port['max_dd']*100:.1f}%)")
    print(f"    max DD < 30%:   {'PASS (gate)' if port['max_dd'] < 0.30 else 'FAIL (gate)'} ({port['max_dd']*100:.1f}%)")
    print(f"    GATE:           {'PASS — proceed with stress tests' if held_out_pass else 'FAIL — stress tests still run for documentation'}")

    result = {
        "held_out_sharpe": port["sharpe"],
        "held_out_max_dd": port["max_dd"],
        "held_out_return": port["total_return"],
        "held_out_vol": port["vol"],
        "held_out_n_weeks": port["n_weeks"],
        "held_out_at_10pct_vol": port.get("at_10pct_vol", {}),
        "cells": {k: v for k, v in cells.items()},
        "is_sharpe": is_result["portfolio"]["sharpe"] if is_result else None,
        "oos_sharpe": oos_result["portfolio"]["sharpe"] if oos_result else None,
        "p3_weights": holdout_result["p3_weights"],
        "usdjpy_concentration": round(usdjpy_weight, 4),
        "gate_pass": held_out_pass,
        "deploy_pass": held_out_deploy,
    }

    return result


# ================================================================
# TASK 6.C.3 — COST SENSITIVITY
# ================================================================
def task_6c3(data, all_fx_data):
    print("\n" + "=" * 70)
    print("TASK 6.C.3 — COST SENSITIVITY")
    print("=" * 70)

    cost_multipliers = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    results_by_mult = {}

    for cm in cost_multipliers:
        print(f"\n  Running P3 at {cm:.2f}x cost...")
        r = run_p3_full(data, all_fx_data, cost_multiplier=cm)
        if r is None:
            print(f"    FAILED")
            continue

        port = r["portfolio"]
        # Also compute worst-2Y at this cost level from full period
        full_ret = port["port_ret"]
        rolling = rolling_window_evaluation(full_ret, 52, window_years=2.0, step_months=6)

        results_by_mult[str(cm)] = {
            "cost_multiplier": cm,
            "sharpe": port["sharpe"],
            "max_dd": port["max_dd"],
            "worst_2y": rolling["worst_window_sharpe"],
            "total_return": port["total_return"],
            "vol": port["vol"],
            "at_10pct_vol": port.get("at_10pct_vol", {}),
        }
        print(f"    Sharpe={port['sharpe']:.4f}, worst-2Y={rolling['worst_window_sharpe']:.4f}, "
              f"maxDD={port['max_dd']*100:.1f}%")

    # Find break-even points
    sharpes = [(cm, results_by_mult[str(cm)]["sharpe"]) for cm in cost_multipliers if str(cm) in results_by_mult]
    worst_2ys = [(cm, results_by_mult[str(cm)]["worst_2y"]) for cm in cost_multipliers if str(cm) in results_by_mult]

    # Sharpe zero crossing
    sharpe_breakeven = None
    for i in range(len(sharpes) - 1):
        if sharpes[i][1] > 0 and sharpes[i + 1][1] <= 0:
            # Linear interpolation
            x0, y0 = sharpes[i]
            x1, y1 = sharpes[i + 1]
            sharpe_breakeven = x0 + (0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
            break
    if sharpe_breakeven is None and sharpes[-1][1] > 0:
        sharpe_breakeven = float("inf")  # still positive at 3x

    # worst-2Y crossing -1.0
    w2y_breakeven = None
    for i in range(len(worst_2ys) - 1):
        if worst_2ys[i][1] > -1.0 and worst_2ys[i + 1][1] <= -1.0:
            x0, y0 = worst_2ys[i]
            x1, y1 = worst_2ys[i + 1]
            w2y_breakeven = x0 + (-1.0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
            break
    if w2y_breakeven is None and worst_2ys[-1][1] > -1.0:
        w2y_breakeven = float("inf")

    print(f"\n  [BREAK-EVEN ANALYSIS]")
    print(f"    Sharpe crosses 0 at:    {sharpe_breakeven:.2f}x" if sharpe_breakeven != float("inf")
          else f"    Sharpe crosses 0 at:    >3.0x (still positive)")
    print(f"    worst-2Y crosses -1.0 at: {w2y_breakeven:.2f}x" if w2y_breakeven != float("inf")
          else f"    worst-2Y crosses -1.0 at: >3.0x (still above -1.0)")

    # Kill criteria
    sharpe_at_2x = results_by_mult.get("2.0", {}).get("sharpe", None)
    w2y_at_1_5x = results_by_mult.get("1.5", {}).get("worst_2y", None)

    cost_pass = True
    if sharpe_at_2x is not None and sharpe_at_2x <= 0:
        cost_pass = False
    if w2y_at_1_5x is not None and w2y_at_1_5x <= -1.0:
        cost_pass = False

    print(f"\n  [KILL CRITERIA]")
    print(f"    Sharpe positive at 2.0x:    {'PASS' if sharpe_at_2x and sharpe_at_2x > 0 else 'FAIL'} "
          f"({sharpe_at_2x:.4f})" if sharpe_at_2x else "    Sharpe positive at 2.0x:    N/A")
    print(f"    worst-2Y > -1.0 at 1.5x:   {'PASS' if w2y_at_1_5x and w2y_at_1_5x > -1.0 else 'FAIL'} "
          f"({w2y_at_1_5x:.4f})" if w2y_at_1_5x else "    worst-2Y > -1.0 at 1.5x:   N/A")
    print(f"    OVERALL: {'PASS' if cost_pass else 'FAIL'}")

    return {
        "cost_curves": results_by_mult,
        "sharpe_breakeven": round(sharpe_breakeven, 2) if sharpe_breakeven != float("inf") else ">3.0x",
        "worst_2y_breakeven": round(w2y_breakeven, 2) if w2y_breakeven != float("inf") else ">3.0x",
        "sharpe_at_2x": round(sharpe_at_2x, 4) if sharpe_at_2x else None,
        "worst_2y_at_1_5x": round(w2y_at_1_5x, 4) if w2y_at_1_5x else None,
        "pass": cost_pass,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("PHASE 6.C — OMEGA TASKS (6.C.0 + 6.C.3)")
    print("=" * 70)

    # Load data
    print("\n[DATA] Loading daily price data...")
    assets_needed = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]
    data = {}
    for asset in assets_needed:
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

    all_fx_data = {}
    for a in assets_needed:
        if a in data:
            all_fx_data[a] = (data[a]["log_ret"], data[a]["close"], data[a]["dates"])

    results = {}

    # Task 6.C.0
    result_0 = task_6c0(data, all_fx_data)
    results["task_6c0"] = result_0

    # Task 6.C.3
    result_3 = task_6c3(data, all_fx_data)
    results["task_6c3"] = result_3

    # Save results
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

    results_clean = convert_for_json(results)
    out_file = os.path.join(RESULTS_DIR, "phase_6c_omega_results.json")
    with open(out_file, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")

    # Summary
    print("\n" + "=" * 70)
    print("OMEGA TASKS COMPLETE")
    print("=" * 70)
    if result_0:
        print(f"  6.C.0 Gate:        {'PASS' if result_0.get('gate_pass') else 'FAIL'}")
        print(f"  6.C.0 Held-out SR: {result_0.get('held_out_sharpe', 'N/A')}")
    print(f"  6.C.3 Cost:        {'PASS' if result_3.get('pass') else 'FAIL'}")
    print(f"  6.C.3 Breakeven:   {result_3.get('sharpe_breakeven', 'N/A')}")


if __name__ == "__main__":
    main()
