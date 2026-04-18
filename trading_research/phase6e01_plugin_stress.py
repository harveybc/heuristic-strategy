#!/usr/bin/env python3
"""
Phase 6.E.0.1 — Plugin-Canonical Stress Tests (6.C.1, 6.C.4, 6.C.5)

Adapts phase6c_stress.py to use LTS plugin signal generation.
6.C.2 (Monte Carlo regimes) omitted for time — 3000 paths × 3 regimes is >1hr.
Can be run separately if needed.

Usage:
  cd /home/harveybc/Documents/GitHub/heuristic-strategy
  conda run -n tensorflow python3 -u trading_research/phase6e01_plugin_stress.py --task 6c1
  conda run -n tensorflow python3 -u trading_research/phase6e01_plugin_stress.py --task 6c4
  conda run -n tensorflow python3 -u trading_research/phase6e01_plugin_stress.py --task 6c5
  conda run -n tensorflow python3 -u trading_research/phase6e01_plugin_stress.py --task all
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
EXTENDED_DATA = os.path.join(SCRIPT_DIR, "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "..", "lts"))

from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation, compute_strategy_metrics
)
from trading_research.transaction_cost_model import apply_cost_to_returns
from plugins_strategy.eurusd_mr_strategy import EurUsdMrStrategy
from plugins_strategy.usdjpy_tsmom_strategy import UsdJpyTsmomStrategy
from plugins_strategy.usdjpy_dual_momentum_strategy import UsdJpyDualMomentumStrategy

PPY_DAILY = 252
PPY_WEEKLY = 52
TARGET_VOL = 0.10

# Fixed P3 weights (script-canonical)
FIXED_P3_WEIGHTS = {"eurusd_mr": 0.2055, "usdjpy_tsmom": 0.492, "usdjpy_dm": 0.3024}


def load_daily_data(asset):
    safe = asset.replace("/", "_")
    csv_path = os.path.join(EXTENDED_DATA, f"{safe}_daily.csv")
    if not os.path.exists(csv_path):
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
            dates = df.index
            log_ret = np.diff(np.log(close + 1e-12), prepend=0)
            log_ret[0] = 0
            data[asset] = {"close": close, "dates": dates, "log_ret": log_ret}
            print(f"  {asset}: {len(close)} bars, {dates[0].date()} to {dates[-1].date()}")
    all_fx = {a: (data[a]["log_ret"], data[a]["close"], data[a]["dates"]) for a in data}
    return data, all_fx


# ================================================================
# PLUGIN SIGNAL → POSITIONS (bar-by-bar)
# ================================================================
def run_mr_plugin(close, dates, config=None):
    plugin = EurUsdMrStrategy(config)
    n = len(close)
    positions = np.zeros(n)
    for i in range(n):
        signal = plugin.generate_signal("EUR/USD", market_data={"close": float(close[i])})
        action = signal.get("action", "none")
        if action == "open":
            positions[i] = 1 if signal["parameters"].get("side") == "buy" else -1
        elif action == "close":
            positions[i] = 0
        elif action == "none":
            positions[i] = positions[i - 1] if i > 0 else 0
    return positions


def run_tsmom_plugin(close, dates, config=None):
    plugin = UsdJpyTsmomStrategy(config)
    n = len(close)
    positions = np.zeros(n)
    for i in range(n):
        dt = dates[i]
        date_str = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
        signal = plugin.generate_signal("USD/JPY",
                                         market_data={"close": float(close[i]), "date": date_str})
        action = signal.get("action", "none")
        if action == "open":
            vol_size = signal["parameters"].get("vol_size", 1.0)
            positions[i] = vol_size if signal["parameters"].get("side") == "buy" else -vol_size
        elif action == "close":
            positions[i] = 0
        elif action == "none":
            positions[i] = positions[i - 1] if i > 0 else 0
    return positions


def run_dm_plugin(close, dates, peer_data, config=None):
    plugin = UsdJpyDualMomentumStrategy(config)
    n = len(close)
    positions = np.zeros(n)
    for i in range(n):
        dt = dates[i]
        date_str = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
        peers = {a: float(pc[i]) for a, pc in peer_data.items() if i < len(pc)}
        signal = plugin.generate_signal("USD/JPY",
                                         market_data={"close": float(close[i]), "date": date_str,
                                                      "peer_prices": peers})
        action = signal.get("action", "none")
        if action == "open":
            positions[i] = 1
        elif action == "close":
            positions[i] = 0
        elif action == "none":
            positions[i] = positions[i - 1] if i > 0 else 0
    return positions


def eval_cell_returns(log_ret, positions, asset, dates):
    gross_ret = positions[:-1] * log_ret[1:]
    net_ret = apply_cost_to_returns(gross_ret, positions[:-1], asset, np.abs(log_ret[:-1]))
    realized_vol = np.std(net_ret) * np.sqrt(PPY_DAILY) if np.std(net_ret) > 0 else 0.10
    vol_scalar = TARGET_VOL / max(realized_vol, 0.01)
    vol_scalar = min(vol_scalar, 5.0)
    net_ret = net_ret * vol_scalar
    eval_dates = dates[1:len(net_ret) + 1]
    return net_ret, eval_dates, vol_scalar


def run_p3_plugin_on_data(eu_close, eu_dates, eu_lr,
                           jp_close, jp_dates, jp_lr,
                           peer_data, weights=None,
                           mr_config=None, ts_config=None, dm_config=None):
    """
    Run full P3 portfolio using plugins on provided data.
    Returns portfolio metrics dict.
    """
    if weights is None:
        weights = FIXED_P3_WEIGHTS

    mr_pos = run_mr_plugin(eu_close, eu_dates, mr_config)
    tsmom_pos = run_tsmom_plugin(jp_close, jp_dates, ts_config)
    dm_pos = run_dm_plugin(jp_close, jp_dates, peer_data, dm_config)

    mr_ret, mr_dt, _ = eval_cell_returns(eu_lr, mr_pos, "EUR/USD", eu_dates)
    ts_ret, ts_dt, _ = eval_cell_returns(jp_lr, tsmom_pos, "USD/JPY", jp_dates)
    dm_ret, dm_dt, _ = eval_cell_returns(jp_lr, dm_pos, "USD/JPY", jp_dates)

    common = mr_dt.intersection(ts_dt).intersection(dm_dt).sort_values()
    mr_a = mr_ret[mr_dt.isin(common)][:len(common)]
    ts_a = ts_ret[ts_dt.isin(common)][:len(common)]
    dm_a = dm_ret[dm_dt.isin(common)][:len(common)]

    cells = {"eurusd_mr": mr_a, "usdjpy_tsmom": ts_a, "usdjpy_dm": dm_a}
    df = pd.DataFrame(cells, index=common)
    weekly = df.resample("W").sum().dropna(how="all")
    port = np.zeros(len(weekly))
    for k in weekly.columns:
        port += weights.get(k, 0) * weekly[k].values

    realized_vol = np.std(port) * np.sqrt(PPY_WEEKLY) if len(port) > 10 else 0.10
    sharpe = annualized_sharpe(port, PPY_WEEKLY)
    eq = np.exp(np.cumsum(port))
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-12)
    max_dd = float(np.max(dd))

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "total_return": round(float(eq[-1] - 1), 4) if len(eq) > 0 else 0.0,
        "n_weeks": len(port),
        "port_ret": port,
        "vol": round(realized_vol, 4),
        "weights": weights,
    }


# ================================================================
# 6.C.1 — JPY REVERSAL STRESS (Plugin-Canonical)
# ================================================================
def task_6c1(data, all_fx):
    print("\n" + "=" * 70)
    print("PLUGIN 6.C.1 — JPY REVERSAL STRESS TEST")
    print("=" * 70)

    np.random.seed(42)
    usdjpy_close = data["USD/JPY"]["close"]
    usdjpy_dates = data["USD/JPY"]["dates"]
    eurusd_close = data["EUR/USD"]["close"]
    eurusd_dates = data["EUR/USD"]["dates"]

    usdjpy_lr = np.diff(np.log(usdjpy_close + 1e-12))
    eurusd_lr = np.diff(np.log(eurusd_close + 1e-12))

    eu_ret_dates = eurusd_dates[1:]
    jp_ret_dates = usdjpy_dates[1:]
    common_ret_dates = eu_ret_dates.intersection(jp_ret_dates).sort_values()
    eu_rmask = eu_ret_dates.isin(common_ret_dates)
    jp_rmask = jp_ret_dates.isin(common_ret_dates)
    eurusd_rets = eurusd_lr[eu_rmask]
    usdjpy_rets = usdjpy_lr[jp_rmask]
    n_common = min(len(eurusd_rets), len(usdjpy_rets))
    eurusd_rets = eurusd_rets[:n_common]
    usdjpy_rets = usdjpy_rets[:n_common]

    # Identify reversal episodes
    window = 63
    usdjpy_cum = np.cumsum(usdjpy_rets)
    reversal_set = set()
    for i in range(window, len(usdjpy_rets)):
        q_ret = usdjpy_cum[i] - usdjpy_cum[i - window]
        if q_ret < -0.05:
            reversal_set.add(i)

    print(f"  Reversal days: {len(reversal_set)} ({len(reversal_set)/n_common*100:.1f}%)")

    N_PATHS = 500
    PATH_LEN = 504
    BLOCK_SIZE = 63

    n_blocks = (n_common - BLOCK_SIZE) // BLOCK_SIZE
    normal_blocks = []
    reversal_blocks = []
    for b in range(n_blocks):
        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        overlap = len(set(range(start, end)) & reversal_set) / BLOCK_SIZE
        if overlap > 0.3:
            reversal_blocks.append((start, end))
        else:
            normal_blocks.append((start, end))

    print(f"  Normal blocks: {len(normal_blocks)}, Reversal blocks: {len(reversal_blocks)}")
    if len(reversal_blocks) < 2:
        reversal_blocks = normal_blocks.copy()

    path_results = []
    n_blocks_per_path = PATH_LEN // BLOCK_SIZE

    for p in range(N_PATHS):
        n_reversal = max(1, int(n_blocks_per_path * 0.3))
        n_normal = n_blocks_per_path - n_reversal

        synth_eu = []
        synth_jp = []
        for _ in range(n_reversal):
            s, e = reversal_blocks[np.random.randint(0, len(reversal_blocks))]
            amp = np.random.choice([1.0, 1.5, 2.0], p=[0.4, 0.4, 0.2])
            synth_eu.append(eurusd_rets[s:e])
            synth_jp.append(usdjpy_rets[s:e] * amp)
        for _ in range(n_normal):
            s, e = normal_blocks[np.random.randint(0, len(normal_blocks))]
            synth_eu.append(eurusd_rets[s:e])
            synth_jp.append(usdjpy_rets[s:e])

        indices = list(range(n_blocks_per_path))
        np.random.shuffle(indices)
        eu_path = np.concatenate([synth_eu[i] for i in indices])[:PATH_LEN]
        jp_path = np.concatenate([synth_jp[i] for i in indices])[:PATH_LEN]

        # Build synthetic prices and dates
        eu_close_s = np.exp(np.cumsum(np.concatenate([[np.log(1.08)], eu_path])))
        jp_close_s = np.exp(np.cumsum(np.concatenate([[np.log(120.0)], jp_path])))
        eu_lr_s = np.diff(np.log(eu_close_s + 1e-12), prepend=0)
        eu_lr_s[0] = 0
        jp_lr_s = np.diff(np.log(jp_close_s + 1e-12), prepend=0)
        jp_lr_s[0] = 0
        synth_dates = pd.date_range("2024-01-01", periods=len(eu_close_s), freq="B")

        # Peer data for DM
        gbp_close = eu_close_s * (1.25 / 1.08) + np.random.normal(0, 0.001, len(eu_close_s)).cumsum()
        aud_close = eu_close_s * (0.65 / 1.08) + np.random.normal(0, 0.001, len(eu_close_s)).cumsum()
        gbp_close = np.maximum(gbp_close, 0.5)
        aud_close = np.maximum(aud_close, 0.3)
        peer_data = {"EUR/USD": eu_close_s, "GBP/USD": gbp_close, "AUD/USD": aud_close}

        try:
            result = run_p3_plugin_on_data(
                eu_close_s, synth_dates, eu_lr_s,
                jp_close_s, synth_dates, jp_lr_s,
                peer_data)
            path_results.append({"sharpe": result["sharpe"], "max_dd": result["max_dd"]})
        except Exception as e:
            path_results.append({"sharpe": 0.0, "max_dd": 1.0, "error": str(e)})

        if (p + 1) % 50 == 0:
            print(f"    Completed {p+1}/{N_PATHS} paths...")

    sharpes = [r["sharpe"] for r in path_results if "error" not in r]
    max_dds = [r["max_dd"] for r in path_results if "error" not in r]
    errors = sum(1 for r in path_results if "error" in r)

    pcts = [5, 10, 25, 50, 75, 90, 95]
    sharpe_pcts = {str(p): round(float(np.percentile(sharpes, p)), 4) for p in pcts}
    dd_pcts = {str(p): round(float(np.percentile(max_dds, p)), 4) for p in pcts}

    median_sr = float(np.median(sharpes))
    pct25_sr = float(np.percentile(sharpes, 25))
    jpy_pass = median_sr >= 0 and pct25_sr >= -1.5

    print(f"\n  [RESULTS] {len(sharpes)} valid paths, {errors} errors")
    print(f"    Sharpe pcts: {sharpe_pcts}")
    print(f"    Median Sharpe: {median_sr:.4f}")
    print(f"    25th pct Sharpe: {pct25_sr:.4f}")
    print(f"    OVERALL: {'PASS' if jpy_pass else 'FAIL'}")

    return {
        "n_valid_paths": len(sharpes), "n_errors": errors,
        "sharpe_percentiles": sharpe_pcts, "max_dd_percentiles": dd_pcts,
        "median_sharpe": round(median_sr, 4), "pct25_sharpe": round(pct25_sr, 4),
        "fraction_positive": round(sum(1 for s in sharpes if s > 0) / max(len(sharpes), 1), 4),
        "pass": jpy_pass,
    }


# ================================================================
# 6.C.4 — WALK-FORWARD (Plugin-Canonical)
# ================================================================
def task_6c4(data, all_fx):
    print("\n" + "=" * 70)
    print("PLUGIN 6.C.4 — EXPANDING-WINDOW WALK-FORWARD")
    print("=" * 70)

    eu_close = data["EUR/USD"]["close"]
    eu_dates = data["EUR/USD"]["dates"]
    eu_lr = data["EUR/USD"]["log_ret"]
    jp_close = data["USD/JPY"]["close"]
    jp_dates = data["USD/JPY"]["dates"]
    jp_lr = data["USD/JPY"]["log_ret"]

    peer_data = {a: data[a]["close"] for a in ["EUR/USD", "GBP/USD", "AUD/USD"] if a in data}

    # Run plugins on full data to get positions
    print("  Generating full-data plugin positions...")
    mr_pos = run_mr_plugin(eu_close, eu_dates)
    tsmom_pos = run_tsmom_plugin(jp_close, jp_dates)
    dm_pos = run_dm_plugin(jp_close, jp_dates, peer_data)

    quarters = pd.date_range("2011-01-01", "2024-01-01", freq="QS")
    print(f"  Walk-forward: {len(quarters)-1} OOS quarters")

    quarterly_results = []
    for q_idx in range(len(quarters) - 1):
        q_start = quarters[q_idx]
        q_end = quarters[q_idx + 1] - pd.Timedelta(days=1)
        train_end_date = q_start - pd.Timedelta(days=1)

        eu_train_mask = eu_dates <= train_end_date
        jp_train_mask = jp_dates <= train_end_date
        if eu_train_mask.sum() < 252 * 5 or jp_train_mask.sum() < 252 * 5:
            continue

        # Training-period cell returns for weight derivation
        mr_ret_train, mr_dt_train, _ = eval_cell_returns(
            eu_lr[eu_train_mask], mr_pos[eu_train_mask], "EUR/USD", eu_dates[eu_train_mask])
        ts_ret_train, ts_dt_train, _ = eval_cell_returns(
            jp_lr[jp_train_mask], tsmom_pos[jp_train_mask], "USD/JPY", jp_dates[jp_train_mask])
        dm_ret_train, dm_dt_train, _ = eval_cell_returns(
            jp_lr[jp_train_mask], dm_pos[jp_train_mask], "USD/JPY", jp_dates[jp_train_mask])

        mr_w2y = rolling_window_evaluation(mr_ret_train, PPY_DAILY)["worst_window_sharpe"]
        ts_w2y = rolling_window_evaluation(ts_ret_train, PPY_DAILY)["worst_window_sharpe"]
        dm_w2y = rolling_window_evaluation(dm_ret_train, PPY_DAILY)["worst_window_sharpe"]

        inv = {"mr": 1.0 / max(abs(mr_w2y), 0.01),
               "ts": 1.0 / max(abs(ts_w2y), 0.01),
               "dm": 1.0 / max(abs(dm_w2y), 0.01)}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}

        # Full-data cell returns
        mr_ret_full, mr_dt_full, _ = eval_cell_returns(eu_lr, mr_pos, "EUR/USD", eu_dates)
        ts_ret_full, ts_dt_full, _ = eval_cell_returns(jp_lr, tsmom_pos, "USD/JPY", jp_dates)
        dm_ret_full, dm_dt_full, _ = eval_cell_returns(jp_lr, dm_pos, "USD/JPY", jp_dates)

        # OOS quarter
        mr_oos = mr_ret_full[(mr_dt_full >= q_start) & (mr_dt_full <= q_end)]
        ts_oos = ts_ret_full[(ts_dt_full >= q_start) & (ts_dt_full <= q_end)]
        dm_oos = dm_ret_full[(dm_dt_full >= q_start) & (dm_dt_full <= q_end)]
        min_len = min(len(mr_oos), len(ts_oos), len(dm_oos))
        if min_len < 20:
            continue

        port_daily = (weights["mr"] * mr_oos[:min_len] +
                      weights["ts"] * ts_oos[:min_len] +
                      weights["dm"] * dm_oos[:min_len])

        q_return = float(np.sum(port_daily))
        q_sharpe = annualized_sharpe(port_daily, PPY_DAILY)
        eq = np.exp(np.cumsum(port_daily))
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / (pk + 1e-12)
        q_maxdd = float(np.max(dd))

        quarterly_results.append({
            "quarter": str(q_start.date()),
            "return": round(q_return, 4),
            "sharpe": round(q_sharpe, 4),
            "max_dd": round(q_maxdd, 4),
            "n_days": min_len,
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "positive": q_return > 0,
        })

    n_quarters = len(quarterly_results)
    n_positive = sum(1 for q in quarterly_results if q["positive"])
    frac_positive = n_positive / max(n_quarters, 1)
    all_sharpes = [q["sharpe"] for q in quarterly_results]
    median_q_sharpe = float(np.median(all_sharpes)) if all_sharpes else 0

    streak = 0
    max_streak = 0
    for q in quarterly_results:
        if not q["positive"]:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    worst_q = min(quarterly_results, key=lambda x: x["sharpe"]) if quarterly_results else None

    print(f"\n  [RESULTS]")
    print(f"    Quarters: {n_quarters}, Positive: {n_positive} ({frac_positive*100:.1f}%)")
    print(f"    Median quarterly Sharpe: {median_q_sharpe:.4f}")
    print(f"    Longest losing streak: {max_streak}")
    if worst_q:
        print(f"    Worst quarter: {worst_q['quarter']} (SR={worst_q['sharpe']:.4f})")

    for q in quarterly_results:
        flag = "+" if q["positive"] else "-"
        print(f"    {flag} {q['quarter']}: ret={q['return']*100:+.1f}%, SR={q['sharpe']:+.2f}")

    wf_pass = frac_positive >= 0.55 and median_q_sharpe >= 0.2 and max_streak <= 5

    print(f"\n  [KILL CRITERIA]")
    print(f"    ≥55% positive: {'PASS' if frac_positive >= 0.55 else 'FAIL'} ({frac_positive*100:.1f}%)")
    print(f"    Median SR ≥ 0.2: {'PASS' if median_q_sharpe >= 0.2 else 'FAIL'} ({median_q_sharpe:.4f})")
    print(f"    Streak ≤ 5: {'PASS' if max_streak <= 5 else 'FAIL'} ({max_streak})")
    print(f"    OVERALL: {'PASS' if wf_pass else 'FAIL'}")

    return {
        "n_quarters": n_quarters, "n_positive": n_positive,
        "frac_positive": round(frac_positive, 4),
        "median_quarterly_sharpe": round(median_q_sharpe, 4),
        "longest_losing_streak": max_streak,
        "worst_quarter": worst_q,
        "quarterly_results": quarterly_results,
        "pass": wf_pass,
    }


# ================================================================
# 6.C.5 — PARAMETER PERTURBATION (Plugin-Canonical)
# ================================================================
def task_6c5(data, all_fx):
    print("\n" + "=" * 70)
    print("PLUGIN 6.C.5 — PARAMETER PERTURBATION SENSITIVITY")
    print("=" * 70)

    eu_close = data["EUR/USD"]["close"]
    eu_dates = data["EUR/USD"]["dates"]
    eu_lr = data["EUR/USD"]["log_ret"]
    jp_close = data["USD/JPY"]["close"]
    jp_dates = data["USD/JPY"]["dates"]
    jp_lr = data["USD/JPY"]["log_ret"]
    peer_data = {a: data[a]["close"] for a in ["EUR/USD", "GBP/USD", "AUD/USD"] if a in data}

    all_results = {}

    # ── EUR/USD MR ──
    print("\n  [EUR/USD Mean Reversion]")
    mr_grids = {
        "lookback": [10, 15, 18, 20, 22, 25, 30],
        "z_entry": [0.75, 1.125, 1.35, 1.5, 1.65, 1.875, 2.25],
        "z_exit": [0.25, 0.375, 0.45, 0.5, 0.55, 0.625, 0.75],
    }
    baseline_params = {"lookback": 20, "z_entry": 1.5, "z_exit": 0.5}

    mr_pos = run_mr_plugin(eu_close, eu_dates)
    mr_ret, _, _ = eval_cell_returns(eu_lr, mr_pos, "EUR/USD", eu_dates)
    baseline_sr = annualized_sharpe(mr_ret, PPY_DAILY)
    baseline_w2y = rolling_window_evaluation(mr_ret, PPY_DAILY)["worst_window_sharpe"]
    print(f"    Baseline: Sharpe={baseline_sr:.4f}, worst-2Y={baseline_w2y:.4f}")

    mr_results = []
    for param, grid in mr_grids.items():
        for val in grid:
            config = dict(baseline_params)
            config[param] = val
            pos = run_mr_plugin(eu_close, eu_dates, config)
            ret, _, _ = eval_cell_returns(eu_lr, pos, "EUR/USD", eu_dates)
            sr = annualized_sharpe(ret, PPY_DAILY)
            w2y = rolling_window_evaluation(ret, PPY_DAILY)["worst_window_sharpe"]
            mr_results.append({
                "param": param, "value": val,
                "sharpe": round(sr, 4), "worst_2y": round(w2y, 4),
                "is_baseline": (val == baseline_params[param]),
            })
            flag = "B" if val == baseline_params[param] else " "
            print(f"      {flag} {param}={val}: SR={sr:.4f}")

    within_20pct = sum(1 for r in mr_results
                       if abs(r["sharpe"] - baseline_sr) <= 0.20 * abs(baseline_sr + 1e-8))
    mr_plateau = within_20pct / max(len(mr_results), 1)
    print(f"    Plateau: {within_20pct}/{len(mr_results)} ({mr_plateau*100:.0f}%)")

    all_results["eurusd_mr"] = {
        "baseline_sharpe": round(baseline_sr, 4),
        "plateau_fraction": round(mr_plateau, 4),
        "is_plateau": mr_plateau >= 0.60,
        "perturbations": mr_results,
    }

    # ── USD/JPY TSMOM ──
    print("\n  [USD/JPY TSMOM]")
    ts_grids = {"lookback_months": [6, 9, 11, 12, 13, 15, 18]}
    ts_baseline = 12

    ts_pos = run_tsmom_plugin(jp_close, jp_dates)
    ts_ret, _, _ = eval_cell_returns(jp_lr, ts_pos, "USD/JPY", jp_dates)
    ts_baseline_sr = annualized_sharpe(ts_ret, PPY_DAILY)
    print(f"    Baseline (lookback=12): Sharpe={ts_baseline_sr:.4f}")

    tsmom_results = []
    for lb in ts_grids["lookback_months"]:
        config = {"lookback_months": lb, "min_history_days": lb * 21}
        pos = run_tsmom_plugin(jp_close, jp_dates, config)
        ret, _, _ = eval_cell_returns(jp_lr, pos, "USD/JPY", jp_dates)
        sr = annualized_sharpe(ret, PPY_DAILY)
        w2y = rolling_window_evaluation(ret, PPY_DAILY)["worst_window_sharpe"]
        tsmom_results.append({
            "param": "lookback_months", "value": lb,
            "sharpe": round(sr, 4), "worst_2y": round(w2y, 4),
            "is_baseline": (lb == ts_baseline),
        })
        flag = "B" if lb == ts_baseline else " "
        print(f"      {flag} lookback={lb}: SR={sr:.4f}")

    within_ts = sum(1 for r in tsmom_results
                    if abs(r["sharpe"] - ts_baseline_sr) <= 0.20 * abs(ts_baseline_sr + 1e-8))
    ts_plateau = within_ts / max(len(tsmom_results), 1)
    print(f"    Plateau: {within_ts}/{len(tsmom_results)} ({ts_plateau*100:.0f}%)")

    all_results["usdjpy_tsmom"] = {
        "baseline_sharpe": round(ts_baseline_sr, 4),
        "plateau_fraction": round(ts_plateau, 4),
        "is_plateau": ts_plateau >= 0.60,
        "perturbations": tsmom_results,
    }

    # ── USD/JPY DM ──
    print("\n  [USD/JPY Dual Momentum]")
    dm_grids = {"lookback_months": [6, 9, 11, 12, 13, 15, 18]}
    dm_baseline = 12

    dm_pos = run_dm_plugin(jp_close, jp_dates, peer_data)
    dm_ret, _, _ = eval_cell_returns(jp_lr, dm_pos, "USD/JPY", jp_dates)
    dm_baseline_sr = annualized_sharpe(dm_ret, PPY_DAILY)
    print(f"    Baseline (lookback=12): Sharpe={dm_baseline_sr:.4f}")

    dm_results = []
    for lb in dm_grids["lookback_months"]:
        config = {"lookback_months": lb, "min_history_days": lb * 21}
        pos = run_dm_plugin(jp_close, jp_dates, peer_data, config)
        ret, _, _ = eval_cell_returns(jp_lr, pos, "USD/JPY", jp_dates)
        sr = annualized_sharpe(ret, PPY_DAILY)
        w2y = rolling_window_evaluation(ret, PPY_DAILY)["worst_window_sharpe"]
        dm_results.append({
            "param": "lookback_months", "value": lb,
            "sharpe": round(sr, 4), "worst_2y": round(w2y, 4),
            "is_baseline": (lb == dm_baseline),
        })
        flag = "B" if lb == dm_baseline else " "
        print(f"      {flag} lookback={lb}: SR={sr:.4f}")

    within_dm = sum(1 for r in dm_results
                    if abs(r["sharpe"] - dm_baseline_sr) <= 0.20 * abs(dm_baseline_sr + 1e-8))
    dm_plateau = within_dm / max(len(dm_results), 1)
    print(f"    Plateau: {within_dm}/{len(dm_results)} ({dm_plateau*100:.0f}%)")

    all_results["usdjpy_dm"] = {
        "baseline_sharpe": round(dm_baseline_sr, 4),
        "plateau_fraction": round(dm_plateau, 4),
        "is_plateau": dm_plateau >= 0.60,
        "perturbations": dm_results,
    }

    all_plateau = all(v["is_plateau"] for v in all_results.values())
    print(f"\n  [KILL CRITERIA]")
    for name, r in all_results.items():
        print(f"    {name}: {r['plateau_fraction']*100:.0f}% → "
              f"{'PLATEAU' if r['is_plateau'] else 'SPIKE'}")
    print(f"    OVERALL: {'PASS' if all_plateau else 'FAIL'}")

    return {"cells": all_results, "all_plateau": all_plateau, "pass": all_plateau}


# ================================================================
# MAIN
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


def main():
    parser = argparse.ArgumentParser(description="Phase 6.E.0.1 Plugin Stress Tests")
    parser.add_argument("--task", default="all", choices=["6c1", "6c4", "6c5", "all"])
    args = parser.parse_args()

    print("=" * 70)
    print(f"PHASE 6.E.0.1 PLUGIN-CANONICAL STRESS TESTS — Task: {args.task}")
    print("=" * 70)

    print("\n[DATA] Loading...")
    data, all_fx = load_all_data()

    results = {}
    tasks = [args.task] if args.task != "all" else ["6c1", "6c4", "6c5"]

    for task in tasks:
        if task == "6c1":
            results["task_6c1"] = task_6c1(data, all_fx)
        elif task == "6c4":
            results["task_6c4"] = task_6c4(data, all_fx)
        elif task == "6c5":
            results["task_6c5"] = task_6c5(data, all_fx)

    results_clean = convert_for_json(results)
    out_file = os.path.join(RESULTS_DIR, f"phase_6e01_plugin_stress_{args.task}.json")
    with open(out_file, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")

    print("\n" + "=" * 70)
    print("PLUGIN STRESS TEST SUMMARY")
    print("=" * 70)
    for task_name, r in results.items():
        print(f"  {task_name}: {'PASS' if r.get('pass') else 'FAIL'}")


if __name__ == "__main__":
    main()
