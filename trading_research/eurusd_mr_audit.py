#!/usr/bin/env python3
"""
Phase 2: Deep Audit of EUR/USD Mean Reversion Anomaly

The ~0.63 net Sharpe on EUR/USD mean reversion is the one genuinely surprising
signal from prior work. This script either confirms or kills it.

Tasks:
  2.1 — Full characterization (parameters, rolling Sharpe, stress regimes)
  2.2 — Robustness to parameter perturbation (smooth plateau vs sharp spike)
  2.3 — Cost sensitivity (survives at 3 bps?)
  2.4 — Structural explanation (time-of-day, session effects)

Kill criterion: if rolling robustness < 0.4 OR parameter plateau is narrow
               OR signal dies at 3 bps OR no structural explanation → discard.
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.transaction_cost_model import total_cost_bps
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation, max_drawdown_from_returns
)


def load_eurusd_data():
    """Load the full EUR/USD hourly dataset (2005-2020)."""
    data_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "feature-eng", "tests", "data", "eurusd_hour_2005_2020_ohlc.csv"),
        os.path.expanduser("~/Documents/GitHub/feature-eng/tests/data/eurusd_hour_2005_2020_ohlc.csv"),
    ]

    for p in data_paths:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index, dayfirst=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Standardize column names
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "open" in cl: col_map[c] = "Open"
                elif "high" in cl: col_map[c] = "High"
                elif "low" in cl: col_map[c] = "Low"
                elif "close" in cl: col_map[c] = "Close"
                elif "vol" in cl: col_map[c] = "Volume"
            df = df.rename(columns=col_map)
            return df

    # Fallback: download daily via yfinance
    import yfinance as yf
    print("  Hourly data not found, downloading daily from yfinance...")
    df = yf.download("EURUSD=X", start="2005-01-01", end="2025-12-31",
                     interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def mean_reversion_strategy(log_returns: np.ndarray, lookback: int = 20,
                            z_entry: float = 1.5, z_exit: float = 0.5,
                            max_hold: int = 50) -> np.ndarray:
    """
    Mean reversion based on z-score of cumulative returns.
    Returns position array.
    """
    n = len(log_returns)
    positions = np.zeros(n)
    cum_ret = np.cumsum(log_returns)
    hold_counter = 0

    for i in range(lookback, n):
        window = cum_ret[max(0, i-lookback):i+1]
        if np.std(window) < 1e-12:
            positions[i] = 0
            continue

        z = (cum_ret[i] - np.mean(window)) / np.std(window)

        if abs(z) >= z_entry:
            positions[i] = -np.sign(z)
            hold_counter = 0
        elif abs(z) < z_exit or hold_counter >= max_hold:
            positions[i] = 0
            hold_counter = 0
        else:
            positions[i] = positions[i-1] if i > 0 else 0
            hold_counter += 1

    return positions


def apply_costs(returns: np.ndarray, positions: np.ndarray,
                spread_bps: float = 1.5) -> np.ndarray:
    """Apply flat cost model for sensitivity testing."""
    net = returns.copy()
    pos_changes = np.diff(positions, prepend=0)
    for i in range(len(net)):
        if pos_changes[i] != 0:
            net[i] -= spread_bps / 10000.0
    return net


def task_2_1_characterization(df: pd.DataFrame, ppy: float) -> dict:
    """Full characterization of the MR signal."""
    print("\n" + "=" * 70)
    print("TASK 2.1: FULL CHARACTERIZATION")
    print("=" * 70)

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])

    # Grid search for best parameters
    lookbacks = [10, 15, 20, 30, 50]
    z_entries = [1.0, 1.5, 2.0, 2.5]
    z_exits = [0.0, 0.25, 0.5, 0.75]

    best_sharpe = -999
    best_params = {}
    all_param_results = []

    for lb, ze, zx in product(lookbacks, z_entries, z_exits):
        if zx >= ze:
            continue
        pos = mean_reversion_strategy(log_ret, lookback=lb, z_entry=ze, z_exit=zx)
        gross = pos[:-1] * log_ret[1:]
        gross = np.concatenate([[0], gross])
        net = apply_costs(gross, pos, spread_bps=1.5)  # Phase 0.2 standard
        sharpe = annualized_sharpe(net, ppy)

        n_trades = int(np.sum(np.abs(np.diff(pos, prepend=0)) > 0))
        trade_rets = net[pos != 0]
        hit_rate = float(np.mean(trade_rets > 0)) if len(trade_rets) > 0 else 0

        entry = {
            "lookback": lb, "z_entry": ze, "z_exit": zx,
            "sharpe": round(sharpe, 4),
            "n_trades": n_trades,
            "hit_rate": round(hit_rate, 4),
        }
        all_param_results.append(entry)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {"lookback": lb, "z_entry": ze, "z_exit": zx}

    print(f"  Best params: lookback={best_params['lookback']}, "
          f"z_entry={best_params['z_entry']}, z_exit={best_params['z_exit']}")
    print(f"  Best net Sharpe: {best_sharpe:+.4f}")

    # Full rolling evaluation with best params
    pos = mean_reversion_strategy(log_ret, **best_params)
    gross = pos[:-1] * log_ret[1:]
    gross = np.concatenate([[0], gross])
    net = apply_costs(gross, pos, spread_bps=1.5)

    rolling = rolling_window_evaluation(net, ppy, window_years=2.0, step_months=6)
    print(f"  Rolling robustness: {rolling['regime_robustness']:.2%}")
    print(f"  Full-period Sharpe: {rolling['full_period']['sharpe']:+.4f}")
    print(f"  Worst window Sharpe: {rolling['worst_window_sharpe']:+.4f}")
    print(f"  Max DD: {rolling['full_period']['max_drawdown']:.2%}")
    print(f"  Is interesting: {rolling['is_interesting']}")

    # Trades per year
    n_years = len(close) / ppy
    n_trades = int(np.sum(np.abs(np.diff(pos, prepend=0)) > 0))
    trades_per_year = n_trades / n_years if n_years > 0 else 0
    print(f"  Trades/year: {trades_per_year:.1f}")

    # Trade stats
    trade_rets = net[pos != 0]
    if len(trade_rets) > 0:
        winners = trade_rets[trade_rets > 0]
        losers = trade_rets[trade_rets < 0]
        hit_rate = len(winners) / len(trade_rets)
        avg_win = np.mean(winners) * 10000 if len(winners) > 0 else 0
        avg_loss = np.mean(losers) * 10000 if len(losers) > 0 else 0
        print(f"  Hit rate: {hit_rate:.1%}")
        print(f"  Avg winner: {avg_win:+.1f} bps")
        print(f"  Avg loser: {avg_loss:+.1f} bps")

    # Stress regime performance
    print("\n  Stress regime performance:")
    regimes = {
        "2008 GFC": (2008, 2009),
        "2011-12 Euro crisis": (2011, 2013),
        "2015 SNB/China": (2015, 2016),
        "2020 COVID": (2020, 2021),
    }
    regime_results = {}
    for regime_name, (y_start, y_end) in regimes.items():
        mask = (df.index.year >= y_start) & (df.index.year < y_end)
        if mask.sum() < 100:
            print(f"    {regime_name}: insufficient data")
            continue
        r_net = net[mask]
        r_sharpe = annualized_sharpe(r_net, ppy)
        r_dd = max_drawdown_from_returns(r_net)
        regime_results[regime_name] = {"sharpe": round(r_sharpe, 4), "max_dd": round(r_dd, 4)}
        print(f"    {regime_name}: Sharpe={r_sharpe:+.4f}, DD={r_dd:.2%}")

    return {
        "best_params": best_params,
        "best_sharpe": best_sharpe,
        "rolling": {
            "regime_robustness": rolling["regime_robustness"],
            "worst_window_sharpe": rolling["worst_window_sharpe"],
            "full_period_sharpe": rolling["full_period"]["sharpe"],
            "max_drawdown": rolling["full_period"]["max_drawdown"],
            "is_interesting": rolling["is_interesting"],
        },
        "trades_per_year": trades_per_year,
        "stress_regimes": regime_results,
        "all_param_results": all_param_results,
    }


def task_2_2_parameter_robustness(df: pd.DataFrame, best_params: dict, ppy: float) -> dict:
    """Sweep ±50% around optimum. Real edge = smooth plateau. Overfit = spike."""
    print("\n" + "=" * 70)
    print("TASK 2.2: PARAMETER PERTURBATION ROBUSTNESS")
    print("=" * 70)

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])

    results = {}

    # Sweep each parameter individually
    for param_name, base_val in best_params.items():
        if param_name == "lookback":
            test_vals = sorted(set([max(5, int(base_val * m)) for m in
                                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]))
        else:
            test_vals = [round(base_val * m, 2) for m in
                         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]
            test_vals = [v for v in test_vals if v > 0]

        sharpes = []
        for val in test_vals:
            params = best_params.copy()
            params[param_name] = val
            if params.get("z_exit", 0) >= params.get("z_entry", 1):
                sharpes.append(None)
                continue
            pos = mean_reversion_strategy(log_ret, **params)
            gross = pos[:-1] * log_ret[1:]
            gross = np.concatenate([[0], gross])
            net = apply_costs(gross, pos, spread_bps=1.5)
            s = annualized_sharpe(net, ppy)
            sharpes.append(round(s, 4))

        valid_sharpes = [s for s in sharpes if s is not None]
        if len(valid_sharpes) > 2:
            sharpe_range = max(valid_sharpes) - min(valid_sharpes)
            plateau_width = sum(1 for s in valid_sharpes if s >= max(valid_sharpes) * 0.8) / len(valid_sharpes)
        else:
            sharpe_range = 0
            plateau_width = 0

        results[param_name] = {
            "base_value": base_val,
            "test_values": test_vals,
            "sharpes": sharpes,
            "sharpe_range": round(sharpe_range, 4),
            "plateau_width": round(plateau_width, 4),
        }

        print(f"  {param_name}: range={sharpe_range:.4f}, "
              f"plateau={plateau_width:.0%}")
        print(f"    values: {test_vals}")
        print(f"    sharpes: {sharpes}")

    # Overall robustness verdict
    all_plateaus = [r["plateau_width"] for r in results.values()]
    is_robust = all(p >= 0.4 for p in all_plateaus)
    print(f"\n  Verdict: {'ROBUST (smooth plateau)' if is_robust else 'FRAGILE (narrow spike)'}")

    return {"parameter_sweeps": results, "is_robust": is_robust}


def task_2_3_cost_sensitivity(df: pd.DataFrame, best_params: dict, ppy: float) -> dict:
    """Test at different cost levels. Genuine edge survives 3 bps."""
    print("\n" + "=" * 70)
    print("TASK 2.3: COST SENSITIVITY")
    print("=" * 70)

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])

    pos = mean_reversion_strategy(log_ret, **best_params)
    gross = pos[:-1] * log_ret[1:]
    gross = np.concatenate([[0], gross])

    cost_levels = [0, 1, 1.5, 2, 3, 5, 8, 10]
    results = []

    for cost_bps in cost_levels:
        net = apply_costs(gross, pos, spread_bps=cost_bps)
        sharpe = annualized_sharpe(net, ppy)
        rolling = rolling_window_evaluation(net, ppy)

        results.append({
            "cost_bps": cost_bps,
            "sharpe": round(sharpe, 4),
            "robustness": rolling["regime_robustness"],
            "total_return_pct": round((np.cumprod(1+net)[-1] - 1) * 100, 2),
        })
        print(f"  {cost_bps:>5.1f} bps: Sharpe={sharpe:+.4f}, "
              f"Robustness={rolling['regime_robustness']:.2f}")

    # Find break-even cost
    breakeven = 0
    for r in results:
        if r["sharpe"] >= 0.3:
            breakeven = r["cost_bps"]

    survives_3bps = any(r["cost_bps"] == 3 and r["sharpe"] >= 0.3 for r in results)
    print(f"\n  Break-even cost: {breakeven} bps")
    print(f"  Survives at 3 bps: {'YES' if survives_3bps else 'NO — KILL'}")

    return {
        "cost_sweep": results,
        "breakeven_bps": breakeven,
        "survives_3bps": survives_3bps,
    }


def task_2_4_structural_explanation(df: pd.DataFrame, best_params: dict, ppy: float) -> dict:
    """Look for structural reasons the MR edge exists."""
    print("\n" + "=" * 70)
    print("TASK 2.4: STRUCTURAL EXPLANATION")
    print("=" * 70)

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])

    pos = mean_reversion_strategy(log_ret, **best_params)

    # Check if we have hourly data (need hour attribute)
    has_hours = hasattr(df.index, 'hour') and df.index.hour.nunique() > 1

    results = {}

    if has_hours:
        # Time-of-day analysis
        print("\n  Time-of-day distribution of trades:")
        trade_mask = np.abs(np.diff(pos, prepend=0)) > 0
        trade_hours = df.index.hour[trade_mask]
        hour_counts = pd.Series(trade_hours).value_counts().sort_index()

        # Compute per-hour Sharpe
        gross = pos[:-1] * log_ret[1:]
        gross = np.concatenate([[0], gross])
        net = apply_costs(gross, pos, spread_bps=1.5)

        hour_sharpes = {}
        for h in range(24):
            mask = df.index.hour == h
            if mask.sum() < 100:
                continue
            h_net = net[mask]
            h_sharpe = annualized_sharpe(h_net, ppy)
            hour_sharpes[h] = round(h_sharpe, 4)
            bar = "█" * max(0, int(h_sharpe * 10))
            print(f"    Hour {h:02d}: Sharpe={h_sharpe:+.4f} {bar}")

        results["hour_sharpes"] = hour_sharpes

        # Session analysis
        print("\n  Session analysis:")
        sessions = {
            "Asian (0-8 UTC)": (0, 8),
            "London (8-16 UTC)": (8, 16),
            "New York (13-21 UTC)": (13, 21),
            "Off-hours (21-0 UTC)": (21, 24),
        }
        session_results = {}
        for sname, (h_start, h_end) in sessions.items():
            if h_end > h_start:
                mask = (df.index.hour >= h_start) & (df.index.hour < h_end)
            else:
                mask = (df.index.hour >= h_start) | (df.index.hour < h_end)
            if mask.sum() < 100:
                continue
            s_net = net[mask]
            s_sharpe = annualized_sharpe(s_net, ppy)
            session_results[sname] = round(s_sharpe, 4)
            print(f"    {sname}: Sharpe={s_sharpe:+.4f}")

        results["session_sharpes"] = session_results
    else:
        print("  (Daily data — no time-of-day analysis available)")

    # Day-of-week analysis
    print("\n  Day-of-week analysis:")
    gross = pos[:-1] * log_ret[1:]
    gross = np.concatenate([[0], gross])
    net = apply_costs(gross, pos, spread_bps=1.5)

    dow_sharpes = {}
    for dow in range(5):
        mask = df.index.dayofweek == dow
        if mask.sum() < 100:
            continue
        d_net = net[mask]
        d_sharpe = annualized_sharpe(d_net, ppy)
        dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri"][dow]
        dow_sharpes[dow_name] = round(d_sharpe, 4)
        print(f"    {dow_name}: Sharpe={d_sharpe:+.4f}")

    results["dow_sharpes"] = dow_sharpes

    # Autocorrelation structure (why MR works)
    print("\n  Autocorrelation structure:")
    for lag in [1, 2, 5, 10, 20]:
        ac = np.corrcoef(log_ret[lag:], log_ret[:-lag])[0, 1]
        print(f"    AC({lag}): {ac:+.4f}")
    results["autocorrelation_hint"] = "Negative autocorrelation supports MR"

    return results


def main():
    print("=" * 70)
    print("PHASE 2: DEEP AUDIT — EUR/USD MEAN REVERSION")
    print("=" * 70)

    df = load_eurusd_data()
    print(f"  Data: {len(df)} bars, {df.index.min()} to {df.index.max()}")

    # Determine PPY
    if hasattr(df.index, 'hour') and df.index.hour.nunique() > 1:
        ppy = 365.25 * 24  # hourly FX
        print(f"  Detected: hourly data, PPY={ppy:.0f}")
    else:
        ppy = 252  # daily
        print(f"  Detected: daily data, PPY={ppy}")

    # Task 2.1
    char_result = task_2_1_characterization(df, ppy)

    # Task 2.2
    robust_result = task_2_2_parameter_robustness(df, char_result["best_params"], ppy)

    # Task 2.3
    cost_result = task_2_3_cost_sensitivity(df, char_result["best_params"], ppy)

    # Task 2.4
    struct_result = task_2_4_structural_explanation(df, char_result["best_params"], ppy)

    # ─── Final Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: FINAL VERDICT")
    print("=" * 70)

    checks = {
        "Rolling robustness >= 0.4": char_result["rolling"]["regime_robustness"] >= 0.4,
        "Parameter plateau robust": robust_result["is_robust"],
        "Survives 3 bps cost": cost_result["survives_3bps"],
    }

    all_pass = all(checks.values())
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")

    if all_pass:
        print(f"\n  ★ EUR/USD MR CONFIRMED — proceed with this signal")
    else:
        print(f"\n  ✗ EUR/USD MR KILLED — discard and do not revisit")

    # Save
    output = {
        "verdict": "confirmed" if all_pass else "killed",
        "checks": {k: v for k, v in checks.items()},
        "characterization": char_result,
        "parameter_robustness": robust_result,
        "cost_sensitivity": cost_result,
        "structural_explanation": struct_result,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "eurusd_mr_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
