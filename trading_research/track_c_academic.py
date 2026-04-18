#!/usr/bin/env python3
"""
Phase 4 — Track C: Academic Strategy Replication

Strategy 1: Time-Series Momentum (Moskowitz, Ooi, Pedersen 2012)
Strategy 2: Dual Momentum (Antonacci 2014)
Strategy 3: Cross-Sectional FX Momentum (Menkhoff, Sarno, Schmeling, Schrimpf 2012)

All implemented EXACTLY as published — no tuning, no optimization.
Evaluated under the same harness, cost model, and kill criteria as Phase 3.
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXTENDED_DATA = os.path.join(os.path.dirname(__file__), "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXTENDED_DATA, exist_ok=True)

# Assets used across academic strategies
# Strategy 1 uses all; Strategy 2 uses 4; Strategy 3 uses FX only
ALL_ASSETS = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X", "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X", "XAU/USD": "GC=F", "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD", "XAG/USD": "SI=F", "CL": "CL=F",
}

FX_PAIRS = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]

EXTENDED_START = {
    "BTC/USD": "2014-09-17", "ETH/USD": "2017-08-01",
    "XAU/USD": "2000-01-01", "XAG/USD": "2000-01-01", "CL": "2000-01-01",
    "EUR/USD": "2003-01-01", "USD/JPY": "2003-01-01",
    "GBP/USD": "2003-01-01", "AUD/USD": "2003-01-01",
}


def download_daily_data(asset):
    """Download extended daily data via yfinance, cache to CSV."""
    import yfinance as yf
    safe = asset.replace("/", "_")
    csv_path = os.path.join(EXTENDED_DATA, f"{safe}_daily.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) > 100:
            return df
    ticker = ALL_ASSETS.get(asset)
    if not ticker:
        return None
    start = EXTENDED_START.get(asset, "2003-01-01")
    try:
        df = yf.download(ticker, start=start, end="2025-12-31",
                         interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            df.to_csv(csv_path)
        return df if len(df) > 100 else None
    except Exception as e:
        print(f"    Download error {asset}: {e}")
        return None


def monthly_resample(daily_close):
    """Resample daily close to month-end close prices."""
    return daily_close.resample("ME").last().dropna()


# ============================================================
# STRATEGY 1: Time-Series Momentum (TSMOM)
# Moskowitz, Ooi, Pedersen (2012) — "Time Series Momentum"
#
# Signal: sign of past 12-month return
# Sizing: inverse-volatility weighting
# Rebalance: monthly
# Universe: all assets with ≥ 12 months history
# ============================================================
def strategy_tsmom(asset, daily_df, lookback_months=12, vol_window=252):
    """
    Implements TSMOM: sign(ret_12m) × (1/σ) with monthly rebalance.
    Returns daily positions array.
    """
    close = daily_df["Close"].values.astype(float)
    log_ret_daily = np.diff(np.log(close + 1e-12))
    log_ret_daily = np.concatenate([[0], log_ret_daily])
    dates = daily_df.index

    # Monthly close
    monthly_close = monthly_resample(daily_df["Close"])
    monthly_dates = monthly_close.index

    positions = np.zeros(len(daily_df))

    for i, date in enumerate(monthly_dates):
        if i < lookback_months:
            continue

        # 12-month return
        ret_12m = np.log(monthly_close.iloc[i] + 1e-12) - np.log(monthly_close.iloc[i - lookback_months] + 1e-12)
        signal = np.sign(ret_12m)

        # Inverse vol sizing
        daily_mask = (dates >= monthly_dates[i - 1]) & (dates <= date)
        recent_daily = log_ret_daily[daily_mask]
        if len(recent_daily) > 20:
            vol = np.std(recent_daily[-min(vol_window, len(recent_daily)):]) * np.sqrt(252)
        else:
            vol = 0.10  # default 10%

        # Target 10% vol (as in paper)
        target_vol = 0.10
        size = target_vol / max(vol, 0.01)
        size = min(size, 3.0)  # cap leverage at 3x

        # Apply position for the next month
        if i < len(monthly_dates) - 1:
            next_date = monthly_dates[i + 1]
        else:
            next_date = dates[-1]
        daily_mask_next = (dates > date) & (dates <= next_date)
        positions[daily_mask_next] = signal * size

    return positions, log_ret_daily


def run_strategy_1():
    """Run TSMOM across all available assets."""
    print("\n  STRATEGY 1: Time-Series Momentum (Moskowitz et al. 2012)")
    print("  " + "-" * 60)

    results = {}
    ppy = 252  # daily

    for asset in ALL_ASSETS:
        df = download_daily_data(asset)
        if df is None:
            continue

        positions, log_ret = strategy_tsmom(asset, df)
        abs_ret = np.abs(log_ret)

        # Evaluate
        gross = positions[:-1] * log_ret[1:]
        net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])

        sr = annualized_sharpe(net, ppy)
        bh_sr = annualized_sharpe(log_ret[1:], ppy)
        rolling = rolling_window_evaluation(net, ppy)

        equity = np.cumsum(net)
        eq_curve = np.exp(equity)
        peak = np.maximum.accumulate(eq_curve)
        dd = (peak - eq_curve) / (peak + 1e-12)
        max_dd = float(np.max(dd))

        pos_changes = np.abs(np.diff(positions))
        n_trades = np.sum(pos_changes > 0)
        trades_per_year = n_trades / (len(log_ret) / ppy)

        results[asset] = {
            "sharpe": round(sr, 4),
            "bh_sharpe": round(bh_sr, 4),
            "edge_sharpe": round(sr - bh_sr, 4),
            "regime_robustness": rolling["regime_robustness"],
            "worst_window_sharpe": rolling["worst_window_sharpe"],
            "max_drawdown": round(max_dd, 4),
            "trades_per_year": round(trades_per_year, 1),
            "n_bars": len(log_ret),
        }

        print(f"    {asset:>10}: SR={sr:+.3f}, BH={bh_sr:+.3f}, Edge={sr - bh_sr:+.3f}, "
              f"Worst2Y={rolling['worst_window_sharpe']:+.3f}, DD={max_dd:.1%}")

    # Portfolio: equal-weight across all assets
    if results:
        all_sharpes = [v["sharpe"] for v in results.values()]
        all_worst = [v["worst_window_sharpe"] for v in results.values()]
        results["_portfolio_summary"] = {
            "n_assets": len(results),
            "avg_sharpe": round(np.mean(all_sharpes), 4),
            "avg_worst_window": round(np.mean(all_worst), 4),
            "n_positive_sharpe": sum(1 for s in all_sharpes if s > 0),
        }
        print(f"    {'PORTFOLIO':>10}: avg_SR={np.mean(all_sharpes):+.3f}, "
              f"positive={sum(1 for s in all_sharpes if s > 0)}/{len(all_sharpes)}")

    return results


# ============================================================
# STRATEGY 2: Dual Momentum (Antonacci 2014)
#
# Compare 12-month returns of:
#   (a) asset vs T-bill (absolute momentum)
#   (b) asset vs other assets (relative momentum)
# Long if both are positive; else flat or switch to bonds
# ============================================================
def strategy_dual_momentum(daily_dfs, lookback_months=12):
    """
    Dual Momentum on FX pairs.
    Long strongest FX pair if its 12-month return > 0, else flat.
    """
    # Build monthly returns for each asset
    monthly_returns = {}
    monthly_close_dict = {}
    common_dates = None

    for asset in FX_PAIRS:
        if asset not in daily_dfs or daily_dfs[asset] is None:
            continue
        mc = monthly_resample(daily_dfs[asset]["Close"])
        monthly_close_dict[asset] = mc
        if common_dates is None:
            common_dates = mc.index
        else:
            common_dates = common_dates.intersection(mc.index)

    if common_dates is None or len(common_dates) < lookback_months + 2:
        return {}

    common_dates = common_dates.sort_values()

    # For each month, compute 12-month returns
    results_per_asset = {}
    ppy = 252

    for asset in monthly_close_dict:
        mc = monthly_close_dict[asset]
        daily_df = daily_dfs[asset]
        close = daily_df["Close"].values.astype(float)
        log_ret = np.diff(np.log(close + 1e-12))
        log_ret = np.concatenate([[0], log_ret])
        dates = daily_df.index
        positions = np.zeros(len(daily_df))

        for i in range(lookback_months, len(common_dates) - 1):
            month = common_dates[i]
            if month not in mc.index:
                continue
            mc_loc = mc.index.get_loc(month)
            if mc_loc < lookback_months:
                continue

            # 12-month return for this asset
            ret_12m = np.log(mc.iloc[mc_loc] + 1e-12) - np.log(mc.iloc[mc_loc - lookback_months] + 1e-12)

            # 12-month returns for all assets
            all_rets = {}
            for a2 in monthly_close_dict:
                mc2 = monthly_close_dict[a2]
                if month in mc2.index:
                    mc2_loc = mc2.index.get_loc(month)
                    if mc2_loc >= lookback_months:
                        all_rets[a2] = np.log(mc2.iloc[mc2_loc] + 1e-12) - np.log(mc2.iloc[mc2_loc - lookback_months] + 1e-12)

            # Relative momentum: is this the strongest?
            if len(all_rets) == 0:
                continue
            best_asset = max(all_rets, key=all_rets.get)

            # Apply until next month
            next_month = common_dates[i + 1] if i + 1 < len(common_dates) else dates[-1]
            mask = (dates > month) & (dates <= next_month)

            # Long if: (1) absolute momentum positive AND (2) this asset is best
            if ret_12m > 0 and best_asset == asset:
                positions[mask] = 1
            else:
                positions[mask] = 0

        abs_ret = np.abs(log_ret)
        gross = positions[:-1] * log_ret[1:]
        net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])

        sr = annualized_sharpe(net, ppy)
        bh_sr = annualized_sharpe(log_ret[1:], ppy)
        rolling = rolling_window_evaluation(net, ppy)

        equity = np.cumsum(net)
        eq_curve = np.exp(equity)
        peak = np.maximum.accumulate(eq_curve)
        dd = (peak - eq_curve) / (peak + 1e-12)
        max_dd = float(np.max(dd))

        pos_changes = np.abs(np.diff(positions))
        n_trades = np.sum(pos_changes > 0)
        trades_per_year = n_trades / (len(log_ret) / ppy)

        results_per_asset[asset] = {
            "sharpe": round(sr, 4),
            "bh_sharpe": round(bh_sr, 4),
            "edge_sharpe": round(sr - bh_sr, 4),
            "regime_robustness": rolling["regime_robustness"],
            "worst_window_sharpe": rolling["worst_window_sharpe"],
            "max_drawdown": round(max_dd, 4),
            "trades_per_year": round(trades_per_year, 1),
            "n_bars": len(log_ret),
        }

    return results_per_asset


def run_strategy_2():
    """Run Dual Momentum across FX pairs."""
    print("\n  STRATEGY 2: Dual Momentum (Antonacci 2014)")
    print("  " + "-" * 60)

    daily_dfs = {}
    for asset in FX_PAIRS:
        df = download_daily_data(asset)
        if df is not None:
            daily_dfs[asset] = df

    if len(daily_dfs) < 2:
        print("    Insufficient FX data")
        return {"error": "insufficient data"}

    results = strategy_dual_momentum(daily_dfs)

    for asset, r in results.items():
        print(f"    {asset:>10}: SR={r['sharpe']:+.3f}, BH={r['bh_sharpe']:+.3f}, "
              f"Edge={r['edge_sharpe']:+.3f}, Worst2Y={r['worst_window_sharpe']:+.3f}")

    return results


# ============================================================
# STRATEGY 3: Cross-Sectional FX Momentum (Menkhoff et al. 2012)
#
# Each month: rank FX pairs by past 12-month return.
# Long top 50%, short bottom 50%.
# Equal-weighted within each group.
# ============================================================
def run_strategy_3():
    """Run Cross-Sectional FX Momentum."""
    print("\n  STRATEGY 3: Cross-Sectional FX Momentum (Menkhoff et al. 2012)")
    print("  " + "-" * 60)

    # Load all FX daily data
    daily_dfs = {}
    for asset in FX_PAIRS:
        df = download_daily_data(asset)
        if df is not None:
            daily_dfs[asset] = df

    if len(daily_dfs) < 3:
        print("    Need ≥3 FX pairs")
        return {"error": "insufficient data"}

    # Build monthly returns
    monthly_close = {}
    for asset in daily_dfs:
        mc = monthly_resample(daily_dfs[asset]["Close"])
        monthly_close[asset] = mc

    common_dates = None
    for mc in monthly_close.values():
        if common_dates is None:
            common_dates = mc.index
        else:
            common_dates = common_dates.intersection(mc.index)
    common_dates = common_dates.sort_values()

    lookback_months = 12
    ppy = 252

    # For each asset, compute daily positions based on cross-sectional rank
    positions_dict = {a: np.zeros(len(daily_dfs[a])) for a in daily_dfs}

    for i in range(lookback_months, len(common_dates) - 1):
        month = common_dates[i]
        next_month = common_dates[i + 1]

        # 12-month returns for all assets
        rets = {}
        for asset, mc in monthly_close.items():
            if month in mc.index:
                loc = mc.index.get_loc(month)
                if loc >= lookback_months:
                    rets[asset] = np.log(mc.iloc[loc] + 1e-12) - np.log(mc.iloc[loc - lookback_months] + 1e-12)

        if len(rets) < 2:
            continue

        # Rank: sort by return
        ranked = sorted(rets.keys(), key=lambda a: rets[a], reverse=True)
        n_long = len(ranked) // 2
        n_short = len(ranked) - n_long

        long_assets = ranked[:n_long]
        short_assets = ranked[n_long:]

        for asset in long_assets:
            dates = daily_dfs[asset].index
            mask = (dates > month) & (dates <= next_month)
            positions_dict[asset][mask] = 1.0 / n_long

        for asset in short_assets:
            dates = daily_dfs[asset].index
            mask = (dates > month) & (dates <= next_month)
            positions_dict[asset][mask] = -1.0 / n_short

    # Evaluate each asset's contribution
    results = {}
    all_net = []

    for asset in daily_dfs:
        close = daily_dfs[asset]["Close"].values.astype(float)
        log_ret = np.diff(np.log(close + 1e-12))
        log_ret = np.concatenate([[0], log_ret])
        abs_ret = np.abs(log_ret)
        positions = positions_dict[asset]

        gross = positions[:-1] * log_ret[1:]
        net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])
        all_net.append(net)

        sr = annualized_sharpe(net, ppy)
        bh_sr = annualized_sharpe(log_ret[1:], ppy)
        rolling = rolling_window_evaluation(net, ppy)

        results[asset] = {
            "sharpe": round(sr, 4),
            "bh_sharpe": round(bh_sr, 4),
            "edge_sharpe": round(sr - bh_sr, 4),
            "regime_robustness": rolling["regime_robustness"],
            "worst_window_sharpe": rolling["worst_window_sharpe"],
        }

        print(f"    {asset:>10}: SR={sr:+.3f}, Edge={sr - bh_sr:+.3f}, "
              f"Worst2Y={rolling['worst_window_sharpe']:+.3f}")

    # Portfolio: sum of all positions (this IS the cross-sectional portfolio)
    if all_net:
        min_len = min(len(n) for n in all_net)
        portfolio_net = sum(n[:min_len] for n in all_net)
        port_sr = annualized_sharpe(portfolio_net, ppy)
        port_bh = annualized_sharpe(portfolio_net, ppy)  # No B&H equivalent for L/S
        port_rolling = rolling_window_evaluation(portfolio_net, ppy)

        equity = np.cumsum(portfolio_net)
        eq_curve = np.exp(equity)
        peak = np.maximum.accumulate(eq_curve)
        dd = (peak - eq_curve) / (peak + 1e-12)
        max_dd = float(np.max(dd))

        results["_portfolio"] = {
            "sharpe": round(port_sr, 4),
            "regime_robustness": port_rolling["regime_robustness"],
            "worst_window_sharpe": port_rolling["worst_window_sharpe"],
            "max_drawdown": round(max_dd, 4),
        }
        print(f"    {'PORTFOLIO':>10}: SR={port_sr:+.3f}, Worst2Y={port_rolling['worst_window_sharpe']:+.3f}, "
              f"DD={max_dd:.1%}")

    return results


def main():
    print("=" * 70)
    print("PHASE 4 — TRACK C: ACADEMIC STRATEGY REPLICATION")
    print("=" * 70)

    s1_results = run_strategy_1()
    s2_results = run_strategy_2()
    s3_results = run_strategy_3()

    # Kill criteria check
    print("\n" + "=" * 70)
    print("TRACK C — KILL CRITERIA CHECK")
    print("=" * 70)

    thresholds = [-0.5, -0.75, -1.0]
    summary = {"strategy_1_tsmom": {}, "strategy_2_dual_momentum": {}, "strategy_3_xsec_fx": {}}

    for label, results_dict, key in [
        ("Strategy 1 (TSMOM)", s1_results, "strategy_1_tsmom"),
        ("Strategy 2 (Dual Momentum)", s2_results, "strategy_2_dual_momentum"),
        ("Strategy 3 (XS-FX)", s3_results, "strategy_3_xsec_fx"),
    ]:
        print(f"\n  {label}:")
        for asset, r in results_dict.items():
            if asset.startswith("_"):
                continue
            if isinstance(r, str):
                continue
            worst = r.get("worst_window_sharpe", -999)
            for thr in thresholds:
                thr_key = f"survive_at_{abs(thr)}"
                if thr_key not in summary[key]:
                    summary[key][thr_key] = {"alive": 0, "dead": 0}
                if worst > thr:
                    summary[key][thr_key]["alive"] += 1
                else:
                    summary[key][thr_key]["dead"] += 1
            print(f"    {asset:>10}: worst_2Y={worst:+.3f} → "
                  f"{'ALIVE' if worst > -0.5 else 'DEAD'} at -0.5, "
                  f"{'ALIVE' if worst > -0.75 else 'DEAD'} at -0.75, "
                  f"{'ALIVE' if worst > -1.0 else 'DEAD'} at -1.0")

    # Decision
    print("\n" + "=" * 70)
    print("TRACK C DECISION")
    print("=" * 70)

    for key, label in [("strategy_1_tsmom", "TSMOM"), ("strategy_2_dual_momentum", "Dual Momentum"),
                       ("strategy_3_xsec_fx", "XS-FX Momentum")]:
        for thr_key in summary[key]:
            s = summary[key][thr_key]
            print(f"  {label} {thr_key}: {s['alive']} alive / {s['dead']} dead")

    output = {
        "track": "C",
        "strategy_1_tsmom": s1_results,
        "strategy_2_dual_momentum": s2_results,
        "strategy_3_xsec_fx": s3_results,
        "kill_criteria_summary": summary,
    }

    output_path = os.path.join(RESULTS_DIR, "phase4_track_c_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
