#!/usr/bin/env python3
"""
Phase 5.5 — Corrective Audit Before Terminal State Decision

Tasks 5.5.1-5.5.5: Oracle-free portfolio rebuild, out-of-sample validation,
proper benchmark comparison, Q3 deployment audit, revised synthesis.

Corrects the 11 issues identified in Phase 5 self-critique.
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe
)
from trading_research.transaction_cost_model import apply_cost_to_returns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXTENDED_DATA = os.path.join(os.path.dirname(__file__), "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXTENDED_DATA, exist_ok=True)

PPY_DAILY = 252
PPY_WEEKLY = 52
TARGET_VOL = 0.10

YFINANCE_TICKERS = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X", "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X", "AUD/JPY": "AUDJPY=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F",
}

EXTENDED_START = {
    "BTC/USD": "2014-09-17", "ETH/USD": "2017-08-01",
    "XAU/USD": "2000-01-01", "XAG/USD": "2000-01-01",
    "EUR/USD": "2003-01-01", "USD/JPY": "2003-01-01", "GBP/USD": "2003-01-01",
    "AUD/USD": "2003-01-01", "AUD/JPY": "2003-01-01", "EUR/JPY": "2003-01-01",
    "GBP/JPY": "2003-01-01",
}

REGIMES = {
    "pre_gfc":   ("2003-01-01", "2007-06-30"),
    "gfc":       ("2007-07-01", "2009-06-30"),
    "qe_era":    ("2009-07-01", "2020-02-28"),
    "covid":     ("2020-03-01", "2021-12-31"),
    "inflation": ("2022-01-01", "2026-12-31"),
}

# OOS split dates (per work plan)
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"  # never used for tuning


# ================================================================
# DATA LOADING
# ================================================================
def load_extended_data(asset, timeframe):
    """Download/load extended price data with caching."""
    import yfinance as yf
    safe = asset.replace("/", "_")
    csv_path = os.path.join(EXTENDED_DATA, f"{safe}_{timeframe}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) > 100:
            return df
    ticker = YFINANCE_TICKERS.get(asset)
    if not ticker:
        return None
    start = EXTENDED_START.get(asset, "2003-01-01")
    interval_map = {"daily": "1d", "weekly": "1wk"}
    interval = interval_map.get(timeframe, "1d")
    try:
        df = yf.download(ticker, start=start, end="2026-12-31",
                         interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            df.to_csv(csv_path)
        return df if len(df) > 100 else None
    except Exception as e:
        print(f"    Download error {asset}: {e}")
        return None


# ================================================================
# ORACLE-FREE STRATEGY IMPLEMENTATIONS
# ================================================================
def run_pure_mr(log_ret, close, lookback=20, z_entry=1.5, z_exit=0.5):
    """Pure mean-reversion (no oracle). Returns positions array."""
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
    """TSMOM: sign of 12-month return, monthly rebalance, inverse-vol sized."""
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
    """Dual Momentum: long if absolute + relative momentum positive."""
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


# ================================================================
# TASK 5.5.1 — CELL CLASSIFICATION AND CORRECTED PORTFOLIO
# ================================================================

# Full cell classification table
ALL_CELLS = [
    # Oracle-free cells (directly deployable)
    {"asset": "EUR/USD", "timeframe": "daily",  "strategy": "pure_mr",         "oracle_free": True,  "source": "phase4_trackA"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "pure_mr",         "oracle_free": True,  "source": "phase4_trackA"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "tsmom",           "oracle_free": True,  "source": "phase4_trackC"},
    {"asset": "XAU/USD", "timeframe": "daily",  "strategy": "tsmom",           "oracle_free": True,  "source": "phase4_trackC"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "dual_momentum",   "oracle_free": True,  "source": "phase4_trackC"},
    # Oracle-dependent cells (σ=10 noisy oracle — NOT directly deployable)
    {"asset": "EUR/USD", "timeframe": "daily",  "strategy": "mean_reversion",  "oracle_free": False, "source": "phase3_oracle"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "mean_reversion",  "oracle_free": False, "source": "phase3_oracle"},
    {"asset": "XAU/USD", "timeframe": "daily",  "strategy": "momentum",        "oracle_free": False, "source": "phase3_oracle"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum",        "oracle_free": False, "source": "phase3_oracle"},
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum",        "oracle_free": False, "source": "phase3_oracle"},
    {"asset": "AUD/USD", "timeframe": "weekly", "strategy": "vol_regime_switch","oracle_free": False, "source": "phase3_oracle"},
    {"asset": "EUR/JPY", "timeframe": "weekly", "strategy": "vol_regime_switch","oracle_free": False, "source": "phase3_oracle"},
]


def compute_cell_returns_oracle_free(cell, all_fx_data=None):
    """
    Compute daily return series for an ORACLE-FREE cell only.
    Returns (dates, net_returns, metadata) or None.
    """
    asset = cell["asset"]
    timeframe = cell["timeframe"]
    strategy = cell["strategy"]

    df = load_extended_data(asset, timeframe)
    if df is None:
        return None

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    dates = df.index
    abs_ret = np.abs(log_ret)
    ppy = periods_per_year_for_timeframe(timeframe)

    # Oracle-free strategies only
    if strategy == "pure_mr":
        positions = run_pure_mr(log_ret, close)
    elif strategy == "tsmom":
        positions = run_tsmom(log_ret, close, dates)
    elif strategy == "dual_momentum":
        if all_fx_data is None:
            return None
        positions = run_dual_momentum(log_ret, close, dates, all_fx_data, asset)
    else:
        return None

    # Net returns
    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])

    # Vol-target to 10%
    realized_vol = np.std(net) * np.sqrt(ppy) if np.std(net) > 0 else 0.10
    vol_scalar = TARGET_VOL / max(realized_vol, 0.01)
    vol_scalar = min(vol_scalar, 5.0)
    net_scaled = net * vol_scalar

    # Return at native frequency — NO forward-fill (fixes critique #11)
    # Weekly cells return weekly dates + weekly returns
    return_dates = dates[1:len(net_scaled) + 1]
    bh_sr = annualized_sharpe(log_ret[1:], ppy)
    sr = annualized_sharpe(net_scaled, ppy)

    rolling = rolling_window_evaluation(net_scaled, ppy)

    metadata = {
        "asset": asset,
        "timeframe": timeframe,
        "strategy": strategy,
        "source": cell["source"],
        "oracle_free": True,
        "sharpe": round(sr, 4),
        "bh_sharpe": round(bh_sr, 4),
        "edge_sharpe": round(sr - bh_sr, 4),
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "regime_robustness": rolling["regime_robustness"],
        "vol_scalar": round(vol_scalar, 4),
        "n_bars": len(net_scaled),
        "ppy": ppy,
    }

    return return_dates, net_scaled, metadata


def task_551():
    """Task 5.5.1: Classify cells, filter, rebuild portfolios."""
    print("=" * 70)
    print("TASK 5.5.1 — ORACLE-FREE PORTFOLIO REBUILD")
    print("=" * 70)

    # 5.5.1.1 — Classification table
    print("\n--- 5.5.1.1: Cell Classification ---")
    print(f"\n  {'Cell':<45} {'Oracle-Free?':<15} {'Source'}")
    print(f"  {'-'*75}")
    for c in ALL_CELLS:
        name = f"{c['asset']}_{c['timeframe']}_{c['strategy']}"
        of = "YES" if c["oracle_free"] else "NO (σ=10)"
        print(f"  {name:<45} {of:<15} {c['source']}")

    oracle_free = [c for c in ALL_CELLS if c["oracle_free"]]
    print(f"\n  Oracle-free cells: {len(oracle_free)}")
    print(f"  Oracle-dependent cells: {len(ALL_CELLS) - len(oracle_free)} (EXCLUDED from portfolio)")

    # Load FX data for dual momentum
    print("\n--- Loading FX data ---")
    fx_pairs = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]
    all_fx_data = {}
    for asset in fx_pairs:
        df = load_extended_data(asset, "daily")
        if df is not None:
            close = df["Close"].values.astype(float)
            log_ret = np.diff(np.log(close + 1e-12))
            log_ret = np.concatenate([[0], log_ret])
            all_fx_data[asset] = (log_ret, close, df.index)
            print(f"  {asset}: {len(df)} bars")

    # 5.5.1.2 — Compute returns and apply edge > 0 filter strictly
    print("\n--- 5.5.1.2: Computing oracle-free cell returns + edge > 0 filter ---")
    cell_returns = {}  # key -> (dates, returns, metadata)
    all_meta = []
    excluded = []

    for cell in oracle_free:
        key = f"{cell['asset']}_{cell['timeframe']}_{cell['strategy']}"
        result = compute_cell_returns_oracle_free(cell, all_fx_data)
        if result is None:
            excluded.append((key, "no_data"))
            continue

        dates, net, meta = result
        all_meta.append(meta)

        # Strict edge > 0 filter
        if meta["edge_sharpe"] <= 0:
            excluded.append((key, f"edge={meta['edge_sharpe']:+.4f} <= 0"))
            print(f"  EXCLUDED {key}: edge={meta['edge_sharpe']:+.4f} <= 0")
            continue

        cell_returns[key] = (dates, net, meta)
        print(f"  INCLUDED {key}: SR={meta['sharpe']:+.3f}, Edge={meta['edge_sharpe']:+.3f}, "
              f"Worst2Y={meta['worst_window_sharpe']:+.3f}")

    # 5.5.1.3 — Cap per-asset representation at 2
    print("\n--- 5.5.1.3: Asset over-representation check ---")
    asset_count = {}
    for key, (_, _, meta) in cell_returns.items():
        a = meta["asset"]
        asset_count[a] = asset_count.get(a, []) + [key]

    for asset, keys in asset_count.items():
        if len(keys) > 2:
            print(f"  ⚠ {asset} has {len(keys)} cells: {keys}")
            # Keep 2 best by edge_sharpe
            sorted_keys = sorted(keys, key=lambda k: cell_returns[k][2]["edge_sharpe"], reverse=True)
            for drop_key in sorted_keys[2:]:
                meta = cell_returns[drop_key][2]
                print(f"    DROPPED {drop_key} (edge={meta['edge_sharpe']:+.3f})")
                excluded.append((drop_key, "asset_cap_2"))
                del cell_returns[drop_key]
        else:
            print(f"  {asset}: {len(keys)} cell(s) — OK")

    print(f"\n  Final oracle-free candidate set: {len(cell_returns)} cells")
    if len(cell_returns) < 2:
        print("  ⚠ INSUFFICIENT CELLS — cannot build portfolio")
        return {"error": "fewer than 2 oracle-free cells with edge > 0",
                "classification": [m for m in all_meta],
                "excluded": excluded}

    # 5.5.1.4 — Build portfolios at WEEKLY frequency (fixes critique #11)
    # Aggregate all cells to weekly returns before combining
    print("\n--- 5.5.1.4-5: Portfolio construction at weekly frequency ---")
    print("  (Using Option B: weekly aggregation to avoid forward-fill artifacts)")

    weekly_returns = {}
    for key, (dates, rets, meta) in cell_returns.items():
        series = pd.Series(rets, index=dates)
        if meta["timeframe"] == "weekly":
            # Already weekly
            weekly_returns[key] = series
        else:
            # Aggregate daily to weekly
            weekly_returns[key] = series.resample("W-FRI").sum()

    # Build aligned weekly dataframe
    df_weekly = pd.DataFrame(weekly_returns)
    df_weekly = df_weekly.dropna(how="all").fillna(0)

    # Need overlap where at least 2 cells have data
    valid = df_weekly.ne(0).sum(axis=1)
    df_weekly = df_weekly[valid >= min(2, len(cell_returns))]

    print(f"  Weekly return matrix: {len(df_weekly)} weeks × {len(df_weekly.columns)} cells")
    print(f"  Period: {df_weekly.index[0].date()} to {df_weekly.index[-1].date()}")

    if len(df_weekly) < 104:  # need 2 years
        print("  ⚠ Insufficient overlap")
        return {"error": "insufficient weekly overlap"}

    # Correlation matrix
    ret_corr = df_weekly.corr()
    print("\n  Weekly return correlations:")
    names = list(ret_corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"    {names[i]} / {names[j]}: {ret_corr.iloc[i, j]:+.3f}")

    # Build portfolios
    portfolios = build_portfolios_weekly(df_weekly, cell_returns)

    # Present at BOTH realized vol and 10% target vol
    print("\n  --- Portfolio Results (realized vol) ---")
    for pname, pdata in portfolios.items():
        if "error" in pdata:
            continue
        print(f"\n  {pname}:")
        print(f"    SR={pdata['sharpe']:+.3f}, Worst2Y={pdata['worst_2y_sharpe']:+.3f}, "
              f"MaxDD={pdata['max_dd']:.1%}, Vol={pdata['realized_vol']:.1%}, "
              f"TotalRet={pdata['total_return']:+.1%}")

    print("\n  --- Portfolio Results (scaled to 10% target vol) ---")
    for pname, pdata in portfolios.items():
        if "error" in pdata:
            continue
        sc = pdata.get("at_10pct_vol", {})
        if sc:
            print(f"\n  {pname} @10% vol:")
            print(f"    SR={sc['sharpe']:+.3f}, Worst2Y={sc['worst_2y_sharpe']:+.3f}, "
                  f"MaxDD={sc['max_dd']:.1%}, Leverage={sc['leverage']:.1f}x")

    # Regime analysis for best portfolio
    best_pname = min(portfolios, key=lambda k: -portfolios[k].get("worst_2y_sharpe", -99)
                     if "error" not in portfolios[k] else -99)
    best = portfolios[best_pname]
    if "weekly_returns" in best:
        print(f"\n  Regime analysis ({best_pname}):")
        regime_results = regime_analysis_weekly(best["weekly_returns"], best["weekly_dates"])
        for rname, rdata in regime_results.items():
            if rdata["sharpe"] is not None:
                print(f"    {rname}: SR={rdata['sharpe']:+.3f} ({rdata['n_weeks']} weeks)")
        best["regime_analysis"] = regime_results

    return {
        "classification": all_meta,
        "excluded": excluded,
        "included_cells": {k: v[2] for k, v in cell_returns.items()},
        "correlation_matrix": ret_corr.round(4).to_dict(),
        "portfolios": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("weekly_returns", "weekly_dates")}
                       for k, v in portfolios.items()},
        "best_portfolio": best_pname,
        "n_cells": len(cell_returns),
        "cell_returns_weekly": df_weekly,  # for OOS use — not serialized
    }


def build_portfolios_weekly(df_weekly, cell_returns):
    """Build P1-P5 from weekly return DataFrame."""
    results = {}
    cell_names = list(df_weekly.columns)

    # P1 — Equal weight
    p1 = df_weekly.mean(axis=1)
    results["P1_equal_weight"] = eval_portfolio_weekly(p1, "P1_equal_weight")

    # P2 — Inverse volatility
    vols = df_weekly.std()
    inv_vol = 1.0 / (vols + 1e-12)
    w2 = inv_vol / inv_vol.sum()
    p2 = (df_weekly * w2).sum(axis=1)
    results["P2_inverse_vol"] = eval_portfolio_weekly(p2, "P2_inverse_vol")
    results["P2_inverse_vol"]["weights"] = {k: round(v, 4) for k, v in w2.items()}

    # P3 — Inverse worst-window
    worst_w = {}
    for name in cell_names:
        meta = cell_returns.get(name, (None, None, {}))[2]
        ww = meta.get("worst_window_sharpe", -2.0) if meta else -2.0
        worst_w[name] = abs(ww) if ww < 0 else 0.01
    inv_ww = {k: 1.0 / max(v, 0.01) for k, v in worst_w.items()}
    total = sum(inv_ww.values())
    w3 = {k: v / total for k, v in inv_ww.items()}
    p3 = sum(df_weekly[k] * w for k, w in w3.items())
    results["P3_inverse_worst_window"] = eval_portfolio_weekly(p3, "P3_inverse_worst_window")
    results["P3_inverse_worst_window"]["weights"] = {k: round(v, 4) for k, v in w3.items()}

    # P4 — Risk parity (inverse variance)
    variances = df_weekly.var()
    inv_var = 1.0 / (variances + 1e-12)
    w4 = inv_var / inv_var.sum()
    p4 = (df_weekly * w4).sum(axis=1)
    results["P4_risk_parity"] = eval_portfolio_weekly(p4, "P4_risk_parity")
    results["P4_risk_parity"]["weights"] = {k: round(v, 4) for k, v in w4.items()}

    # P5 — Hedged pairs (lowest correlation pairs)
    corr = df_weekly.corr()
    n = len(cell_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((cell_names[i], cell_names[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[2])
    selected = pairs[:min(3, len(pairs))]
    if selected:
        pair_cells = set()
        for a, b, _ in selected:
            pair_cells.add(a)
            pair_cells.add(b)
        w5 = {k: 1.0 / len(pair_cells) for k in pair_cells}
        p5 = sum(df_weekly[k] * w for k, w in w5.items() if k in df_weekly.columns)
        results["P5_hedged_pairs"] = eval_portfolio_weekly(p5, "P5_hedged_pairs")
        results["P5_hedged_pairs"]["weights"] = {k: round(v, 4) for k, v in w5.items()}
        results["P5_hedged_pairs"]["selected_pairs"] = [(a, b, round(c, 4)) for a, b, c in selected]
    else:
        results["P5_hedged_pairs"] = {"error": "insufficient pairs"}

    return results


def eval_portfolio_weekly(weekly_ret_series, name):
    """Evaluate a weekly return series."""
    weekly_ret = weekly_ret_series.values
    weekly_dates = weekly_ret_series.index

    sr = annualized_sharpe(weekly_ret, PPY_WEEKLY)
    realized_vol = np.std(weekly_ret) * np.sqrt(PPY_WEEKLY)

    # Rolling 2Y worst window (104 weeks)
    rolling = rolling_window_evaluation(weekly_ret, PPY_WEEKLY,
                                        window_years=2.0, step_months=3)

    # Max drawdown
    equity = np.cumsum(weekly_ret)
    eq_curve = np.exp(equity)
    peak = np.maximum.accumulate(eq_curve)
    dd = (peak - eq_curve) / (peak + 1e-12)
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    total_ret = float(np.exp(np.sum(weekly_ret)) - 1)

    # At 10% target vol
    leverage = TARGET_VOL / max(realized_vol, 0.001)
    scaled_ret = weekly_ret * leverage
    sr_scaled = annualized_sharpe(scaled_ret, PPY_WEEKLY)
    rolling_scaled = rolling_window_evaluation(scaled_ret, PPY_WEEKLY,
                                               window_years=2.0, step_months=3)
    eq_s = np.exp(np.cumsum(scaled_ret))
    peak_s = np.maximum.accumulate(eq_s)
    dd_s = (peak_s - eq_s) / (peak_s + 1e-12)
    max_dd_s = float(np.max(dd_s)) if len(dd_s) > 0 else 0

    return {
        "name": name,
        "sharpe": round(sr, 4),
        "worst_2y_sharpe": rolling["worst_window_sharpe"],
        "max_dd": round(max_dd, 4),
        "realized_vol": round(realized_vol, 4),
        "total_return": round(total_ret, 4),
        "n_weeks": len(weekly_ret),
        "at_10pct_vol": {
            "sharpe": round(sr_scaled, 4),
            "worst_2y_sharpe": rolling_scaled["worst_window_sharpe"],
            "max_dd": round(max_dd_s, 4),
            "leverage": round(leverage, 2),
        },
        "weekly_returns": weekly_ret,   # for OOS/regime analysis
        "weekly_dates": weekly_dates,
    }


def regime_analysis_weekly(weekly_ret, weekly_dates):
    """Compute Sharpe per regime."""
    ret_series = pd.Series(weekly_ret, index=weekly_dates)
    results = {}
    for rname, (start, end) in REGIMES.items():
        mask = (ret_series.index >= start) & (ret_series.index <= end)
        r = ret_series[mask]
        if len(r) < 10:
            results[rname] = {"sharpe": None, "n_weeks": 0}
            continue
        sr = annualized_sharpe(r.values, PPY_WEEKLY)
        results[rname] = {"sharpe": round(sr, 4), "n_weeks": len(r)}
    return results


# ================================================================
# TASK 5.5.2 — OUT-OF-SAMPLE VALIDATION
# ================================================================
def task_552(cell_returns_dict, df_weekly_full):
    """Task 5.5.2: IS construction, OOS evaluation, degradation analysis."""
    print("\n" + "=" * 70)
    print("TASK 5.5.2 — OUT-OF-SAMPLE PORTFOLIO VALIDATION")
    print("=" * 70)

    # Split into train/test
    train_mask = df_weekly_full.index <= TRAIN_END
    test_mask = (df_weekly_full.index >= TEST_START) & (df_weekly_full.index <= TEST_END)

    df_train = df_weekly_full[train_mask]
    df_test = df_weekly_full[test_mask]

    print(f"\n  Training period: {df_train.index[0].date()} to {df_train.index[-1].date()} "
          f"({len(df_train)} weeks)")
    print(f"  Test period:     {df_test.index[0].date()} to {df_test.index[-1].date()} "
          f"({len(df_test)} weeks)")

    if len(df_train) < 104:
        print("  ⚠ Insufficient training data")
        return {"error": "insufficient training data"}
    if len(df_test) < 52:
        print("  ⚠ Insufficient test data")
        return {"error": "insufficient test data"}

    results = {}

    # 5.5.2.2 — Compute weights on TRAINING data only
    print("\n--- 5.5.2.2: In-sample weight construction ---")

    # P1: equal weight (no parameters to fit)
    n_cells = len(df_train.columns)
    w1 = {c: 1.0 / n_cells for c in df_train.columns}

    # P2: inverse vol from training period
    train_vols = df_train.std()
    inv_vol = 1.0 / (train_vols + 1e-12)
    w2 = {c: float(v) for c, v in (inv_vol / inv_vol.sum()).items()}

    # P3: inverse worst-window from training period
    train_worst = {}
    for col in df_train.columns:
        ret = df_train[col].values
        if len(ret) >= 104:
            roll = rolling_window_evaluation(ret, PPY_WEEKLY, window_years=2.0, step_months=3)
            train_worst[col] = abs(roll["worst_window_sharpe"]) if roll["worst_window_sharpe"] < 0 else 0.01
        else:
            train_worst[col] = 2.0  # default penalty
    inv_ww = {k: 1.0 / max(v, 0.01) for k, v in train_worst.items()}
    total_ww = sum(inv_ww.values())
    w3 = {k: v / total_ww for k, v in inv_ww.items()}

    # P4: risk parity from training period
    train_var = df_train.var()
    inv_var = 1.0 / (train_var + 1e-12)
    w4 = {c: float(v) for c, v in (inv_var / inv_var.sum()).items()}

    # P5: hedged pairs from training correlations
    train_corr = df_train.corr()
    cols = list(df_train.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], train_corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[2])
    selected = pairs[:min(3, len(pairs))]
    pair_cells = set()
    for a, b, _ in selected:
        pair_cells.add(a)
        pair_cells.add(b)
    w5 = {k: 1.0 / len(pair_cells) for k in pair_cells} if pair_cells else {}

    frozen_weights = {"P1": w1, "P2": w2, "P3": w3, "P4": w4, "P5": w5}
    print("\n  Frozen weights (from training period only):")
    for pname, w in frozen_weights.items():
        print(f"    {pname}: {', '.join(f'{k}={v:.3f}' for k, v in w.items())}")

    # 5.5.2.3 — Apply frozen weights to IS and OOS
    print("\n--- 5.5.2.3-4: IS vs OOS evaluation ---")

    comparison = {}
    for pname, weights in frozen_weights.items():
        if not weights:
            continue

        # In-sample
        is_ret = sum(df_train[k] * w for k, w in weights.items() if k in df_train.columns)
        is_metrics = eval_portfolio_weekly(is_ret, f"{pname}_IS") if len(is_ret) > 0 else None

        # Out-of-sample (FROZEN weights)
        oos_ret = sum(df_test[k] * w for k, w in weights.items() if k in df_test.columns)
        oos_metrics = eval_portfolio_weekly(oos_ret, f"{pname}_OOS") if len(oos_ret) > 0 else None

        if is_metrics and oos_metrics:
            is_sr = is_metrics["sharpe"]
            oos_sr = oos_metrics["sharpe"]
            degradation = 1.0 - (oos_sr / is_sr) if abs(is_sr) > 0.01 else float("nan")

            comparison[pname] = {
                "is_sharpe": is_sr,
                "oos_sharpe": oos_sr,
                "sharpe_degradation_pct": round(degradation * 100, 1) if not np.isnan(degradation) else None,
                "is_worst_2y": is_metrics["worst_2y_sharpe"],
                "oos_worst_2y": oos_metrics["worst_2y_sharpe"],
                "is_max_dd": is_metrics["max_dd"],
                "oos_max_dd": oos_metrics["max_dd"],
                "is_vol": is_metrics["realized_vol"],
                "oos_vol": oos_metrics["realized_vol"],
                "weights": weights,
            }

            print(f"\n  {pname}:")
            print(f"    IS:  SR={is_sr:+.3f}, Worst2Y={is_metrics['worst_2y_sharpe']:+.3f}, "
                  f"MaxDD={is_metrics['max_dd']:.1%}")
            print(f"    OOS: SR={oos_sr:+.3f}, Worst2Y={oos_metrics['worst_2y_sharpe']:+.3f}, "
                  f"MaxDD={oos_metrics['max_dd']:.1%}")
            deg_str = f"{degradation:.0%}" if not np.isnan(degradation) else "N/A"
            print(f"    Degradation: {deg_str}")

    # Summary
    print("\n  --- Degradation Summary ---")
    robust = []
    fragile = []
    for pname, comp in comparison.items():
        deg = comp.get("sharpe_degradation_pct")
        if deg is not None and deg <= 30:
            robust.append(pname)
        elif deg is not None and deg > 50:
            fragile.append(pname)
        # Between 30-50% is moderate

    if robust:
        print(f"  Robust (≤30% degradation): {', '.join(robust)}")
    if fragile:
        print(f"  Fragile (>50% degradation): {', '.join(fragile)}")
    if not robust and not fragile:
        print(f"  All portfolios show moderate degradation (30-50%)")

    return {
        "frozen_weights": frozen_weights,
        "comparison": comparison,
        "robust_portfolios": robust,
        "fragile_portfolios": fragile,
        "train_period": f"{df_train.index[0].date()} to {df_train.index[-1].date()}",
        "test_period": f"{df_test.index[0].date()} to {df_test.index[-1].date()}",
    }


# ================================================================
# TASK 5.5.3 — PROPER BENCHMARK COMPARISON
# ================================================================
def task_553(portfolios_data):
    """Task 5.5.3: Active managed-futures benchmarks only."""
    print("\n" + "=" * 70)
    print("TASK 5.5.3 — PROPER BENCHMARK COMPARISON (ACTIVE STRATEGIES ONLY)")
    print("=" * 70)

    import yfinance as yf

    # Active managed futures ETFs/funds — NOT passive asset classes
    BENCHMARKS = {
        "WTMF":  {"ticker": "WTMF",  "desc": "WisdomTree Managed Futures (since 2011)"},
        "FMF":   {"ticker": "FMF",   "desc": "First Trust Managed Futures (since 2013)"},
        "DBMF":  {"ticker": "DBMF",  "desc": "iMGP DBi Managed Futures (since 2019)"},
        "KMLM":  {"ticker": "KMLM",  "desc": "KFA Mount Lucas ML Futures (since 2020)"},
        "PQTAX": {"ticker": "PQTAX", "desc": "PIMCO TRENDS Managed Futures (since ~2014)"},
        "GMOM":  {"ticker": "GMOM",  "desc": "Cambria Global Momentum (since 2014)"},
    }

    benchmark_results = {}
    print("\n--- 5.5.3.1-2: Downloading and analyzing active benchmarks ---")

    for name, info in BENCHMARKS.items():
        ticker = info["ticker"]
        print(f"\n  {name} — {info['desc']}:")

        cache_path = os.path.join(RESULTS_DIR, f"benchmark_{ticker}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            try:
                df = yf.download(ticker, start="2005-01-01", end="2026-12-31",
                                 interval="1d", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 0:
                    df.to_csv(cache_path)
            except Exception as e:
                print(f"    ⚠ Download failed: {e}")
                continue

        if df is None or len(df) < 252:
            print(f"    ⚠ Insufficient data ({len(df) if df is not None else 0} bars)")
            continue

        close = df["Close"].values.astype(float)
        log_ret = np.diff(np.log(close + 1e-12))

        # Weekly aggregation for consistency
        weekly = pd.Series(log_ret, index=df.index[1:]).resample("W-FRI").sum()
        weekly = weekly.dropna()

        if len(weekly) < 104:
            print(f"    ⚠ Insufficient weekly data ({len(weekly)} weeks)")
            continue

        sr = annualized_sharpe(weekly.values, PPY_WEEKLY)
        vol = np.std(weekly.values) * np.sqrt(PPY_WEEKLY)
        rolling = rolling_window_evaluation(weekly.values, PPY_WEEKLY,
                                            window_years=2.0, step_months=3)

        result = {
            "ticker": ticker,
            "description": info["desc"],
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "n_weeks": len(weekly),
            "full_sharpe": round(sr, 4),
            "annual_vol": round(vol, 4),
            "worst_2y_sharpe": rolling["worst_window_sharpe"],
            "pass_05": rolling["worst_window_sharpe"] > -0.5,
            "pass_07": rolling["worst_window_sharpe"] > -0.7,
            "pass_10": rolling["worst_window_sharpe"] > -1.0,
        }
        benchmark_results[name] = result
        print(f"    SR={sr:+.3f}, Vol={vol:.1%}, Worst2Y={rolling['worst_window_sharpe']:+.3f}")

    # 5.5.3.3 — Our corrected portfolios (from task 5.5.1)
    print("\n--- 5.5.3.3-4: Side-by-side comparison ---")
    print(f"\n  {'Entity':<35} {'Period':<22} {'Sharpe':>8} {'Worst2Y':>9} "
          f"{'Pass-0.5':>9} {'Pass-0.7':>9} {'Pass-1.0':>9}")
    print(f"  {'-'*100}")

    for name, r in benchmark_results.items():
        p05 = "YES" if r["pass_05"] else "no"
        p07 = "YES" if r["pass_07"] else "no"
        p10 = "YES" if r["pass_10"] else "no"
        period = f"{r['start'][:10]}→{r['end'][:10]}"
        print(f"  {name:<35} {period:<22} {r['full_sharpe']:+8.3f} "
              f"{r['worst_2y_sharpe']:+9.3f} {p05:>9} {p07:>9} {p10:>9}")

    # Our portfolios
    our_portfolios = portfolios_data.get("portfolios", {})
    for pname, pdata in our_portfolios.items():
        if "error" in pdata:
            continue
        w2y = pdata.get("worst_2y_sharpe", -99)
        sr = pdata.get("sharpe", 0)
        p05 = "YES" if w2y > -0.5 else "no"
        p07 = "YES" if w2y > -0.7 else "no"
        p10 = "YES" if w2y > -1.0 else "no"
        print(f"  Our {pname:<30} {'full period':<22} {sr:+8.3f} "
              f"{w2y:+9.3f} {p05:>9} {p07:>9} {p10:>9}")

    # 5.5.3.5 — Threshold recommendation
    print("\n--- 5.5.3.5: Threshold recalibration ---")
    bm_worst2y = [r["worst_2y_sharpe"] for r in benchmark_results.values()]

    if bm_worst2y:
        median_w2y = np.median(bm_worst2y)
        p25_w2y = np.percentile(bm_worst2y, 25)
        n_pass_05 = sum(1 for w in bm_worst2y if w > -0.5)
        n_pass_07 = sum(1 for w in bm_worst2y if w > -0.7)
        n_pass_10 = sum(1 for w in bm_worst2y if w > -1.0)
        n_total = len(bm_worst2y)

        print(f"\n  Benchmark worst-2Y distribution:")
        print(f"    Median: {median_w2y:+.3f}")
        print(f"    25th percentile: {p25_w2y:+.3f}")
        print(f"    Pass -0.5: {n_pass_05}/{n_total} ({100*n_pass_05/n_total:.0f}%)")
        print(f"    Pass -0.7: {n_pass_07}/{n_total} ({100*n_pass_07/n_total:.0f}%)")
        print(f"    Pass -1.0: {n_pass_10}/{n_total} ({100*n_pass_10/n_total:.0f}%)")

        # Recommended threshold: 25th percentile of benchmarks
        # (i.e., "at least as good as the bottom quartile of comparable active funds")
        recommended = round(p25_w2y, 1)
        # But floor at -1.0 (beyond that we're just rationalizing)
        recommended = max(recommended, -1.0)

        # Sensitivity for our portfolios
        our_w2y = [pdata.get("worst_2y_sharpe", -99)
                    for pdata in our_portfolios.values() if "error" not in pdata]

        print(f"\n  RECOMMENDED THRESHOLD: {recommended}")
        print(f"  Justification: 25th percentile of {n_total} active managed-futures benchmarks")
        print(f"  Our portfolios passing at {recommended}: "
              f"{sum(1 for w in our_w2y if w > recommended)}/{len(our_w2y)}")
    else:
        recommended = -0.7
        print("  ⚠ No benchmark data available — defaulting to -0.7")

    return {
        "benchmarks": benchmark_results,
        "recommended_threshold": recommended,
        "benchmark_median_worst_2y": round(float(median_w2y), 4) if bm_worst2y else None,
        "benchmark_p25_worst_2y": round(float(p25_w2y), 4) if bm_worst2y else None,
    }


# ================================================================
# TASK 5.5.4 — Q3 DEPLOYMENT AUDIT
# ================================================================
def task_554():
    """Task 5.5.4: Q3 auto-pause fix and deployment verification."""
    print("\n" + "=" * 70)
    print("TASK 5.5.4 — Q3 DEPLOYMENT AUDIT")
    print("=" * 70)

    # Load existing Q3 config
    config_path = os.path.join(RESULTS_DIR, "phase5_q3_deployment_config.json")
    q3_results_path = os.path.join(RESULTS_DIR, "phase5_q3_deployment_results.json")

    q3_config = {}
    q3_results = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            q3_config = json.load(f)
    if os.path.exists(q3_results_path):
        with open(q3_results_path) as f:
            q3_results = json.load(f)

    historical_max_dd = q3_results.get("backtest", {}).get("max_drawdown", 0.235)
    current_pause = q3_config.get("auto_pause", {}).get("max_drawdown_pct", 0.15)

    print(f"\n--- 5.5.4.1: Auto-pause contradiction ---")
    print(f"  Historical max DD: {historical_max_dd:.1%}")
    print(f"  Current auto-pause: {current_pause:.1%}")
    print(f"  Gap: strategy WOULD HAVE been paused {historical_max_dd - current_pause:.1%} before worst DD")

    # Resolution: Option C (keep 15% pause with explicit resume protocol)
    print(f"\n  Resolution: OPTION C — keep 15% auto-pause + explicit resume protocol")
    print(f"  Rationale: For demo phase, information-gathering > return-maximization")
    print(f"  If paused, resume criteria:")
    print(f"    1. Human review within 48 hours")
    print(f"    2. Check: has volatility normalized? (trailing 20-day vol < 1.5× long-term)")
    print(f"    3. Check: are costs within modeled range? (slippage < 2× expected)")
    print(f"    4. Check: has strategy entered new signal regime? (z-score near zero = no signal)")
    print(f"    5. If 2 of 3 checks pass → resume. Otherwise → continue pause.")

    resume_protocol = {
        "trigger": "auto_pause_at_15pct_dd",
        "review_deadline_hours": 48,
        "resume_checks": [
            "trailing_20d_vol < 1.5x long_term_vol",
            "realized_slippage < 2x modeled_slippage",
            "current_z_score near zero (no active signal)"
        ],
        "resume_threshold": "2 of 3 checks pass",
        "option_selected": "C",
        "rationale": "Demo phase prioritizes information gathering over return maximization"
    }

    # 5.5.4.2 — Deployment status check
    print(f"\n--- 5.5.4.2: Deployment status ---")
    plugin_path = q3_results.get("plugin_path",
                                  "/home/harveybc/Documents/GitHub/lts/plugins_strategy/eurusd_mr_strategy.py")
    plugin_exists = os.path.exists(plugin_path)
    log_dir = os.path.join(RESULTS_DIR, "q3_deployment_logs")
    log_dir_exists = os.path.exists(log_dir)

    print(f"  Strategy plugin: {'EXISTS' if plugin_exists else 'MISSING'} ({plugin_path})")
    print(f"  Log directory: {'EXISTS' if log_dir_exists else 'MISSING'} ({log_dir})")

    # Check if OANDA credentials are configured
    oanda_account = os.environ.get("OANDA_ACCOUNT_ID", "")
    oanda_token = os.environ.get("OANDA_ACCESS_TOKEN", "")
    print(f"  OANDA_ACCOUNT_ID: {'SET' if oanda_account else 'NOT SET'}")
    print(f"  OANDA_ACCESS_TOKEN: {'SET' if oanda_token else 'NOT SET'}")

    if not oanda_account or not oanda_token:
        print(f"\n  ⚠ OANDA credentials not set — demo trading NOT live")
        print(f"  To activate: export OANDA_ACCOUNT_ID=xxx OANDA_ACCESS_TOKEN=xxx")
        live_status = "NOT_LIVE_CREDENTIALS_MISSING"
    else:
        live_status = "CREDENTIALS_CONFIGURED"

    # Check for any trades
    trades_file = os.path.join(log_dir, "trades.jsonl")
    n_trades = 0
    if os.path.exists(trades_file):
        with open(trades_file) as f:
            n_trades = sum(1 for _ in f)
    print(f"  Trades logged: {n_trades}")

    print(f"\n  DEPLOYMENT STATUS: {live_status}")
    if live_status != "CREDENTIALS_CONFIGURED":
        print(f"  Action required: Set OANDA practice account credentials to begin demo trading")
        print(f"  The strategy plugin and monitoring infrastructure are ready")

    return {
        "auto_pause_resolution": resume_protocol,
        "plugin_exists": plugin_exists,
        "log_dir_exists": log_dir_exists,
        "live_status": live_status,
        "n_trades_logged": n_trades,
        "corrected_config": {
            "auto_pause": {
                "max_drawdown_pct": 0.15,
                "resume_protocol": resume_protocol,
            }
        }
    }


# ================================================================
# TASK 5.5.5 — REVISED SYNTHESIS
# ================================================================
def task_555(q1_data, q2_data, q3_data, benchmark_data):
    """Task 5.5.5: Final synthesis with internal consistency."""
    print("\n" + "=" * 70)
    print("TASK 5.5.5 — REVISED SYNTHESIS AND TERMINAL STATE")
    print("=" * 70)

    # Get corrected threshold
    threshold = benchmark_data.get("recommended_threshold", -0.7)

    # Best corrected portfolio
    portfolios = q1_data.get("portfolios", {})
    best_name = None
    best_w2y = -999
    for pname, pdata in portfolios.items():
        if "error" in pdata:
            continue
        w2y = pdata.get("worst_2y_sharpe", -99)
        if w2y > best_w2y:
            best_w2y = w2y
            best_name = pname

    # OOS validation
    oos = q2_data  # task_552 results
    robust = oos.get("robust_portfolios", [])
    comparison = oos.get("comparison", {})

    # Check: does best portfolio pass OOS?
    best_oos_sr = None
    best_oos_w2y = None
    if best_name and best_name.replace("_equal_weight", "").replace("_inverse_vol", "").replace("_inverse_worst_window", "").replace("_risk_parity", "").replace("_hedged_pairs", ""):
        # Map portfolio names
        pname_map = {
            "P1_equal_weight": "P1",
            "P2_inverse_vol": "P2",
            "P3_inverse_worst_window": "P3",
            "P4_risk_parity": "P4",
            "P5_hedged_pairs": "P5",
        }
        mapped = pname_map.get(best_name, "")
        if mapped in comparison:
            best_oos_sr = comparison[mapped].get("oos_sharpe")
            best_oos_w2y = comparison[mapped].get("oos_worst_2y")

    print(f"\n--- Decision inputs ---")
    print(f"  Recalibrated threshold: {threshold}")
    print(f"  Best corrected portfolio: {best_name}")
    print(f"  Best IS worst-2Y: {best_w2y:+.3f}")
    print(f"  Best OOS Sharpe: {best_oos_sr:+.3f}" if best_oos_sr is not None else "  Best OOS Sharpe: N/A")
    print(f"  Best OOS worst-2Y: {best_oos_w2y:+.3f}" if best_oos_w2y is not None else "  Best OOS worst-2Y: N/A")
    print(f"  Oracle-free cells: {q1_data.get('n_cells', 0)}")
    print(f"  Robust portfolios (≤30% OOS degradation): {robust}")
    print(f"  Q3 deployment status: {q3_data.get('live_status', 'unknown')}")

    # Terminal state determination
    print(f"\n--- Terminal state determination ---")

    # Check Terminal 1 criteria
    t1_pass_is = best_w2y > threshold
    t1_pass_oos = best_oos_w2y is not None and best_oos_w2y > threshold
    t1_oos_robust = best_name and any(pname_map.get(best_name, "") == r for r in robust)

    if t1_pass_is and t1_pass_oos and t1_oos_robust:
        terminal = "TERMINAL_1"
        terminal_desc = "DEPLOY PORTFOLIO"
        evidence = (f"{best_name} passes IS (worst2Y={best_w2y:+.3f} > {threshold}) "
                    f"AND OOS (worst2Y={best_oos_w2y:+.3f} > {threshold}) "
                    f"with ≤30% SR degradation")
    elif t1_pass_is and (best_oos_sr is not None and best_oos_sr > 0):
        terminal = "TERMINAL_2_5"
        terminal_desc = "DEPLOY EUR/USD MR + MONITOR PORTFOLIO"
        evidence = (f"{best_name} passes IS (worst2Y={best_w2y:+.3f} > {threshold}) "
                    f"but OOS {'passes' if t1_pass_oos else 'fails'} threshold. "
                    f"OOS SR positive ({best_oos_sr:+.3f}) suggests real but modest edge. "
                    f"Deploy Q3 EUR/USD MR immediately, evaluate portfolio after 90-day observation.")
    elif best_oos_sr is not None and best_oos_sr > 0:
        terminal = "TERMINAL_2"
        terminal_desc = "DEPLOY EUR/USD MR ONLY"
        evidence = (f"Portfolio IS worst2Y={best_w2y:+.3f} {'passes' if t1_pass_is else 'fails'} "
                    f"threshold {threshold}. OOS SR positive but modest. "
                    f"Single-strategy EUR/USD MR deployment is the defensible option.")
    else:
        terminal = "TERMINAL_3"
        terminal_desc = "DOCUMENTED CLOSURE"
        evidence = (f"Corrected portfolio evidence does not support deployment. "
                    f"IS worst2Y={best_w2y:+.3f}, OOS SR={best_oos_sr if best_oos_sr else 'N/A'}. "
                    f"Project closes with comprehensive research documentation.")

    print(f"\n  TERMINAL STATE: {terminal}")
    print(f"  Description: {terminal_desc}")
    print(f"  Evidence: {evidence}")

    return {
        "terminal_state": terminal,
        "terminal_description": terminal_desc,
        "evidence": evidence,
        "threshold_used": threshold,
        "best_portfolio": best_name,
        "best_is_worst2y": best_w2y,
        "best_oos_sharpe": best_oos_sr,
        "best_oos_worst2y": best_oos_w2y,
        "oos_robust_portfolios": robust,
        "q3_status": q3_data.get("live_status"),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  PHASE 5.5 — CORRECTIVE AUDIT BEFORE TERMINAL STATE DECISION     ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Task 5.5.1 — Oracle-free portfolio
    q1_data = task_551()
    if "error" in q1_data:
        print(f"\n  ⚠ Task 5.5.1 failed: {q1_data['error']}")
        # Still continue with other tasks
        q1_data["portfolios"] = {}

    # Task 5.5.2 — OOS validation
    df_weekly = q1_data.pop("cell_returns_weekly", None)
    if df_weekly is not None and len(df_weekly) > 0:
        q2_data = task_552(q1_data.get("included_cells", {}), df_weekly)
    else:
        q2_data = {"error": "no weekly data from task 5.5.1",
                    "robust_portfolios": [], "comparison": {}}

    # Task 5.5.3 — Proper benchmarks
    benchmark_data = task_553(q1_data)

    # Task 5.5.4 — Q3 audit
    q3_data = task_554()

    # Task 5.5.5 — Synthesis
    synthesis = task_555(q1_data, q2_data, q3_data, benchmark_data)

    # Save all results
    output = {
        "phase": "5.5",
        "date": datetime.date.today().isoformat(),
        "task_551_corrected_portfolio": {k: v for k, v in q1_data.items()
                                          if k != "cell_returns_weekly"},
        "task_552_oos_validation": q2_data,
        "task_553_benchmark_comparison": benchmark_data,
        "task_554_q3_audit": q3_data,
        "task_555_synthesis": synthesis,
    }

    output_path = os.path.join(RESULTS_DIR, "phase5_5_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
