#!/usr/bin/env python3
"""
Phase 5 — Question 1: Does Portfolio Diversification Rescue Killed Strategies?

Tasks 1.1-1.6: Select candidate cells, compute per-cell return series,
correlation analysis, construct P1-P5 portfolios, regime analysis, decision.
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import download_asset_data, generate_oracle_signal
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe, max_drawdown
)
from trading_research.audit_noise_budget import get_strategy_positions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXTENDED_DATA = os.path.join(os.path.dirname(__file__), "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXTENDED_DATA, exist_ok=True)

YFINANCE_TICKERS = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X", "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X", "AUD/JPY": "AUDJPY=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F", "CL": "CL=F",
}

EXTENDED_START = {
    "BTC/USD": "2014-09-17", "ETH/USD": "2017-08-01",
    "XAU/USD": "2000-01-01", "XAG/USD": "2000-01-01", "CL": "2000-01-01",
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

# ============================================================
# TASK 1.1 — Candidate cells
# ============================================================
# Cells with edge > 0 and worst2Y > -3.0 from Phase 3-4 results
CANDIDATE_CELLS = [
    # Phase 3.5 survivors (oracle-based, σ=10)
    {"asset": "EUR/USD", "timeframe": "daily",  "strategy": "mean_reversion", "source": "phase3_oracle"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "mean_reversion", "source": "phase3_oracle"},
    {"asset": "XAU/USD", "timeframe": "daily",  "strategy": "momentum",       "source": "phase3_oracle"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum",       "source": "phase3_oracle"},
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum",       "source": "phase3_oracle"},
    {"asset": "AUD/USD", "timeframe": "weekly", "strategy": "vol_regime_switch", "source": "phase3_oracle"},
    {"asset": "EUR/JPY", "timeframe": "weekly", "strategy": "vol_regime_switch", "source": "phase3_oracle"},
    # Phase 4 Track A — pure MR (no oracle)
    {"asset": "EUR/USD", "timeframe": "daily",  "strategy": "pure_mr",        "source": "phase4_trackA"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "pure_mr",        "source": "phase4_trackA"},
    # Phase 4 Track C — academic (best performers)
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "tsmom",          "source": "phase4_trackC"},
    {"asset": "XAU/USD", "timeframe": "daily",  "strategy": "tsmom",          "source": "phase4_trackC"},
    {"asset": "USD/JPY", "timeframe": "daily",  "strategy": "dual_momentum",  "source": "phase4_trackC"},
]

# Target annualized vol for position sizing (10% as per work plan)
TARGET_VOL = 0.10
PPY_DAILY = 252


def load_extended_data(asset, timeframe):
    """Download/load extended price data."""
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
    """TSMOM: sign of 12-month return × inverse-vol, monthly rebalance."""
    monthly_close = pd.Series(close_daily, index=dates).resample("ME").last().dropna()
    monthly_dates = monthly_close.index
    positions = np.zeros(len(log_ret_daily))

    for i in range(lookback_months, len(monthly_dates) - 1):
        ret_12m = np.log(monthly_close.iloc[i] + 1e-12) - np.log(monthly_close.iloc[i - lookback_months] + 1e-12)
        signal = np.sign(ret_12m)

        # Inverse vol
        mask = (dates >= monthly_dates[max(0, i - 1)]) & (dates <= monthly_dates[i])
        recent = log_ret_daily[mask]
        vol = np.std(recent[-min(252, len(recent)):]) * np.sqrt(252) if len(recent) > 20 else 0.10
        size = min(TARGET_VOL / max(vol, 0.01), 3.0)

        # Apply next month
        next_date = monthly_dates[i + 1] if i + 1 < len(monthly_dates) else dates[-1]
        mask_next = (dates > monthly_dates[i]) & (dates <= next_date)
        positions[mask_next] = signal * size

    return positions


def run_dual_momentum(log_ret_daily, close_daily, dates, all_fx_data, asset, lookback_months=12):
    """Dual Momentum: long if absolute + relative momentum positive."""
    mc = pd.Series(close_daily, index=dates).resample("ME").last().dropna()
    monthly_dates = mc.index

    # Monthly closes for all FX
    all_mc = {}
    for a, (lr, cl, dt) in all_fx_data.items():
        all_mc[a] = pd.Series(cl, index=dt).resample("ME").last().dropna()

    positions = np.zeros(len(log_ret_daily))

    common = monthly_dates
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

        # All returns
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


# ============================================================
# TASK 1.2 — Compute per-cell daily return series
# ============================================================
def compute_cell_returns(cell, all_fx_data=None, seed=42):
    """
    Compute daily net return series for a cell.
    Returns (dates, net_returns, positions, metadata) or None.
    """
    asset = cell["asset"]
    timeframe = cell["timeframe"]
    strategy = cell["strategy"]
    source = cell["source"]

    df = load_extended_data(asset, timeframe)
    if df is None:
        return None

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    dates = df.index
    abs_ret = np.abs(log_ret)
    ppy = periods_per_year_for_timeframe(timeframe)

    # Get positions based on strategy type
    if strategy in ("momentum", "mean_reversion", "breakout", "carry_momentum", "vol_regime_switch"):
        oracle = generate_oracle_signal(log_ret, noise_sigma=10.0, horizon=1, seed=seed)
        positions = get_strategy_positions(strategy, log_ret, oracle, close)
    elif strategy == "pure_mr":
        positions = run_pure_mr(log_ret, close)
    elif strategy == "tsmom":
        if timeframe == "daily":
            positions = run_tsmom(log_ret, close, dates)
        else:
            return None
    elif strategy == "dual_momentum":
        if all_fx_data is None:
            return None
        positions = run_dual_momentum(log_ret, close, dates, all_fx_data, asset)
    else:
        return None

    # Compute net returns
    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], asset, abs_ret[:-1])

    # Vol-target sizing: scale returns to 10% target vol
    realized_vol = np.std(net) * np.sqrt(ppy) if np.std(net) > 0 else 0.10
    vol_scalar = TARGET_VOL / max(realized_vol, 0.01)
    vol_scalar = min(vol_scalar, 5.0)  # cap at 5x
    net_scaled = net * vol_scalar

    # If weekly, forward-fill to daily
    if timeframe == "weekly":
        daily_dates = pd.date_range(start=dates[1], end=dates[-1], freq="B")
        weekly_series = pd.Series(net_scaled, index=dates[1:len(net_scaled)+1])
        daily_series = weekly_series.reindex(daily_dates, method="ffill").fillna(0)
        return_dates = daily_series.index
        daily_net = daily_series.values / 5.0  # distribute weekly return across 5 days
    else:
        return_dates = dates[1:len(net_scaled)+1]
        daily_net = net_scaled

    # Sharpe and worst window
    sr = annualized_sharpe(daily_net, PPY_DAILY)
    bh_sr = annualized_sharpe(log_ret[1:], ppy)

    rolling = rolling_window_evaluation(daily_net, PPY_DAILY)

    metadata = {
        "asset": asset,
        "timeframe": timeframe,
        "strategy": strategy,
        "source": source,
        "sharpe": round(sr, 4),
        "bh_sharpe": round(bh_sr, 4),
        "edge_sharpe": round(sr - bh_sr, 4),
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "regime_robustness": rolling["regime_robustness"],
        "vol_scalar": round(vol_scalar, 4),
        "n_bars_original": len(log_ret),
        "n_bars_daily": len(daily_net),
    }

    return return_dates, daily_net, positions, metadata


# ============================================================
# TASK 1.3 — Correlation matrix and temporal overlap
# ============================================================
def compute_correlations(cell_returns_dict):
    """Compute pairwise correlation of daily returns and rolling 2Y Sharpe."""
    cell_names = list(cell_returns_dict.keys())
    n = len(cell_names)

    # Build aligned dataframe
    all_series = {}
    for name, (dates, rets, _, _) in cell_returns_dict.items():
        all_series[name] = pd.Series(rets, index=dates)

    df = pd.DataFrame(all_series)
    df = df.dropna(how="all")
    df = df.fillna(0)

    # Return correlations
    ret_corr = df.corr()

    # Rolling 2Y Sharpe correlations
    window = 252 * 2
    rolling_sharpe = pd.DataFrame()
    for col in df.columns:
        rolling_mean = df[col].rolling(window).mean()
        rolling_std = df[col].rolling(window).std()
        rolling_sharpe[col] = (rolling_mean / (rolling_std + 1e-12)) * np.sqrt(252)

    rolling_sharpe = rolling_sharpe.dropna()
    sharpe_corr = rolling_sharpe.corr() if len(rolling_sharpe) > 10 else pd.DataFrame()

    # Simultaneous bad periods: for each pair, count months both are in worst 10%
    bad_overlap = {}
    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = cell_names[i], cell_names[j]
            if name_i in df.columns and name_j in df.columns:
                # Monthly returns
                monthly = df[[name_i, name_j]].resample("ME").sum().dropna()
                if len(monthly) < 20:
                    continue
                thresh_i = monthly[name_i].quantile(0.10)
                thresh_j = monthly[name_j].quantile(0.10)
                bad_i = monthly[name_i] <= thresh_i
                bad_j = monthly[name_j] <= thresh_j
                both_bad = (bad_i & bad_j).sum()
                bad_overlap[f"{name_i} / {name_j}"] = {
                    "both_bad_months": int(both_bad),
                    "total_months": len(monthly),
                    "pct_overlap": round(100 * both_bad / max(len(monthly), 1), 1),
                }

    return {
        "return_correlation": ret_corr.round(4).to_dict(),
        "sharpe_correlation": sharpe_corr.round(4).to_dict() if len(sharpe_corr) > 0 else {},
        "bad_period_overlap": bad_overlap,
        "common_period": {
            "start": str(df.index[0]) if len(df) > 0 else None,
            "end": str(df.index[-1]) if len(df) > 0 else None,
            "n_days": len(df),
        }
    }


# ============================================================
# TASK 1.4 — Construct candidate portfolios
# ============================================================
def construct_portfolios(cell_returns_dict):
    """Build P1-P5 portfolios and evaluate each."""
    cell_names = list(cell_returns_dict.keys())
    n = len(cell_names)

    # Build aligned dataframe
    all_series = {}
    metadata_dict = {}
    for name, (dates, rets, _, meta) in cell_returns_dict.items():
        all_series[name] = pd.Series(rets, index=dates)
        metadata_dict[name] = meta
    df = pd.DataFrame(all_series).fillna(0)
    df = df.loc[df.index >= df.dropna(how="all").index[0]] if len(df) > 0 else df

    # Common overlap where at least 3 cells have data
    valid_counts = df.ne(0).sum(axis=1)
    df_common = df[valid_counts >= max(3, n // 2)]

    if len(df_common) < 504:  # need at least 2 years
        return {"error": "insufficient common data"}

    results = {}

    # P1 — Equal weight
    p1_ret = df_common.mean(axis=1).values
    results["P1_equal_weight"] = evaluate_portfolio(p1_ret, "P1_equal_weight", df_common.index)

    # P2 — Inverse volatility weight
    vols = df_common.std()
    inv_vol = 1.0 / (vols + 1e-12)
    w2 = inv_vol / inv_vol.sum()
    p2_ret = (df_common * w2).sum(axis=1).values
    results["P2_inverse_vol"] = evaluate_portfolio(p2_ret, "P2_inverse_vol", df_common.index)
    results["P2_inverse_vol"]["weights"] = {k: round(v, 4) for k, v in w2.items()}

    # P3 — Inverse worst-window weight
    worst_windows = {}
    for name in cell_names:
        ww = metadata_dict.get(name, {}).get("worst_window_sharpe", -2.0)
        worst_windows[name] = abs(ww) if ww < 0 else 0.01  # smaller magnitude = heavier weight
    inv_ww = {k: 1.0 / max(v, 0.01) for k, v in worst_windows.items() if k in df_common.columns}
    total_inv_ww = sum(inv_ww.values())
    w3 = {k: v / total_inv_ww for k, v in inv_ww.items()}
    p3_ret = sum(df_common[k] * w for k, w in w3.items()).values
    results["P3_inverse_worst_window"] = evaluate_portfolio(p3_ret, "P3_inverse_worst_window", df_common.index)
    results["P3_inverse_worst_window"]["weights"] = {k: round(v, 4) for k, v in w3.items()}

    # P4 — Risk parity (simplified: inverse variance, then rescaled)
    variances = df_common.var()
    inv_var = 1.0 / (variances + 1e-12)
    w4 = inv_var / inv_var.sum()
    p4_ret = (df_common * w4).sum(axis=1).values
    results["P4_risk_parity"] = evaluate_portfolio(p4_ret, "P4_risk_parity", df_common.index)
    results["P4_risk_parity"]["weights"] = {k: round(v, 4) for k, v in w4.items()}

    # P5 — Hedged pairs: find 3-4 lowest correlation pairs
    corr = df_common.corr()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if cell_names[i] in corr.columns and cell_names[j] in corr.columns:
                c = corr.loc[cell_names[i], cell_names[j]]
                pairs.append((cell_names[i], cell_names[j], c))
    pairs.sort(key=lambda x: x[2])
    selected_pairs = pairs[:min(4, len(pairs))]

    if selected_pairs:
        pair_cells = set()
        for a, b, _ in selected_pairs:
            pair_cells.add(a)
            pair_cells.add(b)
        w5 = {k: 1.0 / len(pair_cells) for k in pair_cells if k in df_common.columns}
        p5_ret = sum(df_common[k] * w for k, w in w5.items()).values
        results["P5_hedged_pairs"] = evaluate_portfolio(p5_ret, "P5_hedged_pairs", df_common.index)
        results["P5_hedged_pairs"]["weights"] = {k: round(v, 4) for k, v in w5.items()}
        results["P5_hedged_pairs"]["selected_pairs"] = [(a, b, round(c, 4)) for a, b, c in selected_pairs]
    else:
        results["P5_hedged_pairs"] = {"error": "insufficient pairs"}

    return results


def evaluate_portfolio(returns, name, dates):
    """Evaluate a portfolio return series."""
    sr = annualized_sharpe(returns, PPY_DAILY)
    rolling = rolling_window_evaluation(returns, PPY_DAILY)

    equity = np.cumsum(returns)
    eq_curve = np.exp(equity)
    peak = np.maximum.accumulate(eq_curve)
    dd = (peak - eq_curve) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    # Realized vol
    realized_vol = np.std(returns) * np.sqrt(PPY_DAILY)

    return {
        "name": name,
        "sharpe": round(sr, 4),
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "regime_robustness": rolling["regime_robustness"],
        "max_drawdown": round(max_dd, 4),
        "realized_vol": round(realized_vol, 4),
        "n_windows": rolling["n_windows"],
        "n_days": len(returns),
        "total_return": round(float(np.exp(np.sum(returns)) - 1), 4),
    }


# ============================================================
# TASK 1.5 — Regime-by-regime analysis
# ============================================================
def regime_analysis(portfolio_returns, dates, portfolio_name):
    """Compute Sharpe per macro regime for a portfolio."""
    ret_series = pd.Series(portfolio_returns, index=dates)
    regime_results = {}

    for regime_name, (start, end) in REGIMES.items():
        mask = (ret_series.index >= start) & (ret_series.index <= end)
        r = ret_series[mask]
        if len(r) < 50:
            regime_results[regime_name] = {"sharpe": None, "n_days": 0}
            continue
        sr = annualized_sharpe(r.values, PPY_DAILY)
        regime_results[regime_name] = {
            "sharpe": round(sr, 4),
            "n_days": len(r),
        }

    positive_regimes = sum(1 for v in regime_results.values()
                          if v.get("sharpe") is not None and v["sharpe"] > 0)
    return {
        "regimes": regime_results,
        "positive_regimes": positive_regimes,
        "total_regimes": sum(1 for v in regime_results.values() if v.get("sharpe") is not None),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 5 — Q1: PORTFOLIO DIVERSIFICATION")
    print("=" * 70)

    # Pre-load FX data for dual momentum
    print("\n--- Loading FX data for dual momentum ---")
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

    # Task 1.2: Compute per-cell return series
    print("\n--- Task 1.2: Computing per-cell return series ---")
    cell_returns = {}
    cell_metadata = []

    for cell in CANDIDATE_CELLS:
        cell_key = f"{cell['asset']}_{cell['timeframe']}_{cell['strategy']}"
        print(f"\n  {cell_key}:")

        result = compute_cell_returns(cell, all_fx_data=all_fx_data)
        if result is None:
            print("    ⚠ No data or unsupported")
            continue

        dates, net, positions, meta = result
        if len(net) < 252:
            print(f"    ⚠ Too short: {len(net)} days")
            continue

        # Filter: edge > 0 and worst2Y > -3.0
        if meta["worst_window_sharpe"] < -3.0:
            print(f"    ⚠ Worst2Y={meta['worst_window_sharpe']:+.3f} < -3.0, excluded")
            continue

        cell_returns[cell_key] = (dates, net, positions, meta)
        cell_metadata.append(meta)
        print(f"    SR={meta['sharpe']:+.3f}, Edge={meta['edge_sharpe']:+.3f}, "
              f"Worst2Y={meta['worst_window_sharpe']:+.3f}, "
              f"VolScalar={meta['vol_scalar']:.2f}x, Bars={meta['n_bars_daily']}")

    print(f"\n  Total cells included: {len(cell_returns)}")
    if len(cell_returns) < 2:
        print("  ⚠ Insufficient cells for portfolio construction")
        output = {"error": "fewer than 2 cells available", "cells": cell_metadata}
        with open(os.path.join(RESULTS_DIR, "phase5_q1_portfolio_results.json"), "w") as f:
            json.dump(output, f, indent=2, default=str)
        return

    # Task 1.3: Correlation analysis
    print("\n--- Task 1.3: Correlation analysis ---")
    corr_results = compute_correlations(cell_returns)
    print(f"  Common period: {corr_results['common_period']['start']} to {corr_results['common_period']['end']}")
    print(f"  ({corr_results['common_period']['n_days']} trading days)")

    # Print correlation matrix
    ret_corr = corr_results["return_correlation"]
    if ret_corr:
        print("\n  Return correlations:")
        names = list(ret_corr.keys())
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if j > i:
                    c = ret_corr.get(n1, {}).get(n2, 0)
                    print(f"    {n1} / {n2}: {c:+.3f}")

    # Bad period overlap
    print("\n  Simultaneous bad periods:")
    for pair, info in corr_results.get("bad_period_overlap", {}).items():
        print(f"    {pair}: {info['both_bad_months']} months ({info['pct_overlap']:.1f}%)")

    # Task 1.4: Portfolio construction
    print("\n--- Task 1.4: Portfolio construction ---")
    portfolio_results = construct_portfolios(cell_returns)

    for pname, presult in portfolio_results.items():
        if isinstance(presult, dict) and "error" not in presult:
            print(f"\n  {pname}:")
            print(f"    Sharpe: {presult['sharpe']:+.3f}")
            print(f"    Worst 2Y: {presult['worst_window_sharpe']:+.3f}")
            print(f"    Max DD: {presult['max_drawdown']:.1%}")
            print(f"    Realized Vol: {presult['realized_vol']:.1%}")
            print(f"    Total Return: {presult['total_return']:+.1%}")
            if "weights" in presult:
                print(f"    Weights: {presult['weights']}")

    # Task 1.5: Regime analysis per portfolio
    print("\n--- Task 1.5: Regime analysis ---")
    # Reconstruct portfolio returns for regime analysis
    all_series = {}
    for name, (dates, rets, _, _) in cell_returns.items():
        all_series[name] = pd.Series(rets, index=dates)
    df = pd.DataFrame(all_series).fillna(0)
    valid_counts = df.ne(0).sum(axis=1)
    df_common = df[valid_counts >= max(3, len(cell_returns) // 2)]

    regime_results_all = {}
    if len(df_common) > 504:
        # P1 regime analysis
        p1_ret = df_common.mean(axis=1).values
        p1_dates = df_common.index
        regime_results_all["P1"] = regime_analysis(p1_ret, p1_dates, "P1")
        print(f"\n  P1 Regime breakdown:")
        for rname, rdata in regime_results_all["P1"]["regimes"].items():
            if rdata["sharpe"] is not None:
                print(f"    {rname}: SR={rdata['sharpe']:+.3f} ({rdata['n_days']} days)")
        print(f"    Positive regimes: {regime_results_all['P1']['positive_regimes']}/{regime_results_all['P1']['total_regimes']}")

        # Do regime analysis for all portfolios
        for pname in ["P2_inverse_vol", "P3_inverse_worst_window", "P4_risk_parity", "P5_hedged_pairs"]:
            pr = portfolio_results.get(pname, {})
            if isinstance(pr, dict) and "weights" in pr:
                w = pr["weights"]
                p_ret = sum(df_common[k] * w_val for k, w_val in w.items() if k in df_common.columns).values
                regime_results_all[pname] = regime_analysis(p_ret, df_common.index, pname)
                pos_reg = regime_results_all[pname]["positive_regimes"]
                tot_reg = regime_results_all[pname]["total_regimes"]
                print(f"\n  {pname} positive regimes: {pos_reg}/{tot_reg}")
                for rname, rdata in regime_results_all[pname]["regimes"].items():
                    if rdata["sharpe"] is not None:
                        print(f"    {rname}: SR={rdata['sharpe']:+.3f}")

    # Task 1.6: Decision
    print("\n" + "=" * 70)
    print("Q1 — DECISION")
    print("=" * 70)

    best_portfolio = None
    best_worst2y = -999

    for pname, presult in portfolio_results.items():
        if not isinstance(presult, dict) or "error" in presult:
            continue
        w2y = presult["worst_window_sharpe"]
        if w2y > best_worst2y:
            best_worst2y = w2y
            best_portfolio = pname

    # Classification
    if best_worst2y > -0.5:
        # Check edge and regime criteria
        best_sr = portfolio_results[best_portfolio]["sharpe"]
        pos_reg = regime_results_all.get(best_portfolio, {}).get("positive_regimes", 0)
        if best_sr > 0.4 and pos_reg >= 4:
            outcome = "OUTCOME_1_RESCUES"
            notes = (f"{best_portfolio} produces SR={best_sr:+.3f}, worst2Y={best_worst2y:+.3f} > -0.5, "
                     f"{pos_reg} positive regimes")
        else:
            outcome = "OUTCOME_1_PARTIAL"
            notes = f"{best_portfolio} worst2Y={best_worst2y:+.3f} > -0.5 but SR={best_sr:+.3f} or regimes={pos_reg}"
    elif best_worst2y > -0.8:
        outcome = "OUTCOME_2_IMPROVES_NOT_THRESHOLD"
        notes = f"Best portfolio {best_portfolio} worst2Y={best_worst2y:+.3f} between -0.5 and -0.8"
    elif best_worst2y > -1.5:
        outcome = "OUTCOME_2_MARGINAL"
        notes = f"Best portfolio {best_portfolio} worst2Y={best_worst2y:+.3f} between -0.8 and -1.5"
    else:
        outcome = "OUTCOME_3_NO_RESCUE"
        notes = f"All portfolios worst2Y < -1.5, diversification does not help"

    print(f"\n  Best portfolio: {best_portfolio}")
    print(f"  Best worst 2Y: {best_worst2y:+.3f}")
    print(f"  Outcome: {outcome}")
    print(f"  {notes}")

    # Assemble output
    output = {
        "question": "Q1",
        "outcome": outcome,
        "outcome_notes": notes,
        "best_portfolio": best_portfolio,
        "best_worst_2y": best_worst2y,
        "n_cells_included": len(cell_returns),
        "cell_metadata": cell_metadata,
        "correlation_analysis": corr_results,
        "portfolios": portfolio_results,
        "regime_analysis": regime_results_all,
    }

    output_path = os.path.join(RESULTS_DIR, "phase5_q1_portfolio_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
