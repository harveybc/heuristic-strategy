#!/usr/bin/env python3
"""
Phase 3.5 — Task 3: Historical Data Extension & Oracle Recomputation

Task 3.1: Download extended historical data for all survivor assets
Task 3.2: Recompute oracle sweep on extended history
Task 3.3: Regime change detection across 5 macro regimes

Distributed execution: use --assets and --output flags for per-machine runs.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import generate_oracle_signal, NOISE_GRID
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe
)
from trading_research.audit_noise_budget import get_strategy_positions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "extended_data")

# yfinance tickers
YFINANCE_TICKERS = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "XAU/USD": "GC=F",
    "XAG/USD": "SI=F",
    "CL": "CL=F",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

# Target start dates for extended history
EXTENDED_START = {
    "BTC/USD": "2014-09-17",
    "ETH/USD": "2017-08-01",
    "XAU/USD": "2000-01-01",
    "XAG/USD": "2000-01-01",
    "CL": "2000-01-01",
    "EUR/USD": "2003-01-01",
    "USD/JPY": "2003-01-01",
    "GBP/USD": "2003-01-01",
    "AUD/USD": "2003-01-01",
    "AUD/JPY": "2003-01-01",
    "EUR/JPY": "2003-01-01",
    "GBP/JPY": "2003-01-01",
}

# Macro regimes for Task 3.3
REGIMES = {
    "pre_gfc": (2000, 2007),
    "gfc_euro_crisis": (2008, 2012),
    "post_crisis_qe": (2013, 2019),
    "covid_stimulus": (2020, 2021),
    "inflation_hikes": (2022, 2025),
}

# Survivor cells
SURVIVORS = [
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "carry_momentum"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "carry_momentum"},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "carry_momentum"},
    {"asset": "EUR/USD", "timeframe": "daily", "strategy": "mean_reversion"},
    {"asset": "USD/JPY", "timeframe": "daily", "strategy": "mean_reversion"},
    {"asset": "GBP/USD", "timeframe": "4h", "strategy": "vol_regime_switch"},
    {"asset": "AUD/USD", "timeframe": "weekly", "strategy": "vol_regime_switch"},
    {"asset": "EUR/JPY", "timeframe": "weekly", "strategy": "vol_regime_switch"},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "momentum"},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "carry_momentum"},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "vol_regime_switch"},
]


def download_extended_data(asset, timeframe):
    """Download extended historical data via yfinance."""
    ticker = YFINANCE_TICKERS.get(asset)
    if not ticker:
        return None

    start = EXTENDED_START.get(asset, "2003-01-01")

    # yfinance interval mapping
    if timeframe == "weekly":
        interval = "1wk"
    elif timeframe == "daily":
        interval = "1d"
    elif timeframe == "4h":
        # yfinance limits intraday to 730 days for 1h, 60 days for shorter
        # For 4h, download 1h and resample
        interval = "1h"
    else:
        interval = "1d"

    print(f"    Downloading {asset} ({ticker}) {timeframe} from {start}...", end=" ", flush=True)

    try:
        if timeframe == "4h":
            # 1h data limited to ~730 days from yfinance
            df = yf.download(ticker, start="2022-01-01", end="2025-12-31",
                             interval="1h", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                df = df.resample("4h").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"
                }).dropna()
        else:
            df = yf.download(ticker, start=start, end="2025-12-31",
                             interval=interval, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if len(df) == 0:
            print("EMPTY")
            return None

        # Data quality checks
        # Check for gaps > 1 week for weekly, > 3 days for daily
        if timeframe in ("weekly", "daily"):
            diffs = pd.Series(df.index).diff()
            max_gap = diffs.max()
            if timeframe == "weekly":
                gap_threshold = pd.Timedelta(days=14)
            else:
                gap_threshold = pd.Timedelta(days=7)

            n_gaps = (diffs > gap_threshold).sum()
            if n_gaps > 0:
                print(f"({n_gaps} gaps > {gap_threshold.days}d)", end=" ")

        # Check for price errors (rolling z-score > 10)
        close = df["Close"].values.astype(float)
        ret = np.diff(np.log(close + 1e-12))
        if len(ret) > 20:
            roll_std = pd.Series(ret).rolling(20).std()
            roll_mean = pd.Series(ret).rolling(20).mean()
            z_scores = (pd.Series(ret) - roll_mean) / (roll_std + 1e-12)
            n_outliers = (np.abs(z_scores) > 10).sum()
            if n_outliers > 0:
                print(f"({n_outliers} outliers |z|>10)", end=" ")

        print(f"{len(df)} bars, {df.index.min().date()} to {df.index.max().date()}")
        return df

    except Exception as e:
        print(f"FAILED: {e}")
        return None


def evaluate_cell_extended(asset, timeframe, strategy, df, seed=42):
    """Evaluate one cell with extended data. Returns comprehensive metrics."""
    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    # Oracle at noise=0
    sig_0 = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    pos_0 = get_strategy_positions(strategy, log_ret, sig_0, close)

    # Net returns
    gross = pos_0[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, pos_0[:-1], asset, abs_ret[:-1])
    oracle_sharpe = annualized_sharpe(net, ppy)

    # B&H
    bh_sharpe = annualized_sharpe(log_ret[1:], ppy)
    edge_sharpe = oracle_sharpe - bh_sharpe

    # Rolling window evaluation
    rolling = rolling_window_evaluation(net, ppy)

    # Noise budget recomputation
    noise_budget = 0.0
    for noise in NOISE_GRID:
        sig_n = generate_oracle_signal(log_ret, noise_sigma=noise, horizon=1, seed=seed)
        pos_n = get_strategy_positions(strategy, log_ret, sig_n, close)
        gross_n = pos_n[:-1] * log_ret[1:]
        net_n = apply_cost_to_returns(gross_n, pos_n[:-1], asset, abs_ret[:-1])
        sr_n = annualized_sharpe(net_n, ppy)
        if sr_n >= 0.3:
            noise_budget = noise
        else:
            break

    # Drawdown
    equity = np.cumsum(net)
    equity_curve = np.exp(equity)
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / (peak + 1e-12)
    max_dd = np.max(dd)

    return {
        "oracle_sharpe": round(oracle_sharpe, 4),
        "bh_sharpe": round(bh_sharpe, 4),
        "edge_sharpe": round(edge_sharpe, 4),
        "noise_budget": noise_budget,
        "regime_robustness": rolling["regime_robustness"],
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "max_drawdown": round(max_dd, 4),
        "n_bars": len(log_ret),
        "n_windows": len(rolling["windows"]),
    }


def regime_analysis(asset, timeframe, strategy, df, seed=42):
    """Task 3.3: Evaluate per macro regime."""
    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    # Full oracle
    sig_0 = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    pos_0 = get_strategy_positions(strategy, log_ret, sig_0, close)
    gross = pos_0[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, pos_0[:-1], asset, abs_ret[:-1])

    # Need DatetimeIndex for regime slicing
    dates = df.index
    regime_results = {}

    for regime_name, (y_start, y_end) in REGIMES.items():
        mask = (dates[1:].year >= y_start) & (dates[1:].year <= y_end)
        regime_net = net[mask]
        regime_bh = log_ret[1:][mask]

        if len(regime_net) < 10:
            regime_results[regime_name] = {"n_bars": 0, "note": "insufficient data"}
            continue

        sr = annualized_sharpe(regime_net, ppy)
        bh_sr = annualized_sharpe(regime_bh, ppy)

        regime_results[regime_name] = {
            "n_bars": int(np.sum(mask)),
            "oracle_sharpe": round(sr, 4),
            "bh_sharpe": round(bh_sr, 4),
            "edge_sharpe": round(sr - bh_sr, 4),
            "years": f"{y_start}-{y_end}",
        }

    # Count regimes with positive edge
    n_positive_edge = sum(
        1 for r in regime_results.values()
        if isinstance(r.get("edge_sharpe"), (int, float)) and r["edge_sharpe"] > 0
    )

    return {
        "regimes": regime_results,
        "n_regimes_positive_edge": n_positive_edge,
        "n_regimes_total": len([r for r in regime_results.values() if r.get("n_bars", 0) > 0]),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Assets to process (default: all survivors)")
    parser.add_argument("--output", default=None,
                        help="Output file path")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3.5 — TASK 3: HISTORICAL DATA EXTENSION")
    print("=" * 70)

    # Filter survivors by requested assets
    cells = SURVIVORS
    if args.assets:
        cells = [c for c in cells if c["asset"] in args.assets]

    # Determine unique (asset, timeframe) pairs
    unique_pairs = {}
    for c in cells:
        key = (c["asset"], c["timeframe"])
        if key not in unique_pairs:
            unique_pairs[key] = []
        unique_pairs[key].append(c["strategy"])

    # Task 3.1: Download extended data
    print("\n--- Task 3.1: Download Extended Historical Data ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    data_cache = {}

    for (asset, timeframe), strategies in unique_pairs.items():
        df = download_extended_data(asset, timeframe)
        if df is not None:
            # Save to disk
            safe_name = f"{asset.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(os.path.join(DATA_DIR, safe_name))
            data_cache[(asset, timeframe)] = df

    # Task 3.2: Recompute oracle sweep on extended data
    print("\n--- Task 3.2: Recompute Oracle Sweep on Extended Data ---")
    results = []

    for cell in cells:
        asset = cell["asset"]
        timeframe = cell["timeframe"]
        strategy = cell["strategy"]
        key = (asset, timeframe)

        print(f"\n  {asset} / {timeframe} / {strategy}...")

        if key not in data_cache:
            print(f"    ⚠ No data, skipping")
            results.append({"error": "no data", **cell})
            continue

        df = data_cache[key]
        metrics = evaluate_cell_extended(asset, timeframe, strategy, df)

        print(f"    Oracle SR: {metrics['oracle_sharpe']:+.3f}, B&H: {metrics['bh_sharpe']:+.3f}, Edge: {metrics['edge_sharpe']:+.3f}")
        print(f"    Noise budget: {metrics['noise_budget']:.2f}σ, Robustness: {metrics['regime_robustness']:.2f}")
        print(f"    Worst window: {metrics['worst_window_sharpe']:+.3f}, Max DD: {metrics['max_drawdown']:.1%}")

        # Task 3.3: Regime analysis
        regime = regime_analysis(asset, timeframe, strategy, df)
        print(f"    Regime analysis: positive edge in {regime['n_regimes_positive_edge']}/{regime['n_regimes_total']} regimes")
        for rname, rdata in regime["regimes"].items():
            if rdata.get("n_bars", 0) > 0:
                print(f"      {rname} ({rdata['years']}): edge={rdata['edge_sharpe']:+.3f}, n={rdata['n_bars']}")

        # Kill criteria
        killed = False
        kill_reasons = []

        # Edge Sharpe drop > 50% vs original (rough check - original was from 2018-2025 yfinance)
        # We'll compare in the integration step since we don't have original here

        if regime["n_regimes_positive_edge"] < 3:
            killed = True
            kill_reasons.append(f"positive edge in only {regime['n_regimes_positive_edge']} of {regime['n_regimes_total']} regimes (need ≥3)")

        if metrics["worst_window_sharpe"] < -0.5:
            killed = True
            kill_reasons.append(f"worst window Sharpe = {metrics['worst_window_sharpe']:.3f} (threshold: -0.5)")

        result = {
            **cell,
            "extended_metrics": metrics,
            "regime_analysis": regime,
            "killed": killed,
            "kill_reasons": kill_reasons,
        }
        results.append(result)

        if killed:
            print(f"    ✗ KILLED: {'; '.join(kill_reasons)}")
        else:
            print(f"    ✓ SURVIVES extended history audit")

    # Summary
    print("\n" + "=" * 70)
    print("TASK 3 SUMMARY")
    print("=" * 70)

    valid = [r for r in results if "error" not in r]
    survived = [r for r in valid if not r["killed"]]
    killed_cells = [r for r in valid if r["killed"]]

    print(f"\n  Total cells evaluated: {len(valid)}")
    print(f"  Survived: {len(survived)}")
    print(f"  Killed: {len(killed_cells)}")

    if survived:
        print(f"\n  Surviving cells:")
        for r in survived:
            m = r["extended_metrics"]
            reg = r["regime_analysis"]
            print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: "
                  f"edge={m['edge_sharpe']:+.3f}, budget={m['noise_budget']:.1f}σ, "
                  f"robustness={m['regime_robustness']:.2f}, "
                  f"regimes={reg['n_regimes_positive_edge']}/{reg['n_regimes_total']}")

    if killed_cells:
        print(f"\n  Killed cells:")
        for r in killed_cells:
            print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: {'; '.join(r['kill_reasons'])}")

    # Save
    output_path = args.output or os.path.join(RESULTS_DIR, "extended_history_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_cells": len(results),
            "n_survived": len(survived),
            "n_killed": len(killed_cells),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
