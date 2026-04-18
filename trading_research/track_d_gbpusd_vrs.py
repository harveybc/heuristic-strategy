#!/usr/bin/env python3
"""
Phase 4 — Track D: GBP/USD 4h VRS Resolution

Acquire 4h data for GBP/USD (as far back as possible),
then apply the full Phase 3.5 audit:
  - Noise budget audit (from audit_noise_budget)
  - Extended history worst-window test
  - Edge over B&H

The vol_regime_switch strategy showed SR=+1.76 at 10σ noise on GBP/USD 4h,
but yfinance only provides ~730 days of intraday data.

This track tries to get longer 4h history and test it properly.
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
    periods_per_year_for_timeframe
)
from trading_research.audit_noise_budget import get_strategy_positions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXTENDED_DATA = os.path.join(os.path.dirname(__file__), "extended_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXTENDED_DATA, exist_ok=True)

ASSET = "GBP/USD"
STRATEGY = "vol_regime_switch"
TICKER = "GBPUSD=X"
PPY_4H = 409.5  # From evaluation_harness for 4h


def acquire_4h_data():
    """
    Try multiple approaches to get maximum GBP/USD 4h history:
    1. Resample 1h data from yfinance (max ~730 days)
    2. Try downloading from alternative sources
    3. Resample daily data to simulate 4h (for extended audit only)
    """
    import yfinance as yf

    results = {}

    # Method 1: yfinance 1h → resample to 4h (best quality, limited history)
    print("  Method 1: yfinance 1h resampled to 4h")
    try:
        df_1h = yf.download(TICKER, period="max", interval="1h",
                            progress=False, auto_adjust=True)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)

        if len(df_1h) > 0:
            # Resample to 4h OHLC
            df_4h = df_1h.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()

            csv_path = os.path.join(EXTENDED_DATA, "GBP_USD_4h_yfinance.csv")
            df_4h.to_csv(csv_path)
            results["yfinance_1h_resampled"] = {
                "bars": len(df_4h),
                "start": str(df_4h.index[0]),
                "end": str(df_4h.index[-1]),
                "days": (df_4h.index[-1] - df_4h.index[0]).days,
                "path": csv_path,
            }
            print(f"    Got {len(df_4h)} bars from {df_4h.index[0]} to {df_4h.index[-1]} "
                  f"({results['yfinance_1h_resampled']['days']} days)")
    except Exception as e:
        print(f"    Failed: {e}")
        results["yfinance_1h_resampled"] = {"error": str(e)}

    # Method 2: yfinance daily data resampled as proxy for extended audit
    print("  Method 2: yfinance daily data (extended history proxy)")
    try:
        df_daily = yf.download(TICKER, start="2003-01-01", end="2025-12-31",
                               interval="1d", progress=False, auto_adjust=True)
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)

        if len(df_daily) > 0:
            csv_path = os.path.join(EXTENDED_DATA, "GBP_USD_daily_extended.csv")
            df_daily.to_csv(csv_path)
            results["daily_extended"] = {
                "bars": len(df_daily),
                "start": str(df_daily.index[0]),
                "end": str(df_daily.index[-1]),
                "days": (df_daily.index[-1] - df_daily.index[0]).days,
                "path": csv_path,
            }
            print(f"    Got {len(df_daily)} daily bars from {df_daily.index[0]} to {df_daily.index[-1]}")
    except Exception as e:
        results["daily_extended"] = {"error": str(e)}

    return results


def evaluate_vrs_on_data(df, label, ppy, noise_sigma=0.0, seed=42):
    """
    Apply vol_regime_switch strategy on price data and evaluate.
    Returns metrics dict.
    """
    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    abs_ret = np.abs(log_ret)

    # Generate oracle signal
    oracle_signal = generate_oracle_signal(log_ret, noise_sigma=noise_sigma, horizon=1, seed=seed)

    # Get VRS positions
    positions = get_strategy_positions(STRATEGY, log_ret, oracle_signal, close)

    # Net returns
    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], ASSET, abs_ret[:-1])

    sr = annualized_sharpe(net, ppy)
    bh_sr = annualized_sharpe(log_ret[1:], ppy)
    rolling = rolling_window_evaluation(net, ppy)

    equity = np.cumsum(net)
    eq_curve = np.exp(equity)
    peak = np.maximum.accumulate(eq_curve)
    dd = (peak - eq_curve) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    n_trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    n_years = len(log_ret) / ppy
    trades_per_year = n_trades / n_years if n_years > 0 else 0

    return {
        "label": label,
        "noise_sigma": noise_sigma,
        "sharpe": round(sr, 4),
        "bh_sharpe": round(bh_sr, 4),
        "edge_sharpe": round(sr - bh_sr, 4),
        "regime_robustness": rolling["regime_robustness"],
        "worst_window_sharpe": rolling["worst_window_sharpe"],
        "max_drawdown": round(max_dd, 4),
        "trades_per_year": round(trades_per_year, 1),
        "n_bars": len(log_ret),
        "rolling_windows": rolling.get("windows", []),
    }


def noise_sweep(df, label, ppy, noise_levels=None):
    """Run strategy across multiple noise levels."""
    if noise_levels is None:
        noise_levels = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0]

    results = []
    for ns in noise_levels:
        r = evaluate_vrs_on_data(df, label, ppy, noise_sigma=ns)
        results.append(r)
        print(f"    σ={ns:>4}: SR={r['sharpe']:+.3f}, Edge={r['edge_sharpe']:+.3f}, "
              f"Worst2Y={r['worst_window_sharpe']:+.3f}, DD={r['max_drawdown']:.1%}")
    return results


def regime_analysis(df, ppy, noise_sigma=0.0, seed=42):
    """Break down performance by macro regime periods."""
    REGIMES = {
        "pre_gfc": ("2003-01-01", "2007-06-30"),
        "gfc": ("2007-07-01", "2009-06-30"),
        "qe_era": ("2009-07-01", "2020-02-28"),
        "covid": ("2020-03-01", "2021-12-31"),
        "inflation": ("2022-01-01", "2025-12-31"),
    }

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    abs_ret = np.abs(log_ret)

    oracle = generate_oracle_signal(log_ret, noise_sigma=noise_sigma, horizon=1, seed=seed)
    positions = get_strategy_positions(STRATEGY, log_ret, oracle, close)

    gross = positions[:-1] * log_ret[1:]
    net = apply_cost_to_returns(gross, positions[:-1], ASSET, abs_ret[:-1])

    dates = df.index
    regime_results = {}

    for regime_name, (start, end) in REGIMES.items():
        mask = (dates[1:] >= start) & (dates[1:] <= end)
        if mask.sum() < 50:
            continue
        regime_net = net[mask[:len(net)]] if len(mask) >= len(net) else net[mask[:len(net)]]
        if len(regime_net) < 50:
            continue
        sr = annualized_sharpe(regime_net, ppy)
        regime_results[regime_name] = {
            "sharpe": round(sr, 4),
            "n_bars": int(mask.sum()),
        }
        print(f"      {regime_name}: SR={sr:+.3f} ({int(mask.sum())} bars)")

    return regime_results


def main():
    print("=" * 70)
    print("PHASE 4 — TRACK D: GBP/USD 4h VRS RESOLUTION")
    print("=" * 70)

    # Step 1: Acquire data
    print("\n--- Step 1: Acquire data ---")
    data_sources = acquire_4h_data()

    # Step 2: Evaluate VRS on yfinance 4h data (limited history)
    print("\n--- Step 2: VRS on yfinance 4h data ---")
    csv_4h = os.path.join(EXTENDED_DATA, "GBP_USD_4h_yfinance.csv")
    results_4h = None
    if os.path.exists(csv_4h):
        df_4h = pd.read_csv(csv_4h, index_col=0, parse_dates=True)
        if len(df_4h) > 100:
            print(f"  {len(df_4h)} bars of 4h data:")
            results_4h = noise_sweep(df_4h, "yfinance_4h", PPY_4H)
    else:
        print("  No 4h data available")

    # Step 3: VRS on daily extended data (as proxy)
    print("\n--- Step 3: VRS on daily data (extended history proxy) ---")
    csv_daily = os.path.join(EXTENDED_DATA, "GBP_USD_daily_extended.csv")
    results_daily = None
    if os.path.exists(csv_daily):
        df_daily = pd.read_csv(csv_daily, index_col=0, parse_dates=True)
        if len(df_daily) > 100:
            ppy_daily = 252
            print(f"  {len(df_daily)} bars of daily data:")
            results_daily = noise_sweep(df_daily, "daily_extended", ppy_daily)

            # Regime analysis on daily
            print("\n  Regime analysis (daily, σ=10):")
            regimes = regime_analysis(df_daily, ppy_daily, noise_sigma=10.0)
    else:
        print("  No daily data available")

    # Step 4: Also test at daily timeframe (not 4h) — since 4h is limited
    print("\n--- Step 4: Daily VRS vs other timeframes for GBP/USD ---")
    import yfinance as yf
    daily_summary = {}

    for tf, interval, ppy_tf in [("daily", "1d", 252), ("weekly", "1wk", 52)]:
        try:
            df = yf.download(TICKER, start="2003-01-01", end="2025-12-31",
                             interval=interval, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                print(f"\n  GBP/USD {tf} ({len(df)} bars):")
                noise_results = noise_sweep(df, f"gbpusd_{tf}", ppy_tf)
                daily_summary[tf] = noise_results
        except Exception as e:
            print(f"  Error for {tf}: {e}")

    # Decision
    print("\n" + "=" * 70)
    print("TRACK D — DECISION")
    print("=" * 70)

    # Check if VRS survives extended daily
    vrs_survives_daily = False
    vrs_4h_best = None

    if results_daily:
        sigma10 = [r for r in results_daily if r["noise_sigma"] == 10.0]
        if sigma10:
            s = sigma10[0]
            vrs_survives_daily = s["worst_window_sharpe"] > -0.5
            print(f"  Daily σ=10: SR={s['sharpe']:+.3f}, worst2Y={s['worst_window_sharpe']:+.3f} → "
                  f"{'SURVIVES' if vrs_survives_daily else 'KILLED'}")

    if results_4h:
        sigma10_4h = [r for r in results_4h if r["noise_sigma"] == 10.0]
        if sigma10_4h:
            vrs_4h_best = sigma10_4h[0]
            print(f"  4h σ=10: SR={vrs_4h_best['sharpe']:+.3f}, worst2Y={vrs_4h_best['worst_window_sharpe']:+.3f}")
            print(f"    NOTE: only {vrs_4h_best['n_bars']} bars (~{vrs_4h_best['n_bars']/(PPY_4H/252):.0f} days) "
                  f"— insufficient for worst-window analysis")

    # Resolution
    if vrs_survives_daily:
        resolution = "VRS_VIABLE_ON_DAILY"
        notes = "VRS survives worst-window on extended daily GBP/USD — consider daily deployment instead of 4h"
    elif vrs_4h_best and vrs_4h_best["sharpe"] > 0.3:
        resolution = "VRS_PROMISING_BUT_UNCONFIRMED"
        notes = (f"4h data too short ({vrs_4h_best['n_bars']} bars) for definitive test. "
                 f"4h Sharpe={vrs_4h_best['sharpe']:+.3f} but daily proxy fails. "
                 f"Recommend sourcing 4h data from OANDA or Dukascopy for proper evaluation.")
    else:
        resolution = "VRS_NOT_VIABLE"
        notes = "GBP/USD VRS fails on extended daily data and has insufficient 4h data"

    print(f"\n  RESOLUTION: {resolution}")
    print(f"  {notes}")

    # Save
    output = {
        "track": "D",
        "asset": ASSET,
        "strategy": STRATEGY,
        "resolution": resolution,
        "resolution_notes": notes,
        "data_sources": data_sources,
        "results_4h": results_4h,
        "results_daily": results_daily,
        "daily_summary": {k: v for k, v in daily_summary.items()} if daily_summary else None,
        "regimes": regimes if 'regimes' in dir() else None,
    }

    output_path = os.path.join(RESULTS_DIR, "phase4_track_d_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
