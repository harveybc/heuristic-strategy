"""
Oracle Sensitivity Test — Phase 0.1 (Rebuilt)

Noise is additive Gaussian on the oracle's predicted log-return, measured in
units of the target horizon's realized volatility (σ).

For each (asset, timeframe, strategy), computes the "noise budget": the noise
level (in σ units) at which net Sharpe drops below 0.3.

A large noise budget = forgiving market. A tiny noise budget = near-perfect
prediction required.

Kill criterion: if after this rework the test still saturates for all assets
at the same noise level, the oracle test is fundamentally broken.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
import os
import time
from typing import Dict, List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.transaction_cost_model import COST_TABLE, YFINANCE_TICKERS, total_cost_bps
from trading_research.evaluation_harness import (
    rolling_window_evaluation, annualized_sharpe, periods_per_year_24h,
    periods_per_year_for_timeframe
)


# ─── Noise grid (in units of realized σ) ────────────────────────────────────
NOISE_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
SHARPE_THRESHOLD = 0.3


# ─── Data download ──────────────────────────────────────────────────────────
def download_asset_data(asset: str, timeframe: str = "daily",
                        start: str = "2005-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and return as DataFrame with DatetimeIndex.
    """
    ticker = YFINANCE_TICKERS.get(asset)
    if ticker is None:
        raise ValueError(f"No yfinance ticker for '{asset}'")

    interval_map = {
        "15min": "15m",    # yfinance max 60 days for intraday
        "1h": "1h",        # yfinance max 730 days for 1h
        "4h": "1h",        # download 1h and resample
        "daily": "1d",
        "weekly": "1wk",
    }
    interval = interval_map.get(timeframe, "1d")

    # For intraday, yfinance limits history
    if timeframe in ("15min",):
        df = yf.download(ticker, period="60d", interval=interval, progress=False, auto_adjust=True)
    elif timeframe in ("1h", "4h"):
        df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {asset} ({ticker})")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Resample 1h -> 4h if needed
    if timeframe == "4h" and interval == "1h":
        df = df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()

    df = df.dropna(subset=["Close"])
    return df


# ─── Strategy implementations ───────────────────────────────────────────────
def strategy_momentum(log_returns: np.ndarray, oracle_signal: np.ndarray,
                      lookback: int = 20, threshold: float = 0.0) -> np.ndarray:
    """
    Directional momentum: enter in direction of N-bar cumulative return
    if magnitude > threshold. Oracle signal scales confidence.
    """
    positions = np.zeros(len(log_returns))
    cum_ret = np.zeros(len(log_returns))

    for i in range(lookback, len(log_returns)):
        cum_ret[i] = np.sum(log_returns[i-lookback:i])
        if abs(cum_ret[i]) > threshold:
            positions[i] = np.sign(cum_ret[i]) * np.sign(oracle_signal[i])
        else:
            positions[i] = 0

    return positions


def strategy_mean_reversion(log_returns: np.ndarray, oracle_signal: np.ndarray,
                            lookback: int = 20, z_entry: float = 1.5,
                            z_exit: float = 0.5) -> np.ndarray:
    """
    Mean reversion: enter counter to N-bar z-score when |z| > z_entry,
    exit when |z| < z_exit.
    """
    positions = np.zeros(len(log_returns))
    prices_cum = np.cumsum(log_returns)

    for i in range(lookback, len(log_returns)):
        window = prices_cum[i-lookback:i+1]
        if np.std(window) < 1e-12:
            positions[i] = 0
            continue
        z = (prices_cum[i] - np.mean(window)) / np.std(window)

        if abs(z) >= z_entry:
            positions[i] = -np.sign(z) * abs(oracle_signal[i])
        elif abs(z) < z_exit:
            positions[i] = 0
        else:
            positions[i] = positions[i-1] if i > 0 else 0

    return np.sign(positions)


def strategy_breakout(log_returns: np.ndarray, oracle_signal: np.ndarray,
                      prices_high: np.ndarray, prices_low: np.ndarray,
                      prices_close: np.ndarray,
                      lookback: int = 20) -> np.ndarray:
    """
    Breakout: enter on break of N-bar high/low with vol filter.
    """
    positions = np.zeros(len(log_returns))

    for i in range(lookback, len(log_returns)):
        hi = np.max(prices_high[i-lookback:i])
        lo = np.min(prices_low[i-lookback:i])
        mid = (hi + lo) / 2
        rng = hi - lo
        if rng < 1e-12:
            positions[i] = 0
            continue

        # Breakout above or below range
        if prices_close[i] > hi:
            positions[i] = 1 * np.sign(oracle_signal[i])
        elif prices_close[i] < lo:
            positions[i] = -1 * np.sign(oracle_signal[i])
        else:
            positions[i] = positions[i-1] if i > 0 else 0

    return np.sign(positions)


def strategy_vol_regime_switch(log_returns: np.ndarray, oracle_signal: np.ndarray,
                               lookback: int = 20, vol_lookback: int = 60,
                               vol_threshold_pctile: float = 50) -> np.ndarray:
    """
    Volatility regime switch: momentum in low-vol regime, MR in high-vol.
    """
    positions = np.zeros(len(log_returns))
    max_lb = max(lookback, vol_lookback)

    for i in range(max_lb, len(log_returns)):
        # Current vol vs historical
        current_vol = np.std(log_returns[i-lookback:i])
        hist_vol = np.std(log_returns[i-vol_lookback:i])

        if current_vol < hist_vol:  # Low vol -> momentum
            cum_ret = np.sum(log_returns[i-lookback:i])
            positions[i] = np.sign(cum_ret) * np.sign(oracle_signal[i])
        else:  # High vol -> mean reversion
            prices_cum = np.cumsum(log_returns[:i+1])
            window = prices_cum[i-lookback:i+1]
            if np.std(window) > 1e-12:
                z = (prices_cum[i] - np.mean(window)) / np.std(window)
                if abs(z) > 1.5:
                    positions[i] = -np.sign(z) * abs(oracle_signal[i])
                else:
                    positions[i] = positions[i-1] if i > 0 else 0
            else:
                positions[i] = 0

    return np.sign(positions)


def strategy_carry_momentum(log_returns: np.ndarray, oracle_signal: np.ndarray,
                            lookback: int = 20) -> np.ndarray:
    """
    Carry + momentum composite (simplified: uses momentum as proxy since
    we don't have interest rate differentials in this oracle test).
    """
    return strategy_momentum(log_returns, oracle_signal, lookback=lookback)


STRATEGY_FUNCTIONS = {
    "momentum": strategy_momentum,
    "mean_reversion": strategy_mean_reversion,
    "breakout": strategy_breakout,
    "vol_regime_switch": strategy_vol_regime_switch,
    "carry_momentum": strategy_carry_momentum,
}


# ─── Oracle with calibrated noise ───────────────────────────────────────────
def generate_oracle_signal(log_returns: np.ndarray, noise_sigma: float,
                           horizon: int = 1, seed: int = 42) -> np.ndarray:
    """
    Generate oracle signal = sign of future N-bar return + calibrated noise.

    The noise is additive Gaussian on the predicted log-return, scaled by
    the realized volatility of returns over the target horizon.

    Parameters
    ----------
    log_returns : array
        Log returns series.
    noise_sigma : float
        Noise level in units of realized return σ. 0 = perfect oracle.
    horizon : int
        Look-ahead horizon in bars.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    array
        Oracle signal (values in [-1, +1] range; sign = direction).
    """
    rng = np.random.default_rng(seed)
    n = len(log_returns)
    signal = np.zeros(n)

    # Future cumulative return for each bar
    for i in range(n - horizon):
        future_ret = np.sum(log_returns[i+1:i+1+horizon])
        signal[i] = future_ret

    # Compute realized vol of the horizon returns
    horizon_returns = np.array([
        np.sum(log_returns[i+1:i+1+horizon])
        for i in range(n - horizon)
    ])
    realized_vol = np.std(horizon_returns)
    if realized_vol < 1e-12:
        realized_vol = 1e-6

    # Add calibrated noise
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma * realized_vol, n)
        signal += noise

    return np.sign(signal)


# ─── Single cell evaluation ─────────────────────────────────────────────────
def evaluate_cell(asset: str, timeframe: str, strategy_name: str,
                  df: pd.DataFrame, noise_levels: List[float] = None,
                  horizon: int = 1, seed: int = 42) -> Dict:
    """
    Evaluate one (asset, timeframe, strategy) cell across all noise levels.

    Returns the noise budget (max noise where Sharpe >= 0.3) and full results.
    """
    if noise_levels is None:
        noise_levels = NOISE_GRID

    close = df["Close"].values.astype(float)
    log_returns = np.diff(np.log(close + 1e-12))
    log_returns = np.concatenate([[0], log_returns])

    high = df["High"].values.astype(float) if "High" in df.columns else close
    low = df["Low"].values.astype(float) if "Low" in df.columns else close
    abs_returns = np.abs(log_returns)

    # Determine annualization
    is_24h = COST_TABLE.get(asset, {}).get("category", "") in (
        "crypto_major", "crypto_alt", "fx_major", "fx_cross", "fx_emerging"
    )
    if is_24h:
        ppy = periods_per_year_24h(timeframe)
    else:
        ppy = periods_per_year_for_timeframe(timeframe)

    results = []
    noise_budget = 0.0

    for noise in noise_levels:
        oracle_signal = generate_oracle_signal(log_returns, noise, horizon, seed)

        # Run strategy
        if strategy_name == "breakout":
            positions = strategy_breakout(log_returns, oracle_signal, high, low, close)
        elif strategy_name in STRATEGY_FUNCTIONS:
            positions = STRATEGY_FUNCTIONS[strategy_name](log_returns, oracle_signal)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Gross returns from positions
        gross_returns = positions[:-1] * log_returns[1:]
        gross_returns = np.concatenate([[0], gross_returns])

        # Apply costs on trade entries
        pos_changes = np.diff(positions, prepend=0)
        trade_mask = pos_changes != 0

        trailing_vol = np.full_like(abs_returns, np.nan)
        for i in range(30, len(abs_returns)):
            trailing_vol[i] = np.mean(abs_returns[i-30:i])
        tv_fill = np.nanmean(abs_returns[:60]) if len(abs_returns) >= 60 else np.nanmean(abs_returns)
        trailing_vol[:30] = tv_fill
        trailing_vol = np.nan_to_num(trailing_vol, nan=tv_fill)

        net_returns = gross_returns.copy()
        for i in range(len(net_returns)):
            if trade_mask[i]:
                cost = total_cost_bps(asset, abs_returns[i], trailing_vol[i]) / 10000.0
                net_returns[i] -= cost

        # Compute metrics
        sharpe = annualized_sharpe(net_returns, ppy)
        equity = np.cumprod(1 + net_returns)
        total_ret = equity[-1] - 1.0
        n_trades = int(trade_mask.sum())
        trade_rets = net_returns[trade_mask]
        hit_rate = float(np.mean(trade_rets > 0)) if len(trade_rets) > 0 else 0.0
        avg_win = float(np.mean(trade_rets[trade_rets > 0]) * 10000) if np.any(trade_rets > 0) else 0.0
        avg_loss = float(np.mean(trade_rets[trade_rets < 0]) * 10000) if np.any(trade_rets < 0) else 0.0
        n_years = len(close) / ppy

        result = {
            "noise": noise,
            "sharpe": round(sharpe, 4),
            "total_return_pct": round(total_ret * 100, 2),
            "n_trades": n_trades,
            "trades_per_year": round(n_trades / n_years, 1) if n_years > 0 else 0,
            "hit_rate": round(hit_rate, 4),
            "avg_win_bps": round(avg_win, 1),
            "avg_loss_bps": round(avg_loss, 1),
        }

        # Rolling window evaluation only for noise=0 (perfect oracle)
        if noise == 0.0:
            rolling = rolling_window_evaluation(net_returns, ppy)
            result["regime_robustness"] = rolling["regime_robustness"]
            result["worst_window_sharpe"] = rolling["worst_window_sharpe"]
            result["is_interesting"] = rolling["is_interesting"]
            result["max_drawdown"] = rolling["full_period"]["max_drawdown"]

        results.append(result)

        if sharpe >= SHARPE_THRESHOLD:
            noise_budget = noise

    # Buy-and-hold benchmark
    bh_returns = log_returns.copy()
    bh_sharpe = annualized_sharpe(bh_returns, ppy)

    return {
        "asset": asset,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "n_bars": len(close),
        "noise_budget": noise_budget,
        "oracle_sharpe": results[0]["sharpe"] if results else 0,
        "bh_sharpe": round(bh_sharpe, 4),
        "noise_results": results,
    }


# ─── Batch evaluation (for distributed execution) ───────────────────────────
def evaluate_batch(cells: List[Tuple[str, str, str]], output_file: str,
                   data_cache: Dict = None):
    """
    Evaluate a batch of (asset, timeframe, strategy) cells.

    Parameters
    ----------
    cells : list of (asset, timeframe, strategy) tuples
    output_file : str
        Path to write JSON results.
    data_cache : dict, optional
        Pre-downloaded data keyed by (asset, timeframe).
    """
    if data_cache is None:
        data_cache = {}

    results = []
    failed = []

    for i, (asset, tf, strat) in enumerate(cells):
        key = (asset, tf)
        print(f"  [{i+1}/{len(cells)}] {asset} / {tf} / {strat}...", end=" ", flush=True)

        try:
            if key not in data_cache:
                data_cache[key] = download_asset_data(asset, tf)

            df = data_cache[key]
            if len(df) < 100:
                print(f"SKIP (only {len(df)} bars)")
                failed.append({"cell": [asset, tf, strat], "error": f"Only {len(df)} bars"})
                continue

            t0 = time.time()
            result = evaluate_cell(asset, tf, strat, df)
            elapsed = time.time() - t0
            result["elapsed_sec"] = round(elapsed, 1)
            results.append(result)

            nb = result["noise_budget"]
            os_sharpe = result["oracle_sharpe"]
            print(f"oracle SR={os_sharpe:+.3f}, budget={nb:.2f}σ ({elapsed:.1f}s)")

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append({"cell": [asset, tf, strat], "error": str(e)})

    # Save
    output = {
        "n_cells": len(cells),
        "n_completed": len(results),
        "n_failed": len(failed),
        "results": results,
        "failed": failed,
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved {len(results)} results to {output_file}")
    return output


# ─── Main: run all cells for this worker ────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oracle Sensitivity Test")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Assets to test (default: all)")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Timeframes to test (default: all)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to test (default: all)")
    parser.add_argument("--output", default="results/oracle_sweep.json",
                        help="Output file path")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker ID for distributed execution")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Total number of workers")
    args = parser.parse_args()

    # Full search grid
    ALL_ASSETS = [
        "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD",
        "AUD/JPY", "EUR/JPY", "GBP/JPY",
        "XAU/USD", "XAG/USD", "CL",
        "BTC/USD", "ETH/USD",
    ]
    ALL_TIMEFRAMES = ["15min", "1h", "4h", "daily", "weekly"]
    ALL_STRATEGIES = [
        "momentum", "mean_reversion", "breakout",
        "carry_momentum", "vol_regime_switch",
    ]
    # Note: "event_driven" requires macro surprise data (Phase 3), excluded from oracle sweep

    assets = args.assets or ALL_ASSETS
    timeframes = args.timeframes or ALL_TIMEFRAMES
    strategies = args.strategies or ALL_STRATEGIES

    # Build cell list
    all_cells = []
    for a in assets:
        for t in timeframes:
            for s in strategies:
                all_cells.append((a, t, s))

    # Distribute across workers
    worker_cells = [c for i, c in enumerate(all_cells) if i % args.n_workers == args.worker_id]

    print("=" * 80)
    print(f"ORACLE SENSITIVITY SWEEP — Worker {args.worker_id}/{args.n_workers}")
    print(f"=" * 80)
    print(f"Total cells: {len(all_cells)}, this worker: {len(worker_cells)}")
    print(f"Assets: {assets}")
    print(f"Timeframes: {timeframes}")
    print(f"Strategies: {strategies}")
    print(f"Noise grid: {NOISE_GRID}")
    print(f"Output: {args.output}")
    print("=" * 80)

    evaluate_batch(worker_cells, args.output)
