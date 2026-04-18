"""
Rolling-Window Evaluation Harness — Phase 0.3

Every strategy evaluation reports standardized metrics with rolling windows.
A strategy is "interesting" if regime_robustness >= 0.5 AND full_period_sharpe >= 0.4.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def annualized_sharpe(returns: np.ndarray, periods_per_year: float = 252) -> float:
    """Annualized Sharpe ratio assuming zero risk-free rate."""
    if len(returns) < 2 or np.std(returns) < 1e-12:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (0.15 = 15% drawdown)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / (peak + 1e-12)
    return float(np.max(dd))


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Max drawdown from a return series, starting equity = 1."""
    equity = np.cumprod(1.0 + returns)
    return max_drawdown(equity)


def compute_strategy_metrics(returns: np.ndarray, periods_per_year: float = 252,
                             trade_mask: np.ndarray = None) -> Dict:
    """
    Compute full metric suite for a strategy.

    Parameters
    ----------
    returns : array
        Net return per bar (after costs).
    periods_per_year : float
        Annualization factor (252 for daily, 252*6.5 for hourly, etc.).
    trade_mask : array, optional
        Boolean mask where trades occurred. If None, inferred from nonzero returns.

    Returns
    -------
    dict
        Full metric dictionary.
    """
    if trade_mask is None:
        trade_mask = returns != 0

    n_bars = len(returns)
    years = n_bars / periods_per_year if periods_per_year > 0 else 1.0

    equity = np.cumprod(1.0 + returns)
    total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0

    # Sharpe
    sharpe = annualized_sharpe(returns, periods_per_year)

    # Drawdown
    dd = max_drawdown(equity)

    # Trade stats
    trade_returns = returns[trade_mask] if trade_mask.any() else np.array([])
    n_trades = int(np.sum(np.abs(np.diff(trade_mask.astype(int))) > 0) / 2) if len(trade_mask) > 1 else 0
    # Simpler: count nonzero return bars as proxy for trade count
    n_trade_bars = int(trade_mask.sum())
    trades_per_year = n_trade_bars / years if years > 0 else 0

    if len(trade_returns) > 0:
        winners = trade_returns[trade_returns > 0]
        losers = trade_returns[trade_returns < 0]
        hit_rate = len(winners) / len(trade_returns)
        avg_winner = float(np.mean(winners)) if len(winners) > 0 else 0.0
        avg_loser = float(np.mean(losers)) if len(losers) > 0 else 0.0
    else:
        hit_rate = 0.0
        avg_winner = 0.0
        avg_loser = 0.0

    return {
        "sharpe": round(sharpe, 4),
        "total_return": round(total_return, 6),
        "max_drawdown": round(dd, 6),
        "n_trade_bars": n_trade_bars,
        "trades_per_year": round(trades_per_year, 1),
        "hit_rate": round(hit_rate, 4),
        "avg_winner_bps": round(avg_winner * 10000, 2),
        "avg_loser_bps": round(avg_loser * 10000, 2),
        "n_bars": n_bars,
        "years": round(years, 2),
    }


def rolling_window_evaluation(returns: np.ndarray, periods_per_year: float = 252,
                              window_years: float = 2.0, step_months: int = 6,
                              sharpe_threshold: float = 0.3) -> Dict:
    """
    Evaluate strategy across rolling windows.

    Parameters
    ----------
    returns : array
        Net return series.
    periods_per_year : float
        Annualization factor.
    window_years : float
        Window size in years.
    step_months : int
        Step size in months.
    sharpe_threshold : float
        Threshold for "passing" a window.

    Returns
    -------
    dict with:
        - full_period: metrics for entire series
        - windows: list of per-window metrics
        - regime_robustness: fraction of windows with Sharpe > threshold
        - is_interesting: regime_robustness >= 0.5 AND full_period Sharpe >= 0.4
    """
    window_bars = int(window_years * periods_per_year)
    step_bars = int(step_months / 12.0 * periods_per_year)

    # Full period
    full_metrics = compute_strategy_metrics(returns, periods_per_year)

    # Rolling windows
    windows = []
    start = 0
    while start + window_bars <= len(returns):
        w_returns = returns[start:start + window_bars]
        w_metrics = compute_strategy_metrics(w_returns, periods_per_year)
        w_metrics["start_bar"] = start
        w_metrics["end_bar"] = start + window_bars
        windows.append(w_metrics)
        start += step_bars

    # Robustness
    if len(windows) > 0:
        passing = sum(1 for w in windows if w["sharpe"] >= sharpe_threshold)
        regime_robustness = passing / len(windows)
        sharpes = [w["sharpe"] for w in windows]
        worst_window_sharpe = min(sharpes)
        worst_window_dd = max(w["max_drawdown"] for w in windows)
    else:
        regime_robustness = 0.0
        worst_window_sharpe = 0.0
        worst_window_dd = 0.0

    is_interesting = (regime_robustness >= 0.5) and (full_metrics["sharpe"] >= 0.4)

    return {
        "full_period": full_metrics,
        "n_windows": len(windows),
        "regime_robustness": round(regime_robustness, 4),
        "worst_window_sharpe": round(worst_window_sharpe, 4),
        "worst_window_dd": round(worst_window_dd, 6),
        "is_interesting": is_interesting,
        "windows": windows,
    }


def periods_per_year_for_timeframe(timeframe: str) -> float:
    """Map timeframe string to annualization factor."""
    mapping = {
        "15min": 252 * 6.5 * 4,    # ~6552 for US equities; adjust for 24h markets
        "1h": 252 * 6.5,           # ~1638 for equities
        "4h": 252 * 6.5 / 4,      # ~409.5
        "daily": 252,
        "weekly": 52,
    }
    # For 24h markets (crypto, FX)
    mapping_24h = {
        "15min": 365 * 24 * 4,     # 35040
        "1h": 365 * 24,            # 8760
        "4h": 365 * 6,             # 2190
        "daily": 365,
        "weekly": 52,
    }
    return mapping.get(timeframe, 252)


def periods_per_year_24h(timeframe: str) -> float:
    """Annualization factor for 24h markets (crypto, FX)."""
    mapping = {
        "15min": 365.25 * 24 * 4,
        "1h": 365.25 * 24,
        "4h": 365.25 * 6,
        "daily": 365.25,
        "weekly": 52,
    }
    return mapping.get(timeframe, 252)


if __name__ == "__main__":
    # Quick self-test with random returns
    np.random.seed(42)
    fake_returns = np.random.normal(0.0002, 0.01, 2520)  # ~10 years daily
    result = rolling_window_evaluation(fake_returns, 252)
    print(f"Full Sharpe: {result['full_period']['sharpe']:.3f}")
    print(f"Robustness: {result['regime_robustness']:.1%}")
    print(f"Interesting: {result['is_interesting']}")
    print(f"Windows: {result['n_windows']}")
