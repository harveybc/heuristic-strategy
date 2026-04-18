"""
Transaction Cost Model — Phase 0.2

Single source of truth for all trading cost calculations.
Per-asset-class spread, commission, and volatility-dependent slippage.
Used by every backtest and oracle test in the framework.
"""
import numpy as np


# ─── Cost tables ────────────────────────────────────────────────────────────
# All spreads in basis points (round-trip)
# Commissions in native units as specified
# Slippage: base bps scaled by sqrt(vol_ratio) where vol_ratio = bar_vol / trailing_avg_vol

COST_TABLE = {
    # FX Majors
    "EUR/USD":  {"spread_bps": 1.5, "commission": 0.0, "slip_base_bps": 0.3, "slip_vol_scale": True, "category": "fx_major"},
    "USD/JPY":  {"spread_bps": 1.5, "commission": 0.0, "slip_base_bps": 0.3, "slip_vol_scale": True, "category": "fx_major"},
    "GBP/USD":  {"spread_bps": 1.5, "commission": 0.0, "slip_base_bps": 0.3, "slip_vol_scale": True, "category": "fx_major"},
    "AUD/USD":  {"spread_bps": 1.5, "commission": 0.0, "slip_base_bps": 0.3, "slip_vol_scale": True, "category": "fx_major"},
    # FX Cross / Emerging
    "AUD/JPY":  {"spread_bps": 4.0, "commission": 0.0, "slip_base_bps": 0.5, "slip_vol_scale": True, "category": "fx_cross"},
    "EUR/JPY":  {"spread_bps": 4.0, "commission": 0.0, "slip_base_bps": 0.5, "slip_vol_scale": True, "category": "fx_cross"},
    "GBP/JPY":  {"spread_bps": 4.0, "commission": 0.0, "slip_base_bps": 0.5, "slip_vol_scale": True, "category": "fx_cross"},
    "USD/MXN":  {"spread_bps": 4.0, "commission": 0.0, "slip_base_bps": 0.5, "slip_vol_scale": True, "category": "fx_emerging"},
    # Equity Index ETF
    "SPY":      {"spread_bps": 1.0, "commission_per_share": 0.005, "slip_base_bps": 0.5, "slip_vol_scale": False, "category": "equity_etf"},
    # Commodity Futures (tick-based converted to bps at runtime)
    "CL":       {"spread_bps": 3.0, "commission_round": 5.0, "slip_base_bps": 3.0, "slip_vol_scale": True, "category": "commodity_future"},
    "XAU/USD":  {"spread_bps": 3.0, "commission_round": 5.0, "slip_base_bps": 2.0, "slip_vol_scale": True, "category": "commodity"},
    "XAG/USD":  {"spread_bps": 5.0, "commission_round": 5.0, "slip_base_bps": 3.0, "slip_vol_scale": True, "category": "commodity"},
    # Crypto Major
    "BTC/USD":  {"spread_bps": 10.0, "commission_bps": 10.0, "slip_base_bps": 15.0, "slip_vol_scale": True, "category": "crypto_major"},
    "ETH/USD":  {"spread_bps": 10.0, "commission_bps": 10.0, "slip_base_bps": 15.0, "slip_vol_scale": True, "category": "crypto_major"},
    # Crypto Alt
    "SOL/USD":  {"spread_bps": 20.0, "commission_bps": 10.0, "slip_base_bps": 30.0, "slip_vol_scale": True, "category": "crypto_alt"},
    "BNB/USD":  {"spread_bps": 20.0, "commission_bps": 10.0, "slip_base_bps": 30.0, "slip_vol_scale": True, "category": "crypto_alt"},
}

# yfinance tickers for data download
YFINANCE_TICKERS = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X", "AUD/JPY": "AUDJPY=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "USD/MXN": "MXN=X",
    "SPY": "SPY", "CL": "CL=F",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F",
    "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD", "BNB/USD": "BNB-USD",
}


def get_cost_params(asset: str) -> dict:
    """Get cost parameters for an asset. Raises KeyError if unknown."""
    if asset not in COST_TABLE:
        raise KeyError(f"Unknown asset '{asset}'. Known: {list(COST_TABLE.keys())}")
    return COST_TABLE[asset].copy()


def total_cost_bps(asset: str, bar_vol: float = None, trailing_vol: float = None) -> float:
    """
    Compute total round-trip cost in basis points for one trade.

    Parameters
    ----------
    asset : str
        Asset name matching COST_TABLE keys.
    bar_vol : float, optional
        Current bar's absolute return (for vol-dependent slippage).
    trailing_vol : float, optional
        Trailing 30-bar average absolute return.

    Returns
    -------
    float
        Total cost in bps (spread + commission + slippage).
    """
    p = COST_TABLE[asset]
    cost = p["spread_bps"]

    # Commission component
    if "commission_bps" in p:
        cost += p["commission_bps"]
    # For per-share / per-round commissions, caller should convert to bps externally
    # Here we add a flat estimate if present
    if "commission_round" in p:
        cost += 1.0  # ~1 bps flat estimate for futures commission

    # Slippage
    slip = p["slip_base_bps"]
    if p.get("slip_vol_scale", False) and bar_vol is not None and trailing_vol is not None and trailing_vol > 0:
        vol_ratio = bar_vol / trailing_vol
        slip *= np.sqrt(max(vol_ratio, 0.1))  # floor at 0.1 to avoid near-zero
    cost += slip

    return cost


def apply_cost_to_returns(returns: np.ndarray, positions: np.ndarray, asset: str,
                          abs_returns: np.ndarray = None) -> np.ndarray:
    """
    Apply transaction costs to a return series based on position changes.

    Parameters
    ----------
    returns : array
        Gross strategy returns per bar.
    positions : array
        Position at each bar (+1, 0, -1).
    asset : str
        Asset name for cost lookup.
    abs_returns : array, optional
        Absolute returns for vol-dependent slippage. If None, uses |returns|.

    Returns
    -------
    array
        Net returns after costs.
    """
    if abs_returns is None:
        abs_returns = np.abs(returns)

    # Trailing 30-bar average vol
    trailing_vol = np.full_like(abs_returns, np.nan)
    for i in range(30, len(abs_returns)):
        trailing_vol[i] = np.mean(abs_returns[i-30:i])
    trailing_vol[:30] = np.nanmean(abs_returns[:60]) if len(abs_returns) >= 60 else np.nanmean(abs_returns)

    # Position changes = trade events
    pos_changes = np.diff(positions, prepend=0)
    trade_mask = pos_changes != 0

    net_returns = returns.copy()
    for i in range(len(returns)):
        if trade_mask[i]:
            cost = total_cost_bps(asset, abs_returns[i], trailing_vol[i]) / 10000.0
            net_returns[i] -= cost

    return net_returns


if __name__ == "__main__":
    print("Transaction Cost Model")
    print("=" * 70)
    print(f"{'Asset':<12} {'Category':<16} {'Spread':>8} {'Total(calm)':>12} {'Total(2x vol)':>14}")
    print("-" * 70)
    for asset in COST_TABLE:
        calm = total_cost_bps(asset, bar_vol=0.001, trailing_vol=0.001)
        stressed = total_cost_bps(asset, bar_vol=0.002, trailing_vol=0.001)
        cat = COST_TABLE[asset]["category"]
        spread = COST_TABLE[asset]["spread_bps"]
        print(f"{asset:<12} {cat:<16} {spread:>7.1f} {calm:>11.1f} {stressed:>13.1f}")
