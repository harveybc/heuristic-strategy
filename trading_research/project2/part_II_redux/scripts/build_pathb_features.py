#!/usr/bin/env python3
"""
Build Path B feature matrix from raw OHLCV data.
Computes the 12 F-6 features + 6-bar forward log return target.
Output: CSV with features + target column, ready for rolling orchestrator Path B.
"""
import argparse
import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12 F-6 technical features from OHLCV data."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    n = len(df)

    result = pd.DataFrame(index=df.index)

    # 1. ADX (14-period)
    period = 14
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                        np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                         np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.insert(dm_plus, 0, 0)
    dm_minus = np.insert(dm_minus, 0, 0)

    atr_arr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    di_plus = 100 * pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values / np.maximum(atr_arr, 1e-10)
    di_minus = 100 * pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values / np.maximum(atr_arr, 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    result['adx'] = adx

    # 2. DI Spread
    result['di_spread'] = di_plus - di_minus

    # 3. ATR% (14-period ATR / close)
    result['atr_pct'] = atr_arr / np.maximum(close, 1e-10)

    # 4. ATR Ratio (short/long ATR)
    atr_short = pd.Series(tr).ewm(span=7, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=28, adjust=False).mean().values
    result['atr_ratio'] = atr_short / np.maximum(atr_long, 1e-10)

    # 5. BB Width % (20-period, 2 std)
    bb_period = 20
    sma20 = pd.Series(close).rolling(bb_period).mean().values
    std20 = pd.Series(close).rolling(bb_period).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    result['bb_width_pct'] = (bb_upper - bb_lower) / np.maximum(sma20, 1e-10)

    # 6. BB Position (where price sits in BB)
    result['bb_position'] = (close - bb_lower) / np.maximum(bb_upper - bb_lower, 1e-10)

    # 7. RSI (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    result['rsi'] = 100 - 100 / (1 + rs)

    # 8. ROC-12 (12-period rate of change)
    roc_period = 12
    roc = np.full(n, np.nan)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / np.maximum(close[:-roc_period], 1e-10)
    result['roc_12'] = roc

    # 9. Price vs EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    result['price_vs_ema50'] = (close - ema50) / np.maximum(ema50, 1e-10)

    # 10. EMA Alignment (EMA20 - EMA50) / EMA50
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    result['ema_alignment'] = (ema20 - ema50) / np.maximum(ema50, 1e-10)

    # 11. Stochastic K (14-period)
    stoch_period = 14
    stoch_k = np.full(n, np.nan)
    for i in range(stoch_period - 1, n):
        h = np.max(high[i - stoch_period + 1:i + 1])
        l = np.min(low[i - stoch_period + 1:i + 1])
        stoch_k[i] = 100 * (close[i] - l) / max(h - l, 1e-10)
    result['stoch_k'] = stoch_k

    # 12. MACD Histogram
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    result['macd_hist'] = macd_line - signal

    return result


def compute_target(close: np.ndarray, forward_bars: int = 6) -> np.ndarray:
    """Compute forward log return target."""
    target = np.full(len(close), np.nan)
    target[:-forward_bars] = np.log(close[forward_bars:] / np.maximum(close[:-forward_bars], 1e-10))
    return target


def main():
    parser = argparse.ArgumentParser(description='Build Path B feature matrix')
    parser.add_argument('--data', required=True, help='Raw OHLCV CSV')
    parser.add_argument('--output', required=True, help='Output features CSV')
    parser.add_argument('--forward_bars', type=int, default=6, help='Forward bars for target')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    date_col = None
    for c in ['DateTime', 'DATE_TIME', 'Date', 'date', 'datetime']:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    print(f"Loaded {len(df)} bars from {args.data}")

    # Compute features
    features = compute_features(df)
    print(f"Computed {len(features.columns)} features: {list(features.columns)}")

    # Compute target
    features['target'] = compute_target(df['Close'].values, args.forward_bars)

    # Drop NaN rows (warmup period)
    valid = features.dropna()
    print(f"Valid rows after NaN drop: {len(valid)} (dropped {len(features) - len(valid)})")

    # Save
    valid.to_csv(args.output)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
