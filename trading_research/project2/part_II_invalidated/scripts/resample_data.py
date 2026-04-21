#!/usr/bin/env python3
"""
Resample Script — Project 2 Part II Stage II-1.2.b

Resamples 1h EUR/USD data to 4h (primary) and daily (for CI-2).
Proper OHLCV aggregation: O=first, H=max, L=min, C=last, V=sum.

Output:
  data/processed/eurusd_4h_2005_2024.csv
  data/processed/eurusd_daily_2005_2024.csv

Usage:
  python resample_data.py --input data/raw/eurusd_1h_2005_2024.csv
                          --output_dir data/processed/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str, label: str) -> pd.DataFrame:
    """
    Resample OHLCV data with proper aggregation.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        rule: Pandas resample rule ('4h', '1D', etc.)
        label: Human-readable label for logging
    
    Returns:
        Resampled DataFrame
    """
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    resampled = df.resample(rule).agg(agg_dict)
    
    # Drop rows where all OHLC are NaN (weekends/holidays)
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in resampled.columns]
    resampled = resampled.dropna(subset=price_cols, how='all')
    
    # Forward-fill any remaining NaN in Volume
    if 'Volume' in resampled.columns:
        resampled['Volume'] = resampled['Volume'].fillna(0).astype(int)
    
    print(f"  {label}: {len(df)} → {len(resampled)} bars")
    print(f"    Date range: {resampled.index.min()} to {resampled.index.max()}")
    print(f"    Close range: {resampled['Close'].min():.5f} to {resampled['Close'].max():.5f}")
    
    # Sanity check: High >= Low
    if 'High' in resampled.columns and 'Low' in resampled.columns:
        invalid = (resampled['High'] < resampled['Low']).sum()
        if invalid > 0:
            print(f"    [WARN] {invalid} bars where High < Low after resample")
    
    return resampled


def main():
    parser = argparse.ArgumentParser(description="Resample 1h OHLCV to 4h and daily")
    parser.add_argument("--input", required=True,
                        help="Input 1h CSV file")
    parser.add_argument("--output_dir", default="data/processed/",
                        help="Output directory")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"RESAMPLE EUR/USD DATA")
    print(f"{'='*60}")
    
    # Load
    df = pd.read_csv(args.input)
    
    # Auto-detect date column
    date_col = None
    for candidate in ['DateTime', 'DATE_TIME', 'Date', 'date', 'datetime', 'Datetime']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Input: {len(df)} bars, {df.index.min()} to {df.index.max()}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resample to 4h (primary experimental timeframe)
    print(f"\n--- Resampling to 4h ---")
    df_4h = resample_ohlcv(df, '4h', '1h → 4h')
    output_4h = os.path.join(args.output_dir, "eurusd_4h_2005_2024.csv")
    df_4h.to_csv(output_4h, index=True, index_label='DateTime')
    print(f"  ✓ Saved to {output_4h}")
    
    # Resample to daily (for CI-2 and secondary analysis)
    print(f"\n--- Resampling to daily ---")
    df_daily = resample_ohlcv(df, '1D', '1h → daily')
    output_daily = os.path.join(args.output_dir, "eurusd_daily_2005_2024.csv")
    df_daily.to_csv(output_daily, index=True, index_label='DateTime')
    print(f"  ✓ Saved to {output_daily}")
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"  1h bars:    {len(df)}")
    print(f"  4h bars:    {len(df_4h)}")
    print(f"  Daily bars: {len(df_daily)}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
