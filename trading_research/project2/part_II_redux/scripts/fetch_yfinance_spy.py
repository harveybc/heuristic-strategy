#!/usr/bin/env python3
"""Fetch SPY daily data from yfinance.

Usage:
    python scripts/fetch_yfinance_spy.py --output data/raw/yfinance/spy_daily_1993_2025.csv
"""
import argparse
import os

import yfinance as yf
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Fetch SPY from yfinance')
    parser.add_argument('--output', default='data/raw/yfinance/spy_daily_1993_2025.csv')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Fetching SPY from yfinance")
    print(f"{'='*60}")
    
    spy = yf.download('SPY', start='1993-01-29', end='2025-12-31', interval='1d', progress=False)
    
    if len(spy) == 0:
        print("  ERROR: No SPY data fetched")
        return
    
    # Flatten multi-level columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    spy = spy[['Open', 'High', 'Low', 'Close', 'Volume']]
    spy.index.name = 'DateTime'
    spy = spy.dropna()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    spy.to_csv(args.output)
    
    print(f"  Bars: {len(spy)}")
    print(f"  Range: {spy.index.min()} to {spy.index.max()}")
    print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()
