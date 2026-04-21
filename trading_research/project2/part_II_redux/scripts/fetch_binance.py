#!/usr/bin/env python3
"""Fetch BTC/USDT from Binance public API (no auth needed).

Usage:
    python scripts/fetch_binance.py --output_4h data/raw/binance/btcusd_4h_2017_2025.csv --output_daily data/raw/binance/btcusd_daily_2017_2025.csv
"""
import argparse
import os
import time

import pandas as pd
import requests


def fetch_binance_klines(symbol='BTCUSDT', interval='4h', start_date='2017-08-17', end_date='2025-12-31'):
    """Fetch klines from Binance REST API with pagination."""
    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    current_start = start_ms
    limit = 1000
    
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': limit,
        }
        
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        if not data:
            break
        
        all_data.extend(data)
        
        # Move to next batch
        current_start = data[-1][0] + 1
        
        if len(data) < limit:
            break
        
        time.sleep(0.2)  # Rate limit
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['DateTime'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df.set_index('DateTime')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[~df.index.duplicated(keep='first')]
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch Binance BTC/USDT')
    parser.add_argument('--output_4h', default='data/raw/binance/btcusd_4h_2017_2025.csv')
    parser.add_argument('--output_daily', default='data/raw/binance/btcusd_daily_2017_2025.csv')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Fetching Binance BTC/USDT")
    print(f"{'='*60}")
    
    # Fetch 4h
    print("  Fetching 4h bars...")
    df_4h = fetch_binance_klines(interval='4h')
    if len(df_4h) > 0:
        os.makedirs(os.path.dirname(args.output_4h), exist_ok=True)
        df_4h.to_csv(args.output_4h)
        print(f"  4h: {len(df_4h)} bars, {df_4h.index.min()} to {df_4h.index.max()}")
        print(f"  Saved to: {args.output_4h}")
    else:
        print("  WARNING: No 4h data fetched")
    
    # Fetch daily
    print("  Fetching daily bars...")
    df_daily = fetch_binance_klines(interval='1d')
    if len(df_daily) > 0:
        os.makedirs(os.path.dirname(args.output_daily), exist_ok=True)
        df_daily.to_csv(args.output_daily)
        print(f"  Daily: {len(df_daily)} bars, {df_daily.index.min()} to {df_daily.index.max()}")
        print(f"  Saved to: {args.output_daily}")
    else:
        print("  WARNING: No daily data fetched")


if __name__ == '__main__':
    main()
