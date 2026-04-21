#!/usr/bin/env python3
"""Consolidate all raw data into processed timeframes.

Usage:
    python scripts/consolidate_data.py
"""
import os
import sys

import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def load_hourly(filepath):
    """Load a 1h CSV with DateTime index."""
    df = pd.read_csv(filepath, parse_dates=['DateTime'], index_col='DateTime')
    df.columns = [c.capitalize() for c in df.columns]  # Normalize
    return df.sort_index()


def load_daily(filepath):
    """Load a daily CSV with DateTime index."""
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    df.columns = [c.capitalize() for c in df.columns]
    df.index.name = 'DateTime'
    return df.sort_index()


def resample_ohlcv(df, freq):
    """Resample OHLCV to a lower frequency."""
    result = df.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna(subset=['Open'])
    return result


def process_fx(asset_name, hourly_path):
    """Process FX asset: 1h → 4h, daily, weekly."""
    print(f"\n  Processing {asset_name} from {hourly_path}")
    
    if not os.path.exists(hourly_path):
        print(f"    SKIP: {hourly_path} not found")
        return {}
    
    df_1h = load_hourly(hourly_path)
    print(f"    1h bars: {len(df_1h)} ({df_1h.index.min()} to {df_1h.index.max()})")
    
    # Drop Saturday bars (artifacts — FX doesn't trade Saturday)
    n_sat = (df_1h.index.dayofweek == 5).sum()
    if n_sat > 0:
        df_1h = df_1h[df_1h.index.dayofweek != 5]
        print(f"    Dropped {n_sat} Saturday artifact bars")
    
    outputs = {}
    
    # Save 1h
    path_1h = os.path.join(PROC_DIR, f'{asset_name}_1h.csv')
    df_1h.to_csv(path_1h)
    outputs['1h'] = (path_1h, len(df_1h))
    
    # 4h
    df_4h = resample_ohlcv(df_1h, '4h')
    df_4h = df_4h[df_4h.index.dayofweek != 5]  # Drop Saturday artifacts
    path_4h = os.path.join(PROC_DIR, f'{asset_name}_4h.csv')
    df_4h.to_csv(path_4h)
    outputs['4h'] = (path_4h, len(df_4h))
    print(f"    4h bars: {len(df_4h)}")
    
    # Daily
    df_daily = resample_ohlcv(df_1h, '1D')
    df_daily = df_daily[df_daily.index.dayofweek != 5]  # Drop Saturday artifacts
    path_daily = os.path.join(PROC_DIR, f'{asset_name}_daily.csv')
    df_daily.to_csv(path_daily)
    outputs['daily'] = (path_daily, len(df_daily))
    print(f"    Daily bars: {len(df_daily)}")
    
    # Weekly (anchor to Friday close)
    df_weekly = resample_ohlcv(df_1h, 'W-FRI')
    path_weekly = os.path.join(PROC_DIR, f'{asset_name}_weekly.csv')
    df_weekly.to_csv(path_weekly)
    outputs['weekly'] = (path_weekly, len(df_weekly))
    print(f"    Weekly bars: {len(df_weekly)}")
    
    return outputs


def process_equity(asset_name, daily_path):
    """Process equity asset: daily → weekly."""
    print(f"\n  Processing {asset_name} from {daily_path}")
    
    if not os.path.exists(daily_path):
        print(f"    SKIP: {daily_path} not found")
        return {}
    
    df_daily = load_daily(daily_path)
    print(f"    Daily bars: {len(df_daily)} ({df_daily.index.min()} to {df_daily.index.max()})")
    
    outputs = {}
    
    # Save daily
    path_daily = os.path.join(PROC_DIR, f'{asset_name}_daily.csv')
    df_daily.to_csv(path_daily)
    outputs['daily'] = (path_daily, len(df_daily))
    
    # Weekly (anchor to Friday close)
    df_weekly = resample_ohlcv(df_daily, 'W-FRI')
    path_weekly = os.path.join(PROC_DIR, f'{asset_name}_weekly.csv')
    df_weekly.to_csv(path_weekly)
    outputs['weekly'] = (path_weekly, len(df_weekly))
    print(f"    Weekly bars: {len(df_weekly)}")
    
    return outputs


def process_crypto(asset_name, path_4h):
    """Process crypto asset: 4h base → daily, weekly."""
    print(f"\n  Processing {asset_name} from {path_4h}")
    
    if not os.path.exists(path_4h):
        print(f"    SKIP: {path_4h} not found")
        return {}
    
    df_4h = load_daily(path_4h)  # Same CSV format
    print(f"    4h bars: {len(df_4h)} ({df_4h.index.min()} to {df_4h.index.max()})")
    
    outputs = {}
    
    # Save 4h
    path_4h_out = os.path.join(PROC_DIR, f'{asset_name}_4h.csv')
    df_4h.to_csv(path_4h_out)
    outputs['4h'] = (path_4h_out, len(df_4h))
    
    # Daily
    df_daily = resample_ohlcv(df_4h, '1D')
    path_daily = os.path.join(PROC_DIR, f'{asset_name}_daily.csv')
    df_daily.to_csv(path_daily)
    outputs['daily'] = (path_daily, len(df_daily))
    print(f"    Daily bars: {len(df_daily)}")
    
    # Weekly (anchor to Friday close for consistency)
    df_weekly = resample_ohlcv(df_4h, 'W-FRI')
    path_weekly = os.path.join(PROC_DIR, f'{asset_name}_weekly.csv')
    df_weekly.to_csv(path_weekly)
    outputs['weekly'] = (path_weekly, len(df_weekly))
    print(f"    Weekly bars: {len(df_weekly)}")
    
    return outputs


def main():
    print(f"\n{'='*60}")
    print("Consolidating All Data Sources")
    print(f"{'='*60}")
    
    os.makedirs(PROC_DIR, exist_ok=True)
    
    inventory = {}
    
    # FX assets from HistData
    for asset in ['eurusd', 'usdjpy']:
        hourly_path = os.path.join(RAW_DIR, 'histdata', f'{asset}_1h_2005_2025.csv')
        outputs = process_fx(asset, hourly_path)
        inventory[asset] = outputs
    
    # SPY from yfinance
    spy_path = os.path.join(RAW_DIR, 'yfinance', 'spy_daily_1993_2025.csv')
    outputs = process_equity('spy', spy_path)
    inventory['spy'] = outputs
    
    # BTC from Binance
    btc_4h_path = os.path.join(RAW_DIR, 'binance', 'btcusd_4h_2017_2025.csv')
    outputs = process_crypto('btcusd', btc_4h_path)
    inventory['btcusd'] = outputs
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONSOLIDATION SUMMARY")
    print(f"{'='*60}")
    
    for asset, timeframes in inventory.items():
        print(f"\n  {asset.upper()}:")
        for tf, (path, count) in timeframes.items():
            print(f"    {tf:8s}: {count:>8,} bars  → {os.path.basename(path)}")
    
    # Save inventory
    import json
    inv_path = os.path.join(PROC_DIR, 'inventory.json')
    inv_data = {asset: {tf: {'path': p, 'bars': c} for tf, (p, c) in tfs.items()} 
                for asset, tfs in inventory.items()}
    with open(inv_path, 'w') as f:
        json.dump(inv_data, f, indent=2)
    print(f"\n  Inventory saved to: {inv_path}")


if __name__ == '__main__':
    main()
