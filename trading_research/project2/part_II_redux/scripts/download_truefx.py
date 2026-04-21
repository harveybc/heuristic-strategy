#!/usr/bin/env python3
"""Download TrueFX tick data and resample to 1h bars.

Usage:
    source .env && python scripts/download_truefx.py --asset EURUSD --output data/raw/truefx/eurusd_1h_2009_2025.csv
"""
import argparse
import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests


def download_truefx_month(session, username, password, asset, year, month):
    """Download a single month of TrueFX tick data."""
    # TrueFX bulk download URL pattern
    month_str = f"{month:02d}"
    
    # TrueFX uses format: EURUSD-2009-01.zip
    url = f"https://www.truefx.com/dev/data/{year}/{asset}-{year}-{month_str}.zip"
    
    # Alternative pattern with full month name
    month_names = {
        1: 'JANUARY', 2: 'FEBRUARY', 3: 'MARCH', 4: 'APRIL',
        5: 'MAY', 6: 'JUNE', 7: 'JULY', 8: 'AUGUST',
        9: 'SEPTEMBER', 10: 'OCTOBER', 11: 'NOVEMBER', 12: 'DECEMBER'
    }
    
    # Try the truefx.com/dev/data/ bulk endpoint
    urls_to_try = [
        f"https://www.truefx.com/dev/data/{year}/{month_names[month]}-{year}/{asset}-{year}-{month_str}.zip",
        f"https://www.truefx.com/dev/data/{asset}/{year}/{asset}-{year}-{month_str}.zip",
        f"http://www.truefx.com/dev/data/{year}/{month_names[month]}-{year}/{asset}-{year}-{month_str}.zip",
    ]
    
    for url in urls_to_try:
        try:
            resp = session.get(url, timeout=60, auth=(username, password))
            if resp.status_code == 200 and len(resp.content) > 100:
                return resp.content
        except requests.RequestException:
            continue
    
    return None


def parse_truefx_ticks(zip_content):
    """Parse TrueFX tick CSV from zip content.
    
    TrueFX format: asset,timestamp,bid,ask
    timestamp is milliseconds since epoch or YYYYMMDD HH:MM:SS.mmm
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                return None
            
            all_dfs = []
            for csv_name in csv_files:
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, 
                                     names=['asset', 'timestamp', 'bid', 'ask'])
                    all_dfs.append(df)
            
            if not all_dfs:
                return None
            
            df = pd.concat(all_dfs, ignore_index=True)
            
            # Parse timestamp
            sample = str(df['timestamp'].iloc[0])
            if sample.isdigit() and len(sample) > 10:
                # Milliseconds since epoch
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            
            # Mid price
            df['mid'] = (df['bid'] + df['ask']) / 2
            
            return df[['datetime', 'mid', 'bid', 'ask']]
    except Exception as e:
        print(f"    Parse error: {e}")
        return None


def ticks_to_hourly(ticks_df):
    """Convert tick data to 1h OHLCV bars using mid price."""
    ticks_df = ticks_df.set_index('datetime').sort_index()
    
    hourly = ticks_df['mid'].resample('1h').agg(
        Open='first', High='max', Low='min', Close='last'
    ).dropna()
    
    # Volume = tick count
    hourly['Volume'] = ticks_df['mid'].resample('1h').count()
    hourly = hourly[hourly['Volume'] > 0]
    
    return hourly


def main():
    parser = argparse.ArgumentParser(description='Download TrueFX data')
    parser.add_argument('--asset', default='EURUSD', help='Asset (EURUSD, USDJPY)')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--start_year', type=int, default=2009)
    parser.add_argument('--end_year', type=int, default=2025)
    args = parser.parse_args()
    
    username = os.environ.get('TRUEFX_USER')
    password = os.environ.get('TRUEFX_PASS')
    if not username or not password:
        print("ERROR: TRUEFX_USER and TRUEFX_PASS must be set in environment")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"TrueFX Download: {args.asset}")
    print(f"{'='*60}")
    
    session = requests.Session()
    all_hourly = []
    
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            print(f"  {args.asset} {year}-{month:02d}...", end=' ')
            
            zip_content = download_truefx_month(session, username, password, 
                                                 args.asset, year, month)
            if zip_content is None:
                print("SKIP (not available)")
                continue
            
            ticks = parse_truefx_ticks(zip_content)
            if ticks is None or len(ticks) == 0:
                print("SKIP (no ticks)")
                continue
            
            hourly = ticks_to_hourly(ticks)
            all_hourly.append(hourly)
            print(f"{len(ticks)} ticks → {len(hourly)} bars")
            
            time.sleep(0.5)  # Rate limit
    
    if not all_hourly:
        print("\n  WARNING: No TrueFX data downloaded. TrueFX may require web login first.")
        print("  Creating empty placeholder. Cross-validation will be skipped.")
        # Create empty CSV with header
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']).to_csv(
            args.output, index=False)
        return
    
    combined = pd.concat(all_hourly)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined.index.name = 'DateTime'
    
    print(f"\n  Total hourly bars: {len(combined)}")
    print(f"  Range: {combined.index.min()} to {combined.index.max()}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output)
    print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()
