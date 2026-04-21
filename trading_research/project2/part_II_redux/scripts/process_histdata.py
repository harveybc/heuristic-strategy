#!/usr/bin/env python3
"""Process HistData 1-minute ASCII zips into 1h OHLCV CSVs.

Usage:
    python scripts/process_histdata.py --zip_dir /path/to/zips --asset eurusd --output data/raw/histdata/eurusd_1h_2005_2025.csv
"""
import argparse
import glob
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd


def parse_histdata_csv(filepath: str) -> pd.DataFrame:
    """Parse a single HistData ASCII CSV file.
    
    HistData format variants:
      - YYYYMMDD HHMMSS;O;H;L;C;V  (semicolon-separated, datetime in one or two cols)
      - DateTime,Open,High,Low,Close,Volume  (comma-separated)
    """
    # Try semicolon first (most common HistData format)
    try:
        df = pd.read_csv(filepath, sep=';', header=None, 
                         names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'],
                         dtype={'datetime_str': str})
        if len(df.columns) == 6 and df['open'].dtype in [np.float64, np.int64]:
            # Parse datetime: "20050103 170000" format
            df['datetime'] = pd.to_datetime(df['datetime_str'].str.strip(), format='%Y%m%d %H%M%S')
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            return df
    except Exception:
        pass

    # Try comma-separated with header
    try:
        df = pd.read_csv(filepath, sep=',')
        cols_lower = [c.lower().strip() for c in df.columns]
        df.columns = cols_lower
        if 'datetime' in cols_lower or 'date' in cols_lower:
            dt_col = 'datetime' if 'datetime' in cols_lower else 'date'
            df['datetime'] = pd.to_datetime(df[dt_col])
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            return df
    except Exception:
        pass

    # Try semicolon with 7 columns (date and time separate)
    try:
        df = pd.read_csv(filepath, sep=';', header=None)
        if df.shape[1] == 7:
            df.columns = ['date_str', 'time_str', 'open', 'high', 'low', 'close', 'volume']
            df['datetime'] = pd.to_datetime(df['date_str'].str.strip() + ' ' + df['time_str'].str.strip(),
                                            format='%Y%m%d %H%M%S')
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            return df
    except Exception:
        pass

    raise ValueError(f"Could not parse HistData file: {filepath}")


def process_asset(zip_dir: str, asset: str, output_path: str):
    """Process all HistData zips for an asset into a single 1h CSV."""
    # Find zips
    patterns = [
        os.path.join(zip_dir, f'*{asset.upper()}*'),
        os.path.join(zip_dir, f'*{asset.lower()}*'),
        os.path.join(zip_dir, '*.zip'),
    ]
    
    zip_files = []
    for pattern in patterns:
        zip_files = sorted(glob.glob(pattern))
        zip_files = [f for f in zip_files if f.endswith('.zip')]
        if zip_files:
            break
    
    if not zip_files:
        print(f"  ERROR: No zip files found in {zip_dir}")
        sys.exit(1)
    
    print(f"  Found {len(zip_files)} zip files")
    
    all_dfs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for zf_path in sorted(zip_files):
            basename = os.path.basename(zf_path)
            print(f"    Processing {basename}...", end=' ')
            try:
                with zipfile.ZipFile(zf_path, 'r') as zf:
                    zf.extractall(tmpdir)
                    csv_files = [os.path.join(tmpdir, f) for f in zf.namelist() 
                                if f.lower().endswith('.csv')]
                    
                    for csv_path in csv_files:
                        df = parse_histdata_csv(csv_path)
                        all_dfs.append(df)
                        print(f"{len(df)} bars", end=' ')
                    
                    # Clean up extracted files
                    for f in zf.namelist():
                        fpath = os.path.join(tmpdir, f)
                        if os.path.exists(fpath):
                            os.remove(fpath)
                print()
            except Exception as e:
                print(f"ERROR: {e}")
                sys.exit(1)
    
    if not all_dfs:
        print("  ERROR: No data parsed from any zip file")
        sys.exit(1)
    
    # Concatenate and sort
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values('datetime', inplace=True)
    combined.drop_duplicates(subset='datetime', keep='first', inplace=True)
    combined.set_index('datetime', inplace=True)
    
    print(f"  Combined 1-min bars: {len(combined)}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    
    # Verify monotonic
    assert combined.index.is_monotonic_increasing, "Timestamps not monotonically increasing!"
    
    # Resample to 1h OHLCV
    hourly = combined.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['open'])
    
    print(f"  Hourly bars: {len(hourly)}")
    print(f"  Date range: {hourly.index.min()} to {hourly.index.max()}")
    
    # Validate: realistic bar count
    years = (hourly.index.max() - hourly.index.min()).days / 365.25
    bars_per_year = len(hourly) / years
    print(f"  Bars/year: {bars_per_year:.0f} (expect ~6,200 for FX)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hourly.index.name = 'DateTime'
    hourly.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    hourly.to_csv(output_path)
    print(f"  Saved to: {output_path}")
    
    return hourly


def main():
    parser = argparse.ArgumentParser(description='Process HistData zips to 1h OHLCV')
    parser.add_argument('--zip_dir', required=True, help='Directory containing HistData zip files')
    parser.add_argument('--asset', required=True, help='Asset name (e.g., eurusd, usdjpy)')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Processing HistData: {args.asset.upper()}")
    print(f"{'='*60}")
    print(f"  Zip directory: {args.zip_dir}")
    
    process_asset(args.zip_dir, args.asset, args.output)


if __name__ == '__main__':
    main()
