#!/usr/bin/env python3
"""Fetch FRED macro data using FRED API.

Usage:
    source .env && python scripts/fetch_fred_macro.py --output_daily data/raw/fred/macro_daily_1990_2025.csv --output_monthly data/raw/fred/macro_monthly_1990_2025.csv
"""
import argparse
import os
import sys

import pandas as pd
from fredapi import Fred


SERIES_MAP = {
    'DGS10': 'US_10Y_Yield',
    'DTWEXBGS': 'DXY_Broad',
    'VIXCLS': 'VIX',
    'CPIAUCSL': 'CPI',
    'UNRATE': 'Unemployment',
    'DFF': 'Fed_Funds_Rate',
    'IR3TIB01EZM156N': 'EU_3M_Rate',
    'IRLTLT01EZM156N': 'EU_Long_Term_Rate',
    'IRLTLT01JPM156N': 'JP_Long_Term_Rate',
}


def main():
    parser = argparse.ArgumentParser(description='Fetch FRED macro data')
    parser.add_argument('--output_daily', default='data/raw/fred/macro_daily_1990_2025.csv')
    parser.add_argument('--output_monthly', default='data/raw/fred/macro_monthly_1990_2025.csv')
    args = parser.parse_args()
    
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        print("ERROR: FRED_API_KEY must be set in environment")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Fetching FRED Macro Data")
    print(f"{'='*60}")
    
    fred = Fred(api_key=api_key)
    
    daily_series = {}
    monthly_series = {}
    
    for series_id, name in SERIES_MAP.items():
        print(f"  Fetching {series_id} ({name})...", end=' ')
        try:
            data = fred.get_series(series_id, observation_start='1990-01-01', observation_end='2025-12-31')
            print(f"{len(data)} observations")
            
            # Determine frequency
            if len(data) > 5000:  # Daily
                daily_series[name] = data
            else:  # Monthly
                monthly_series[name] = data
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Build daily DataFrame
    if daily_series:
        df_daily = pd.DataFrame(daily_series)
        df_daily.index.name = 'Date'
        df_daily = df_daily.ffill()  # Forward-fill gaps
        
        os.makedirs(os.path.dirname(args.output_daily), exist_ok=True)
        df_daily.to_csv(args.output_daily)
        print(f"\n  Daily macro: {len(df_daily)} rows, cols: {list(df_daily.columns)}")
        print(f"  Range: {df_daily.index.min()} to {df_daily.index.max()}")
        print(f"  Saved to: {args.output_daily}")
    
    # Build monthly DataFrame
    if monthly_series:
        df_monthly = pd.DataFrame(monthly_series)
        df_monthly.index.name = 'Date'
        
        os.makedirs(os.path.dirname(args.output_monthly), exist_ok=True)
        df_monthly.to_csv(args.output_monthly)
        print(f"\n  Monthly macro: {len(df_monthly)} rows, cols: {list(df_monthly.columns)}")
        print(f"  Range: {df_monthly.index.min()} to {df_monthly.index.max()}")
        print(f"  Saved to: {args.output_monthly}")
    
    # Also create combined daily (monthly forward-filled to daily)
    if daily_series and monthly_series:
        # Resample monthly to daily via forward-fill
        combined = df_daily.copy()
        for name, series in monthly_series.items():
            daily_resampled = series.resample('D').ffill()
            combined[name] = daily_resampled.reindex(combined.index, method='ffill')
        
        # Add derived features
        if 'Fed_Funds_Rate' in combined.columns and 'EU_3M_Rate' in combined.columns:
            combined['US_EU_Rate_Diff'] = combined['Fed_Funds_Rate'] - combined['EU_3M_Rate']
        if 'Fed_Funds_Rate' in combined.columns and 'JP_Long_Term_Rate' in combined.columns:
            combined['US_JP_Rate_Diff'] = combined['US_10Y_Yield'] - combined['JP_Long_Term_Rate']
        
        combined_path = args.output_daily.replace('macro_daily', 'macro_combined_daily')
        combined.to_csv(combined_path)
        print(f"\n  Combined daily macro: {len(combined)} rows, cols: {list(combined.columns)}")
        print(f"  Saved to: {combined_path}")


if __name__ == '__main__':
    main()
