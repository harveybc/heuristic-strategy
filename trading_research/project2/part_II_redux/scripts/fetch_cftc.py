#!/usr/bin/env python3
"""Fetch CFTC Commitments of Traders data for EUR and JPY FX.

Uses CFTC bulk annual CSVs from cftc.gov.

Usage:
    python scripts/fetch_cftc.py --output_eur data/raw/cftc/eur_weekly_2000_2025.csv --output_jpy data/raw/cftc/jpy_weekly_2000_2025.csv
"""
import argparse
import io
import os
import zipfile

import pandas as pd
import requests


# CFTC futures-only report URLs
CFTC_URLS = {
    'current': 'https://www.cftc.gov/dea/newcot/deafut.txt',
    # Annual history zips
    'history': 'https://www.cftc.gov/files/dea/history/deacot{year}.zip',
    # Combined
    'fut_fin_combined': 'https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip',
}

# Contract market codes for FX
# EUR: 099741 (EURO FX - CHICAGO MERCANTILE EXCHANGE)
# JPY: 097741 (JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE)
ASSET_CODES = {
    'EUR': '099741',
    'JPY': '097741',
}


def fetch_cftc_year(year):
    """Fetch CFTC data for a single year."""
    url = f'https://www.cftc.gov/files/dea/history/deacot{year}.zip'
    
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            # Try alternate format
            url = f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip'
            resp = requests.get(url, timeout=60)
        
        if resp.status_code != 200:
            return None
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_files = [f for f in zf.namelist() if f.lower().endswith('.txt') or f.lower().endswith('.csv')]
            if not csv_files:
                return None
            
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f, low_memory=False)
                return df
    except Exception as e:
        print(f"    Year {year} error: {e}")
        return None


def extract_fx_positions(df, asset_code, asset_name):
    """Extract FX non-commercial net positioning from CFTC data."""
    # Filter by market name (most reliable across years)
    asset_keywords = {'EUR': 'EURO FX', 'JPY': 'JAPANESE YEN'}
    keyword = asset_keywords.get(asset_name, asset_name)
    
    # Find the market name column
    name_col = None
    for col in df.columns:
        if 'market' in col.lower() and 'name' in col.lower():
            name_col = col
            break
    
    if name_col is None:
        name_col = df.columns[0]  # First column is typically market name
    
    mask = df[name_col].astype(str).str.upper().str.contains(keyword, na=False)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        # Try CFTC code as fallback
        for col in df.columns:
            if 'cftc' in col.lower() and 'contract' in col.lower() and 'code' in col.lower():
                mask = df[col].astype(str).str.strip() == asset_code
                subset = df[mask].copy()
                break
    
    if len(subset) == 0:
        print(f"    No rows found for {asset_name} (keyword={keyword})")
        return pd.DataFrame()
    
    # Find date column — prefer YYYY-MM-DD format
    date_col = None
    for col in subset.columns:
        if 'yyyy-mm-dd' in col.lower():
            date_col = col
            break
    if date_col is None:
        for col in subset.columns:
            if 'date' in col.lower() and 'yymmdd' not in col.lower():
                date_col = col
                break
    if date_col is None:
        for col in subset.columns:
            if 'date' in col.lower():
                date_col = col
                break
    
    if date_col is None:
        print(f"    No date column found for {asset_name}")
        return pd.DataFrame()
    
    # Find position columns (non-commercial long/short)
    long_col = short_col = None
    for col in subset.columns:
        col_lower = col.lower()
        if 'noncommercial' in col_lower and 'long' in col_lower and 'spread' not in col_lower:
            if long_col is None:  # Take first match (All)
                long_col = col
        elif 'noncommercial' in col_lower and 'short' in col_lower:
            if short_col is None:
                short_col = col
    
    if long_col is None or short_col is None:
        print(f"    Could not find non-commercial position columns for {asset_name}")
        print(f"    Available columns: {[c for c in subset.columns if 'long' in c.lower() or 'short' in c.lower() or 'noncomm' in c.lower()]}")
        return pd.DataFrame()
    
    result = pd.DataFrame({
        'Date': pd.to_datetime(subset[date_col]),
        f'{asset_name}_Long': pd.to_numeric(subset[long_col], errors='coerce'),
        f'{asset_name}_Short': pd.to_numeric(subset[short_col], errors='coerce'),
    })
    
    result[f'{asset_name}_Net_Long'] = result[f'{asset_name}_Long'] - result[f'{asset_name}_Short']
    result[f'{asset_name}_Net_Change'] = result[f'{asset_name}_Net_Long'].diff()
    result = result.dropna(subset=[f'{asset_name}_Long'])
    result = result.sort_values('Date')
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Fetch CFTC COT data')
    parser.add_argument('--output_eur', default='data/raw/cftc/eur_weekly_2000_2025.csv')
    parser.add_argument('--output_jpy', default='data/raw/cftc/jpy_weekly_2000_2025.csv')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Fetching CFTC Commitments of Traders")
    print(f"{'='*60}")
    
    all_dfs = []
    for year in range(2000, 2026):
        print(f"  Year {year}...", end=' ')
        df = fetch_cftc_year(year)
        if df is not None:
            all_dfs.append(df)
            print(f"{len(df)} rows")
        else:
            print("SKIP")
    
    if not all_dfs:
        print("  ERROR: No CFTC data fetched")
        return
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total CFTC rows: {len(combined)}")
    
    for asset_name, asset_code in ASSET_CODES.items():
        output = args.output_eur if asset_name == 'EUR' else args.output_jpy
        print(f"\n  Extracting {asset_name} FX positions...")
        
        result = extract_fx_positions(combined, asset_code, asset_name)
        
        if len(result) == 0:
            print(f"    WARNING: No {asset_name} positions extracted")
            continue
        
        os.makedirs(os.path.dirname(output), exist_ok=True)
        result.to_csv(output, index=False)
        print(f"    {asset_name}: {len(result)} weekly observations")
        print(f"    Range: {result['Date'].min()} to {result['Date'].max()}")
        print(f"    Saved to: {output}")


if __name__ == '__main__':
    main()
