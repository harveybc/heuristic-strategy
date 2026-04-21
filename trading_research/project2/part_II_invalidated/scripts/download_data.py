#!/usr/bin/env python3
"""
Data Download Script — Project 2 Part II Stage II-1.2.a

Downloads EUR/USD 1h bars 2005-01-01 to 2024-12-31.

Sources:
  - HistData bulk download (2005-2020): pre-downloaded CSV files
  - OANDA v20 API (2020-2024 top-up): requires OANDA_TOKEN env var

Data quality checks per F-5 §2.3:
  - Missing bar percentage
  - Duplicate detection
  - Price sanity (EUR/USD typically 0.8–1.6)
  - Weekend filter (remove Saturday/Sunday bars)
  - Timezone normalisation to UTC

Output: data/raw/eurusd_1h_2005_2024.csv

Usage:
  python download_data.py [--histdata_dir path/to/histdata/csvs]
                          [--oanda_token TOKEN]
                          [--output data/raw/eurusd_1h_2005_2024.csv]
                          [--start_year 2005] [--end_year 2024]
"""

import argparse
import os
import glob
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_BARS_PER_YEAR_1H = 6240  # ~260 trading days * 24h
PRICE_MIN = 0.80   # EUR/USD sanity lower bound
PRICE_MAX = 1.65   # EUR/USD sanity upper bound


# ---------------------------------------------------------------------------
# HistData loader
# ---------------------------------------------------------------------------

def load_histdata_csvs(histdata_dir: str) -> pd.DataFrame:
    """
    Load HistData bulk-downloaded CSV files.
    
    HistData format: either 6-column (DateTime;Open;High;Low;Close;Volume)
    or 5-column (DateTime,Open,High,Low,Close) depending on download format.
    """
    csv_files = sorted(glob.glob(os.path.join(histdata_dir, "*.csv")))
    if not csv_files:
        # Try zip extraction patterns
        csv_files = sorted(glob.glob(os.path.join(histdata_dir, "**/*.csv"), recursive=True))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in {histdata_dir}")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} HistData CSV files")
    
    frames = []
    for fpath in csv_files:
        try:
            # Try semicolon separator first (HistData ASCII format)
            df = pd.read_csv(fpath, sep=';', header=None)
            if df.shape[1] < 5:
                # Try comma separator
                df = pd.read_csv(fpath, sep=',', header=None)
            
            if df.shape[1] == 6:
                df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            elif df.shape[1] == 5:
                df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
                df['Volume'] = 0
            elif df.shape[1] == 7:
                df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Extra']
                df = df.drop(columns=['Extra'])
            else:
                print(f"  [SKIP] {os.path.basename(fpath)}: unexpected {df.shape[1]} columns")
                continue
            
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', dayfirst=False)
            frames.append(df)
            print(f"  Loaded {os.path.basename(fpath)}: {len(df)} bars")
        except Exception as e:
            print(f"  [WARN] Failed to load {os.path.basename(fpath)}: {e}")
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values('DateTime', inplace=True)
    combined.drop_duplicates(subset='DateTime', keep='first', inplace=True)
    return combined


# ---------------------------------------------------------------------------
# OANDA v20 API loader
# ---------------------------------------------------------------------------

def download_oanda(token: str, instrument: str = "EUR_USD",
                   granularity: str = "H1",
                   start_date: str = "2020-01-01",
                   end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Download candles from OANDA v20 REST API.
    Requires: pip install oandapyV20
    """
    try:
        import oandapyV20
        from oandapyV20.endpoints.instruments import InstrumentsCandles
    except ImportError:
        print("[ERROR] oandapyV20 not installed. Install with: pip install oandapyV20")
        print("        Alternatively, provide HistData CSVs covering 2020-2024.")
        return pd.DataFrame()

    client = oandapyV20.API(access_token=token, environment="practice")
    
    all_candles = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_start < end_dt:
        # OANDA limits to 5000 candles per request
        chunk_end = min(current_start + timedelta(days=200), end_dt)
        
        params = {
            "from": current_start.strftime("%Y-%m-%dT00:00:00Z"),
            "to": chunk_end.strftime("%Y-%m-%dT00:00:00Z"),
            "granularity": granularity,
            "price": "M",  # Mid prices
        }
        
        try:
            req = InstrumentsCandles(instrument=instrument, params=params)
            response = client.request(req)
            candles = response.get("candles", [])
            
            for c in candles:
                if c.get("complete", False):
                    mid = c["mid"]
                    all_candles.append({
                        "DateTime": pd.to_datetime(c["time"]),
                        "Open": float(mid["o"]),
                        "High": float(mid["h"]),
                        "Low": float(mid["l"]),
                        "Close": float(mid["c"]),
                        "Volume": int(c.get("volume", 0)),
                    })
            
            print(f"  OANDA {current_start.date()} → {chunk_end.date()}: {len(candles)} candles")
        except Exception as e:
            print(f"  [WARN] OANDA API error for {current_start.date()}: {e}")
        
        current_start = chunk_end
    
    if not all_candles:
        return pd.DataFrame()
    
    return pd.DataFrame(all_candles)


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback)
# ---------------------------------------------------------------------------

def generate_synthetic_eurusd(start_year: int = 2005, end_year: int = 2024) -> pd.DataFrame:
    """
    Generate synthetic EUR/USD 1h data for pipeline testing.
    NOT for research use — only for infrastructure validation.
    Uses geometric Brownian motion with EUR/USD-like parameters.
    """
    print("[INFO] Generating synthetic EUR/USD data for infrastructure testing")
    print("[WARNING] This is synthetic data — NOT suitable for research experiments")
    
    np.random.seed(42)
    
    # Generate business-hours only (24h weekdays)
    dates = pd.bdate_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='h')
    # Filter to weekdays only
    dates = dates[dates.weekday < 5]
    
    n = len(dates)
    
    # GBM parameters calibrated to EUR/USD
    mu = 0.0  # No drift (FX is ~zero drift long-term)
    sigma = 0.0005  # Hourly volatility (~8% annualised)
    S0 = 1.20  # Starting price
    
    # Generate returns
    dt = 1.0 / (252 * 24)
    returns = np.random.normal(mu * dt, sigma, n)
    log_prices = np.log(S0) + np.cumsum(returns)
    close = np.exp(log_prices)
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.0003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0003, n)))
    open_price = np.roll(close, 1)
    open_price[0] = S0
    volume = np.random.randint(100, 10000, n)
    
    df = pd.DataFrame({
        'DateTime': dates[:n],
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    return df


# ---------------------------------------------------------------------------
# Data quality checks per F-5 §2.3
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame) -> dict:
    """Run data quality checks and return report dict."""
    report = {}
    
    # 1. Total bars
    report['total_bars'] = len(df)
    
    # 2. Date range
    report['date_range'] = f"{df['DateTime'].min()} to {df['DateTime'].max()}"
    report['years_covered'] = (df['DateTime'].max() - df['DateTime'].min()).days / 365.25
    
    # 3. Duplicates
    dupes = df.duplicated(subset='DateTime').sum()
    report['duplicates'] = int(dupes)
    if dupes > 0:
        print(f"  [WARN] {dupes} duplicate timestamps found — removing")
        df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
    
    # 4. Missing bars estimate
    years = df['DateTime'].dt.year.unique()
    expected_total = len(years) * EXPECTED_BARS_PER_YEAR_1H
    actual = len(df)
    missing_pct = max(0, (expected_total - actual) / expected_total * 100)
    report['expected_bars'] = expected_total
    report['actual_bars'] = actual
    report['missing_bar_pct'] = round(missing_pct, 2)
    
    # 5. Price sanity
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            below = (df[col] < PRICE_MIN).sum()
            above = (df[col] > PRICE_MAX).sum()
            if below > 0 or above > 0:
                print(f"  [WARN] {col}: {below} below {PRICE_MIN}, {above} above {PRICE_MAX}")
            report[f'{col}_min'] = float(df[col].min())
            report[f'{col}_max'] = float(df[col].max())
    
    # 6. NaN check
    nan_count = df[price_cols].isna().sum().sum()
    report['nan_count'] = int(nan_count)
    if nan_count > 0:
        print(f"  [WARN] {nan_count} NaN values in price columns")
    
    # 7. Weekend bars
    if pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        weekend = df['DateTime'].dt.weekday >= 5
        report['weekend_bars'] = int(weekend.sum())
    
    # 8. High >= Low check
    if 'High' in df.columns and 'Low' in df.columns:
        invalid_hl = (df['High'] < df['Low']).sum()
        report['invalid_high_low'] = int(invalid_hl)
        if invalid_hl > 0:
            print(f"  [WARN] {invalid_hl} bars where High < Low")
    
    return report


def filter_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Saturday and Sunday bars."""
    if pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        mask = df['DateTime'].dt.weekday < 5
        removed = len(df) - mask.sum()
        if removed > 0:
            print(f"  Removed {removed} weekend bars")
        return df[mask].copy()
    return df


def normalise_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all timestamps are UTC."""
    if df['DateTime'].dt.tz is not None:
        df['DateTime'] = df['DateTime'].dt.tz_convert('UTC').dt.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download EUR/USD 1h data")
    parser.add_argument("--histdata_dir", default=None,
                        help="Directory containing HistData CSV files")
    parser.add_argument("--oanda_token", default=None,
                        help="OANDA v20 API token (or set OANDA_TOKEN env var)")
    parser.add_argument("--output", 
                        default="data/raw/eurusd_1h_2005_2024.csv",
                        help="Output CSV path")
    parser.add_argument("--start_year", type=int, default=2005)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for testing (not for research)")
    args = parser.parse_args()

    oanda_token = args.oanda_token or os.environ.get("OANDA_TOKEN")
    
    print(f"\n{'='*60}")
    print(f"EUR/USD 1H DATA DOWNLOAD")
    print(f"{'='*60}")
    print(f"Period: {args.start_year} - {args.end_year}")
    print(f"Output: {args.output}")
    
    frames = []
    
    # Source 1: HistData
    if args.histdata_dir:
        print(f"\n--- Loading HistData from {args.histdata_dir} ---")
        hd = load_histdata_csvs(args.histdata_dir)
        if len(hd) > 0:
            frames.append(hd)
            print(f"HistData: {len(hd)} bars loaded")
    
    # Source 2: OANDA API
    if oanda_token:
        print(f"\n--- Downloading from OANDA API ---")
        # Only download years not covered by HistData
        oanda_start = "2020-01-01"
        if frames:
            last_date = frames[0]['DateTime'].max()
            if last_date.year >= 2020:
                oanda_start = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d")
        
        oa = download_oanda(oanda_token, start_date=oanda_start,
                            end_date=f"{args.end_year}-12-31")
        if len(oa) > 0:
            frames.append(oa)
            print(f"OANDA: {len(oa)} bars loaded")
    
    # Source 3: Synthetic (fallback for testing)
    if not frames or args.synthetic:
        if not frames:
            print("\n[INFO] No data sources available. Generating synthetic data for testing.")
        df_synthetic = generate_synthetic_eurusd(args.start_year, args.end_year)
        frames.append(df_synthetic)
    
    # Combine
    df = pd.concat(frames, ignore_index=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.sort_values('DateTime', inplace=True)
    df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
    
    # Filter by requested year range
    df = df[(df['DateTime'].dt.year >= args.start_year) & 
            (df['DateTime'].dt.year <= args.end_year)]
    
    # Timezone normalisation
    df = normalise_timezone(df)
    
    # Weekend filter
    df = filter_weekends(df)
    
    # Quality checks
    print(f"\n--- Data Quality Checks ---")
    report = run_quality_checks(df)
    for key, val in report.items():
        print(f"  {key}: {val}")
    
    # Reset index and save
    df = df.reset_index(drop=True)
    
    # Ensure column order
    cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved {len(df)} bars to {args.output}")
    
    # Save quality report
    report_path = args.output.replace('.csv', '_quality_report.json')
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Quality report saved to {report_path}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
