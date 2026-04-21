#!/usr/bin/env python3
"""
Macro Data Download — Project 2 Part II Stage II-1.2.d

Downloads macro/calendar data for CI-3 analysis:
  - FRED: US 10Y yield, DXY, VIX, CPI, unemployment rate
  - CFTC: EUR net positioning (Commitment of Traders)

All series forward-filled to avoid look-ahead bias.

Output:
  data/raw/macro_fred_monthly.csv
  data/raw/cftc_eur_weekly.csv

Usage:
  python download_macro_data.py [--fred_api_key KEY]
                                [--output_dir data/raw/]
                                [--start_date 2005-01-01]
                                [--end_date 2024-12-31]
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FRED data download
# ---------------------------------------------------------------------------

FRED_SERIES = {
    'DGS10': 'US_10Y_Yield',       # US 10-Year Treasury Yield
    'DTWEXBGS': 'DXY_Broad',       # Trade-Weighted US Dollar Index (broad)
    'VIXCLS': 'VIX',               # CBOE Volatility Index
    'CPIAUCSL': 'CPI',             # Consumer Price Index
    'UNRATE': 'Unemployment',      # Unemployment Rate
    'DEXUSEU': 'EURUSD_FRED',      # EUR/USD from FRED (for cross-check)
    'FEDFUNDS': 'Fed_Funds_Rate',  # Federal Funds Rate
    'IR3TIB01EZM156N': 'EU_3M_Rate',  # Eurozone 3-month interbank rate
}


def download_fred_data(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download macro series from FRED API."""
    try:
        from fredapi import Fred
    except ImportError:
        print("[ERROR] fredapi not installed. Install with: pip install fredapi")
        print("        Generating synthetic macro data for testing.")
        return _generate_synthetic_macro(start_date, end_date)

    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, col_name in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            frames[col_name] = data
            print(f"  ✓ {col_name} ({series_id}): {len(data)} observations")
        except Exception as e:
            print(f"  ✗ {col_name} ({series_id}): {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = 'Date'

    # Forward-fill to avoid look-ahead bias
    df = df.ffill()

    # Compute derived features
    if 'Fed_Funds_Rate' in df.columns and 'EU_3M_Rate' in df.columns:
        df['US_EU_Rate_Diff'] = df['Fed_Funds_Rate'] - df['EU_3M_Rate']

    return df


def _generate_synthetic_macro(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate synthetic macro data for testing."""
    print("[INFO] Generating synthetic macro data for pipeline testing")
    dates = pd.bdate_range(start=start_date, end=end_date, freq='MS')  # Monthly
    np.random.seed(123)

    n = len(dates)
    df = pd.DataFrame({
        'US_10Y_Yield': np.cumsum(np.random.normal(0, 0.1, n)) + 3.0,
        'DXY_Broad': np.cumsum(np.random.normal(0, 0.5, n)) + 100.0,
        'VIX': np.abs(np.cumsum(np.random.normal(0, 1, n)) + 20),
        'CPI': np.cumsum(np.abs(np.random.normal(0.2, 0.1, n))) + 200,
        'Unemployment': np.clip(np.cumsum(np.random.normal(0, 0.2, n)) + 5, 3, 15),
        'Fed_Funds_Rate': np.clip(np.cumsum(np.random.normal(0, 0.1, n)) + 2, 0, 6),
        'EU_3M_Rate': np.clip(np.cumsum(np.random.normal(0, 0.08, n)) + 1, -0.5, 5),
    }, index=dates)

    df['US_EU_Rate_Diff'] = df['Fed_Funds_Rate'] - df['EU_3M_Rate']
    df.index.name = 'Date'
    return df


# ---------------------------------------------------------------------------
# CFTC Commitment of Traders
# ---------------------------------------------------------------------------

def download_cftc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download CFTC Commitment of Traders data for EUR.
    Uses cot_reports package or falls back to synthetic.
    """
    try:
        import cot_reports
        df = cot_reports.cot_year(year=2024, cot_report_type='traders_in_financial_futures')
        # Filter for EUR
        eur_mask = df['Market_and_Exchange_Names'].str.contains('EURO', case=False, na=False)
        eur_data = df[eur_mask].copy()
        if len(eur_data) > 0:
            print(f"  ✓ CFTC EUR data: {len(eur_data)} reports")
            return eur_data
    except ImportError:
        pass
    except Exception as e:
        print(f"  [WARN] CFTC download failed: {e}")

    # Fallback: synthetic
    print("[INFO] Generating synthetic CFTC data for pipeline testing")
    dates = pd.bdate_range(start=start_date, end=end_date, freq='W-TUE')
    np.random.seed(456)
    n = len(dates)

    df = pd.DataFrame({
        'EUR_Net_Long': np.cumsum(np.random.normal(0, 5000, n)).astype(int),
        'EUR_Long': (np.abs(np.cumsum(np.random.normal(0, 3000, n))) + 100000).astype(int),
        'EUR_Short': (np.abs(np.cumsum(np.random.normal(0, 3000, n))) + 80000).astype(int),
        'EUR_Net_Change': np.random.normal(0, 5000, n).astype(int),
    }, index=dates)

    df.index.name = 'Date'
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download macro/CFTC data")
    parser.add_argument("--fred_api_key", default=None,
                        help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--output_dir", default="data/raw/",
                        help="Output directory")
    parser.add_argument("--start_date", default="2005-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for testing")
    args = parser.parse_args()

    fred_key = args.fred_api_key or os.environ.get("FRED_API_KEY")

    print(f"\n{'='*60}")
    print(f"MACRO DATA DOWNLOAD")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # FRED data
    print(f"\n--- FRED Macro Data ---")
    if fred_key and not args.synthetic:
        macro_df = download_fred_data(fred_key, args.start_date, args.end_date)
    else:
        if not args.synthetic:
            print("[INFO] No FRED API key. Set FRED_API_KEY env var or pass --fred_api_key")
        macro_df = _generate_synthetic_macro(args.start_date, args.end_date)

    macro_path = os.path.join(args.output_dir, "macro_fred_monthly.csv")
    macro_df.to_csv(macro_path, index=True)
    print(f"✓ Saved {len(macro_df)} rows to {macro_path}")

    # CFTC data
    print(f"\n--- CFTC EUR Positioning ---")
    cftc_df = download_cftc_data(args.start_date, args.end_date)
    cftc_path = os.path.join(args.output_dir, "cftc_eur_weekly.csv")
    cftc_df.to_csv(cftc_path, index=True)
    print(f"✓ Saved {len(cftc_df)} rows to {cftc_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
