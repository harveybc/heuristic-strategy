#!/usr/bin/env python3
"""
Phase 3.5 — Task 4: Fix COT Data Download & Integrate into Feature Store

Uses CFTC Socrata API (modern endpoint) for recent COT data
and direct historical ZIPs for older data.

CFTC Contract Codes:
  EUR/USD → 099741, USD/JPY → 097741, GBP/USD → 096742,
  AUD/USD → 232741, XAU/USD → 088691, XAG/USD → 084691, CL → 067651
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import urllib.request
import urllib.error
import io
import zipfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FEATURE_STORE_DIR = os.path.join(os.path.dirname(__file__), "feature_store")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# CFTC contract code -> our asset name mapping
COT_CONTRACTS = {
    "099741": "EUR/USD",   # Euro FX
    "097741": "USD/JPY",   # Japanese Yen
    "096742": "GBP/USD",   # British Pound
    "232741": "AUD/USD",   # Australian Dollar
    "088691": "XAU/USD",   # Gold
    "084691": "XAG/USD",   # Silver
    "067651": "CL",        # Crude Oil WTI
}

# Socrata resource IDs
SOCRATA_RESOURCES = {
    "legacy_combined": "6dca-aqww",
    "disaggregated": "72hh-3qpy",
    "tff": "gpe5-46if",
}


def download_socrata_cot(resource_id, contract_code, limit=50000):
    """Download COT data from CFTC Socrata API."""
    base_url = f"https://publicreporting.cftc.gov/resource/{resource_id}.json"
    params = f"?$where=cftc_contract_market_code='{contract_code}'&$limit={limit}&$order=report_date_as_yyyy_mm_dd"
    url = base_url + params

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) trading-research/1.0")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
        if data:
            df = pd.DataFrame(data)
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"    Socrata error for {contract_code}: {e}")
        return pd.DataFrame()


def download_cftc_zip(year):
    """Download historical COT data from CFTC ZIP files."""
    # Legacy combined futures format
    url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) trading-research/1.0")

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # Usually contains one CSV file
            names = zf.namelist()
            csv_name = [n for n in names if n.endswith('.txt') or n.endswith('.csv')][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)
            return df
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return pd.DataFrame()
        print(f"    ZIP download error for {year}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    ZIP download error for {year}: {e}")
        return pd.DataFrame()


def extract_cot_features(df, contract_code, asset_name):
    """Extract COT features from raw CFTC data."""
    # Different column naming between Socrata JSON and ZIP CSV
    # Try both conventions

    # Identify date column
    date_col = None
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if "report_date" in cl or "as_of_date" in cl:
            date_col = c
            break
    if date_col is None:
        # Try first column
        date_col = df.columns[0]

    # Identify contract code column
    code_col = None
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if "contract_market_code" in cl or "cftc_contract" in cl:
            code_col = c
            break

    if code_col is not None:
        mask = df[code_col].astype(str).str.strip() == str(contract_code)
        df = df[mask].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Parse dates
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    # Extract positions - try multiple column name conventions
    features = {}

    # Non-commercial (speculator) positions
    nc_long_cols = [c for c in df.columns if any(x in c.lower() for x in ["noncommercial_positions_long", "non_comm_positions_long", "noncomm_long"])]
    nc_short_cols = [c for c in df.columns if any(x in c.lower() for x in ["noncommercial_positions_short", "non_comm_positions_short", "noncomm_short"])]

    # Commercial (hedger) positions
    comm_long_cols = [c for c in df.columns if any(x in c.lower() for x in ["commercial_positions_long", "comm_positions_long"]) and "non" not in c.lower()]
    comm_short_cols = [c for c in df.columns if any(x in c.lower() for x in ["commercial_positions_short", "comm_positions_short"]) and "non" not in c.lower()]

    # Open interest
    oi_cols = [c for c in df.columns if "open_interest" in c.lower() and "change" not in c.lower()]

    def safe_numeric(series):
        return pd.to_numeric(series, errors="coerce")

    if nc_long_cols and nc_short_cols:
        nc_long = safe_numeric(df[nc_long_cols[0]])
        nc_short = safe_numeric(df[nc_short_cols[0]])
        features["net_noncommercial"] = nc_long - nc_short
    
    if comm_long_cols and comm_short_cols:
        comm_long = safe_numeric(df[comm_long_cols[0]])
        comm_short = safe_numeric(df[comm_short_cols[0]])
        features["net_commercial"] = comm_long - comm_short

    if oi_cols:
        features["open_interest"] = safe_numeric(df[oi_cols[0]])

    if not features:
        return pd.DataFrame()

    result = pd.DataFrame(features)

    # Derived features
    if "net_noncommercial" in result.columns:
        # 156-week (3-year) z-score
        roll = result["net_noncommercial"].rolling(156, min_periods=52)
        result["net_nc_zscore_3y"] = (
            (result["net_noncommercial"] - roll.mean()) / (roll.std() + 1e-12)
        )
        # Change
        result["net_nc_change"] = result["net_noncommercial"].diff()

    if "open_interest" in result.columns:
        result["oi_pct_change"] = result["open_interest"].pct_change()

    return result


def download_all_cot():
    """Download COT data for all relevant contracts."""
    print("  Downloading COT data from CFTC Socrata API...")

    all_cot = {}

    for contract_code, asset_name in COT_CONTRACTS.items():
        print(f"\n    {asset_name} (code: {contract_code})...")

        # Try Socrata first (modern, clean)
        df_socrata = download_socrata_cot(SOCRATA_RESOURCES["legacy_combined"], contract_code)

        if len(df_socrata) > 0:
            features = extract_cot_features(df_socrata, contract_code, asset_name)
            if len(features) > 0:
                print(f"      Socrata: {len(features)} weekly observations, {features.index.min().date()} to {features.index.max().date()}")
                all_cot[asset_name] = features
                continue

        # Fallback: try historical ZIPs
        print(f"      Socrata failed/empty, trying historical ZIPs...")
        frames = []
        for year in range(2006, 2026):
            df_zip = download_cftc_zip(year)
            if len(df_zip) > 0:
                features = extract_cot_features(df_zip, contract_code, asset_name)
                if len(features) > 0:
                    frames.append(features)
                    print(f"      {year}: {len(features)} obs", end="")
            time.sleep(0.5)  # Rate limit

        if frames:
            combined = pd.concat(frames)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            print(f"\n      ZIP total: {len(combined)} observations, {combined.index.min().date()} to {combined.index.max().date()}")
            all_cot[asset_name] = combined
        else:
            print(f"      ⚠ No data available for {asset_name}")

    return all_cot


def integrate_into_feature_store(all_cot):
    """Add COT features to existing daily feature CSVs."""
    print("\n  Integrating COT features into feature store...")
    os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

    for asset_name, cot_df in all_cot.items():
        safe_name = asset_name.replace("/", "_")
        csv_path = os.path.join(FEATURE_STORE_DIR, f"{safe_name}_daily.csv")

        if not os.path.exists(csv_path):
            print(f"    ⚠ No feature file for {asset_name}, skipping")
            continue

        # Load existing features
        daily_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # COT is weekly (released Friday, data as of Tuesday)
        # Lag by 3 business days to avoid look-ahead
        cot_lagged = cot_df.copy()
        cot_lagged.index = cot_lagged.index + pd.Timedelta(days=3)

        # Forward-fill weekly COT to daily
        for col in cot_df.columns:
            cot_col = f"cot_{col}"
            # Reindex to daily frequency with ffill
            cot_series = cot_lagged[col].reindex(daily_df.index, method="ffill")
            daily_df[cot_col] = cot_series

        # Save updated
        daily_df.to_csv(csv_path)
        n_cot_cols = len([c for c in daily_df.columns if c.startswith("cot_")])
        pct_filled = daily_df[[c for c in daily_df.columns if c.startswith("cot_")]].notna().mean().mean() * 100
        print(f"    {asset_name}: added {n_cot_cols} COT columns, {pct_filled:.0f}% coverage")

    # Save standalone COT file
    cot_output = os.path.join(FEATURE_STORE_DIR, "cot_data.json")
    cot_summary = {}
    for asset_name, cot_df in all_cot.items():
        cot_summary[asset_name] = {
            "n_observations": len(cot_df),
            "date_range": [str(cot_df.index.min().date()), str(cot_df.index.max().date())],
            "columns": list(cot_df.columns),
        }
    with open(cot_output, "w") as f:
        json.dump(cot_summary, f, indent=2)
    print(f"\n  COT summary saved to {cot_output}")


def main():
    print("=" * 70)
    print("PHASE 3.5 — TASK 4: FIX COT DATA DOWNLOAD")
    print("=" * 70)

    all_cot = download_all_cot()

    if not all_cot:
        print("\n  ⚠ No COT data downloaded. All methods failed.")
        print("  Consider manual download from CFTC website.")
        return

    print(f"\n  Successfully downloaded COT data for {len(all_cot)} assets:")
    for asset, df in all_cot.items():
        print(f"    {asset}: {len(df)} obs, columns: {list(df.columns)}")

    integrate_into_feature_store(all_cot)

    # Save results
    output_path = os.path.join(RESULTS_DIR, "cot_download_results.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "n_assets": len(all_cot),
        "assets": {
            asset: {
                "n_observations": len(df),
                "date_range": [str(df.index.min().date()), str(df.index.max().date())],
                "columns": list(df.columns),
            }
            for asset, df in all_cot.items()
        }
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
