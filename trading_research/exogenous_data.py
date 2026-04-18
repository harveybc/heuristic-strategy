#!/usr/bin/env python3
"""
Phase 3: Exogenous Data Enrichment

Downloads and integrates macro/external features:
  3.1 — Always-on macro features (FRED, VIX, DXY)
  3.2 — CFTC COT reports
  3.3 — Crypto-specific features (funding rates)

Output: unified feature store per (asset, timeframe) as parquet files.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "feature_store")


# ─── Task 3.1: Always-on macro features ─────────────────────────────────────

def download_fred_series(series_id: str, start: str = "2005-01-01",
                         end: str = "2025-12-31") -> pd.Series:
    """Download a FRED series. Uses pandas_datareader if available, else yfinance proxy."""
    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_id, "fred", start, end)
        return df.iloc[:, 0]
    except ImportError:
        pass

    # Fallback: try FRED CSV URL directly
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
        df = pd.read_csv(url)
        # FRED CSVs may use 'DATE' or 'date' or first column as date
        date_col = None
        for c in df.columns:
            if c.upper() == 'DATE':
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        s = df.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce")
        return s.dropna()
    except Exception as e:
        print(f"    ⚠ Failed to download {series_id}: {e}")
        return pd.Series(dtype=float)


def get_macro_features(start: str = "2005-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Download macro features from FRED and Yahoo Finance."""
    print("  Downloading macro features...")

    features = {}

    # FRED series
    fred_series = {
        "DGS10": "US 10Y yield",
        "DGS2": "US 2Y yield",
        "T10Y2Y": "10Y-2Y spread",
        "DFF": "Fed Funds Rate",
    }
    for sid, desc in fred_series.items():
        print(f"    {sid} ({desc})...", end=" ", flush=True)
        s = download_fred_series(sid, start, end)
        if len(s) > 0:
            features[sid] = s
            print(f"{len(s)} obs")
        else:
            print("FAILED")

    # VIX from Yahoo Finance
    print("    VIX...", end=" ", flush=True)
    try:
        vix = yf.download("^VIX", start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        features["VIX"] = vix["Close"]
        print(f"{len(vix)} obs")
    except Exception as e:
        print(f"FAILED: {e}")

    # DXY proxy (from DX-Y.NYB)
    print("    DXY...", end=" ", flush=True)
    try:
        dxy = yf.download("DX-Y.NYB", start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)
        features["DXY"] = dxy["Close"]
        print(f"{len(dxy)} obs")
    except Exception as e:
        print(f"FAILED: {e}")

    if not features:
        return pd.DataFrame()

    # Align to daily
    df = pd.DataFrame(features)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.ffill()  # Forward-fill daily macro data

    # Derived features
    if "DGS10" in df.columns and "DGS2" in df.columns:
        df["yield_spread"] = df["DGS10"] - df["DGS2"]
    if "VIX" in df.columns:
        df["VIX_pctile_60d"] = df["VIX"].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    if "DXY" in df.columns:
        df["DXY_ret_20d"] = np.log(df["DXY"]).diff(20)

    print(f"  ✓ Macro features: {list(df.columns)}, {len(df)} rows")
    return df


# ─── Task 3.2: CFTC COT data ────────────────────────────────────────────────

def get_cot_data() -> pd.DataFrame:
    """
    Download CFTC Commitment of Traders data.
    Uses Quandl/CFTC public data.
    """
    print("  Downloading COT data...")

    # COT data is available from CFTC.gov as compressed CSVs
    # For simplicity, try the Quandl endpoint first
    cot_tickers = {
        "EURUSD_COT": "099741",  # Euro FX
        "JPYUSD_COT": "097741",  # Japanese Yen
        "GBPUSD_COT": "096742",  # British Pound
        "AUDUSD_COT": "232741",  # Australian Dollar
        "GOLD_COT": "088691",    # Gold
        "CRUDE_COT": "067651",   # Crude Oil
    }

    try:
        # Try downloading from CFTC public data
        base_url = "https://www.cftc.gov/dea/newcot/deafut.txt"
        print(f"    Trying CFTC direct download...")
        df = pd.read_csv(base_url, low_memory=False)
        print(f"    ✓ Got {len(df)} rows from CFTC")

        # Extract relevant columns: noncommercial long, short, spreading
        # This is a simplified extraction — real implementation needs more work
        return df
    except Exception as e:
        print(f"    ⚠ COT download failed: {e}")
        print(f"    → COT data will be added manually later if needed")
        return pd.DataFrame()


# ─── Task 3.3: Crypto features ──────────────────────────────────────────────

def get_crypto_features(start: str = "2018-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Download crypto-specific features (funding rates, volumes)."""
    print("  Downloading crypto features...")

    features = {}

    # BTC and ETH volumes from Yahoo Finance as proxy
    for asset, ticker in [("BTC", "BTC-USD"), ("ETH", "ETH-USD")]:
        print(f"    {asset} volume data...", end=" ", flush=True)
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            features[f"{asset}_volume"] = df["Volume"]
            features[f"{asset}_vol_20d"] = df["Close"].pct_change().rolling(20).std()
            features[f"{asset}_ret_20d"] = np.log(df["Close"]).diff(20)
            print(f"{len(df)} obs")
        except Exception as e:
            print(f"FAILED: {e}")

    # BTC dominance proxy (BTC market cap / total crypto)
    # Not directly available from yfinance, skip for now
    print(f"    BTC dominance: not available from yfinance, skipping")

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill()

    print(f"  ✓ Crypto features: {list(df.columns)}, {len(df)} rows")
    return df


# ─── Feature store builder ──────────────────────────────────────────────────

def build_feature_store(assets: list = None, timeframes: list = None):
    """
    Build unified feature store for all (asset, timeframe) combinations.
    """
    if assets is None:
        assets = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD",
                  "AUD/JPY", "EUR/JPY", "GBP/JPY",
                  "XAU/USD", "CL", "BTC/USD", "ETH/USD"]
    if timeframes is None:
        timeframes = ["daily"]  # Start with daily, extend later

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download shared features
    macro_df = get_macro_features()
    crypto_df = get_crypto_features()
    cot_df = get_cot_data()

    results = {}

    for asset in assets:
        for tf in timeframes:
            key = f"{asset.replace('/', '_')}_{tf}"
            print(f"\n  Building: {asset} / {tf}...")

            try:
                from trading_research.oracle_sensitivity import download_asset_data
                price_df = download_asset_data(asset, tf)
            except Exception as e:
                print(f"    ⚠ Failed to download {asset}/{tf}: {e}")
                continue

            # Merge macro features (align to price index)
            if not macro_df.empty:
                # For intraday data, forward-fill daily macro to each bar
                macro_aligned = macro_df.reindex(price_df.index, method="ffill")
                for col in macro_aligned.columns:
                    price_df[f"macro_{col}"] = macro_aligned[col]

            # Merge crypto features if applicable
            if asset in ("BTC/USD", "ETH/USD") and not crypto_df.empty:
                crypto_aligned = crypto_df.reindex(price_df.index, method="ffill")
                for col in crypto_aligned.columns:
                    price_df[f"crypto_{col}"] = crypto_aligned[col]

            # Technical features (basic)
            close = price_df["Close"].values.astype(float)
            ret = np.diff(np.log(close + 1e-12))
            ret = np.concatenate([[0], ret])
            price_df["log_return"] = ret
            price_df["abs_return"] = np.abs(ret)
            price_df["vol_20"] = price_df["abs_return"].rolling(20).std()
            price_df["ret_20"] = price_df["log_return"].rolling(20).sum()
            price_df["z_score_20"] = (
                (price_df["Close"] - price_df["Close"].rolling(20).mean()) /
                (price_df["Close"].rolling(20).std() + 1e-12)
            )

            # Save
            out_path = os.path.join(OUTPUT_DIR, f"{key}.csv")
            price_df.to_csv(out_path)
            results[key] = {
                "n_rows": len(price_df),
                "n_features": len(price_df.columns),
                "date_range": [str(price_df.index.min()), str(price_df.index.max())],
                "features": list(price_df.columns),
            }
            print(f"    ✓ {len(price_df)} rows, {len(price_df.columns)} features → {out_path}")

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Feature store manifest saved to {manifest_path}")

    return results


def main():
    print("=" * 70)
    print("PHASE 3: EXOGENOUS DATA ENRICHMENT")
    print("=" * 70)

    build_feature_store()


if __name__ == "__main__":
    main()
