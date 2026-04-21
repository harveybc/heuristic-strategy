#!/usr/bin/env python3
"""Stage II-7.1.c: Fetch CoinMetrics Community daily metrics (no auth).

Outputs:
  - data/raw/coinmetrics/btc_daily_metrics_2019_2025.csv
  - data/raw/coinmetrics/eth_daily_metrics_2019_2025.csv
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CM_DIR = os.path.join(BASE_DIR, "data", "raw", "coinmetrics")
BASE_URL = "https://community-api.coinmetrics.io/v4"


CANDIDATE_METRICS = [
    "AdrActCnt",      # active addresses
    "TxCnt",          # transaction count
    "FeeMeanNtv",     # mean tx fee (native units)
    "CapRealUSD",     # realized cap
    "NVTAdj90",       # nvt proxy
    "NVTAdj",
    "DiffMean",       # difficulty
    "HashRate",       # hash rate
]


def _get_json(url: str, params: Dict[str, str] | None = None) -> dict:
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=45)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {}


def get_available_assets() -> set[str]:
    payload = _get_json(f"{BASE_URL}/catalog/assets")
    return {row.get("asset") for row in payload.get("data", []) if row.get("asset")}


def fetch_single_metric(asset: str, metric: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "assets": asset,
        "metrics": metric,
        "frequency": "1d",
        "start_time": start_date,
        "end_time": end_date,
        "page_size": "10000",
    }

    rows: List[dict] = []
    url = f"{BASE_URL}/timeseries/asset-metrics"

    while url:
        payload = _get_json(url, params=params)
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")
        params = None
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["time", "asset", metric])

    df = pd.DataFrame(rows)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if metric in df.columns:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df[["time", "asset", metric]].dropna(subset=["time"]).sort_values("time")


def fetch_asset_metrics(asset: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, List[str]]:
    merged: pd.DataFrame | None = None
    selected: List[str] = []

    for metric in CANDIDATE_METRICS:
        try:
            mdf = fetch_single_metric(asset, metric, start_date=start_date, end_date=end_date)
        except requests.HTTPError:
            continue

        if mdf.empty:
            continue

        selected.append(metric)
        if merged is None:
            merged = mdf
        else:
            merged = merged.merge(mdf, on=["time", "asset"], how="outer")

    if merged is None:
        return pd.DataFrame(), []

    merged = merged.sort_values("time").drop_duplicates(subset=["time", "asset"])
    return merged, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CoinMetrics Community metrics for Stage II-7")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    os.makedirs(RAW_CM_DIR, exist_ok=True)

    print("=" * 72)
    print("STAGE II-7.1.c — COINMETRICS COMMUNITY")
    print("=" * 72)

    available_assets = get_available_assets()

    for asset in ["btc", "eth"]:
        print(f"\nAsset: {asset}")
        if asset not in available_assets:
            print("  WARNING: asset unavailable in current community API catalog")
            continue

        df, selected_metrics = fetch_asset_metrics(asset=asset, start_date=args.start, end_date=args.end)
        print(f"  Selected metrics: {', '.join(selected_metrics) if selected_metrics else 'none'}")
        if df.empty:
            print("  WARNING: no rows fetched")
            continue

        out_path = os.path.join(RAW_CM_DIR, f"{asset}_daily_metrics_{args.start[:4]}_{args.end[:4]}.csv")
        df.to_csv(out_path, index=False)
        print(f"  rows={len(df):,} range={df['time'].iloc[0]} -> {df['time'].iloc[-1]} saved={out_path}")


if __name__ == "__main__":
    main()
