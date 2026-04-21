#!/usr/bin/env python3
"""Stage II-7.1.d: Fetch Blockchain.com supplementary BTC metrics.

Outputs:
  - data/raw/blockchain_com/btc_metrics_2019_2025.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BC_DIR = os.path.join(BASE_DIR, "data", "raw", "blockchain_com")
BASE_URL = "https://api.blockchain.info/charts"


METRICS: Dict[str, str] = {
    "mempool_size": "mempool-size",
    "confirmed_tx_per_block": "n-transactions-per-block",
    "hash_rate": "hash-rate",
}


def fetch_chart(metric_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    chart_name = METRICS[metric_key]
    url = f"{BASE_URL}/{chart_name}"
    params = {
        "timespan": "all",
        "format": "json",
        "sampled": "false",
    }

    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    payload = resp.json()

    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(columns=["time", metric_key])

    df = pd.DataFrame(values)
    df["time"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.rename(columns={"y": metric_key})[["time", metric_key]]
    df = df[(df["time"] >= pd.Timestamp(start_date, tz="UTC")) & (df["time"] <= pd.Timestamp(end_date, tz="UTC"))]
    df = df.sort_values("time").drop_duplicates(subset=["time"])

    # Normalize mixed-frequency chart output to daily bars for consistent joins.
    df["time"] = df["time"].dt.floor("D")
    df = df.groupby("time", as_index=False)[metric_key].mean()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Blockchain.com supplementary metrics for Stage II-7")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    os.makedirs(RAW_BC_DIR, exist_ok=True)

    print("=" * 72)
    print("STAGE II-7.1.d — BLOCKCHAIN.COM SUPPLEMENTARY")
    print("=" * 72)

    merged = None
    for metric_key in METRICS:
        df = fetch_chart(metric_key, start_date=args.start, end_date=args.end)
        print(f"  {metric_key}: rows={len(df):,}")
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="time", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No Blockchain.com data fetched")

    merged = merged.sort_values("time")
    out_path = os.path.join(RAW_BC_DIR, f"btc_metrics_{args.start[:4]}_{args.end[:4]}.csv")
    merged.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(merged):,} range={merged['time'].iloc[0]} -> {merged['time'].iloc[-1]}")


if __name__ == "__main__":
    main()
