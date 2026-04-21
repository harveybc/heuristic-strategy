#!/usr/bin/env python3
"""Stage II-7.1.b: Fetch Binance perpetual funding rates (no auth).

Outputs CSV files under data/raw/binance/:
  - funding_btcusdt_2019_2025.csv
  - funding_ethusdt_2019_2025.csv
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BINANCE_DIR = os.path.join(BASE_DIR, "data", "raw", "binance")
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    rows: List[dict] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        data = None
        for attempt in range(5):
            try:
                resp = requests.get(FUNDING_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

        if not data:
            break

        rows.extend(data)
        current_start = int(data[-1]["fundingTime"]) + 1

        if len(data) < 1000:
            break

        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

    out = df[["symbol", "funding_time", "funding_rate"]].copy()
    out = out.dropna().drop_duplicates(subset=["symbol", "funding_time"]).sort_values("funding_time")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance funding rates for Stage II-7")
    parser.add_argument("--start", default="2019-09-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    os.makedirs(RAW_BINANCE_DIR, exist_ok=True)

    print("=" * 72)
    print("STAGE II-7.1.b — BINANCE FUNDING")
    print("=" * 72)

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\nFetching funding {symbol} ...")
        df = fetch_funding(symbol, args.start, args.end)

        if df.empty:
            print("  WARNING: no rows fetched")
            continue

        out_path = os.path.join(RAW_BINANCE_DIR, f"funding_{symbol.lower()}_{args.start[:4]}_{args.end[:4]}.csv")
        df.to_csv(out_path, index=False)
        print(
            f"  rows={len(df):,} range={df['funding_time'].iloc[0]} -> {df['funding_time'].iloc[-1]} saved={out_path}"
        )


if __name__ == "__main__":
    main()
