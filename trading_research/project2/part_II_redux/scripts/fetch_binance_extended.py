#!/usr/bin/env python3
"""Stage II-7.1.a: Fetch extended Binance OHLCV (public API, no auth).

Outputs Parquet files under data/raw/binance/:
  - btcusdt_5m_2019_2025.parquet
  - btcusdt_15m_2019_2025.parquet
  - btcusdt_1h_2019_2025.parquet
  - ethusdt_5m_2019_2025.parquet
  - ethusdt_15m_2019_2025.parquet
  - ethusdt_1h_2019_2025.parquet
  - ethusdt_4h_2019_2025.parquet
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, List

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BINANCE_DIR = os.path.join(BASE_DIR, "data", "raw", "binance")
KLINES_URL = "https://api.binance.com/api/v3/klines"


def _ensure_parquet_support() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Parquet writing requires pyarrow. Install in current env: pip install pyarrow"
        ) from exc


def fetch_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    pause_s: float = 0.15,
) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    rows: List[list] = []
    current_start = start_ms
    limit = 1000

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }

        data = None
        for attempt in range(5):
            try:
                resp = requests.get(KLINES_URL, params=params, timeout=30)
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
        current_start = int(data[-1][0]) + 1

        if len(data) < limit:
            break

        time.sleep(pause_s)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["DateTime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df[["DateTime", "open", "high", "low", "close", "volume", "quote_volume", "trades"]].copy()
    out = out.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "quote_volume": "QuoteVolume",
            "trades": "Trades",
        }
    )
    out = out.dropna().drop_duplicates(subset=["DateTime"]).sort_values("DateTime")
    return out


def iter_targets() -> Iterable[tuple[str, str]]:
    for tf in ["5m", "15m", "1h"]:
        yield ("BTCUSDT", tf)
    for tf in ["5m", "15m", "1h", "4h"]:
        yield ("ETHUSDT", tf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch extended Binance OHLCV for Stage II-7")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    _ensure_parquet_support()
    os.makedirs(RAW_BINANCE_DIR, exist_ok=True)

    print("=" * 72)
    print("STAGE II-7.1.a — BINANCE OHLCV EXTENSION")
    print("=" * 72)

    for symbol, interval in iter_targets():
        print(f"\nFetching {symbol} {interval} ...")
        df = fetch_klines(symbol=symbol, interval=interval, start_date=args.start, end_date=args.end)

        if df.empty:
            print("  WARNING: no rows fetched")
            continue

        out_name = f"{symbol.lower()}_{interval}_{args.start[:4]}_{args.end[:4]}.parquet"
        out_path = os.path.join(RAW_BINANCE_DIR, out_name)
        df.to_parquet(out_path, index=False)

        print(
            f"  rows={len(df):,} range={df['DateTime'].iloc[0]} -> {df['DateTime'].iloc[-1]} saved={out_path}"
        )


if __name__ == "__main__":
    main()
