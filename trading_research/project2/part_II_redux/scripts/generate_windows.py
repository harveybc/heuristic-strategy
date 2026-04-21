#!/usr/bin/env python3
"""
Generate rolling window manifest for walk-forward experiments.

Creates anchored expanding-window or fixed-size rolling-window manifests
for a given OHLCV dataset and temporal boundaries.

Usage:
  python generate_windows.py --data data/processed/btcusd_4h.csv \
      --is_start 2017-08-17 --is_end 2019-12-31 \
      --train_pct 0.6 --val_pct 0.2 --test_pct 0.2 \
      --trigger yearly --output data/windows/btcusd_4h_yearly.json

  python generate_windows.py --data data/processed/btcusd_4h.csv \
      --is_start 2017-08-17 --is_end 2019-12-31 \
      --trigger monthly --min_train_bars 1000 \
      --output data/windows/btcusd_4h_monthly.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np


def generate_yearly_expanding(df, is_start, is_end, min_train_bars=500):
    """Anchored expanding windows with yearly test periods.

    Train: is_start → year boundary (expanding)
    Val: next 6 months
    Test: following 6 months

    For short datasets (e.g. BTC IS=2017-2019), adapts to available data.
    """
    windows = []
    is_data = df[is_start:is_end]

    if len(is_data) == 0:
        return windows

    first_year = is_data.index[0].year
    last_year = is_data.index[-1].year

    # Generate yearly boundaries
    for year in range(first_year + 1, last_year + 1):
        # Train: is_start → end of (year-1)
        train_end = f"{year-1}-12-31"
        # Val: first half of year
        val_start = f"{year}-01-01"
        val_end = f"{year}-06-30"
        # Test: second half of year
        test_start = f"{year}-07-01"
        test_end = f"{year}-12-31"

        # Clip to IS period
        if test_end > is_end:
            test_end = is_end
        if val_end > is_end:
            # Not enough room for both val and test
            continue

        train_slice = df[is_start:train_end]
        val_slice = df[val_start:val_end]
        test_slice = df[test_start:test_end]

        if len(train_slice) < min_train_bars:
            continue
        if len(val_slice) < 50 or len(test_slice) < 50:
            continue

        windows.append({
            "id": len(windows) + 1,
            "train_start": is_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_bars": len(train_slice),
            "val_bars": len(val_slice),
            "test_bars": len(test_slice),
        })

    return windows


def generate_monthly_expanding(df, is_start, is_end, min_train_bars=500,
                                val_months=1, test_months=1):
    """Anchored expanding windows with monthly test periods."""
    windows = []
    is_data = df[is_start:is_end]

    if len(is_data) == 0:
        return windows

    # Generate monthly boundaries
    months = pd.date_range(start=is_start, end=is_end, freq='MS')

    # Need at least min_train_bars + val + test months
    for i in range(len(months)):
        train_end_dt = months[i] - pd.Timedelta(days=1)
        train_slice = df[is_start:str(train_end_dt.date())]

        if len(train_slice) < min_train_bars:
            continue

        # Validation: next val_months months
        val_start_dt = months[i]
        val_end_idx = i + val_months
        if val_end_idx >= len(months):
            break
        val_end_dt = months[val_end_idx] - pd.Timedelta(days=1)

        # Test: next test_months months
        test_start_dt = months[val_end_idx]
        test_end_idx = val_end_idx + test_months
        if test_end_idx >= len(months):
            test_end_dt = pd.Timestamp(is_end)
        else:
            test_end_dt = months[test_end_idx] - pd.Timedelta(days=1)

        if str(test_end_dt.date()) > is_end:
            test_end_dt = pd.Timestamp(is_end)

        val_slice = df[str(val_start_dt.date()):str(val_end_dt.date())]
        test_slice = df[str(test_start_dt.date()):str(test_end_dt.date())]

        if len(val_slice) < 20 or len(test_slice) < 20:
            continue

        windows.append({
            "id": len(windows) + 1,
            "train_start": is_start,
            "train_end": str(train_end_dt.date()),
            "val_start": str(val_start_dt.date()),
            "val_end": str(val_end_dt.date()),
            "test_start": str(test_start_dt.date()),
            "test_end": str(test_end_dt.date()),
            "train_bars": len(train_slice),
            "val_bars": len(val_slice),
            "test_bars": len(test_slice),
        })

    return windows


def generate_weekly_expanding(df, is_start, is_end, min_train_bars=500,
                               val_weeks=2, test_weeks=2):
    """Anchored expanding windows with weekly test periods."""
    windows = []
    is_data = df[is_start:is_end]

    if len(is_data) == 0:
        return windows

    weeks = pd.date_range(start=is_start, end=is_end, freq='W-MON')

    for i in range(len(weeks)):
        train_end_dt = weeks[i] - pd.Timedelta(days=1)
        train_slice = df[is_start:str(train_end_dt.date())]

        if len(train_slice) < min_train_bars:
            continue

        val_start_dt = weeks[i]
        val_end_idx = i + val_weeks
        if val_end_idx >= len(weeks):
            break
        val_end_dt = weeks[val_end_idx] - pd.Timedelta(days=1)

        test_start_dt = weeks[val_end_idx]
        test_end_idx = val_end_idx + test_weeks
        if test_end_idx >= len(weeks):
            test_end_dt = pd.Timestamp(is_end)
        else:
            test_end_dt = weeks[test_end_idx] - pd.Timedelta(days=1)

        if str(test_end_dt.date()) > is_end:
            test_end_dt = pd.Timestamp(is_end)

        val_slice = df[str(val_start_dt.date()):str(val_end_dt.date())]
        test_slice = df[str(test_start_dt.date()):str(test_end_dt.date())]

        if len(val_slice) < 10 or len(test_slice) < 10:
            continue

        windows.append({
            "id": len(windows) + 1,
            "train_start": is_start,
            "train_end": str(train_end_dt.date()),
            "val_start": str(val_start_dt.date()),
            "val_end": str(val_end_dt.date()),
            "test_start": str(test_start_dt.date()),
            "test_end": str(test_end_dt.date()),
            "train_bars": len(train_slice),
            "val_bars": len(val_slice),
            "test_bars": len(test_slice),
        })

    return windows


def main():
    parser = argparse.ArgumentParser(description="Generate rolling window manifest")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--is_start", default="2005-01-01", help="IS start date")
    parser.add_argument("--is_end", default="2019-12-31", help="IS end date")
    parser.add_argument("--trigger", default="yearly",
                        choices=["yearly", "monthly", "weekly"],
                        help="Retraining trigger frequency")
    parser.add_argument("--min_train_bars", type=int, default=500,
                        help="Minimum training bars per window")
    parser.add_argument("--val_months", type=int, default=1,
                        help="Validation months (for monthly trigger)")
    parser.add_argument("--test_months", type=int, default=1,
                        help="Test months (for monthly trigger)")
    parser.add_argument("--val_weeks", type=int, default=2,
                        help="Validation weeks (for weekly trigger)")
    parser.add_argument("--test_weeks", type=int, default=2,
                        help="Test weeks (for weekly trigger)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]
    print(f"Loaded {args.data}: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # Clip to available data
    actual_start = max(str(df.index[0].date()), args.is_start)
    actual_end = min(str(df.index[-1].date()), args.is_end)
    print(f"IS period: {actual_start} to {actual_end}")

    if args.trigger == "yearly":
        windows = generate_yearly_expanding(df, actual_start, actual_end,
                                            min_train_bars=args.min_train_bars)
    elif args.trigger == "monthly":
        windows = generate_monthly_expanding(df, actual_start, actual_end,
                                              min_train_bars=args.min_train_bars,
                                              val_months=args.val_months,
                                              test_months=args.test_months)
    elif args.trigger == "weekly":
        windows = generate_weekly_expanding(df, actual_start, actual_end,
                                             min_train_bars=args.min_train_bars,
                                             val_weeks=args.val_weeks,
                                             test_weeks=args.test_weeks)

    # Build manifest
    manifest = {
        "asset": os.path.basename(args.data).replace(".csv", ""),
        "trigger": args.trigger,
        "is_start": actual_start,
        "is_end": actual_end,
        "min_train_bars": args.min_train_bars,
        "total_windows": len(windows),
        "generated": datetime.now().isoformat(),
        "windows": windows,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(windows)} windows ({args.trigger} trigger)")
    for w in windows:
        print(f"  Window {w['id']}: train={w['train_bars']:,} bars "
              f"({w['train_start']}→{w['train_end']}), "
              f"val={w['val_bars']:,}, test={w['test_bars']:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
