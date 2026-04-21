#!/usr/bin/env python3
"""
Monthly Window Manifest Generator for Experiment A2.

Generates anchored expanding windows with MONTHLY test steps.
- Train: anchored from 2005, expanding
- Val: 3 months
- Test: 1 month
- Step: 1 month
- First test: 2008-01 (after 3yr train min)
- Last test: 2019-12
- This produces ~144 windows (12 years × 12 months)

Usage:
  python gen_monthly_manifest.py \
      --data data/processed/eurusd_4h_2005_2024.csv \
      --output data/windows/monthly_manifest.json
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


def generate_monthly_windows(
    start_date: str = "2005-01-01",
    end_date: str = "2019-12-31",
    train_min_months: int = 36,  # 3 years
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    embargo_bars: int = 6,
    df: pd.DataFrame = None
) -> dict:
    """Generate monthly-stepped anchored expanding windows."""
    windows = []
    window_id = 1

    anchor = pd.Timestamp(start_date)
    end_limit = pd.Timestamp(end_date)

    # First train end = anchor + train_min_months
    train_end = anchor + relativedelta(months=train_min_months) - relativedelta(days=1)

    while True:
        val_start = train_end + relativedelta(days=1)
        val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)
        test_start = val_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end > end_limit:
            break

        # Count bars if data available
        train_bars = val_bars = test_bars = 0
        if df is not None:
            train_bars = len(df[(df.index >= str(anchor.date())) & (df.index <= str(train_end.date()))])
            val_bars = len(df[(df.index >= str(val_start.date())) & (df.index <= str(val_end.date()))])
            test_bars = len(df[(df.index >= str(test_start.date())) & (df.index <= str(test_end.date()))])

        window = {
            "id": window_id,
            "train_start": str(anchor.date()),
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_bars": train_bars,
            "val_bars": val_bars,
            "test_bars": test_bars,
            "embargo_bars": embargo_bars,
        }
        windows.append(window)

        window_id += 1
        train_end = train_end + relativedelta(months=step_months)

    manifest = {
        "design": "anchored_expanding_monthly",
        "asset": "EURUSD",
        "timeframe": "4h",
        "embargo_bars": embargo_bars,
        "data_start": start_date,
        "data_end": str(end_limit.date()),
        "train_min_months": train_min_months,
        "val_months": val_months,
        "test_months": test_months,
        "step_months": step_months,
        "total_windows": len(windows),
        "windows": windows,
        "held_out": {
            "start": "2020-01-01",
            "end": "2024-12-31",
            "bars": len(df[(df.index >= "2020-01-01")]) if df is not None else 0,
        },
        "generated_at": datetime.now().isoformat(),
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Monthly manifest generator for A2")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="data/windows/monthly_manifest.json")
    parser.add_argument("--embargo_bars", type=int, default=6)
    args = parser.parse_args()

    df = None
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
        date_col = None
        for c in ['DateTime', 'DATE_TIME', 'Date', 'date', 'datetime']:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        print(f"Data: {len(df)} bars")

    manifest = generate_monthly_windows(embargo_bars=args.embargo_bars, df=df)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {manifest['total_windows']} monthly windows → {args.output}")


if __name__ == "__main__":
    main()
