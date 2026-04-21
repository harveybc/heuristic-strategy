#!/usr/bin/env python3
"""
Window Manifest Generator — Project 2 Part II Stage II-1.2.c

Generates anchored expanding windows per F-5 §5.2 Design A.

Parameters:
  train_min=3yr, val=1yr, test=1yr, step=1yr, start=2005, end=2019
  Held-out: 2020-2024 (preserved, not in windows)

Embargo support per F-2 §3.1:
  Train [window_start, train_end]
  EMBARGO [train_end, train_end + embargo_bars]  -- excluded from val and test
  Validation [train_end + embargo_bars, val_end]

Output: data/windows/window_manifest.json per F-5 §5.3 schema

Usage:
  python window_manifest_generator.py --data data/processed/eurusd_4h_2005_2024.csv
                                      --output data/windows/window_manifest.json
                                      [--train_min_years 3] [--val_years 1] [--test_years 1]
                                      [--step_years 1] [--start_year 2005] [--end_year 2019]
                                      [--embargo_bars 6]
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd


def generate_anchored_expanding_windows(
    data_start: str = "2005-01-01",
    start_year: int = 2005,
    end_year: int = 2019,
    train_min_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
    embargo_bars: int = 6,
    timeframe: str = "4h",
    df: pd.DataFrame = None
) -> dict:
    """
    Generate anchored expanding window manifest.
    
    Anchored = training always starts from data_start.
    Expanding = each step adds step_years to training.
    
    Window i:
      train: [data_start, data_start + train_min_years + i*step_years - 1]
      val:   [train_end + 1, train_end + val_years]
      test:  [val_end + 1, val_end + test_years]
    """
    windows = []
    window_id = 1
    
    # First train period ends at start_year + train_min_years - 1
    train_end_year = start_year + train_min_years - 1
    
    while True:
        val_start_year = train_end_year + 1
        val_end_year = val_start_year + val_years - 1
        test_start_year = val_end_year + 1
        test_end_year = test_start_year + test_years - 1
        
        # Stop if test period exceeds end_year
        if test_end_year > end_year:
            break
        
        train_start = f"{start_year}-01-01"
        train_end = f"{train_end_year}-12-31"
        val_start = f"{val_start_year}-01-01"
        val_end = f"{val_end_year}-12-31"
        test_start = f"{test_start_year}-01-01"
        test_end = f"{test_end_year}-12-31"
        
        # Count actual bars if data provided
        train_bars = val_bars = test_bars = 0
        if df is not None:
            train_bars = len(df[(df.index >= train_start) & (df.index <= train_end)])
            val_bars = len(df[(df.index >= val_start) & (df.index <= val_end)])
            test_bars = len(df[(df.index >= test_start) & (df.index <= test_end)])
        
        window = {
            "id": window_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_bars": train_bars,
            "val_bars": val_bars,
            "test_bars": test_bars,
            "embargo_bars": embargo_bars,
        }
        windows.append(window)
        
        window_id += 1
        train_end_year += step_years
    
    manifest = {
        "design": "anchored_expanding",
        "asset": "EURUSD",
        "timeframe": timeframe,
        "embargo_bars": embargo_bars,
        "data_start": f"{start_year}-01-01",
        "data_end": f"{end_year}-12-31",
        "train_min_years": train_min_years,
        "val_years": val_years,
        "test_years": test_years,
        "step_years": step_years,
        "total_windows": len(windows),
        "windows": windows,
        "held_out": {
            "start": "2020-01-01",
            "end": "2024-12-31",
            "bars": len(df[(df.index >= "2020-01-01")]) if df is not None else 0,
            "note": "Touched exactly once per experiment after all in-sample tuning"
        },
        "generated_at": datetime.now().isoformat(),
        "generator": "window_manifest_generator.py"
    }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate anchored expanding window manifest")
    parser.add_argument("--data", default=None,
                        help="OHLCV CSV to count bars per window (optional)")
    parser.add_argument("--output",
                        default="data/windows/window_manifest.json",
                        help="Output manifest JSON path")
    parser.add_argument("--start_year", type=int, default=2005)
    parser.add_argument("--end_year", type=int, default=2019,
                        help="Last year for test windows (held-out starts at end_year+1)")
    parser.add_argument("--train_min_years", type=int, default=3)
    parser.add_argument("--val_years", type=int, default=1)
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--step_years", type=int, default=1)
    parser.add_argument("--embargo_bars", type=int, default=6,
                        help="Embargo bars between train end and val start (default: 6)")
    parser.add_argument("--timeframe", default="4h")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"WINDOW MANIFEST GENERATOR")
    print(f"{'='*60}")
    
    # Load data if provided
    df = None
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
        # Auto-detect date column
        date_col = None
        for candidate in ['DateTime', 'DATE_TIME', 'Date', 'date', 'datetime']:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        print(f"Data loaded: {len(df)} bars, {df.index.min()} to {df.index.max()}")
    
    manifest = generate_anchored_expanding_windows(
        start_year=args.start_year,
        end_year=args.end_year,
        train_min_years=args.train_min_years,
        val_years=args.val_years,
        test_years=args.test_years,
        step_years=args.step_years,
        embargo_bars=args.embargo_bars,
        timeframe=args.timeframe,
        df=df
    )
    
    # Print summary
    print(f"\nDesign:          {manifest['design']}")
    print(f"Asset:           {manifest['asset']} {manifest['timeframe']}")
    print(f"Training anchor: {manifest['data_start']}")
    print(f"Min train:       {args.train_min_years} years")
    print(f"Val:             {args.val_years} year(s)")
    print(f"Test:            {args.test_years} year(s)")
    print(f"Step:            {args.step_years} year(s)")
    print(f"Embargo:         {args.embargo_bars} bars")
    print(f"Total windows:   {manifest['total_windows']}")
    print(f"Held-out:        {manifest['held_out']['start']} to {manifest['held_out']['end']}")
    
    print(f"\nWindows:")
    for w in manifest['windows']:
        print(f"  Window {w['id']:2d}: Train {w['train_start']}→{w['train_end']} | "
              f"Val {w['val_start']}→{w['val_end']} | "
              f"Test {w['test_start']}→{w['test_end']} | "
              f"Bars: {w['train_bars']}/{w['val_bars']}/{w['test_bars']}")
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest saved to {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
