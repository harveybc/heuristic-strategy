#!/usr/bin/env python3
"""
Change-Point Window Manifest Generator for Experiment A3 (UL-2).

Uses Wasserstein distance on rolling feature distributions to detect
regime changes. When a regime change is detected, a new retraining
window is triggered.

Approach:
  1. Slide a trailing window (default 252 bars = ~1 month 4h) over data
  2. Compute rolling Wasserstein distance between current window and
     reference distribution (from training start)
  3. When distance exceeds threshold, trigger a retraining point
  4. Build anchored expanding windows with val/test around each trigger

Usage:
  python gen_changepoint_manifest.py \
      --data data/processed/eurusd_4h_2005_2024.csv \
      --output data/windows/changepoint_manifest.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


def detect_change_points(
    df: pd.DataFrame,
    detection_start: str = "2008-01-01",
    detection_end: str = "2019-12-31",
    ref_window: int = 1560,   # ~6 months of 4h bars for reference
    test_window: int = 252,   # ~1 month of 4h bars for current
    step: int = 126,          # check every ~2 weeks
    threshold_percentile: float = 90.0,  # trigger when Wasserstein > P90
    min_gap_bars: int = 1560,  # minimum ~6 months between triggers
    features: list = None
) -> list:
    """
    Detect regime change points using Wasserstein distance.
    
    Returns list of (date_str, distance) tuples for each change point.
    """
    if features is None:
        # Use returns and volatility as regime features
        close_col = None
        for c in ['Close', 'CLOSE', 'close']:
            if c in df.columns:
                close_col = c
                break
        if close_col is None:
            close_col = df.columns[3]  # Assume OHLC ordering
        
        df = df.copy()
        df['_returns'] = df[close_col].pct_change()
        df['_volatility'] = df['_returns'].rolling(60).std()
        df['_momentum'] = df[close_col].pct_change(60)
        features = ['_returns', '_volatility', '_momentum']
    
    df_clean = df[features].dropna()
    
    # Only look at detection period
    mask = (df_clean.index >= detection_start) & (df_clean.index <= detection_end)
    detect_range = df_clean.loc[mask]
    
    if len(detect_range) < test_window + step:
        return []
    
    # Reference distribution: first ref_window bars of the full dataset
    ref_data = df_clean.iloc[:ref_window]
    
    # Compute Wasserstein distances
    distances = []
    indices = range(0, len(detect_range) - test_window, step)
    
    for i in indices:
        current = detect_range.iloc[i:i + test_window]
        # Multivariate Wasserstein approximation: mean of per-feature distances
        dist = 0.0
        for feat in features:
            ref_vals = ref_data[feat].dropna().values
            cur_vals = current[feat].dropna().values
            if len(ref_vals) > 10 and len(cur_vals) > 10:
                dist += stats.wasserstein_distance(ref_vals, cur_vals)
        dist /= len(features)
        distances.append((detect_range.index[i + test_window - 1], dist))
    
    if not distances:
        return []
    
    # Compute threshold from all distances
    all_dists = [d[1] for d in distances]
    threshold = np.percentile(all_dists, threshold_percentile)
    
    # Find change points (above threshold with minimum gap)
    change_points = []
    last_trigger_idx = -min_gap_bars
    
    for date, dist in distances:
        idx = df_clean.index.get_loc(date)
        if dist > threshold and (idx - last_trigger_idx) >= min_gap_bars:
            change_points.append((str(date.date()) if hasattr(date, 'date') else str(date)[:10], float(dist)))
            last_trigger_idx = idx
    
    return change_points


def build_changepoint_manifest(
    df: pd.DataFrame,
    change_points: list,
    anchor: str = "2005-01-01",
    val_months: int = 3,
    test_months: int = 6,
    embargo_bars: int = 6,
    end_limit: str = "2019-12-31"
) -> dict:
    """Build manifest from detected change points.
    
    Each change point triggers a retraining:
    - Train: anchor → change_point
    - Val: next val_months
    - Test: next test_months
    """
    from dateutil.relativedelta import relativedelta
    
    windows = []
    window_id = 1
    
    for cp_date_str, cp_dist in change_points:
        cp_date = pd.Timestamp(cp_date_str)
        
        train_start = anchor
        train_end = cp_date_str
        
        val_start = cp_date + relativedelta(days=1)
        val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)
        test_start = val_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)
        
        if test_end > pd.Timestamp(end_limit):
            # Truncate test to end_limit
            test_end = pd.Timestamp(end_limit)
            if test_start > test_end:
                continue
        
        train_bars = len(df[(df.index >= train_start) & (df.index <= train_end)])
        val_bars = len(df[(df.index >= str(val_start.date())) & (df.index <= str(val_end.date()))])
        test_bars = len(df[(df.index >= str(test_start.date())) & (df.index <= str(test_end.date()))])
        
        if train_bars < 1000 or test_bars < 50:
            continue
        
        window = {
            "id": window_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_bars": train_bars,
            "val_bars": val_bars,
            "test_bars": test_bars,
            "embargo_bars": embargo_bars,
            "trigger_distance": cp_dist,
        }
        windows.append(window)
        window_id += 1
    
    manifest = {
        "design": "change_point_triggered",
        "asset": "EURUSD",
        "timeframe": "4h",
        "embargo_bars": embargo_bars,
        "detection_method": "wasserstein_distance",
        "total_windows": len(windows),
        "change_points_detected": len(change_points),
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
    parser = argparse.ArgumentParser(description="Change-point manifest generator for A3")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="data/windows/changepoint_manifest.json")
    parser.add_argument("--embargo_bars", type=int, default=6)
    parser.add_argument("--threshold_percentile", type=float, default=90.0)
    parser.add_argument("--min_gap_bars", type=int, default=1560,
                        help="Minimum bars between triggers (~6 months at 4h)")
    args = parser.parse_args()

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
    print(f"Data: {len(df)} bars, {df.index.min()} → {df.index.max()}")

    print("Detecting change points...")
    cps = detect_change_points(
        df,
        threshold_percentile=args.threshold_percentile,
        min_gap_bars=args.min_gap_bars
    )
    print(f"Change points detected: {len(cps)}")
    for date, dist in cps:
        print(f"  {date}: Wasserstein distance = {dist:.6f}")

    manifest = build_changepoint_manifest(
        df, cps, embargo_bars=args.embargo_bars
    )
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {manifest['total_windows']} change-point windows → {args.output}")


if __name__ == "__main__":
    main()
