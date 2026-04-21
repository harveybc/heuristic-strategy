#!/usr/bin/env python3
"""
Information Coefficient Analysis — Project 2 Part II Stage II-1.3.c

Per F-2 §3.3 (MEDIUM priority addition from Jansen framework):
  - Compute IC per feature vs forward returns at horizons (1, 6, 24 bars)
  - Compute rolling 1-year IC mean, std, IR (IC/std) per feature
  - Diagnostic only — does not gate model training

If IC analysis shows surprisingly high IC contradicting F-6 null finding,
escalate for re-examination.

Output: ic_analysis_phase0.csv + ic_analysis_report.json

Usage:
  python ic_analysis.py --data data/processed/eurusd_4h_2005_2024.csv
                        [--features_file data/processed/eurusd_4h_features.csv]
                        [--horizons 1 6 24]
                        [--output_dir data/processed/]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(prices: pd.Series, horizons: list) -> pd.DataFrame:
    """Compute forward log-returns at multiple horizons."""
    fwd = {}
    for h in horizons:
        fwd[f'fwd_ret_{h}'] = np.log(prices.shift(-h) / prices)
    return pd.DataFrame(fwd, index=prices.index)


def compute_ic(feature: pd.Series, returns: pd.Series) -> dict:
    """
    Compute Information Coefficient (rank correlation) between
    feature and forward returns.
    """
    # Align and drop NaN
    aligned = pd.concat([feature, returns], axis=1).dropna()
    if len(aligned) < 30:
        return {'ic': np.nan, 'p_value': np.nan, 'n': len(aligned)}
    
    ic, p_val = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return {
        'ic': float(ic),
        'p_value': float(p_val),
        'n': int(len(aligned)),
    }


def compute_rolling_ic(feature: pd.Series, returns: pd.Series,
                        window: int = 1560) -> pd.Series:
    """
    Rolling IC with specified window.
    Default window=1560 (1 year of 4h bars: 260 days * 6 bars/day).
    """
    aligned = pd.concat([feature, returns], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    # Compute rolling correlation between the two series
    rolling_ic = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
    return rolling_ic


def run_ic_analysis(features_df: pd.DataFrame, prices: pd.Series,
                     horizons: list = [1, 6, 24],
                     rolling_window: int = 1560) -> dict:
    """
    Run full IC analysis across all features and horizons.
    
    Returns:
        Dict with per-feature, per-horizon IC results.
    """
    # Compute forward returns
    fwd_returns = compute_forward_returns(prices, horizons)
    
    results = []
    feature_cols = [c for c in features_df.columns 
                    if c.upper() not in ('DATE', 'DATE_TIME', 'DATETIME',
                                         'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME')]
    
    print(f"\nAnalysing {len(feature_cols)} features × {len(horizons)} horizons")
    print(f"Rolling window: {rolling_window} bars")
    
    for col in feature_cols:
        for h in horizons:
            ret_col = f'fwd_ret_{h}'
            if ret_col not in fwd_returns.columns:
                continue
            
            # Full-sample IC
            ic_result = compute_ic(features_df[col], fwd_returns[ret_col])
            
            # Rolling IC statistics
            rolling_ic = compute_rolling_ic(features_df[col], fwd_returns[ret_col],
                                            window=rolling_window)
            
            ic_mean = float(rolling_ic.mean()) if len(rolling_ic) > 0 else np.nan
            ic_std = float(rolling_ic.std()) if len(rolling_ic) > 0 else np.nan
            ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
            
            row = {
                'feature': col,
                'horizon': h,
                'ic_full_sample': ic_result['ic'],
                'ic_p_value': ic_result['p_value'],
                'ic_n_obs': ic_result['n'],
                'ic_rolling_mean': ic_mean,
                'ic_rolling_std': ic_std,
                'ic_ir': ic_ir,
                'significant_5pct': ic_result['p_value'] < 0.05 if not np.isnan(ic_result['p_value']) else False,
            }
            results.append(row)
            
            sig = "***" if row['significant_5pct'] else "   "
            print(f"  {col:25s} h={h:3d}: IC={ic_result['ic']:+.4f} "
                  f"(p={ic_result['p_value']:.4f}) {sig} "
                  f"Rolling: μ={ic_mean:+.4f} σ={ic_std:.4f} IR={ic_ir:+.3f}")
    
    return results


def check_f6_contradiction(results: list, ic_threshold: float = 0.10) -> list:
    """
    Check if any feature shows IC that contradicts F-6 null finding.
    F-6 found no lagged causal structure at 4h.
    If IC > threshold consistently, it may indicate exploitable signal
    that PCMCI missed (different test, different conditions).
    
    Returns list of flagged features.
    """
    flags = []
    for r in results:
        if abs(r.get('ic_full_sample', 0)) > ic_threshold and r.get('significant_5pct', False):
            flags.append({
                'feature': r['feature'],
                'horizon': r['horizon'],
                'ic': r['ic_full_sample'],
                'p_value': r['ic_p_value'],
                'note': f"IC={r['ic_full_sample']:.4f} exceeds ±{ic_threshold} threshold. "
                        f"May contradict F-6 null. Investigate before committing to Path B."
            })
    return flags


def main():
    parser = argparse.ArgumentParser(description="IC Analysis for Phase 0 features")
    parser.add_argument("--data", required=True,
                        help="OHLCV CSV with DateTime index")
    parser.add_argument("--features_file", default=None,
                        help="Pre-computed features CSV (if not provided, generates from OHLCV)")
    parser.add_argument("--horizons", nargs='+', type=int, default=[1, 6, 24],
                        help="Forward return horizons in bars (default: 1 6 24)")
    parser.add_argument("--rolling_window", type=int, default=1560,
                        help="Rolling window for IC stats (default: 1560 = ~1 year of 4h)")
    parser.add_argument("--output_dir", default="data/processed/",
                        help="Output directory")
    parser.add_argument("--ic_threshold", type=float, default=0.10,
                        help="IC threshold for F-6 contradiction flag")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"INFORMATION COEFFICIENT ANALYSIS")
    print(f"{'='*60}")
    
    # Load price data
    df = pd.read_csv(args.data)
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
    
    prices = df['Close'] if 'Close' in df.columns else df.iloc[:, -1]
    print(f"Price data: {len(prices)} bars, {prices.index.min()} to {prices.index.max()}")
    
    # Load or compute features
    if args.features_file and os.path.exists(args.features_file):
        features_df = pd.read_csv(args.features_file)
        for candidate in ['DateTime', 'DATE_TIME', 'Date', 'date', 'datetime']:
            if candidate in features_df.columns:
                features_df[candidate] = pd.to_datetime(features_df[candidate])
                features_df.set_index(candidate, inplace=True)
                break
        print(f"Features loaded: {features_df.shape}")
    else:
        # Generate basic technical features inline
        print("No features file — computing basic technical indicators...")
        features_df = _compute_basic_features(df)
        print(f"Generated {features_df.shape[1]} features")
    
    # Run IC analysis
    results = run_ic_analysis(features_df, prices,
                               horizons=args.horizons,
                               rolling_window=args.rolling_window)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "ic_analysis_phase0.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ IC analysis saved to {csv_path}")
    
    # Check for F-6 contradictions
    flags = check_f6_contradiction(results, ic_threshold=args.ic_threshold)
    
    # Summary report
    report = {
        "total_features": len(set(r['feature'] for r in results)),
        "horizons_tested": args.horizons,
        "rolling_window": args.rolling_window,
        "significant_ics": sum(1 for r in results if r.get('significant_5pct', False)),
        "total_tests": len(results),
        "f6_contradiction_flags": flags,
        "interpretation": "",
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    if flags:
        report["interpretation"] = (
            f"WARNING: {len(flags)} feature-horizon combinations show IC "
            f"exceeding ±{args.ic_threshold} with p<0.05. This may contradict "
            f"F-6 null finding. Review before Path B commitment."
        )
        print(f"\n⚠️  F-6 CONTRADICTION FLAGS: {len(flags)} features")
        for f in flags:
            print(f"   {f['feature']} h={f['horizon']}: IC={f['ic']:.4f} (p={f['p_value']:.4f})")
    else:
        report["interpretation"] = (
            f"No features exceed IC ±{args.ic_threshold} threshold. "
            f"Consistent with F-6 null finding at 4h."
        )
        print(f"\n✓ No F-6 contradictions found. Consistent with causal null at 4h.")
    
    report_path = os.path.join(args.output_dir, "ic_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Report saved to {report_path}")
    
    # Print summary table
    print(f"\n--- IC Summary (top features by |IC|) ---")
    results_df_sorted = results_df.reindex(
        results_df['ic_full_sample'].abs().sort_values(ascending=False).index
    ).head(20)
    print(results_df_sorted[['feature', 'horizon', 'ic_full_sample', 'ic_p_value',
                              'ic_rolling_mean', 'ic_ir', 'significant_5pct']].to_string())
    
    print(f"\n{'='*60}\n")


def _compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical features from OHLCV data."""
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 3]
    high = df['High'] if 'High' in df.columns else df.iloc[:, 1]
    low = df['Low'] if 'Low' in df.columns else df.iloc[:, 2]
    
    features = pd.DataFrame(index=df.index)
    
    # Returns
    features['log_return'] = np.log(close / close.shift(1))
    features['return_5'] = close.pct_change(5)
    features['return_20'] = close.pct_change(20)
    
    # Moving averages
    features['ema_14'] = close.ewm(span=14).mean()
    features['ema_50'] = close.ewm(span=50).mean()
    features['ema_200'] = close.ewm(span=200).mean()
    features['ema_14_50_ratio'] = features['ema_14'] / features['ema_50']
    
    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ATR (14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean()
    
    # Bollinger Bands position
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['bb_position'] = (close - bb_mid) / (2 * bb_std.replace(0, 1e-10))
    
    # Momentum
    features['momentum_10'] = close - close.shift(10)
    
    # Volatility
    features['volatility_20'] = features['log_return'].rolling(20).std()
    
    # Volume (if available)
    if 'Volume' in df.columns:
        features['volume_sma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return features.dropna()


if __name__ == "__main__":
    main()
