#!/usr/bin/env python3
"""Cross-validate HistData vs TrueFX 1h bars on overlap period.

Usage:
    python scripts/validate_histdata_truefx.py --asset eurusd
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Cross-validate HistData vs TrueFX')
    parser.add_argument('--asset', default='eurusd')
    args = parser.parse_args()
    
    histdata_path = os.path.join(BASE_DIR, 'data', 'raw', 'histdata', f'{args.asset}_1h_2005_2025.csv')
    truefx_path = os.path.join(BASE_DIR, 'data', 'raw', 'truefx', f'{args.asset}_1h_2009_2025.csv')
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation: HistData vs TrueFX ({args.asset.upper()})")
    print(f"{'='*60}")
    
    if not os.path.exists(histdata_path):
        print(f"  ERROR: HistData file not found: {histdata_path}")
        sys.exit(1)
    
    if not os.path.exists(truefx_path):
        print(f"  WARNING: TrueFX file not found: {truefx_path}")
        print("  SKIP cross-validation (TrueFX may not have downloaded)")
        return
    
    hd = pd.read_csv(histdata_path, parse_dates=['DateTime'], index_col='DateTime')
    tf = pd.read_csv(truefx_path, parse_dates=['DateTime'], index_col='DateTime')
    
    if len(tf) == 0:
        print("  WARNING: TrueFX file is empty. Skipping cross-validation.")
        return
    
    print(f"  HistData: {len(hd)} bars ({hd.index.min()} to {hd.index.max()})")
    print(f"  TrueFX:   {len(tf)} bars ({tf.index.min()} to {tf.index.max()})")
    
    # Align on overlap period
    overlap_start = max(hd.index.min(), tf.index.min())
    overlap_end = min(hd.index.max(), tf.index.max())
    
    hd_overlap = hd[overlap_start:overlap_end]
    tf_overlap = tf[overlap_start:overlap_end]
    
    # Join on index
    common_idx = hd_overlap.index.intersection(tf_overlap.index)
    
    if len(common_idx) == 0:
        print("  ERROR: No overlapping timestamps found")
        return
    
    hd_aligned = hd_overlap.loc[common_idx]
    tf_aligned = tf_overlap.loc[common_idx]
    
    print(f"\n  Overlap: {overlap_start} to {overlap_end}")
    print(f"  Common bars: {len(common_idx)}")
    
    # Compare Close prices
    diff = np.abs(hd_aligned['Close'] - tf_aligned['Close'])
    
    # For FX, 1 pip = 0.0001 (EUR/USD) or 0.01 (USD/JPY)
    if 'jpy' in args.asset.lower():
        pip = 0.01
    else:
        pip = 0.0001
    
    within_1pip = (diff <= 1 * pip).mean() * 100
    within_2pip = (diff <= 2 * pip).mean() * 100
    within_5pip = (diff <= 5 * pip).mean() * 100
    
    print(f"\n  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Max diff: {diff.max():.6f} ({diff.max() / pip:.1f} pips)")
    print(f"  Median diff: {diff.median():.6f}")
    
    print(f"\n  Agreement rates:")
    print(f"    Within 1 pip: {within_1pip:.1f}%")
    print(f"    Within 2 pip: {within_2pip:.1f}%")
    print(f"    Within 5 pip: {within_5pip:.1f}%")
    
    # PASS/FAIL threshold: >95% agree within 2 pips
    passed = within_2pip > 95.0
    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status} (threshold: >95% within 2 pips, actual: {within_2pip:.1f}%)")
    
    if not passed:
        print(f"  WARNING: Cross-validation FAILED. HistData and TrueFX disagree significantly.")
        print(f"  Creating ESCALATION file...")
        escalation_path = os.path.join(BASE_DIR, 'deliverables', 'ESCALATION_cross_validation.md')
        os.makedirs(os.path.dirname(escalation_path), exist_ok=True)
        with open(escalation_path, 'w') as f:
            f.write(f"# ESCALATION: Cross-Validation Failure\n\n")
            f.write(f"Asset: {args.asset.upper()}\n")
            f.write(f"Within 2 pips: {within_2pip:.1f}% (threshold: >95%)\n")
            f.write(f"Max diff: {diff.max():.6f} ({diff.max() / pip:.1f} pips)\n")


if __name__ == '__main__':
    main()
