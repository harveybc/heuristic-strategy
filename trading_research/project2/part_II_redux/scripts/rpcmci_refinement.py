#!/usr/bin/env python3
"""
Stage II-4: CI-2 Refinement via Rolling PCMCI+ (RPCMCI) on BTC/USD 4h.

Runs PCMCI+ on overlapping windows of the IS period to check if the
rsi→return causal link (found at lag-6 with MCI=-0.2459 in full IS)
is temporally stable or regime-specific.

Windows: Yearly expanding windows of the IS data.
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure tigramite is available
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError:
    print("ERROR: tigramite not installed. Run: pip install tigramite")
    sys.exit(1)


def compute_features(df):
    """Compute the 12 F-6 technical features + 6-bar forward log return."""
    close = df['close']
    high = df['high']
    low = df['low']

    # ADX (simplified via DI spread proxy)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx = dx.rolling(14).mean()
    di_spread = plus_di - minus_di

    # ATR-based
    atr_pct = atr14 / close * 100
    atr_ratio = atr14 / atr14.rolling(120).mean()

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width_pct = (bb_upper - bb_lower) / sma20 * 100
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)

    # ROC
    roc_12 = close.pct_change(12) * 100

    # EMA
    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    price_vs_ema50 = (close - ema50) / ema50 * 100

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    ema_alignment = (ema12 - ema26) / atr14

    # Stoch K
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-10)

    # MACD histogram
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line

    # Target: 6-bar forward log return
    target = np.log(close.shift(-6) / close)

    features = pd.DataFrame({
        'adx': adx,
        'di_spread': di_spread,
        'atr_pct': atr_pct,
        'atr_ratio': atr_ratio,
        'bb_width_pct': bb_width_pct,
        'bb_position': bb_position,
        'rsi': rsi,
        'roc_12': roc_12,
        'price_vs_ema50': price_vs_ema50,
        'ema_alignment': ema_alignment,
        'stoch_k': stoch_k,
        'macd_hist': macd_hist,
        'target_fwd_6bar_logret': target,
    }, index=df.index)

    return features.dropna()


def run_pcmci_window(data_array, var_names, tau_max=10, pc_alpha=0.01,
                      alpha_level=0.05, max_samples=5000):
    """Run PCMCI+ on a data window. Returns results dict."""
    n_samples = data_array.shape[0]
    if n_samples > max_samples:
        data_array = data_array[-max_samples:]

    dataframe = pp.DataFrame(data_array, var_names=var_names)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)

    target_idx = len(var_names) - 1  # last column is target
    lagged_links = []

    for feat_idx in range(len(var_names) - 1):
        for tau in range(1, tau_max + 1):
            val = results['val_matrix'][feat_idx, target_idx, tau]
            pval = results['p_matrix'][feat_idx, target_idx, tau]
            if pval < alpha_level:
                lagged_links.append({
                    'feature': var_names[feat_idx],
                    'tau': tau,
                    'mci': float(val),
                    'p_value': float(pval),
                })

    return {
        'n_samples': int(data_array.shape[0]),
        'lagged_links': lagged_links,
        'n_lagged_links': len(lagged_links),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rolling PCMCI+ for CI-2 refinement")
    parser.add_argument("--data", required=True, help="BTC 4h CSV")
    parser.add_argument("--is_start", default="2017-08-17")
    parser.add_argument("--is_end", default="2019-12-31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau_max", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]
    is_data = df[args.is_start:args.is_end]
    print(f"IS data: {len(is_data)} bars ({args.is_start} to {args.is_end})")

    # Compute features
    features = compute_features(is_data)
    print(f"Features computed: {len(features)} rows, {len(features.columns)} columns")
    var_names = list(features.columns)
    data_array = features.values.astype(np.float64)

    # Remove any rows with NaN/Inf
    mask = np.all(np.isfinite(data_array), axis=1)
    data_array = data_array[mask]
    print(f"Valid rows: {data_array.shape[0]}")

    results = {
        "stage": "II-4",
        "asset": "btcusd_4h",
        "is_period": f"{args.is_start} to {args.is_end}",
        "generated": datetime.now().isoformat(),
        "params": {
            "tau_max": args.tau_max,
            "pc_alpha": 0.01,
            "alpha_level": 0.05,
            "max_samples": args.max_samples,
        },
        "windows": [],
    }

    # Define rolling windows: expanding yearly
    window_configs = [
        ("full_is", 0, len(data_array)),
    ]

    # Also do yearly sub-windows
    # Features have datetime index, get years
    valid_features = features[mask]
    years = sorted(valid_features.index.year.unique())
    for year in years:
        year_mask = np.array(valid_features.index.year == year)
        year_data = data_array[year_mask[:len(data_array)]] if year_mask.sum() > 0 else None
        if year_data is not None and len(year_data) > 200:
            window_configs.append((f"year_{year}", None, None))

    # Run full IS PCMCI+
    print(f"\n--- Running PCMCI+ on full IS ({data_array.shape[0]} bars) ---")
    t0 = time.time()
    full_result = run_pcmci_window(data_array, var_names, tau_max=args.tau_max,
                                     max_samples=args.max_samples)
    elapsed = time.time() - t0
    full_result['window'] = 'full_is'
    full_result['elapsed_s'] = round(elapsed, 1)
    results['windows'].append(full_result)
    print(f"  Time: {elapsed:.1f}s, Links: {full_result['n_lagged_links']}")
    for link in full_result['lagged_links']:
        print(f"    {link['feature']} t-{link['tau']}: MCI={link['mci']:.4f}, p={link['p_value']:.6f}")

    # Run per-year windows
    for year in years:
        year_mask = np.array(valid_features.index.year == year)
        year_data = data_array[year_mask[:len(data_array)]] if year_mask.sum() > 0 else None
        if year_data is None or len(year_data) < 200:
            print(f"\n--- Year {year}: SKIP (only {len(year_data) if year_data is not None else 0} bars) ---")
            continue

        print(f"\n--- Year {year} ({len(year_data)} bars) ---")
        t0 = time.time()
        yr_result = run_pcmci_window(year_data, var_names, tau_max=args.tau_max,
                                      max_samples=args.max_samples)
        elapsed = time.time() - t0
        yr_result['window'] = f'year_{year}'
        yr_result['elapsed_s'] = round(elapsed, 1)
        results['windows'].append(yr_result)
        print(f"  Time: {elapsed:.1f}s, Links: {yr_result['n_lagged_links']}")
        for link in yr_result['lagged_links']:
            print(f"    {link['feature']} t-{link['tau']}: MCI={link['mci']:.4f}, p={link['p_value']:.6f}")

    # Also run first-half vs second-half
    mid = len(data_array) // 2
    for label, start, end in [("first_half", 0, mid), ("second_half", mid, len(data_array))]:
        subset = data_array[start:end]
        print(f"\n--- {label} ({len(subset)} bars) ---")
        t0 = time.time()
        half_result = run_pcmci_window(subset, var_names, tau_max=args.tau_max,
                                        max_samples=args.max_samples)
        elapsed = time.time() - t0
        half_result['window'] = label
        half_result['elapsed_s'] = round(elapsed, 1)
        results['windows'].append(half_result)
        print(f"  Time: {elapsed:.1f}s, Links: {half_result['n_lagged_links']}")
        for link in half_result['lagged_links']:
            print(f"    {link['feature']} t-{link['tau']}: MCI={link['mci']:.4f}, p={link['p_value']:.6f}")

    # Summary
    all_links = []
    for w in results['windows']:
        for link in w['lagged_links']:
            all_links.append(f"{link['feature']}@t-{link['tau']}")
    unique_links = set(all_links)
    stable_links = [l for l in unique_links if all_links.count(l) >= 2]

    results['summary'] = {
        'total_windows': len(results['windows']),
        'all_unique_links': list(unique_links),
        'stable_links_appear_2plus': stable_links,
        'classification': 'alpha' if len(stable_links) > 0 else ('beta' if len(unique_links) > 0 else 'gamma'),
    }

    print(f"\n{'='*60}")
    print(f"RPCMCI SUMMARY")
    print(f"{'='*60}")
    print(f"Total windows: {len(results['windows'])}")
    print(f"Unique causal links: {unique_links}")
    print(f"Stable links (≥2 windows): {stable_links}")
    print(f"Classification: {results['summary']['classification'].upper()}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
