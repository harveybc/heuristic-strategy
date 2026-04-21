#!/usr/bin/env python3
"""
Stage II-0.5: Cross-Asset Causal Comparison — 14 PCMCI+ Runs
=============================================================

Runs PCMCI+ with RobustParCorr on 4 assets × multiple timeframes.
Classifies each as α (strong causal), β (weak causal), or γ (null).

Usage:
    python scripts/cross_asset_causal.py --runs all
    python scripts/cross_asset_causal.py --runs eurusd_4h,eurusd_daily,eurusd_daily_macro,eurusd_weekly
    python scripts/cross_asset_causal.py --runs btcusd_4h,btcusd_daily,btcusd_weekly

Input:  data/processed/*.csv + data/raw/fred/*.csv + data/raw/cftc/*.csv
Output: deliverables/causal_results_II05.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DELIV_DIR = os.path.join(BASE_DIR, 'deliverables')

# Add feature-eng to path
FEATURE_ENG_PATH = os.path.expanduser('~/Documents/GitHub/feature-eng')
if os.path.isdir(FEATURE_ENG_PATH):
    sys.path.insert(0, FEATURE_ENG_PATH)

FEATURE_COLS = [
    'adx', 'di_spread', 'atr_pct', 'atr_ratio',
    'bb_width_pct', 'bb_position', 'rsi', 'roc_12',
    'price_vs_ema50', 'ema_alignment', 'stoch_k', 'macd_hist'
]

# In-sample period: 2005-01-01 to 2019-12-31
IS_START = '2005-01-01'
IS_END = '2019-12-31'

# 14 analysis runs
ALL_RUNS = [
    'eurusd_4h', 'eurusd_daily', 'eurusd_daily_macro', 'eurusd_weekly',
    'usdjpy_4h', 'usdjpy_daily', 'usdjpy_daily_macro', 'usdjpy_weekly',
    'spy_daily', 'spy_daily_macro', 'spy_weekly',
    'btcusd_4h', 'btcusd_daily', 'btcusd_weekly',
]


# ─── Data Loading ──────────────────────────────────────────────────

def load_ohlcv(asset, timeframe, full=False):
    """Load processed OHLCV data for asset/timeframe.

    If full=False (default), filter to IS period.
    If full=True, return all data (for feature warmup on short datasets).
    """
    filename = f'{asset}_{timeframe}.csv'
    filepath = os.path.join(PROC_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")

    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]

    if full:
        print(f"    Loaded {filename}: {len(df):,} bars (full dataset for warmup)")
    else:
        # Filter to in-sample period
        df = df[IS_START:IS_END]
        print(f"    Loaded {filename}: {len(df):,} bars (IS period)")
    return df


def load_macro_features(asset):
    """Load macro features for FX/equity assets."""
    # FRED combined daily
    fred_path = os.path.join(RAW_DIR, 'fred', 'macro_combined_daily_1990_2025.csv')
    if not os.path.exists(fred_path):
        print(f"    WARNING: FRED data not found at {fred_path}")
        return pd.DataFrame()

    fred = pd.read_csv(fred_path, parse_dates=[0], index_col=0)
    fred.columns = [c.lower() for c in fred.columns]

    # CFTC positioning
    if asset == 'eurusd':
        cftc_path = os.path.join(RAW_DIR, 'cftc', 'eur_weekly_2000_2025.csv')
        pos_col = 'eur_net_long'
        macro_cols = ['us_eu_rate_diff', 'dxy_broad', 'vix', 'eur_net_pos']
    elif asset == 'usdjpy':
        cftc_path = os.path.join(RAW_DIR, 'cftc', 'jpy_weekly_2000_2025.csv')
        pos_col = 'jpy_net_long'
        macro_cols = ['us_jp_rate_diff', 'dxy_broad', 'vix', 'jpy_net_pos']
    elif asset == 'spy':
        cftc_path = None
        pos_col = None
        macro_cols = ['us_10y_yield', 'dxy_broad', 'vix']
    else:
        return pd.DataFrame()

    # Build macro df
    macro = pd.DataFrame(index=fred.index)

    # Map column names
    col_map = {
        'us_eu_rate_diff': 'us_eu_rate_diff',
        'us_jp_rate_diff': 'us_jp_rate_diff',
        'us_10y_yield': 'us_10y_yield',
        'dxy_broad': 'dxy_broad',
        'vix': 'vix',
    }

    for target_col in macro_cols:
        if target_col.endswith('_net_pos'):
            continue  # Handle CFTC separately
        source_col = col_map.get(target_col, target_col)
        if source_col in fred.columns:
            macro[target_col] = fred[source_col]
        else:
            print(f"    WARNING: Missing FRED column {source_col}")

    # Add CFTC positioning
    if cftc_path and os.path.exists(cftc_path):
        cftc = pd.read_csv(cftc_path, parse_dates=[0], index_col=0)
        cftc.columns = [c.lower() for c in cftc.columns]
        # Drop duplicate dates (keep last)
        cftc = cftc[~cftc.index.duplicated(keep='last')]
        if pos_col in cftc.columns:
            pos_name = macro_cols[-1]  # eur_net_pos or jpy_net_pos
            macro[pos_name] = cftc[pos_col]
    elif cftc_path:
        print(f"    WARNING: CFTC file not found: {cftc_path}")

    # Forward-fill macro (daily→lower freq alignment)
    macro = macro.ffill()
    print(f"    Macro features loaded: {list(macro.columns)}")
    return macro


# ─── Feature Computation ──────────────────────────────────────────

def compute_technical_features(ohlcv):
    """Compute 12 technical features from OHLCV."""
    from app.regime_detector import compute_regime_features
    features = compute_regime_features(ohlcv)
    # Keep only the 12 standard feature columns
    available = [c for c in FEATURE_COLS if c in features.columns]
    return features[available]


def compute_forward_return(ohlcv, horizon=6):
    """Compute forward log return at given horizon."""
    return np.log(ohlcv['close'].shift(-horizon) / ohlcv['close']) * 100


# ─── PCMCI+ Analysis ──────────────────────────────────────────────

def run_pcmci_plus(data_matrix, var_names, label,
                   tau_max=10, pc_alpha=0.01, alpha_level=0.05,
                   max_samples=5000):
    """Run PCMCI+ and extract causal links to the last variable (target)."""
    import tigramite
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    print(f"\n{'='*70}")
    print(f"PCMCI+ — {label}")
    print(f"{'='*70}")

    # Subsample if too large (PCMCI+ is O(N*p^2) or worse)
    if len(data_matrix) > max_samples:
        idx = np.sort(np.random.RandomState(42).choice(
            len(data_matrix), max_samples, replace=False))
        data_matrix = data_matrix[idx]
        print(f"  Subsampled to {max_samples} from {len(data_matrix)}")

    dataframe = pp.DataFrame(data_matrix,
                             var_names=[n[:14] for n in var_names])

    print(f"  Variables: {len(var_names)} ({len(var_names)-1} features + target)")
    print(f"  Samples: {data_matrix.shape[0]:,}")
    print(f"  τ_max={tau_max}, pc_α={pc_alpha}, α_level={alpha_level}")

    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    print(f"  Running PCMCI+ ...", flush=True)
    t0 = time.time()
    results = pcmci.run_pcmciplus(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        contemp_collider_rule='majority',
        conflict_resolution=True
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    graph = results['graph']
    val_matrix = results['val_matrix']
    p_matrix = results['p_matrix']

    target_idx = len(var_names) - 1

    # Extract links to target
    print(f"\n  Significant causal links → fwd_ret_6 (α={alpha_level}):")
    print(f"  {'Feature':>16s}  {'Lag':>4s}  {'MCI':>10s}  {'p-value':>10s}  Link")
    print(f"  {'─'*16}  {'─'*4}  {'─'*10}  {'─'*10}  ────────")

    causal_links = {}
    lagged_links = {}
    contemp_links = {}

    for i in range(len(var_names) - 1):
        for tau in range(tau_max + 1):
            if graph[i, target_idx, tau] not in ('', '   '):
                link_type = graph[i, target_idx, tau].strip()
                if '-->' in link_type or 'o->' in link_type:
                    pval = float(p_matrix[i, target_idx, tau])
                    val = float(val_matrix[i, target_idx, tau])
                    if pval < alpha_level:
                        feat_name = var_names[i]
                        entry = {
                            'lag': int(tau),
                            'mci_value': round(val, 6),
                            'p_value': round(pval, 8),
                            'link_type': link_type
                        }
                        causal_links.setdefault(feat_name, []).append(entry)
                        if tau == 0:
                            contemp_links[feat_name] = entry
                        else:
                            lagged_links.setdefault(feat_name, []).append(entry)

                        lag_str = f"t-{tau}" if tau > 0 else "t=0"
                        print(f"  {feat_name:>16s}  {lag_str:<4s}  {val:+10.4f}  "
                              f"{pval:10.6f}  {link_type}")

    if not causal_links:
        print(f"  (none)")

    # Autodependency
    auto_links = []
    for tau in range(1, tau_max + 1):
        if graph[target_idx, target_idx, tau] not in ('', '   '):
            link_type = graph[target_idx, target_idx, tau].strip()
            if '-->' in link_type:
                pval = float(p_matrix[target_idx, target_idx, tau])
                val = float(val_matrix[target_idx, target_idx, tau])
                if pval < alpha_level:
                    auto_links.append({
                        'lag': int(tau),
                        'mci_value': round(val, 6),
                        'p_value': round(pval, 8)
                    })

    n_lagged = sum(len(v) for v in lagged_links.values())
    n_contemp = len(contemp_links)

    # Classify: α, β, γ
    classification = classify_result(lagged_links)

    print(f"\n  Summary:")
    print(f"    Lagged links: {n_lagged}")
    print(f"    Contemporaneous links: {n_contemp}")
    print(f"    Auto links: {len(auto_links)}")
    print(f"    Classification: {classification}")

    strongest_lagged = None
    if lagged_links:
        all_lagged = []
        for feat, entries in lagged_links.items():
            for e in entries:
                all_lagged.append({**e, 'feature': feat})
        all_lagged.sort(key=lambda x: abs(x['mci_value']), reverse=True)
        strongest_lagged = all_lagged[0]
        print(f"    Strongest lagged: {strongest_lagged['feature']} "
              f"lag={strongest_lagged['lag']} MCI={strongest_lagged['mci_value']:+.4f}")

    return {
        'label': label,
        'n_variables': len(var_names),
        'n_samples': int(data_matrix.shape[0]),
        'tau_max': tau_max,
        'pc_alpha': pc_alpha,
        'alpha_level': alpha_level,
        'elapsed_seconds': round(elapsed, 1),
        'causal_links': causal_links,
        'lagged_links': lagged_links,
        'contemporaneous_links': contemp_links,
        'autodependency': auto_links,
        'n_lagged_links': n_lagged,
        'n_contemporaneous_links': n_contemp,
        'has_lagged_causal_structure': n_lagged > 0,
        'strongest_lagged_link': strongest_lagged,
        'classification': classification,
    }


def classify_result(lagged_links):
    """Classify as α, β, or γ per §7.4."""
    if not lagged_links:
        return 'γ'

    max_mci = 0
    min_p = 1.0
    for feat, entries in lagged_links.items():
        for e in entries:
            if abs(e['mci_value']) > max_mci:
                max_mci = abs(e['mci_value'])
            if e['p_value'] < min_p:
                min_p = e['p_value']

    # α: ≥1 lagged link with MCI > 0.10 and p < 0.01
    has_alpha = False
    for feat, entries in lagged_links.items():
        for e in entries:
            if abs(e['mci_value']) > 0.10 and e['p_value'] < 0.01:
                has_alpha = True
                break

    if has_alpha:
        return 'α'

    # β: ≥1 lagged link with MCI ∈ [0.05, 0.10] and p < 0.05
    has_beta = False
    for feat, entries in lagged_links.items():
        for e in entries:
            if 0.05 <= abs(e['mci_value']) <= 0.10 and e['p_value'] < 0.05:
                has_beta = True
                break

    if has_beta:
        return 'β'

    return 'γ'


# ─── Run Execution ─────────────────────────────────────────────────

def execute_run(run_name):
    """Execute a single PCMCI+ run."""
    parts = run_name.split('_')

    # Parse asset and timeframe
    if 'macro' in parts:
        asset = parts[0]
        timeframe = parts[1]
        use_macro = True
    else:
        asset = parts[0]
        timeframe = parts[1]
        use_macro = False

    print(f"\n{'#'*70}")
    print(f"# RUN: {run_name}")
    print(f"# Asset={asset}, Timeframe={timeframe}, Macro={use_macro}")
    print(f"{'#'*70}")

    # Load OHLCV
    ohlcv = load_ohlcv(asset, timeframe)
    if len(ohlcv) == 0:
        return {'run': run_name, 'error': 'No data in IS period', 'classification': 'ERROR'}

    # Compute features on IS data first
    tech = compute_technical_features(ohlcv)
    fwd_ret = compute_forward_return(ohlcv, horizon=6)
    tech['fwd_ret_6'] = fwd_ret

    # Check if too few clean samples — if so, retry with full data for warmup
    all_cols_check = [c for c in FEATURE_COLS if c in tech.columns] + ['fwd_ret_6']
    clean_check = tech[all_cols_check].dropna()
    if len(clean_check) < 200:
        print(f"    Only {len(clean_check)} clean IS samples — retrying with full-data warmup")
        ohlcv_full = load_ohlcv(asset, timeframe, full=True)
        tech_full = compute_technical_features(ohlcv_full)
        fwd_ret_full = compute_forward_return(ohlcv_full, horizon=6)
        tech_full['fwd_ret_6'] = fwd_ret_full
        # Filter back to IS period after feature computation
        tech = tech_full[IS_START:IS_END]
        ohlcv = ohlcv_full[IS_START:IS_END]

    # Add macro features if requested
    extra_cols = []
    if use_macro:
        macro = load_macro_features(asset)
        if not macro.empty:
            # Align macro to OHLCV index (forward-fill for date alignment)
            macro_aligned = macro.reindex(ohlcv.index, method='ffill')
            for col in macro_aligned.columns:
                tech[col] = macro_aligned[col]
                extra_cols.append(col)

    # Drop NaN rows
    all_cols = [c for c in FEATURE_COLS if c in tech.columns] + extra_cols + ['fwd_ret_6']
    tech_clean = tech[all_cols].dropna()
    print(f"    Clean samples: {len(tech_clean):,} (from {len(ohlcv):,} bars)")

    if len(tech_clean) < 200:
        return {'run': run_name, 'error': f'Too few samples: {len(tech_clean)}',
                'classification': 'ERROR'}

    # Build data matrix
    data = tech_clean.values.astype(np.float64)

    # Verify finite
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]

    var_names = all_cols
    label = run_name.upper()

    result = run_pcmci_plus(data, var_names, label)
    result['run'] = run_name
    result['asset'] = asset
    result['timeframe'] = timeframe
    result['has_macro'] = use_macro

    return result


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Stage II-0.5 Cross-Asset Causal')
    parser.add_argument('--runs', type=str, default='all',
                        help='Comma-separated run names or "all"')
    parser.add_argument('--output', type=str,
                        default=os.path.join(DELIV_DIR, 'causal_results_II05.json'))
    args = parser.parse_args()

    os.makedirs(DELIV_DIR, exist_ok=True)

    if args.runs == 'all':
        runs = ALL_RUNS
    else:
        runs = [r.strip() for r in args.runs.split(',')]

    print(f"\n{'='*70}")
    print(f"STAGE II-0.5: CROSS-ASSET CAUSAL COMPARISON")
    print(f"{'='*70}")
    print(f"  Runs: {len(runs)}")
    print(f"  IS period: {IS_START} to {IS_END}")
    print(f"  Output: {args.output}")

    # Load existing results if appending
    all_results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
            if 'runs' in existing:
                all_results = existing['runs']
                print(f"  Loaded {len(all_results)} existing results")

    total_start = time.time()

    for i, run_name in enumerate(runs):
        print(f"\n  [{i+1}/{len(runs)}] {run_name}")

        try:
            result = execute_run(run_name)
            all_results[run_name] = result

            # Save checkpoint after each run
            output = {
                'stage': 'II-0.5',
                'timestamp': datetime.now().isoformat(),
                'is_period': f'{IS_START} to {IS_END}',
                'completed_runs': len(all_results),
                'total_runs': len(ALL_RUNS),
                'runs': all_results,
            }
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"  [CHECKPOINT] Saved {len(all_results)} results")

        except Exception as e:
            print(f"  ERROR in {run_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[run_name] = {
                'run': run_name,
                'error': str(e),
                'classification': 'ERROR'
            }

    total_elapsed = time.time() - total_start

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Run':<25s}  {'Class':>5s}  {'Lagged':>6s}  {'Contemp':>7s}  "
          f"{'Strongest':>12s}  {'Time':>6s}")
    print(f"  {'─'*25}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*12}  {'─'*6}")

    for run_name in ALL_RUNS:
        if run_name in all_results:
            r = all_results[run_name]
            if 'error' in r:
                print(f"  {run_name:<25s}  ERROR  {r['error']}")
                continue
            cls = r.get('classification', '?')
            n_lag = r.get('n_lagged_links', 0)
            n_con = r.get('n_contemporaneous_links', 0)
            strongest = r.get('strongest_lagged_link')
            s_str = f"{strongest['feature'][:8]}={strongest['mci_value']:+.3f}" if strongest else "—"
            t_str = f"{r.get('elapsed_seconds', 0):.0f}s"
            print(f"  {run_name:<25s}  {cls:>5s}  {n_lag:>6d}  {n_con:>7d}  "
                  f"{s_str:>12s}  {t_str:>6s}")

    print(f"\n  Total time: {total_elapsed:.0f}s")
    print(f"  Results: {args.output}")


if __name__ == '__main__':
    main()
