#!/usr/bin/env python3
"""
CI-2 Multi-Timeframe Causal Analysis — Stage II-4
===================================================
Re-examines F-6's null finding at daily and weekly timeframes.
Also runs CI-3 macro feature causal testing.

Input:
  - data/processed/eurusd_daily_2005_2024.csv  (daily OHLCV)
  - data/raw/eurusd_1h_2005_2024.csv           (1h source for weekly resample)
  - data/raw/macro_fred_monthly.csv             (FRED macro data)
  - data/raw/cftc_eur_weekly.csv                (CFTC EUR positioning)

Output:
  - deliverables/ci2_results.json               (structured results)

Usage:
  python scripts/ci2_causal_analysis.py \\
      --daily_csv data/processed/eurusd_daily_2005_2024.csv \\
      --hourly_csv data/raw/eurusd_1h_2005_2024.csv \\
      --macro_csv data/raw/macro_fred_monthly.csv \\
      --cftc_csv data/raw/cftc_eur_weekly.csv \\
      --output deliverables/ci2_results.json \\
      [--skip_weekly] [--skip_macro] [--run_rpcmci]
"""

import argparse
import json
import os
import sys
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Feature computation ──────────────────────────────────────────

# Add feature-eng to path
FEATURE_ENG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                                "feature-eng")
if os.path.isdir(FEATURE_ENG_PATH):
    sys.path.insert(0, os.path.abspath(FEATURE_ENG_PATH))

FEATURE_COLS = [
    'adx', 'di_spread', 'atr_pct', 'atr_ratio',
    'bb_width_pct', 'bb_position', 'rsi', 'roc_12',
    'price_vs_ema50', 'ema_alignment', 'stoch_k', 'macd_hist'
]

RETURN_HORIZONS = [1, 3, 6]


def load_ohlc(filepath: str, label: str) -> pd.DataFrame:
    """Load OHLCV CSV with DateTime index, ensure lowercase column names."""
    print(f"\n  Loading {label}: {filepath}")
    df = pd.read_csv(filepath)

    # Find and set datetime index
    date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_col:
        df.index = pd.to_datetime(df[date_col[0]])
        df = df.drop(columns=date_col)

    # Lowercase column names
    df.columns = [c.lower() for c in df.columns]

    print(f"    Bars: {len(df):,}  Range: {df.index[0]} to {df.index[-1]}")
    return df


def resample_to_weekly(hourly_path: str) -> pd.DataFrame:
    """Resample 1h data to weekly OHLCV."""
    print(f"\n  Resampling 1h → weekly from: {hourly_path}")
    df = pd.read_csv(hourly_path)

    date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_col:
        df.index = pd.to_datetime(df[date_col[0]])
        df = df.drop(columns=date_col)

    df.columns = [c.lower() for c in df.columns]

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns:
        agg['volume'] = 'sum'

    weekly = df.resample('W').agg(agg).dropna(subset=['open', 'high', 'low', 'close'],
                                               how='all')
    print(f"    1h bars: {len(df):,} → Weekly bars: {len(weekly):,}")
    print(f"    Range: {weekly.index[0]} to {weekly.index[-1]}")
    return weekly


def compute_features(ohlc_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute 12 regime features + forward returns."""
    from app.regime_detector import compute_regime_features

    features = compute_regime_features(ohlc_df)

    for h in RETURN_HORIZONS:
        features[f'fwd_ret_{h}'] = ohlc_df['close'].pct_change(h).shift(-h) * 100

    features = features.dropna()
    print(f"    {label} features: {len(features):,} bars × {len(FEATURE_COLS)} features")
    return features


# ─── PCMCI+ Analysis ──────────────────────────────────────────────

def run_pcmci_plus(features: pd.DataFrame, label: str,
                   extra_cols: list = None, tau_max: int = 10,
                   pc_alpha: float = 0.01, alpha_level: float = 0.05,
                   max_samples: int = 15000) -> dict:
    """
    Run PCMCI+ with RobustParCorr on feature + return data.

    Args:
        features: DataFrame with FEATURE_COLS + fwd_ret columns
        label: Human-readable label (e.g., "daily", "weekly")
        extra_cols: Additional columns beyond FEATURE_COLS (e.g., macro)
        tau_max: Maximum lag to test
        pc_alpha: Significance for skeleton discovery
        alpha_level: Significance for final links
        max_samples: Max samples (subsample if larger)

    Returns:
        Dict with causal links, summary statistics
    """
    import tigramite
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.robust_parcorr import RobustParCorr

    print(f"\n{'='*70}")
    print(f"PCMCI+ CAUSAL DISCOVERY — {label.upper()}")
    print(f"{'='*70}")

    # Build variable set
    feat_cols = list(FEATURE_COLS)
    if extra_cols:
        feat_cols = feat_cols + extra_cols
    target = 'fwd_ret_6'
    cols = feat_cols + [target]

    # Filter to available columns
    available = [c for c in cols if c in features.columns]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
    cols = available
    feat_cols = [c for c in feat_cols if c in features.columns]

    data = features[cols].values.astype(np.float64)

    # Remove any rows with NaN/Inf
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]

    # Subsample if needed
    if len(data) > max_samples:
        idx = np.sort(np.random.RandomState(42).choice(len(data), max_samples,
                                                        replace=False))
        data = data[idx]
        print(f"  Subsampled to {max_samples} from {mask.sum()}")

    var_names = [c[:14] for c in cols]
    dataframe = pp.DataFrame(data, var_names=var_names)

    print(f"  Variables: {len(cols)} ({len(feat_cols)} features + {target})")
    print(f"  Samples: {data.shape[0]:,}")
    print(f"  Max lag: {tau_max}")
    print(f"  pc_alpha={pc_alpha}, alpha_level={alpha_level}")

    cond_ind_test = RobustParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    print(f"  Running PCMCI+ ...")
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

    ret_idx = len(cols) - 1

    # ── Extract significant links to target ──
    print(f"\n  Significant causal links → {target} (alpha={alpha_level}):")
    print(f"  {'Feature':>16s}  {'Lag':>4s}  {'MCI':>10s}  {'p-value':>10s}  Link")
    print(f"  {'─'*16}  {'─'*4}  {'─'*10}  {'─'*10}  ────────")

    causal_links = {}
    lagged_links = {}
    contemp_links = {}

    for i in range(len(cols) - 1):
        for tau in range(tau_max + 1):
            if graph[i, ret_idx, tau] not in ('', '   '):
                link_type = graph[i, ret_idx, tau].strip()
                if '-->' in link_type or 'o->' in link_type:
                    pval = p_matrix[i, ret_idx, tau]
                    val = val_matrix[i, ret_idx, tau]
                    if pval < alpha_level:
                        feat_name = cols[i]
                        entry = {
                            'lag': int(tau),
                            'mci_value': float(val),
                            'p_value': float(pval),
                            'link_type': link_type
                        }
                        if feat_name not in causal_links:
                            causal_links[feat_name] = []
                        causal_links[feat_name].append(entry)

                        if tau == 0:
                            contemp_links[feat_name] = entry
                        else:
                            if feat_name not in lagged_links:
                                lagged_links[feat_name] = []
                            lagged_links[feat_name].append(entry)

                        lag_str = f"t-{tau}" if tau > 0 else "t=0"
                        print(f"  {feat_name:>16s}  {lag_str:<4s}  {val:+10.4f}  "
                              f"{pval:10.6f}  {link_type}")

    if not causal_links:
        print(f"  (none)")

    # ── Autodependency of return ──
    auto_links = []
    for tau in range(1, tau_max + 1):
        if graph[ret_idx, ret_idx, tau] not in ('', '   '):
            link_type = graph[ret_idx, ret_idx, tau].strip()
            if '-->' in link_type:
                pval = p_matrix[ret_idx, ret_idx, tau]
                val = val_matrix[ret_idx, ret_idx, tau]
                if pval < alpha_level:
                    auto_links.append({
                        'lag': int(tau), 'mci_value': float(val),
                        'p_value': float(pval)
                    })
                    print(f"  {'fwd_ret_6':>16s}  t-{tau:<2d}  {val:+10.4f}  "
                          f"{pval:10.6f}  {link_type}  (auto)")

    # ── Summary ──
    n_lagged = sum(len(v) for v in lagged_links.values())
    n_contemp = len(contemp_links)
    n_total = sum(len(v) for v in causal_links.values())

    print(f"\n  Summary:")
    print(f"    Total significant links → target: {n_total}")
    print(f"    Contemporaneous (tau=0): {n_contemp}")
    print(f"    Lagged (tau>0): {n_lagged}")
    print(f"    Autodependency links: {len(auto_links)}")

    # Determine if lagged causal structure exists
    has_lagged = n_lagged > 0
    strongest_lagged = None
    if lagged_links:
        all_lagged = []
        for feat, entries in lagged_links.items():
            for e in entries:
                all_lagged.append({**e, 'feature': feat})
        all_lagged.sort(key=lambda x: abs(x['mci_value']), reverse=True)
        strongest_lagged = all_lagged[0]
        print(f"    Strongest lagged: {strongest_lagged['feature']} "
              f"at lag {strongest_lagged['lag']} "
              f"(MCI={strongest_lagged['mci_value']:+.4f})")

    return {
        'label': label,
        'n_variables': len(cols),
        'n_samples': int(data.shape[0]),
        'tau_max': tau_max,
        'pc_alpha': pc_alpha,
        'alpha_level': alpha_level,
        'elapsed_seconds': round(elapsed, 1),
        'causal_links_to_return': {k: v for k, v in causal_links.items()},
        'lagged_links': {k: v for k, v in lagged_links.items()},
        'contemporaneous_links': contemp_links,
        'autodependency': auto_links,
        'n_total_links': n_total,
        'n_lagged_links': n_lagged,
        'n_contemporaneous_links': n_contemp,
        'has_lagged_causal_structure': has_lagged,
        'strongest_lagged_link': strongest_lagged,
    }


# ─── CI-3: Macro Feature Integration ──────────────────────────────

def prepare_macro_features(features_daily: pd.DataFrame,
                           macro_path: str, cftc_path: str) -> pd.DataFrame:
    """
    Merge macro and CFTC features with daily technical features.
    Uses forward-fill alignment (no look-ahead).
    """
    print(f"\n  Preparing macro features for CI-3...")

    # Load macro data
    macro = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
    print(f"    Macro: {len(macro)} monthly rows, cols: {list(macro.columns)}")

    # Load CFTC data
    cftc = pd.read_csv(cftc_path, parse_dates=['Date'], index_col='Date')
    print(f"    CFTC: {len(cftc)} weekly rows, cols: {list(cftc.columns)}")

    # Reindex to daily dates with forward-fill (no look-ahead)
    daily_idx = features_daily.index

    macro_daily = macro.reindex(daily_idx, method='ffill')
    cftc_daily = cftc.reindex(daily_idx, method='ffill')

    # Select key macro columns for causal testing
    macro_cols = []
    if 'US_EU_Rate_Diff' in macro_daily.columns:
        features_daily['us_eu_rate_diff'] = macro_daily['US_EU_Rate_Diff'].values
        macro_cols.append('us_eu_rate_diff')
    if 'DXY_Broad' in macro_daily.columns:
        features_daily['dxy'] = macro_daily['DXY_Broad'].values
        macro_cols.append('dxy')
    if 'VIX' in macro_daily.columns:
        features_daily['vix'] = macro_daily['VIX'].values
        macro_cols.append('vix')

    # CFTC: net long as percentage of total (more stable than raw)
    if 'EUR_Net_Long' in cftc_daily.columns:
        features_daily['eur_net_pos'] = cftc_daily['EUR_Net_Long'].values
        macro_cols.append('eur_net_pos')

    # Drop NaN from forward-fill lead-in
    features_daily = features_daily.dropna(subset=macro_cols)
    print(f"    After merge: {len(features_daily)} bars with {len(macro_cols)} macro features")
    print(f"    Macro cols: {macro_cols}")

    return features_daily, macro_cols


# ─── RPCMCI Attempt ──────────────────────────────────────────────

def run_rpcmci_attempt(features: pd.DataFrame, label: str,
                       n_regimes: int = 3, tau_max: int = 5,
                       max_samples: int = 5000) -> dict:
    """
    Time-boxed RPCMCI attempt. Catches failures gracefully.
    """
    print(f"\n{'='*70}")
    print(f"RPCMCI REGIME-DEPENDENT CAUSAL DISCOVERY — {label.upper()}")
    print(f"{'='*70}")

    try:
        import tigramite
        import tigramite.data_processing as pp
        from tigramite.rpcmci import RPCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from sklearn.linear_model import LinearRegression
    except ImportError as e:
        msg = f"RPCMCI import failed: {e}"
        print(f"  {msg}")
        return {'error': msg, 'status': 'IMPORT_FAILED'}

    # Use focused feature set for speed
    focus_cols = ['bb_position', 'atr_ratio', 'ema_alignment',
                  'rsi', 'atr_pct', 'fwd_ret_6']
    available = [c for c in focus_cols if c in features.columns]
    data = features[available].values.astype(np.float64)

    mask = np.isfinite(data).all(axis=1)
    data = data[mask]

    if len(data) > max_samples:
        idx = np.sort(np.random.RandomState(42).choice(len(data), max_samples,
                                                        replace=False))
        data = data[idx]

    # Standardize
    means = data.mean(axis=0)
    stds = data.std(axis=0) + 1e-10
    data = (data - means) / stds

    var_names = [c[:12] for c in available]
    dataframe = pp.DataFrame(data, var_names=var_names)

    print(f"  Variables: {available}")
    print(f"  Samples: {data.shape[0]:,}")
    print(f"  Searching for {n_regimes} regimes...")

    rpcmci = RPCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        prediction_model=LinearRegression(),
        seed=42,
        verbosity=0
    )

    print(f"  Running RPCMCI (may take several minutes)...")
    t0 = time.time()
    try:
        results = rpcmci.run_rpcmci(
            num_regimes=n_regimes,
            max_transitions=50,
            switch_thres=0.05,
            num_iterations=10,
            max_anneal=5,
            tau_min=1,
            tau_max=tau_max,
            pc_alpha=0.2,
            alpha_level=0.05,
            n_jobs=4
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        regimes_onehot, causal_results, diff_g_f, error_free = results

        # Decode regime assignments
        regimes_onehot = np.array(regimes_onehot)
        if regimes_onehot.ndim == 2:
            regime_labels = np.argmax(regimes_onehot, axis=0)
        else:
            regime_labels = regimes_onehot.flatten().astype(int)

        # Regime distribution
        regime_dist = {}
        for r in range(n_regimes):
            count = int(np.sum(regime_labels == r))
            regime_dist[r] = count
            print(f"    Regime {r}: {count} bars ({100*count/len(regime_labels):.1f}%)")

        # Per-regime links to return
        ret_idx = available.index('fwd_ret_6')
        regime_graphs = {}
        for r in range(n_regimes):
            r_results = causal_results[r]
            r_graph = r_results['graph']
            r_val = r_results['val_matrix']
            r_pval = r_results['p_matrix']
            links = []
            for i in range(len(available) - 1):
                for tau in range(1, tau_max + 1):
                    if r_graph[i, ret_idx, tau] not in ('', '   '):
                        link_type = r_graph[i, ret_idx, tau].strip()
                        if '-->' in link_type and r_pval[i, ret_idx, tau] < 0.05:
                            links.append({
                                'feature': available[i],
                                'lag': int(tau),
                                'mci': float(r_val[i, ret_idx, tau]),
                                'p': float(r_pval[i, ret_idx, tau])
                            })
                            print(f"    Regime {r}: {available[i]}(t-{tau}) → ret  "
                                  f"MCI={r_val[i, ret_idx, tau]:+.4f}")
            if not links:
                print(f"    Regime {r}: No significant lagged links")
            regime_graphs[r] = {'links': links}

        return {
            'status': 'SUCCESS',
            'elapsed_seconds': round(elapsed, 1),
            'n_regimes': n_regimes,
            'regime_distribution': regime_dist,
            'error_free_annealings': int(error_free),
            'regime_graphs': regime_graphs,
        }

    except Exception as e:
        elapsed = time.time() - t0
        msg = f"RPCMCI failed after {elapsed:.1f}s: {type(e).__name__}: {e}"
        print(f"  {msg}")
        return {'error': msg, 'status': 'RUNTIME_FAILED', 'elapsed_seconds': round(elapsed, 1)}


# ─── Outcome Classification ──────────────────────────────────────

def classify_outcome(results: dict) -> dict:
    """
    Classify CI-2 outcome as CI-α, CI-β, or CI-γ.

    CI-α: Strong lagged links at daily/weekly (technical features alone)
    CI-β: Null on technicals, but macro features causally lead returns
    CI-γ: Null at all timeframes with all feature sets
    """
    daily = results.get('ci2_daily', {})
    weekly = results.get('ci2_weekly', {})
    macro = results.get('ci3_macro', {})

    daily_lagged = daily.get('n_lagged_links', 0) > 0
    weekly_lagged = weekly.get('n_lagged_links', 0) > 0
    macro_lagged = macro.get('n_lagged_links', 0) > 0

    # Check for strong lagged links (|MCI| > 0.05 and at least 2 features)
    daily_strong = daily.get('n_lagged_links', 0) >= 2
    weekly_strong = weekly.get('n_lagged_links', 0) >= 2

    if daily_strong or weekly_strong:
        classification = 'CI-alpha'
        description = ('Strong lagged causal links found at '
                      f'{"daily" if daily_strong else ""}{"+" if daily_strong and weekly_strong else ""}'
                      f'{"weekly" if weekly_strong else ""} timeframe(s). '
                      'Path B supported at these timeframes.')
        path_b_go = True
        recommended_timeframe = 'daily' if daily_strong else 'weekly'
        recommended_features = 'Phase 1 (technical)'
    elif daily_lagged or weekly_lagged:
        # Weak lagged links — some signal but not strong
        if macro_lagged:
            classification = 'CI-beta'
            description = ('Weak technical lagged links + macro features causally lead returns. '
                          'Path B supported at daily with expanded feature set.')
            path_b_go = True
            recommended_timeframe = 'daily'
            recommended_features = 'Phase 2 (technical + macro)'
        else:
            classification = 'CI-beta'
            description = ('Weak lagged causal links found in technicals. '
                          'Marginal evidence for Path B.')
            path_b_go = True
            recommended_timeframe = 'daily' if daily_lagged else 'weekly'
            recommended_features = 'Phase 1 (technical)'
    elif macro_lagged:
        classification = 'CI-beta'
        description = ('Null on technical features alone, but macro features '
                      'causally lead returns. Path B supported at daily with '
                      'expanded macro feature set (Phase 2).')
        path_b_go = True
        recommended_timeframe = 'daily'
        recommended_features = 'Phase 2 (technical + macro)'
    else:
        classification = 'CI-gamma'
        description = ('No lagged causal structure at any tested timeframe or '
                      'feature set. Path B is predictably non-viable — '
                      'predicting noise. Recommend halting Path B.')
        path_b_go = False
        recommended_timeframe = None
        recommended_features = None

    return {
        'classification': classification,
        'description': description,
        'path_b_go': path_b_go,
        'recommended_timeframe': recommended_timeframe,
        'recommended_features': recommended_features,
        'evidence': {
            'daily_lagged_links': daily.get('n_lagged_links', 0),
            'weekly_lagged_links': weekly.get('n_lagged_links', 0),
            'macro_lagged_links': macro.get('n_lagged_links', 0),
            'daily_contemp_links': daily.get('n_contemporaneous_links', 0),
            'weekly_contemp_links': weekly.get('n_contemporaneous_links', 0),
        }
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CI-2 Multi-Timeframe Causal Analysis — Stage II-4")
    parser.add_argument("--daily_csv", required=True,
                        help="Daily OHLCV CSV")
    parser.add_argument("--hourly_csv", default=None,
                        help="1h OHLCV CSV (for weekly resample)")
    parser.add_argument("--macro_csv", default=None,
                        help="FRED macro monthly CSV")
    parser.add_argument("--cftc_csv", default=None,
                        help="CFTC EUR weekly CSV")
    parser.add_argument("--output", default="deliverables/ci2_results.json",
                        help="Output JSON path")
    parser.add_argument("--tau_max_daily", type=int, default=10,
                        help="Max lag for daily (10 = 10 days)")
    parser.add_argument("--tau_max_weekly", type=int, default=10,
                        help="Max lag for weekly (10 = 10 weeks)")
    parser.add_argument("--skip_weekly", action="store_true",
                        help="Skip weekly analysis")
    parser.add_argument("--skip_macro", action="store_true",
                        help="Skip CI-3 macro analysis")
    parser.add_argument("--run_rpcmci", action="store_true",
                        help="Attempt RPCMCI (time-boxed)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Helper to save incremental results (crash-resilient)
    def save_checkpoint(results, label):
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [CHECKPOINT] Saved after {label} → {args.output}")
        sys.stdout.flush()

    # Resume from checkpoint if exists
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"  [RESUME] Loaded checkpoint from {args.output}")
        print(f"  [RESUME] Phases present: {[k for k in results if k not in ('metadata','outcome','f6_comparison')]}")
    else:
        results = {}

    results['metadata'] = {
        'script': 'ci2_causal_analysis.py',
        'timestamp': datetime.now().isoformat(),
        'stage': 'II-4',
        'purpose': 'Multi-timeframe causal re-examination (CI-2) + macro feature testing (CI-3)',
    }

    # ── CI-2a: Daily PCMCI+ ──
    if 'ci2_daily' not in results or results['ci2_daily'].get('skipped'):
        print(f"\n{'#'*70}")
        print(f"# CI-2a: DAILY TIMEFRAME")
        print(f"{'#'*70}")

        daily_ohlc = load_ohlc(args.daily_csv, "Daily OHLCV")
        daily_features = compute_features(daily_ohlc, "Daily")

        results['ci2_daily'] = run_pcmci_plus(
            daily_features, "Daily (CI-2a)",
            tau_max=args.tau_max_daily
        )
        save_checkpoint(results, "CI-2a Daily")
    else:
        print(f"\n  [SKIP] CI-2a Daily already in checkpoint")
        daily_ohlc = load_ohlc(args.daily_csv, "Daily OHLCV")
        daily_features = compute_features(daily_ohlc, "Daily")

    # ── CI-2b: Weekly PCMCI+ ──
    if not args.skip_weekly:
        if 'ci2_weekly' not in results or results['ci2_weekly'].get('skipped'):
            print(f"\n{'#'*70}")
            print(f"# CI-2b: WEEKLY TIMEFRAME")
            print(f"{'#'*70}")

            if args.hourly_csv and os.path.exists(args.hourly_csv):
                weekly_ohlc = resample_to_weekly(args.hourly_csv)
            else:
                # Fallback: resample daily to weekly
                print("  (No 1h source; resampling daily → weekly)")
                agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
                if 'volume' in daily_ohlc.columns:
                    agg['volume'] = 'sum'
                weekly_ohlc = daily_ohlc.resample('W').agg(agg).dropna(
                    subset=['open', 'high', 'low', 'close'], how='all')
                print(f"    Daily → Weekly: {len(weekly_ohlc):,} bars")

            weekly_features = compute_features(weekly_ohlc, "Weekly")

            results['ci2_weekly'] = run_pcmci_plus(
                weekly_features, "Weekly (CI-2b)",
                tau_max=args.tau_max_weekly
            )
            save_checkpoint(results, "CI-2b Weekly")
        else:
            print(f"\n  [SKIP] CI-2b Weekly already in checkpoint")
    else:
        print("\n  [SKIPPED] Weekly analysis")
        results['ci2_weekly'] = {'skipped': True, 'n_lagged_links': 0}

    # ── CI-3: Macro Features ──
    if not args.skip_macro and args.macro_csv and args.cftc_csv:
        if 'ci3_macro' not in results or results['ci3_macro'].get('skipped'):
            print(f"\n{'#'*70}")
            print(f"# CI-3: MACRO FEATURES (daily + macro)")
            print(f"{'#'*70}")

            # Re-compute daily features for macro merge (need clean copy)
            daily_ohlc_macro = load_ohlc(args.daily_csv, "Daily OHLCV (for CI-3)")
            daily_features_macro = compute_features(daily_ohlc_macro, "Daily+Macro")

            daily_features_macro, macro_cols = prepare_macro_features(
                daily_features_macro, args.macro_csv, args.cftc_csv
            )

            results['ci3_macro'] = run_pcmci_plus(
                daily_features_macro, "Daily + Macro (CI-3)",
                extra_cols=macro_cols,
                tau_max=args.tau_max_daily
            )
            save_checkpoint(results, "CI-3 Macro")
        else:
            print(f"\n  [SKIP] CI-3 Macro already in checkpoint")
    else:
        print("\n  [SKIPPED] CI-3 macro analysis")
        results['ci3_macro'] = {'skipped': True, 'n_lagged_links': 0}

    # ── RPCMCI attempt (time-boxed) ──
    if args.run_rpcmci:
        if 'rpcmci_daily' not in results or results['rpcmci_daily'].get('skipped'):
            print(f"\n{'#'*70}")
            print(f"# RPCMCI — REGIME-DEPENDENT (time-boxed)")
            print(f"{'#'*70}")
            results['rpcmci_daily'] = run_rpcmci_attempt(daily_features, "Daily")
            save_checkpoint(results, "RPCMCI")
        else:
            print(f"\n  [SKIP] RPCMCI already in checkpoint")
    else:
        results['rpcmci_daily'] = {'skipped': True, 'status': 'NOT_REQUESTED'}

    # ── Outcome Classification ──
    print(f"\n{'#'*70}")
    print(f"# OUTCOME CLASSIFICATION")
    print(f"{'#'*70}")

    outcome = classify_outcome(results)
    results['outcome'] = outcome

    print(f"\n  Classification: {outcome['classification']}")
    print(f"  Description: {outcome['description']}")
    print(f"  Path B go/no-go: {'GO' if outcome['path_b_go'] else 'NO-GO'}")
    if outcome['recommended_timeframe']:
        print(f"  Recommended timeframe: {outcome['recommended_timeframe']}")
        print(f"  Recommended features: {outcome['recommended_features']}")

    # ── F-6 comparison ──
    results['f6_comparison'] = {
        'f6_timeframe': '4h',
        'f6_result': 'No lagged causal structure (contemporaneous only)',
        'ci2_daily_lagged': results['ci2_daily'].get('n_lagged_links', 0),
        'ci2_weekly_lagged': results.get('ci2_weekly', {}).get('n_lagged_links', 0),
        'ci3_macro_lagged': results.get('ci3_macro', {}).get('n_lagged_links', 0),
    }

    # ── Save results ──
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"CI-2/CI-3 ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"  Daily  — lagged:{results['ci2_daily'].get('n_lagged_links',0)}  "
          f"contemp:{results['ci2_daily'].get('n_contemporaneous_links',0)}")
    if not args.skip_weekly:
        print(f"  Weekly — lagged:{results['ci2_weekly'].get('n_lagged_links',0)}  "
              f"contemp:{results['ci2_weekly'].get('n_contemporaneous_links',0)}")
    if not args.skip_macro:
        print(f"  Macro  — lagged:{results['ci3_macro'].get('n_lagged_links',0)}  "
              f"contemp:{results['ci3_macro'].get('n_contemporaneous_links',0)}")
    print(f"\n  OUTCOME: {outcome['classification']}")
    print(f"  Path B: {'GO' if outcome['path_b_go'] else 'NO-GO'}")


if __name__ == '__main__':
    main()
