#!/usr/bin/env python3
"""Data Validation Gate — 6 tests per asset × timeframe.

Tests per §6 of Project2_part2redux.md:
  1. Bar count realistic (±10% of expected)
  2. Weekend gap (FX + equity: no weekend bars)
  3. Fat-tail kurtosis (>4 for real data, GBM≈3)
  4. Volatility clustering (ACF(1) of r² > 0.05)
  5. Tiny nonzero return autocorrelation (|ACF(1)| > 0.005)
  6. Combined no-GBM fingerprint (Ljung-Box, Jarque-Bera, Runs test)

Usage:
    python scripts/validate_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')


# Expected bar counts (±10% tolerance)
# FX daily includes Sunday session bars (~1084 extra over Mon-Fri)
EXPECTED_COUNTS = {
    'eurusd_1h': 130000,
    'eurusd_4h': 33000,
    'eurusd_daily': 6500,
    'eurusd_weekly': 1100,
    'usdjpy_1h': 130000,
    'usdjpy_4h': 33000,
    'usdjpy_daily': 6500,
    'usdjpy_weekly': 1100,
    'spy_daily': 8000,
    'spy_weekly': 1700,
    'btcusd_4h': 18000,
    'btcusd_daily': 3000,
    'btcusd_weekly': 450,
}

# Assets exempt from weekend gap test
CRYPTO_ASSETS = {'btcusd'}


def load_processed(asset_tf):
    """Load a processed CSV."""
    path = os.path.join(PROC_DIR, f'{asset_tf}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.capitalize() for c in df.columns]
    return df


def compute_returns(df):
    """Compute log returns from Close prices."""
    return np.log(df['Close'] / df['Close'].shift(1)).dropna()


def acf_lag1(series):
    """Compute ACF at lag 1."""
    n = len(series)
    mean = series.mean()
    c0 = ((series - mean) ** 2).sum()
    c1 = ((series.iloc[:-1].values - mean) * (series.iloc[1:].values - mean)).sum()
    return c1 / c0 if c0 > 0 else 0


def runs_test(series):
    """Non-parametric runs test on sign of returns."""
    signs = np.sign(series.values)
    signs = signs[signs != 0]  # Remove zeros
    n = len(signs)
    if n < 20:
        return 1.0  # Not enough data
    
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            runs += 1
    
    # Expected runs and variance under null (use float to avoid int overflow)
    n1, n2 = float(n_pos), float(n_neg)
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    
    if var_runs <= 0:
        return 1.0
    
    z = (runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p_value


def ljung_box_test(series, lags=10):
    """Ljung-Box Q test."""
    n = len(series)
    q_stat = 0
    mean = series.mean()
    c0 = ((series - mean) ** 2).sum() / n
    
    for k in range(1, lags + 1):
        ck = ((series.iloc[:-k].values - mean) * (series.iloc[k:].values - mean)).sum() / n
        rho_k = ck / c0 if c0 > 0 else 0
        q_stat += (rho_k ** 2) / (n - k)
    
    q_stat *= n * (n + 2)
    p_value = 1 - stats.chi2.cdf(q_stat, lags)
    
    return p_value


def test_bar_count(df, asset_tf, expected):
    """Test 1: Bar count within ±10% of expected."""
    actual = len(df)
    lower = expected * 0.9
    upper = expected * 1.1
    passed = lower <= actual <= upper
    return {
        'test': 'bar_count',
        'expected': expected,
        'actual': actual,
        'tolerance': '±10%',
        'range': f'[{lower:.0f}, {upper:.0f}]',
        'passed': passed,
    }


# FX assets have legitimate Sunday session bars (market opens Sun ~22:00 UTC)
FX_ASSETS = {'eurusd', 'usdjpy'}


def test_weekend_gap(df, asset_tf):
    """Test 2: No Saturday bars for FX/equity (Sunday FX is legitimate)."""
    parts = asset_tf.rsplit('_', 1)
    asset = parts[0]
    timeframe = parts[1] if len(parts) > 1 else ''
    
    if asset in CRYPTO_ASSETS:
        return {'test': 'weekend_gap', 'passed': True, 'note': 'Crypto exempt (24/7)'}
    
    # Weekly bars are anchored to a specific day — exempt from weekend test
    if timeframe == 'weekly':
        return {'test': 'weekend_gap', 'passed': True, 'note': 'Weekly exempt (anchor day)'}
    
    if asset in FX_ASSETS:
        # FX: only check Saturday bars (Sunday session is legitimate)
        saturday_bars = df[df.index.dayofweek == 5]
        n_saturday = len(saturday_bars)
        passed = n_saturday == 0
        sunday_bars = df[df.index.dayofweek == 6]
        return {
            'test': 'weekend_gap',
            'saturday_bars': n_saturday,
            'sunday_bars': len(sunday_bars),
            'note': 'FX: Saturday=closed, Sunday=legitimate session',
            'passed': passed,
        }
    else:
        # Equity: no weekend bars at all
        weekend_bars = df[df.index.dayofweek >= 5]
        n_weekend = len(weekend_bars)
        passed = n_weekend == 0
        return {
            'test': 'weekend_gap',
            'weekend_bars': n_weekend,
            'passed': passed,
        }


def test_kurtosis(returns):
    """Test 3: Fat-tail kurtosis > 4."""
    kurt = stats.kurtosis(returns, fisher=False)  # Excess=False gives raw kurtosis
    passed = kurt > 4
    return {
        'test': 'fat_tail_kurtosis',
        'kurtosis': round(float(kurt), 2),
        'threshold': '>4 (GBM≈3)',
        'passed': passed,
    }


def test_vol_clustering(returns):
    """Test 4: ACF(1) of squared returns > 0.05."""
    r_squared = returns ** 2
    acf1 = acf_lag1(r_squared)
    passed = acf1 > 0.05
    return {
        'test': 'volatility_clustering',
        'acf1_r_squared': round(float(acf1), 4),
        'threshold': '>0.05',
        'passed': passed,
    }


def test_return_autocorrelation(returns):
    """Test 5: |ACF(1)| of raw returns > 0.005."""
    acf1 = acf_lag1(returns)
    passed = abs(acf1) > 0.005
    return {
        'test': 'return_autocorrelation',
        'acf1_returns': round(float(acf1), 6),
        'threshold': '|ACF|>0.005',
        'passed': passed,
    }


def test_no_gbm_fingerprint(returns):
    """Test 6: Combined no-GBM fingerprint (≥2 of 3 tests reject null)."""
    # Ljung-Box on r²
    r_squared = returns ** 2
    lb_p = ljung_box_test(r_squared)
    lb_reject = lb_p < 0.01
    
    # Jarque-Bera on returns
    jb_stat, jb_p = stats.jarque_bera(returns)
    jb_reject = jb_p < 0.01
    
    # Runs test on sign(r)
    runs_p = runs_test(returns)
    runs_reject = runs_p < 0.05
    
    n_reject = sum([lb_reject, jb_reject, runs_reject])
    passed = n_reject >= 2
    
    return {
        'test': 'no_gbm_fingerprint',
        'ljung_box_r2': {'p': round(float(lb_p), 6), 'reject': lb_reject},
        'jarque_bera': {'p': round(float(jb_p), 6), 'reject': jb_reject},
        'runs_test': {'p': round(float(runs_p), 6), 'reject': runs_reject},
        'rejections': n_reject,
        'threshold': '≥2 of 3',
        'passed': passed,
    }


def run_all_tests(asset_tf):
    """Run all 6 tests on a single asset × timeframe."""
    df = load_processed(asset_tf)
    if df is None:
        return {'asset_tf': asset_tf, 'error': 'File not found', 'all_passed': False}
    
    returns = compute_returns(df)
    expected = EXPECTED_COUNTS.get(asset_tf, len(df))
    
    results = {
        'asset_tf': asset_tf,
        'bars': len(df),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'tests': [],
    }
    
    results['tests'].append(test_bar_count(df, asset_tf, expected))
    results['tests'].append(test_weekend_gap(df, asset_tf))
    results['tests'].append(test_kurtosis(returns))
    results['tests'].append(test_vol_clustering(returns))
    results['tests'].append(test_return_autocorrelation(returns))
    results['tests'].append(test_no_gbm_fingerprint(returns))
    
    results['all_passed'] = all(t['passed'] for t in results['tests'])
    
    return results


def main():
    print(f"\n{'='*70}")
    print("DATA VALIDATION GATE (Stage II-0b)")
    print(f"{'='*70}")
    
    # Discover all processed files
    asset_tfs = sorted(EXPECTED_COUNTS.keys())
    
    all_results = []
    any_failure = False
    
    for asset_tf in asset_tfs:
        print(f"\n  {asset_tf.upper()}")
        print(f"  {'─'*50}")
        
        result = run_all_tests(asset_tf)
        all_results.append(result)
        
        if 'error' in result:
            print(f"    ERROR: {result['error']}")
            any_failure = True
            continue
        
        for test in result['tests']:
            status = "PASS" if test['passed'] else "FAIL"
            print(f"    [{status}] {test['test']}", end='')
            
            if 'kurtosis' in test:
                print(f" = {test['kurtosis']}", end='')
            elif 'acf1_r_squared' in test:
                print(f" = {test['acf1_r_squared']}", end='')
            elif 'acf1_returns' in test:
                print(f" = {test['acf1_returns']}", end='')
            elif 'rejections' in test:
                print(f" = {test['rejections']}/3", end='')
            elif 'actual' in test:
                print(f" = {test['actual']} (expected ~{test['expected']})", end='')
            elif 'weekend_bars' in test:
                print(f" = {test.get('weekend_bars', 'N/A')} weekend bars", end='')
            
            print()
        
        status = "ALL PASS" if result['all_passed'] else "SOME FAIL"
        print(f"    → {status}")
        
        if not result['all_passed']:
            any_failure = True
    
    # Save results
    output_path = os.path.join(BASE_DIR, 'deliverables', 'validation_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    if any_failure:
        print("VALIDATION GATE: FAIL — see results above")
        print(f"Results saved to: {output_path}")
        sys.exit(1)
    else:
        print("VALIDATION GATE: ALL PASS")
        print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
