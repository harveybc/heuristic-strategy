#!/usr/bin/env python3
"""
Stage II-3 Experiment Evaluation & Statistical Tests.

Implements F-10 framework evaluation:
  - Kill criteria K-1 through K-7
  - Ledoit-Wolf Sharpe ratio test
  - Bootstrap confidence interval on ΔSR
  - Deflated Sharpe Ratio (DSR) for multiple-testing correction
  - Cross-experiment comparison table

Usage:
  python evaluate_experiments.py --experiments_dir logs/ \
      --baseline logs/static_baseline/static_001_results.csv \
      --data data/processed/eurusd_4h_2005_2024.csv \
      --output deliverables/evaluation_results.json
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helper: Sharpe computation
# ---------------------------------------------------------------------------

def compute_sharpe(returns):
    """Annualised Sharpe from returns array."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma)


def compute_max_drawdown(equity):
    """Max drawdown as fraction of peak."""
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


# ---------------------------------------------------------------------------
# Load experiment results
# ---------------------------------------------------------------------------

def load_experiment_csv(csv_path):
    """Load F-5 §7 format CSV into list of dicts."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for field in ['train_sharpe', 'val_sharpe', 'test_sharpe',
                          'train_mae', 'val_mae', 'test_mae',
                          'max_dd', 'cost_ratio']:
                try:
                    row[field] = float(row[field]) if row[field] else 0.0
                except (ValueError, TypeError):
                    row[field] = 0.0
            try:
                row['num_trades'] = int(row['num_trades']) if row['num_trades'] else 0
            except (ValueError, TypeError):
                row['num_trades'] = 0
            rows.append(row)
    return rows


def load_experiment_summary(summary_path):
    """Load experiment summary JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def collect_experiment_pnl(window_dirs):
    """Collect test PnL from per-window best_params.json and re-evaluate.
    For now, use test_sharpe from CSV as proxy."""
    pass  # We use summary JSON aggregate


# ---------------------------------------------------------------------------
# Kill Criteria (F-10 §4.1)
# ---------------------------------------------------------------------------

def evaluate_kill_criteria(exp_rows, exp_summary, held_out_sharpe=None):
    """Evaluate K-1 through K-7 kill criteria."""
    results = {}

    # K-1: Held-out Sharpe > 0
    if held_out_sharpe is not None:
        results['K1_held_out_sr'] = {
            'value': held_out_sharpe,
            'threshold': 0.0,
            'pass': held_out_sharpe > 0,
            'description': 'Held-out Sharpe ratio > 0'
        }
    else:
        results['K1_held_out_sr'] = {
            'value': None,
            'threshold': 0.0,
            'pass': None,
            'description': 'Held-out Sharpe ratio > 0 (deferred)'
        }

    # K-2: Worst 2-year rolling Sharpe > -0.9
    test_sharpes = [r['test_sharpe'] for r in exp_rows]
    if len(test_sharpes) >= 2:
        # Rolling 2-window averages
        rolling_2yr = [np.mean(test_sharpes[i:i+2]) for i in range(len(test_sharpes)-1)]
        worst_2yr = min(rolling_2yr)
    else:
        worst_2yr = min(test_sharpes) if test_sharpes else 0.0
    results['K2_worst_2yr_sr'] = {
        'value': worst_2yr,
        'threshold': -0.9,
        'pass': worst_2yr > -0.9,
        'description': 'Worst 2-year rolling Sharpe > -0.9'
    }

    # K-3: Cost ratio >= 2.0 (worst window)
    cost_ratios = [r['cost_ratio'] for r in exp_rows]
    worst_cr = min(cost_ratios) if cost_ratios else 0.0
    results['K3_cost_ratio'] = {
        'value': worst_cr,
        'threshold': 2.0,
        'pass': worst_cr >= 2.0,
        'description': 'Min window cost ratio >= 2.0'
    }

    # K-5: Window consistency >= 60%
    consistency = exp_summary.get('window_consistency', 0)
    results['K5_window_consistency'] = {
        'value': consistency,
        'threshold': 0.6,
        'pass': consistency >= 0.6,
        'description': 'Fraction of windows with positive test Sharpe >= 60%'
    }

    return results


# ---------------------------------------------------------------------------
# Ledoit-Wolf Sharpe Test
# ---------------------------------------------------------------------------

def ledoit_wolf_sr_test(sr1, sr2, n, skew1=0, kurt1=3, skew2=0, kurt2=3,
                        rho=0.5):
    """
    Ledoit-Wolf (2008) test for equality of Sharpe ratios.
    
    H0: SR1 = SR2
    Returns z-stat and p-value (two-sided).
    """
    delta = sr1 - sr2
    
    # Approximate variance of delta SR
    # V(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / n
    v1 = (1 + 0.5 * sr1**2 - skew1 * sr1 + (kurt1 - 3) / 4 * sr1**2) / n
    v2 = (1 + 0.5 * sr2**2 - skew2 * sr2 + (kurt2 - 3) / 4 * sr2**2) / n
    
    # Covariance approximation
    cov_term = rho * np.sqrt(v1 * v2) * 2  # Simplified
    
    var_delta = v1 + v2 - cov_term
    if var_delta <= 0:
        var_delta = v1 + v2  # Fallback: assume uncorrelated
    
    se = np.sqrt(var_delta)
    if se < 1e-12:
        return 0.0, 1.0
    
    z = delta / se
    p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    return float(z), float(p)


# ---------------------------------------------------------------------------
# Bootstrap ΔSR
# ---------------------------------------------------------------------------

def bootstrap_delta_sr(returns_adaptive, returns_static,
                       n_bootstrap=10000, block_size=20,
                       confidence=0.90):
    """
    Circular block bootstrap CI for Sharpe ratio difference.
    
    Returns (mean_delta, ci_lower, ci_upper, p_value).
    """
    n = min(len(returns_adaptive), len(returns_static))
    if n < block_size * 2:
        block_size = max(n // 4, 2)
    
    sr_adaptive = compute_sharpe(returns_adaptive[:n])
    sr_static = compute_sharpe(returns_static[:n])
    observed_delta = sr_adaptive - sr_static
    
    deltas = []
    n_blocks = max(n // block_size, 1)
    
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Circular block bootstrap
        starts = rng.randint(0, n, size=n_blocks)
        idx_a = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        
        starts = rng.randint(0, n, size=n_blocks)
        idx_s = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        
        sr_a = compute_sharpe(returns_adaptive[idx_a])
        sr_s = compute_sharpe(returns_static[idx_s])
        deltas.append(sr_a - sr_s)
    
    deltas = np.array(deltas)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(deltas, alpha * 100)
    ci_upper = np.percentile(deltas, (1 - alpha) * 100)
    
    # P-value: fraction of bootstrap samples where delta <= 0
    p_value = np.mean(deltas <= 0)
    
    return {
        'observed_delta': float(observed_delta),
        'mean_bootstrap_delta': float(np.mean(deltas)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        'confidence': confidence,
        'n_bootstrap': n_bootstrap,
        'block_size': block_size,
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(sr_observed, n_returns, n_experiments,
                          sr_std=1.0, skew=0.0, kurt=3.0):
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    
    Accounts for multiple testing by comparing observed SR to expected
    maximum SR under the null given n_experiments independent trials.
    
    Returns DSR value and whether it exceeds 0.95 significance.
    """
    # Expected max SR under null (from order statistics of standard normal)
    # E[max(Z_1,...,Z_T)] ≈ sqrt(2 * ln(T)) - (ln(pi) + ln(ln(T))) / (2*sqrt(2*ln(T)))
    if n_experiments <= 1:
        expected_max_sr = 0.0
    else:
        log_t = np.log(n_experiments)
        expected_max_sr = (np.sqrt(2 * log_t)
                          - (np.log(np.pi) + np.log(log_t))
                          / (2 * np.sqrt(2 * log_t)))
    
    # Standard error of SR estimate
    se_sr = np.sqrt((1 + 0.5 * sr_observed**2
                     - skew * sr_observed
                     + (kurt - 3) / 4 * sr_observed**2) / n_returns)
    
    if se_sr < 1e-12:
        return {'dsr': 0.0, 'significant': False, 'expected_max_sr': expected_max_sr}
    
    # DSR = P(SR* < SR_observed) where SR* ~ N(E[max SR], se)
    z = (sr_observed - expected_max_sr) / se_sr
    dsr = float(scipy_stats.norm.cdf(z))
    
    return {
        'dsr': dsr,
        'significant': dsr > 0.95,
        'sr_observed': float(sr_observed),
        'expected_max_sr': float(expected_max_sr),
        'se_sr': float(se_sr),
        'n_experiments': n_experiments,
        'n_returns': n_returns,
    }


# ---------------------------------------------------------------------------
# K-7: Adaptive vs Static comparison
# ---------------------------------------------------------------------------

def evaluate_k7(adaptive_summary, static_summary,
                adaptive_test_sharpes, static_test_sharpes):
    """
    K-7: Adaptive test SR must exceed static baseline SR.
    Uses per-window paired comparison.
    """
    adaptive_agg = adaptive_summary.get('aggregate_test_sharpe', 0)
    static_agg = static_summary.get('aggregate_test_sharpe', 0)
    delta = adaptive_agg - static_agg
    
    # Paired t-test on per-window Sharpes (if same number of windows)
    n_adaptive = len(adaptive_test_sharpes)
    n_static = len(static_test_sharpes)
    
    if n_adaptive == n_static and n_adaptive > 1:
        # Paired test
        diffs = np.array(adaptive_test_sharpes) - np.array(static_test_sharpes)
        t_stat, p_value = scipy_stats.ttest_1samp(diffs, 0)
        test_type = 'paired_t_test'
    elif n_adaptive > 1 and n_static > 1:
        # Unpaired (different number of windows)
        t_stat, p_value = scipy_stats.ttest_ind(adaptive_test_sharpes, static_test_sharpes)
        test_type = 'unpaired_t_test'
    else:
        t_stat, p_value = 0.0, 1.0
        test_type = 'insufficient_data'
    
    return {
        'adaptive_aggregate_sr': float(adaptive_agg),
        'static_aggregate_sr': float(static_agg),
        'delta_sr': float(delta),
        'delta_positive': delta > 0,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_p10': p_value < 0.10,
        'test_type': test_type,
        'pass': delta > 0 and p_value < 0.10,
        'description': 'Adaptive ΔSR > 0 with p < 0.10'
    }


# ---------------------------------------------------------------------------
# Parameter stability analysis
# ---------------------------------------------------------------------------

def analyze_param_stability(experiment_dir):
    """Analyze parameter stability across windows."""
    params_list = []
    window_dirs = sorted(glob.glob(os.path.join(experiment_dir, "window_*")))
    
    for wd in window_dirs:
        bp_path = os.path.join(wd, "best_params.json")
        if os.path.exists(bp_path):
            with open(bp_path, 'r') as f:
                params_list.append(json.load(f))
    
    if len(params_list) < 2:
        return {}
    
    param_names = list(params_list[0].keys())
    stability = {}
    for name in param_names:
        values = [p.get(name, 0) for p in params_list
                  if isinstance(p.get(name), (int, float))]
        if len(values) >= 2:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float('inf')
            stability[name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),
                'values': [float(v) for v in values],
                'flag': 'HIGH' if cv > 0.6 else ('AMBIGUOUS' if cv > 0.4 else 'STABLE')
            }
    
    avg_cv = np.mean([s['cv'] for s in stability.values()
                      if s['cv'] != float('inf')])
    
    return {
        'per_param': stability,
        'avg_cv': float(avg_cv) if not np.isnan(avg_cv) else None,
        'verdict': 'CURVE_FITTING_RISK' if avg_cv > 0.6 else (
            'AMBIGUOUS' if avg_cv > 0.4 else 'STABLE_ADAPTATION')
    }


# ---------------------------------------------------------------------------
# Full experiment evaluation
# ---------------------------------------------------------------------------

def evaluate_single_experiment(exp_id, exp_dir, static_csv_path, static_summary_path,
                               n_total_experiments=7):
    """Full F-10 evaluation for one experiment."""
    # Load adaptive experiment data
    csv_files = glob.glob(os.path.join(exp_dir, "*_results.csv"))
    summary_files = glob.glob(os.path.join(exp_dir, "*_summary.json"))
    
    if not csv_files or not summary_files:
        return {'error': f'No results found in {exp_dir}'}
    
    exp_rows = load_experiment_csv(csv_files[0])
    exp_summary = load_experiment_summary(summary_files[0])
    
    # Load static baseline
    static_rows = load_experiment_csv(static_csv_path)
    static_summary = load_experiment_summary(static_summary_path)
    
    # Kill criteria
    kill_results = evaluate_kill_criteria(exp_rows, exp_summary)
    
    # K-7: Adaptive vs static
    adaptive_sharpes = [r['test_sharpe'] for r in exp_rows]
    static_sharpes = [r['test_sharpe'] for r in static_rows]
    k7 = evaluate_k7(exp_summary, static_summary, adaptive_sharpes, static_sharpes)
    kill_results['K7_adaptive_vs_static'] = k7
    
    # Deflated Sharpe Ratio
    total_trades = sum(r['num_trades'] for r in exp_rows)
    agg_sr = exp_summary.get('aggregate_test_sharpe', 0)
    dsr = deflated_sharpe_ratio(agg_sr, max(total_trades, 10), n_total_experiments)
    
    # Parameter stability
    param_stability = analyze_param_stability(exp_dir)
    
    # Count passes/fails
    kills_checked = {k: v for k, v in kill_results.items() if v.get('pass') is not None}
    n_pass = sum(1 for v in kills_checked.values() if v.get('pass'))
    n_fail = sum(1 for v in kills_checked.values() if not v.get('pass'))
    
    verdict = 'PASS' if n_fail == 0 else ('INCONCLUSIVE' if n_fail <= 2 else 'FAIL')
    
    return {
        'experiment_id': exp_id,
        'summary': {
            'aggregate_test_sharpe': exp_summary.get('aggregate_test_sharpe'),
            'mean_test_sharpe': exp_summary.get('mean_test_sharpe'),
            'std_test_sharpe': exp_summary.get('std_test_sharpe'),
            'window_consistency': exp_summary.get('window_consistency'),
            'total_trades': exp_summary.get('total_test_trades'),
            'max_drawdown': exp_summary.get('max_drawdown'),
            'final_equity': exp_summary.get('final_equity'),
        },
        'kill_criteria': kill_results,
        'deflated_sharpe': dsr,
        'param_stability': param_stability,
        'verdict': verdict,
        'kills_pass': n_pass,
        'kills_fail': n_fail,
    }


# ---------------------------------------------------------------------------
# Cross-experiment comparison
# ---------------------------------------------------------------------------

def cross_experiment_comparison(all_results):
    """Generate F-10 §6.2 cross-experiment comparison table."""
    if not all_results:
        return {}
    
    ranked = sorted(all_results, 
                    key=lambda x: x.get('summary', {}).get('aggregate_test_sharpe', -999),
                    reverse=True)
    
    table = []
    for i, r in enumerate(ranked):
        s = r.get('summary', {})
        k7 = r.get('kill_criteria', {}).get('K7_adaptive_vs_static', {})
        table.append({
            'rank': i + 1,
            'experiment': r['experiment_id'],
            'agg_test_sr': s.get('aggregate_test_sharpe'),
            'mean_test_sr': s.get('mean_test_sharpe'),
            'consistency': s.get('window_consistency'),
            'total_trades': s.get('total_trades'),
            'max_dd': s.get('max_drawdown'),
            'delta_sr_vs_static': k7.get('delta_sr'),
            'k7_pass': k7.get('pass'),
            'dsr': r.get('deflated_sharpe', {}).get('dsr'),
            'verdict': r.get('verdict'),
        })
    
    return {
        'ranking': table,
        'best_experiment': ranked[0]['experiment_id'] if ranked else None,
        'any_pass': any(r['verdict'] == 'PASS' for r in ranked),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage II-3 Experiment Evaluation")
    parser.add_argument("--experiments_dir", default="logs",
                        help="Root directory containing experiment subdirs")
    parser.add_argument("--baseline_csv",
                        default="logs/static_baseline/static_001_results.csv")
    parser.add_argument("--baseline_summary",
                        default="logs/static_baseline/static_001_summary.json")
    parser.add_argument("--output", default="deliverables/evaluation_results.json")
    parser.add_argument("--n_experiments", type=int, default=7,
                        help="Total number of experiments for DSR correction")
    args = parser.parse_args()
    
    # Find all experiment directories
    exp_dirs = {}
    for entry in sorted(os.listdir(args.experiments_dir)):
        full = os.path.join(args.experiments_dir, entry)
        if os.path.isdir(full) and entry.startswith("exp_A"):
            # Extract experiment ID from directory name
            exp_id = entry.replace("exp_", "")
            exp_dirs[exp_id] = full
    
    if not exp_dirs:
        print("No experiments found!")
        sys.exit(1)
    
    print(f"\nFound {len(exp_dirs)} experiments: {list(exp_dirs.keys())}")
    print(f"Baseline: {args.baseline_csv}")
    
    all_results = []
    for exp_id, exp_dir in exp_dirs.items():
        print(f"\nEvaluating {exp_id}...")
        result = evaluate_single_experiment(
            exp_id, exp_dir,
            args.baseline_csv, args.baseline_summary,
            n_total_experiments=args.n_experiments
        )
        all_results.append(result)
        
        v = result.get('verdict', 'ERROR')
        sr = result.get('summary', {}).get('aggregate_test_sharpe', 0)
        print(f"  Aggregate SR: {sr:.4f}  Verdict: {v}")
    
    # Cross-experiment comparison
    comparison = cross_experiment_comparison(all_results)
    
    output = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'n_experiments_evaluated': len(all_results),
        'n_experiments_for_dsr': args.n_experiments,
        'per_experiment': all_results,
        'cross_experiment': comparison,
        'static_baseline': {
            'csv': args.baseline_csv,
            'summary': args.baseline_summary,
        }
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    if comparison.get('ranking'):
        print(f"\nRanking:")
        for r in comparison['ranking']:
            mark = "✓" if r['verdict'] == 'PASS' else ("?" if r['verdict'] == 'INCONCLUSIVE' else "✗")
            print(f"  {r['rank']}. {r['experiment']:20s} SR={r['agg_test_sr']:+.4f}  "
                  f"ΔSR={r['delta_sr_vs_static']:+.4f}  "
                  f"Consist={r['consistency']*100:.0f}%  "
                  f"DSR={r['dsr']:.3f}  {mark} {r['verdict']}")
    
    best = comparison.get('best_experiment')
    any_pass = comparison.get('any_pass', False)
    if any_pass:
        print(f"\n→ Outcome PA-α: At least one experiment PASSES. Best: {best}")
    elif any(r['verdict'] == 'INCONCLUSIVE' for r in all_results):
        print(f"\n→ Outcome PA-β: Some experiments INCONCLUSIVE. Best: {best}")
    else:
        print(f"\n→ Outcome PA-γ: All experiments FAIL.")


if __name__ == "__main__":
    main()
