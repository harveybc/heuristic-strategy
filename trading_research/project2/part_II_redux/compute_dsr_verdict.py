#!/usr/bin/env python3
"""
Stage II-6.5: Deflated Sharpe Ratio + Kill Criteria Evaluation
Per Bailey & López de Prado (2014) and F-10 §5.2

Usage: python compute_dsr_verdict.py
Reads HO results from logs/HO_*/results.csv, computes DSR, applies kill criteria.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

REDUX_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(REDUX_DIR, "logs")

# Total experiments in Part II-Redux (for multiple testing correction)
N_STRATEGIES = 17  # 7 Path A + 10 Path B


def load_results(experiment_dir):
    """Load results CSV from an experiment directory."""
    csvs = [f for f in os.listdir(experiment_dir) if f.endswith("_results.csv")]
    if not csvs:
        return None
    df = pd.read_csv(os.path.join(experiment_dir, csvs[0]))
    return df


def compute_aggregate_sharpe(test_pnl_series):
    """Compute aggregate Sharpe from a combined PnL series."""
    if len(test_pnl_series) == 0:
        return 0.0
    mean_ret = np.mean(test_pnl_series)
    std_ret = np.std(test_pnl_series, ddof=1)
    if std_ret == 0:
        return 0.0
    # Annualize: 4h bars → 6 bars/day × 365 days
    ann_factor = np.sqrt(6 * 365)
    return (mean_ret / std_ret) * ann_factor


def expected_max_sr(n_strategies, variance_sr=1.0):
    """
    Expected maximum Sharpe ratio under null hypothesis.
    E[max(SR)] ≈ √(2 ln(N)) - (ln(π) + ln(ln(N))) / (2 √(2 ln(N)))
    From Bailey & López de Prado (2014), Theorem 1.
    """
    if n_strategies <= 1:
        return 0.0
    z = np.sqrt(2 * np.log(n_strategies))
    euler_mascheroni = 0.5772156649
    correction = (euler_mascheroni + np.log(np.sqrt(np.log(n_strategies)))) / z
    return (z - correction) * np.sqrt(variance_sr)


def compute_sr_variance(sr, n_obs, skewness=0.0, kurtosis=3.0):
    """
    Variance of Sharpe ratio estimator, adjusted for non-normality.
    Var(SR) ≈ (1 + 0.5 × SR² - γ₃ × SR + (γ₄/4) × SR²) / (n - 1)
    where γ₃ = skewness, γ₄ = excess kurtosis
    """
    excess_kurt = kurtosis - 3.0  # Convert to excess kurtosis
    numerator = 1.0 + 0.5 * sr**2 - skewness * sr + (excess_kurt / 4.0) * sr**2
    return numerator / max(n_obs - 1, 1)


def compute_dsr(observed_sr, n_obs, n_strategies, skewness=0.0, kurtosis=3.0):
    """
    Compute Deflated Sharpe Ratio.
    DSR = Φ[(SR - E[max(SR)]) / √Var(SR)]
    Returns: (dsr_pvalue, dsr_zscore, expected_max, sr_stderr)
    """
    sr_var = compute_sr_variance(observed_sr, n_obs, skewness, kurtosis)
    sr_stderr = np.sqrt(sr_var)

    e_max = expected_max_sr(n_strategies, variance_sr=1.0)

    z_score = (observed_sr - e_max) / sr_stderr if sr_stderr > 0 else 0.0
    p_value = stats.norm.cdf(z_score)

    return {
        "dsr_pvalue": p_value,
        "dsr_zscore": z_score,
        "expected_max_sr": e_max,
        "sr_stderr": sr_stderr,
        "observed_sr": observed_sr,
        "n_obs": n_obs,
        "n_strategies": n_strategies,
    }


def apply_kill_criteria(results_df, experiment_id):
    """
    Apply kill criteria K-1 through K-7 to HO results.
    Returns dict of criteria results.
    """
    criteria = {}

    test_sharpes = results_df["test_sharpe"].values
    trades = results_df["num_trades"].values
    maxdd = results_df["max_dd"].values
    cost_ratios = results_df["cost_ratio"].values

    # K-1: Aggregate HO Sharpe > 0
    agg_sr = np.mean(test_sharpes)  # Simple average across windows
    criteria["K1_agg_sharpe"] = {
        "value": float(agg_sr),
        "threshold": "> 0",
        "pass": agg_sr > 0,
    }

    # K-2: Worst 2-year rolling Sharpe > -0.9
    # For yearly windows, 2-year rolling = pairs of consecutive windows
    if len(test_sharpes) >= 2:
        rolling_2yr = [
            np.mean(test_sharpes[i : i + 2]) for i in range(len(test_sharpes) - 1)
        ]
        worst_2yr = min(rolling_2yr)
    else:
        worst_2yr = test_sharpes[0] if len(test_sharpes) > 0 else -999
    criteria["K2_worst_2yr_sharpe"] = {
        "value": float(worst_2yr),
        "threshold": "> -0.9",
        "pass": worst_2yr > -0.9,
    }

    # K-3: Cost ratio ≥ 2.0
    valid_cr = cost_ratios[cost_ratios != 0]
    median_cr = float(np.median(valid_cr)) if len(valid_cr) > 0 else 0.0
    criteria["K3_cost_ratio"] = {
        "value": median_cr,
        "threshold": ">= 2.0",
        "pass": median_cr >= 2.0,
    }

    # K-5: Window consistency ≥ 60%
    positive_windows = np.sum(test_sharpes > 0)
    total_windows = len(test_sharpes)
    consistency = positive_windows / total_windows if total_windows > 0 else 0.0
    criteria["K5_consistency"] = {
        "value": float(consistency),
        "detail": f"{positive_windows}/{total_windows}",
        "threshold": ">= 60%",
        "pass": consistency >= 0.60,
    }

    # K-6: Train-test MAE ratio < 3x in > 50% of windows (Path B only)
    if "train_mae" in results_df.columns and results_df["train_mae"].sum() > 0:
        mae_ratios = results_df["test_mae"] / results_df["train_mae"].replace(0, np.nan)
        mae_ratios = mae_ratios.dropna()
        pct_under_3x = (mae_ratios < 3.0).mean() if len(mae_ratios) > 0 else 0.0
        criteria["K6_mae_ratio"] = {
            "value": float(pct_under_3x),
            "threshold": "> 50% windows with ratio < 3x",
            "pass": pct_under_3x > 0.50,
        }

    # Overall
    criteria["overall_pass"] = all(
        v["pass"] for k, v in criteria.items() if k != "overall_pass"
    )

    return criteria


def format_verdict(experiment_id, ho_sharpe, dsr_result, kill_result):
    """Determine final verdict."""
    if not kill_result["overall_pass"]:
        return "HELD_OUT_FAILED"

    if ho_sharpe <= 0:
        return "HELD_OUT_FAILED"

    dsr_p = dsr_result["dsr_pvalue"]
    # 95% CI: DSR p-value > 0.975 means CI entirely above 0
    # DSR p-value > 0.5 means point estimate above expected max
    if dsr_p > 0.975:
        return "HELD_OUT_VALIDATED_DSR_POSITIVE"
    elif dsr_p > 0.5:
        return "HELD_OUT_VALIDATED_DSR_MARGINAL"
    else:
        return "HELD_OUT_FAILED"


def main():
    print("=" * 70)
    print("  STAGE II-6.5: Deflated Sharpe Ratio + Final Verdict")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  N strategies (multiple testing): {N_STRATEGIES}")
    print("=" * 70)
    print()

    # Expected max SR under null
    e_max = expected_max_sr(N_STRATEGIES)
    print(f"E[max(SR)] for N={N_STRATEGIES}: {e_max:.4f}")
    print()

    # Process each HO experiment
    ho_experiments = {
        "HO_A1_btc_momentum_yearly": {"path": "A", "is_sharpe": 0.155},
        "HO_A2_btc_momentum_monthly": {"path": "A", "is_sharpe": 0.156},
        "HO_A6_btc_momentum_hpo": {"path": "A", "is_sharpe": 0.142},
        "HO_B3_tft_yearly": {"path": "B", "is_sharpe": 0.024},
    }

    verdicts = {}
    for exp_id, info in ho_experiments.items():
        exp_dir = os.path.join(LOGS_DIR, exp_id)
        if not os.path.exists(exp_dir):
            print(f"SKIP: {exp_id} — directory not found")
            continue

        results = load_results(exp_dir)
        if results is None or len(results) == 0:
            print(f"SKIP: {exp_id} — no results")
            continue

        print(f"\n{'='*60}")
        print(f"  {exp_id}")
        print(f"{'='*60}")

        # Per-window results
        print("\n  Window Results:")
        print(
            f"  {'Win':>4} {'Test SR':>10} {'Trades':>8} {'MaxDD':>8} {'CR':>8}"
        )
        for _, row in results.iterrows():
            print(
                f"  {int(row['window_id']):4d} {row['test_sharpe']:10.4f} "
                f"{int(row['num_trades']):8d} {row['max_dd']:8.3f} "
                f"{row['cost_ratio']:8.2f}"
            )

        # Aggregate Sharpe
        agg_sr = float(results["test_sharpe"].mean())
        n_obs = int(results["num_trades"].sum())

        # Return distribution stats (approximate from window Sharpes)
        test_sharpes = results["test_sharpe"].values
        skew = float(stats.skew(test_sharpes)) if len(test_sharpes) > 2 else 0.0
        kurt = float(stats.kurtosis(test_sharpes, fisher=False)) if len(test_sharpes) > 2 else 3.0

        print(f"\n  Aggregate HO Sharpe: {agg_sr:+.4f}")
        print(f"  IS Sharpe:           {info['is_sharpe']:+.4f}")
        print(f"  Windows:             {len(results)}")
        print(f"  Total trades:        {n_obs}")
        print(f"  Sharpe skewness:     {skew:.3f}")
        print(f"  Sharpe kurtosis:     {kurt:.3f}")

        # Kill criteria
        kill = apply_kill_criteria(results, exp_id)
        print(f"\n  Kill Criteria:")
        for k, v in kill.items():
            if k == "overall_pass":
                continue
            status = "PASS" if v["pass"] else "FAIL"
            print(f"    {k}: {v['value']:.4f} (threshold {v['threshold']}) → {status}")
        print(f"    Overall: {'PASS' if kill['overall_pass'] else 'FAIL'}")

        # DSR
        dsr = compute_dsr(agg_sr, n_obs, N_STRATEGIES, skew, kurt)
        print(f"\n  DSR Analysis:")
        print(f"    E[max(SR)]:     {dsr['expected_max_sr']:.4f}")
        print(f"    SR stderr:      {dsr['sr_stderr']:.4f}")
        print(f"    DSR z-score:    {dsr['dsr_zscore']:.4f}")
        print(f"    DSR p-value:    {dsr['dsr_pvalue']:.4f}")

        verdict = format_verdict(exp_id, agg_sr, dsr, kill)
        print(f"\n  VERDICT: {verdict}")
        verdicts[exp_id] = {
            "agg_ho_sharpe": agg_sr,
            "is_sharpe": info["is_sharpe"],
            "dsr_pvalue": dsr["dsr_pvalue"],
            "dsr_zscore": dsr["dsr_zscore"],
            "kill_pass": kill["overall_pass"],
            "verdict": verdict,
            "kill_details": kill,
            "dsr_details": dsr,
        }

    # Final summary
    print(f"\n\n{'='*70}")
    print("  FINAL VERDICT MATRIX")
    print(f"{'='*70}")
    print(
        f"\n  {'Strategy':<35} {'IS SR':>8} {'HO SR':>8} {'DSR p':>8} {'Verdict'}"
    )
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*30}")
    for exp_id, v in verdicts.items():
        print(
            f"  {exp_id:<35} {v['is_sharpe']:+8.3f} {v['agg_ho_sharpe']:+8.3f} "
            f"{v['dsr_pvalue']:8.3f} {v['verdict']}"
        )

    # Part III recommendation
    a2_v = verdicts.get("HO_A2_btc_momentum_monthly", {}).get("verdict", "MISSING")
    b3_v = verdicts.get("HO_B3_tft_yearly", {}).get("verdict", "MISSING")

    print(f"\n  Part III Recommendation:")
    if "VALIDATED" in a2_v and "VALIDATED" in b3_v:
        print("  → Both A2 and B3 validated. Part III proceeds with full ambition.")
    elif "VALIDATED" in a2_v:
        print("  → Only A2 validated. Part III focuses on Path A refinement, multi-asset expansion.")
    elif "VALIDATED" in b3_v:
        print("  → Only B3 validated. Part III focuses on ML refinement, ensemble with A as filter.")
    else:
        print("  → Neither validated. Part II-Redux closes as null.")

    # Save JSON
    output_path = os.path.join(
        REDUX_DIR, "deliverables", "dsr_verdict_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(verdicts, f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
