#!/usr/bin/env python3
"""
Phase 3.5 — Task 1: Audit the 10σ Noise Budget Ceiling

For each 10σ cell, trace oracle decisions at noise=0 vs noise=10σ:
  - Sign-flip probability (fraction of bars where action differs)
  - Signal-to-noise ratio at decision time
  - Classification: honest ceiling / trivially saturated / ambiguous

If saturated, compute replacement metrics:
  - Minimum prediction accuracy needed for breakeven
  - Magnitude sensitivity (scale predictions by 0.5–1.5)
"""
import numpy as np
import pandas as pd
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import (
    download_asset_data, generate_oracle_signal
)
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_for_timeframe
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# The 10σ cells from Phase 1.3 survivors
TEN_SIGMA_CELLS = [
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "carry_momentum"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "carry_momentum"},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "carry_momentum"},
    {"asset": "EUR/USD", "timeframe": "daily", "strategy": "mean_reversion"},
    {"asset": "USD/JPY", "timeframe": "daily", "strategy": "mean_reversion"},
    {"asset": "GBP/USD", "timeframe": "4h", "strategy": "vol_regime_switch"},
    {"asset": "AUD/USD", "timeframe": "weekly", "strategy": "vol_regime_switch"},
    {"asset": "EUR/JPY", "timeframe": "weekly", "strategy": "vol_regime_switch"},
]

# Detailed audit for these 4 canonical cells (1.1 + 1.2)
CANONICAL_CELLS = [
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum"},
    {"asset": "GBP/USD", "timeframe": "4h", "strategy": "vol_regime_switch"},
    {"asset": "EUR/USD", "timeframe": "daily", "strategy": "mean_reversion"},
]


def get_strategy_positions(strategy_name, log_returns, oracle_signal, close=None):
    """Run strategy with given oracle signal, return positions array."""
    n = len(log_returns)
    positions = np.zeros(n)

    if strategy_name == "momentum":
        lookback = 20
        for i in range(lookback, n):
            cum = np.sum(log_returns[i-lookback:i])
            if abs(cum) > 0.001:
                positions[i] = np.sign(cum) * oracle_signal[i]
            else:
                positions[i] = oracle_signal[i]

    elif strategy_name == "carry_momentum":
        # Same as momentum in oracle test (simplified carry)
        lookback = 20
        for i in range(lookback, n):
            cum = np.sum(log_returns[i-lookback:i])
            if abs(cum) > 0.001:
                positions[i] = np.sign(cum) * oracle_signal[i]
            else:
                positions[i] = oracle_signal[i]

    elif strategy_name == "mean_reversion":
        lookback = 20
        z_entry = 1.5
        z_exit = 0.5
        cum_ret = np.cumsum(log_returns)
        for i in range(lookback, n):
            window = cum_ret[max(0, i-lookback):i+1]
            std = np.std(window)
            if std < 1e-12:
                continue
            z = (cum_ret[i] - np.mean(window)) / std
            if abs(z) > z_entry:
                positions[i] = -np.sign(z) * abs(oracle_signal[i])
            elif abs(z) < z_exit:
                positions[i] = 0
            else:
                positions[i] = positions[i-1] if i > 0 else 0

    elif strategy_name == "breakout":
        lookback = 20
        if close is not None:
            for i in range(lookback, n):
                high = np.max(close[i-lookback:i])
                low = np.min(close[i-lookback:i])
                if close[i] > high:
                    positions[i] = oracle_signal[i]
                elif close[i] < low:
                    positions[i] = -oracle_signal[i]
                else:
                    positions[i] = positions[i-1] if i > 0 else 0
        else:
            positions = oracle_signal.copy()

    elif strategy_name == "vol_regime_switch":
        lookback = 20
        vol_lookback = 60
        cum_ret = np.cumsum(log_returns)
        for i in range(max(lookback, vol_lookback), n):
            recent_vol = np.std(log_returns[i-lookback:i])
            long_vol = np.std(log_returns[i-vol_lookback:i])
            is_low_vol = recent_vol < long_vol

            if is_low_vol:
                # Momentum in low vol
                cum = np.sum(log_returns[i-lookback:i])
                positions[i] = np.sign(cum) * oracle_signal[i]
            else:
                # Mean reversion in high vol
                window = cum_ret[max(0, i-lookback):i+1]
                std = np.std(window)
                if std > 1e-12:
                    z = (cum_ret[i] - np.mean(window)) / std
                    positions[i] = -np.sign(z) * abs(oracle_signal[i])
    else:
        positions = oracle_signal.copy()

    return np.sign(positions)


def trace_cell(asset, timeframe, strategy, seed=42):
    """
    Full trace of one cell: compare actions at noise=0 vs noise=10σ.
    Returns detailed diagnostics.
    """
    print(f"\n  Tracing {asset} / {timeframe} / {strategy}...")

    df = download_asset_data(asset, timeframe)
    if df is None or len(df) < 100:
        return {"error": "insufficient data"}

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])

    # Oracle signals at noise=0 and noise=10
    sig_0 = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    sig_10 = generate_oracle_signal(log_ret, noise_sigma=10.0, horizon=1, seed=seed)

    # Run strategy with both signals
    pos_0 = get_strategy_positions(strategy, log_ret, sig_0, close)
    pos_10 = get_strategy_positions(strategy, log_ret, sig_10, close)

    # Core diagnostic: sign-flip probability
    # Only count bars where at least one position is nonzero
    active_mask = (pos_0 != 0) | (pos_10 != 0)
    n_active = np.sum(active_mask)

    if n_active == 0:
        return {"error": "no active bars"}

    flips = np.sum(pos_0[active_mask] != pos_10[active_mask])
    sign_flip_prob = flips / n_active

    # Signal-to-noise ratio per bar
    # The oracle predicts sign of next return; noise is 10 * realized_vol
    horizon_returns = log_ret[1:]  # future returns
    realized_vol = np.std(horizon_returns) if len(horizon_returns) > 0 else 1e-12
    noise_std = 10.0 * realized_vol

    # For each bar, |true_prediction| / noise_std
    abs_predictions = np.abs(horizon_returns)
    snr_per_bar = abs_predictions / (noise_std + 1e-12)
    median_snr = np.median(snr_per_bar)
    mean_snr = np.mean(snr_per_bar)
    pct_snr_above_1 = np.mean(snr_per_bar > 1.0)

    # Compute Sharpe at noise=0 and noise=10
    ppy = periods_per_year_for_timeframe(timeframe)

    gross_ret_0 = pos_0[:-1] * log_ret[1:]
    gross_ret_10 = pos_10[:-1] * log_ret[1:]

    abs_ret = np.abs(log_ret)
    net_ret_0 = apply_cost_to_returns(gross_ret_0, pos_0[:-1], asset, abs_ret[:-1])
    net_ret_10 = apply_cost_to_returns(gross_ret_10, pos_10[:-1], asset, abs_ret[:-1])

    sharpe_0 = annualized_sharpe(net_ret_0, ppy)
    sharpe_10 = annualized_sharpe(net_ret_10, ppy)

    # B&H sharpe for reference
    bh_ret = log_ret[1:]
    bh_sharpe = annualized_sharpe(bh_ret, ppy)

    # Classify
    if sign_flip_prob >= 0.40:
        classification = "honest_ceiling"
    elif sign_flip_prob < 0.20:
        classification = "trivially_saturated"
    else:
        classification = "ambiguous"

    result = {
        "asset": asset,
        "timeframe": timeframe,
        "strategy": strategy,
        "n_bars": len(log_ret),
        "n_active_bars": int(n_active),
        "sign_flip_probability": round(sign_flip_prob, 4),
        "median_snr_at_10sigma": round(median_snr, 4),
        "mean_snr_at_10sigma": round(mean_snr, 4),
        "pct_bars_snr_above_1": round(pct_snr_above_1, 4),
        "sharpe_noise_0": round(sharpe_0, 4),
        "sharpe_noise_10": round(sharpe_10, 4),
        "bh_sharpe": round(bh_sharpe, 4),
        "classification": classification,
        "realized_vol": round(realized_vol, 6),
        "noise_std_10sigma": round(noise_std, 6),
    }

    print(f"    Sign-flip prob: {sign_flip_prob:.1%}")
    print(f"    Median SNR at 10σ: {median_snr:.4f}")
    print(f"    Sharpe@0σ: {sharpe_0:+.3f}, Sharpe@10σ: {sharpe_10:+.3f}, B&H: {bh_sharpe:+.3f}")
    print(f"    Classification: {classification}")

    return result


def compute_replacement_metrics(asset, timeframe, strategy, seed=42):
    """
    For saturated/ambiguous cells, compute:
    1. Minimum directional accuracy needed
    2. Magnitude sensitivity (scale predictions)
    """
    print(f"    Computing replacement metrics for {asset}/{timeframe}/{strategy}...")

    df = download_asset_data(asset, timeframe)
    if df is None or len(df) < 100:
        return {}

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    # 1. Minimum accuracy needed: binary search
    # Use the perfect oracle positions and progressively randomize
    sig_perfect = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    pos_perfect = get_strategy_positions(strategy, log_ret, sig_perfect, close)

    # Test at different accuracy levels
    accuracy_results = []
    rng = np.random.RandomState(seed)
    for target_accuracy in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0]:
        sharpes = []
        for s in range(20):  # 20 trials per accuracy level
            rng2 = np.random.RandomState(seed + s)
            # Flip (1-accuracy) fraction of signals
            corrupt_sig = sig_perfect.copy()
            flip_mask = rng2.random(len(corrupt_sig)) > target_accuracy
            corrupt_sig[flip_mask] *= -1

            pos = get_strategy_positions(strategy, log_ret, corrupt_sig, close)
            gross = pos[:-1] * log_ret[1:]
            net = apply_cost_to_returns(gross, pos[:-1], asset, abs_ret[:-1])
            sharpes.append(annualized_sharpe(net, ppy))

        mean_sr = np.mean(sharpes)
        accuracy_results.append({
            "accuracy": target_accuracy,
            "mean_sharpe": round(mean_sr, 4),
            "std_sharpe": round(np.std(sharpes), 4),
        })

    # Find minimum accuracy for breakeven (Sharpe >= 0.3)
    min_accuracy_breakeven = None
    for ar in accuracy_results:
        if ar["mean_sharpe"] >= 0.3:
            min_accuracy_breakeven = ar["accuracy"]
            break

    # 2. Magnitude sensitivity: scale predictions
    magnitude_results = []
    for scale in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        scaled_sig = sig_perfect * scale
        pos = get_strategy_positions(strategy, log_ret, scaled_sig, close)
        gross = pos[:-1] * log_ret[1:]
        net = apply_cost_to_returns(gross, pos[:-1], asset, abs_ret[:-1])
        sr = annualized_sharpe(net, ppy)
        magnitude_results.append({
            "scale": scale,
            "sharpe": round(sr, 4),
        })

    print(f"      Min accuracy for breakeven: {min_accuracy_breakeven}")
    print(f"      Magnitude sensitivity range: {magnitude_results[0]['sharpe']:.3f} to {magnitude_results[-1]['sharpe']:.3f}")

    return {
        "min_accuracy_breakeven": min_accuracy_breakeven,
        "accuracy_curve": accuracy_results,
        "magnitude_sensitivity": magnitude_results,
    }


def main():
    print("=" * 70)
    print("PHASE 3.5 — TASK 1: AUDIT 10σ NOISE BUDGET CEILING")
    print("=" * 70)

    # Task 1.1 + 1.2: Trace canonical cells
    print("\n--- Task 1.1–1.2: Detailed Trace of 10σ Cells ---")
    trace_results = []
    for cell in TEN_SIGMA_CELLS:
        result = trace_cell(cell["asset"], cell["timeframe"], cell["strategy"])
        trace_results.append(result)

    # Task 1.3: Classification summary
    print("\n" + "=" * 70)
    print("TASK 1.3: CLASSIFICATION SUMMARY")
    print("=" * 70)

    honest = [r for r in trace_results if r.get("classification") == "honest_ceiling"]
    saturated = [r for r in trace_results if r.get("classification") == "trivially_saturated"]
    ambiguous = [r for r in trace_results if r.get("classification") == "ambiguous"]

    print(f"\n  Honest ceiling (sign-flip ≥ 40%): {len(honest)}")
    for r in honest:
        print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: flip={r['sign_flip_probability']:.1%}")

    print(f"\n  Trivially saturated (sign-flip < 20%): {len(saturated)}")
    for r in saturated:
        print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: flip={r['sign_flip_probability']:.1%}")

    print(f"\n  Ambiguous (20-40%): {len(ambiguous)}")
    for r in ambiguous:
        print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: flip={r['sign_flip_probability']:.1%}")

    # Task 1.4: Replacement metrics for saturated/ambiguous cells
    print("\n" + "=" * 70)
    print("TASK 1.4: REPLACEMENT METRICS FOR NON-HONEST CELLS")
    print("=" * 70)

    for r in trace_results:
        if r.get("classification") in ("trivially_saturated", "ambiguous"):
            replacement = compute_replacement_metrics(
                r["asset"], r["timeframe"], r["strategy"]
            )
            r["replacement_metrics"] = replacement

    # Kill criterion check
    print("\n" + "=" * 70)
    print("KILL CRITERION CHECK")
    print("=" * 70)
    n_saturated = len(saturated)
    n_total = len([r for r in trace_results if "error" not in r])
    print(f"\n  {n_saturated} of {n_total} 10σ cells are trivially saturated")
    if n_saturated >= 6:
        print("  ⚠ KILL CRITERION TRIGGERED: 6+ of 7 10σ cells saturated")
        print("  → Reprioritize Phase 4 toward Tier 2 (3σ) XAU cells")
    else:
        print("  ✓ Kill criterion not triggered")

    # Save results
    output_path = os.path.join(RESULTS_DIR, "noise_budget_audit.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_cells_audited": len(trace_results),
            "n_honest": len(honest),
            "n_saturated": len(saturated),
            "n_ambiguous": len(ambiguous),
            "kill_criterion_triggered": n_saturated >= 6,
            "cells": trace_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
