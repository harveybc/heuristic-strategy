#!/usr/bin/env python3
"""
Phase 3.5 — Task 2: Edge-Over-Buy-and-Hold Reranking

For all 14 surviving cells:
  2.1 — Compute Edge Sharpe = Oracle Sharpe - B&H Sharpe
  2.2 — Identify B&H-dominated cells (noise budget relative to B&H ≤ 1σ)
  2.3 — Decompose into long-only / short-only / long+short Sharpes
"""
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import (
    download_asset_data, generate_oracle_signal, NOISE_GRID
)
from trading_research.transaction_cost_model import apply_cost_to_returns
from trading_research.evaluation_harness import (
    annualized_sharpe, periods_per_year_for_timeframe
)
from trading_research.audit_noise_budget import get_strategy_positions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# All 14 survivors from Phase 1.3
SURVIVORS = [
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "momentum", "noise_budget": 10.0},
    {"asset": "BTC/USD", "timeframe": "weekly", "strategy": "carry_momentum", "noise_budget": 10.0},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "momentum", "noise_budget": 10.0},
    {"asset": "XAU/USD", "timeframe": "weekly", "strategy": "carry_momentum", "noise_budget": 10.0},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "momentum", "noise_budget": 10.0},
    {"asset": "XAG/USD", "timeframe": "weekly", "strategy": "carry_momentum", "noise_budget": 10.0},
    {"asset": "EUR/USD", "timeframe": "daily", "strategy": "mean_reversion", "noise_budget": 10.0},
    {"asset": "USD/JPY", "timeframe": "daily", "strategy": "mean_reversion", "noise_budget": 10.0},
    {"asset": "GBP/USD", "timeframe": "4h", "strategy": "vol_regime_switch", "noise_budget": 10.0},
    {"asset": "AUD/USD", "timeframe": "weekly", "strategy": "vol_regime_switch", "noise_budget": 10.0},
    {"asset": "EUR/JPY", "timeframe": "weekly", "strategy": "vol_regime_switch", "noise_budget": 10.0},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "momentum", "noise_budget": 3.0},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "carry_momentum", "noise_budget": 3.0},
    {"asset": "XAU/USD", "timeframe": "daily", "strategy": "vol_regime_switch", "noise_budget": 3.0},
]


def evaluate_cell_edge(cell, seed=42):
    """
    Full edge-over-B&H analysis for one cell.
    Returns: edge sharpe, noise budget relative to B&H, long/short decomposition.
    """
    asset = cell["asset"]
    timeframe = cell["timeframe"]
    strategy = cell["strategy"]
    print(f"\n  {asset} / {timeframe} / {strategy}...")

    df = download_asset_data(asset, timeframe)
    if df is None or len(df) < 100:
        return {"error": "insufficient data", **cell}

    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close + 1e-12))
    log_ret = np.concatenate([[0], log_ret])
    ppy = periods_per_year_for_timeframe(timeframe)
    abs_ret = np.abs(log_ret)

    # B&H returns and Sharpe
    bh_ret = log_ret[1:]
    bh_sharpe = annualized_sharpe(bh_ret, ppy)

    # --- Task 2.1: Edge Sharpe at noise=0 ---
    sig_0 = generate_oracle_signal(log_ret, noise_sigma=0.0, horizon=1, seed=seed)
    pos_full = get_strategy_positions(strategy, log_ret, sig_0, close)

    gross_full = pos_full[:-1] * log_ret[1:]
    net_full = apply_cost_to_returns(gross_full, pos_full[:-1], asset, abs_ret[:-1])
    oracle_sharpe = annualized_sharpe(net_full, ppy)
    edge_sharpe = oracle_sharpe - bh_sharpe

    # --- Task 2.1: Noise budget relative to B&H ---
    # Find noise level where oracle Sharpe drops to B&H Sharpe
    noise_budget_vs_bh = 0.0
    for noise in NOISE_GRID:
        sig_n = generate_oracle_signal(log_ret, noise_sigma=noise, horizon=1, seed=seed)
        pos_n = get_strategy_positions(strategy, log_ret, sig_n, close)
        gross_n = pos_n[:-1] * log_ret[1:]
        net_n = apply_cost_to_returns(gross_n, pos_n[:-1], asset, abs_ret[:-1])
        sr_n = annualized_sharpe(net_n, ppy)
        if sr_n >= bh_sharpe:
            noise_budget_vs_bh = noise
        else:
            break

    # --- Task 2.2: B&H dominated? ---
    is_bh_dominated = noise_budget_vs_bh <= 1.0

    # --- Task 2.3: Long-only / Short-only decomposition ---
    pos_long_only = np.where(pos_full > 0, pos_full, 0)
    pos_short_only = np.where(pos_full < 0, pos_full, 0)

    gross_long = pos_long_only[:-1] * log_ret[1:]
    gross_short = pos_short_only[:-1] * log_ret[1:]

    net_long = apply_cost_to_returns(gross_long, pos_long_only[:-1], asset, abs_ret[:-1])
    net_short = apply_cost_to_returns(gross_short, pos_short_only[:-1], asset, abs_ret[:-1])

    sharpe_long = annualized_sharpe(net_long, ppy)
    sharpe_short = annualized_sharpe(net_short, ppy)

    # Long/short trade counts
    n_long_bars = np.sum(pos_full > 0)
    n_short_bars = np.sum(pos_full < 0)
    n_flat_bars = np.sum(pos_full == 0)

    result = {
        **cell,
        "n_bars": len(log_ret),
        "oracle_sharpe": round(oracle_sharpe, 4),
        "bh_sharpe": round(bh_sharpe, 4),
        "edge_sharpe": round(edge_sharpe, 4),
        "noise_budget_vs_bh": noise_budget_vs_bh,
        "is_bh_dominated": is_bh_dominated,
        "sharpe_long_only": round(sharpe_long, 4),
        "sharpe_short_only": round(sharpe_short, 4),
        "sharpe_long_short": round(oracle_sharpe, 4),
        "n_long_bars": int(n_long_bars),
        "n_short_bars": int(n_short_bars),
        "n_flat_bars": int(n_flat_bars),
        "long_pct": round(n_long_bars / max(len(pos_full), 1) * 100, 1),
        "short_pct": round(n_short_bars / max(len(pos_full), 1) * 100, 1),
    }

    print(f"    Oracle SR: {oracle_sharpe:+.3f}, B&H SR: {bh_sharpe:+.3f}, Edge SR: {edge_sharpe:+.3f}")
    print(f"    Budget vs B&H: {noise_budget_vs_bh:.2f}σ {'(B&H DOMINATED)' if is_bh_dominated else ''}")
    print(f"    Long-only: {sharpe_long:+.3f}, Short-only: {sharpe_short:+.3f}")
    print(f"    Long {result['long_pct']:.0f}% / Short {result['short_pct']:.0f}% / Flat {100-result['long_pct']-result['short_pct']:.0f}%")

    return result


def main():
    print("=" * 70)
    print("PHASE 3.5 — TASK 2: EDGE-OVER-BUY-AND-HOLD RERANKING")
    print("=" * 70)

    results = []
    for cell in SURVIVORS:
        r = evaluate_cell_edge(cell)
        results.append(r)

    # Sort by edge_sharpe descending
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x["edge_sharpe"], reverse=True)

    # Summary table
    print("\n" + "=" * 70)
    print("UPDATED RANKING BY EDGE SHARPE")
    print("=" * 70)
    print(f"{'Rank':>4} {'Asset':<10} {'TF':<7} {'Strategy':<20} {'Oracle':>7} {'B&H':>7} {'Edge':>7} {'BvBH':>6} {'Long':>7} {'Short':>7} {'Dom':>4}")
    print("-" * 100)
    for i, r in enumerate(valid):
        dom = "YES" if r["is_bh_dominated"] else ""
        print(f"{i+1:>4} {r['asset']:<10} {r['timeframe']:<7} {r['strategy']:<20} "
              f"{r['oracle_sharpe']:>+7.3f} {r['bh_sharpe']:>+7.3f} {r['edge_sharpe']:>+7.3f} "
              f"{r['noise_budget_vs_bh']:>5.1f}σ {r['sharpe_long_only']:>+7.3f} {r['sharpe_short_only']:>+7.3f} "
              f"{dom:>4}")

    # B&H dominated cells
    bh_dominated = [r for r in valid if r["is_bh_dominated"]]
    non_dominated = [r for r in valid if not r["is_bh_dominated"]]
    print(f"\n  B&H dominated (removed): {len(bh_dominated)}")
    for r in bh_dominated:
        print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: "
              f"edge={r['edge_sharpe']:+.3f}, budget_vs_bh={r['noise_budget_vs_bh']:.1f}σ")

    print(f"\n  Surviving (not B&H dominated): {len(non_dominated)}")
    for r in non_dominated:
        print(f"    {r['asset']}/{r['timeframe']}/{r['strategy']}: "
              f"edge={r['edge_sharpe']:+.3f}, budget_vs_bh={r['noise_budget_vs_bh']:.1f}σ")

    # Kill criterion
    if len(non_dominated) < 5:
        print(f"\n  ⚠ WARNING: Only {len(non_dominated)} cells survive after B&H filter")
    else:
        print(f"\n  ✓ {len(non_dominated)} cells survive B&H filter")

    # Save
    output_path = os.path.join(RESULTS_DIR, "edge_over_bh_ranking.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_tested": len(results),
            "n_bh_dominated": len(bh_dominated),
            "n_surviving": len(non_dominated),
            "ranked_results": valid,
            "bh_dominated": bh_dominated,
            "non_dominated": non_dominated,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
