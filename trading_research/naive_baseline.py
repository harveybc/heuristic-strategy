#!/usr/bin/env python3
"""
Phase 1.3: Naive Baseline Test

For the top N cells from the oracle sweep, compare against:
1. Buy-and-hold Sharpe
2. Random entry baseline (100 seeds, matching trade frequency + SL/TP)

Only keep cells where oracle Sharpe materially exceeds B&H AND random (p < 0.05).

Kill criterion: if fewer than 5 cells survive, search space is empty.
"""
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trading_research.oracle_sensitivity import download_asset_data, evaluate_cell, NOISE_GRID
from trading_research.transaction_cost_model import total_cost_bps, COST_TABLE
from trading_research.evaluation_harness import (
    annualized_sharpe, rolling_window_evaluation,
    periods_per_year_24h, periods_per_year_for_timeframe
)


def random_baseline(log_returns: np.ndarray, trade_freq: float, asset: str,
                    abs_returns: np.ndarray, ppy: float,
                    n_seeds: int = 100) -> dict:
    """
    Run random entry strategy matching the given trade frequency.

    Returns distribution of Sharpe ratios.
    """
    n = len(log_returns)
    sharpes = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1000)

        # Random positions with matching frequency
        positions = np.zeros(n)
        in_trade = False
        for i in range(1, n):
            if not in_trade:
                if rng.random() < trade_freq:
                    positions[i] = rng.choice([-1, 1])
                    in_trade = True
            else:
                # Hold for random duration (5-30 bars)
                hold_prob = 1.0 / 15.0  # avg 15 bar hold
                if rng.random() < hold_prob:
                    in_trade = False
                    positions[i] = 0
                else:
                    positions[i] = positions[i-1]

        # Returns with costs
        gross_ret = positions[:-1] * log_returns[1:]
        gross_ret = np.concatenate([[0], gross_ret])

        pos_changes = np.diff(positions, prepend=0)
        trade_mask = pos_changes != 0
        trailing_vol = np.full_like(abs_returns, np.nanmean(abs_returns))
        for j in range(30, len(abs_returns)):
            trailing_vol[j] = np.mean(abs_returns[j-30:j])

        net_ret = gross_ret.copy()
        for j in range(len(net_ret)):
            if trade_mask[j]:
                cost = total_cost_bps(asset, abs_returns[j], trailing_vol[j]) / 10000.0
                net_ret[j] -= cost

        sharpes.append(annualized_sharpe(net_ret, ppy))

    return {
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
        "p5": float(np.percentile(sharpes, 5)),
        "p95": float(np.percentile(sharpes, 95)),
        "sharpes": [round(s, 4) for s in sharpes],
    }


def test_cell(cell: dict, top_n: int = 30) -> dict:
    """Test one cell against baselines."""
    asset = cell["asset"]
    tf = cell["timeframe"]
    strat = cell["strategy"]

    print(f"  {asset} / {tf} / {strat}...", end=" ", flush=True)

    try:
        df = download_asset_data(asset, tf)
        close = df["Close"].values.astype(float)
        log_returns = np.diff(np.log(close + 1e-12))
        log_returns = np.concatenate([[0], log_returns])
        abs_returns = np.abs(log_returns)

        is_24h = COST_TABLE.get(asset, {}).get("category", "") in (
            "crypto_major", "crypto_alt", "fx_major", "fx_cross", "fx_emerging"
        )
        ppy = periods_per_year_24h(tf) if is_24h else periods_per_year_for_timeframe(tf)

        # B&H
        bh_sharpe = annualized_sharpe(log_returns, ppy)

        # Estimate trade frequency from oracle result
        oracle_trades_per_year = cell.get("noise_results", [{}])[0].get("trades_per_year", 50)
        trade_freq_per_bar = oracle_trades_per_year / ppy if ppy > 0 else 0.01

        # Random baseline
        rng_result = random_baseline(log_returns, trade_freq_per_bar, asset,
                                     abs_returns, ppy, n_seeds=100)

        oracle_sharpe = cell["oracle_sharpe"]

        # Statistical test: is oracle better than random at p < 0.05?
        rng_sharpes = np.array(rng_result["sharpes"])
        p_value = float(np.mean(rng_sharpes >= oracle_sharpe))

        # Does oracle beat B&H?
        beats_bh = oracle_sharpe > bh_sharpe + 0.1  # margin of 0.1

        survives = (p_value < 0.05) and beats_bh

        result = {
            "asset": asset,
            "timeframe": tf,
            "strategy": strat,
            "oracle_sharpe": oracle_sharpe,
            "bh_sharpe": round(bh_sharpe, 4),
            "random_mean_sharpe": round(rng_result["mean"], 4),
            "random_std_sharpe": round(rng_result["std"], 4),
            "random_p95_sharpe": round(rng_result["p95"], 4),
            "p_value_vs_random": round(p_value, 4),
            "beats_bh": beats_bh,
            "survives": survives,
            "noise_budget": cell["noise_budget"],
        }

        status = "SURVIVES ★" if survives else "killed"
        print(f"oracle={oracle_sharpe:+.3f} bh={bh_sharpe:+.3f} "
              f"rng_p={p_value:.3f} → {status}")
        return result

    except Exception as e:
        print(f"FAILED: {e}")
        return {
            "asset": asset, "timeframe": tf, "strategy": strat,
            "error": str(e), "survives": False,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/oracle_sweep_merged.json")
    parser.add_argument("--output", default="results/baseline_test.json")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    all_results = data.get("results", data.get("top30", data.get("all_results", [])))
    # Sort by noise_budget descending to get top cells
    all_results.sort(key=lambda x: x.get("noise_budget", 0), reverse=True)
    top_cells = all_results[:args.top_n]
    print(f"Testing top {len(top_cells)} cells against baselines...")
    print("=" * 80)

    results = []
    for cell in top_cells:
        r = test_cell(cell)
        results.append(r)

    survivors = [r for r in results if r.get("survives", False)]
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(survivors)} of {len(results)} cells survive")
    print(f"{'='*80}")

    if len(survivors) < 5:
        print(f"\n⚠⚠⚠ KILL CRITERION TRIGGERED: only {len(survivors)} survivors")
        print("The search space may be empty. See Phase 5 decision tree.")
    else:
        print(f"\n✓ {len(survivors)} cells pass baseline test. Ready for Phase 4.")

    for r in survivors:
        print(f"  ★ {r['asset']}/{r['timeframe']}/{r['strategy']}: "
              f"SR={r['oracle_sharpe']:+.3f}, budget={r['noise_budget']:.2f}σ")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n_tested": len(results),
            "n_survivors": len(survivors),
            "kill_criterion_triggered": len(survivors) < 5,
            "survivors": survivors,
            "all_results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
