"""
Walk-Forward Optimizer (WFO) for heuristic-strategy.

Implements anchored expanding-window walk-forward optimization:
  - Splits full dataset into yearly folds
  - For each fold: GA optimizes on training data, evaluates ONCE on unseen test year
  - Aggregates only OOS (out-of-sample) trades = true performance

This is the only honest way to evaluate a trading strategy.
"""

import json
import time
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from deap import base, creator, tools
from tqdm import tqdm
import os as _os

_QUIET = _os.environ.get("STRATEGY_QUIET", "0") == "1"


def _sharpe_fitness(profit, trades_list, min_trades=15):
    """
    Sharpe-adjusted fitness that penalizes insufficient trades.

    Returns Sharpe * sqrt(N) where N = number of trades.
    Strategies with < min_trades trades get -1e6 (rejected).
    """
    n = len(trades_list)
    if n < min_trades:
        return -1e6

    returns = [t['pnl'] for t in trades_list]
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    if std_r < 1e-10:
        # All trades identical — suspicious, penalize
        return mean_r * np.sqrt(n) if mean_r > 0 else -1e6

    sharpe = mean_r / std_r
    return sharpe * np.sqrt(n)


def _evaluate_with_sharpe(individual, plugin, base_data, config, min_trades=15):
    """
    Evaluate a candidate using Sharpe-based fitness instead of raw profit.
    """
    result = plugin.evaluate_candidate(
        individual, base_data, None, None, config
    )
    if isinstance(result, tuple) and len(result) == 2:
        profit, stats = result
    else:
        profit = result[0] if isinstance(result, tuple) else result
        stats = {"num_trades": 0}

    trades_list = getattr(plugin, "trades", [])
    fitness = _sharpe_fitness(profit, trades_list, min_trades)
    return (fitness,)


def _run_ga_on_slice(plugin, train_data, config,
                     population_size=30, num_generations=20,
                     min_trades=15):
    """
    Run GA optimization on a training data slice.
    Returns best parameters dict.
    """
    optimizable_params = plugin.get_optimizable_params()
    num_params = len(optimizable_params)

    if not hasattr(creator, "WFOFitnessMax"):
        creator.create("WFOFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "WFOIndividual"):
        creator.create("WFOIndividual", list, fitness=creator.WFOFitnessMax)

    toolbox = base.Toolbox()

    def random_attr(param):
        _, low, high = param
        return random.uniform(low, high)

    toolbox.register(
        "individual",
        lambda: creator.WFOIndividual([random_attr(p) for p in optimizable_params])
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return _evaluate_with_sharpe(ind, plugin, train_data, config, min_trades)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Bounds repair: clip parameters to valid ranges after crossover/mutation
    bounds = [(low, high) for _, low, high in optimizable_params]

    def _repair_bounds(ind):
        for i, (lo, hi) in enumerate(bounds):
            ind[i] = max(lo, min(hi, ind[i]))
        return ind

    random.seed(42)
    population = toolbox.population(n=population_size)

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    cxpb = config.get("crossover_probability", 0.5)
    mutpb = config.get("mutation_probability", 0.2)

    for gen in range(1, num_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                _repair_bounds(c1)
                _repair_bounds(c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                _repair_bounds(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]

        if not _QUIET:
            best_fit = max(fits)
            avg_fit = np.mean(fits)
            print(f"  Gen {gen:2d}: Best Sharpe*√N = {best_fit:.3f}, "
                  f"Avg = {avg_fit:.3f}")

    best_ind = tools.selBest(population, 1)[0]
    best_params = {
        name: best_ind[i]
        for i, (name, _, _) in enumerate(optimizable_params)
    }
    return best_params, best_ind.fitness.values[0]


def run_walk_forward(plugin, full_data, config,
                     train_years=3,
                     first_test_year=2013,
                     last_test_year=2020,
                     population_size=30,
                     num_generations=20,
                     min_trades=10):
    """
    Rolling-window walk-forward optimization.

    For each test year Y:
      - Train on [Y - train_years, Y-1]
      - Test on [Y, Y]
      - Record OOS trades

    Args:
        plugin: Strategy plugin instance
        full_data: DataFrame with DatetimeIndex, columns OPEN/HIGH/LOW/CLOSE
        config: Strategy configuration dict
        train_years: Number of years in rolling training window
        first_test_year: First year to use as test set
        last_test_year: Last year to use as test set
        population_size: GA population per fold
        num_generations: GA generations per fold
        min_trades: Minimum trades required for valid fitness

    Returns:
        dict with fold results, aggregated OOS metrics
    """
    start_time = time.time()

    if not isinstance(full_data.index, pd.DatetimeIndex):
        raise ValueError("full_data must have a DatetimeIndex")

    print(f"\n{'='*70}")
    print(f"WALK-FORWARD OPTIMIZATION (Rolling {train_years}yr window)")
    print(f"{'='*70}")
    print(f"Data: {full_data.index.min().date()} to {full_data.index.max().date()} "
          f"({len(full_data)} bars)")
    print(f"Train window: {train_years} years (rolling)")
    print(f"Test folds: {first_test_year} to {last_test_year} "
          f"({last_test_year - first_test_year + 1} folds)")
    print(f"GA: pop={population_size}, gen={num_generations}")
    print(f"Min trades per fold: {min_trades}")
    print(f"Fitness: Sharpe * √N (penalize < {min_trades} trades)")
    print(f"{'='*70}\n")

    fold_results = []
    all_oos_trades = []

    for test_year in range(first_test_year, last_test_year + 1):
        fold_start = time.time()
        train_start_year = test_year - train_years
        train_end_year = test_year - 1

        # Split data by year — rolling window
        train_mask = (full_data.index.year >= train_start_year) & \
                     (full_data.index.year <= train_end_year)
        test_mask = full_data.index.year == test_year

        train_data = full_data[train_mask].copy()
        test_data = full_data[test_mask].copy()

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"[FOLD {test_year}] SKIP — insufficient data "
                  f"(train={len(train_data)}, test={len(test_data)})")
            continue

        print(f"[FOLD {test_year}] Train: {train_start_year}-{train_end_year} "
              f"({len(train_data)} bars) | "
              f"Test: {test_year} ({len(test_data)} bars)")

        # --- Phase 1: GA optimization on training data ---
        print(f"  Optimizing on training data...")
        train_min_trades = max(min_trades, int(min_trades * train_years / 2))

        best_params, best_fitness = _run_ga_on_slice(
            plugin, train_data, config,
            population_size=population_size,
            num_generations=num_generations,
            min_trades=train_min_trades
        )

        print(f"  Best params: { {k: round(v, 4) for k, v in best_params.items()} }")
        print(f"  Train fitness (Sharpe*√N): {best_fitness:.3f}")

        # --- Phase 2: Evaluate ONCE on unseen test year ---
        print(f"  Evaluating on OOS test year {test_year}...")
        opt_params = plugin.get_optimizable_params()
        candidate = [best_params[name] for name, _, _ in opt_params]

        result = plugin.evaluate_candidate(
            candidate, test_data, None, None, config
        )
        if isinstance(result, tuple) and len(result) == 2:
            oos_profit, oos_stats = result
        else:
            oos_profit = result[0] if isinstance(result, tuple) else result
            oos_stats = {}

        oos_trades = getattr(plugin, "trades", [])
        n_oos = len(oos_trades)
        oos_wins = sum(1 for t in oos_trades if t['pnl'] > 0)
        oos_win_pct = (oos_wins / n_oos * 100) if n_oos > 0 else 0
        oos_pnl_list = [t['pnl'] for t in oos_trades]
        oos_sharpe = (np.mean(oos_pnl_list) / (np.std(oos_pnl_list) + 1e-10)
                      if n_oos > 1 else 0)

        fold_time = time.time() - fold_start

        fold_info = {
            "test_year": test_year,
            "train_range": f"{train_start_year}-{train_end_year}",
            "train_bars": len(train_data),
            "test_bars": len(test_data),
            "best_params": best_params,
            "train_fitness": best_fitness,
            "oos_profit": oos_profit,
            "oos_trades": n_oos,
            "oos_win_pct": oos_win_pct,
            "oos_sharpe": oos_sharpe,
            "fold_time_sec": fold_time,
        }
        fold_results.append(fold_info)

        # Tag OOS trades with fold info
        for t in oos_trades:
            t["wfo_fold"] = test_year
            t["wfo_train_range"] = f"{train_start_year}-{train_end_year}"
        all_oos_trades.extend(oos_trades)

        print(f"  OOS Result: Profit=${oos_profit:.2f}, "
              f"Trades={n_oos}, Win%={oos_win_pct:.1f}%, "
              f"Sharpe={oos_sharpe:.2f} "
              f"({fold_time:.0f}s)")
        print()

    # ─── Aggregate OOS results ───
    total_time = time.time() - start_time
    total_oos_profit = sum(f["oos_profit"] for f in fold_results)
    total_oos_trades = sum(f["oos_trades"] for f in fold_results)
    total_oos_wins = sum(1 for t in all_oos_trades if t['pnl'] > 0)
    total_win_pct = (total_oos_wins / total_oos_trades * 100
                     if total_oos_trades > 0 else 0)

    if total_oos_trades > 1:
        all_pnl = [t['pnl'] for t in all_oos_trades]
        agg_sharpe = np.mean(all_pnl) / (np.std(all_pnl) + 1e-10)
        max_dd_pips = max((t.get('max_dd', 0) for t in all_oos_trades), default=0)
    else:
        agg_sharpe = 0
        max_dd_pips = 0

    # Equity curve from OOS trades
    equity = 10000.0
    equity_curve = [equity]
    peak = equity
    max_drawdown_usd = 0
    for t in all_oos_trades:
        equity += t['pnl']
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_drawdown_usd:
            max_drawdown_usd = dd

    print(f"\n{'='*70}")
    print(f"WALK-FORWARD RESULTS (OUT-OF-SAMPLE ONLY)")
    print(f"{'='*70}")
    print(f"Folds: {len(fold_results)}")
    print(f"Total OOS Profit:    ${total_oos_profit:,.2f}")
    print(f"Total OOS Trades:    {total_oos_trades}")
    print(f"Win Rate:            {total_win_pct:.1f}%")
    print(f"Aggregate Sharpe:    {agg_sharpe:.3f}")
    print(f"Max DD (trade pips): {max_dd_pips:.0f}")
    print(f"Max DD (equity USD): ${max_drawdown_usd:,.2f}")
    print(f"Final Equity:        ${equity:,.2f}")
    print(f"Total Time:          {total_time:.0f}s")
    print()

    # Per-fold summary table
    print(f"{'Year':<6} {'Train':<12} {'OOS Profit':>12} {'Trades':>8} "
          f"{'Win%':>7} {'Sharpe':>8} {'Params'}")
    print("-" * 90)
    for f in fold_results:
        params_str = ", ".join(
            f"{k}={v:.3f}" for k, v in f["best_params"].items()
        )
        print(f"{f['test_year']:<6} {f['train_range']:<12} "
              f"${f['oos_profit']:>10,.2f} {f['oos_trades']:>8} "
              f"{f['oos_win_pct']:>6.1f}% {f['oos_sharpe']:>7.2f}  "
              f"{params_str}")
    print(f"{'='*70}\n")

    # Profitable folds
    profitable_folds = sum(1 for f in fold_results if f["oos_profit"] > 0)
    print(f"Profitable folds: {profitable_folds}/{len(fold_results)} "
          f"({profitable_folds/max(len(fold_results),1)*100:.0f}%)")

    return {
        "fold_results": fold_results,
        "all_oos_trades": all_oos_trades,
        "total_oos_profit": total_oos_profit,
        "total_oos_trades": total_oos_trades,
        "total_win_pct": total_win_pct,
        "aggregate_sharpe": agg_sharpe,
        "max_drawdown_usd": max_drawdown_usd,
        "final_equity": equity,
        "equity_curve": equity_curve,
        "total_time_sec": total_time,
    }
