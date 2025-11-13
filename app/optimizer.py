import backtrader as bt
import random
import datetime
from deap import base, creator, tools
import time
from tqdm import tqdm
import multiprocessing
import json

# Global variables for optimization
_plugin = None
_base_data = None
_hourly_predictions = None
_daily_predictions = None
_config = None
_current_epoch = 1  # Global variable to hold current epoch number
_plugin_param_count = 0  # Number of parameters belonging to the strategy plugin
_base_param_bounds = None  # Bounds for plugin parameters only (name, low, high)

def _parse_offsets_from_cols(cols):
    import re
    offs = []
    for i, c in enumerate(cols, start=1):
        m = re.search(r"H(\d+)$", str(c))
        offs.append(int(m.group(1)) if m else i)
    return offs

def _aggregate_mae_and_naive(base_df, preds_df, selected_cols):
    """
    Compute mean MAE across selected prediction columns and mean MAE for a naive baseline (y_hat = y_t).
    Returns (mae, naive_mae). If no valid comparisons, returns (None, None).
    """
    import pandas as pd
    import numpy as np
    if preds_df is None or preds_df.empty or base_df is None or base_df.empty:
        return None, None
    cols = [c for c in selected_cols if c in preds_df.columns]
    if not cols:
        return None, None
    offsets = _parse_offsets_from_cols(cols)
    maes = []
    naive_maes = []
    price = base_df["CLOSE"] if "CLOSE" in base_df.columns else None
    if price is None:
        return None, None
    for col, off in zip(cols, offsets):
        forecast_times = preds_df.index + pd.Timedelta(hours=int(off))
        actual = price.reindex(forecast_times)
        actual.index = preds_df.index
        pred = preds_df[col]
        valid = actual.notna() & pred.notna()
        if valid.sum() == 0:
            continue
        mae = float(np.mean(np.abs(actual[valid].values - pred[valid].values)))
        # Naive baseline: predict y_{t+off} = y_t
        naive = price.reindex(preds_df.index)
        naive_mae = float(np.mean(np.abs(actual[valid].values - naive[valid].values)))
        maes.append(mae)
        naive_maes.append(naive_mae)
    if not maes:
        return None, None
    return float(np.mean(maes)), float(np.mean(naive_maes))

def _compute_uniform_offsets(max_horizon: int, num_predictions: int):
    """
    Compute uniformly spaced integer offsets between 1 and max_horizon (inclusive).
    Returns a sorted list of ints with length == num_predictions.
    """
    import numpy as np
    if max_horizon is None or num_predictions is None:
        return None
    max_horizon = int(max_horizon)
    num_predictions = int(num_predictions)
    if max_horizon < 1 or num_predictions < 1:
        return []
    if num_predictions == 1:
        return [max_horizon]
    offs = np.linspace(1, max_horizon, num=num_predictions)
    offs = np.round(offs).astype(int).tolist()
    offs[0] = 1
    offs[-1] = max_horizon
    uniq = []
    for o in offs:
        if 1 <= o <= max_horizon and (not uniq or o != uniq[-1]):
            uniq.append(o)
    if len(uniq) != num_predictions:
        step = max(1, max_horizon // (num_predictions - 1))
        uniq = list(range(1, 1 + step * (num_predictions - 1), step))
        uniq[-1] = max_horizon
    return uniq

def init_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Initializes the optimizer with the provided plugin and datasets.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config, _num_generations
    _plugin = plugin
    _base_data = base_data
    _hourly_predictions = hourly_predictions
    _daily_predictions = daily_predictions
    _config = config
    _num_generations = config.get("num_generations", 10)
    print("[INIT] Optimizer initialized with strategy plugin.")


def evaluate_individual(individual):
    """
    Evaluates a candidate strategy parameter set.
    Prints the current epoch number along with the candidate.
    Expects the plugin's evaluate_candidate() method to return either:
         - A tuple: (profit, stats) where stats is a dict containing keys 'num_trades', 'win_pct', 'max_dd', 'sharpe'
         - Or a single-value tuple (profit,)
    This function always returns (profit, stats) to satisfy downstream requirements.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config, _current_epoch, _num_generations
    if _plugin is None:
        print("[EVALUATE] ERROR: _plugin is None!")
        return -1e6, {}
    
    # Print the candidate and current epoch information.
    print(f"[EVALUATE][Epoch {_current_epoch}/{_num_generations}] Evaluating candidate (genome): {individual}")
    
    # Split individual's genome into plugin parameters and prediction parameters (if present)
    global _plugin_param_count
    plugin_params = individual
    hourly_df = _hourly_predictions
    daily_df = _daily_predictions
    base_df = _base_data
    cfg = dict(_config) if _config is not None else {}

    # Clamp plugin parameters to configured bounds if available
    if _plugin_param_count:
        try:
            # Ensure plugin_params is a mutable list slice
            if not isinstance(plugin_params, list):
                plugin_params = list(plugin_params)
            if _base_param_bounds and len(_base_param_bounds) >= _plugin_param_count:
                for i in range(_plugin_param_count):
                    name, low, high = _base_param_bounds[i]
                    v = plugin_params[i]
                    if v < low:
                        v = low
                    elif v > high:
                        v = high
                    plugin_params[i] = v
        except Exception:
            # Fail-safe: do not crash evaluation if bounds missing/misaligned
            pass

    if _plugin_param_count and len(individual) > _plugin_param_count:
        plugin_params = individual[:_plugin_param_count]
        # Extract prediction config (rounded to int and clamped to supported ranges)
        st_max = int(round(individual[_plugin_param_count + 0]))
        st_n   = int(round(individual[_plugin_param_count + 1]))
        lt_max = int(round(individual[_plugin_param_count + 2]))
        lt_n   = int(round(individual[_plugin_param_count + 3]))

        # Clamp to supported bounds
        st_max = max(2, min(48, st_max))
        st_n   = max(2, min(48, st_n))
        st_n   = min(st_n, st_max)

        lt_max = max(24, min(144, lt_max))
        lt_n   = max(24, min(144, lt_n))
        lt_n   = min(lt_n, lt_max)

        # Compute offsets and select columns from superset predictions
        st_offsets = _compute_uniform_offsets(st_max, st_n)
        lt_offsets = _compute_uniform_offsets(lt_max, lt_n)

        # Build column names and filter by availability
        st_cols = [f"Prediction_H{o}" for o in st_offsets]
        lt_cols = [f"Prediction_H{o}" for o in lt_offsets]
        st_cols = [c for c in st_cols if c in hourly_df.columns]
        lt_cols = [c for c in lt_cols if c in daily_df.columns]

        hourly_df = hourly_df[st_cols]
        daily_df = daily_df[lt_cols]

        # Construct uncertainties matching shapes using defaults, for consistency
        import pandas as pd
        unc_h_val = cfg.get("default_uncertainty_short_term", 0.0002)
        unc_d_val = cfg.get("default_uncertainty_long_term", 0.0047)
        # Name uncertainty columns to avoid overlap with prediction columns
        unc_hourly_cols = [f"Uncertainty_h_{i+1}" for i in range(len(hourly_df.columns))]
        unc_daily_cols = [f"Uncertainty_d_{i+1}" for i in range(len(daily_df.columns))]
        unc_hourly_df = pd.DataFrame(unc_h_val, index=hourly_df.index, columns=unc_hourly_cols)
        unc_daily_df = pd.DataFrame(unc_d_val, index=daily_df.index, columns=unc_daily_cols)
        cfg["uncertainty_hourly"] = unc_hourly_df
        cfg["uncertainty_daily"] = unc_daily_df
        cfg["hourly_columns"] = st_cols
        cfg["daily_columns"] = lt_cols

        # Align base and predictions to common index
        common_idx = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
        if not common_idx.empty:
            base_df = base_df.loc[common_idx]
            hourly_df = hourly_df.loc[common_idx]
            daily_df = daily_df.loc[common_idx]

    result = _plugin.evaluate_candidate(plugin_params, base_df, hourly_df, daily_df, cfg)
    
    # Normalize to (profit, stats)
    if isinstance(result, tuple) and len(result) == 2:
        profit, stats = result
    elif isinstance(result, tuple) and len(result) == 1:
        profit, stats = result[0], {}
    else:
        profit, stats = result, {}
    
    print(f"[EVALUATE][Epoch {_current_epoch}/{_num_generations}] Candidate => "
          f"Profit: {profit:.2f}, "
          f"Trades: {stats.get('num_trades', 0)}, "
          f"Win%: {stats.get('win_pct', 0):.1f}, "
          f"MaxDD: {stats.get('max_dd', 0):.2f}, "
          f"Sharpe: {stats.get('sharpe', 0):.2f}")
    return profit, stats


def run_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config,
                  base_val=None, hourly_val=None, daily_val=None,
                  base_test=None, hourly_test=None, daily_test=None):
    """
    Runs the optimizer using DEAP to optimize the strategy parameters.
    Displays a TQDM progress bar for the evaluation of individuals in each generation.
    Prints detailed candidate evaluation information including the current epoch.
    Saves the best found parameters as JSON if config['save_config'] is provided.
    """
    global _current_epoch, _base_data, _hourly_predictions, _daily_predictions
    from tqdm import tqdm
    import random
    import multiprocessing
    import json
    from deap import base, creator, tools

    # Initialize globals
    init_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config)

    optimizable_params = plugin.get_optimizable_params()
    num_params = len(optimizable_params)
    # Extend with prediction configuration parameters (treated as continuous, rounded later)
    # Ranges per user requirement:
    #  - short_term_max_horizon: [2, 48]
    #  - short_term_num_predictions: [2, 48]
    #  - long_term_max_horizon: [24, 144]
    #  - long_term_num_predictions: [24, 144]
    global _plugin_param_count, _base_param_bounds
    _plugin_param_count = num_params
    pred_params = [
        ("short_term_max_horizon", 2, 48),
        ("short_term_num_predictions", 2, 48),
        ("long_term_max_horizon", 24, 144),
        ("long_term_num_predictions", 24, 144),
    ]
    optimizable_params = optimizable_params + pred_params
    # Store bounds for base plugin params only
    _base_param_bounds = list(optimizable_params[:_plugin_param_count])
    total_params = len(optimizable_params)
    print(f"Optimizable Parameters ({total_params}):")
    for name, low, high in optimizable_params:
        print(f"  {name}: [{low}, {high}]")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def random_attr(param):
        name, low, high = param
        return random.uniform(low, high)

    # Register GA primitives
    toolbox.register("individual", lambda: creator.Individual([random_attr(p) for p in optimizable_params]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)
    population_size = config.get("population_size", 20)
    num_generations = config.get("num_generations", 50)
    cxpb = config.get("crossover_probability", 0.5)
    mutpb = config.get("mutation_probability", 0.2)

    print("Starting Genetic Algorithm Optimization")
    disable_mp = config.get("disable_multiprocessing", False)
    if not disable_mp:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    population = toolbox.population(n=population_size)
    print("[OPTIMIZATION] Evaluating initial population...")
    if disable_mp:
        fitnesses = []
        with tqdm(total=len(population), desc="Initial eval", unit="cand") as pbar:
            for ind in population:
                result = toolbox.evaluate(ind)
                ind.fitness.values = (result[0],)
                fitnesses.append(result)
                pbar.update(1)
    else:
        fitnesses = []
        with tqdm(total=len(population), desc="Initial eval", unit="cand") as pbar:
            for result in toolbox.map(toolbox.evaluate, population):
                fitnesses.append(result)
                pbar.update(1)
        for ind, f in zip(population, fitnesses):
            ind.fitness.values = (f[0],)

    print(f"  Evaluated {len(population)} individuals initially.")

    # Early stopping state
    patience = int(config.get("patience", 0) or 0)
    best_val_profit = None
    epochs_without_improve = 0
    best_ind_overall = None
    best_ind_train_profit = None
    best_ind_val_profit = None
    best_ind_genome_snapshot = None  # immutable copy of genome at last validation improvement

    for gen in range(1, num_generations):
        _current_epoch = gen+1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if disable_mp:
            with tqdm(total=len(invalid_ind), desc=f"Epoch {gen} eval", unit="cand") as pbar:
                for ind in invalid_ind:
                    fit = toolbox.evaluate(ind)
                    ind.fitness.values = (fit[0],)
                    pbar.update(1)
        else:
            fitnesses = []
            with tqdm(total=len(invalid_ind), desc=f"Epoch {gen} eval", unit="cand") as pbar:
                for fit in toolbox.map(toolbox.evaluate, invalid_ind):
                    fitnesses.append(fit)
                    pbar.update(1)
            for ind, f in zip(invalid_ind, fitnesses):
                ind.fitness.values = (f[0],)

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        gen_best_ind = tools.selBest(population, 1)[0]
        gen_best_profit = gen_best_ind.fitness.values[0]
        print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits) / len(fits):.2f}")

        # Validation evaluation for early stopping
        if base_val is not None and hourly_val is not None and daily_val is not None:
            # Evaluate gen_best_ind on validation data through evaluate_individual-like path
            # Temporarily swap global datasets
            prev_base, prev_hourly, prev_daily = _base_data, _hourly_predictions, _daily_predictions
            _base_data, _hourly_predictions, _daily_predictions = base_val, hourly_val, daily_val
            val_profit, _ = evaluate_individual(gen_best_ind)
            # Restore originals
            _base_data, _hourly_predictions, _daily_predictions = prev_base, prev_hourly, prev_daily
            print(f"  Validation Profit (gen {gen}): {val_profit:.2f}")

            improved = (best_val_profit is None) or (val_profit > best_val_profit)
            if improved:
                best_val_profit = val_profit
                epochs_without_improve = 0
                best_ind_overall = gen_best_ind
                best_ind_train_profit = gen_best_profit
                best_ind_val_profit = val_profit
                # Snapshot genome to ensure immutability even if underlying Individual changes later
                best_ind_genome_snapshot = list(gen_best_ind)
            else:
                epochs_without_improve += 1
                print(f"  No validation improvement. Patience counter: {epochs_without_improve}/{patience}")

            # Per-epoch summary with patience and last best profit (validation)
            best_pf_str = (f"{best_val_profit:.2f}" if best_val_profit is not None else "N/A")
            print(f"[Epoch {gen}] LastBestProfit={best_pf_str} | Patience={epochs_without_improve}/{patience}")

            if patience > 0 and epochs_without_improve >= patience and not improved:
                print("Early stopping triggered: validation profit did not improve within patience.")
                break

        else:
            # If no validation set, track best training individual
            if best_ind_overall is None or gen_best_profit > (best_ind_train_profit or -1e9):
                best_ind_overall = gen_best_ind
                best_ind_train_profit = gen_best_profit

            # Per-epoch summary without validation
            best_train_str = (f"{best_ind_train_profit:.2f}" if best_ind_train_profit is not None else "N/A")
            # Patience is effectively disabled when no validation set; print 0/patience for clarity
            print(f"[Epoch {gen}] LastBestProfit={best_train_str} | Patience={0}/{patience}")

    from deap import tools
    # Determine final champion (early stop may have set best_ind_overall)
    if best_ind_overall is None:
        best_ind = tools.selBest(population, 1)[0]
        best_ind_train_profit = best_ind.fitness.values[0]
        best_ind_genome_snapshot = list(best_ind)
    else:
        # Rebuild an Individual from snapshot if available to avoid accidental later mutation side-effects
        if best_ind_genome_snapshot is not None:
            best_ind = creator.Individual(best_ind_genome_snapshot)
        else:
            best_ind = best_ind_overall
    # Build raw params from genome
    best_params_raw = {name: best_ind[i] for i, (name, _, _) in enumerate(optimizable_params)}

    # Sanitize and clamp integer prediction params before printing/saving
    def _ri(x):
        try:
            return int(round(float(x)))
        except Exception:
            return int(x)

    best_params = dict(best_params_raw)
    # Short-term
    if "short_term_max_horizon" in best_params:
        st_max = _ri(best_params["short_term_max_horizon"])
        st_max = max(2, min(48, st_max))
        best_params["short_term_max_horizon"] = st_max
    if "short_term_num_predictions" in best_params:
        st_n = _ri(best_params["short_term_num_predictions"])
        st_n = max(2, min(48, st_n))
        # ensure st_n <= st_max (if st_max present)
        if "short_term_max_horizon" in best_params:
            st_n = min(st_n, best_params["short_term_max_horizon"])
        best_params["short_term_num_predictions"] = st_n
    # Long-term
    if "long_term_max_horizon" in best_params:
        lt_max = _ri(best_params["long_term_max_horizon"])
        lt_max = max(24, min(144, lt_max))
        best_params["long_term_max_horizon"] = lt_max
    if "long_term_num_predictions" in best_params:
        lt_n = _ri(best_params["long_term_num_predictions"])
        lt_n = max(24, min(144, lt_n))
        if "long_term_max_horizon" in best_params:
            lt_n = min(lt_n, best_params["long_term_max_horizon"])
        best_params["long_term_num_predictions"] = lt_n

    print("[OPTIMIZATION] Best parameter set found:")
    for name in best_params_raw.keys():
        val = best_params.get(name, best_params_raw[name])
        # Print ints without decimals; floats with 4 decimals
        if isinstance(val, int):
            print(f"  {name} = {val}")
        else:
            print(f"  {name} = {float(val):.4f}")
    # Determine achieved profit for logging without relying on fitness tuple (may be empty if snapshot)
    achieved_profit = None
    if best_ind_val_profit is not None:
        achieved_profit = best_ind_val_profit
    elif best_ind_train_profit is not None:
        achieved_profit = best_ind_train_profit
    else:
        try:
            achieved_profit = best_ind.fitness.values[0]
        except Exception:
            achieved_profit = None
    if achieved_profit is not None:
        print(f"Achieved Profit: {achieved_profit:.2f}")
    else:
        print("Achieved Profit: N/A (will be computed in champion evaluation)")

    if not disable_mp:
        pool.close()

    # Save the best parameters as JSON if configured (using config["save_config"] for best parameters).
    if config.get("save_config"):
        try:
            with open(config["save_config"], "w") as f:
                json.dump(best_params, f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_config']}: {e}")

    # Evaluate champion on train, validation, and test (if provided)
    # Train (current globals already point to train/base_for_opt)
    train_profit, train_stats = evaluate_individual(best_ind)

    # Validation
    val_profit, val_stats = None, {}
    if base_val is not None and hourly_val is not None and daily_val is not None:
        prev_base, prev_hourly, prev_daily = _base_data, _hourly_predictions, _daily_predictions
        _base_data, _hourly_predictions, _daily_predictions = base_val, hourly_val, daily_val
        val_profit, val_stats = evaluate_individual(best_ind)
        _base_data, _hourly_predictions, _daily_predictions = prev_base, prev_hourly, prev_daily

    # Test
    test_profit, test_stats = None, {}
    if base_test is not None and hourly_test is not None and daily_test is not None:
        prev_base, prev_hourly, prev_daily = _base_data, _hourly_predictions, _daily_predictions
        _base_data, _hourly_predictions, _daily_predictions = base_test, hourly_test, daily_test
        test_profit, test_stats = evaluate_individual(best_ind)
        _base_data, _hourly_predictions, _daily_predictions = prev_base, prev_hourly, prev_daily

    # Print compact summary
    def fmt_stats(s):
        if not s:
            return "-"
        return (
            f"Trades={s.get('num_trades', 0)}, Win%={s.get('win_pct', 0):.1f}, "
            f"MaxDD={s.get('max_dd', 0):.2f}, Sharpe={s.get('sharpe', 0):.2f}"
        )

    print("\nChampion evaluation summary:")
    # Determine selected columns based on best_params (if present)
    st_cols_sel = []
    lt_cols_sel = []
    try:
        st_max = int(best_params.get("short_term_max_horizon")) if "short_term_max_horizon" in best_params else None
        st_n   = int(best_params.get("short_term_num_predictions")) if "short_term_num_predictions" in best_params else None
        lt_max = int(best_params.get("long_term_max_horizon")) if "long_term_max_horizon" in best_params else None
        lt_n   = int(best_params.get("long_term_num_predictions")) if "long_term_num_predictions" in best_params else None
        if st_max and st_n:
            st_cols_sel = [f"Prediction_H{o}" for o in _compute_uniform_offsets(st_max, st_n)]
        if lt_max and lt_n:
            lt_cols_sel = [f"Prediction_H{o}" for o in _compute_uniform_offsets(lt_max, lt_n)]
    except Exception:
        pass

    # Compute MAE / Naive MAE for train (current globals)
    train_mae, train_naive_mae = _aggregate_mae_and_naive(_base_data, _hourly_predictions.join(_daily_predictions, how="inner"), st_cols_sel + lt_cols_sel)
    print(f"  Train:       Profit={train_profit:.2f} | {fmt_stats(train_stats)}" + (f" | MAE={train_mae:.6f} | NaiveMAE={train_naive_mae:.6f}" if train_mae is not None else ""))
    if val_profit is not None:
        # Temporarily swap to validation data to compute MAE
        prev_base, prev_hourly, prev_daily = _base_data, _hourly_predictions, _daily_predictions
        _base_data, _hourly_predictions, _daily_predictions = base_val, hourly_val, daily_val
        val_mae, val_naive_mae = _aggregate_mae_and_naive(_base_data, _hourly_predictions.join(_daily_predictions, how="inner"), st_cols_sel + lt_cols_sel)
        _base_data, _hourly_predictions, _daily_predictions = prev_base, prev_hourly, prev_daily
        print(f"  Validation:  Profit={val_profit:.2f} | {fmt_stats(val_stats)}" + (f" | MAE={val_mae:.6f} | NaiveMAE={val_naive_mae:.6f}" if val_mae is not None else ""))
    if test_profit is not None:
        prev_base, prev_hourly, prev_daily = _base_data, _hourly_predictions, _daily_predictions
        _base_data, _hourly_predictions, _daily_predictions = base_test, hourly_test, daily_test
        test_mae, test_naive_mae = _aggregate_mae_and_naive(_base_data, _hourly_predictions.join(_daily_predictions, how="inner"), st_cols_sel + lt_cols_sel)
        _base_data, _hourly_predictions, _daily_predictions = prev_base, prev_hourly, prev_daily
        print(f"  Test:        Profit={test_profit:.2f} | {fmt_stats(test_stats)}" + (f" | MAE={test_mae:.6f} | NaiveMAE={test_naive_mae:.6f}" if test_mae is not None else ""))

    result_payload = {
        "best_parameters": best_params,
        "profit": train_profit,
        "stats": train_stats,
        "validation_profit": val_profit,
        "validation_stats": val_stats,
        "test_profit": test_profit,
        "test_stats": test_stats,
    }

    # Attach MAE metrics to payload
    result_payload.update({
        "train_mae": train_mae,
        "train_naive_mae": train_naive_mae,
    })
    if val_profit is not None:
        result_payload.update({
            "validation_mae": val_mae,
            "validation_naive_mae": val_naive_mae,
        })
    if test_profit is not None:
        result_payload.update({
            "test_mae": test_mae,
            "test_naive_mae": test_naive_mae,
        })

    return result_payload

if __name__ == '__main__':
    print("Standalone testing of optimizer not supported; run via main pipeline.")
