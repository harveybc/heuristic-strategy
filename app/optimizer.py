import random
import multiprocessing
import json

from deap import base, creator, tools
from tqdm import tqdm

_active_optimizer_instance = None

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

def _aggregate_mae_and_naive_for_sets(base_df, df_cols_pairs):
    """
    Compute MAE and naive MAE across multiple (preds_df, selected_cols) pairs.
    Each pair is processed independently to avoid column name overlaps.
    Returns (mae, naive_mae) averaged across all valid columns.
    """
    import pandas as pd
    import numpy as np
    if base_df is None or base_df.empty:
        return None, None
    price = base_df["CLOSE"] if "CLOSE" in base_df.columns else None
    if price is None:
        return None, None
    all_maes = []
    all_naives = []
    for preds_df, cols in df_cols_pairs:
        if preds_df is None or preds_df.empty:
            continue
        cols = [c for c in (cols or []) if c in preds_df.columns] or list(preds_df.columns)
        offsets = _parse_offsets_from_cols(cols)
        for col, off in zip(cols, offsets):
            forecast_times = preds_df.index + pd.Timedelta(hours=int(off))
            actual = price.reindex(forecast_times)
            actual.index = preds_df.index
            pred = preds_df[col]
            valid = actual.notna() & pred.notna()
            if valid.sum() == 0:
                continue
            mae = float(np.mean(np.abs(actual[valid].values - pred[valid].values)))
            naive = price.reindex(preds_df.index)
            naive_mae = float(np.mean(np.abs(actual[valid].values - naive[valid].values)))
            all_maes.append(mae)
            all_naives.append(naive_mae)
    if not all_maes:
        return None, None
    return float(np.mean(all_maes)), float(np.mean(all_naives))

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

def _mp_safe_evaluate(individual):
    if _active_optimizer_instance is None:
        print("[EVALUATE] ERROR: No active optimizer plugin instance.")
        return -1e6, {}
    return _active_optimizer_instance.evaluate_individual(individual)


class Plugin:
    """Optimizer plugin wrapping the GA/DEAP optimization pipeline."""

    plugin_params = {
        "population_size": 20,
        "num_generations": 50,
        "crossover_probability": 0.5,
        "mutation_probability": 0.2,
        "disable_multiprocessing": True,
        "patience": 0,
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self._strategy_plugin = None
        self._base_data = None
        self._hourly_predictions = None
        self._daily_predictions = None
        self._config = {}
        self._num_generations = 0
        self._current_epoch = 1
        self._plugin_param_count = 0
        self._base_param_bounds = []

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return dict(self.params)

    def init_optimizer(self, strategy_plugin, base_data, hourly_predictions, daily_predictions, config):
        global _active_optimizer_instance

        self._strategy_plugin = strategy_plugin
        self._base_data = base_data
        self._hourly_predictions = hourly_predictions
        self._daily_predictions = daily_predictions
        self._config = dict(config) if config is not None else {}
        self._num_generations = int(config.get("num_generations", self.params.get("num_generations", 10)))
        self._current_epoch = 1
        self._plugin_param_count = 0
        self._base_param_bounds = []
        _active_optimizer_instance = self
        print("[INIT] Optimizer initialized with strategy plugin.")

    def _clone_config(self):
        return dict(self._config) if self._config is not None else {}

    def evaluate_individual(self, individual):
        if self._strategy_plugin is None:
            print("[EVALUATE] ERROR: Strategy plugin is not set!")
            return -1e6, {}

        print(
            f"[EVALUATE][Epoch {self._current_epoch}/{self._num_generations}] Evaluating candidate (genome): {individual}"
        )

        plugin_params = list(individual)
        hourly_df = self._hourly_predictions
        daily_df = self._daily_predictions
        base_df = self._base_data
        cfg = self._clone_config()

        if self._plugin_param_count:
            try:
                if self._base_param_bounds and len(self._base_param_bounds) >= self._plugin_param_count:
                    for i in range(self._plugin_param_count):
                        name, low, high = self._base_param_bounds[i]
                        value = plugin_params[i]
                        if value < low:
                            value = low
                        elif value > high:
                            value = high
                        plugin_params[i] = value
            except Exception:
                pass

        if self._plugin_param_count and len(individual) > self._plugin_param_count:
            plugin_params = individual[: self._plugin_param_count]
            st_max = int(round(individual[self._plugin_param_count + 0]))
            st_n = int(round(individual[self._plugin_param_count + 1]))
            lt_max = int(round(individual[self._plugin_param_count + 2]))
            lt_n = int(round(individual[self._plugin_param_count + 3]))

            st_max = max(2, min(48, st_max))
            st_n = max(2, min(48, st_n))
            st_n = min(st_n, st_max)

            lt_max = max(24, min(144, lt_max))
            lt_n = max(24, min(144, lt_n))
            lt_n = min(lt_n, lt_max)

            st_offsets = _compute_uniform_offsets(st_max, st_n)
            lt_offsets = _compute_uniform_offsets(lt_max, lt_n)

            st_cols = [f"Prediction_H{o}" for o in st_offsets or []]
            lt_cols = [f"Prediction_H{o}" for o in lt_offsets or []]
            st_cols = [c for c in st_cols if hourly_df is not None and c in hourly_df.columns]
            lt_cols = [c for c in lt_cols if daily_df is not None and c in daily_df.columns]

            if hourly_df is not None and st_cols:
                hourly_df = hourly_df[st_cols]
            if daily_df is not None and lt_cols:
                daily_df = daily_df[lt_cols]

            import pandas as pd

            unc_h_val = cfg.get("default_uncertainty_short_term", 0.0002)
            unc_d_val = cfg.get("default_uncertainty_long_term", 0.0047)
            if hourly_df is not None and len(hourly_df.columns) > 0:
                unc_hourly_cols = [f"Uncertainty_h_{i+1}" for i in range(len(hourly_df.columns))]
                cfg["uncertainty_hourly"] = pd.DataFrame(unc_h_val, index=hourly_df.index, columns=unc_hourly_cols)
            if daily_df is not None and len(daily_df.columns) > 0:
                unc_daily_cols = [f"Uncertainty_d_{i+1}" for i in range(len(daily_df.columns))]
                cfg["uncertainty_daily"] = pd.DataFrame(unc_d_val, index=daily_df.index, columns=unc_daily_cols)
            cfg["hourly_columns"] = st_cols
            cfg["daily_columns"] = lt_cols

            common_idx = base_df.index
            if hourly_df is not None:
                common_idx = common_idx.intersection(hourly_df.index)
            if daily_df is not None:
                common_idx = common_idx.intersection(daily_df.index)
            if not common_idx.empty:
                base_df = base_df.loc[common_idx]
                if hourly_df is not None:
                    hourly_df = hourly_df.loc[common_idx]
                if daily_df is not None:
                    daily_df = daily_df.loc[common_idx]

        result = self._strategy_plugin.evaluate_candidate(plugin_params, base_df, hourly_df, daily_df, cfg)

        if isinstance(result, tuple) and len(result) == 2:
            profit, stats = result
        elif isinstance(result, tuple) and len(result) == 1:
            profit, stats = result[0], {}
        else:
            profit, stats = result, {}

        print(
            f"[EVALUATE][Epoch {self._current_epoch}/{self._num_generations}] Candidate => "
            f"Profit: {profit:.2f}, "
            f"Trades: {stats.get('num_trades', 0)}, "
            f"Win%: {stats.get('win_pct', 0):.1f}, "
            f"MaxDD: {stats.get('max_dd', 0):.2f}, "
            f"Sharpe: {stats.get('sharpe', 0):.2f}, "
            f"Early%: {stats.get('early_close_pct', 0):.1f}, "
            f"TP%: {stats.get('tp_close_pct', 0):.1f}, "
            f"SL%: {stats.get('sl_close_pct', 0):.1f}"
        )
        return profit, stats

    def run_optimizer(
        self,
        strategy_plugin,
        base_data,
        hourly_predictions,
        daily_predictions,
        config,
        base_val=None,
        hourly_val=None,
        daily_val=None,
        base_test=None,
        hourly_test=None,
        daily_test=None,
    ):
        self.init_optimizer(strategy_plugin, base_data, hourly_predictions, daily_predictions, config)

        optimizable_params = strategy_plugin.get_optimizable_params()
        num_params = len(optimizable_params)
        pred_params = [
            ("short_term_max_horizon", 2, 48),
            ("short_term_num_predictions", 2, 48),
            ("long_term_max_horizon", 24, 144),
            ("long_term_num_predictions", 24, 144),
        ]
        optimizable_params = optimizable_params + pred_params
        self._plugin_param_count = num_params
        self._base_param_bounds = list(optimizable_params[:num_params])

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

        toolbox.register("individual", lambda: creator.Individual([random_attr(p) for p in optimizable_params]))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _mp_safe_evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        random.seed(42)
        population_size = config.get("population_size", self.params.get("population_size", 20))
        num_generations = config.get("num_generations", self.params.get("num_generations", 50))
        cxpb = config.get("crossover_probability", self.params.get("crossover_probability", 0.5))
        mutpb = config.get("mutation_probability", self.params.get("mutation_probability", 0.2))

        print("Starting Genetic Algorithm Optimization")
        disable_mp = config.get("disable_multiprocessing", self.params.get("disable_multiprocessing", True))
        pool = None
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

        patience = int(config.get("patience", self.params.get("patience", 0)) or 0)
        best_val_profit = None
        epochs_without_improve = 0
        best_ind_overall = None
        best_ind_train_profit = None
        best_ind_val_profit = None
        best_ind_genome_snapshot = None

        for gen in range(1, num_generations):
            self._current_epoch = gen + 1
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
            print(
                f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits) / len(fits):.2f}"
            )

            if base_val is not None and hourly_val is not None and daily_val is not None:
                prev_base, prev_hourly, prev_daily = self._base_data, self._hourly_predictions, self._daily_predictions
                self._base_data, self._hourly_predictions, self._daily_predictions = base_val, hourly_val, daily_val
                val_profit, _ = self.evaluate_individual(gen_best_ind)
                self._base_data, self._hourly_predictions, self._daily_predictions = (
                    prev_base,
                    prev_hourly,
                    prev_daily,
                )
                print(f"  Validation Profit (gen {gen}): {val_profit:.2f}")

                improved = (best_val_profit is None) or (val_profit > best_val_profit)
                if improved:
                    best_val_profit = val_profit
                    epochs_without_improve = 0
                    best_ind_overall = gen_best_ind
                    best_ind_train_profit = gen_best_profit
                    best_ind_val_profit = val_profit
                    best_ind_genome_snapshot = list(gen_best_ind)
                else:
                    epochs_without_improve += 1
                    print(
                        f"  No validation improvement. Patience counter: {epochs_without_improve}/{patience}"
                    )

                best_pf_str = f"{best_val_profit:.2f}" if best_val_profit is not None else "N/A"
                print(f"[Epoch {gen}] LastBestProfit={best_pf_str} | Patience={epochs_without_improve}/{patience}")

                if patience > 0 and epochs_without_improve >= patience and not improved:
                    print("Early stopping triggered: validation profit did not improve within patience.")
                    break

            else:
                if best_ind_overall is None or gen_best_profit > (best_ind_train_profit or -1e9):
                    best_ind_overall = gen_best_ind
                    best_ind_train_profit = gen_best_profit

                best_train_str = f"{best_ind_train_profit:.2f}" if best_ind_train_profit is not None else "N/A"
                print(f"[Epoch {gen}] LastBestProfit={best_train_str} | Patience={0}/{patience}")

        if best_ind_overall is None:
            best_ind = tools.selBest(population, 1)[0]
            best_ind_train_profit = best_ind.fitness.values[0]
            best_ind_genome_snapshot = list(best_ind)
        else:
            if best_ind_genome_snapshot is not None:
                best_ind = creator.Individual(best_ind_genome_snapshot)
            else:
                best_ind = best_ind_overall

        best_params_raw = {name: best_ind[i] for i, (name, _, _) in enumerate(optimizable_params)}

        def _ri(value):
            try:
                return int(round(float(value)))
            except Exception:
                return int(value)

        best_params = dict(best_params_raw)
        if "short_term_max_horizon" in best_params:
            st_max = _ri(best_params["short_term_max_horizon"])
            st_max = max(2, min(48, st_max))
            best_params["short_term_max_horizon"] = st_max
        if "short_term_num_predictions" in best_params:
            st_n = _ri(best_params["short_term_num_predictions"])
            st_n = max(2, min(48, st_n))
            if "short_term_max_horizon" in best_params:
                st_n = min(st_n, best_params["short_term_max_horizon"])
            best_params["short_term_num_predictions"] = st_n
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
            if isinstance(val, int):
                print(f"  {name} = {val}")
            else:
                print(f"  {name} = {float(val):.4f}")

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

        if pool is not None:
            pool.close()

        if config.get("save_config"):
            try:
                with open(config["save_config"], "w") as f:
                    json.dump(best_params, f, indent=4, default=str)
                print(f"Best parameters saved to {config['save_config']}.")
            except Exception as e:
                print(f"Failed to save best parameters to {config['save_config']}: {e}")

        train_profit, train_stats = self.evaluate_individual(best_ind)

        val_profit, val_stats = None, {}
        if base_val is not None and hourly_val is not None and daily_val is not None:
            prev_base, prev_hourly, prev_daily = self._base_data, self._hourly_predictions, self._daily_predictions
            self._base_data, self._hourly_predictions, self._daily_predictions = base_val, hourly_val, daily_val
            val_profit, val_stats = self.evaluate_individual(best_ind)
            self._base_data, self._hourly_predictions, self._daily_predictions = (
                prev_base,
                prev_hourly,
                prev_daily,
            )

        test_profit, test_stats = None, {}
        if base_test is not None and hourly_test is not None and daily_test is not None:
            prev_base, prev_hourly, prev_daily = self._base_data, self._hourly_predictions, self._daily_predictions
            self._base_data, self._hourly_predictions, self._daily_predictions = base_test, hourly_test, daily_test
            test_profit, test_stats = self.evaluate_individual(best_ind)
            self._base_data, self._hourly_predictions, self._daily_predictions = (
                prev_base,
                prev_hourly,
                prev_daily,
            )

        def fmt_stats(stat_dict):
            if not stat_dict:
                return "-"
            return (
                f"Trades={stat_dict.get('num_trades', 0)}, Win%={stat_dict.get('win_pct', 0):.1f}, "
                f"MaxDD={stat_dict.get('max_dd', 0):.2f}, Sharpe={stat_dict.get('sharpe', 0):.2f}, "
                f"Early%={stat_dict.get('early_close_pct', 0):.1f}, TP%={stat_dict.get('tp_close_pct', 0):.1f}, SL%={stat_dict.get('sl_close_pct', 0):.1f}"
            )

        print("\nChampion evaluation summary:")
        st_cols_sel = []
        lt_cols_sel = []
        try:
            st_max = int(best_params.get("short_term_max_horizon")) if "short_term_max_horizon" in best_params else None
            st_n = int(best_params.get("short_term_num_predictions")) if "short_term_num_predictions" in best_params else None
            lt_max = int(best_params.get("long_term_max_horizon")) if "long_term_max_horizon" in best_params else None
            lt_n = int(best_params.get("long_term_num_predictions")) if "long_term_num_predictions" in best_params else None
            if st_max and st_n:
                st_cols_sel = [f"Prediction_H{o}" for o in _compute_uniform_offsets(st_max, st_n)]
            if lt_max and lt_n:
                lt_cols_sel = [f"Prediction_H{o}" for o in _compute_uniform_offsets(lt_max, lt_n)]
        except Exception:
            pass

        train_mae, train_naive_mae = _aggregate_mae_and_naive_for_sets(
            self._base_data,
            [
                (self._hourly_predictions, st_cols_sel),
                (self._daily_predictions, lt_cols_sel),
            ],
        )
        print(
            f"  Train:       Profit={train_profit:.2f} | {fmt_stats(train_stats)}"
            + (f" | MAE={train_mae:.6f} | NaiveMAE={train_naive_mae:.6f}" if train_mae is not None else "")
        )

        if val_profit is not None:
            prev_base, prev_hourly, prev_daily = self._base_data, self._hourly_predictions, self._daily_predictions
            self._base_data, self._hourly_predictions, self._daily_predictions = base_val, hourly_val, daily_val
            val_mae, val_naive_mae = _aggregate_mae_and_naive_for_sets(
                self._base_data,
                [
                    (self._hourly_predictions, st_cols_sel),
                    (self._daily_predictions, lt_cols_sel),
                ],
            )
            self._base_data, self._hourly_predictions, self._daily_predictions = (
                prev_base,
                prev_hourly,
                prev_daily,
            )
            print(
                f"  Validation:  Profit={val_profit:.2f} | {fmt_stats(val_stats)}"
                + (f" | MAE={val_mae:.6f} | NaiveMAE={val_naive_mae:.6f}" if val_mae is not None else "")
            )

        if test_profit is not None:
            prev_base, prev_hourly, prev_daily = self._base_data, self._hourly_predictions, self._daily_predictions
            self._base_data, self._hourly_predictions, self._daily_predictions = base_test, hourly_test, daily_test
            test_mae, test_naive_mae = _aggregate_mae_and_naive_for_sets(
                self._base_data,
                [
                    (self._hourly_predictions, st_cols_sel),
                    (self._daily_predictions, lt_cols_sel),
                ],
            )
            self._base_data, self._hourly_predictions, self._daily_predictions = (
                prev_base,
                prev_hourly,
                prev_daily,
            )
            print(
                f"  Test:        Profit={test_profit:.2f} | {fmt_stats(test_stats)}"
                + (f" | MAE={test_mae:.6f} | NaiveMAE={test_naive_mae:.6f}" if test_mae is not None else "")
            )

        result_payload = {
            "best_parameters": best_params,
            "profit": train_profit,
            "stats": train_stats,
            "validation_profit": val_profit,
            "validation_stats": val_stats,
            "test_profit": test_profit,
            "test_stats": test_stats,
            "train_mae": train_mae,
            "train_naive_mae": train_naive_mae,
        }
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
