import json
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import neat
from neat import nn
import numpy as np

from app.optimizer import (
    _aggregate_mae_and_naive_for_sets,
)


class _EarlyStopException(Exception):
    """Internal helper used to exit the NEAT loop when patience is exhausted."""


class Plugin:
    """Optimizer plugin that drives the Heuristic Strategy via NEAT."""

    plugin_params = {
        "population_size": 150,
        "max_generations": 80,
        "fitness_threshold": 2500.0,
        "elitism": 2,
        "survival_threshold": 0.2,
        "stagnation_limit": 15,
        "weight_mutation_power": 2.5,
        "bias_mutation_power": 0.5,
        "compatibility_threshold": 3.0,
        "activation_default": "tanh",
        "aggregation_default": "sum",
    "patience": 10,
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self._strategy_plugin = None
        self._base_data = None
        self._hourly_predictions = None
        self._daily_predictions = None
        self._config = {}
        self._feature_vector = []
        self._feature_names = []
        self._optimizable_params: List[Tuple[str, float, float]] = []
        self._num_generations = 0
        self._current_epoch = 1
        self._last_generation_fitness: List[float] = []
        self._last_genome_records: Dict[int, Dict] = {}
        self._last_generation_best_key: Optional[int] = None
        self._best_record: Optional[Dict] = None
        self._best_val_profit: Optional[float] = None
        self._epochs_without_improve = 0
        self._patience = int(self.params.get("patience", 10))
        self._validation_frames: Optional[Tuple] = None
        self._test_frames: Optional[Tuple] = None
        self._best_genome_snapshot = None

    # ------------------------------------------------------------------
    # Shared plugin contract helpers
    # ------------------------------------------------------------------
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return dict(self.params)

    # ------------------------------------------------------------------
    # Core NEAT helpers
    # ------------------------------------------------------------------
    def init_optimizer(self, strategy_plugin, base_data, hourly_predictions, daily_predictions, config):
        self._strategy_plugin = strategy_plugin
        self._base_data = base_data
        self._hourly_predictions = hourly_predictions
        self._daily_predictions = daily_predictions
        self._config = dict(config) if config is not None else {}
        self._feature_vector, self._feature_names = self._build_feature_vector()
        self._num_generations = int(self.params.get("max_generations", 80))
        self._current_epoch = 1
        self._last_generation_fitness = []
        self._last_genome_records = {}
        self._last_generation_best_key = None
        self._best_record = None
        self._best_val_profit = None
        self._epochs_without_improve = 0
        self._patience = int(self._config.get("patience", self.params.get("patience", 10)) or 0)
        self._best_genome_snapshot = None

    def evaluate_individual(self, candidate: List[float]):
        if self._strategy_plugin is None:
            print("[EVALUATE] ERROR: Strategy plugin is not set!")
            return -1e6, {}

        print(
            f"[EVALUATE][Epoch {self._current_epoch}/{self._num_generations}] Evaluating candidate (genome): {candidate}"
        )

        cfg = dict(self._config)
        result = self._strategy_plugin.evaluate_candidate(
            candidate,
            self._base_data,
            self._hourly_predictions,
            self._daily_predictions,
            cfg,
        )

        if isinstance(result, tuple):
            if len(result) == 2:
                profit, stats = result
            else:
                profit, stats = result[0], {}
        else:
            profit, stats = result, {}

        print(
            f"[EVALUATE][Epoch {self._current_epoch}/{self._num_generations}] Candidate => "
            f"Profit: {profit:.2f}, "
            f"Trades: {stats.get('num_trades', 0)}, "
            f"Win%: {stats.get('win_pct', 0):.1f}, "
            f"MaxDD: {stats.get('max_dd', 0):.2f}, "
            f"Sharpe: {stats.get('sharpe', 0):.2f}"
        )
        return profit, stats

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_feature_vector(self) -> Tuple[List[float], List[str]]:
        names = [
            "st_mae_ratio",
            "lt_mae_ratio",
            "volatility",
            "trend_slope",
            "prediction_spread",
            "uncertainty_ratio",
            "drawdown_ratio",
            "momentum_osc",
            "hour_sin",
            "hour_cos",
        ]

        def _safe(val, fallback=0.0):
            if val is None or np.isnan(val):
                return fallback
            return float(val)

        base_df = self._base_data
        hourly = self._hourly_predictions
        daily = self._daily_predictions

        st_cols = list(getattr(hourly, "columns", [])) if hourly is not None else []
        lt_cols = list(getattr(daily, "columns", [])) if daily is not None else []

        st_mae, st_naive = _aggregate_mae_and_naive_for_sets(
            base_df,
            [(hourly, st_cols)],
        )
        lt_mae, lt_naive = _aggregate_mae_and_naive_for_sets(
            base_df,
            [(daily, lt_cols)],
        )

        def _ratio(a, b):
            if a is None or b in (None, 0):
                return 1.0
            if b == 0:
                return 1.0
            return float(a) / float(b)

        st_ratio = _ratio(st_mae, st_naive)
        lt_ratio = _ratio(lt_mae, lt_naive)

        volatility = 0.0
        trend_slope = 0.0
        prediction_spread = 0.0
        uncertainty_ratio = 1.0
        drawdown_ratio = 0.0
        momentum_osc = 0.0
        hour_sin = 0.0
        hour_cos = 0.0

        if base_df is not None and not base_df.empty:
            close = base_df["CLOSE"] if "CLOSE" in base_df.columns else base_df.iloc[:, 0]
            pct = close.pct_change().rolling(24).std()
            volatility = _safe(pct.mean(), 0.0)

            window = close.tail(48)
            if len(window) >= 2:
                x = np.arange(len(window))
                slope, _ = np.polyfit(x, window.values, 1)
                trend_slope = float(slope)

            recent = close.tail(72)
            if not recent.empty:
                max_close = recent.max()
                min_close = recent.min()
                drawdown_ratio = _ratio(max_close - min_close, max_close if max_close else 1.0)

            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            momentum = ema_fast - ema_slow
            momentum_osc = _safe(momentum.tail(1).iloc[0], 0.0)

            last_dt = base_df.index[-1]
            if hasattr(last_dt, "hour"):
                hour = last_dt.hour
                hour_sin = math.sin(2 * math.pi * hour / 24.0)
                hour_cos = math.cos(2 * math.pi * hour / 24.0)

        if hourly is not None and daily is not None and not hourly.empty and not daily.empty:
            aligned_idx = hourly.index.intersection(daily.index)
            if not aligned_idx.empty:
                h_vals = hourly.loc[aligned_idx]
                d_vals = daily.loc[aligned_idx]
                h_mean = h_vals.mean(axis=1)
                d_mean = d_vals.mean(axis=1)
                prediction_spread = _safe(np.mean(np.abs(h_mean.values - d_mean.values)), 0.0)

        u_hourly = self._config.get("uncertainty_hourly")
        u_daily = self._config.get("uncertainty_daily")
        if u_hourly is not None and not u_hourly.empty and u_daily is not None and not u_daily.empty:
            uh = np.abs(u_hourly.values).mean()
            ud = np.abs(u_daily.values).mean()
            uncertainty_ratio = _ratio(ud, uh if uh else 1.0)

        vector = [
            st_ratio,
            lt_ratio,
            volatility,
            trend_slope,
            prediction_spread,
            uncertainty_ratio,
            drawdown_ratio,
            momentum_osc,
            hour_sin,
            hour_cos,
        ]
        vector = [0.0 if np.isnan(v) else float(v) for v in vector]
        if not vector:
            vector = [0.0]
            names = ["bias"]
        return vector, names

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _decode_output_vector(self, output: List[float]) -> List[float]:
        decoded = []
        for value, (name, low, high) in zip(output, self._optimizable_params):
            span = high - low
            mapped = low + span * self._sigmoid(value)
            decoded.append(mapped)
        # Ensure RR thresholds are valid
        if len(decoded) >= 5:
            lower_rr = decoded[3]
            upper_rr = max(decoded[4], lower_rr + 0.1)
            decoded[4] = upper_rr
        return decoded

    def _build_neat_config(self, num_inputs: int) -> neat.Config:
        pop_size = int(self.params.get('population_size', 150))

        cfg_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.params.get('fitness_threshold', 2500.0)}
population_size       = {pop_size}
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = {self.params.get('activation_default', 'tanh')}
activation_mutate_rate  = 0.0
activation_options      = sigmoid tanh relu
aggregation_default     = {self.params.get('aggregation_default', 'sum')}
aggregation_mutate_rate = 0.0
aggregation_options     = sum mean max
bias_init_mean          = 0.0
bias_init_stdev         = {float(self.params.get('bias_mutation_power', 0.5))}
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = {float(self.params.get('bias_mutation_power', 0.5))}
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {len(self._optimizable_params)}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = {float(self.params.get('weight_mutation_power', 2.5))}
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = {float(self.params.get('compatibility_threshold', 3.0))}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {int(self.params.get('stagnation_limit', 15))}
species_elitism      = {int(self.params.get('elitism', 2))}

[DefaultReproduction]
elitism            = {int(self.params.get('elitism', 2))}
survival_threshold = {float(self.params.get('survival_threshold', 0.2))}
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(cfg_text)
            tmp_path = tmp.name
        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                tmp_path,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return config

    # ------------------------------------------------------------------
    # Generation lifecycle helpers
    # ------------------------------------------------------------------
    def _eval_genomes(self, genomes, neat_config):
        self._last_generation_fitness = []
        self._last_genome_records = {}
        self._last_generation_best_key = None

        for genome_id, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, neat_config)
            outputs = net.activate(self._feature_vector)
            candidate = self._decode_output_vector(outputs)
            profit, stats = self.evaluate_individual(candidate)
            genome.fitness = profit
            record = {
                "candidate": candidate,
                "train_profit": profit,
                "train_stats": stats,
                "genome": genome,
            }
            self._last_generation_fitness.append(profit)
            self._last_genome_records[genome_id] = record
            if self._last_generation_best_key is None or profit > self._last_genome_records[
                self._last_generation_best_key
            ]["train_profit"]:
                self._last_generation_best_key = genome_id

    def _evaluate_on_dataset(self, candidate, dataset_tuple):
        if dataset_tuple is None:
            return None, {}
        base_df, hourly_df, daily_df = dataset_tuple
        if base_df is None or hourly_df is None or daily_df is None:
            return None, {}
        prev = (self._base_data, self._hourly_predictions, self._daily_predictions)
        self._base_data, self._hourly_predictions, self._daily_predictions = dataset_tuple
        profit, stats = self.evaluate_individual(candidate)
        self._base_data, self._hourly_predictions, self._daily_predictions = prev
        return profit, stats

    def _serialize_genome(self, genome):
        if genome is None:
            return {}
        nodes = {}
        for key, node in genome.nodes.items():
            nodes[str(key)] = {
                "bias": getattr(node, "bias", 0.0),
                "response": getattr(node, "response", 0.0),
                "activation": getattr(node, "activation", ""),
                "aggregation": getattr(node, "aggregation", ""),
            }
        connections = {}
        for (in_key, out_key), conn in genome.connections.items():
            connections[f"{in_key}->{out_key}"] = {
                "weight": getattr(conn, "weight", 0.0),
                "enabled": getattr(conn, "enabled", True),
            }
        return {"nodes": nodes, "connections": connections}

    def _handle_generation_complete(self, generation: int, max_generations: int):
        if not self._last_generation_fitness:
            print(f"Generation {generation}: No fitness evaluations produced.")
            return False

        max_profit = max(self._last_generation_fitness)
        avg_profit = sum(self._last_generation_fitness) / len(self._last_generation_fitness)
        print(
            f"Generation {generation}: Max Profit = {max_profit:.2f}, Avg Profit = {avg_profit:.2f}"
        )

        best_key = self._last_generation_best_key
        record = self._last_genome_records.get(best_key)
        if record is None:
            return False

        val_profit = None
        val_stats = {}
        if self._validation_frames is not None:
            val_profit, val_stats = self._evaluate_on_dataset(record["candidate"], self._validation_frames)
            if val_profit is not None:
                print(f"  Validation Profit (gen {generation}): {val_profit:.2f}")

        improved = False
        if self._validation_frames is not None and val_profit is not None:
            improved = (self._best_val_profit is None) or (val_profit > self._best_val_profit)
            if improved:
                self._best_val_profit = val_profit
                self._epochs_without_improve = 0
                self._best_record = {
                    **record,
                    "val_profit": val_profit,
                    "val_stats": val_stats,
                }
                self._best_genome_snapshot = self._serialize_genome(record["genome"])
            else:
                self._epochs_without_improve += 1
                print(
                    f"  No validation improvement. Patience counter: {self._epochs_without_improve}/{self._patience}"
                )
            best_pf_str = f"{self._best_val_profit:.2f}" if self._best_val_profit is not None else "N/A"
            print(f"[Epoch {generation}] LastBestProfit={best_pf_str} | Patience={self._epochs_without_improve}/{self._patience}")
            if self._patience > 0 and self._epochs_without_improve >= self._patience and not improved:
                print("Early stopping triggered: validation profit did not improve within patience.")
                raise _EarlyStopException()
        else:
            if self._best_record is None or record["train_profit"] > self._best_record.get("train_profit", -1e9):
                self._best_record = record
                self._best_genome_snapshot = self._serialize_genome(record["genome"])
            best_train = self._best_record.get("train_profit") if self._best_record else None
            best_train_str = f"{best_train:.2f}" if best_train is not None else "N/A"
            print(f"[Epoch {generation}] LastBestProfit={best_train_str} | Patience={0}/{self._patience}")

        train_profit = record["train_profit"]
        train_stats = record["train_stats"] or {}
        val_sharpe = val_stats.get("sharpe", 0.0) if val_stats else 0.0
        print(
            f"[NEAT] Epoch {generation} Champion => Train Profit={train_profit:.2f}, "
            f"Sharpe={train_stats.get('sharpe', 0.0):.2f}"
        )
        if val_profit is not None:
            print(
                f"[NEAT] Epoch {generation} Validation => Profit={val_profit:.2f}, "
                f"Sharpe={val_sharpe:.2f}"
            )

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        self._validation_frames = None
        self._test_frames = None
        if base_val is not None and hourly_val is not None and daily_val is not None:
            self._validation_frames = (base_val, hourly_val, daily_val)
        if base_test is not None and hourly_test is not None and daily_test is not None:
            self._test_frames = (base_test, hourly_test, daily_test)

        self._optimizable_params = strategy_plugin.get_optimizable_params()
        total_params = len(self._optimizable_params)
        print(f"Optimizable Parameters ({total_params}):")
        for name, low, high in self._optimizable_params:
            print(f"  {name}: [{low}, {high}]")

        neat_config = self._build_neat_config(len(self._feature_vector))
        population = neat.Population(neat_config)

        max_generations = int(self.params.get("max_generations", 80))
        self._num_generations = max_generations
        try:
            for gen in range(1, max_generations + 1):
                self._current_epoch = gen
                population.run(self._eval_genomes, 1)
                self._handle_generation_complete(gen, max_generations)
        except _EarlyStopException:
            pass

        if self._best_record is None:
            if self._last_generation_best_key is not None:
                self._best_record = self._last_genome_records.get(self._last_generation_best_key)

        if self._best_record is None:
            raise RuntimeError("NEAT failed to evaluate any candidates.")

        best_candidate = self._best_record["candidate"]
        best_params = {
            name: best_candidate[idx]
            for idx, (name, _, _) in enumerate(self._optimizable_params)
        }

        print("[OPTIMIZATION] Best parameter set found:")
        for k, v in best_params.items():
            print(f"  {k} = {float(v):.4f}")

        if config.get("save_config"):
            try:
                with open(config["save_config"], "w") as f:
                    json.dump(best_params, f, indent=4, default=str)
                print(f"Best parameters saved to {config['save_config']}.")
            except Exception as e:
                print(f"Failed to save best parameters to {config['save_config']}: {e}")

        train_profit = self._best_record.get("train_profit")
        train_profit = 0.0 if train_profit is None else float(train_profit)
        train_stats = self._best_record.get("train_stats", {})
        val_profit = self._best_record.get("val_profit")
        val_profit = None if val_profit is None else float(val_profit)
        val_stats = self._best_record.get("val_stats", {})
        test_profit = None
        test_stats = {}
        if self._test_frames is not None:
            test_profit, test_stats = self._evaluate_on_dataset(best_candidate, self._test_frames)
            if test_profit is not None:
                test_profit = float(test_profit)

        # Compute MAE metrics for summary just like GA plugin
        st_cols_sel = list(getattr(self._hourly_predictions, "columns", []))
        lt_cols_sel = list(getattr(self._daily_predictions, "columns", []))

        train_mae, train_naive_mae = _aggregate_mae_and_naive_for_sets(
            self._base_data,
            [
                (self._hourly_predictions, st_cols_sel),
                (self._daily_predictions, lt_cols_sel),
            ],
        )

        def fmt_stats(stat_dict):
            if not stat_dict:
                return "-"
            return (
                f"Trades={stat_dict.get('num_trades', 0)}, Win%={stat_dict.get('win_pct', 0):.1f}, "
                f"MaxDD={stat_dict.get('max_dd', 0):.2f}, Sharpe={stat_dict.get('sharpe', 0):.2f}"
            )

        print("\nChampion evaluation summary:")
        print(
            f"  Train:       Profit={train_profit:.2f} | {fmt_stats(train_stats)}"
            + (f" | MAE={train_mae:.6f} | NaiveMAE={train_naive_mae:.6f}" if train_mae is not None else "")
        )

        if val_profit is not None:
            prev_frames = (self._base_data, self._hourly_predictions, self._daily_predictions)
            self._base_data, self._hourly_predictions, self._daily_predictions = self._validation_frames
            val_mae, val_naive_mae = _aggregate_mae_and_naive_for_sets(
                self._base_data,
                [
                    (self._hourly_predictions, st_cols_sel),
                    (self._daily_predictions, lt_cols_sel),
                ],
            )
            self._base_data, self._hourly_predictions, self._daily_predictions = prev_frames
            print(
                f"  Validation:  Profit={val_profit:.2f} | {fmt_stats(val_stats)}"
                + (f" | MAE={val_mae:.6f} | NaiveMAE={val_naive_mae:.6f}" if val_mae is not None else "")
            )
        else:
            val_mae = val_naive_mae = None

        test_mae = test_naive_mae = None
        if test_profit is not None:
            prev_frames = (self._base_data, self._hourly_predictions, self._daily_predictions)
            self._base_data, self._hourly_predictions, self._daily_predictions = self._test_frames
            test_mae, test_naive_mae = _aggregate_mae_and_naive_for_sets(
                self._base_data,
                [
                    (self._hourly_predictions, st_cols_sel),
                    (self._daily_predictions, lt_cols_sel),
                ],
            )
            self._base_data, self._hourly_predictions, self._daily_predictions = prev_frames
            print(
                f"  Test:        Profit={test_profit:.2f} | {fmt_stats(test_stats)}"
                + (f" | MAE={test_mae:.6f} | NaiveMAE={test_naive_mae:.6f}" if test_mae is not None else "")
            )

        result_payload = {
            "best_parameters": {
                **best_params,
                "neat_champion_genome": self._best_genome_snapshot,
            },
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
            result_payload.update(
                {
                    "validation_mae": val_mae,
                    "validation_naive_mae": val_naive_mae,
                }
            )
        if test_profit is not None:
            result_payload.update(
                {
                    "test_mae": test_mae,
                    "test_naive_mae": test_naive_mae,
                }
            )

        return result_payload
