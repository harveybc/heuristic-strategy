import json
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import neat
from neat import nn
import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from app.optimizer import (
    _aggregate_mae_and_naive_for_sets,
)


class _EarlyStopException(Exception):
    """Internal helper used to exit the NEAT loop when patience is exhausted."""


class Plugin:
    """Optimizer plugin that drives the Heuristic Strategy via NEAT."""

    plugin_params = {
        "population_size": 35,
        "max_generations": 2500,
        "fitness_threshold": 1e9,
        "elitism": 2,
        "survival_threshold": 0.2,
        "min_species_size": 2,
        "stagnation_limit": 7,
        "weight_mutation_power": 2.5,
        "bias_mutation_power": 0.5,
        "compatibility_threshold": 3.4,
        "activation_default": "identity",
        "activation_functions": ["identity", "sigmoid", "tanh", "relu"],
        "aggregation_default": "sum",
        "no_fitness_termination": False,
        "single_structural_mutation": True,
        "structural_mutation_surer": "default",
        "patience": 30,
        "show_progress_bar": True,
        "target_species_count": 0,
        "compatibility_adjust_rate": 0.1,
        "enable_neat_default_reporter": True,
        "validation_improvement_epsilon": 1e-9,
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
        self._current_population_index = 0
        self._current_population_total = 0
        self._use_progress_bar = True
        self._eval_context = "train"
        self._active_uncertainty_hourly = None
        self._active_uncertainty_daily = None
        self._current_candidate_net = None
        self._neat_config = None
        self._stdout_reporter = None
        self._stats_reporter = None

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
        self._use_progress_bar = bool(
            self._config.get("show_progress_bar", self.params.get("show_progress_bar", True))
        )
        self._eval_context = "train"
        self._active_uncertainty_hourly = None
        self._active_uncertainty_daily = None
        self._current_candidate_net = None
        self._neat_config = None
        self._stdout_reporter = None
        self._stats_reporter = None

    def evaluate_individual(self, candidate: List[float]):
        if self._strategy_plugin is None:
            print("[EVALUATE] ERROR: Strategy plugin is not set!")
            return -1e6, {}

        progress_note = ""
        if self._current_population_total:
            progress_note = f"[{self._current_population_index}/{self._current_population_total}] "

        patience_note = self._format_patience_status()
        context = getattr(self, "_eval_context", "train") or "train"
        context_tag = context.upper()

        print(
            f"[EVALUATE][{context_tag}][Epoch {self._current_epoch}/{self._num_generations}] {progress_note}"
            f"Evaluating candidate (genome): {candidate} | {patience_note}"
        )

        cfg = dict(self._config)
        candidate_net = getattr(self, "_current_candidate_net", None)
        if candidate_net is not None:
            cfg["neat_network"] = candidate_net
            cfg["neat_feature_names"] = list(self._feature_names)
            cfg["neat_param_bounds"] = list(self._optimizable_params)
        if self._active_uncertainty_hourly is not None:
            cfg["uncertainty_hourly"] = self._active_uncertainty_hourly
        if self._active_uncertainty_daily is not None:
            cfg["uncertainty_daily"] = self._active_uncertainty_daily
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
            f"[EVALUATE][{context_tag}][Epoch {self._current_epoch}/{self._num_generations}] Candidate => "
            f"Profit: {profit:.2f}, "
            f"Trades: {stats.get('num_trades', 0)}, "
            f"Win%: {stats.get('win_pct', 0):.1f}, "
            f"MaxDD: {stats.get('max_dd', 0):.2f}, "
            f"Sharpe: {stats.get('sharpe', 0):.2f} | {patience_note}"
        )
        return profit, stats

    def _format_patience_status(self) -> str:
        patience_limit = self._patience if isinstance(self._patience, int) else 0
        if patience_limit <= 0:
            return "Patience=off"
        best_pf = self._best_val_profit
        best_pf_str = f"{best_pf:.2f}" if best_pf is not None else "N/A"
        return f"Patience={self._epochs_without_improve}/{patience_limit} | BestVal={best_pf_str}"

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_feature_vector(self) -> Tuple[List[float], List[str]]:
        prediction_frame, column_order = self._build_prediction_frame()
        features: List[float] = []
        names: List[str] = []

        if prediction_frame is not None and column_order:
            target_idx = self._select_prediction_timestamp(prediction_frame.index)
            if target_idx is not None and target_idx in prediction_frame.index:
                row = prediction_frame.loc[target_idx]
            else:
                row = prediction_frame.iloc[-1]
                target_idx = prediction_frame.index[-1]

            for col in column_order:
                value = row.get(col, 0.0)
                if value is None:
                    value = 0.0
                else:
                    try:
                        if np.isnan(value):
                            value = 0.0
                    except TypeError:
                        pass
                try:
                    value = float(value)
                except Exception:
                    value = 0.0
                features.append(value)
                names.append(col)

            print(f"[NEAT][Features] Using prediction snapshot at {target_idx}")
        else:
            features = [0.0]
            names = ["prediction_bias"]

        positional = self._compute_positional_features()
        features.extend(positional.values())
        names.extend(positional.keys())
        return features, names

    def _build_prediction_frame(self) -> Tuple[Optional[pd.DataFrame], List[str]]:
        hourly = self._hourly_predictions
        daily = self._daily_predictions
        if hourly is None or hourly.empty or daily is None or daily.empty:
            return None, []

        renamed_h = {col: f"Prediction_h_{i + 1}" for i, col in enumerate(hourly.columns)}
        renamed_d = {col: f"Prediction_d_{i + 1}" for i, col in enumerate(daily.columns)}
        hourly_df = hourly.rename(columns=renamed_h).copy()
        daily_df = daily.rename(columns=renamed_d).copy()

        merged = hourly_df.join(daily_df, how="outer")
        context = getattr(self, "_eval_context", "train").upper()

        num_hourly = hourly_df.shape[1]
        num_daily = daily_df.shape[1]
        strategy_defaults = getattr(self._strategy_plugin, "params", {}) or {}
        default_short = strategy_defaults.get("default_uncertainty_short_term", 0.0)
        default_long = strategy_defaults.get("default_uncertainty_long_term", 0.0)

        uncertainty_hourly = self._prepare_uncertainty_frame(
            self._config.get("uncertainty_hourly"),
            num_hourly,
            "Uncertainty_h",
            default_short,
            hourly_df.index if not hourly_df.empty else merged.index,
        )
        uncertainty_daily = self._prepare_uncertainty_frame(
            self._config.get("uncertainty_daily"),
            num_daily,
            "Uncertainty_d",
            default_long,
            daily_df.index if not daily_df.empty else merged.index,
        )

        def _safe_join(current, extra, label):
            if extra is None or extra.empty:
                if context != "TRAIN":
                    print(f"[{context}][Uncertainty] {label} frame missing or empty; skipping join.")
                return current
            before_rows = len(current)
            joined = current.join(extra, how="left")
            if joined.empty and before_rows > 0:
                print(
                    f"[{context}][Uncertainty] Left-join with {label} dropped all rows. "
                    "Retrying without this uncertainty frame."
                )
                return current
            return joined

        merged = _safe_join(merged, uncertainty_hourly, "hourly")
        merged = _safe_join(merged, uncertainty_daily, "daily")

        if merged.empty:
            print(f"[{context}][Features] Prediction frame empty after merging uncertainties.")
            return None, []

        column_order: List[str] = list(hourly_df.columns) + list(daily_df.columns)
        if uncertainty_hourly is not None:
            column_order += list(uncertainty_hourly.columns)
        if uncertainty_daily is not None:
            column_order += list(uncertainty_daily.columns)

        merged = merged.sort_index()
        merged = merged[column_order]
        self._active_uncertainty_hourly = uncertainty_hourly
        self._active_uncertainty_daily = uncertainty_daily
        return merged, column_order

    def _prepare_uncertainty_frame(
        self,
        frame: Optional[pd.DataFrame],
        expected_cols: int,
        prefix: str,
        default_value: float,
        reference_index,
    ) -> Optional[pd.DataFrame]:
        if expected_cols == 0:
            return None

        columns = [f"{prefix}_{i + 1}" for i in range(expected_cols)]
        if frame is None or frame.empty:
            if reference_index is None or len(reference_index) == 0:
                return None
            data = np.full((len(reference_index), expected_cols), default_value, dtype=float)
            return pd.DataFrame(data, index=reference_index, columns=columns)

        df = frame.copy()
        df = df.iloc[:, :expected_cols]
        current_cols = df.shape[1]
        if current_cols < expected_cols:
            pad_count = expected_cols - current_cols
            pad_columns = [f"__pad_{idx}" for idx in range(pad_count)]
            pad_df = pd.DataFrame(
                default_value,
                index=df.index,
                columns=pad_columns,
            )
            df = pd.concat([df, pad_df], axis=1)
            df = df.iloc[:, :expected_cols]
        df.columns = columns
        if reference_index is not None:
            try:
                reference_index = pd.Index(reference_index)
                df = df.reindex(reference_index)
            except Exception:
                df = df.reindex(reference_index, fill_value=default_value)
            if df.isnull().values.any():
                df = df.fillna(default_value)
        return df

    def _select_prediction_timestamp(self, prediction_index) -> Optional[pd.Timestamp]:
        if prediction_index is None or len(prediction_index) == 0:
            return None

        base_df = self._base_data
        if base_df is not None and not base_df.empty:
            last_dt = base_df.index[-1]
            if hasattr(last_dt, "to_pydatetime"):
                last_dt = last_dt.to_pydatetime()
            target_hour = last_dt.replace(minute=0, second=0, microsecond=0)
            if target_hour in prediction_index:
                return target_hour
            earlier = prediction_index[prediction_index <= target_hour]
            if len(earlier) > 0:
                return earlier[-1]

        return prediction_index[-1]

    def _compute_positional_features(self) -> Dict[str, float]:
        defaults = {"hour_sin": 0.0, "hour_cos": 1.0, "dow_sin": 0.0, "dow_cos": 1.0}
        base_df = self._base_data
        if base_df is None or base_df.empty:
            return defaults

        dt = base_df.index[-1]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()

        hour = dt.hour + (dt.minute / 60.0)
        hour_angle = 2 * math.pi * hour / 24.0
        dow = dt.weekday()
        dow_angle = 2 * math.pi * dow / 7.0

        return {
            "hour_sin": math.sin(hour_angle),
            "hour_cos": math.cos(hour_angle),
            "dow_sin": math.sin(dow_angle),
            "dow_cos": math.cos(dow_angle),
        }

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
        def _bool_str(value):
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "false"}:
                    return normalized
            return "true" if bool(value) else "false"

        no_fitness_termination = _bool_str(self.params.get('no_fitness_termination', False))
        single_structural_mutation = _bool_str(self.params.get('single_structural_mutation', True))
        structural_mutation_surer = str(self.params.get('structural_mutation_surer', 'default')).strip().lower()
        if structural_mutation_surer not in {"true", "false", "default"}:
            structural_mutation_surer = "default"
        min_species_size = max(1, int(self.params.get('min_species_size', 2)))

        cfg_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.params.get('fitness_threshold', 2500.0)}
pop_size              = {pop_size}
no_fitness_termination = {no_fitness_termination}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = {self.params.get('activation_default', 'identity')}
activation_mutate_rate  = 0.1
activation_options      = identity sigmoid tanh relu
aggregation_default     = {self.params.get('aggregation_default', 'sum')}
aggregation_mutate_rate = 0.0
aggregation_options     = sum mean max
bias_init_mean          = 0.0
bias_init_stdev         = {float(self.params.get('bias_mutation_power', 0.5))}
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = {float(self.params.get('bias_mutation_power', 0.5))}
bias_mutate_rate        = 0.07
bias_replace_rate       = 0.01
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.2
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
feed_forward            = True
initial_connection      = fs_neat_hidden
node_add_prob           = 0.1
node_delete_prob        = 0.05
num_hidden              = 10
num_inputs              = {num_inputs}
num_outputs             = {len(self._optimizable_params)}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = {float(self.params.get('weight_mutation_power', 2.5))}
weight_mutate_rate      = 0.3
weight_replace_rate     = 0.01
single_structural_mutation = {single_structural_mutation}
structural_mutation_surer = {structural_mutation_surer}

[DefaultSpeciesSet]
compatibility_threshold = {float(self.params.get('compatibility_threshold', 3.0))}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {int(self.params.get('stagnation_limit', 15))}
species_elitism      = {int(self.params.get('elitism', 2))}

[DefaultReproduction]
elitism            = {int(self.params.get('elitism', 2))}
survival_threshold = {float(self.params.get('survival_threshold', 0.2))}
min_species_size   = {min_species_size}
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

        genome_list = list(genomes)
        total = len(genome_list)
        self._current_population_total = total
        progress_iter = genome_list
        progress_bar = None
        if self._use_progress_bar and total > 0:
            try:
                progress_bar = tqdm(
                    genome_list,
                    total=total,
                    desc=f"Epoch {self._current_epoch} eval",
                    unit="genome",
                    leave=False,
                )
                progress_iter = progress_bar
            except Exception:
                progress_bar = None

        for idx, (genome_id, genome) in enumerate(progress_iter, 1):
            self._eval_context = "train"
            self._current_population_index = idx
            net = nn.FeedForwardNetwork.create(genome, neat_config)
            self._current_candidate_net = net
            try:
                outputs = net.activate(self._feature_vector)
                candidate = self._decode_output_vector(outputs)
                profit, stats = self.evaluate_individual(candidate)
            finally:
                self._current_candidate_net = None
            if progress_bar is not None:
                progress_bar.set_postfix({"profit": f"{profit:.2f}"}, refresh=False)
            genome.fitness = profit
            record = {
                "candidate": candidate,
                "train_profit": profit,
                "train_stats": stats,
                "genome": genome,
                "net": net,
            }
            self._last_generation_fitness.append(profit)
            self._last_genome_records[genome_id] = record
            if self._last_generation_best_key is None or profit > self._last_genome_records[
                self._last_generation_best_key
            ]["train_profit"]:
                self._last_generation_best_key = genome_id

        if progress_bar is not None:
            progress_bar.close()
        self._current_population_index = 0
        self._current_population_total = 0

    def _evaluate_on_dataset(self, candidate, dataset_tuple, dataset_label="validation", genome=None, net=None):
        if dataset_tuple is None:
            return None, {}
        base_df, hourly_df, daily_df = dataset_tuple
        if base_df is None or hourly_df is None or daily_df is None:
            return None, {}

        prev = (
            self._base_data,
            self._hourly_predictions,
            self._daily_predictions,
            self._feature_vector,
            self._feature_names,
            getattr(self, "_eval_context", "train"),
            self._active_uncertainty_hourly,
            self._active_uncertainty_daily,
            getattr(self, "_current_candidate_net", None),
        )

        self._base_data, self._hourly_predictions, self._daily_predictions = dataset_tuple
        self._feature_vector, self._feature_names = self._build_feature_vector()

        context_label = (dataset_label or "validation").lower()
        self._eval_context = context_label

        if context_label in {"validation", "test"}:
            try:
                print(
                    f"[{context_label.upper()}][Dataset] base_rows={len(base_df)}, hourly_rows={len(hourly_df)}, "
                    f"daily_rows={len(daily_df)}, hourly_cols={hourly_df.shape[1]}, daily_cols={daily_df.shape[1]}"
                )
                idx_summary = (
                    f"{hourly_df.index.min()} -> {hourly_df.index.max()}"
                    if len(hourly_df.index) > 0
                    else "<empty>"
                )
                print(f"[{context_label.upper()}][Dataset] hourly_index_range={idx_summary}")
            except Exception as diag_err:
                print(f"[{context_label.upper()}][Dataset] Unable to summarize dataset: {diag_err}")

        validation_trades_requested = bool(
            context_label == "validation" and self._config.get("show_validation_trades")
        )
        prev_show_trades_marker = object()
        prev_show_trades = self._config.get("show_trades", prev_show_trades_marker)
        if validation_trades_requested:
            print("[VALIDATION] show_validation_trades enabled -> printing trades for validation dataset only.")
            self._config["show_trades"] = True

        candidate_net = net
        if candidate_net is None and genome is not None and self._neat_config is not None:
            try:
                candidate_net = nn.FeedForwardNetwork.create(genome, self._neat_config)
            except Exception as build_err:
                print(f"[{context_label.upper()}][NEAT] Failed to rebuild network for dataset eval: {build_err}")
                candidate_net = None
        self._current_candidate_net = candidate_net

        try:
            profit, stats = self.evaluate_individual(candidate)
        finally:
            (
                self._base_data,
                self._hourly_predictions,
                self._daily_predictions,
                self._feature_vector,
                self._feature_names,
            ) = prev[:5]
            self._eval_context = prev[5]
            self._active_uncertainty_hourly = prev[6]
            self._active_uncertainty_daily = prev[7]
            self._current_candidate_net = prev[8]

            if validation_trades_requested:
                if prev_show_trades is prev_show_trades_marker:
                    self._config.pop("show_trades", None)
                else:
                    self._config["show_trades"] = prev_show_trades

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
            val_profit, val_stats = self._evaluate_on_dataset(
                record["candidate"],
                self._validation_frames,
                dataset_label="validation",
                genome=record.get("genome"),
                net=record.get("net"),
            )
            if val_profit is not None:
                print(f"  Validation Profit (gen {generation}): {val_profit:.2f}")

        improved = False
        if self._validation_frames is not None and val_profit is not None:
            prev_best = self._best_val_profit
            prev_best_str = f"{prev_best:.2f}" if prev_best is not None else "N/A"
            epsilon = self._config.get(
                "validation_improvement_epsilon",
                self.params.get("validation_improvement_epsilon", 0.0),
            )
            try:
                epsilon = float(epsilon)
            except (TypeError, ValueError):
                epsilon = 0.0
            if epsilon < 0:
                epsilon = 0.0
            baseline = prev_best if prev_best is not None else float("-inf")
            improved = (prev_best is None) or (val_profit > (baseline + epsilon))
            print(
                f"[VALIDATION][Epoch {generation}] Profit={val_profit:.2f} | PrevBest={prev_best_str} | "
                f"Improved={'YES' if improved else 'NO'} | Epsilon={epsilon:.2g}"
            )
            if improved:
                self._best_val_profit = val_profit
                self._epochs_without_improve = 0
                self._best_record = {
                    **record,
                    "val_profit": val_profit,
                    "val_stats": val_stats,
                }
                self._best_genome_snapshot = self._serialize_genome(record["genome"])
                print(
                    f"[VALIDATION][Epoch {generation}] New BestVal={self._best_val_profit:.2f} (patience reset)"
                )
            else:
                self._epochs_without_improve += 1
                print(
                    f"  No validation improvement. Patience counter: {self._epochs_without_improve}/{self._patience}"
                )
            best_pf_str = f"{self._best_val_profit:.2f}" if self._best_val_profit is not None else "N/A"
            patience_limit = self._patience if self._patience else 0
            status = "IMPROVED" if improved else "WAITING"
            print(
                f"[EarlyStop][Epoch {generation}] Status={status} | ValProfit={val_profit:.2f} | "
                f"BestValProfit={best_pf_str} | Patience={self._epochs_without_improve}/{patience_limit}"
            )
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
        self._neat_config = neat_config
        population = neat.Population(neat_config)
        if self._config.get("enable_neat_default_reporter", self.params.get("enable_neat_default_reporter", True)):
            try:
                self._stdout_reporter = neat.reporting.StdOutReporter(True)
                population.add_reporter(self._stdout_reporter)
            except Exception as reporter_err:
                print(f"[NEAT][Reporter] Failed to attach StdOutReporter: {reporter_err}")
        try:
            self._stats_reporter = neat.StatisticsReporter()
            population.add_reporter(self._stats_reporter)
        except Exception as stats_err:
            print(f"[NEAT][Reporter] Failed to attach StatisticsReporter: {stats_err}")

        max_generations = int(self.params.get("max_generations", 80))
        self._num_generations = max_generations
        try:
            for gen in range(1, max_generations + 1):
                self._current_epoch = gen
                population.run(self._eval_genomes, 1)
                self._handle_generation_complete(gen, max_generations)
                self._log_species_summary(population, gen)
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
            test_profit, test_stats = self._evaluate_on_dataset(
                best_candidate,
                self._test_frames,
                dataset_label="test",
                genome=self._best_record.get("genome") if self._best_record else None,
                net=self._best_record.get("net") if self._best_record else None,
            )
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

    def _log_species_summary(self, population, generation):
        species_container = getattr(population, "species", None)
        if species_container is None:
            return
        species_dict = getattr(species_container, "species", None)
        if not species_dict:
            print(f"[NEAT][Species][Epoch {generation}] No active species")
            return
        compat_threshold = None
        try:
            compat_threshold = population.config.species_set_config.compatibility_threshold
        except Exception:
            compat_threshold = None
        species_count = len(species_dict)
        thresh_str = f"{compat_threshold:.2f}" if compat_threshold is not None else "N/A"
        print(
            f"[NEAT][Species][Epoch {generation}] count={species_count} | compatibility_threshold={thresh_str}"
        )
        sizes = []
        stagnated_species = 0
        for species_id, species in species_dict.items():
            members = getattr(species, "members", {}) or {}
            size = len(members)
            sizes.append(size)
            best_fitness = None
            for genome in members.values():
                fitness = getattr(genome, "fitness", None)
                if fitness is None:
                    continue
                if best_fitness is None or fitness > best_fitness:
                    best_fitness = fitness
            best_str = f"{best_fitness:.2f}" if best_fitness is not None else "N/A"
            age = getattr(species, "age", "?")
            stag = getattr(species, "last_improved", "?")
            try:
                last_improved = int(stag)
                current_gen = getattr(species, "created", 0) + age
                stagnation_span = current_gen - last_improved if last_improved is not None else age
                if (
                    last_improved is not None
                    and self.params.get("stagnation_limit") is not None
                    and stagnation_span >= int(self.params.get("stagnation_limit", 0))
                ):
                    stagnated_species += 1
            except Exception:
                pass
            print(
                f"  Species {species_id}: size={size}, best_profit={best_str}, age={age}, last_improved={stag}"
            )

        if sizes:
            avg_size = sum(sizes) / len(sizes)
            print(
                f"    Size stats => avg={avg_size:.2f}, min={min(sizes)}, max={max(sizes)}, total={sum(sizes)}"
            )

        genome_config = getattr(population.config, "genome_config", None)
        rep_distances: List[float] = []
        if genome_config is not None:
            representatives = []
            for species in species_dict.values():
                rep = getattr(species, "representative", None)
                if rep is not None:
                    representatives.append(rep)
            for idx in range(len(representatives)):
                for jdx in range(idx + 1, len(representatives)):
                    try:
                        dist = representatives[idx].distance(representatives[jdx], genome_config)
                        rep_distances.append(dist)
                    except Exception:
                        continue
        if rep_distances:
            avg_dist = sum(rep_distances) / len(rep_distances)
            print(
                f"    Representative distance stats => avg={avg_dist:.4f}, min={min(rep_distances):.4f}, "
                f"max={max(rep_distances):.4f}, samples={len(rep_distances)}"
            )
        else:
            print("    Representative distance stats => insufficient data")

        self._maybe_adjust_compatibility(population, species_count, stagnated_species)

    def _maybe_adjust_compatibility(self, population, species_count: int, stagnated_species: int = 0):
        if population is None or species_count <= 0:
            return
        target = self._config.get(
            "target_species_count",
            self.params.get("target_species_count", 0),
        )
        try:
            target = float(target)
        except (TypeError, ValueError):
            target = 0.0
        if target <= 0:
            return
        adjust_rate = self._config.get(
            "compatibility_adjust_rate",
            self.params.get("compatibility_adjust_rate", 0.0),
        )
        try:
            adjust_rate = float(adjust_rate)
        except (TypeError, ValueError):
            adjust_rate = 0.0
        if adjust_rate <= 0:
            return

        threshold = getattr(population.config.species_set_config, "compatibility_threshold", None)
        if threshold is None:
            return

        lower_bound = target * 0.8
        upper_bound = target * 1.2
        new_threshold = None
        direction = None
        reason = None

        if species_count < lower_bound:
            new_threshold = max(0.1, threshold * (1 - adjust_rate))
            direction = "decreased"
            reason = "below target"
        elif species_count > upper_bound:
            if stagnated_species <= 0:
                return
            new_threshold = threshold * (1 + adjust_rate)
            direction = "increased"
            reason = f"above target with {stagnated_species} stagnated"

        if new_threshold is None or abs(new_threshold - threshold) < 1e-6:
            return

        population.config.species_set_config.compatibility_threshold = new_threshold
        try:
            population.species.species_set.compatibility_threshold = new_threshold
        except Exception:
            pass
        print(
            f"[NEAT][Species] Auto-adjusted compatibility threshold {direction} to {new_threshold:.3f} "
            f"(target={target:.1f}, actual={species_count}, stagnated={stagnated_species}, reason={reason})"
        )
