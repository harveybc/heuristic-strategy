# NEAT Optimizer & Adaptive Strategy Plan

## Objectives
- Introduce a NEAT (NeuroEvolution of Augmenting Topologies) optimizer plugin that slots into the existing `heuristic_strategy.optimizer_plugins` entry point group and speaks the same contract as the GA/DEAP implementation.
- Produce an adaptive strategy plugin that mirrors the current long/short behavior but exposes a narrow, NEAT-focused parameter surface (exactly the five canonical knobs used today).
- Keep compatibility with `app/main.py`'s dual config merge, `data_processor.run_processing_pipeline`, and the plugin loader requirements captured in `.github/copilot-instructions.md`.
- Preserve deterministic logging/remote-config flow (`merge_config`, `config_handler`, `remote_*`) while adding NEAT-specific config defaults and CLI flags.

## Existing Building Blocks
- Strategy plugin contract: provides `plugin_params`, `set_params`, `get_debug_info`, `get_optimizable_params`, `evaluate_candidate`, plus strategy-specific helpers (see `heuristic_strategy_plugins/plugin_long_short_predictions.py`).
- Optimizer plugin contract: `plugin_params`, `set_params`, `get_debug_info`, `init_optimizer`, `evaluate_individual`, `run_optimizer` (see `app/optimizer.py`).
- Data pipeline already assembles `base_data`, `hourly_predictions`, `daily_predictions`, and optional validation/test supersets before calling `optimizer_plugin.run_optimizer`.
- Utility helpers inside `app/optimizer.py` such as `_aggregate_mae_and_naive`, `_aggregate_mae_and_naive_for_sets`, `_compute_uniform_offsets`, `_mp_safe_evaluate` can be reused/imported.

## Proposed NEAT Optimizer Plugin

### Contract snapshot
- **Inputs**: `strategy_plugin`, `base_data`, `hourly_predictions`, `daily_predictions`, merged `config`, optional validation/test supersets.
- **Outputs**: dict with `best_parameters` (the five strategy knobs), `train_profit`, `val_profit`, `test_profit`, `stats`, and metadata mirroring GA plugin keys.
- **Methods**: same as GA plugin plus NEAT-specific helpers (`_build_neat_config`, `_extract_features`, `_decode_output_vector`).

### Plugin parameters (10 NEAT hyperparameters + headcount)
1. `population_size` – default 150 (classic NEAT default, controls species sizes).
2. `max_generations` – default 80 (upper bound for evolution loop, distinct from GA `num_generations`).
3. `fitness_threshold` – default 2500.0 (stop early if best fitness crosses this profit level).
4. `elitism` – default 2 (how many top genomes survive unchanged each generation).
5. `survival_threshold` – default 0.2 (fraction of each species allowed to reproduce).
6. `stagnation_limit` – default 15 (species stagnation generations before reset).
7. `weight_mutation_power` – default 2.5 (scales perturbations applied to connection weights).
8. `bias_mutation_power` – default 0.5 (range for bias mutations in node genes).
9. `compatibility_threshold` – default 3.0 (species separation constant).
10. `activation_default` – default `tanh` (seed activation for new nodes; available choices enumerated via config/CLI).
11. `aggregation_default` – default `sum` (controls node aggregation; still part of top knobs but kept separate for clarity).

> **Note**: population size + ten knobs satisfy the "top 10 hyperparameters" ask; we treat aggregation default as the tenth distinct setting because NEAT differentiates activation vs aggregation.

### Feature engineering for genomes
Each genome receives a fixed feature vector derived once per dataset snapshot. Proposed features (computed via helpers inside plugin):
- `st_mae_ratio`: MAE/naive MAE for short-term superset columns (reuses `_aggregate_mae_and_naive`).
- `lt_mae_ratio`: same for long-term superset.
- `volatility`: rolling std dev of `base_data['CLOSE']` over 24h normalized by price level.
- `trend_slope`: linear regression slope over last 48h normalized.
- `prediction_spread`: average |daily_pred - hourly_pred| at overlapping horizons.
- `uncertainty_ratio`: mean daily uncertainty / mean hourly uncertainty (from config injections).
- `drawdown_memory`: ratio of recent min to max of price (captures risk environment).
- `momentum_osc`: difference of two EMAs (fast=12, slow=26) normalized by pip cost.
- `hour_of_day_sin`, `hour_of_day_cos`: cyclical encoding to let NEAT adapt to session effects.

The feature vector is fed to each network's forward pass; the NEAT genome produces five continuous outputs each mapped to strategy parameter ranges.

### Output decoding (ensuring only five knobs controlled)
Outputs correspond exactly to `[profit_threshold, tp_multiplier, sl_multiplier, lower_rr_threshold, upper_rr_threshold]`. We scale/tanh outputs to bounds defined by `strategy_plugin.get_optimizable_params()`:
- Use sigmoid mapping to [low, high] per param.
- Snap to min/max as needed.
- Keep `upper_rr_threshold >= lower_rr_threshold + 0.1` to avoid invalid combos (enforced after decoding).

### Run loop outline
1. **Initialization**: clone config, cache datasets, compute global feature vector.
2. **NEAT config build**: instantiate `neat.Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation)` with values from plugin params; embed aggregator/hyperparams via an in-memory `.ini` string to avoid writing to disk.
3. **Population**: `neat.Population(config)` with reporters (stdout + statistics) respecting logging style.
4. **Evaluation**: Each genome -> feedforward net -> decode 5 params -> call `strategy_plugin.evaluate_candidate` with `base_data`, `hourly_predictions`, `daily_predictions`, config (plus subset selection logic identical to GA plugin using `_compute_uniform_offsets`). Optionally share the superset down-selection: treat 2 additional outputs as `st_max/lt_max`? The requirement specified "only five main strategy parameters dynamically", so NEAT will not reselect horizons; reuse whatever dataset selection user provided.
5. **Fitness**: Use resulting profit as `fitness`; optionally penalize large drawdowns via `profit - penalty * risk` (configurable weight).
6. **Validation/test**: After each generation or when best candidate improves, run candidate on validation/test supersets when provided.
7. **Stopping**: break if `fitness_threshold` reached or `max_generations` exhausted.
8. **Return**: pack best parameters + profits + stats dict consistent with GA plugin outputs.

### Parallelization & reproducibility
- NEAT supports `ParallelEvaluator`. We'll reuse `_active_optimizer_instance` + `_mp_safe_evaluate` for compatibility when `config['disable_multiprocessing']` is False.
- Random seeds derived from config (`random_seed`, fallback 42) to keep deterministic runs.

## Adaptive Strategy Plugin

### Goals
- Live under `heuristic_strategy_plugins/plugin_adaptive_neat.py`.
- Reuse the current HeuristicStrategy internals but cleanly separate the five optimizable parameters from structural params (pip cost, volume, leverage, etc.).
- Provide a method `set_dynamic_params(vector)` so the NEAT optimizer (or any optimizer) can optionally override the five knobs between evaluations without mutating other defaults.

### Behavior
- `plugin_params`: same as long/short plugin except `profit_threshold`, `tp_multiplier`, `sl_multiplier`, `lower_rr_threshold`, `upper_rr_threshold` move into a `dynamic_defaults` sub-dict to highlight NEAT control.
- `get_optimizable_params`: returns the same bounds but flagged as "NEAT_CONTROLLED" to document the contract; GA/NEAT can ignore the flag.
- `evaluate_candidate`: identical to current logic but reading runtime params from a helper `self._extract_candidate(individual)` that clamps to ranges and optionally logs NEAT-provided metadata.
- `HeuristicStrategy`: inherits from the existing class or composes it (import from long_short module) to avoid code duplication; plan is to extract shared base class later, but for MVP we can instantiate the existing strategy and pass down the candidate vector.
- Input predictions/uncertainties identical (merging hourly/daily + uncertainties). Output stats/trades identical for compatibility.

### I/O patterns
- **Inputs**: `(candidate_vector, base_data, hourly_predictions, daily_predictions, config)`
- **Outputs**: `(profit: float, stats: dict)` with stats keys unchanged so the downstream summary writer keeps working.
- Documented expectation that candidate_vector contains exactly the five NEAT-controlled knobs, in the same order as GA plugin uses today.

## Config & Wiring Changes
- `app/config.py`: add `neat_optimizer` defaults (e.g., `"optimizer_plugin": "ga_optimizer"` stays but add `"available_optimizer_plugins": ["ga_optimizer", "neat_optimizer"]` comment + NEAT hyperparameter defaults).
- `app/cli.py`: new flags `--activation_default`, `--compatibility_threshold`, etc., auto-injected via `merge_config`.
- `setup.py`: register `neat_optimizer=heuristic_optimizer_plugins.neat_optimizer:Plugin` and `adaptive_neat_strategy=heuristic_strategy_plugins.plugin_adaptive_neat:Plugin`.
- `requirements.txt`: add `neat-python>=0.92` (and ensure `numpy`, `scipy` already covered).

## Testing Strategy
1. **Unit-ish**: Create lightweight test harness under `tests/` that instantiates the NEAT plugin with a stub strategy returning deterministic profits to verify decode + feature extraction.
2. **Smoke**: Run `pytest tests -k optimizer` (if such tests exist) or `heuristic.sh --help` to ensure entry points resolve.
3. **End-to-end**: Execute `heuristic_strategy.sh --load_config examples/config/phase_1/phase_1_strategy_config.json --optimizer_plugin neat_optimizer --strategy adaptive_neat_strategy --max_generations 2 --population_size 10` (small numbers) to confirm pipeline completes.

## Assumptions & Open Questions
- Exact NEAT input/output spec was not provided; the feature list above is inferred from available data and GA helper functions.
- Adaptive strategy is expected to mirror today’s long/short logic; if additional behaviors (e.g., intra-day parameter switching) are desired, we’ll extend after baseline parity is confirmed.
- Risk penalty formula (`profit - penalty * max_drawdown_ratio`) needs stakeholder-provided weight; plan assumes config key `neat_drawdown_penalty` (default 0.0).

## Next Steps
1. Scaffold optimizer/strategy files and register them.
2. Port shared helpers from `app/optimizer.py` into a reusable module (or import functions directly) to avoid duplication.
3. Implement NEAT evaluation loop with deterministic logging and optional multiprocessing.
4. Validate with small configs, then document usage in `README.md` / examples.
