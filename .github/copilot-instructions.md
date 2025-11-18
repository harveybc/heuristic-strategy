# Copilot instructions

- Workspace = four coordinated repos (`predictor`, `preprocessor`, `feature-eng`, `heuristic-strategy`). They share a common plugin/config-merger stack: every CLI call loads defaults from `app/config.py`, merges JSON/remote config, then plugin defaults, then CLI flags via `config_merger.merge_config`.
- Typical workflow (production-quality runs use Conda py3.9): preprocess with `preprocessor`, engineer features with `feature-eng`, train predictors with `predictor`, then evaluate/optimize trading rules with `heuristic-strategy`. Example command: `sh heuristic_strategy.sh --load_config examples/config/phase_1/phase_1_strategy_config.json`.
- Plugins are packaged via setuptools entry points. Strategy plugins live under `heuristic_strategy.plugins`; optimizer plugins under `heuristic_strategy.optimizer_plugins`. Load them with `app.plugin_loader.load_plugin`. When adding or renaming plugins always:
  1. Register entry points in `setup.py`.
  2. Provide `plugin_params`, `set_params`, and `get_debug_info` so `merge_config` can inject defaults.
  3. Keep constructor state-free; runtime config arrives through `set_params`.
- `heuristic-strategy/app/main.py` performs two config passes: first without plugin params, then after instantiating strategy + optimizer plugins, it re-runs `merge_config` with both parameter dictionaries and replays `set_params`. Preserve this flow if you touch argument parsing.
- Optimizer responsibilities now live inside the GA plugin (`app/optimizer.py`). It mirrors the strategy-plugin interface and exposes `init_optimizer`, `evaluate_individual`, and `run_optimizer`. When creating new optimizers, follow that contract and ensure multiprocessing paths rely on the `_mp_safe_evaluate` helper.
- `app/data_processor.py` is the staging area: it loads/generates datasets, injects uncertainty frames into `config`, and calls the optimizer plugin. Do not bypass it; instead, extend the helper functions (`_create_predictions_at_offsets`, `_apply_gaussian_noise`, etc.) when new data prep is required.
- Prediction files: hourly/daily CSVs aligned on `DATE_TIME`. If files are missing, the pipeline auto-builds "ideal" predictions + Gaussian noise (controlled by `gaussian_noise_*` keys). Respect this auto-generation when integrating new models.
- Validation/early-stopping: GA runs with train/validation/test supersets built inside `run_processing_pipeline`. If you add new datasets, make sure they flow through `build_supersets` so offsets/indices stay aligned.
- Remote/replicable workflows rely on `config_handler` helpers (`remote_load_config`, `remote_save_config`, `remote_log`). Never short-circuit these; instead, add new remote flags to `app/cli.py` and pass them through merge_config.
- Testing: quick smoke = `pytest` inside each repo (requires `set_env.sh` for path setup). Optimizer changes should at least run `pytest tests -k optimizer` (if available) or execute `heuristic_strategy.sh --help` to ensure entry points resolve.
- Logging: The system is verbose by design. Preserve existing `print` statements (especially `[INIT]`, `[EVALUATE]`, progress summaries) so OLAP ingestion scripts can parse logs deterministically.
