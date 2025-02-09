import sys
import json
import pandas as pd
from typing import Any, Dict

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.data_processor import (
    process_data,
    load_and_evaluate_model,
    run_prediction_pipeline
)
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args


def main():
    """
    Main entry point for the heuristic_strategy application.

    This function orchestrates the following steps:
      1. Parse CLI arguments and any unknown arguments.
      2. Load default configuration and merge with remote and local configuration files.
      3. Merge CLI and unknown arguments (first pass without plugin-specific parameters).
      4. Load the specified plugin from the 'heuristic_strategy.plugins' namespace and set its parameters.
      5. Merge configuration again including plugin defaults.
      6. Depending on the configuration, either load/evaluate an existing model or run the prediction/backtesting pipeline.
      7. Save the current configuration locally and remotely.
      8. (Within the pipeline) save optimization results and simulated trades (and optionally print them).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Remote configuration file loading.
    if args.remote_load_config:
        try:
            file_config = remote_load_config(
                args.remote_load_config,
                args.username,
                args.password
            )
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Local configuration file loading.
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # First pass: Merge config with CLI arguments and unknown arguments (without plugin-specific parameters).
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # If CLI did not provide a plugin, use the one in config.
    if not cli_args.get('plugin'):
        cli_args['plugin'] = config.get('plugin', 'default')

    plugin_name = cli_args['plugin']
    print(f"Loading plugin: {plugin_name}")
    try:
        # Load the specified plugin from the 'heuristic_strategy.plugins' namespace.
        plugin_class, _ = load_plugin('heuristic_strategy.plugins', plugin_name)
        plugin = plugin_class()
        # Override plugin parameters with the merged configuration.
        plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Second pass: Merge configuration with the plugin's parameters.
    print("Merging configuration with CLI arguments and unknown args (second pass, with plugin params)...")
    config = merge_config(config, plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    # Decision: Load and evaluate an existing model or run the prediction/backtesting pipeline.
    if config.get('load_model'):
        print("Loading and evaluating model...")
        try:
            load_and_evaluate_model(config, plugin)
        except Exception as e:
            print(f"Model evaluation failed: {e}")
            sys.exit(1)
    else:
        print("Processing and running prediction pipeline...")
        # The pipeline now also saves results and trades as CSV files.
        trading_info, trades = run_prediction_pipeline(config, plugin)

    # Save the current configuration locally if a save path is specified.
    if config.get('save_config'):
        try:
            save_config(config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")

    # Save the current configuration remotely if a remote save endpoint is specified.
    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config.get('username'), config.get('password'))
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")


if __name__ == "__main__":
    main()
