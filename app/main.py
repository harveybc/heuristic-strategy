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
from app.data_processor import process_data, run_processing_pipelins
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    """
    Main entry point for the heuristic_strategy application.

    This function orchestrates:
      1. Parsing CLI arguments.
      2. Loading default, remote, and local configurations.
      3. Merging configurations with CLI and unknown args.
      4. Loading the specified plugin from the 'heuristic_strategy.plugins' namespace.
      5. Merging plugin parameters.
      6. Running the processing pipeline (optimization and trade simulation).
      7. Saving configurations remotely (if specified).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    print("Merging configuration with CLI and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # If no plugin is provided on CLI, use the one defined in configuration.
    if not cli_args.get('plugin'):
        cli_args['plugin'] = config.get('plugin', 'default')

    plugin_name = cli_args['plugin']
    print(f"Loading plugin: {plugin_name}")
    try:
        # Load the plugin from the 'heuristic_strategy.plugins' namespace.
        plugin_class, _ = load_plugin('heuristic_strategy.plugins', plugin_name)
        plugin = plugin_class()
        # Set plugin parameters based on merged configuration.
        plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize plugin '{plugin_name}': {e}")
        sys.exit(1)

    print("Merging configuration with CLI and unknown args (second pass, with plugin params)...")
    config = merge_config(config, plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    if config.get('load_model'):
        print("Warning: 'load_model' is not applicable for trading strategy plugins. Ignoring this parameter.")

    print("Processing and running optimization pipeline...")
    trading_info, trades = run_processing_pipelins(config, plugin)

    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config.get('username'), config.get('password'))
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")

if __name__ == "__main__":
    main()
