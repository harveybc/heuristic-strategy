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
    The main entry point of the Predictor application.

    This function orchestrates the overall workflow:
    1. Parses command-line arguments.
    2. Loads default and file-based configurations (remote and/or local).
    3. Merges configurations with CLI arguments and unknown arguments.
    4. Loads and configures the specified plugin.
    5. Depending on the configuration, either loads and evaluates an existing model
       or processes data through the prediction pipeline.
    6. Saves configurations locally and/or remotely as specified.

    The configuration merging process ensures that the application can dynamically
    adjust its behavior based on user inputs and configuration files, including
    limiting the number of rows read from training and validation datasets using
    `max_steps_train` and `max_steps_test`.

    Raises:
        Exception: Propagates any exception that occurs during execution.
    """
    print("Parsing initial arguments...")
    # Parse command-line arguments and any unknown arguments
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    # Initialize configuration with default values
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Remote configuration file loading
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

    # Local configuration file loading
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # First pass: Merge config with CLI args and unknown args WITHOUT plugin-specific parameters
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    # We give empty dicts for plugin_params in this pass
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # If CLI did not provide a plugin, use whatever is in config['plugin']
    if not cli_args.get('plugin'):
        cli_args['plugin'] = config.get('plugin', 'ann')

    plugin_name = cli_args['plugin']
    print(f"Loading plugin: {plugin_name}")
    try:
        # Load the specified plugin from the 'predictor.plugins' namespace
        plugin_class, _ = load_plugin('predictor.plugins', plugin_name)
        plugin = plugin_class()
        # Override plugin parameters with the merged configuration
        plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Second pass: Merge config with the plugin's parameters (if any)
    print("Merging configuration with CLI arguments and unknown args (second pass, with plugin params)...")
    # Now we pass the plugin's parameters as the first dict so they're recognized properly
    config = merge_config(config, plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    # Decision point: Load and evaluate an existing model or run the prediction pipeline
    if config['load_model']:
        print("Loading and evaluating model...")
        try:
            load_and_evaluate_model(config, plugin)
        except Exception as e:
            print(f"Model evaluation failed: {e}")
            sys.exit(1)
    else:
        print("Processing and running prediction pipeline...")
        run_prediction_pipeline(config, plugin)
        
    # Save the current configuration locally if a save path is specified
    if 'save_config' in config and config['save_config']:
        try:
            save_config(config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")

    # Save the current configuration remotely if a remote save endpoint is specified
    if 'remote_save_config' in config and config['remote_save_config']:
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config['username'], config['password'])
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")


if __name__ == "__main__":
    main()
