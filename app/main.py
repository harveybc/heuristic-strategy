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
from app.data_processor import process_data, run_processing_pipeline
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    """
    Orquesta la ejecución completa del sistema, incluyendo la optimización (si se configura)
    y la ejecución del pipeline completo (preprocesamiento, entrenamiento, predicción y evaluación).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Carga remota de configuración si se solicita
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Carga local de configuración si se solicita
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # Primera fusión de la configuración (sin parámetros específicos de plugins)
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # Selección del plugins
    if not cli_args.get('predictor_plugin'):
        cli_args['predictor_plugin'] = config.get('predictor_plugin', 'default_predictor')
    plugin_name = config.get('predictor_plugin', 'default_predictor')
    
    
    # --- CARGA DE PLUGINS ---

    # Load strategy plugin
    if not cli_args.get('plugin'):
        cli_args['plugin'] = config.get('plugin', 'default')

    plugin_name = cli_args['plugin']
    print(f"Loading strategy plugin: {plugin_name}")
    try:
        plugin_class, _ = load_plugin('heuristic_strategy.plugins', plugin_name)
        plugin = plugin_class()
        plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize strategy plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Load optimizer plugin
    if not cli_args.get('optimizer_plugin'):
        cli_args['optimizer_plugin'] = config.get('optimizer_plugin', 'ga_optimizer')

    optimizer_plugin_name = cli_args['optimizer_plugin']
    print(f"Loading optimizer plugin: {optimizer_plugin_name}")
    try:
        optimizer_plugin_class, _ = load_plugin('heuristic_strategy.optimizer_plugins', optimizer_plugin_name)
        optimizer_plugin = optimizer_plugin_class()
        optimizer_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize optimizer plugin '{optimizer_plugin_name}': {e}")
        sys.exit(1)

    print("Merging configuration with CLI and unknown args (second pass, with plugin params)...")
    config = merge_config(config, plugin.plugin_params, optimizer_plugin.plugin_params, file_config, cli_args, unknown_args_dict)
    plugin.set_params(**config)
    optimizer_plugin.set_params(**config)

    if config.get('load_model'):
        print("Warning: 'load_model' is not applicable for trading strategy plugins. Ignoring this parameter.")

    print("Processing and running optimization pipeline...")
    trading_info, trades = run_processing_pipeline(config, plugin, optimizer_plugin)

    if config.get('save_config'):
        try:
            # Remove any datasets (pandas DataFrames) from the configuration before saving.
            serializable_config = {
            key: (value.to_dict() if hasattr(value, "to_dict") else value)
            for key, value in config.items()
            if not isinstance(value, pd.DataFrame)
            }
            
            save_config(serializable_config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")


    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config.get('username'), config.get('password'))
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")

if __name__ == "__main__":
    main()
