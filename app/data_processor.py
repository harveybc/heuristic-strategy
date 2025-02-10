import pandas as pd
import numpy as np
import os
import time
import sys
from app.data_handler import load_csv

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# This module processes three CSV datasets:
#
#   1. Hourly predictions: Next n_hourly predictions (columns for t+1 to t+n_hourly).
#   2. Daily predictions: For each tick (hourly timestamp), predictions for the next n_daily days
#                        (e.g. t+24, t+48, …, t+24*n_daily).
#   3. Base dataset: Actual rates per tick.
#
# If a date column is specified via config (key "date_column"), it is converted to a datetime index
# and the datasets are aligned to the common date range.
#
# If either the hourly_predictions_file or daily_predictions_file is None, the corresponding
# predictions are auto-generated from the base dataset using the value in config["time_horizon"]:
#   - Hourly predictions: the next time_horizon ticks (t+1 ... t+time_horizon)
#   - Daily predictions: the values at t+24, t+48, ... , t+24*time_horizon
#
# After processing, the pipeline calls an optimizer module (app.optimizer) which uses the strategy
# plugin’s optimization interface (get_optimizable_params and evaluate_candidate) to search for the
# best parameters.
#
# Additionally, the pipeline saves a CSV file containing the optimization results (summary) and another
# CSV file containing simulated trades (if available). The trades file includes open and exit prices,
# profit, maximum drawdown, and the date of each trade. If the 'print_trades' flag is set in the configuration,
# the simulated trades are also printed to the console.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    
    For each row in the base dataset, the prediction is a block of the next 'horizon' values.
    
    Args:
        df (pd.DataFrame): Base dataset.
        horizon (int): Number of future ticks to predict.
    
    Returns:
        pd.DataFrame: DataFrame of hourly predictions.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.
    
    For each row in the base dataset, the prediction is a block of values at offsets 24, 48, ..., 24*horizon.
    
    Args:
        df (pd.DataFrame): Base dataset.
        horizon (int): Number of future days (each day = 24 ticks) to predict.
    
    Returns:
        pd.DataFrame: DataFrame of daily predictions.
    """
    blocks = []
    for i in range(len(df) - horizon * 24):
        block = []
        for d in range(1, horizon+1):
            block.extend(df.iloc[i + d*24].values.flatten())
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon*24])

def process_data(config):
    """
    Process three datasets required for trading strategy optimization:
      - Hourly predictions (columns: predictions from t+1 to t+n_hourly)
      - Daily predictions (each row holds predictions for t+24, t+48, ..., t+24*n_daily)
      - Base dataset (actual rates per tick)
    
    If a date column is specified via config["date_column"], it is converted to a datetime index
    and the datasets are aligned.
    
    If either predictions file is None, predictions are auto-generated from the base dataset
    using config["time_horizon"].
    
    Args:
        config (dict): Configuration with keys including:
            "hourly_predictions_file", "daily_predictions_file", "base_dataset_file",
            "headers", "date_column", "time_horizon"
    
    Returns:
        dict: Dictionary with keys "hourly", "daily", "base" holding processed DataFrames.
    """
    headers = config.get("headers", True)
    print("Loading datasets...")
    
    if config.get("hourly_predictions_file"):
        hourly_df = load_csv(config["hourly_predictions_file"], headers=headers)
        print(f"Loaded hourly predictions from file: {config['hourly_predictions_file']}")
    else:
        hourly_df = None
        print("No hourly predictions file provided. Will auto-calculate hourly predictions.")

    if config.get("daily_predictions_file"):
        daily_df = load_csv(config["daily_predictions_file"], headers=headers)
        print(f"Loaded daily predictions from file: {config['daily_predictions_file']}")
    else:
        daily_df = None
        print("No daily predictions file provided. Will auto-calculate daily predictions.")

    base_df = load_csv(config["base_dataset_file"], headers=headers)
    print(f"Base dataset loaded: {base_df.shape}")

    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-calculating predictions.")
        print("Auto-calculating hourly predictions using base dataset...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])
        print(f"Auto-generated hourly predictions: {hourly_df.shape}")

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-calculating predictions.")
        print("Auto-calculating daily predictions using base dataset...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])
        print(f"Auto-generated daily predictions: {daily_df.shape}")

    print("\nDatasets loaded:")
    print(f"  Hourly predictions: {hourly_df.shape}")
    print(f"  Daily predictions:  {daily_df.shape}")
    print(f"  Base dataset:       {base_df.shape}")

    date_column = config.get("date_column")
    if date_column:
        for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
                print(f"{label} dataset: Converted '{date_column}' to datetime index (shape: {df.shape}).")
            else:
                print(f"Warning: '{date_column}' not found in {label} dataset.")

    if (hasattr(hourly_df.index, 'min') and hasattr(daily_df.index, 'min') and hasattr(base_df.index, 'min')):
        try:
            common_start = max(hourly_df.index.min(), daily_df.index.min(), base_df.index.min())
            common_end   = min(hourly_df.index.max(), daily_df.index.max(), base_df.index.max())
            hourly_df = hourly_df.loc[common_start:common_end]
            daily_df  = daily_df.loc[common_start:common_end]
            base_df   = base_df.loc[common_start:common_end]
            print(f"Datasets aligned to common date range: {common_start} to {common_end}")
        except Exception as e:
            print("Error aligning datasets by date:", e)

    for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
        df[df.columns] = df[df.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        print(f"{label} dataset: Converted columns to numeric (final shape: {df.shape}).")

    return {
        "hourly": hourly_df,
        "daily": daily_df,
        "base": base_df
    }

def run_prediction_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    This function processes the three datasets and then—if the plugin supports the optimization interface—
    calls the optimizer module.
    
    The plugin must implement:
       - get_optimizable_params()
       - evaluate_candidate(individual, base_data, hourly_predictions, daily_predictions, config)
    
    After optimization, the pipeline saves a CSV file with optimization results (config["results_file"])
    and, if available, a CSV file with simulated trades (config["trades_file"]). If config["print_trades"] is True,
    simulated trades are printed.
    
    Args:
        config (dict): Configuration dictionary.
        plugin (object): Strategy plugin instance.
    
    Returns:
        tuple: (trading_info, trades)
    """
    start_time = time.time()
    print("\n=== Starting Trading Strategy Optimization Pipeline ===")
    
    datasets = process_data(config)
    hourly_preds = datasets["hourly"]
    daily_preds  = datasets["daily"]
    base_data    = datasets["base"]
    
    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base rates:         {base_data.shape}")
    
    if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
        print("\nPlugin supports optimization interface. Running optimizer...")
        from app.optimizer import run_optimizer
        trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
    else:
        print("\nPlugin does not support optimization. Exiting.")
        trading_info = {}

    print("\n=== Trading Strategy Optimization Results ===")
    if isinstance(trading_info, dict):
        for key, value in trading_info.items():
            print(f"{key}: {value}")
    else:
        print("Optimizer did not return the expected trading information dictionary.")
    
    if config.get("results_file"):
        try:
            results_df = pd.DataFrame([trading_info])
            results_df.to_csv(config["results_file"], index=False)
            print(f"Optimization results saved to {config['results_file']}.")
        except Exception as e:
            print(f"Failed to save optimization results: {e}")
    
    trades = getattr(plugin, "trades", None)
    if trades is not None and config.get("trades_file"):
        try:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(config["trades_file"], index=False)
            print(f"Simulated trades saved to {config['trades_file']}.")
            if config.get("print_trades"):
                print("\nSimulated Trades:")
                print(trades_df)
        except Exception as e:
            print(f"Failed to save simulated trades: {e}")
    
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    
    return trading_info, trades

if __name__ == "__main__":
    # Example usage:
    # from app.plugins.plugin_long_short_predictions import Plugin as StrategyPlugin
    # config = {
    #     "hourly_predictions_file": None,
    #     "daily_predictions_file": None,
    #     "base_dataset_file": "data/base_dataset.csv",
    #     "headers": True,
    #     "date_column": "DATE_TIME",
    #     "time_horizon": 6,
    #     "population_size": 20,
    #     "num_generations": 50,
    #     "crossover_probability": 0.5,
    #     "mutation_probability": 0.2,
    #     "results_file": "optimization_results.csv",
    #     "trades_file": "simulated_trades.csv",
    #     "print_trades": True
    # }
    # plugin = StrategyPlugin()
    # run_prediction_pipeline(config, plugin)
    pass
