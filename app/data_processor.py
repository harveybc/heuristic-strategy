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
# After processing, the pipeline calls an optimizer module (optimizer.py) which uses the strategy plugin’s
# optimization interface (get_optimizable_params and evaluate_candidate) to search for the best parameters.
#
# Additionally, the pipeline saves a CSV file containing the optimization results (summary) and another CSV
# file containing simulated trades (if available). The trades file includes open and exit prices, profit,
# maximum drawdown, and the date of each trade. If the 'print_trades' flag is set in the configuration,
# the simulated trades are also printed to the console.
# =============================================================================

def process_data(config):
    """
    Process three datasets required for trading strategy optimization:
      - Hourly predictions (columns: predictions from t+1 to t+n_hourly)
      - Daily predictions (each row holds predictions for t+24, t+48, ..., t+24*n_daily)
      - Base dataset (actual rates per tick)
    
    If a date column is specified (via config["date_column"]), that column is converted to a 
    datetime index and the datasets are aligned to the common date range.
    
    Args:
        config (dict): Configuration with keys:
            - "hourly_predictions_file": Path to the hourly predictions CSV.
            - "daily_predictions_file": Path to the daily predictions CSV.
            - "base_dataset_file": Path to the base dataset CSV.
            - "headers" (optional): Boolean indicating if CSV files have headers.
            - "date_column" (optional): Name of the date/time column.
    
    Returns:
        dict: Dictionary with keys "hourly", "daily", "base" holding processed DataFrames.
    """
    headers = config.get("headers", True)
    print("Loading datasets...")
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers)
    daily_df  = load_csv(config["daily_predictions_file"], headers=headers)
    base_df   = load_csv(config["base_dataset_file"], headers=headers)
    
    print("Datasets loaded:")
    print(f"  Hourly predictions: {hourly_df.shape}")
    print(f"  Daily predictions:  {daily_df.shape}")
    print(f"  Base dataset:       {base_df.shape}")
    
    # Convert date column to datetime index if provided.
    date_column = config.get("date_column")
    if date_column:
        for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
                print(f"{label} dataset: Converted '{date_column}' to datetime index (shape: {df.shape}).")
            else:
                print(f"Warning: '{date_column}' not found in {label} dataset.")
    
    # Align datasets by their common date range if indices are datetime.
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
    
    # Convert all non-index columns to numeric (filling missing values with zero).
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
    
    This function processes the three datasets (hourly predictions, daily predictions, and base rates)
    and then—if the provided plugin supports the optimization interface—calls the optimizer module.
    
    The plugin must implement:
       - get_optimizable_params()
       - evaluate_candidate(individual, base_data, hourly_predictions, daily_predictions, config)
    
    Additionally, after optimization, the pipeline saves a CSV file containing the optimization results (summary)
    and, if available, a CSV file with simulated trades. If the configuration flag 'print_trades' is set,
    the simulated trades are printed to the console.
    
    Args:
        config (dict): Configuration dictionary.
        plugin (object): Strategy plugin instance.
    
    Returns:
        tuple: (trading_info, trades) where trading_info is a dict with the optimization summary,
               and trades is a list of trade dictionaries (or None if not available).
    """
    start_time = time.time()
    print("\n=== Starting Trading Strategy Optimization Pipeline ===")
    
    # Process and align datasets.
    datasets = process_data(config)
    hourly_preds = datasets["hourly"]
    daily_preds  = datasets["daily"]
    base_data    = datasets["base"]
    
    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base rates:         {base_data.shape}")
    
    # Run the optimizer if the plugin supports the optimization interface.
    if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
        print("\nPlugin supports optimization interface. Running optimizer...")
        from optimizer import run_optimizer
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
    
    # Save optimization results to CSV if 'results_file' is specified.
    if config.get("results_file"):
        try:
            results_df = pd.DataFrame([trading_info])
            results_df.to_csv(config["results_file"], index=False)
            print(f"Optimization results saved to {config['results_file']}.")
        except Exception as e:
            print(f"Failed to save optimization results: {e}")
    
    # Save simulated trades to CSV if available and if 'trades_file' is specified.
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
    # Example usage (update file paths and parameters as needed):
    #
    # from strategy_plugin import StrategyPlugin
    # config = {
    #     "hourly_predictions_file": "data/hourly_predictions.csv",
    #     "daily_predictions_file": "data/daily_predictions.csv",
    #     "base_dataset_file": "data/base_dataset.csv",
    #     "headers": True,
    #     "date_column": "DATE_TIME",
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
