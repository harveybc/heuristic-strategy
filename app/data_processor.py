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
# 1. Hourly predictions: Contains the next n_hourly predictions (columns for t+1 to t+n_hourly).
# 2. Daily predictions: For each tick (hourly timestamp), contains predictions for the next n_daily days 
#    (i.e. predictions for t+24, t+48, …, t+24*n_daily).
# 3. Base dataset: Contains the actual rates per tick.
#
# If a date column is provided in the configuration (using key "date_column"), it is converted to a 
# datetime index and the datasets are aligned based on the common date range.
#
# After processing, the pipeline calls the optimizer from the current strategy plugin via the 
# "optimize_strategy" method. The plugin is expected to run the trading simulation (e.g., with backtrader)
# and return a dictionary with trading performance details.
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
        config (dict): Configuration dictionary with keys:
            - "hourly_predictions_file": Path to the hourly predictions CSV.
            - "daily_predictions_file": Path to the daily predictions CSV.
            - "base_dataset_file": Path to the base dataset CSV.
            - "headers" (optional): Boolean indicating if CSV files have headers.
            - "date_column" (optional): Name of the column containing date/time information.
    
    Returns:
        dict: Dictionary with keys "hourly", "daily", "base" whose values are processed pandas DataFrames.
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
    
    # Convert date column to datetime index if provided
    date_column = config.get("date_column")
    if date_column:
        for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
                print(f"{label} dataset: Converted '{date_column}' to datetime index (shape: {df.shape}).")
            else:
                print(f"Warning: '{date_column}' not found in {label} dataset.")
    
    # Align datasets by their common date range if all indices are datetime
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
    
    # Convert all non-index columns to numeric and fill missing values with zero
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
    and then passes the processed data along with the configuration to the strategy plugin’s
    optimizer via its 'optimize_strategy' method. The plugin is expected to run a trading simulation
    (e.g., with backtrader using a DEAP-based optimizer) and return a dictionary with the trading results.
    
    Args:
        config (dict): Configuration dictionary containing file paths and parameters.
        plugin (object): Strategy plugin that implements the optimize_strategy() method.
    
    Returns:
        None
    """
    start_time = time.time()
    print("\n=== Starting Trading Strategy Optimization Pipeline ===")
    
    # Process and align datasets
    datasets = process_data(config)
    hourly_preds = datasets["hourly"]
    daily_preds  = datasets["daily"]
    base_data    = datasets["base"]
    
    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base rates:         {base_data.shape}")
    
    # Call the plugin optimizer with the processed data
    print("\nInvoking strategy plugin optimizer...")
    try:
        trading_info = plugin.optimize_strategy(
            base_data=base_data,
            hourly_predictions=hourly_preds,
            daily_predictions=daily_preds,
            config=config
        )
    except Exception as e:
        print("Error during strategy optimization:", e)
        sys.exit(1)
    
    # Display the trading simulation results
    print("\n=== Trading Strategy Optimization Results ===")
    if isinstance(trading_info, dict):
        for key, value in trading_info.items():
            print(f"{key}: {value}")
    else:
        print("Optimizer did not return the expected trading information dictionary.")
    
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # When running this module directly, ensure a valid configuration dict and plugin instance are provided.
    # Example usage:
    #
    # from app.strategy_plugin import StrategyPlugin
    # config = {
    #     "hourly_predictions_file": "data/hourly_predictions.csv",
    #     "daily_predictions_file": "data/daily_predictions.csv",
    #     "base_dataset_file": "data/base_dataset.csv",
    #     "headers": True,
    #     "date_column": "DATE_TIME"
    # }
    # plugin = StrategyPlugin()  # This plugin must implement optimize_strategy()
    # run_prediction_pipeline(config, plugin)
    #
    pass
