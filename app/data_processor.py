import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls `run_optimizer()` from `app.optimizer` to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets.
    
    - Loads the hourly predictions, daily predictions (if provided), and the base dataset.
    - If 'max_steps' is provided in config, each dataset is truncated to the first max_steps rows.
    - If a predictions file is not provided, predictions are auto-generated using config["time_horizon"].
    - If a date_column is provided:
        * For the Base dataset: converts that column to datetime and sets it as the index (named "DATE_TIME").
        * For the predictions datasets: if the date_column is present, they are processed normally;
          otherwise, it is assumed that the predictions are aligned with the base dataset, and the base's
          index (truncated to the number of rows in the predictions) is assigned.
    - Finally, the datasets are aligned by their common date range and all columns are converted to numeric.
    
    Args:
        config (dict): Configuration with keys including:
            - "hourly_predictions_file"
            - "daily_predictions_file"
            - "base_dataset_file"
            - "headers"
            - "date_column"
            - "time_horizon"
            - "max_steps"
    
    Returns:
        dict: Dictionary with keys "hourly", "daily", "base" holding the processed DataFrames.
    """
    import pandas as pd
    import numpy as np
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Truncate datasets to max_steps if provided.
    max_steps = config.get("max_steps")
    if max_steps is not None:
        if hourly_df is not None:
            hourly_df = hourly_df.iloc[:max_steps]
        if daily_df is not None:
            daily_df = daily_df.iloc[:max_steps]
        base_df = base_df.iloc[:max_steps]
        print(f"Datasets truncated to the first {max_steps} rows.")

    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    print("\nDatasets loaded:")
    print(f"  Hourly predictions: {hourly_df.shape}")
    print(f"  Daily predictions:  {daily_df.shape}")
    print(f"  Base dataset:       {base_df.shape}")

    date_column = config.get("date_column")
    if date_column:
        # Process Base dataset.
        if date_column in base_df.columns:
            base_df[date_column] = pd.to_datetime(base_df[date_column])
            base_df.set_index(date_column, inplace=True)
            base_df.index.name = "DATE_TIME"
        else:
            # If not found, assume base_df's index is already datetime.
            if not pd.api.types.is_datetime64_any_dtype(base_df.index):
                print(f"Warning: '{date_column}' not found in Base dataset. Using default index.")
            else:
                base_df.index.name = "DATE_TIME"

        # Process predictions datasets.
        for label, df in zip(["Hourly", "Daily"], [hourly_df, daily_df]):
            if df is not None:
                if date_column in df.columns:
                    df[date_column] = pd.to_datetime(df[date_column])
                    df.set_index(date_column, inplace=True)
                    df.index.name = "DATE_TIME"
                else:
                    print(f"Warning: '{date_column}' not found in {label} dataset. Assuming predictions are aligned with Base.")
                    # Replace index with Base's index (truncate to the length of the predictions).
                    df.index = base_df.index[:len(df)]
                    df.index.name = "DATE_TIME"
    else:
        # No date_column provided; ensure predictions have a datetime index.
        for label, df in zip(["Hourly", "Daily"], [hourly_df, daily_df]):
            if df is not None and not pd.api.types.is_datetime64_any_dtype(df.index):
                print(f"Warning: {label} predictions have no datetime index. Assigning Base index.")
                df.index = base_df.index[:len(df)]
                df.index.name = "DATE_TIME"

    # Align datasets by common date range.
    try:
        common_start = max(hourly_df.index.min(), daily_df.index.min(), base_df.index.min())
        common_end = min(hourly_df.index.max(), daily_df.index.max(), base_df.index.max())
        hourly_df = hourly_df.loc[common_start:common_end]
        daily_df = daily_df.loc[common_start:common_end]
        base_df = base_df.loc[common_start:common_end]
        print(f"Datasets aligned to common date range: {common_start} to {common_end}")
    except Exception as e:
        print("Error aligning datasets by date:", e)

    # Convert all columns to numeric and fill missing values.
    for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
        df[df.columns] = df[df.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        print(f"{label} dataset: Converted columns to numeric (final shape: {df.shape}).")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("rel_volume", plugin.params["rel_volume"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"])
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "rel_volume": candidate[3],
                "lower_rr_threshold": candidate[4],
                "upper_rr_threshold": candidate[5],
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)



if __name__ == "__main__":
    pass
