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
    """
    headers = config.get("headers", True)
    print("Loading datasets...")

    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

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
        for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
            else:
                print(f"Warning: '{date_column}' not found in {label} dataset.")

    try:
        common_start = max(hourly_df.index.min(), daily_df.index.min(), base_df.index.min())
        common_end = min(hourly_df.index.max(), daily_df.index.max(), base_df.index.max())
        hourly_df, daily_df, base_df = hourly_df.loc[common_start:common_end], daily_df.loc[common_start:common_end], base_df.loc[common_start:common_end]
        print(f"Datasets aligned to common date range: {common_start} to {common_end}")
    except Exception as e:
        print("Error aligning datasets by date:", e)

    for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
        df[df.columns] = df[df.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        print(f"{label} dataset: Converted columns to numeric (final shape: {df.shape}).")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}

def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    """
    start_time = time.time()
    print("\n=== Starting Trading Strategy Optimization Pipeline ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base rates:         {base_data.shape}")

    if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
        print("\nPlugin supports optimization. Running optimizer...")
        trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
    else:
        print("\nPlugin does not support optimization. Exiting.")
        trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

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
        except Exception as e:
            print(f"Failed to save simulated trades: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, trades

if __name__ == "__main__":
    pass
