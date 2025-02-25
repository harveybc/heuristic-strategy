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
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets (base, hourly predictions, daily predictions) are properly aligned to the common date range.
      Additionally, preserves the full base dataset (base_full) for later evaluation.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)
    print(f"Base dataset loaded: {base_df.shape}")

    # Conserva una copia completa del base dataset para la evaluación
    base_df_full = base_df.copy()

    # Auto-generate predictions if files are missing
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

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Ensure base_df has a datetime index as well.
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if "DATE_TIME" in base_df.columns:
            base_df.index = pd.to_datetime(base_df["DATE_TIME"])
        else:
            raise ValueError("base_df does not have a DATE_TIME column.")

    # Align all datasets to the common date range (intersection of their datetime indexes)
    common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found among base, hourly, and daily predictions.")

    # Aquí se recortan los tres datasets al rango común
    base_df = base_df.loc[common_index]         # <-- Alinea el dataset base
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that all datasets have the same number of rows
    if not (len(base_df) == len(hourly_df) == len(daily_df)):
        raise ValueError("After alignment, the number of rows in base, hourly, and daily predictions do not match!")

    # Print aligned date ranges
    print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df, "base_full": base_df_full}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - Before sending data to the strategy plugin, computes and prints a table with MAE and R² for each prediction horizon,
      using as target the CLOSE value from the full base dataset.
    - If config["load_parameters"] is provided, loads candidate parameters from the specified JSON file and evaluates the strategy once.
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd, time
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    from sklearn.metrics import mean_absolute_error, r2_score

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds = datasets["hourly"]
    daily_preds = datasets["daily"]
    base_data = datasets["base"]         # Aligned base (issuance times)
    base_full = datasets["base_full"]      # Full base dataset

    # Calculate error metrics for each prediction horizon

    # For hourly predictions: each column corresponds to the forecast for h hours ahead.
    n_hourly = hourly_preds.shape[1]
    hourly_results = []
    for h in range(1, n_hourly + 1):
        # For each forecast, the target is CLOSE at (timestamp + h hours)
        forecast_times = hourly_preds.index + pd.Timedelta(hours=h)
        actual = base_full.reindex(forecast_times)["CLOSE"]
        # Reassign index so that actual aligns element-wise with pred
        actual.index = hourly_preds.index
        pred = hourly_preds.iloc[:, h - 1]
        valid = actual.notna()
        if valid.sum() == 0:
            mae = None
            r2 = None
        else:
            mae = mean_absolute_error(actual[valid], pred[valid])
            r2 = r2_score(actual[valid], pred[valid])
        hourly_results.append({"Horizon (hours)": h, "MAE": mae, "R2": r2})
    
    df_hourly = pd.DataFrame(hourly_results)

    # For daily predictions: each column corresponds to the forecast for d days ahead (24*d hours).
    n_daily = daily_preds.shape[1]
    daily_results = []
    for d in range(1, n_daily + 1):
        forecast_times = daily_preds.index + pd.Timedelta(hours=24 * d)
        actual = base_full.reindex(forecast_times)["CLOSE"]
        # Reassign index so that actual aligns with the predictions
        actual.index = daily_preds.index
        pred = daily_preds.iloc[:, d - 1]
        valid = actual.notna()
        if valid.sum() == 0:
            mae = None
            r2 = None
        else:
            mae = mean_absolute_error(actual[valid], pred[valid])
            r2 = r2_score(actual[valid], pred[valid])
        daily_results.append({"Horizon (days)": d, "MAE": mae, "R2": r2})
    
    df_daily = pd.DataFrame(daily_results)

    # Print the error metrics tables
    print("\nError Metrics for Hourly Predictions:")
    print(df_hourly.to_string(index=False))
    print("\nError Metrics for Daily Predictions:")
    print(df_daily.to_string(index=False))

    # Additional verification: print final aligned date ranges before sending data to the plugin
    print(f"\nFinal Base dataset date range: {base_data.index.min()} to {base_data.index.max()}")
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Base dataset:       {base_data.shape}")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")

    # Proceed with sending data to the plugin (evaluation or optimization)
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

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

    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

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
