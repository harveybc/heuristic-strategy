import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer
from sklearn.metrics import mean_absolute_error, r2_score
import json

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
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values
            if row_vals.ndim == 0:
                continue
            block_values.extend(row_vals.flatten())
        if block_values:
            blocks.append(block_values)
    if not blocks:
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)

def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Loads uncertainties if provided.
    - Ensures all datasets (base, hourly predictions, daily predictions, uncertainties) 
      are properly aligned to the common date range.
    - Truncates all datasets to config['max_steps'] rows (if specified).
      Additionally, preserves the full base dataset (base_full) for later evaluation.
    """
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load predictor json configuration files if provided
    hourly_predictions_file = None
    daily_predictions_file = None
    uncertainty_hourly_file = None
    uncertainty_daily_file = None

    if config.get("predictor_hourly_config_file"):
        print(f"Loading hourly predictor config from {config['predictor_hourly_config_file']}")
        with open(config["predictor_hourly_config_file"], "r") as f:
            predictor_hourly_config = json.load(f)
        config["hourly_predictions_file"] = predictor_hourly_config.get("output_file")
        config["uncertainty_hourly_file"] = predictor_hourly_config.get("uncertainties_file")

    if config.get("predictor_daily_config_file"):
        print(f"Loading daily predictor config from {config['predictor_daily_config_file']}")
        with open(config["predictor_daily_config_file"], "r") as f:
            predictor_daily_config = json.load(f)
        config["daily_predictions_file"] = predictor_daily_config.get("output_file")
        config["uncertainty_daily_file"] = predictor_daily_config.get("uncertainties_file")
        # Extract the base filename path as the predictor_daily_config_file except its extension
        base_filename = predictor_daily_config.get("results_file")
        base_filename = base_filename.rsplit(".", 1)[0]
        config["save_config"] = base_filename + "_config_out.json"
        config["save_log"] = base_filename + "_debug_log.json"
        config["balance_plot_file"] = base_filename + "_balance_plot.png"
        config["trades_csv_file"] = base_filename + "_trades.csv"
        config["summary_csv_file"] = base_filename + "_summary.csv"
        config["save_parameters"] = base_filename + "_parameters.json"

        

    # Load predictions and base
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)
    print(f"Base dataset loaded: {base_df.shape}")

    # Keep a full copy for later evaluation
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

    # Load uncertainties if available
    uncertainty_hourly_df = None
    uncertainty_daily_df = None


    if config.get("uncertainty_hourly_file"):
        uncertainty_hourly_df = load_csv(config["uncertainty_hourly_file"], headers=headers)
    if config.get("uncertainty_daily_file"):
        uncertainty_daily_df = load_csv(config["uncertainty_daily_file"], headers=headers)

    # Ensure all datasets have a datetime index based on DATE_TIME column.
    def ensure_datetime(df, name):
        if not isinstance(df.index, pd.DatetimeIndex):
            if "DATE_TIME" in df.columns:
                df.index = pd.to_datetime(df["DATE_TIME"])
            else:
                raise ValueError(f"{name} does not have a DATE_TIME column.")
        return df

    base_df = ensure_datetime(base_df, "base_df")
    hourly_df = ensure_datetime(hourly_df, "hourly_df")
    daily_df = ensure_datetime(daily_df, "daily_df")
    if uncertainty_hourly_df is not None:
        uncertainty_hourly_df = ensure_datetime(uncertainty_hourly_df, "uncertainty_hourly_df")
    if uncertainty_daily_df is not None:
        uncertainty_daily_df = ensure_datetime(uncertainty_daily_df, "uncertainty_daily_df")

    # Compute common index across all datasets (only include uncertainties if available)
    common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
    if uncertainty_hourly_df is not None:
        common_index = common_index.intersection(uncertainty_hourly_df.index)
    if uncertainty_daily_df is not None:
        common_index = common_index.intersection(uncertainty_daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found among base, predictions, and uncertainties.")

    # Trim all datasets to the common date range
    base_df = base_df.loc[common_index]
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]
    if uncertainty_hourly_df is not None:
        uncertainty_hourly_df = uncertainty_hourly_df.loc[common_index]
    if uncertainty_daily_df is not None:
        uncertainty_daily_df = uncertainty_daily_df.loc[common_index]

    # Apply max_steps if provided: truncate all datasets to the same number of rows.
    if "max_steps" in config:
        max_steps = config["max_steps"]
        base_df = base_df.iloc[:max_steps]
        hourly_df = hourly_df.iloc[:max_steps]
        daily_df = daily_df.iloc[:max_steps]
        if uncertainty_hourly_df is not None:
            uncertainty_hourly_df = uncertainty_hourly_df.iloc[:max_steps]
        if uncertainty_daily_df is not None:
            uncertainty_daily_df = uncertainty_daily_df.iloc[:max_steps]



    # ----- START: Insert this code block -----

    # Select only the configured columns for each dataframe AFTER alignment and truncation
    # The DATE_TIME column is already set as the index by ensure_datetime

    print("Selecting configured columns...")

    # Select configured hourly prediction columns
    if "hourly_columns" in config:
        try:
            # Ensure all specified columns exist before selection
            missing_cols = [col for col in config["hourly_columns"] if col not in hourly_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required hourly columns: {missing_cols}")
            hourly_df = hourly_df[config["hourly_columns"]]
            print(f"Selected hourly columns: {list(hourly_df.columns)}")
        except KeyError as e:
            raise KeyError(f"Error selecting hourly columns: {e}. Check config['hourly_columns'] and the input file.") from e
        except ValueError as e:
            raise ValueError(f"Error validating hourly columns: {e}") from e
    else:
        print("Warning: 'hourly_columns' not specified in config. Returning all available columns for hourly predictions.")

    # Select configured daily prediction columns
    if "daily_columns" in config:
        try:
            # Ensure all specified columns exist before selection
            missing_cols = [col for col in config["daily_columns"] if col not in daily_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required daily columns: {missing_cols}")
            daily_df = daily_df[config["daily_columns"]]
            print(f"Selected daily columns: {list(daily_df.columns)}")
        except KeyError as e:
            raise KeyError(f"Error selecting daily columns: {e}. Check config['daily_columns'] and the input file.") from e
        except ValueError as e:
            raise ValueError(f"Error validating daily columns: {e}") from e
    else:
        print("Warning: 'daily_columns' not specified in config. Returning all available columns for daily predictions.")

    # Select configured hourly uncertainty columns (if dataframe exists)
    if uncertainty_hourly_df is not None:
        if "uncertainty_hourly_columns" in config:
            try:
                # Ensure all specified columns exist before selection
                missing_cols = [col for col in config["uncertainty_hourly_columns"] if col not in uncertainty_hourly_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required uncertainty hourly columns: {missing_cols}")
                uncertainty_hourly_df = uncertainty_hourly_df[config["uncertainty_hourly_columns"]]
                print(f"Selected uncertainty hourly columns: {list(uncertainty_hourly_df.columns)}")
            except KeyError as e:
                raise KeyError(f"Error selecting uncertainty hourly columns: {e}. Check config['uncertainty_hourly_columns'] and the input file.") from e
            except ValueError as e:
                raise ValueError(f"Error validating uncertainty hourly columns: {e}") from e
        else:
            print("Warning: 'uncertainty_hourly_columns' not specified in config. Returning all available columns for hourly uncertainty.")

    # Select configured daily uncertainty columns (if dataframe exists)
    if uncertainty_daily_df is not None:
        if "uncertainty_daily_columns" in config:
            try:
                # Ensure all specified columns exist before selection
                missing_cols = [col for col in config["uncertainty_daily_columns"] if col not in uncertainty_daily_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required uncertainty daily columns: {missing_cols}")
                uncertainty_daily_df = uncertainty_daily_df[config["uncertainty_daily_columns"]]
                print(f"Selected uncertainty daily columns: {list(uncertainty_daily_df.columns)}")
            except KeyError as e:
                raise KeyError(f"Error selecting uncertainty daily columns: {e}. Check config['uncertainty_daily_columns'] and the input file.") from e
            except ValueError as e:
                raise ValueError(f"Error validating uncertainty daily columns: {e}") from e
        else:
            print("Warning: 'uncertainty_daily_columns' not specified in config. Returning all available columns for daily uncertainty.")

    # ----- END: Insert this code block -----


    # Print aligned date ranges and shapes.  <-- Existing code starts here
    print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
    # ... (rest of the existing print statements and the return statement) ...
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")
    if uncertainty_hourly_df is not None:
        print(f"Aligned Hourly uncertainties range: {uncertainty_hourly_df.index.min()} to {uncertainty_hourly_df.index.max()}")
    if uncertainty_daily_df is not None:
        print(f"Aligned Daily uncertainties range: {uncertainty_daily_df.index.min()} to {uncertainty_daily_df.index.max()}")

    return {
        "hourly": hourly_df,
        "daily": daily_df,
        "base": base_df,
        "base_full": base_df_full,
        "uncertainty_hourly": uncertainty_hourly_df,
        "uncertainty_daily": uncertainty_daily_df
    }

def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - Computes and prints error metrics for each prediction horizon.
    - If config["load_parameters"] is provided, loads candidate parameters and evaluates the strategy once.
    - Otherwise, runs full optimization via run_optimizer().
    - Renames the balance plot and saves trades and summary CSV files.
    - Saves best parameters if in optimization mode.
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
    base_data = datasets["base"]         # Aligned base dataset
    base_full = datasets["base_full"]      # Full base dataset

    # Inject processed uncertainties into the config so the plugin can use them
    config["uncertainty_hourly"] = datasets.get("uncertainty_hourly")
    config["uncertainty_daily"] = datasets.get("uncertainty_daily")

    # Calculate error metrics for hourly predictions
    n_hourly = hourly_preds.shape[1]
    hourly_results = []
    for h in range(1, n_hourly + 1):
        forecast_times = hourly_preds.index + pd.Timedelta(hours=h)
        actual = base_full.reindex(forecast_times)["CLOSE"]
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

    # Calculate error metrics for daily predictions
    n_daily = daily_preds.shape[1]
    daily_results = []
    for d in range(1, n_daily + 1):
        forecast_times = daily_preds.index + pd.Timedelta(hours=24 * d)
        actual = base_full.reindex(forecast_times)["CLOSE"]
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

    print("\nError Metrics for Hourly Predictions:")
    print(df_hourly.to_string(index=False))
    print("\nError Metrics for Daily Predictions:")
    print(df_daily.to_string(index=False))

    print(f"\nFinal Base dataset date range: {base_data.index.min()} to {base_data.index.max()}")
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Base dataset:       {base_data.shape}")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")

    # Proceed with evaluation or optimization using the processed uncertainties as well.
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
            result = plugin.evaluate_candidate(best_ind, data, preds, daily, cfg)
            if isinstance(result, tuple) and len(result) == 2:
                profit, stats = result
            else:
                profit = result[0] if isinstance(result, tuple) else result
                stats = {}

            # Unpack result into profit and stats
            profit, stats = result

            # Build trading_info with profit plus one entry per stat key/value
            trading_info = {
                "profit": profit,
                **stats,
                "best_parameters": {
                    "profit_threshold": candidate[0],
                    "tp_multiplier":    candidate[1],
                    "sl_multiplier":    candidate[2],
                    "lower_rr_threshold": candidate[3],
                    "upper_rr_threshold": candidate[4],
                    "time_horizon":     candidate[5]
                }
            }
        else:
            trading_info = {}
    else:
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            # Run optimizer and then flatten the 'stats' dict into top-level keys
            _raw_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            _stats = _raw_info.pop("stats", {}) or {}
            trading_info = {"initial_capital":10000,**_raw_info, **_stats}
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
            info = trading_info.copy()
            info.pop("best_parameters", None)
            df = pd.DataFrame([info])
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
