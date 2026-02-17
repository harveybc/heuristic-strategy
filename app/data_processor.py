import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from sklearn.metrics import mean_absolute_error, r2_score
import json

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer plugin provided at runtime.
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

def create_daily_predictions(df, horizon, bars_per_day=24):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets
    t+bars_per_day, t+2*bars_per_day, ... t+horizon*bars_per_day.
    Works on pandas.Series or single-column DataFrame.

    Args:
        bars_per_day: Number of bars in one day (24 for 1h, 6 for 4h).
    """
    # If someone passed a single-column DataFrame, pull out the Series
    if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        series = df  # assume it's a Series

    nrows = len(series)
    required_rows = horizon * bars_per_day

    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    for i in range(nrows - required_rows):
        block_values = []
        for d in range(1, horizon + 1):
            idx = i + d * bars_per_day
            if idx >= nrows:
                break
            val = series.iloc[idx]
            # if it's array‐like, flatten; otherwise just append the scalar
            if isinstance(val, (np.ndarray, pd.Series, list)):
                block_values.extend(np.asarray(val).flatten())
            else:
                block_values.append(val)
        if block_values:
            blocks.append(block_values)

    if not blocks:
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    daily_idx = series.index[: (nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)


# =============================================================================
# NEW: Flexible generation at uniform offsets
# =============================================================================
def _compute_uniform_offsets(max_horizon: int, num_predictions: int):
    """
    Compute uniformly spaced integer offsets between 1 and max_horizon (inclusive).
    Requires 1 <= num_predictions <= max_horizon. Returns a sorted list of ints.
    Example: max_horizon=6, num_predictions=3 -> [1, 3, 6]
    """
    if max_horizon is None or num_predictions is None:
        return None
    if not isinstance(max_horizon, int) or not isinstance(num_predictions, int):
        raise ValueError("max_horizon and num_predictions must be integers.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1.")
    if num_predictions < 1:
        raise ValueError("num_predictions must be >= 1.")
    if num_predictions > max_horizon:
        raise ValueError("num_predictions must be <= max_horizon to avoid duplicate offsets.")

    if num_predictions == 1:
        return [max_horizon]

    # Use linspace to include endpoints 1 and max_horizon
    offs = np.linspace(1, max_horizon, num=num_predictions)
    offs = np.round(offs).astype(int).tolist()

    # Ensure strictly increasing and bounded within [1, max_horizon]
    offs[0] = 1
    offs[-1] = max_horizon
    # Deduplicate while preserving order
    unique_offs = []
    for o in offs:
        if 1 <= o <= max_horizon and (not unique_offs or o != unique_offs[-1]):
            unique_offs.append(o)
    if len(unique_offs) != num_predictions:
        # Fallback to deterministic spacing without duplicates
        step = max(1, max_horizon // (num_predictions - 1))
        unique_offs = list(range(1, 1 + step * (num_predictions - 1), step))
        unique_offs[-1] = max_horizon
    return unique_offs


def _create_predictions_at_offsets(df_or_series, offsets):
    """
    Build a prediction matrix at specified hour-based offsets relative to each row.
    Offsets are positive integers (hours ahead). Works with Series or single-col DF.
    Returns a DataFrame with columns aligned to offsets.
    """
    # Normalize to Series of base values
    if isinstance(df_or_series, pd.DataFrame) and df_or_series.shape[1] == 1:
        series = df_or_series.iloc[:, 0]
    elif isinstance(df_or_series, pd.DataFrame):
        # If a multi-column DF is passed, use CLOSE if available; else first column
        series = df_or_series["CLOSE"] if "CLOSE" in df_or_series.columns else df_or_series.iloc[:, 0]
    else:
        series = df_or_series

    max_off = int(max(offsets))
    nrows = len(series)
    if nrows <= max_off:
        print("Warning: Not enough rows to create predictions at requested offsets. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    for i in range(nrows - max_off):
        row_vals = []
        for off in offsets:
            idx = i + int(off)
            val = series.iloc[idx]
            if isinstance(val, (np.ndarray, pd.Series, list)):
                row_vals.extend(np.asarray(val).flatten())
            else:
                row_vals.append(val)
        blocks.append(row_vals)

    idx_out = series.index[: nrows - max_off]
    df_out = pd.DataFrame(blocks, index=idx_out)
    return df_out


def _apply_gaussian_noise(df: pd.DataFrame, mean: float, std: float, label: str = "") -> pd.DataFrame:
    """
    Apply element-wise Gaussian noise to a numeric DataFrame.
    If both mean and std are 0.0, returns df unchanged.
    """
    try:
        mu = float(mean or 0.0)
        sd = float(std or 0.0)
    except Exception:
        mu, sd = 0.0, 0.0
    if mu == 0.0 and sd == 0.0:
        return df
    if df is None or df.empty:
        return df
    noise = np.random.normal(loc=mu, scale=sd, size=df.shape)
    df_noisy = df.copy()
    df_noisy[:] = df_noisy.values + noise
    tag = f" ({label})" if label else ""
    print(f"Applied Gaussian noise{tag}: mean={mu}, std={sd} to predictions with shape {df.shape}.")
    return df_noisy

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
    base_filename = ""
    prefix = config.get("prefix", "_best_daily")
    # If Gaussian noise is enabled (non-zero), append parameters to filenames
    gmu = float(config.get("gaussian_noise_mean", 0.0) or 0.0)
    gsd = float(config.get("gaussian_noise_stddev", 0.0) or 0.0)
    noise_suffix = ""
    if abs(gmu) > 0.0 or abs(gsd) > 0.0:
        # keep it filename-safe; small rounding for readability
        noise_suffix = f"_gn_mu{gmu:.4f}_sd{gsd:.4f}"
        # also store for downstream reference
        config["noise_suffix"] = noise_suffix
    if config.get("predictor_hourly_config_file"):
        print(f"Loading hourly predictor config from {config['predictor_hourly_config_file']}")
        with open(config["predictor_hourly_config_file"], "r") as f:
            predictor_hourly_config = json.load(f)
        config["hourly_predictions_file"] = predictor_hourly_config.get("output_file")
        config["uncertainty_hourly_file"] = predictor_hourly_config.get("uncertainties_file")
        # Extract the base filename path as the predictor_daily_config_file except its extension
        # and append the prefix to the filename
        # specify the base directory to save the results(hourly/daily) from the config use_hourly config parameter
        if config.get("use_hourly", False):
            base_filename = predictor_hourly_config.get("results_file")
        base_filename = base_filename.rsplit(".", 1)[0]
        config["save_config"] = base_filename + prefix + noise_suffix + "_config_out.json"
        config["save_log"] = base_filename + prefix + noise_suffix + "_debug_log.json"
        config["balance_plot_file"] = base_filename + prefix + noise_suffix + "_balance_plot.png"
        config["trades_csv_file"] = base_filename + prefix + noise_suffix + "_trades.csv"
        config["summary_csv_file"] = base_filename + prefix + noise_suffix + "_summary.csv"
        if config.get("load_parameters"):
            config["save_parameters"] = base_filename + prefix + noise_suffix + "_parameters.json"

        

    if config.get("predictor_daily_config_file"):
        print(f"Loading daily predictor config from {config['predictor_daily_config_file']}")
        with open(config["predictor_daily_config_file"], "r") as f:
            predictor_daily_config = json.load(f)
        config["daily_predictions_file"] = predictor_daily_config.get("output_file")
        config["uncertainty_daily_file"] = predictor_daily_config.get("uncertainties_file")
        # Extract the base filename path as the predictor_daily_config_file except its extension
        # and append the prefix to the filename
        # specify the base directory to save the results(hourly/daily) from the config use_hourly config parameter
        if not(config.get("use_hourly", False)):
            base_filename = predictor_daily_config.get("results_file")
        base_filename = base_filename.rsplit(".", 1)[0]
        config["save_config"] = base_filename + prefix + noise_suffix + "_config_out.json"
        config["save_log"] = base_filename + prefix + noise_suffix + "_debug_log.json"
        config["balance_plot_file"] = base_filename + prefix + noise_suffix + "_balance_plot.png"
        config["trades_csv_file"] = base_filename + prefix + noise_suffix + "_trades.csv"
        config["summary_csv_file"] = base_filename + prefix + noise_suffix + "_summary.csv"
        if config.get("load_parameters"):
            config["save_parameters"] = base_filename + prefix + noise_suffix + "_parameters.json"

        

    # Load predictions and base
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)
    print(f"Base dataset loaded: {base_df.shape}")

    # If no predictor config files set names, append noise suffix to any preconfigured output filenames
    def _append_suffix_to_filename(path: str, suffix: str) -> str:
        if not path or not suffix:
            return path
        if suffix in path:
            return path
        import os
        base, ext = os.path.splitext(path)
        return f"{base}{suffix}{ext}"

    if noise_suffix:
        for k in [
            "save_config",
            "save_log",
            "balance_plot_file",
            "trades_csv_file",
            "summary_csv_file",
            "save_parameters",
        ]:
            if config.get(k):
                config[k] = _append_suffix_to_filename(config[k], noise_suffix)

    # Keep a full copy for later evaluation
    base_df_full = base_df.copy()

    # Auto-generate predictions if files are missing
    # … inside process_data() …

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        # New dynamic configuration (optional)
        st_max = config.get("short_term_max_horizon")
        st_n = config.get("short_term_num_predictions")
        if st_max and st_n:
            print("Auto-generating hourly predictions (dynamic offsets)...")
            offsets_h = _compute_uniform_offsets(int(st_max), int(st_n))
            hourly_df = _create_predictions_at_offsets(base_df["CLOSE"], offsets_h)
            # Generate and persist column names into config for downstream selection
            hourly_cols = [f"Prediction_H{h}" for h in offsets_h]
            hourly_df.columns = hourly_cols
            config["hourly_columns"] = hourly_cols
            # Prepare matching uncertainty column names for later use
            config["uncertainty_hourly_columns"] = [f"Uncertainty_H{h}" for h in offsets_h]
            # Apply Gaussian noise if requested
            hourly_df = _apply_gaussian_noise(
                hourly_df,
                config.get("gaussian_noise_hourly_mean", config.get("gaussian_noise_mean", 0.0)),
                config.get("gaussian_noise_hourly_stddev", config.get("gaussian_noise_stddev", 0.0)),
                label="hourly",
            )
        else:
            if "time_horizon" not in config or not config["time_horizon"]:
                raise ValueError("time_horizon must be provided when auto-generating predictions.")
            print("Auto-generating hourly predictions...")
            # only use the CLOSE series so each block is horizon-length, not horizon * ncols
            hourly_df = create_hourly_predictions(base_df["CLOSE"], config["time_horizon"])
            if config.get("hourly_columns"):
                hourly_df.columns = config["hourly_columns"]
            else:
                hourly_df.columns = [f"Prediction_H{i}" for i in range(1, config["time_horizon"] + 1)]
            # Apply Gaussian noise if requested
            hourly_df = _apply_gaussian_noise(
                hourly_df,
                config.get("gaussian_noise_hourly_mean", config.get("gaussian_noise_mean", 0.0)),
                config.get("gaussian_noise_hourly_stddev", config.get("gaussian_noise_stddev", 0.0)),
                label="hourly",
            )

    if daily_df is None:
        # New dynamic configuration (optional)
        lt_max = config.get("long_term_max_horizon")
        lt_n = config.get("long_term_num_predictions")
        if lt_max and lt_n:
            print("Auto-generating daily (long-term) predictions (dynamic offsets)...")
            offsets_d = _compute_uniform_offsets(int(lt_max), int(lt_n))
            daily_df = _create_predictions_at_offsets(base_df["CLOSE"], offsets_d)
            daily_cols = [f"Prediction_H{h}" for h in offsets_d]
            daily_df.columns = daily_cols
            config["daily_columns"] = daily_cols
            config["uncertainty_daily_columns"] = [f"Uncertainty_H{h}" for h in offsets_d]
            # Apply Gaussian noise if requested
            daily_df = _apply_gaussian_noise(
                daily_df,
                config.get("gaussian_noise_daily_mean", config.get("gaussian_noise_mean", 0.0)),
                config.get("gaussian_noise_daily_stddev", config.get("gaussian_noise_stddev", 0.0)),
                label="daily",
            )
        else:
            if "time_horizon" not in config or not config["time_horizon"]:
                raise ValueError("time_horizon must be provided when auto-generating predictions.")
            print("Auto-generating daily predictions...")
            # likewise here
            bpd = config.get("bars_per_day", 24)
            daily_df = create_daily_predictions(base_df["CLOSE"], config["time_horizon"], bars_per_day=bpd)
            if config.get("daily_columns"):
                daily_df.columns = config["daily_columns"]
            else:
                daily_df.columns = [f"Prediction_H{bpd*i}" for i in range(1, config["time_horizon"] + 1)]
            # Apply Gaussian noise if requested
            daily_df = _apply_gaussian_noise(
                daily_df,
                config.get("gaussian_noise_daily_mean", config.get("gaussian_noise_mean", 0.0)),
                config.get("gaussian_noise_daily_stddev", config.get("gaussian_noise_stddev", 0.0)),
                label="daily",
            )

    # Load uncertainties if available
    uncertainty_hourly_df = None
    uncertainty_daily_df = None


    if config.get("uncertainty_hourly_file"):
        uncertainty_hourly_df = load_csv(config["uncertainty_hourly_file"], headers=headers)
    else:
        # set a constant value for the uncertainty of config["default_uncertainty_short_term"] with the same size as the hourly_df
        if hourly_df is not None:
            uncertainty_hourly_df = pd.DataFrame(config["default_uncertainty_short_term"], index=hourly_df.index, columns=hourly_df.columns)
            print(f"Uncertainty hourly dataset created with constant value: {config['default_uncertainty_short_term']}")
            # assign the column names to the uncertainty_hourly_df with the specified in: config["uncertainty_hourly_columns"]
            if config.get("uncertainty_hourly_columns"):
                uncertainty_hourly_df.columns = config["uncertainty_hourly_columns"]
            else:
                # fallback for legacy hourly: 1..time_horizon
                uncertainty_hourly_df.columns = [f"Uncertainty_H{i}" for i in range(1, config["time_horizon"] + 1)]
            # assign the same DATE_TIME column to the uncertainty_hourly_df with the same index as the hourly_df
            uncertainty_hourly_df["DATE_TIME"] = hourly_df.index
            uncertainty_hourly_df.set_index("DATE_TIME", inplace=True)

    

    if config.get("uncertainty_daily_file"):
        uncertainty_daily_df = load_csv(config["uncertainty_daily_file"], headers=headers)
    else:
        # set a constant value for the uncertainty of config["default_uncertainty_long_term"] with the same size as the daily_df
        if daily_df is not None:
            uncertainty_daily_df = pd.DataFrame(config["default_uncertainty_long_term"], index=daily_df.index, columns=daily_df.columns)
            print(f"Uncertainty daily dataset created with constant value: {config['default_uncertainty_long_term']}")
            # assign the column names to the uncertainty_daily_df with the specified in: config["uncertainty_daily_columns"]
            if config.get("uncertainty_daily_columns"):
                uncertainty_daily_df.columns = config["uncertainty_daily_columns"]
            else:
                # fallback for legacy daily: 24,48,...
                _bpd = config.get("bars_per_day", 24)
                uncertainty_daily_df.columns = [f"Uncertainty_H{_bpd*i}" for i in range(1, config["time_horizon"] + 1)]
            # assign the same DATE_TIME column to the uncertainty_daily_df with the same index as the daily_df
            uncertainty_daily_df["DATE_TIME"] = daily_df.index
            uncertainty_daily_df.set_index("DATE_TIME", inplace=True)
            


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

def run_processing_pipeline(config, plugin, optimizer_plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - Computes and prints error metrics for each prediction horizon.
    - If config["load_parameters"] is provided, loads candidate parameters and evaluates the strategy once.
    - Otherwise, runs full optimization via the provided optimizer plugin.
    - Renames the balance plot and saves trades and summary CSV files.
    - Saves best parameters if in optimization mode.
    """
    import json, os, pandas as pd, time
    from sklearn.metrics import mean_absolute_error, r2_score

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    def _apply_neat_prediction_defaults(cfg):
        optimizer_name = str(cfg.get("optimizer_plugin", "")).strip().lower()
        class_name = getattr(getattr(optimizer_plugin, "__class__", None), "__name__", "").lower()
        is_neat = "neat" in optimizer_name or "neat" in class_name
        if not is_neat:
            return

        bpd = cfg.get("bars_per_day", 24)
        overrides = {
            "short_term_max_horizon": bpd,       # 1 day of bars
            "short_term_num_predictions": bpd,
            "long_term_max_horizon": bpd * 5,    # 5 days of bars
            "long_term_num_predictions": bpd * 5,
        }
        applied = []
        for key, value in overrides.items():
            if cfg.get(key) is None:
                cfg[key] = value
                applied.append(f"{key}={value}")
        if applied:
            print(
                "[NEAT][Config] Applied fixed prediction horizons for NEAT optimizer: "
                + ", ".join(applied)
            )

    _apply_neat_prediction_defaults(config)

    datasets = process_data(config)
    hourly_preds = datasets["hourly"]
    daily_preds = datasets["daily"]
    base_data = datasets["base"]         # Aligned base dataset
    base_full = datasets["base_full"]      # Full base dataset

    # Inject processed uncertainties into the config so the plugin can use them
    config["uncertainty_hourly"] = datasets.get("uncertainty_hourly")
    config["uncertainty_daily"] = datasets.get("uncertainty_daily")

    # Calculate error metrics for hourly predictions
    # Use offsets parsed from column names (e.g., Prediction_H1, Prediction_H3, ...)
    import re
    def _parse_col_offsets(cols):
        offs = []
        for i, c in enumerate(cols, start=1):
            m = re.search(r"H(\d+)$", str(c))
            offs.append(int(m.group(1)) if m else i)
        return offs

    hourly_offsets = _parse_col_offsets(hourly_preds.columns)
    hourly_results = []
    for idx, off in enumerate(hourly_offsets):
        forecast_times = hourly_preds.index + pd.Timedelta(hours=int(off))
        actual = base_full.reindex(forecast_times)["CLOSE"]
        actual.index = hourly_preds.index
        pred = hourly_preds.iloc[:, idx]
        valid = actual.notna()
        if valid.sum() == 0:
            mae = None
            r2 = None
        else:
            mae = mean_absolute_error(actual[valid], pred[valid])
            r2 = r2_score(actual[valid], pred[valid])
        hourly_results.append({"Horizon (hours)": int(off), "MAE": mae, "R2": r2})
    df_hourly = pd.DataFrame(hourly_results)

    # Calculate error metrics for daily predictions
    daily_offsets = _parse_col_offsets(daily_preds.columns)
    daily_results = []
    # Determine if classic daily (multiples of 24) or dynamic hours
    is_classic_daily = all(o % 24 == 0 for o in daily_offsets)
    for idx, off in enumerate(daily_offsets):
        forecast_times = daily_preds.index + pd.Timedelta(hours=int(off))
        actual = base_full.reindex(forecast_times)["CLOSE"]
        actual.index = daily_preds.index
        pred = daily_preds.iloc[:, idx]
        valid = actual.notna()
        if valid.sum() == 0:
            mae = None
            r2 = None
        else:
            mae = mean_absolute_error(actual[valid], pred[valid])
            r2 = r2_score(actual[valid], pred[valid])
        key = "Horizon (days)" if is_classic_daily else "Horizon (hours)"
        val = (off // 24) if is_classic_daily else off
        daily_results.append({key: int(val), "MAE": mae, "R2": r2})
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
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"])
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            optimizer_plugin.init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = optimizer_plugin.evaluate_individual(candidate)
            if isinstance(result, tuple) and len(result) == 2:
                profit, stats = result
            else:
                profit = result[0] if isinstance(result, tuple) else result
                stats = {}

            # Build trading_info with profit plus one entry per stat key/value
            trading_info = {
                "profit": profit,
                **stats,
                "best_parameters": {
                    "profit_threshold": candidate[0],
                    "tp_multiplier":    candidate[1],
                    "sl_multiplier":    candidate[2],
                    "lower_rr_threshold": candidate[3],
                    "upper_rr_threshold": candidate[4]
                    
                }
            }
        else:
            trading_info = {}
    else:
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            # For optimization, provide supersets so per-candidate selection doesn't require recomputation
            # Short-term superset: H1..H48, Long-term superset: H1..H144 (all in hours)
            try:
                hourly_cols = list(hourly_preds.columns)
                daily_cols = list(daily_preds.columns)
                def build_supersets(base_df):
                    hs = _create_predictions_at_offsets(base_df["CLOSE"], hourly_offsets)
                    if hs.empty:
                        raise ValueError(
                            "Validation/Test hourly prediction superset is empty. "
                            "Ensure the dataset has more rows than the maximum hourly horizon."
                        )
                    hs.columns = hourly_cols
                    ds = _create_predictions_at_offsets(base_df["CLOSE"], daily_offsets)
                    if ds.empty:
                        raise ValueError(
                            "Validation/Test daily prediction superset is empty. "
                            "Ensure the dataset has more rows than the maximum daily horizon."
                        )
                    ds.columns = daily_cols
                    # Apply Gaussian noise to supersets if configured
                    hs = _apply_gaussian_noise(
                        hs,
                        config.get("gaussian_noise_mean", 0.0),
                        config.get("gaussian_noise_stddev", 0.0),
                        label="superset-hourly",
                    )
                    ds = _apply_gaussian_noise(
                        ds,
                        config.get("gaussian_noise_mean", 0.0),
                        config.get("gaussian_noise_stddev", 0.0),
                        label="superset-daily",
                    )
                    common_idx = base_df.index.intersection(hs.index).intersection(ds.index)
                    if common_idx.empty:
                        raise ValueError(
                            "Validation/Test supersets have no overlapping timestamps after alignment."
                        )
                    return base_df.loc[common_idx], hs.loc[common_idx], ds.loc[common_idx]

                # Train supersets
                base_for_opt, hourly_superset, daily_superset = build_supersets(base_data)

                # Validation supersets (optional)
                base_val = None; hourly_val = None; daily_val = None
                if config.get("validation_dataset_file"):
                    base_val_raw = load_csv(config["validation_dataset_file"], headers=config.get("headers", True))
                    if not isinstance(base_val_raw.index, pd.DatetimeIndex):
                        if "DATE_TIME" in base_val_raw.columns:
                            base_val_raw.index = pd.to_datetime(base_val_raw["DATE_TIME"])
                        else:
                            raise ValueError("Validation dataset does not have a DATE_TIME column.")
                    base_val_raw = base_val_raw.iloc[: config.get("max_steps", len(base_val_raw))]
                    base_val, hourly_val, daily_val = build_supersets(base_val_raw)

                # Test supersets (optional)
                base_test = None; hourly_test = None; daily_test = None
                if config.get("test_dataset_file"):
                    base_test_raw = load_csv(config["test_dataset_file"], headers=config.get("headers", True))
                    if not isinstance(base_test_raw.index, pd.DatetimeIndex):
                        if "DATE_TIME" in base_test_raw.columns:
                            base_test_raw.index = pd.to_datetime(base_test_raw["DATE_TIME"])
                        else:
                            raise ValueError("Test dataset does not have a DATE_TIME column.")
                    base_test_raw = base_test_raw.iloc[: config.get("max_steps", len(base_test_raw))]
                    base_test, hourly_test, daily_test = build_supersets(base_test_raw)
            except Exception as e:
                print(f"Warning: failed to generate prediction supersets for optimization, falling back to current frames: {e}")
                base_for_opt = base_data
                hourly_superset = hourly_preds
                daily_superset = daily_preds
                base_val = hourly_val = daily_val = None
                base_test = hourly_test = daily_test = None

            # Run optimizer and then flatten the 'stats' dict into top-level keys
            _raw_info = optimizer_plugin.run_optimizer(
                plugin,
                base_for_opt,
                hourly_superset,
                daily_superset,
                config,
                base_val,
                hourly_val,
                daily_val,
                base_test,
                hourly_test,
                daily_test,
            )
            _stats = _raw_info.pop("stats", {}) or {}

            # Flatten validation and test stats if present
            if "validation_stats" in _raw_info:
                val_stats = _raw_info.pop("validation_stats")
                if val_stats:
                    for k, v in val_stats.items():
                        _raw_info[f"val_{k}"] = v
            
            if "test_stats" in _raw_info:
                test_stats = _raw_info.pop("test_stats")
                if test_stats:
                    for k, v in test_stats.items():
                        _raw_info[f"test_{k}"] = v

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
            # Add Gaussian noise parameters to summary columns
            info["gaussian_noise_mean"] = config.get("gaussian_noise_mean", 0.0)
            info["gaussian_noise_stddev"] = config.get("gaussian_noise_stddev", 0.0)
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
