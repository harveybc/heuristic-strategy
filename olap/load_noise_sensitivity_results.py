#!/usr/bin/env python3  # Specify that this script should be run with Python 3 when executed directly.
# -*- coding: utf-8 -*-  # Declare UTF-8 encoding for this source file.

"""
Loader script for the `noise_sensitivity_olap` PostgreSQL database.

This script:
- Walks a directory looking for summary CSV files matching the Gaussian noise pattern.
- For each summary, loads the associated configuration and parameters JSON files.
- Inserts or updates records in the `experiments` and `performance` tables.

The script assumes filenames of the form:
- summary_gn_mu0.0020_sd0.0020.csv
- config_out_gn_mu0.0020_sd0.0020.json
- parameters_gn_mu0.0020_sd0.0020.json
"""

import os  # Import os to work with file system paths and environment variables.
import re  # Import re to use regular expressions for parsing mu and sd from filenames.
import json  # Import json to serialize and deserialize JSON data.
import ast  # Import ast to safely evaluate literal Python structures from strings.
from typing import Any, Dict, Optional, Tuple  # Import typing helpers for clarity and type hints.

import pandas as pd  # Import pandas to read CSV files into DataFrames for convenient handling.
from sqlalchemy import create_engine, text  # Import SQLAlchemy tools for database connections and SQL execution.

# Define the environment variable name that stores the database URL.
NOISE_OLAP_DB_URL_ENV: str = "NOISE_OLAP_DB_URL"

# Define a default database URL to use if the environment variable is not set.
# IMPORTANT: Replace 'your_pg_user' and 'your_pg_password' with valid credentials in your environment.
DEFAULT_DB_URL: str = (
    "postgresql+psycopg2://your_pg_user:your_pg_password@localhost:5432/noise_sensitivity_olap"
)

# Compile a regular expression to extract Gaussian noise parameters from filenames.
# This pattern:
# - Looks for "gn_mu" followed by a number with optional decimal part.
# - Then an underscore "_" or dot ".".
# - Then "sd" followed by a number with optional decimal part.
# Critically, it does NOT allow an extra trailing dot, so it will not consume the "." before ".csv".
NOISE_PATTERN = re.compile(
    r"gn_mu(?P<mu>\d+(?:\.\d+)?)[_.]sd(?P<sd>\d+(?:\.\d+)?)",  # Capture groups for mu and sd without trailing extension dot.
    re.IGNORECASE,  # Ignore case to be tolerant of filename variations.
)


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine for the noise_sensitivity_olap database.

    The function reads the connection string from the NOISE_OLAP_DB_URL environment variable,
    falling back to DEFAULT_DB_URL if the variable is not defined.
    """
    # Read the database URL from the environment variable or use the default if not set.
    db_url: str = os.getenv(NOISE_OLAP_DB_URL_ENV, DEFAULT_DB_URL)
    # Create a SQLAlchemy engine using the resolved database URL.
    engine = create_engine(db_url, echo=False, future=True)
    # Return the created engine to the caller.
    return engine


def parse_noise_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse the mu and sd components from a filename using the NOISE_PATTERN regex.

    Args:
        filename: Basename of the file (without directory), e.g. 'summary_gn_mu0.0020_sd0.0020.csv'.

    Returns:
        A tuple (mu_str, sd_str) if the pattern is found, otherwise None.

    This function also strips any trailing dots from the captured strings as an extra safety measure.
    """
    # Search the precompiled noise pattern in the provided filename string.
    match = NOISE_PATTERN.search(filename)
    # If there is no match for the noise pattern, return None to indicate failure.
    if not match:
        return None
    # Extract the raw mu component from the 'mu' named group.
    mu_str: str = match.group("mu")
    # Extract the raw sd component from the 'sd' named group.
    sd_str: str = match.group("sd")
    # Strip any trailing dot characters from mu_str to guard against malformed captures.
    mu_str = mu_str.rstrip(".")
    # Strip any trailing dot characters from sd_str to guard against malformed captures.
    sd_str = sd_str.rstrip(".")
    # Return the cleaned mu and sd strings as a tuple.
    return mu_str, sd_str


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file from the given path if the file exists.

    Args:
        path: Absolute or relative path to the JSON file.

    Returns:
        The parsed JSON object as a dictionary, or None if the file does not exist.
    """
    # Check if the file exists on the filesystem.
    if not os.path.isfile(path):
        # If the file does not exist, return None to signal that it is missing.
        return None
    # Open the file in read mode with UTF-8 encoding to handle text safely.
    with open(path, "r", encoding="utf-8") as f:
        # Deserialize the JSON content from the file into a Python object.
        data = json.load(f)
    # Return the parsed JSON object (expected to be a dict).
    return data


def safe_parse_stats_column(value: Any) -> Dict[str, Any]:
    """
    Safely parse a stats column such as 'validation_stats' or 'test_stats'.

    The column is expected to be a string representation of a Python dictionary.
    This function uses ast.literal_eval to safely convert the string into a dict.

    Args:
        value: The stats column value from the DataFrame row.

    Returns:
        A dictionary containing the parsed stats, or an empty dict on failure or if the value is NaN.
    """
    # If the value is None or a NaN float from pandas, return an empty dictionary.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    # If the value is already a dictionary, return it unchanged.
    if isinstance(value, dict):
        return value
    try:
        # Use ast.literal_eval to safely evaluate the string into a Python object.
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        # If parsing fails due to syntax or value errors, return an empty dictionary.
        return {}
    # If the parsed object is not a dict, normalize to an empty dictionary for consistency.
    if not isinstance(parsed, dict):
        return {}
    # Return the successfully parsed dictionary.
    return parsed


def upsert_experiment(
    engine: Any,
    experiment_key: str,
    mu_str: str,
    sd_str: str,
    summary_row: Dict[str, Any],
    config_data: Optional[Dict[str, Any]],
    params_data: Optional[Dict[str, Any]],
) -> int:
    """
    Insert or update an experiment row and return the experiment ID.

    This function:
    - Uses the experiment_key to perform an upsert via PostgreSQL's ON CONFLICT.
    - Persists both explicit numeric fields and raw JSON for config and parameters.

    Args:
        engine: SQLAlchemy engine connected to the target database.
        experiment_key: Unique key based on noise parameters, e.g. 'gn_mu0.0020_sd0.0020'.
        mu_str: String representation of the Gaussian noise mean parsed from filename.
        sd_str: String representation of the Gaussian noise standard deviation parsed from filename.
        summary_row: Dictionary with metrics from the summary CSV row.
        config_data: Parsed configuration JSON dictionary, or None if missing.
        params_data: Parsed parameters JSON dictionary, or None if missing.

    Returns:
        The integer primary key of the experiment row in the experiments table.
    """
    # Read the Gaussian noise mean from the summary row if present.
    mu_val = summary_row.get("gaussian_noise_mean")
    # If the summary row does not contain the mean, fall back to converting mu_str to float.
    if mu_val is None:
        mu_val = float(mu_str)
    # Read the Gaussian noise standard deviation from the summary row if present.
    sd_val = summary_row.get("gaussian_noise_stddev")
    # If the summary row does not contain the stddev, fall back to converting sd_str to float.
    if sd_val is None:
        sd_val = float(sd_str)

    # Initialize configuration-derived fields with default None values.
    base_dataset_file = None
    prefix = None
    max_trades_per_5days = None
    # If a configuration dictionary is available, extract relevant fields.
    if config_data is not None:
        # Extract the base dataset file path from the configuration if it exists.
        base_dataset_file = config_data.get("base_dataset_file")
        # Extract any prefix field used to differentiate scenarios or datasets.
        prefix = config_data.get("prefix")
        # Extract the maximum allowed trades per 5-day window if defined.
        max_trades_per_5days = config_data.get("max_trades_per_5days")

    # Initialize parameter-derived fields with default None values.
    profit_threshold = None
    tp_multiplier = None
    sl_multiplier = None
    lower_rr_threshold = None
    upper_rr_threshold = None
    short_term_max_horizon = None
    short_term_num_predictions = None
    long_term_max_horizon = None
    long_term_num_predictions = None

    # If a parameters dictionary is available, extract the known strategy parameters.
    if params_data is not None:
        # Extract the profit threshold parameter if present.
        profit_threshold = params_data.get("profit_threshold")
        # Extract the take-profit multiplier for trade exits if present.
        tp_multiplier = params_data.get("tp_multiplier")
        # Extract the stop-loss multiplier for trade exits if present.
        sl_multiplier = params_data.get("sl_multiplier")
        # Extract the lower risk-reward threshold used in the optimization if present.
        lower_rr_threshold = params_data.get("lower_rr_threshold")
        # Extract the upper risk-reward threshold used in the optimization if present.
        upper_rr_threshold = params_data.get("upper_rr_threshold")
        # Extract the maximum horizon for short-term forecasts if present.
        short_term_max_horizon = params_data.get("short_term_max_horizon")
        # Extract the number of short-term predictions used per decision if present.
        short_term_num_predictions = params_data.get("short_term_num_predictions")
        # Extract the maximum horizon for long-term forecasts if present.
        long_term_max_horizon = params_data.get("long_term_max_horizon")
        # Extract the number of long-term predictions used per decision if present.
        long_term_num_predictions = params_data.get("long_term_num_predictions")

    # Define the SQL upsert statement for the experiments table with ON CONFLICT.
    sql = text(
        """
        INSERT INTO experiments (
            experiment_key,
            gaussian_noise_mean,
            gaussian_noise_stddev,
            base_dataset_file,
            prefix,
            max_trades_per_5days,
            config_json,
            parameters_json,
            profit_threshold,
            tp_multiplier,
            sl_multiplier,
            lower_rr_threshold,
            upper_rr_threshold,
            short_term_max_horizon,
            short_term_num_predictions,
            long_term_max_horizon,
            long_term_num_predictions
        ) VALUES (
            :experiment_key,
            :gaussian_noise_mean,
            :gaussian_noise_stddev,
            :base_dataset_file,
            :prefix,
            :max_trades_per_5days,
            :config_json,
            :parameters_json,
            :profit_threshold,
            :tp_multiplier,
            :sl_multiplier,
            :lower_rr_threshold,
            :upper_rr_threshold,
            :short_term_max_horizon,
            :short_term_num_predictions,
            :long_term_max_horizon,
            :long_term_num_predictions
        )
        ON CONFLICT (experiment_key) DO UPDATE
        SET
            gaussian_noise_mean = EXCLUDED.gaussian_noise_mean,
            gaussian_noise_stddev = EXCLUDED.gaussian_noise_stddev,
            base_dataset_file = EXCLUDED.base_dataset_file,
            prefix = EXCLUDED.prefix,
            max_trades_per_5days = EXCLUDED.max_trades_per_5days,
            config_json = EXCLUDED.config_json,
            parameters_json = EXCLUDED.parameters_json,
            profit_threshold = EXCLUDED.profit_threshold,
            tp_multiplier = EXCLUDED.tp_multiplier,
            sl_multiplier = EXCLUDED.sl_multiplier,
            lower_rr_threshold = EXCLUDED.lower_rr_threshold,
            upper_rr_threshold = EXCLUDED.upper_rr_threshold,
            short_term_max_horizon = EXCLUDED.short_term_max_horizon,
            short_term_num_predictions = EXCLUDED.short_term_num_predictions,
            long_term_max_horizon = EXCLUDED.long_term_max_horizon,
            long_term_num_predictions = EXCLUDED.long_term_num_predictions
        RETURNING id;
        """
    )

    # Serialize the configuration dictionary to a JSON string if present.
    config_json_str = json.dumps(config_data) if config_data is not None else None
    # Serialize the parameters dictionary to a JSON string if present.
    params_json_str = json.dumps(params_data) if params_data is not None else None

    # Execute the upsert inside a transaction to ensure atomic behavior.
    with engine.begin() as conn:
        # Execute the SQL statement with the corresponding parameters.
        result = conn.execute(
            sql,
            {
                "experiment_key": experiment_key,
                "gaussian_noise_mean": mu_val,
                "gaussian_noise_stddev": sd_val,
                "base_dataset_file": base_dataset_file,
                "prefix": prefix,
                "max_trades_per_5days": max_trades_per_5days,
                "config_json": config_json_str,
                "parameters_json": params_json_str,
                "profit_threshold": profit_threshold,
                "tp_multiplier": tp_multiplier,
                "sl_multiplier": sl_multiplier,
                "lower_rr_threshold": lower_rr_threshold,
                "upper_rr_threshold": upper_rr_threshold,
                "short_term_max_horizon": short_term_max_horizon,
                "short_term_num_predictions": short_term_num_predictions,
                "long_term_max_horizon": long_term_max_horizon,
                "long_term_num_predictions": long_term_num_predictions,
            },
        )
        # Fetch the single row returned by the INSERT ... RETURNING id clause.
        row = result.fetchone()
    # Convert the returned ID to an integer to ensure consistent typing.
    experiment_id = int(row[0])
    # Return the experiment ID to the caller.
    return experiment_id


def upsert_performance(
    engine: Any,
    experiment_id: int,
    summary_row: Dict[str, Any],
) -> None:
    """
    Insert or update a performance row for the given experiment.

    The uniqueness is enforced by the experiment_id column with ON CONFLICT,
    ensuring exactly one performance record per experiment.

    Args:
        engine: SQLAlchemy engine connected to the target database.
        experiment_id: Primary key of the corresponding experiment row.
        summary_row: Dictionary with metrics from the summary CSV row.
    """
    # Safely parse the validation_stats column into a dictionary.
    validation_stats = safe_parse_stats_column(summary_row.get("validation_stats"))
    # Safely parse the test_stats column into a dictionary.
    test_stats = safe_parse_stats_column(summary_row.get("test_stats"))

    # Define the upsert SQL statement for the performance table.
    sql = text(
        """
        INSERT INTO performance (
            experiment_id,
            initial_capital,
            profit,
            validation_profit,
            test_profit,
            num_trades_total,
            win_pct_total,
            max_dd_total,
            sharpe_total,
            risk_total,
            train_mae,
            train_naive_mae,
            validation_mae,
            validation_naive_mae,
            test_mae,
            test_naive_mae,
            validation_num_trades,
            validation_win_pct,
            validation_max_dd,
            validation_sharpe,
            validation_risk,
            test_num_trades,
            test_win_pct,
            test_max_dd,
            test_sharpe,
            test_risk,
            summary_row_json
        ) VALUES (
            :experiment_id,
            :initial_capital,
            :profit,
            :validation_profit,
            :test_profit,
            :num_trades_total,
            :win_pct_total,
            :max_dd_total,
            :sharpe_total,
            :risk_total,
            :train_mae,
            :train_naive_mae,
            :validation_mae,
            :validation_naive_mae,
            :test_mae,
            :test_naive_mae,
            :validation_num_trades,
            :validation_win_pct,
            :validation_max_dd,
            :validation_sharpe,
            :validation_risk,
            :test_num_trades,
            :test_win_pct,
            :test_max_dd,
            :test_sharpe,
            :test_risk,
            :summary_row_json
        )
        ON CONFLICT (experiment_id) DO UPDATE
        SET
            initial_capital = EXCLUDED.initial_capital,
            profit = EXCLUDED.profit,
            validation_profit = EXCLUDED.validation_profit,
            test_profit = EXCLUDED.test_profit,
            num_trades_total = EXCLUDED.num_trades_total,
            win_pct_total = EXCLUDED.win_pct_total,
            max_dd_total = EXCLUDED.max_dd_total,
            sharpe_total = EXCLUDED.sharpe_total,
            risk_total = EXCLUDED.risk_total,
            train_mae = EXCLUDED.train_mae,
            train_naive_mae = EXCLUDED.train_naive_mae,
            validation_mae = EXCLUDED.validation_mae,
            validation_naive_mae = EXCLUDED.validation_naive_mae,
            test_mae = EXCLUDED.test_mae,
            test_naive_mae = EXCLUDED.test_naive_mae,
            validation_num_trades = EXCLUDED.validation_num_trades,
            validation_win_pct = EXCLUDED.validation_win_pct,
            validation_max_dd = EXCLUDED.validation_max_dd,
            validation_sharpe = EXCLUDED.validation_sharpe,
            validation_risk = EXCLUDED.validation_risk,
            test_num_trades = EXCLUDED.test_num_trades,
            test_win_pct = EXCLUDED.test_win_pct,
            test_max_dd = EXCLUDED.test_max_dd,
            test_sharpe = EXCLUDED.test_sharpe,
            test_risk = EXCLUDED.test_risk,
            summary_row_json = EXCLUDED.summary_row_json;
        """
    )

    # Build a parameter dictionary mapping each named placeholder to a value.
    params = {
        "experiment_id": experiment_id,  # Use the provided experiment primary key.
        "initial_capital": summary_row.get("initial_capital"),  # Initial capital from summary.
        "profit": summary_row.get("profit"),  # Total profit from summary.
        "validation_profit": summary_row.get("validation_profit"),  # Validation segment profit.
        "test_profit": summary_row.get("test_profit"),  # Test segment profit.
        "num_trades_total": summary_row.get("num_trades"),  # Total number of trades.
        "win_pct_total": summary_row.get("win_pct"),  # Overall win percentage.
        "max_dd_total": summary_row.get("max_dd"),  # Overall maximum drawdown.
        "sharpe_total": summary_row.get("sharpe"),  # Overall Sharpe ratio.
        "risk_total": summary_row.get("risk"),  # Overall risk metric from summary.
        "train_mae": summary_row.get("train_mae"),  # Training MAE.
        "train_naive_mae": summary_row.get("train_naive_mae"),  # Training naive MAE.
        "validation_mae": summary_row.get("validation_mae"),  # Validation MAE.
        "validation_naive_mae": summary_row.get("validation_naive_mae"),  # Validation naive MAE.
        "test_mae": summary_row.get("test_mae"),  # Test MAE.
        "test_naive_mae": summary_row.get("test_naive_mae"),  # Test naive MAE.
        "validation_num_trades": validation_stats.get("num_trades"),  # Validation trades count.
        "validation_win_pct": validation_stats.get("win_pct"),  # Validation win percentage.
        "validation_max_dd": validation_stats.get("max_dd"),  # Validation maximum drawdown.
        "validation_sharpe": validation_stats.get("sharpe"),  # Validation Sharpe ratio.
        "validation_risk": validation_stats.get("risk"),  # Validation risk metric.
        "test_num_trades": test_stats.get("num_trades"),  # Test trades count.
        "test_win_pct": test_stats.get("win_pct"),  # Test win percentage.
        "test_max_dd": test_stats.get("max_dd"),  # Test maximum drawdown.
        "test_sharpe": test_stats.get("sharpe"),  # Test Sharpe ratio.
        "test_risk": test_stats.get("risk"),  # Test risk metric.
        "summary_row_json": json.dumps(summary_row),  # Store entire summary row as JSON for extensibility.
    }

    # Execute the performance upsert inside a transaction context.
    with engine.begin() as conn:
        # Execute the SQL statement with the assembled parameter dictionary.
        conn.execute(sql, params)


def process_summary_file(engine: Any, base_dir: str, summary_path: str) -> None:
    """
    Process a single summary CSV file and load its data into the OLAP cube.

    Steps:
    - Parse mu and sd from the filename.
    - Read the summary CSV row.
    - Locate and load the corresponding config and parameters JSON files.
    - Upsert into experiments and performance tables.

    Args:
        engine: SQLAlchemy engine connected to the target database.
        base_dir: Directory containing summary/config/parameters files.
        summary_path: Full path to the summary CSV file to be processed.
    """
    # Extract the basename of the summary file to parse noise-related components.
    filename = os.path.basename(summary_path)
    # Attempt to parse mu and sd components from the filename.
    noise_parts = parse_noise_from_filename(filename)
    # If the pattern was not found, log a warning and skip this file.
    if noise_parts is None:
        print(f"[WARN] Skipping file without recognized noise pattern: {filename}")
        return
    # Unpack mu and sd string components from the parsed tuple.
    mu_str, sd_str = noise_parts
    # Build a canonical experiment key string from the parsed mu and sd values.
    experiment_key = f"gn_mu{mu_str}_sd{sd_str}"

    # Read the summary CSV into a pandas DataFrame.
    df = pd.read_csv(summary_path)
    # If there are no rows in the summary file, log a warning and skip it.
    if df.empty:
        print(f"[WARN] Empty summary file, skipping: {filename}")
        return
    # Assume there is one summary row per experiment, so take the first row.
    row_dict = df.iloc[0].to_dict()

    # Build the common suffix used for the JSON filenames.
    suffix = f"gn_mu{mu_str}_sd{sd_str}"
    # Construct the expected config JSON filename based on the suffix.
    config_filename = f"config_out_{suffix}.json"
    # Construct the expected parameters JSON filename based on the suffix.
    parameters_filename = f"parameters_{suffix}.json"

    # Compute absolute paths to the config and parameters files in the base directory.
    config_path = os.path.join(base_dir, config_filename)
    parameters_path = os.path.join(base_dir, parameters_filename)

    # Attempt to load the configuration JSON file, if present.
    config_data = load_json_if_exists(config_path)
    # If the configuration JSON is missing, log a warning but proceed with partial data.
    if config_data is None:
        print(f"[WARN] Config file not found for {experiment_key}: {config_filename}")

    # Attempt to load the parameters JSON file, if present.
    params_data = load_json_if_exists(parameters_path)
    # If the parameters JSON is missing, log a warning but proceed with partial data.
    if params_data is None:
        print(f"[WARN] Parameters file not found for {experiment_key}: {parameters_filename}")

    # Upsert the experiment row into the experiments table and obtain its primary key.
    experiment_id = upsert_experiment(
        engine=engine,
        experiment_key=experiment_key,
        mu_str=mu_str,
        sd_str=sd_str,
        summary_row=row_dict,
        config_data=config_data,
        params_data=params_data,
    )

    # Upsert the performance metrics row for this experiment.
    upsert_performance(
        engine=engine,
        experiment_id=experiment_id,
        summary_row=row_dict,
    )

    # Log an informational message declaring that this experiment was loaded successfully.
    print(f"[INFO] Loaded experiment {experiment_key} (id={experiment_id}) from {filename}")


def load_all_experiments(base_dir: str) -> None:
    """
    Load all experiments from the given base directory into the OLAP cube.

    The function scans the base directory for summary CSV files whose names start
    with 'summary_gn_mu' and end with '.csv', and processes each one.

    Args:
        base_dir: Directory containing summary_*.csv, config_out_*.json, parameters_*.json.
    """
    # Create a SQLAlchemy engine for interaction with the database.
    engine = get_engine()

    # Iterate over all entries in the specified base directory.
    for entry in os.listdir(base_dir):
        # Process only files that look like summary CSVs with Gaussian noise suffix.
        if entry.startswith("summary_gn_mu") and entry.endswith(".csv"):
            # Build the absolute path to the summary CSV file.
            summary_path = os.path.join(base_dir, entry)
            # Process the identified summary file to load its experiment data.
            process_summary_file(engine, base_dir, summary_path)


if __name__ == "__main__":
    # If this module is executed as a script, parse command-line arguments.

    import argparse  # Import argparse to handle command-line argument parsing.

    # Create an ArgumentParser to specify script usage and options.
    parser = argparse.ArgumentParser(
        description="Load Gaussian-noise strategy experiments into noise_sensitivity_olap."
    )
    # Add a positional argument specifying the directory containing the result files.
    parser.add_argument(
        "base_dir",
        help="Directory containing summary_*.csv, config_out_*.json, parameters_*.json files.",
    )

    # Parse the command-line arguments into a Namespace object.
    args = parser.parse_args()

    # Invoke the top-level loader function using the provided base directory.
    load_all_experiments(args.base_dir)
