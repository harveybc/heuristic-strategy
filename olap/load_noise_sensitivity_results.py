#!/usr/bin/env python3  # Specify Python 3 as the interpreter for this script.
# -*- coding: utf-8 -*-  # Declare UTF-8 encoding for this source file.

"""
Loader script for the `noise_sensitivity_olap` PostgreSQL database.

This script:
- Walks a directory looking for summary CSV files matching the Gaussian noise pattern.
- For each summary, loads the associated configuration and parameters JSON files.
- Inserts or updates records in the `experiments` and `performance` tables.

Expected filenames for each experiment:
- summary_gn_mu0.0020_sd0.0020.csv
- config_out_gn_mu0.0020_sd0.0020.json
- parameters_gn_mu0.0020_sd0.0020.json
"""

import os  # Import os to work with file paths and environment variables.
import re  # Import re for regular expressions to parse mu and sd from filenames.
import json  # Import json to handle JSON encoding and decoding.
import ast  # Import ast to safely evaluate Python literals from strings.
from typing import Any, Dict, Optional, Tuple  # Import typing helpers for clarity and hints.

import pandas as pd  # Import pandas to read CSV files into DataFrames.
from sqlalchemy import create_engine, text  # Import SQLAlchemy engine and text for executing SQL statements.

# Name of the environment variable providing the database URL.
NOISE_OLAP_DB_URL_ENV: str = "NOISE_OLAP_DB_URL"

# Default database URL if NOISE_OLAP_DB_URL is not set.
# IMPORTANT: Replace with your actual credentials or provide the URL via environment.
DEFAULT_DB_URL: str = (
    "postgresql+psycopg2://your_pg_user:your_pg_password@localhost:5432/noise_sensitivity_olap"
)

# Regular expression to capture Gaussian noise parameters from filenames without consuming extensions.
# Pattern explanation:
# - 'gn_mu' literal prefix.
# - mu: one or more digits, optionally followed by a '.' and more digits.
# - separator: '_' or '.' between mu and sd.
# - 'sd' literal prefix, followed by sd: digits with optional decimal part.
NOISE_PATTERN = re.compile(
    r"gn_mu(?P<mu>\d+(?:\.\d+)?)[_.]sd(?P<sd>\d+(?:\.\d+)?)",
    re.IGNORECASE,  # Case-insensitive to be robust against filename variations.
)


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine for the noise_sensitivity_olap database.

    The database URL is obtained from the NOISE_OLAP_DB_URL environment variable,
    falling back to DEFAULT_DB_URL if not set.
    """
    # Resolve the database URL from the environment variable or use the default.
    db_url: str = os.getenv(NOISE_OLAP_DB_URL_ENV, DEFAULT_DB_URL)
    # Create and return a SQLAlchemy engine with the resolved URL.
    return create_engine(db_url, echo=False, future=True)


def parse_noise_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse the mu and sd components from a filename using the NOISE_PATTERN regex.

    Args:
        filename: The basename of the file, e.g. 'summary_gn_mu0.0020_sd0.0020.csv'.

    Returns:
        A tuple (mu_str, sd_str) if the pattern is found, otherwise None.

    Any trailing dots are stripped from the captured components as a safety measure.
    """
    # Search for the Gaussian noise pattern in the filename.
    match = NOISE_PATTERN.search(filename)
    # If there is no match, return None to indicate that parsing failed.
    if not match:
        return None
    # Extract the mu component from the named capturing group.
    mu_str: str = match.group("mu")
    # Extract the sd component from the named capturing group.
    sd_str: str = match.group("sd")
    # Strip any trailing dots from mu_str to avoid issues from malformed filenames.
    mu_str = mu_str.rstrip(".")
    # Strip any trailing dots from sd_str similarly.
    sd_str = sd_str.rstrip(".")
    # Return the cleaned mu and sd strings as a tuple.
    return mu_str, sd_str


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file from the given path if the file exists.

    Args:
        path: Full path to the JSON file.

    Returns:
        The parsed JSON object as a dictionary, or None if the file does not exist.
    """
    # Check whether the specified path refers to an existing file.
    if not os.path.isfile(path):
        # If the file does not exist, return None to signal missing JSON.
        return None
    # Open the JSON file for reading in text mode with UTF-8 encoding.
    with open(path, "r", encoding="utf-8") as f:
        # Parse the JSON content from the file into a Python object.
        data = json.load(f)
    # Return the parsed dictionary.
    return data


def safe_parse_stats_column(value: Any) -> Dict[str, Any]:
    """
    Safely parse a stats column such as 'validation_stats' or 'test_stats'.

    The column is expected to be a string representation of a Python dictionary.
    This function uses ast.literal_eval to safely convert the string into a dict.

    Args:
        value: The column value from the DataFrame row.

    Returns:
        A dictionary containing the parsed stats, or an empty dict on failure or NaN.
    """
    # If the value is None or a NaN float from pandas, return an empty dict.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    # If the value is already a dictionary, return it directly.
    if isinstance(value, dict):
        return value
    try:
        # Attempt to parse the string as a Python literal using literal_eval.
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        # If parsing fails, return an empty dict to avoid interrupting the load.
        return {}
    # Normalize the result to a dict; if not a dict, return an empty one.
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

    Uses the experiment_key to perform an upsert via PostgreSQL's ON CONFLICT.

    Args:
        engine: SQLAlchemy engine connected to the target database.
        experiment_key: Unique experiment key, e.g. 'gn_mu0.0020_sd0.0020'.
        mu_str: Gaussian noise mean as a string parsed from the filename.
        sd_str: Gaussian noise standard deviation as a string parsed from the filename.
        summary_row: Dictionary containing metrics from the summary CSV row.
        config_data: Configuration JSON dictionary, or None if missing.
        params_data: Parameters JSON dictionary, or None if missing.

    Returns:
        The integer primary key of the experiment in the experiments table.
    """
    # Read the Gaussian noise mean from the summary row if present.
    mu_val = summary_row.get("gaussian_noise_mean")
    # If not present, convert mu_str to float as a fallback.
    if mu_val is None:
        mu_val = float(mu_str)
    # Read the Gaussian noise standard deviation from the summary row if present.
    sd_val = summary_row.get("gaussian_noise_stddev")
    # If not present, convert sd_str to float as a fallback.
    if sd_val is None:
        sd_val = float(sd_str)

    # Initialize configuration-derived fields with None.
    base_dataset_file = None
    prefix = None
    max_trades_per_5days = None
    # If configuration data exists, extract relevant fields.
    if config_data is not None:
        base_dataset_file = config_data.get("base_dataset_file")
        prefix = config_data.get("prefix")
        max_trades_per_5days = config_data.get("max_trades_per_5days")

    # Initialize parameter-derived fields with None.
    profit_threshold = None
    tp_multiplier = None
    sl_multiplier = None
    lower_rr_threshold = None
    upper_rr_threshold = None
    short_term_max_horizon = None
    short_term_num_predictions = None
    long_term_max_horizon = None
    long_term_num_predictions = None

    # If parameters data exists, extract known parameter fields.
    if params_data is not None:
        profit_threshold = params_data.get("profit_threshold")
        tp_multiplier = params_data.get("tp_multiplier")
        sl_multiplier = params_data.get("sl_multiplier")
        lower_rr_threshold = params_data.get("lower_rr_threshold")
        upper_rr_threshold = params_data.get("upper_rr_threshold")
        short_term_max_horizon = params_data.get("short_term_max_horizon")
        short_term_num_predictions = params_data.get("short_term_num_predictions")
        long_term_max_horizon = params_data.get("long_term_max_horizon")
        long_term_num_predictions = params_data.get("long_term_num_predictions")

    # Define the upsert SQL statement for the experiments table.
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

    # Serialize configuration and parameters dictionaries as JSON strings.
    config_json_str = json.dumps(config_data) if config_data is not None else None
    params_json_str = json.dumps(params_data) if params_data is not None else None

    # Execute the upsert transaction and fetch the experiment ID.
    with engine.begin() as conn:
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
        row = result.fetchone()
    experiment_id = int(row[0])
    return experiment_id


def upsert_performance(
    engine: Any,
    experiment_id: int,
    summary_row: Dict[str, Any],
) -> None:
    """
    Insert or update a performance row for the given experiment.

    Uniqueness is enforced by the experiment_id column with ON CONFLICT.

    Args:
        engine: SQLAlchemy engine connected to the target database.
        experiment_id: Primary key of the experiment row.
        summary_row: Dictionary containing metrics from the summary CSV row.
    """
    # Parse validation_stats safely into a dictionary.
    validation_stats = safe_parse_stats_column(summary_row.get("validation_stats"))
    # Parse test_stats safely into a dictionary.
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

    # Build the parameters mapping from the summary row and parsed stats.
    params = {
        "experiment_id": experiment_id,
        "initial_capital": summary_row.get("initial_capital"),
        "profit": summary_row.get("profit"),
        "validation_profit": summary_row.get("validation_profit"),
        "test_profit": summary_row.get("test_profit"),
        "num_trades_total": summary_row.get("num_trades"),
        "win_pct_total": summary_row.get("win_pct"),
        "max_dd_total": summary_row.get("max_dd"),
        "sharpe_total": summary_row.get("sharpe"),
        "risk_total": summary_row.get("risk"),
        "train_mae": summary_row.get("train_mae"),
        "train_naive_mae": summary_row.get("train_naive_mae"),
        "validation_mae": summary_row.get("validation_mae"),
        "validation_naive_mae": summary_row.get("validation_naive_mae"),
        "test_mae": summary_row.get("test_mae"),
        "test_naive_mae": summary_row.get("test_naive_mae"),
        "validation_num_trades": validation_stats.get("num_trades"),
        "validation_win_pct": validation_stats.get("win_pct"),
        "validation_max_dd": validation_stats.get("max_dd"),
        "validation_sharpe": validation_stats.get("sharpe"),
        "validation_risk": validation_stats.get("risk"),
        "test_num_trades": test_stats.get("num_trades"),
        "test_win_pct": test_stats.get("win_pct"),
        "test_max_dd": test_stats.get("max_dd"),
        "test_sharpe": test_stats.get("sharpe"),
        "test_risk": test_stats.get("risk"),
        "summary_row_json": json.dumps(summary_row),
    }

    # Execute the SQL upsert inside a transaction context.
    with engine.begin() as conn:
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
        summary_path: Full path to the summary CSV file being processed.
    """
    # Extract only the filename part from the full path for parsing.
    filename = os.path.basename(summary_path)
    # Try to parse the Gaussian noise parameters (mu and sd) from the filename.
    noise_parts = parse_noise_from_filename(filename)
    # If no valid pattern is found, log a warning and skip this file.
    if noise_parts is None:
        print(f"[WARN] Skipping file without recognized noise pattern: {filename}")
        return
    # Unpack mu and sd string components.
    mu_str, sd_str = noise_parts
    # Build a canonical experiment key using the parsed noise components.
    experiment_key = f"gn_mu{mu_str}_sd{sd_str}"

    # Read the summary CSV file into a pandas DataFrame.
    df = pd.read_csv(summary_path)
    # If the summary file is empty, log a warning and skip processing.
    if df.empty:
        print(f"[WARN] Empty summary file, skipping: {filename}")
        return
    # Assume there is exactly one experiment per summary file; take the first row.
    row_dict = df.iloc[0].to_dict()

    # Build the suffix part shared by config and parameters JSON filenames.
    suffix = f"gn_mu{mu_str}_sd{sd_str}"
    # Compose the expected config JSON filename.
    config_filename = f"config_out_{suffix}.json"
    # Compose the expected parameters JSON filename.
    parameters_filename = f"parameters_{suffix}.json"

    # Construct absolute paths to the config and parameters files.
    config_path = os.path.join(base_dir, config_filename)
    parameters_path = os.path.join(base_dir, parameters_filename)

    # Attempt to load the configuration JSON file from disk.
    config_data = load_json_if_exists(config_path)
    if config_data is None:
        # If config is missing, log a warning.
        print(f"[WARN] Config file not found for {experiment_key}: {config_filename}")
    else:
        # If config is loaded successfully, log an informational message.
        print(f"[INFO] Loaded config JSON for {experiment_key}: {config_filename}")

    # Attempt to load the parameters JSON file from disk.
    params_data = load_json_if_exists(parameters_path)
    if params_data is None:
        # If parameters file is missing, log a warning.
        print(f"[WARN] Parameters file not found for {experiment_key}: {parameters_filename}")
    else:
        # If parameters are loaded successfully, log an informational message.
        print(f"[INFO] Loaded parameters JSON for {experiment_key}: {parameters_filename}")

    # Upsert the experiment row into the experiments table and retrieve its ID.
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

    # Log a final informational message summarizing the processed experiment.
    print(f"[INFO] Loaded experiment {experiment_key} (id={experiment_id}) from {filename}")


def load_all_experiments(base_dir: str) -> None:
    """
    Load all experiments from the given base directory into the OLAP cube.

    The function scans the base directory for summary CSV files whose names start
    with 'summary_gn_mu' and end with '.csv', and processes each one.

    Args:
        base_dir: Directory containing summary_*.csv, config_out_*.json, parameters_*.json.
    """
    # Create a SQLAlchemy engine for interacting with the database.
    engine = get_engine()

    # Iterate over all entries in the specified base directory.
    for entry in os.listdir(base_dir):
        # Restrict processing to files matching the expected summary filename pattern.
        if entry.startswith("summary_gn_mu") and entry.endswith(".csv"):
            # Build the full path to the summary CSV file.
            summary_path = os.path.join(base_dir, entry)
            # Process this summary file and load its experiment into the OLAP cube.
            process_summary_file(engine, base_dir, summary_path)


if __name__ == "__main__":
    # When this module is executed as a script, parse command-line arguments.

    import argparse  # Import argparse to parse command line options.

    # Create an ArgumentParser describing the script functionality.
    parser = argparse.ArgumentParser(
        description="Load Gaussian-noise strategy experiments into noise_sensitivity_olap."
    )
    # Add a positional argument specifying the directory containing results and configs.
    parser.add_argument(
        "base_dir",
        help="Directory containing summary_*.csv, config_out_*.json, parameters_*.json files.",
    )

    # Parse arguments provided on the command line.
    args = parser.parse_args()

    # Invoke the bulk loading function using the specified base directory.
    load_all_experiments(args.base_dir)
