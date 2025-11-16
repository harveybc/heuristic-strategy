
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loader script for the `noise_sensitivity_olap` database.

This script:
- Walks a directory looking for summary CSV files matching the Gaussian noise pattern.
- For each summary, loads the associated configuration and parameters JSON files.
- Inserts or updates records in the `experiments` and `performance` tables.
"""

import os  # Import os to work with file paths and environment variables.
import re  # Import re to use regular expressions for filename parsing.
import json  # Import json to serialize dictionaries for JSONB columns.
import ast  # Import ast to safely parse Python literal strings (for *_stats columns).
from typing import Any, Dict, Optional, Tuple  # Import typing helpers for clarity.

import pandas as pd  # Import pandas to read CSV files easily.
from sqlalchemy import create_engine, text  # Import SQLAlchemy engine and SQL text.

# Environment variable name for database URL used by this loader.
NOISE_OLAP_DB_URL_ENV: str = "NOISE_OLAP_DB_URL"

# Default database URL as a fallback; must be customized by the user.
DEFAULT_DB_URL: str = (
    "postgresql+psycopg2://your_pg_user:your_pg_password@localhost:5432/noise_sensitivity_olap"
)

# Precompiled regex to match the Gaussian noise suffix in filenames.
NOISE_PATTERN = re.compile(
    r"gn_mu(?P<mu>[0-9.]+)_sd(?P<sd>[0-9.]+)",  # Pattern capturing mu and sd numeric strings.
    re.IGNORECASE,  # Ignore case just in case filenames vary.
)


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine for the noise_sensitivity_olap database.

    Reads the connection string from NOISE_OLAP_DB_URL or falls back to DEFAULT_DB_URL.
    """
    # Read the URL from environment, otherwise use the default.
    db_url: str = os.getenv(NOISE_OLAP_DB_URL_ENV, DEFAULT_DB_URL)
    # Create the SQLAlchemy engine with future=True for new-style API behavior.
    engine = create_engine(db_url, echo=False, future=True)
    # Return the engine handle.
    return engine


def parse_noise_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse the mu and sd components from a filename using the NOISE_PATTERN regex.

    Returns:
        A tuple (mu_str, sd_str) if the pattern is found, otherwise None.
    """
    # Search the pattern in the provided filename.
    match = NOISE_PATTERN.search(filename)
    # If there is no match, return None to signal failure.
    if not match:
        return None
    # Extract the matched mu string.
    mu_str: str = match.group("mu")
    # Extract the matched sd string.
    sd_str: str = match.group("sd")
    # Return both strings as a tuple.
    return mu_str, sd_str


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file from the given path if it exists.

    Returns:
        The parsed JSON object as a dictionary, or None if the file is missing.
    """
    # Check if the requested JSON file exists.
    if not os.path.isfile(path):
        # If the file does not exist, return None.
        return None
    # Open the file in text mode with UTF-8 encoding.
    with open(path, "r", encoding="utf-8") as f:
        # Parse the JSON content from the file.
        data = json.load(f)
    # Return the parsed dictionary.
    return data


def safe_parse_stats_column(value: Any) -> Dict[str, Any]:
    """
    Safely parse the 'validation_stats' and 'test_stats' columns.

    The column is expected to be a string representing a Python dictionary. We use
    ast.literal_eval to safely convert the string into a dictionary.

    Returns:
        A dictionary with the parsed stats, or an empty dict on failure or NaN.
    """
    # If the value is missing or NaN in pandas, return an empty dict immediately.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    # If the value is already a dictionary, simply return it.
    if isinstance(value, dict):
        return value
    try:
        # Attempt to parse the string representation safely.
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        # On any parsing error, return an empty dict to avoid crashing the loader.
        return {}
    # Ensure the result is a dictionary, otherwise normalize to empty.
    if not isinstance(parsed, dict):
        return {}
    # Return the parsed dictionary if successful.
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

    This function uses the experiment_key to perform an upsert via
    PostgreSQL's ON CONFLICT clause and returns the resulting primary key.
    """
    # Convert mu and sd strings to decimal-compatible values.
    mu_val = summary_row.get("gaussian_noise_mean")
    # If the CSV does not contain the numeric mean, fall back to direct conversion.
    if mu_val is None:
        mu_val = float(mu_str)
    # Convert sd from the summary or fall back to float(sd_str).
    sd_val = summary_row.get("gaussian_noise_stddev")
    if sd_val is None:
        sd_val = float(sd_str)

    # Extract configuration-derived fields with safe defaults.
    base_dataset_file = None
    prefix = None
    max_trades_per_5days = None
    if config_data is not None:
        # Base dataset file path from config, if present.
        base_dataset_file = config_data.get("base_dataset_file")
        # Optional prefix used in your strategy naming.
        prefix = config_data.get("prefix")
        # Maximum trades per 5 days if defined in the configuration.
        max_trades_per_5days = config_data.get("max_trades_per_5days")

    # Extract parameter-derived fields with safe defaults.
    profit_threshold = None
    tp_multiplier = None
    sl_multiplier = None
    lower_rr_threshold = None
    upper_rr_threshold = None
    short_term_max_horizon = None
    short_term_num_predictions = None
    long_term_max_horizon = None
    long_term_num_predictions = None

    if params_data is not None:
        # Extract individual fields from parameters JSON if present.
        profit_threshold = params_data.get("profit_threshold")
        tp_multiplier = params_data.get("tp_multiplier")
        sl_multiplier = params_data.get("sl_multiplier")
        lower_rr_threshold = params_data.get("lower_rr_threshold")
        upper_rr_threshold = params_data.get("upper_rr_threshold")
        short_term_max_horizon = params_data.get("short_term_max_horizon")
        short_term_num_predictions = params_data.get("short_term_num_predictions")
        long_term_max_horizon = params_data.get("long_term_max_horizon")
        long_term_num_predictions = params_data.get("long_term_num_predictions")

    # Build the SQL statement using PostgreSQL's ON CONFLICT for upsert behavior.
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

    # Serialize config and parameters dicts to JSON so they can be stored in JSONB.
    config_json_str = json.dumps(config_data) if config_data is not None else None
    params_json_str = json.dumps(params_data) if params_data is not None else None

    # Execute the upsert and fetch the resulting experiment ID inside a transaction.
    with engine.begin() as conn:
        # Execute the SQL command with the parameter dictionary.
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
        # Fetch the first row containing the returned ID.
        row = result.fetchone()
    # Extract the experiment ID from the returned row.
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

    Uses ON CONFLICT on the experiment_id field to ensure at most one row
    per experiment is stored in the performance table.
    """
    # Safely parse validation_stats and test_stats fields.
    validation_stats = safe_parse_stats_column(summary_row.get("validation_stats"))
    test_stats = safe_parse_stats_column(summary_row.get("test_stats"))

    # Build the SQL upsert statement for the performance table.
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

    # Compose a parameter dictionary mapping each expected field.
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

    # Execute the SQL statement inside a transaction.
    with engine.begin() as conn:
        # Execute the upsert using the parameter dictionary.
        conn.execute(sql, params)


def process_summary_file(engine: Any, base_dir: str, summary_path: str) -> None:
    """
    Process a single summary CSV file:

    - Parse mu/sd from the filename.
    - Read the CSV row.
    - Load config and parameters JSON files.
    - Upsert experiment and performance rows.
    """
    # Extract the filename from the full path to parse the noise parameters.
    filename = os.path.basename(summary_path)
    # Parse mu and sd from the filename using the precompiled regex.
    noise_parts = parse_noise_from_filename(filename)
    # If the pattern is not found, skip this file.
    if noise_parts is None:
        print(f"[WARN] Skipping file without noise pattern: {filename}")
        return
    # Unpack mu and sd string components from the parsed tuple.
    mu_str, sd_str = noise_parts
    # Build an experiment key string based on mu and sd.
    experiment_key = f"gn_mu{mu_str}_sd{sd_str}"

    # Read the summary CSV into a pandas DataFrame.
    df = pd.read_csv(summary_path)
    # If the summary is empty, there is nothing to process.
    if df.empty:
        print(f"[WARN] Empty summary file, skipping: {filename}")
        return
    # Assuming one row per experiment; take the first row.
    row_dict = df.iloc[0].to_dict()

    # Construct filenames for config and parameters based on the suffix.
    suffix = f"gn_mu{mu_str}_sd{sd_str}"
    config_filename = f"config_out_{suffix}.json"
    parameters_filename = f"parameters_{suffix}.json"

    # Build absolute paths to the config and parameters JSON files.
    config_path = os.path.join(base_dir, config_filename)
    parameters_path = os.path.join(base_dir, parameters_filename)

    # Attempt to load both configuration and parameters JSON files.
    config_data = load_json_if_exists(config_path)
    if config_data is None:
        print(f"[WARN] Config file not found for {experiment_key}: {config_filename}")

    params_data = load_json_if_exists(parameters_path)
    if params_data is None:
        print(f"[WARN] Parameters file not found for {experiment_key}: {parameters_filename}")

    # Upsert the experiment row and retrieve its primary key.
    experiment_id = upsert_experiment(
        engine=engine,
        experiment_key=experiment_key,
        mu_str=mu_str,
        sd_str=sd_str,
        summary_row=row_dict,
        config_data=config_data,
        params_data=params_data,
    )

    # Upsert the corresponding performance row.
    upsert_performance(
        engine=engine,
        experiment_id=experiment_id,
        summary_row=row_dict,
    )

    # Print a simple info message indicating success.
    print(f"[INFO] Loaded experiment {experiment_key} (id={experiment_id}) from {filename}")


def load_all_experiments(base_dir: str) -> None:
    """
    Load all experiments from the given base directory.

    The function looks for summary CSV files following the pattern:
    'summary_gn_mu*_sd*.csv' and processes each one.
    """
    # Create a SQLAlchemy engine for database interactions.
    engine = get_engine()

    # Walk the base directory searching for summary CSV files.
    for entry in os.listdir(base_dir):
        # Process only files whose names start with 'summary_gn_mu' and end with '.csv'.
        if entry.startswith("summary_gn_mu") and entry.endswith(".csv"):
            # Build the full path to the summary CSV file.
            summary_path = os.path.join(base_dir, entry)
            # Process the discovered summary file.
            process_summary_file(engine, base_dir, summary_path)


if __name__ == "__main__":
    # When executed as a script, parse the base directory from the command line.

    import argparse  # Import argparse for command-line argument parsing.

    # Create an ArgumentParser instance with a short description.
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

    # Call the loader using the provided base directory path.
    load_all_experiments(args.base_dir)
