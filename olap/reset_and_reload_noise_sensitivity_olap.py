#!/usr/bin/env python3  # Indicate that this script should be executed with Python 3.
# -*- coding: utf-8 -*-  # Declare UTF-8 as the encoding for this source file.

"""
Convenience script to reset the `noise_sensitivity_olap` schema and reload data.

WARNING:
- This script DROPS the public schema in the target database, deleting all tables.
- Use only if you are sure you want to rebuild the OLAP cube from scratch.
"""

import os  # Import os to access environment variables.
from typing import Any  # Import Any for type hints.

from sqlalchemy import create_engine, text  # Import SQLAlchemy engine and text for SQL execution.

# Import the creation and loading functions from the local modules.
import schema_noise_sensitivity_olap  # Import the schema creation module.
import load_noise_sensitivity_results  # Import the data loading module.

# Environment variable used for the database URL.
NOISE_OLAP_DB_URL_ENV: str = "NOISE_OLAP_DB_URL"

# Default database URL; must point to noise_sensitivity_olap database.
# IMPORTANT: Replace user and password according to your environment or set NOISE_OLAP_DB_URL.
DEFAULT_DB_URL: str = (
    "postgresql+psycopg2://your_pg_user:your_pg_password@localhost:5432/noise_sensitivity_olap"
)


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine pointing at the noise_sensitivity_olap database.

    The connection URL is read from the NOISE_OLAP_DB_URL environment variable,
    falling back to DEFAULT_DB_URL if the variable is not set.
    """
    # Resolve the database URL from environment or use the defined default.
    db_url: str = os.getenv(NOISE_OLAP_DB_URL_ENV, DEFAULT_DB_URL)
    # Create and return an engine instance for the resolved URL.
    return create_engine(db_url, echo=False, future=True)


def reset_schema() -> None:
    """
    Drop and recreate the public schema in the target database.

    This operation removes all tables, views, and other objects in the public schema.
    """
    # Obtain an engine for SQL execution.
    engine = get_engine()
    # Prepare SQL command to drop the public schema including dependent objects.
    drop_sql = text("DROP SCHEMA IF EXISTS public CASCADE;")
    # Prepare SQL command to recreate the public schema.
    create_sql = text("CREATE SCHEMA public;")

    # Print an informational message before dropping the schema.
    print("[INFO] Dropping and recreating 'public' schema in noise_sensitivity_olap...")
    # Execute both statements within a transaction context.
    with engine.begin() as conn:
        # Drop the public schema if it currently exists.
        conn.execute(drop_sql)
        # Recreate the public schema empty.
        conn.execute(create_sql)
    # Print an informational message after schema recreation completes.
    print("[INFO] Schema reset completed (public schema dropped and recreated).")


if __name__ == "__main__":
    # When executed as a script, parse command-line arguments.

    import argparse  # Import argparse for parsing command-line arguments.

    # Create the ArgumentParser with a short description of this script.
    parser = argparse.ArgumentParser(
        description="Reset the noise_sensitivity_olap schema and reload all experiments."
    )

    # Add a positional argument specifying the base directory for result files.
    parser.add_argument(
        "base_dir",
        help="Directory containing summary_*.csv, config_out_*.json, parameters_*.json files.",
    )

    # Parse the arguments from sys.argv into a Namespace object.
    args = parser.parse_args()

    # Reset the public schema, dropping all objects in it.
    reset_schema()

    # Print an informational message before recreating schema objects.
    print("[INFO] Creating OLAP schema (tables and views)...")
    # Recreate the schema objects (tables and views) using the schema module.
    schema_noise_sensitivity_olap.create_schema()
    # Print an informational message after schema creation completes.
    print("[INFO] OLAP schema creation completed.")

    # Print an informational message before reloading all experiment data.
    print(f"[INFO] Reloading experiments from directory: {args.base_dir}")
    # Reload all experiments from the provided directory using the loader module.
    load_noise_sensitivity_results.load_all_experiments(args.base_dir)
    # Print an informational message after all experiments have been processed.
    print("[INFO] Reload process finished.")
