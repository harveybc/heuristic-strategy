#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convenience script to reset the `noise_sensitivity_olap` schema and reload data.

WARNING:
- This script DROPS the public schema in the target database, deleting all tables.
- Use only if you are sure you want to rebuild the OLAP cube from scratch.
"""

import os  # Import os to access environment variables.
from typing import Any  # Import Any for type hints.

from sqlalchemy import create_engine, text  # Import SQLAlchemy engine and text.

# Import the creation and loading functions from the local modules.
import schema_noise_sensitivity_olap  # Import the schema creation module.
import load_noise_sensitivity_results  # Import the data loading module.

# Environment variable used for the database URL.
NOISE_OLAP_DB_URL_ENV: str = "NOISE_OLAP_DB_URL"

# Default database URL; must point to noise_sensitivity_olap database.
DEFAULT_DB_URL: str = (
    "postgresql+psycopg2://your_pg_user:your_pg_password@localhost:5432/noise_sensitivity_olap"
)


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine pointing at the noise_sensitivity_olap database.
    """
    # Read the URL from environment or use the default if none is set.
    db_url: str = os.getenv(NOISE_OLAP_DB_URL_ENV, DEFAULT_DB_URL)
    # Create and return the engine.
    return create_engine(db_url, echo=False, future=True)


def reset_schema() -> None:
    """
    Drop and recreate the public schema in the target database.

    This operation removes all tables, views, and other objects in the public schema.
    """
    # Create an engine for SQL execution.
    engine = get_engine()
    # SQL command to drop the public schema with all dependent objects.
    drop_sql = text("DROP SCHEMA IF EXISTS public CASCADE;")
    # SQL command to recreate the public schema.
    create_sql = text("CREATE SCHEMA public;")

    # Execute both commands inside a transaction context.
    with engine.begin() as conn:
        # Drop the public schema if it exists.
        conn.execute(drop_sql)
        # Recreate the public schema.
        conn.execute(create_sql)


if __name__ == "__main__":
    # When executed as a script, parse command-line arguments.

    import argparse  # Import argparse for argument parsing.

    # Create the ArgumentParser with a short description.
    parser = argparse.ArgumentParser(
        description="Reset the noise_sensitivity_olap schema and reload all experiments."
    )

    # Add a positional argument specifying the base directory for result files.
    parser.add_argument(
        "base_dir",
        help="Directory containing summary_*.csv, config_out_*.json, parameters_*.json files.",
    )

    # Parse the arguments from sys.argv.
    args = parser.parse_args()

    # Reset the public schema in the target database.
    reset_schema()
    # Recreate the schema objects (tables and views).
    schema_noise_sensitivity_olap.create_schema()
    # Reload all experiments from the provided directory.
    load_noise_sensitivity_results.load_all_experiments(args.base_dir)
