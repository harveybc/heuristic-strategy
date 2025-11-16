#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Schema creation script for the `noise_sensitivity_olap` PostgreSQL database.

This script:
- Connects to the target PostgreSQL database.
- Creates the `experiments` and `performance` tables if they do not exist.
- Creates a helper view `mae_vs_performance` for Metabase visualizations.
"""

import os  # Import os to read environment variables for configuration.
from typing import Any  # Import Any for type hints in function signatures.

from sqlalchemy import (  # Import core SQLAlchemy functions and types.
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Numeric,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB  # Import JSONB type for Postgres.

# Define the environment variable name used for the database URL.
NOISE_OLAP_DB_URL_ENV: str = "NOISE_OLAP_DB_URL"

# Define a default database URL as a fallback if the environment variable is not set.
# IMPORTANT: You MUST replace 'your_pg_user' and 'your_pg_password' with real credentials.
DEFAULT_DB_URL: str = (
    "postgresql+psycopg2://your_pg_user:your_pg_password@localhost:5432/noise_sensitivity_olap"
)


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine connected to the target database.

    The connection string is read from the NOISE_OLAP_DB_URL environment variable,
    falling back to DEFAULT_DB_URL if the environment variable is not defined.
    """
    # Read the database URL from the environment variable if present.
    db_url: str = os.getenv(NOISE_OLAP_DB_URL_ENV, DEFAULT_DB_URL)
    # Create a SQLAlchemy engine using the database URL.
    engine = create_engine(db_url, echo=False, future=True)
    # Return the created engine to the caller.
    return engine


def create_schema() -> None:
    """
    Create the `experiments` and `performance` tables and the `mae_vs_performance` view.

    The function is idempotent: it can be executed multiple times without failing
    because it uses `IF NOT EXISTS` semantics where appropriate.
    """
    # Create a SQLAlchemy engine instance.
    engine = get_engine()
    # Create a MetaData object to hold table definitions.
    metadata = MetaData()

    # Define the `experiments` table representing noise-level experiments.
    experiments = Table(
        "experiments",  # Name of the table in the database.
        metadata,  # MetaData instance registering this table.
        Column("id", Integer, primary_key=True),  # Surrogate primary key.
        Column(
            "experiment_key",
            String,
            nullable=False,
            unique=True,
            doc="String key based on noise parameters, e.g. 'gn_mu0.0020_sd0.0020'.",
        ),
        Column(
            "gaussian_noise_mean",
            Numeric,
            nullable=False,
            doc="Gaussian noise mean used in this experiment.",
        ),
        Column(
            "gaussian_noise_stddev",
            Numeric,
            nullable=False,
            doc="Gaussian noise standard deviation used in this experiment.",
        ),
        Column(
            "base_dataset_file",
            String,
            nullable=True,
            doc="Original dataset file used for this experiment.",
        ),
        Column(
            "prefix",
            String,
            nullable=True,
            doc="Optional prefix used to distinguish strategy modes or datasets.",
        ),
        Column(
            "max_trades_per_5days",
            Integer,
            nullable=True,
            doc="Maximum allowed trades per 5-day period from the configuration.",
        ),
        Column(
            "config_json",
            JSONB,
            nullable=True,
            doc="Raw configuration JSON stored to keep all fields for future analysis.",
        ),
        Column(
            "parameters_json",
            JSONB,
            nullable=True,
            doc="Raw parameters JSON with all optimized strategy parameters.",
        ),
        Column(
            "profit_threshold",
            Numeric,
            nullable=True,
            doc="Optimized profit threshold parameter from the strategy.",
        ),
        Column(
            "tp_multiplier",
            Numeric,
            nullable=True,
            doc="Take-profit multiplier used in the strategy.",
        ),
        Column(
            "sl_multiplier",
            Numeric,
            nullable=True,
            doc="Stop-loss multiplier used in the strategy.",
        ),
        Column(
            "lower_rr_threshold",
            Numeric,
            nullable=True,
            doc="Lower risk-reward threshold used in optimization.",
        ),
        Column(
            "upper_rr_threshold",
            Numeric,
            nullable=True,
            doc="Upper risk-reward threshold used in optimization.",
        ),
        Column(
            "short_term_max_horizon",
            Integer,
            nullable=True,
            doc="Maximum horizon for short-term predictions.",
        ),
        Column(
            "short_term_num_predictions",
            Integer,
            nullable=True,
            doc="Number of short-term predictions made per decision.",
        ),
        Column(
            "long_term_max_horizon",
            Integer,
            nullable=True,
            doc="Maximum horizon for long-term predictions.",
        ),
        Column(
            "long_term_num_predictions",
            Integer,
            nullable=True,
            doc="Number of long-term predictions made per decision.",
        ),
    )

    # Define the `performance` table representing error metrics and trading results.
    performance = Table(
        "performance",  # Name of the table in the database.
        metadata,  # MetaData instance registering this table.
        Column("id", Integer, primary_key=True),  # Surrogate primary key.
        Column(
            "experiment_id",
            Integer,
            ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
            doc="Foreign key pointing to the experiment this performance belongs to.",
        ),
        Column(
            "initial_capital",
            Numeric,
            nullable=True,
            doc="Starting capital used in the backtest.",
        ),
        Column(
            "profit",
            Numeric,
            nullable=True,
            doc="Total profit from the backtest.",
        ),
        Column(
            "validation_profit",
            Numeric,
            nullable=True,
            doc="Profit computed on the validation data segment.",
        ),
        Column(
            "test_profit",
            Numeric,
            nullable=True,
            doc="Profit computed on the test data segment.",
        ),
        Column(
            "num_trades_total",
            Integer,
            nullable=True,
            doc="Total number of trades executed across the full backtest.",
        ),
        Column(
            "win_pct_total",
            Numeric,
            nullable=True,
            doc="Overall win percentage across the full backtest.",
        ),
        Column(
            "max_dd_total",
            Numeric,
            nullable=True,
            doc="Maximum drawdown observed across the full backtest.",
        ),
        Column(
            "sharpe_total",
            Numeric,
            nullable=True,
            doc="Overall Sharpe ratio for the full backtest.",
        ),
        Column(
            "risk_total",
            Numeric,
            nullable=True,
            doc="Global risk value reported in the summary.",
        ),
        Column(
            "train_mae",
            Numeric,
            nullable=True,
            doc="Training MAE for the forecasting model.",
        ),
        Column(
            "train_naive_mae",
            Numeric,
            nullable=True,
            doc="Naive baseline MAE for the training data.",
        ),
        Column(
            "validation_mae",
            Numeric,
            nullable=True,
            doc="Validation MAE for the forecasting model.",
        ),
        Column(
            "validation_naive_mae",
            Numeric,
            nullable=True,
            doc="Naive baseline MAE for the validation data.",
        ),
        Column(
            "test_mae",
            Numeric,
            nullable=True,
            doc="Test MAE for the forecasting model.",
        ),
        Column(
            "test_naive_mae",
            Numeric,
            nullable=True,
            doc="Naive baseline MAE for the test data.",
        ),
        Column(
            "validation_num_trades",
            Integer,
            nullable=True,
            doc="Number of trades executed on the validation segment.",
        ),
        Column(
            "validation_win_pct",
            Numeric,
            nullable=True,
            doc="Win percentage on the validation segment.",
        ),
        Column(
            "validation_max_dd",
            Numeric,
            nullable=True,
            doc="Maximum drawdown on the validation segment.",
        ),
        Column(
            "validation_sharpe",
            Numeric,
            nullable=True,
            doc="Sharpe ratio on the validation segment.",
        ),
        Column(
            "validation_risk",
            Numeric,
            nullable=True,
            doc="Risk value on the validation segment.",
        ),
        Column(
            "test_num_trades",
            Integer,
            nullable=True,
            doc="Number of trades executed on the test segment.",
        ),
        Column(
            "test_win_pct",
            Numeric,
            nullable=True,
            doc="Win percentage on the test segment.",
        ),
        Column(
            "test_max_dd",
            Numeric,
            nullable=True,
            doc="Maximum drawdown on the test segment.",
        ),
        Column(
            "test_sharpe",
            Numeric,
            nullable=True,
            doc="Sharpe ratio on the test segment.",
        ),
        Column(
            "test_risk",
            Numeric,
            nullable=True,
            doc="Risk value on the test segment.",
        ),
        Column(
            "summary_row_json",
            JSONB,
            nullable=True,
            doc="Raw JSON version of the summary CSV row for extensibility.",
        ),
        UniqueConstraint(
            "experiment_id",
            name="uq_performance_experiment_id",
        ),
    )

    # Create all registered tables in the database if they do not yet exist.
    metadata.create_all(engine)

    # Create the helper view for MAE vs performance visualization.
    view_sql = text(
        """
        CREATE OR REPLACE VIEW mae_vs_performance AS
        SELECT
            e.id AS experiment_id,
            e.experiment_key,
            e.gaussian_noise_mean,
            e.gaussian_noise_stddev,
            p.profit,
            p.sharpe_total,
            p.train_mae,
            p.train_naive_mae,
            p.validation_mae,
            p.validation_naive_mae,
            p.test_mae,
            p.test_naive_mae
        FROM experiments e
        JOIN performance p
            ON p.experiment_id = e.id;
        """
    )

    # Execute the view creation inside a transaction context.
    with engine.begin() as conn:
        # Execute the SQL text to create or replace the view.
        conn.execute(view_sql)


if __name__ == "__main__":
    # Execute schema creation when the script is run as a main program.
    create_schema()
