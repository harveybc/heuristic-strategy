# config.py

DEFAULT_VALUES = {
    "hourly_predictions_file": None,
    "daily_predictions_file": None,
    "uncertainty_hourly_file":None,
    "uncertainty_daily_file": None,
    "base_dataset_file": "examples/data/phase_1/phase_1_base_d1.csv",
    "validation_dataset_file": "examples/data/phase_1/phase_1_base_d2.csv",
    "test_dataset_file": "examples/data/phase_1/phase_1_base_d3.csv",
    
    #"hourly_predictions_file": None,
    #"daily_predictions_file": None,
    #"uncertainty_hourly_file":None,
    #"uncertainty_daily_file": None,
    "predictor_hourly_config_file": None,
    "predictor_daily_config_file": None,
    "prefix": "best_daily_",
    "max_trades_per_5days": 3,
    "date_column": "DATE_TIME",
    "plugin": "default",
    "optimizer_plugin": "ga_optimizer",
    "time_horizon": 6,
    "population_size": 25,
    "num_generations": 250,
    # Early stopping patience (epochs without validation improvement before stopping)
    "patience": 15,
    #Gaussian noise parameters for data augmentation during training
    "gaussian_noise_mean": 0.02,
    "gaussian_noise_stddev": 0.02,
    "crossover_probability": 0.2,
    "mutation_probability": 0.1,
    "use_hourly": False,
    "load_config": None,
    "save_config": "config_out.json",
    "remote_log": None,
    "remote_load_config": None,
    "remote_save_config": None,
    "username": None,
    "password": None,
    "save_log": "debug_log.json",
    "quiet_mode": False,
    "force_date": False,
    "headers": True,
    "disable_multiprocessing": True,
    #output files for balance plot, trades csv and summary in a csv with all possible statistics
    "balance_plot_file": "balance_plot.png",
    "trades_csv_file": "trades.csv",
    "summary_csv_file": "summary.csv",
    "strategy_name": "Heuristic Strategy",
    "max_steps": 6300,
    "save_parameters": "parameters.json",
    "load_parameters": None,
    #"load_parameters": "parameters.json",
    #"use_normalization_json": "tests/data/phase_1/phase_1_normalizer_debug_out.json",
    "use_normalization_json": None,

    # Default uncertainty values used if none are provided:
    "default_uncertainty_short_term": 0.0002,
    #"default_uncertainty_short_term": 0.000001,
    "default_uncertainty_long_term": 0.0047,
    #"default_uncertainty_long_term": 0.000001,
    "daily_columns": ["Prediction_H24", "Prediction_H48", "Prediction_H72", "Prediction_H96", "Prediction_H120", "Prediction_H144"],
    "hourly_columns": ["Prediction_H1", "Prediction_H2", "Prediction_H3", "Prediction_H4", "Prediction_H5", "Prediction_H6"],
    "uncertainty_daily_columns": ["Uncertainty_H24", "Uncertainty_H48", "Uncertainty_H72", "Uncertainty_H96", "Uncertainty_H120", "Uncertainty_H144"],
    "uncertainty_hourly_columns": ["Uncertainty_H1", "Uncertainty_H2", "Uncertainty_H3", "Uncertainty_H4", "Uncertainty_H5", "Uncertainty_H6"],
    "use_first_match": True,
    
    # ---------------------------------------------------------------------
    # NEW CONFIGURABLE PREDICTION SETTINGS (OPTIONAL)
    # ---------------------------------------------------------------------
    # These allow dynamic generation of short-term ("hourly") and long-term
    # ("daily" but actually multi-hour) prediction horizons.
    # If any of these are omitted, legacy behaviour is preserved:
    #   - Short term defaults to using `time_horizon` successive 1h steps.
    #   - Long term defaults to using `time_horizon` successive 24h steps.
    # To activate the new logic, provide BOTH max_horizon and num_predictions
    # for the desired timeframe. Predictions will then be uniformly spaced
    # (inclusive) between t+1 and t+max_horizon. Column names are generated
    # accordingly (e.g. max_horizon=6, num_predictions=3 => H1,H3,H6).
    # These can be made optimizable by the strategy plugin.
    # ---------------------------------------------------------------------
    # Short-term (hourly) dynamic settings
    "short_term_max_horizon": None,        # e.g. 6
    "short_term_num_predictions": None,   # e.g. 3
    # Long-term (daily/multi-hour) dynamic settings
    "long_term_max_horizon": None,        # e.g. 144 (hours)
    "long_term_num_predictions": None,    # e.g. 6 or 12
    # NOTE: Periodicity implied by uniform spacing; an explicit periodicity
    # parameter can be derived as (max_horizon-1)/(num_predictions-1) when
    # num_predictions > 1. If a future explicit parameter is needed, it can
    # be added here and handled inside data_processor.
}
