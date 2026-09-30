# config.py

DEFAULT_VALUES = {
    "hourly_predictions_file": "tests\\data\\phase_2_3_cnn_1h_prediction_d3.csv",
    "daily_predictions_file": "tests\\data\\phase_2_3_cnn_1d_prediction_d3.csv",
    #"daily_predictions_file": "tests\\data\\ann_predictions_daily_d3.csv",
    #"hourly_predictions_file": "tests\\data\\lstm_predictions_hourly_d1.csv",
    #"daily_predictions_file": "tests\\data\\cnn_predictions_daily_d1.csv",
    #"hourly_predictions_file": "tests\\data\\lstm_predictions_hourly_d3.csv",
    #"daily_predictions_file": "tests\\data\\cnn_predictions_daily_d3.csv",
    #"hourly_predictions_file": "tests\\data\\ideal_predictions_hourly_d3.csv",
    #"daily_predictions_file": "tests\\data\\ideal_predictions_daily_d3.csv",
    #"hourly_predictions_file": None,
    #"daily_predictions_file":  None,
    
    "base_dataset_file": "tests\\data\\phase_2_3_base_d3.csv",

    #"base_dataset_file": "tests\\data\\phase_1_base_d1.csv",
    "date_column": "DATE_TIME",
    "plugin": "default",
    "time_horizon": 6,
    "population_size": 20,
    "num_generations": 30,
    "crossover_probability": 0.5,
    "mutation_probability": 0.2,
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
    #"use_normalization_json": "tests\\data\\phase_1_normalizer_debug_out.json",
    "use_normalization_json": None,
    # Prediction source: "CSV" (default) or "API" (Prediction Provider)
    "prediction_source": "CSV",
    "pp_api_url": "http://127.0.0.1:8000",
    "pp_timeout": 5.0,
    # Explicit resolution of the D/E contradiction. Not a recovered historical run.
    # S07 2026-09-30: the retained 241-cell manifest (run_out/native_conditional_20260927_full/
    # manifest.json, sha256 958726415c09ced866c7d8164dfe4e49b99b4ccb8e0e5e7059fa26effdbba94f)
    # records execution.exit_variant = "E" and plugin_params.exit_variant = "E". Verified by reading
    # the manifest, not by rerunning the sweep.
    "exit_variant": "E",
    "exit_variant_resolution": "explicit_resolution_not_recovered_historical_run",
    "historical_run_recovered": False,
    "sweep_241_exit_variant": "E",
    "sweep_241_manifest_sha256": "958726415c09ced866c7d8164dfe4e49b99b4ccb8e0e5e7059fa26effdbba94f",
    # Protective (bracket/intrabar) orders are NOT part of the replication baseline. A run that
    # uses them is the separately named experiment below and must set it explicitly.
    "protective_orders_experiment": "PROTECTIVE_BROKER_ORDERS_EXPERIMENT_NOT_RUN",
    "use_protective_broker_orders": False,
}
