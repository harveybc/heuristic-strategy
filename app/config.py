# config.py

DEFAULT_VALUES = {
    #"hourly_predictions_file": "tests\\data\\ann_predictions_hourly_d3.csv",
    #"daily_predictions_file": "tests\\data\\ann_predictions_daily_d3.csv",
    "hourly_predictions_file": "tests\\data\\lstm_predictions_hourly_d1.csv",
    "daily_predictions_file": "tests\\data\\cnn_predictions_daily_d1.csv",
    #"hourly_predictions_file": "tests\\data\\lstm_predictions_hourly_d3.csv",
    #"daily_predictions_file": "tests\\data\\cnn_predictions_daily_d3.csv",
    
    #"hourly_predictions_file": "tests\\data\\ideal_predictions_hourly_d3.csv",
    #"daily_predictions_file": "tests\\data\\ideal_predictions_daily_d3.csv",
    #"hourly_predictions_file": None,
    #"daily_predictions_file":  None,
    
    #"base_dataset_file": "tests\\data\\phase_1_base_d3.csv",
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
    "use_normalization_json": None
     
}
