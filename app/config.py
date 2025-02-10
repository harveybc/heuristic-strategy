# config.py

DEFAULT_VALUES = {
    "hourly_predictions_file": None,
    "daily_predictions_file": None,
    "base_dataset_file": "..\\predictor\\examples\\results\\phase_1\\phase_1_base_d3.csv",
    "date_column": "DATE_TIME",
    "plugin": "default",
    "time_horizon": 6,
    "population_size": 20,
    "num_generations": 100,
    "crossover_probability": 0.5,
    "mutation_probability": 0.2,
    "load_config": "config.json",
    "save_config": "config_out.json",
    "remote_log": None,
    "remote_load_config": None,
    "remote_save_config": None,
    "username": None,
    "password": None,
    "save_log": "debug_log.json",
    "quiet_mode": False,
    "force_date": False,
    "headers": True
}
