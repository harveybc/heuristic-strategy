import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="heuristic_strategy: A trading strategy tester with backtrader."
    )
    # Data files for processing and backtesting
    parser.add_argument('--hourly_predictions_file', type=str, help='Path to the CSV file with hourly predictions.')
    parser.add_argument('--daily_predictions_file', type=str, help='Path to the CSV file with daily predictions.')
    parser.add_argument('--base_dataset_file', type=str, help='Path to the CSV file with actual rates (base dataset).')
    parser.add_argument('--date_column', type=str, help='Name of the date/time column in the CSV files.')
    
    # Plugin selection
    parser.add_argument('--plugin', type=str, help='Name of the strategy plugin to use.')
    
    # Optimizer configuration parameters
    parser.add_argument('--max_steps', type=int, help='Max number of ticks (steps)')
    parser.add_argument('--population_size', type=int, help='Population size for the optimizer.')
    parser.add_argument('--num_generations', type=int, help='Number of generations for the optimizer.')
    parser.add_argument('--crossover_probability', type=float, help='Crossover probability for the optimizer.')
    parser.add_argument('--mutation_probability', type=float, help='Mutation probability for the optimizer.')
    
    # Configuration saving and loading
    parser.add_argument('--load_config', type=str, help='Path to load a configuration file.')
    parser.add_argument('--save_config', type=str, help='Path to save the current configuration.')
    parser.add_argument('--remote_log', type=str, help='URL of a remote API endpoint for saving debug variables in JSON format.')
    parser.add_argument('--remote_load_config', type=str, help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('--remote_save_config', type=str, help='URL of a remote API endpoint for saving configuration in JSON format.')
    parser.add_argument('--username', type=str, help='Username for the API endpoint.')
    parser.add_argument('--password', type=str, help='Password for the API endpoint.')
    parser.add_argument('--save_log', type=str, help='Path to save the current debug information.')
    
    # Miscellaneous flags
    parser.add_argument('--quiet_mode', action='store_true', help='Suppress output messages.')
    parser.add_argument('--force_date', action='store_true', help='Include date in the output CSV files.')
    parser.add_argument('--headers', action='store_true', help='Indicate if the CSV files have headers.')
    
    return parser.parse_known_args()
