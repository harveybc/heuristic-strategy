# Heuristic Strategy Optimizer

## Description

The **Heuristic Strategy Optimizer** is a comprehensive tool for optimizing and testing heuristic-based trading strategies. It leverages **Backtrader** for historical data simulation, **DEAP** for genetic algorithm-based optimization, and a flexible **plugin-based system** for extensibility. The optimizer supports external plugins using Python's built-in plugin system, allowing users to integrate custom trading strategies and optimizers seamlessly.

## Installation Instructions

To install and set up the heuristic strategy optimizer, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/heuristic-strategy.git
    cd heuristic-strategy
    ```

2. **Add the cloned directory to the Windows or Linux PYTHONPATH environment variable**:

   In Windows, a restart of the command prompt may be required for the PYTHONPATH variable to be usable.
   Confirm the directory is correctly added by running:

   - On Windows:
     ```bash
     echo %PYTHONPATH%
     ```
   - On Linux:
     ```bash
     echo $PYTHONPATH
     ```

   If the cloned repo directory appears in the PYTHONPATH, continue to the next step.

3. **Create and Activate a Virtual Environment (Anaconda is required)**:

   - **Using `conda`**:
     ```bash
     conda create --name strategy-env python=3.9
     conda activate strategy-env
     ```

4. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5. **Build the Package**:
    ```bash
    python -m build
    ```

6. **Install the Package**:
    ```bash
    pip install .
    ```

7. **(Optional) Run the optimizer**:
    - On Windows, run:
      ```bash
      heuristic_strategy.bat --load_config examples\config\phase_1\phase_1_strategy_config.json
      ```
    - On Linux, run:
      ```bash
      sh heuristic_strategy.sh --load_config examples/config/phase_1/phase_1_strategy_config.json
      ```

8. **(Optional) Run Tests**:
    - On Windows:
      ```bash
      set_env.bat
      pytest
      ```
    - On Linux:
      ```bash
      sh ./set_env.sh
      pytest
      ```

9. **(Optional) Generate Documentation**:
    ```bash
    pdoc --html -o docs app
    ```

10. **(Optional) Install Nvidia CUDA GPU support**:
    Refer to: [Readme - CUDA](https://github.com/harveybc/heuristic-strategy/blob/master/README_CUDA.md)

## Usage

Example configuration JSON files are located in `examples/config`. For a list of parameters available via CLI or in a config JSON file, use:
```bash
heuristic_strategy.bat --help
```

After executing the optimization pipeline, the optimizer generates multiple output files:
- **output_file**: CSV file with results of the optimized strategy
- **results_file**: CSV file containing aggregated results for multiple iterations
- **performance_plot_file**: PNG image displaying strategy performance over time
- **parameter_plot_file**: PNG image visualizing the evolution of strategy parameters

The application supports several command-line arguments to control its behavior, for example:
```bash
heuristic_strategy.bat --load_config examples/config/phase_1/phase_1_strategy_config.json --generations 100 --population 50
```

### Directory Structure

```
heuristic-strategy/
│
├── app/                                 # Main application package
│   ├── __init__.py                      # Package initialization
│   ├── cli.py                           # Command-line interface handling
│   ├── config.py                        # Default configuration values
│   ├── config_handler.py                # Configuration management
│   ├── config_merger.py                 # Configuration merging logic
│   ├── data_handler.py                   # Data loading and saving functions
│   ├── data_processor.py                 # Core data processing pipeline
│   ├── main.py                           # Application entry point
│   ├── plugin_loader.py                  # Dynamic plugin loading system
│   ├── backtester.py                      # Backtesting utilities
│   ├── optimizer.py                      # Optimization logic using DEAP
│   └── plugins/                          # Plugin directory
│       ├── strategy_plugin_basic.py      # Basic trading strategy
│       ├── strategy_plugin_moving_avg.py # Moving Average trading strategy
│       ├── strategy_plugin_rsi.py        # RSI-based trading strategy
│       └── strategy_plugin_custom.py     # Custom trading strategy template
│
├── tests/                               # Test suite directory
│   ├── __init__.py                     # Test package initialization
│   ├── conftest.py                     # pytest configuration
│   ├── acceptance_tests/               # User acceptance tests
│   ├── integration_tests/              # Integration test modules
│   ├── system_tests/                   # System-wide test cases
│   └── unit_tests/                      # Unit test modules
│
├── examples/                            # Example files directory
│   ├── config/                          # Example configuration files
│   ├── data/                            # Example trading data
│   ├── results/                         # Example output results
│   └── scripts/                         # Example execution scripts
│       └── run_phase_1.bat              # Phase 1 execution script
│
├── concatenate_csv.py                   # CSV file manipulation utility
├── setup.py                             # Package installation script
├── heuristic_strategy.bat               # Windows execution script
├── heuristic_strategy.sh                # Linux execution script
├── set_env.bat                          # Windows environment setup
├── set_env.sh                           # Linux environment setup
├── requirements.txt                     # Python dependencies
├── LICENSE.txt                          # Project license
└── README_CUDA.md                       # GPU acceleration instructions
```

This updated repository follows the same structure and functionality as the original predictor repository, while adapting the focus towards heuristic trading strategy optimization.
