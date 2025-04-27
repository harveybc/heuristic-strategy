#!/bin/bash

CONFIG_DIR="examples/config/phase_1_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_1/phase_1_cnn_25200_1h_config.json" --base_dataset_file "examples/data/phase_1/phase_1_base_d3.csv" --load_parameters "examples/results/best_phase_3_1_low_freq_trading.json" --prefix "_best_hourly_low_freq" --max_trades_per_5days 3
done

CONFIG_DIR="examples/config/phase_2_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_2/phase_2_4_cnn_1h_config.json" --base_dataset_file "examples/data/phase_2_4/base_d3.csv" --load_parameters "examples/results/best_phase_3_1_low_freq_trading.json" --prefix "_best_hourly_low_freq" --max_trades_per_5days 3
done

CONFIG_DIR="examples/config/phase_3_1_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_3_1/phase_3_1_cnn_1h_config.json" --base_dataset_file "examples/data/phase_3/base_d6.csv" --load_parameters "examples/results/best_phase_3_1_low_freq_trading.json" --prefix "_best_hourly_low_freq" --max_trades_per_5days 3
done



echo "All configurations processed."
