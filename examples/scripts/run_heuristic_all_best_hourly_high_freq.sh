#!/bin/bash

CONFIG_DIR="examples/config/phase_1_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_1/phase_1_cnn_25200_1h_config.json" --base_dataset_file "examples/data/phase_1/phase_1_base_d3.csv" --load_parameters "examples/results/phase_2_1_daily/phase_2_1_ann_25200_1d_results_high_frequency_parameters.json" --prefix "_best_hourly_high_freq" --max_trades_per_5days 20
done

CONFIG_DIR="examples/config/phase_2_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_2/phase_2_4_cnn_1h_config.json" --base_dataset_file "examples/data/phase_2_4/base_d3.csv" --load_parameters "examples/results/phase_2_1_daily/phase_2_1_ann_25200_1d_results_high_frequency_parameters.json" --prefix "_best_hourly_high_freq" --max_trades_per_5days 20
done

CONFIG_DIR="examples/config/phase_3_1_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_3_1/phase_3_1_cnn_1h_config.json" --base_dataset_file "examples/data/phase_3/base_d6.csv" --load_parameters "examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_high_frequency_parameters.json" --prefix "_best_hourly_high_freq" --max_trades_per_5days 20
done



echo "All configurations processed."
