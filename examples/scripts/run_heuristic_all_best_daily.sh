#!/bin/bash

CONFIG_DIR="examples/config/phase_1"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_hourly_config_file "$file" --predictor_daily_config_file "examples/config/phase_1_daily/phase_1_ann_25200_1d_config.json" --base_dataset_file "examples/data/phase_1/phase_1_base_d3.csv" --load_parameters "examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_parameters.json" --prefix "_best_daily"
done

CONFIG_DIR="examples/config/phase_2"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_hourly_config_file "$file" --predictor_daily_config_file "examples/config/phase_2_daily/phase_2_4_ann_1d_config.json" --base_dataset_file "examples/data/phase_2_4/base_d3.csv" --load_parameters "examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_parameters.json" --prefix "_best_daily"
done

CONFIG_DIR="examples/config/phase_3_1"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./heuristic.sh --predictor_hourly_config_file "$file" --predictor_daily_config_file "examples/config/phase_3_1_daily/phase_3_1_cnn_1d_config.json" --base_dataset_file "examples/data/phase_3/base_d6.csv" --load_parameters "examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_parameters.json" --prefix "_best_daily"
done



echo "All configurations processed."
