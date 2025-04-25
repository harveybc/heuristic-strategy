#!/bin/bash

CONFIG_DIR="examples/config/phase_1_daily"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./.sh heuristic.sh --predictor_daily_config_file "$file" --predictor_hourly_config_file "examples/config/phase_1/phase_1_cnn_25200_1h_config.json" --base_dataset_file "examples/data/phase_1/phase_1_base_d3.csv"
done




echo "All configurations processed."
