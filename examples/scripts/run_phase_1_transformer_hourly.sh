#!/bin/bash
# Define the configuration directory
CONFIG_DIR="examples/config/phase_1"

# Process the JSON configuration files
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_1575_1h_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_3150_1h_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_6300_1h_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_12600_1h_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_25200_1h_config.json"

echo All hourly configurations processed.