#!/bin/bash
# Define the configuration directory
CONFIG_DIR="examples/config/phase_1_daily"

# Process the JSON configuration files
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_1575_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_3150_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_6300_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_12600_1d_config.json"
sh predictor.sh  --load_config "$CONFIG_DIR/phase_1_transformer_25200_1d_config.json"

echo All daily configurations processed.