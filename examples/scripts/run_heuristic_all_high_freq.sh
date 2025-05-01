#!/usr/bin/env bash
set -euo pipefail

PHASES=(phase_1 phase_2 phase_3_1 phase_3_2)

# per‐phase base datasets
declare -A BASE_DATA_MAP=(
    [phase_1]="examples/data/phase_1/phase_1_base_d3.csv"
    [phase_2]="examples/data/phase_2_4/base_d3.csv"
    [phase_3_1]="examples/data/phase_3/base_d6.csv"
    [phase_3_2]="examples/data/phase_3/base_d6.csv"
)

# per‐phase hourly configs
declare -A HOURLY_CFG_MAP=(
    [phase_1]="examples/config/phase_1/phase_1_cnn_25200_1h_config.json"
    [phase_2]="examples/config/phase_2/phase_2_2_cnn_1h_config.json"
    [phase_3_1]="examples/config/phase_3_1/phase_3_1_cnn_1h_config.json"
    [phase_3_2]="examples/config/phase_3_2/phase_3_2_cnn_1h_config.json"
)

# per‐phase load‐parameters
declare -A LOAD_PARAMS=(
    [phase_1]="examples/results/phase_1_daily/phase_1_ann_25200_1d_results_high_frequency_parameters.json"
    [phase_2]="examples/results/phase_2_daily/phase_2_ann_25200_1d_results_high_frequency_parameters.json"
    [phase_3_1]="examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_high_frequency_parameters.json"
    [phase_3_2]="examples/results/phase_3_2_daily/phase_3_2_cnn_25200_1d_results_high_frequency_parameters.json"
)

PREFIX="_best_hourly_high_freq"
MAX_TRADES=20

for PH in "${PHASES[@]}"; do
    DAILY_DIR="examples/config/${PH}_daily"
    for daily_cfg in "$DAILY_DIR"/*.json; do
        echo "Running heuristic.sh with:"
        echo "  daily : $daily_cfg"
        echo "  hourly: ${HOURLY_CFG_MAP[$PH]}"
        echo "  base  : ${BASE_DATA_MAP[$PH]}"
        echo "  load  : ${LOAD_PARAMS[$PH]}"
        sh ./heuristic.sh \
            --predictor_daily_config_file  "$daily_cfg" \
            --predictor_hourly_config_file "${HOURLY_CFG_MAP[$PH]}" \
            --base_dataset_file           "${BASE_DATA_MAP[$PH]}" \
            --load_parameters             "${LOAD_PARAMS[$PH]}" \
            --prefix                      "$PREFIX" \
            --max_trades_per_5days        "$MAX_TRADES"
    done
done

echo "All configurations processed."
