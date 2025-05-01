#!/bin/bash
set -euo pipefail

# phases
PHASES=(phase_1 phase_2 phase_3_1 phase_3_2)

# override per‐phase base datasets
declare -A BASE_DATA_MAP=(
    [phase_1]="examples/data/phase_1/phase_1_base_d3.csv"
    [phase_2]="examples/data/phase_2_4/base_d3.csv"
    [phase_3_1]="examples/data/phase_3/base_d6.csv"
    [phase_3_2]="examples/data/phase_3/base_d6.csv"
)

# override per‐phase hourly configs
declare -A HOURLY_CFG_MAP=(
    [phase_1]="examples/config/phase_1/phase_1_cnn_25200_1h_config.json"
    [phase_2]="examples/config/phase_2/phase_2_2_cnn_1h_config.json"
    [phase_3_1]="examples/config/phase_3_1/phase_3_1_cnn_1h_config.json"
    [phase_3_2]="examples/config/phase_3_2/phase_3_2_cnn_1h_config.json"
)

# common args
PREFIX="_best_hourly_low_freq"
MAX_TRADES=3

# your existing load‐parameters
LOAD_PARAMS=(
    phase_1:examples/results/phase_1_daily/phase_1_ann_25200_1d_results_high_frequency_parameters.json
    phase_2:examples/results/phase_2_daily/phase_2_ann_25200_1d_results_high_frequency_parameters.json
    phase_3_1:examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_high_frequency_parameters.json
    phase_3_2:examples/results/phase_3_2_daily/phase_3_2_cnn_25200_1d_results_high_frequency_parameters.json
)

for PH in "${PHASES[@]}"; do
    CONFIG_DIR="examples/config/${PH}_daily"
    DAILY_DIR="${CONFIG_DIR}"
    BASE_DATA="${BASE_DATA_MAP[$PH]}"
    HOURLY_CFG="${HOURLY_CFG_MAP[$PH]}"

    # pick the correct load‐parameters for this phase
    for entry in "${LOAD_PARAMS[@]}"; do
        key=${entry%%:*}
        val=${entry#*:}
        [[ $key == $PH ]] && LOAD_PARAM="$val"
    done

    for daily_json in "$DAILY_DIR"/*.json; do
        echo "Running heuristic.sh"
        echo "  daily : $daily_json"
        echo "  hourly: $HOURLY_CFG"
        echo "  base  : $BASE_DATA"
        echo "  load  : $LOAD_PARAM"

        sh ./heuristic.sh \
            --predictor_daily_config_file  "$daily_json" \
            --predictor_hourly_config_file "$HOURLY_CFG" \
            --base_dataset_file           "$BASE_DATA" \
            --load_parameters             "$LOAD_PARAM" \
            --prefix                      "$PREFIX" \
            --max_trades_per_5days        "$MAX_TRADES"
    done
done

echo "All configurations processed."
