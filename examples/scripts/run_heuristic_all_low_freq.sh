#!/bin/bash
set -euo pipefail

# list your phases here
PHASES=(phase_1 phase_2 phase_3_1 phase_3_2)

# common args
PREFIX="_low_freq"
MAX_TRADES=3
USE_HOURLY=""
LOAD_PARAMS=(
    # adjust these if they differ by phase
    phase_1:examples/results/phase_2_1_daily/phase_2_1_ann_25200_1d_results_low_frequency_parameters.json
    phase_2:examples/results/phase_2_1_daily/phase_2_1_ann_25200_1d_results_low_frequency_parameters.json
    phase_3_1:examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_low_frequency_parameters.json
    phase_3_2:examples/results/phase_3_1_daily/phase_3_1_cnn_25200_1d_results_low_frequency_parameters.json
)

for PH in "${PHASES[@]}"; do
    CONFIG_DIR="examples/config/${PH}"
    DAILY_DIR="${CONFIG_DIR}_daily"
    BASE_DATA="examples/data/${PH/phase_/phase_}_${PH/_1/}_d$( [ "$PH" = "phase_1" ] && echo 3 || echo 6 ).csv"

    # pick the correct load‐parameters for this phase
    for entry in "${LOAD_PARAMS[@]}"; do
        key=${entry%%:*}; val=${entry#*:}
        [[ $key == $PH ]] && LOAD_PARAM="$val"
    done

    for file in "$CONFIG_DIR"/*.json; do
        base=$(basename "$file")
        # replace _1d_ → _1h_ in the filename
        daily_base="${base/_1d_/_1h_}"
        daily_file="$DAILY_DIR/$daily_base"

        echo "Running heuristic.sh"
        echo "  hourly: $file"
        echo "  daily : $daily_file"

        sh ./heuristic.sh \
            --predictor_hourly_config_file "$file" \
            --predictor_daily_config_file "$daily_file" \
            --base_dataset_file "$BASE_DATA" \
            --load_parameters "$LOAD_PARAM" \
            --prefix "$PREFIX" \
            --max_trades_per_5days "$MAX_TRADES" \
            $USE_HOURLY
    done
done

echo "All configurations processed."
