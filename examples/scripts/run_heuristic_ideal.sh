#!/bin/bash
#bash heuristic.sh --base_dataset_file examples/data/phase_3/base_d6.csv --save_parameters examples/results/ideal_high_frequency_parameters.json --prefix _high_frequency --max_trades_per_5days 20
#bash heuristic.sh --base_dataset_file examples/data/phase_3/base_d6.csv --save_parameters examples/results/ideal_low_frequency_parameters.json --prefix _low_frequency --max_trades_per_5days 3
bash heuristic.sh --base_dataset_file examples/data/phase_3/base_d6.csv --load_parameters examples/results/phase_3_2_daily/phase_3_2_cnn_25200_1d_results_high_frequency_parameters.json --prefix _high_frequency --max_trades_per_5days 20
bash heuristic.sh --base_dataset_file examples/data/phase_3/base_d6.csv --load_parameters examples/results/phase_3_2_daily/phase_3_2_cnn_25200_1d_results_low_frequency_parameters.json --prefix _low_frequency --max_trades_per_5days 3

echo "All configurations processed."
