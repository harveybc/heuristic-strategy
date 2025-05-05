#!/bin/bash
bash heuristic.sh --predictor_daily_config_file examples/config/phase_2_daily/phase_2_1_ann_1d_config.json --predictor_hourly_config_file examples/config/phase_2/phase_2_1_cnn_1h_config.json --base_dataset_file examples/data/phase_2_1/base_d1.csv --prefix _high_frequency --max_trades_per_5days 20 --max_steps 30000
bash heuristic.sh --predictor_daily_config_file examples/config/phase_3_1_daily/phase_3_1_cnn_1d_config.json --predictor_hourly_config_file examples/config/phase_3_1/phase_3_1_cnn_1h_config.json --base_dataset_file examples/data/phase_3/base_d4.csv --prefix _high_frequency --max_trades_per_5days 20 --max_steps 30000
bash heuristic.sh --predictor_daily_config_file examples/config/phase_2_daily/phase_2_1_ann_1d_config.json --predictor_hourly_config_file examples/config/phase_2/phase_2_1_cnn_1h_config.json --base_dataset_file examples/data/phase_2_1/base_d4.csv --prefix _low_frequency --max_trades_per_5days 3 --max_steps 30000
bash heuristic.sh --predictor_daily_config_file examples/config/phase_3_1_daily/phase_3_1_cnn_1d_config.json --predictor_hourly_config_file examples/config/phase_3_1/phase_3_1_cnn_1h_config.json --base_dataset_file examples/data/phase_3/base_d4.csv --prefix _low_frequency --max_trades_per_5days 3 --max_steps 30000
bash examples/scripts/run_heuristic_all.sh
echo "All configurations processed."
