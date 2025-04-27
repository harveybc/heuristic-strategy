#!/bin/bash

sh examples/scripts/run_heuristic_all_best_daily_low_freq.sh
sh examples/scripts/run_heuristic_all_best_hourly_low_freq.sh
sh examples/scripts/run_heuristic_all_best_daily_high_freq.sh
sh examples/scripts/run_heuristic_all_best_hourly_high_freq.sh

echo "All configurations processed."
