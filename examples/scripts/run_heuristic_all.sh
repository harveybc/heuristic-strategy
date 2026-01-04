#!/bin/bash

# Force the shell to NOT exit on error, just in case
set +e 

echo "--- Starting Heuristic Low Freq ---"
bash examples/scripts/run_heuristic_all_low_freq.sh; echo "Finished with exit code $?"

echo "--- Starting Heuristic High Freq ---"
bash examples/scripts/run_heuristic_all_high_freq.sh; echo "Finished with exit code $?"

echo "--- Starting Heuristic Phase 2.5 ---"
bash examples/scripts/run_heuristic_phase_2_5.sh; echo "Finished with exit code $?"

echo "--- Starting Heuristic Ideal ---"
bash examples/scripts/run_heuristic_ideal.sh; echo "Finished with exit code $?"

echo "DONE: All configurations processed regardless of individual script results."
