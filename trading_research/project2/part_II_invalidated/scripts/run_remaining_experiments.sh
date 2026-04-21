#!/bin/bash
# run_remaining_experiments.sh — Launch A3, A5 after checking A1/A4/A2 status
# Run from part_II directory on Omega

set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

export PYTHONPATH="/home/harveybc/Documents/GitHub/heuristic-strategy:$PYTHONPATH"
export STRATEGY_QUIET=1

CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate tensorflow

echo "=== Running A3: Change-Point Triggered Retraining ==="
python infrastructure/rolling_orchestrator.py \
    --manifest data/windows/changepoint_manifest.json \
    --data data/processed/eurusd_4h_2005_2024.csv \
    --output_dir logs/exp_A3 \
    --experiment_id A3_changepoint_ga \
    --path A \
    --strategy_plugin regime_adaptive \
    --embargo_bars 6 \
    --population_size 30 \
    --num_generations 20 \
    --min_trades 5

echo ""
echo "=== Running A5: regime_wfo + Yearly GA ==="
python infrastructure/rolling_orchestrator.py \
    --manifest data/windows/window_manifest.json \
    --data data/processed/eurusd_4h_2005_2024.csv \
    --output_dir logs/exp_A5 \
    --experiment_id A5_wfo_yearly_ga \
    --path A \
    --strategy_plugin regime_wfo \
    --embargo_bars 6 \
    --population_size 30 \
    --num_generations 20 \
    --min_trades 5

echo ""
echo "=== All remaining experiments complete ==="
