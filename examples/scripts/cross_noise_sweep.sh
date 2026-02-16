#!/usr/bin/env bash
# Cross-sensitivity sweep: vary hourly and daily noise independently.
# Runs the heuristic strategy with gaussian noise applied to predictions.
# The trick: run two separate experiments per (h_noise, d_noise) pair
# using heuristic.sh directly, one adding noise to ideal predictions.
#
# Since the current codebase applies the same noise to both,
# we use a Python wrapper that patches the data pipeline.

set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH=./
export STRATEGY_QUIET=1

python3 examples/scripts/cross_noise_sweep.py "$@"
