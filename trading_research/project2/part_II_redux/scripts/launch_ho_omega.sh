#!/bin/bash
# Stage II-6.3: Path A Held-Out Evaluation — OMEGA (local)
# Runs: A1 yearly HO (6 windows) + A6 HPO yearly HO (6 windows)
set -e

REDUX_DIR="$HOME/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II_redux"
HS_DIR="$HOME/Documents/GitHub/heuristic-strategy"
FE_DIR="$HOME/Documents/GitHub/feature-eng"
export PYTHONPATH="$HS_DIR:$FE_DIR:$PYTHONPATH"
export STRATEGY_QUIET=1

DATA="$REDUX_DIR/data/processed/btcusd_4h.csv"
YEARLY_HO="$REDUX_DIR/data/windows/btcusd_4h_yearly_ho.json"
ORCH="$REDUX_DIR/infrastructure/rolling_orchestrator.py"
LOGDIR="$REDUX_DIR/logs"

run_ho() {
    local exp_id=$1
    local plugin=$2
    local extra_args="${3:-}"

    echo "============================================"
    echo "  HO LAUNCH: $exp_id"
    echo "  Plugin: $plugin"
    echo "  Manifest: btcusd_4h_yearly_ho.json (6 windows, 2020-2025)"
    echo "  Started: $(date)"
    echo "============================================"

    local outdir="$LOGDIR/$exp_id"
    mkdir -p "$outdir"

    python "$ORCH" \
        --manifest "$YEARLY_HO" \
        --data "$DATA" \
        --output_dir "$outdir" \
        --experiment_id "$exp_id" \
        --path A \
        --strategy_plugin "$plugin" \
        --population_size 30 \
        --num_generations 20 \
        --min_trades 5 \
        $extra_args \
        2>&1 | tee "$outdir/${exp_id}.log"

    echo ""
    echo "  DONE: $exp_id at $(date)"
    echo ""
}

echo "=== OMEGA HO EVALUATION: A1 + A6 (2020-2025) ==="
echo "Start time: $(date)"

# A1: btc_momentum yearly HO (baseline GA: pop=30, gen=20)
run_ho "HO_A1_btc_momentum_yearly" "btc_momentum"

# A6: btc_momentum yearly HO with larger HPO (pop=80, gen=50)
run_ho "HO_A6_btc_momentum_hpo" "btc_momentum" "--population_size 80 --num_generations 50"

echo "=== OMEGA HO COMPLETE at $(date) ==="
