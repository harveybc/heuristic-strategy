#!/bin/bash
# Stage II-6.3: Path A Held-Out Evaluation — GAMMA (remote)
# Runs: A2 monthly HO (72 windows, 2020-2025)
set -e

REDUX_DIR="$HOME/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II_redux"
HS_DIR="$HOME/Documents/GitHub/heuristic-strategy"
FE_DIR="$HOME/Documents/GitHub/feature-eng"
export PYTHONPATH="$HS_DIR:$FE_DIR:$PYTHONPATH"
export STRATEGY_QUIET=1

DATA="$REDUX_DIR/data/processed/btcusd_4h.csv"
MONTHLY_HO="$REDUX_DIR/data/windows/btcusd_4h_monthly_ho.json"
ORCH="$REDUX_DIR/infrastructure/rolling_orchestrator.py"
LOGDIR="$REDUX_DIR/logs"

echo "============================================"
echo "  HO LAUNCH: A2 btc_momentum monthly"
echo "  Plugin: btc_momentum"
echo "  Manifest: btcusd_4h_monthly_ho.json (72 windows, 2020-2025)"
echo "  Started: $(date)"
echo "============================================"

OUTDIR="$LOGDIR/HO_A2_btc_momentum_monthly"
mkdir -p "$OUTDIR"

python "$ORCH" \
    --manifest "$MONTHLY_HO" \
    --data "$DATA" \
    --output_dir "$OUTDIR" \
    --experiment_id "HO_A2_btc_momentum_monthly" \
    --path A \
    --strategy_plugin "btc_momentum" \
    --population_size 30 \
    --num_generations 20 \
    --min_trades 5 \
    2>&1 | tee "$OUTDIR/HO_A2_btc_momentum_monthly.log"

echo ""
echo "  DONE: HO_A2 at $(date)"
echo "=== GAMMA HO COMPLETE ==="
