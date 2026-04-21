#!/bin/bash
# Stage II-3: Path A Experiment Launcher
# Usage: ./launch_path_a.sh <experiment> [machine]
# Experiments: A1..A7 or ALL
# Machines: omega, dragon, gamma

set -e

REDUX_DIR="$HOME/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II_redux"
HS_DIR="$HOME/Documents/GitHub/heuristic-strategy"
FE_DIR="$HOME/Documents/GitHub/feature-eng"
export PYTHONPATH="$HS_DIR:$FE_DIR:$PYTHONPATH"
export STRATEGY_QUIET=1

DATA="$REDUX_DIR/data/processed/btcusd_4h.csv"
YEARLY="$REDUX_DIR/data/windows/btcusd_4h_yearly.json"
MONTHLY="$REDUX_DIR/data/windows/btcusd_4h_monthly.json"
WEEKLY="$REDUX_DIR/data/windows/btcusd_4h_weekly.json"
ORCH="$REDUX_DIR/infrastructure/rolling_orchestrator.py"
LOGDIR="$REDUX_DIR/logs"

run_experiment() {
    local exp_id=$1
    local manifest=$2
    local plugin=$3
    local extra_args="${4:-}"

    echo "============================================"
    echo "  LAUNCHING: $exp_id"
    echo "  Plugin: $plugin"
    echo "  Manifest: $(basename $manifest)"
    echo "============================================"

    local outdir="$LOGDIR/$exp_id"
    mkdir -p "$outdir"

    python "$ORCH" \
        --manifest "$manifest" \
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
    echo "  DONE: $exp_id (log: $outdir/${exp_id}.log)"
    echo ""
}

case "${1:-}" in
    A1)
        # btc_momentum + DEAP GA + Yearly
        run_experiment "A1_btc_momentum_yearly" "$YEARLY" "btc_momentum"
        ;;
    A2)
        # btc_momentum + DEAP GA + Monthly
        run_experiment "A2_btc_momentum_monthly" "$MONTHLY" "btc_momentum"
        ;;
    A3)
        # btc_momentum + DEAP GA + Monthly (proxy for change-point; same monthly with stricter min_trades)
        run_experiment "A3_btc_momentum_changepoint" "$MONTHLY" "btc_momentum" "--min_trades 8 --population_size 50 --num_generations 30"
        ;;
    A4)
        # regime_adaptive + DEAP GA + Yearly + GMM refit
        run_experiment "A4_regime_adaptive_gmm_yearly" "$YEARLY" "regime_adaptive" "--gmm_refit"
        ;;
    A5)
        # regime_wfo + DEAP GA + Yearly
        run_experiment "A5_regime_wfo_yearly" "$YEARLY" "regime_wfo"
        ;;
    A6)
        # btc_momentum + DEAP GA + Yearly (larger pop for HPO)
        run_experiment "A6_btc_momentum_hpo" "$YEARLY" "btc_momentum" "--population_size 80 --num_generations 50"
        ;;
    A7)
        # regime_adaptive + DEAP GA + Weekly + GMM refit
        run_experiment "A7_regime_adaptive_gmm_weekly" "$WEEKLY" "regime_adaptive" "--gmm_refit"
        ;;
    OMEGA)
        # Run Omega batch: A1, A2, A3
        run_experiment "A1_btc_momentum_yearly" "$YEARLY" "btc_momentum"
        run_experiment "A2_btc_momentum_monthly" "$MONTHLY" "btc_momentum"
        run_experiment "A3_btc_momentum_changepoint" "$MONTHLY" "btc_momentum" "--min_trades 8 --population_size 50 --num_generations 30"
        echo "=== OMEGA BATCH COMPLETE ==="
        ;;
    GAMMA)
        # Run Gamma batch: A4, A5, A6, A7
        run_experiment "A4_regime_adaptive_gmm_yearly" "$YEARLY" "regime_adaptive" "--gmm_refit"
        run_experiment "A5_regime_wfo_yearly" "$YEARLY" "regime_wfo"
        run_experiment "A6_btc_momentum_hpo" "$YEARLY" "btc_momentum" "--population_size 80 --num_generations 50"
        run_experiment "A7_regime_adaptive_gmm_weekly" "$WEEKLY" "regime_adaptive" "--gmm_refit"
        echo "=== GAMMA BATCH COMPLETE ==="
        ;;
    *)
        echo "Usage: $0 {A1|A2|A3|A4|A5|A6|A7|OMEGA|GAMMA}"
        echo ""
        echo "Omega (A1-A3): btc_momentum yearly/monthly/changepoint"
        echo "Gamma (A4-A7): regime_adaptive + regime_wfo + btc_momentum HPO"
        echo ""
        exit 1
        ;;
esac
