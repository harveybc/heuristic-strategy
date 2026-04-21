#!/bin/bash
# Stage II-5: Path B Experiment Launcher
# Usage: ./launch_path_b.sh <experiment> [machine]
# Experiments: B1..B6 or ALL
# Machines: omega, dragon, gamma

set -e

REDUX_DIR="$HOME/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II_redux"
HS_DIR="$HOME/Documents/GitHub/heuristic-strategy"
FE_DIR="$HOME/Documents/GitHub/feature-eng"
PRED_DIR="$HOME/Documents/GitHub/predictor"
export PYTHONPATH="$HS_DIR:$FE_DIR:$PRED_DIR:$PYTHONPATH"
export STRATEGY_QUIET=1

# Ensure CUDA libs are on LD_LIBRARY_PATH (needed for SSH non-interactive sessions)
NVIDIA_PKGS="$HOME/anaconda3/envs/tensorflow/lib/python3.12/site-packages/nvidia"
if [[ -d "$NVIDIA_PKGS" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_PKGS/cuda_runtime/lib:$NVIDIA_PKGS/cublas/lib:$NVIDIA_PKGS/cudnn/lib:$NVIDIA_PKGS/cufft/lib:$NVIDIA_PKGS/curand/lib:$NVIDIA_PKGS/cusolver/lib:$NVIDIA_PKGS/cusparse/lib:$NVIDIA_PKGS/nccl/lib:$NVIDIA_PKGS/nvjitlink/lib:$NVIDIA_PKGS/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_PKGS/cuda_nvcc"
fi

DATA="$REDUX_DIR/data/processed/btcusd_4h_features.csv"
RAW_DATA="$REDUX_DIR/data/processed/btcusd_4h.csv"
YEARLY="$REDUX_DIR/data/windows/btcusd_4h_yearly.json"
ORCH="$REDUX_DIR/infrastructure/rolling_orchestrator.py"
LOGDIR="$REDUX_DIR/logs"

run_experiment() {
    local exp_id=$1
    local plugin=$2
    local extra_args="${3:-}"

    echo "============================================"
    echo "  LAUNCHING: $exp_id"
    echo "  Predictor: $plugin"
    echo "  Manifest: btcusd_4h_yearly.json"
    echo "============================================"

    local outdir="$LOGDIR/$exp_id"
    mkdir -p "$outdir"

    python "$ORCH" \
        --manifest "$YEARLY" \
        --data "$DATA" \
        --raw_data "$RAW_DATA" \
        --output_dir "$outdir" \
        --experiment_id "$exp_id" \
        --path B \
        --predictor_plugin "$plugin" \
        --epochs 200 \
        --early_patience 20 \
        --batch_size 64 \
        $extra_args \
        2>&1 | tee "$outdir/${exp_id}.log"

    echo ""
    echo "  DONE: $exp_id (log: $outdir/${exp_id}.log)"
    echo ""
}

case "${1:-}" in
    B1)
        # CNN
        run_experiment "B1_cnn_yearly" "cnn"
        ;;
    B2)
        # LSTM
        run_experiment "B2_lstm_yearly" "lstm"
        ;;
    B3)
        # TFT (Temporal Fusion Transformer)
        run_experiment "B3_tft_yearly" "tft"
        ;;
    B4)
        # TCN (Temporal Convolutional Network)
        run_experiment "B4_tcn_yearly" "tcn"
        ;;
    B5)
        # ANN (baseline)
        run_experiment "B5_ann_yearly" "ann"
        ;;
    B6)
        # Transformer
        run_experiment "B6_transformer_yearly" "transformer"
        ;;
    OMEGA)
        # Omega batch: B1, B2
        run_experiment "B1_cnn_yearly" "cnn"
        run_experiment "B2_lstm_yearly" "lstm"
        echo "=== OMEGA PATH B BATCH COMPLETE ==="
        ;;
    DRAGON)
        # Dragon batch: B3, B4, B5, B6
        run_experiment "B3_tft_yearly" "tft"
        run_experiment "B4_tcn_yearly" "tcn"
        run_experiment "B5_ann_yearly" "ann"
        run_experiment "B6_transformer_yearly" "transformer"
        echo "=== DRAGON PATH B BATCH COMPLETE ==="
        ;;
    GAMMA)
        # Gamma batch: binary versions B1b-B4b
        run_experiment "B1b_binary_cnn_yearly" "binary_cnn"
        run_experiment "B2b_binary_lstm_yearly" "binary_lstm"
        run_experiment "B3b_binary_tft_yearly" "binary_tft"
        run_experiment "B4b_binary_tcn_yearly" "binary_tcn"
        echo "=== GAMMA PATH B BATCH COMPLETE ==="
        ;;
    BINARY)
        # Binary batch (for Dragon/any GPU machine): B1b-B4b
        run_experiment "B1b_binary_cnn_yearly" "binary_cnn"
        run_experiment "B2b_binary_lstm_yearly" "binary_lstm"
        run_experiment "B3b_binary_tft_yearly" "binary_tft"
        run_experiment "B4b_binary_tcn_yearly" "binary_tcn"
        echo "=== BINARY PATH B BATCH COMPLETE ==="
        ;;
    *)
        echo "Usage: $0 {B1|B2|B3|B4|B5|B6|OMEGA|DRAGON|GAMMA|BINARY}"
        echo ""
        echo "Omega (B1-B2): CNN, LSTM"
        echo "Dragon (B3-B6): TFT, TCN, ANN, Transformer"
        echo "Gamma (B1b-B4b): binary_cnn, binary_lstm, binary_tft, binary_tcn"
        echo ""
        exit 1
        ;;
esac
