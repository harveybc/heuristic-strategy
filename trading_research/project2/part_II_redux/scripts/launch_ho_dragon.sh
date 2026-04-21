#!/bin/bash
# Stage II-6.4: Path B Held-Out Evaluation — DRAGON (remote, GPU)
# Runs: B3 TFT regression yearly HO (6 windows, 2020-2025)
set -e

REDUX_DIR="$HOME/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II_redux"
HS_DIR="$HOME/Documents/GitHub/heuristic-strategy"
FE_DIR="$HOME/Documents/GitHub/feature-eng"
PRED_DIR="$HOME/Documents/GitHub/predictor"
export PYTHONPATH="$HS_DIR:$FE_DIR:$PRED_DIR:$PYTHONPATH"
export STRATEGY_QUIET=1

# CUDA environment for Dragon
NVIDIA_PKGS="$HOME/anaconda3/envs/tensorflow/lib/python3.12/site-packages/nvidia"
if [[ -d "$NVIDIA_PKGS" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_PKGS/cuda_runtime/lib:$NVIDIA_PKGS/cublas/lib:$NVIDIA_PKGS/cudnn/lib:$NVIDIA_PKGS/cufft/lib:$NVIDIA_PKGS/curand/lib:$NVIDIA_PKGS/cusolver/lib:$NVIDIA_PKGS/cusparse/lib:$NVIDIA_PKGS/nccl/lib:$NVIDIA_PKGS/nvjitlink/lib:$NVIDIA_PKGS/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_PKGS/cuda_nvcc"
fi

DATA="$REDUX_DIR/data/processed/btcusd_4h_features.csv"
RAW_DATA="$REDUX_DIR/data/processed/btcusd_4h.csv"
YEARLY_HO="$REDUX_DIR/data/windows/btcusd_4h_yearly_ho.json"
ORCH="$REDUX_DIR/infrastructure/rolling_orchestrator.py"
LOGDIR="$REDUX_DIR/logs"

echo "============================================"
echo "  HO LAUNCH: B3 TFT regression yearly"
echo "  Predictor: tft"
echo "  Manifest: btcusd_4h_yearly_ho.json (6 windows, 2020-2025)"
echo "  Started: $(date)"
echo "============================================"

OUTDIR="$LOGDIR/HO_B3_tft_yearly"
mkdir -p "$OUTDIR"

python "$ORCH" \
    --manifest "$YEARLY_HO" \
    --data "$DATA" \
    --raw_data "$RAW_DATA" \
    --output_dir "$OUTDIR" \
    --experiment_id "HO_B3_tft_yearly" \
    --path B \
    --predictor_plugin "tft" \
    --epochs 200 \
    --early_patience 20 \
    --batch_size 64 \
    2>&1 | tee "$OUTDIR/HO_B3_tft_yearly.log"

echo ""
echo "  DONE: HO_B3 at $(date)"
echo "=== DRAGON HO COMPLETE ==="
