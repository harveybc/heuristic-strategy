#!/bin/bash
# Setup remote machine for Stage II-3 experiments
# Usage: ./setup_remote.sh <hostname>
set -e

HOST=${1:?Usage: setup_remote.sh <hostname>}
HS_ROOT="$HOME/Documents/GitHub/heuristic-strategy"
REMOTE_DIR="~/Documents/GitHub/heuristic-strategy"
P2_DATA="$HS_ROOT/trading_research/project2/part_II"

echo "=== Setting up $HOST for Stage II-3 ==="

# 1. Create remote directory structure
echo "[1/4] Creating directories on $HOST..."
ssh $HOST "mkdir -p ~/Documents/GitHub/heuristic-strategy/trading_research/project2/part_II/{infrastructure,data/{windows,processed,raw},logs,scripts}"

# 2. Sync heuristic-strategy code (excluding large/unnecessary dirs)
echo "[2/4] Syncing heuristic-strategy code..."
rsync -az --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.egg-info' \
    --exclude='logs/' \
    --exclude='.tox' \
    --exclude='build/' \
    --exclude='dist/' \
    "$HS_ROOT/" "$HOST:$REMOTE_DIR/"

# 3. Sync project2 data files
echo "[3/4] Syncing project2 data..."
rsync -az --progress \
    "$P2_DATA/data/" "$HOST:$REMOTE_DIR/trading_research/project2/part_II/data/"
rsync -az --progress \
    "$P2_DATA/infrastructure/" "$HOST:$REMOTE_DIR/trading_research/project2/part_II/infrastructure/"
rsync -az --progress \
    "$P2_DATA/scripts/" "$HOST:$REMOTE_DIR/trading_research/project2/part_II/scripts/"

# 4. Install dependencies
echo "[4/4] Installing Python dependencies on $HOST..."
ssh $HOST "python3 -m pip install --user --quiet pandas numpy deap scikit-learn backtrader 2>&1 | tail -5"

echo ""
echo "=== Verifying installation on $HOST ==="
ssh $HOST "python3 -c 'import pandas, numpy, deap, sklearn, backtrader; print(\"All dependencies OK\")'"

echo "=== $HOST ready for experiments ==="
