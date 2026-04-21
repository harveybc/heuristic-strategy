#!/bin/bash
# Deploy WFO code + data to remote machines and launch fold subsets
# Usage: bash deploy_wfo.sh

set -e

DRAGON_HOST="harveybc@192.168.1.235"
GAMMA_HOST="harveybc@192.168.0.106"
SSH_PORT=62024
REMOTE_DIR="/home/harveybc/Documents/GitHub/heuristic-strategy"
CONDA_ENV="tensorflow"

# Files to sync
FILES=(
    "app/walk_forward_optimizer.py"
    "app/plugins/plugin_regime_wfo.py"
    "run_wfo.py"
    "tests/data/eurusd_hour_2005_2020.csv"
    "setup.py"
)

deploy_to() {
    local HOST=$1
    local NAME=$2
    echo "=== Deploying to $NAME ($HOST) ==="

    # Ensure dirs exist
    ssh -p $SSH_PORT $HOST "mkdir -p $REMOTE_DIR/app/plugins $REMOTE_DIR/tests/data" 2>&1

    # Sync files
    for f in "${FILES[@]}"; do
        echo "  Syncing $f..."
        scp -P $SSH_PORT "$f" "$HOST:$REMOTE_DIR/$f" 2>&1
    done

    # Install plugin
    echo "  Installing plugin..."
    ssh -p $SSH_PORT $HOST "export PATH=/home/harveybc/anaconda3/envs/$CONDA_ENV/bin:\$PATH && cd $REMOTE_DIR && pip install -e . 2>&1 | tail -1"

    echo "=== $NAME ready ==="
}

launch_on() {
    local HOST=$1
    local NAME=$2
    local FIRST_YEAR=$3
    local LAST_YEAR=$4
    local LOG_FILE="wfo_${NAME}_${FIRST_YEAR}_${LAST_YEAR}.log"

    echo "=== Launching WFO on $NAME: folds $FIRST_YEAR-$LAST_YEAR ==="
    ssh -p $SSH_PORT $HOST "export PATH=/home/harveybc/anaconda3/envs/$CONDA_ENV/bin:\$PATH && cd $REMOTE_DIR && nohup python run_wfo.py --train_years 3 --first_test_year $FIRST_YEAR --last_test_year $LAST_YEAR --population_size 20 --num_generations 15 --min_trades 10 --save_results wfo_results_${FIRST_YEAR}_${LAST_YEAR}.json --save_trades wfo_oos_trades_${FIRST_YEAR}_${LAST_YEAR}.csv > $LOG_FILE 2>&1 & echo PID:\$!"
}

echo "=== WFO Distribution Deployment ==="
echo "Dragon: folds 2014-2016"
echo "Gamma:  folds 2017-2019"
echo ""

# Deploy
cd /home/harveybc/Documents/GitHub/heuristic-strategy
deploy_to "$DRAGON_HOST" "dragon"
deploy_to "$GAMMA_HOST" "gamma"

echo ""

# Launch
launch_on "$DRAGON_HOST" "dragon" 2014 2016
launch_on "$GAMMA_HOST" "gamma" 2017 2019

echo ""
echo "=== All machines launched ==="
echo "Monitor with:"
echo "  ssh -p $SSH_PORT $DRAGON_HOST 'tail -5 $REMOTE_DIR/wfo_dragon_2014_2016.log'"
echo "  ssh -p $SSH_PORT $GAMMA_HOST 'tail -5 $REMOTE_DIR/wfo_gamma_2017_2019.log'"
