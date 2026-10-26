#!/bin/bash
# DIAGNOSTIC pipeline-control unit: run_unit.sh <name> <data-dir> <contract v13|c2> <seeds> [extra pipeline_controls args...]
# Same E3-M command as the WF_V13 / WF_C2_13 units (tools/expert_guided_clean_round.py e3m_cmd), fold wf2022.
set -u
N=$1; DD=$2; K=$3; SEEDS=$4; shift 4
D=/var/tmp/pipeline-controls-20261026; CODE=/var/tmp/hs-controls-20261026; PY=$HOME/anaconda3/envs/trading-stack/bin/python
ADM=${ADM:-/var/tmp/fs-timing/admitted}
REC=$(ls $DD/recipe/CONSUMER_RECIPE_wf2022_DIAGNOSTIC_*.json)
OUT=$D/out/DIAGNOSTIC_${N}
if [ "$K" = "v13" ]; then AC=$D/ACTION_CONTRACT_v1_3.json; EXTRA=""; else AC=$D/ACTION_CONTRACT_C2_v1_3.json; EXTRA="--outcomes ${OUTCOMES:-$D/outcomes/OUTCOMES_ENTRY_wf2022_DIAGNOSTIC_REALBARS.npz}"; fi
mkdir -p $D/outcomes
cd $CODE || exit 1
echo "$(date -u +%FT%TZ) start $N" >> $D/LOG
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP:-4} MKL_NUM_THREADS=${OMP:-4} $HOME/.local/bin/crispdm-run -q -m 3G -t ${WALL:-7200s} -n ctrl-$N -- $PY tools/pipeline_controls.py run-e3m "$@" -- \
  run --admitted $ADM --csv-dir /var/tmp/pivot-detector-private/sel15 --store $DD/store/full_pool_eurusd_h1.parquet --recipe $REC \
  --out $OUT --variants FULL --units 2 --depth 2 --head 16 --dropout 0.3 --wd 0.01 --max-epochs ${EPOCHS:-1000} --scaling fixed_train \
  --nonstationary diff --split-contract wf2022 --agent-state none --exec-contract sac --min-orders-per-week 2.0 --labels zigzag \
  --closer-rule hold_through --action-contract $AC --strategy-fitness sharpe_lcb --seeds $SEEDS --cpu $EXTRA > $OUT.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) end $N rc=$rc" >> $D/LOG
exit $rc
