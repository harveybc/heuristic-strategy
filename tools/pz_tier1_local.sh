#!/bin/bash
# PZ3 (docs/short_horizon/clean_round/PZ_PREREGISTRO_2026_10_30.md section 3): tier 1 of the behaviour matrix for the
# pragmatic strategy PZ, perfect heads, full_f4, the formal chain's tier-1 queue; two crispdm jobs x 1500M (the 3 GiB
# local budget). Usage: OUT=<dir> DATA=<dir with admitted/ and sel15/> bash tools/pz_tier1_local.sh ; STOP flag: $OUT/STOP
# (checked between chunks; a running chunk finishes its cell; results are resumable: a finished CELL_*.json is skipped).
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:?}; DATA=${DATA:?}; JOBS=${JOBS:-2}; MEM=${MEM:-1500M}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
CELLS=$CODE/docs/short_horizon/clean_round/evidence/pz_20261030/inputs/TIER1_CELLS.json
export CUDA_VISIBLE_DEVICES=""
mkdir -p $OUT/logs
[ -f $OUT/STOP ] && { echo "STOP flag present"; exit 0; }
cd $CODE
git rev-parse HEAD > $OUT/CODE_SHA.txt 2>/dev/null || true
for k in $(seq 0 $((JOBS - 1))); do
  ( $CR -q -W 7200 -m $MEM -t 14400s -n pz-tier1-$k -- $PY tools/strategy_behaviour_matrix.py run --cells $CELLS --fold wf2022 \
      --admitted $DATA/admitted --csv-dir $DATA/sel15 --seeds 0,1,2 --chunk $k/$JOBS --strategy pz \
      --pz-params docs/short_horizon/clean_round/contracts/PZ_PARAMS.json \
      --params docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json --out $OUT/results > $OUT/logs/chunk_$k.log 2>&1
    echo "$(date -u +%FT%TZ) chunk $k exit $?" >> $OUT/PROGRESS ) &
  sleep 3
done
wait
echo ALL > $OUT/ALL_DONE
