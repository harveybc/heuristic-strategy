#!/bin/bash
# PZ5 (PZ_PREREGISTRO_2026_10_30 section 5; owner order 2026-10-30): the staged DEAP of the PZ genes, one crispdm scope
# (forked world-major workers inside), resumable: re-running this script continues where a STOP left it.
#   ROOT=<dir> DATA=<dir with admitted/ and sel15/> [JOBS=3] [MEM=2800M] bash tools/pz_deap_local.sh
# Order: gate-mirror on the launch sha (PROPERTY_PASSED_<sha> in $ROOT) -> stage 1 (grid; its degeneracy gate) -> stage 2
# (its gen-0 degeneracy gate) -> stage 3 -> stage 2b only when the S3 rule says so. Soft stop: touch $ROOT/STOP.
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${ROOT:?}; DATA=${DATA:?}; JOBS=${JOBS:-3}; MEM=${MEM:-2800M}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
SUB=$CODE/docs/short_horizon/clean_round/evidence/pz_20261030/inputs/DEAP_SUBSET_F4.json
PAR=$CODE/docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json
export CUDA_VISIBLE_DEVICES=""
mkdir -p $ROOT
cd $CODE
SHA=$(git log -1 --format=%h -- app tools)                 # the launch sha = the last commit touching the code (docs-only commits keep it)
[ -n "$(git status --porcelain -- app tools)" ] && { echo "DIRTY_TREE: the launch sha must be a committed tree" | tee -a $ROOT/LAUNCH.log; exit 2; }
run(){ $CR -q -W 7200 -m $MEM -t 345600s -n pz-deap-$1 -- $PY tools/strategy_pz_deap.py "$@" --subset $SUB --admitted $DATA/admitted --csv-dir $DATA/sel15 --params $PAR --jobs $JOBS --out $ROOT; }
step(){ echo "$(date -u +%FT%TZ) $*" >> $ROOT/LAUNCH.log; }
[ -f $ROOT/STOP ] && { step "STOP flag present"; exit 0; }
if [ ! -f $ROOT/PROPERTY_PASSED_$SHA ]; then
  step "gate-mirror on $SHA"; run gate-mirror --code-sha $SHA > $ROOT/gate_mirror.log 2>&1; rc=$?
  [ $rc -ne 0 ] && { step "gate-mirror exit $rc: NOT launching"; exit $rc; }
fi
[ -f $ROOT/STAGE1_RESULT.json ] || { step "stage1 on $SHA"; run stage1 >> $ROOT/stage1.log 2>&1; rc=$?; [ $rc -ne 0 ] && { step "stage1 exit $rc"; exit $rc; }; }
[ -f $ROOT/stage2/RESULT.json ] || { step "stage2 on $SHA"; run stage2 >> $ROOT/stage2.log 2>&1; rc=$?; [ $rc -ne 0 ] && { step "stage2 exit $rc"; exit $rc; }; }
[ -f $ROOT/STAGE3_RESULT.json ] || { step "stage3 on $SHA"; run stage3 >> $ROOT/stage3.log 2>&1; rc=$?; [ $rc -ne 0 ] && { step "stage3 exit $rc"; exit $rc; }; }
if [ -f $ROOT/STAGE1_RESULT_S3.json ] && [ ! -f $ROOT/stage2b/RESULT.json ]; then
  step "stage2b on $SHA"; run stage2 --stage1-result STAGE1_RESULT_S3.json --tag stage2b --init $ROOT/stage2/RESULT.json >> $ROOT/stage2b.log 2>&1 || { step "stage2b failed"; exit 1; }
fi
step "PZ5 DONE"; echo DONE > $ROOT/PZ5_DONE
