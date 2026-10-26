#!/bin/bash
# FULL parametric layer sweeps (owner 2026-10-26): TP/SL families {cell, param} x volume families {risk, param}, every
# coefficient optimised with NSGA-II on the selection block of the simulated outputs, scored once on the decision block.
# Usage: run_strategy_sim_full.sh <root> <code> [jobs]  (the truth files of run_strategy_sim_sweeps.sh must exist)
set -u
ROOT=$1; CODE=$2; JOBS=${3:-4}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
ADM=${ADM:-$ROOT/data/admitted}; CSV=${CSV:-$ROOT/data/sel15}
export CUDA_VISIBLE_DEVICES=""
log(){ echo "$(date -u +%FT%TZ) [strategy-sim-full] $*" >> $ROOT/LOG; }
mkdir -p $ROOT/runs $ROOT/logs
cd $CODE || exit 1
FOLDS=${FOLDS:-"wf2022 wf2023"}
LEVELS=${LEVELS:-"0.5,0.62,0.7,0.8,1.0"}
SEEDS=${SEEDS:-"0,1,2"}
JOBLIST=$ROOT/JOBS_FULL.txt; : > $JOBLIST
for f in $FOLDS; do
  for fam in cell_param param_param param_risk cell_risk; do echo "$f base $fam $LEVELS" >> $JOBLIST; done
  for v in cost2 regime onesided missing10 cap2 cal12; do echo "$f $v param_param $LEVELS" >> $JOBLIST; done
done
log "start FULL families folds=[$FOLDS] jobs=$JOBS"
run_job(){ f=$1; v=$2; fam=$3; lv=$4
  $CR -q -W ${MAXWAIT:-43200} -m ${MEM:-3G} -t 43200s -n simfull-$f-$v-$fam -- $PY tools/strategy_output_simulation.py run --fold $f --admitted $ADM --csv-dir $CSV --truth-dir $ROOT/truth --variant $v --gate perbar --family $fam --levels $lv --seeds $SEEDS --out $ROOT/runs > $ROOT/logs/full_${f}_${v}_${fam}.log 2>&1 && log "full $f $v $fam DONE" || log "full $f $v $fam FAILED (see logs/full_${f}_${v}_${fam}.log)"
}
export -f run_job log; export ROOT CODE PY CR ADM CSV SEEDS MEM MAXWAIT
cat $JOBLIST | xargs -P $JOBS -L 1 bash -c 'run_job $0 $1 $2 $3'
log "FULL sweeps end"
