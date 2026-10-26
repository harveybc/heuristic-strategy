#!/bin/bash
# Intervention battery (owner 2026-10-26) on the frozen strategies of the simulation sweeps: base scenario, per-bar
# gate, families c3 and param_param, quality levels 0.62 / 0.70 / 0.80 / 1.00, both folds; decision block only.
# Waits for the sweep drivers' end markers. Usage: run_strategy_sim_interventions.sh <root> <code> [jobs] [draws]
set -u
ROOT=$1; CODE=$2; JOBS=${3:-4}; DRAWS=${4:-6}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
ADM=${ADM:-$ROOT/data/admitted}; CSV=${CSV:-$ROOT/data/sel15}
export CUDA_VISIBLE_DEVICES=""
log(){ echo "$(date -u +%FT%TZ) [strategy-sim-int] $*" >> $ROOT/LOG; }
mkdir -p $ROOT/interventions $ROOT/logs
cd $CODE || exit 1
for i in $(seq 1 288); do grep -q "FULL sweeps end" $ROOT/LOG && grep -q "sweeps end" $ROOT/LOG && break; sleep 300; done
log "start interventions jobs=$JOBS draws=$DRAWS"
JOBLIST=$ROOT/JOBS_INT.txt; : > $JOBLIST
for f in ${FOLDS:-wf2022 wf2023}; do for lv in ${LEVELS:-0.62 0.70 0.80 1.00}; do
  [ -f $ROOT/runs/RUN_${f}_base_perbar_q${lv}.json ] && echo "$ROOT/runs/RUN_${f}_base_perbar_q${lv}.json" >> $JOBLIST
  [ -f $ROOT/runs/RUN_${f}_base_perbar_param_param_q${lv}.json ] && echo "$ROOT/runs/RUN_${f}_base_perbar_param_param_q${lv}.json" >> $JOBLIST
done; done
run_job(){ r=$1; n=$(basename $r .json)
  [ -f $ROOT/interventions/INT_${n#RUN_}.json ] && return
  $CR -q -W ${MAXWAIT:-43200} -m ${MEM:-3G} -t 43200s -n simint-${n#RUN_} -- $PY tools/strategy_intervention.py --run $r --truth-dir $ROOT/truth --admitted $ADM --csv-dir $CSV --draws $DRAWS --out $ROOT/interventions/INT_${n#RUN_}.json > $ROOT/logs/int_${n#RUN_}.log 2>&1 && log "int ${n#RUN_} DONE" || log "int ${n#RUN_} FAILED (see logs/int_${n#RUN_}.log)"
}
export -f run_job log; export ROOT CODE PY CR ADM CSV DRAWS MEM MAXWAIT
cat $JOBLIST | xargs -P $JOBS -L 1 bash -c 'run_job $0'
log "interventions end"
