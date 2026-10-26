#!/bin/bash
# Strategy simulation sweeps (owner 2026-10-26): the real C3 strategy + engine fed by simulated head outputs of declared
# quality. Builds the engine truth per fold and variant, then runs every (fold, variant, quality level) configuration
# with the declared noise seeds, each under its own memory cap; a bounded number of configurations run concurrently
# (CPU only). Resumable: finished RUN_*.json files are skipped. Usage: run_strategy_sim_sweeps.sh <root> <code> [jobs]
set -u
# every launch queues for admission up to MAXWAIT (default 12 h): the host admits by reserved memory, not by cores
ROOT=$1; CODE=$2; JOBS=${3:-3}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
ADM=${ADM:-$ROOT/data/admitted}; CSV=${CSV:-$ROOT/data/sel15}
export CUDA_VISIBLE_DEVICES=""
log(){ echo "$(date -u +%FT%TZ) [strategy-sim] $*" >> $ROOT/LOG; }
mkdir -p $ROOT/truth $ROOT/runs $ROOT/logs
cd $CODE || exit 1
FOLDS=${FOLDS:-"wf2022 wf2023"}
LEVELS=${LEVELS:-"0.5,0.55,0.62,0.7,0.8,0.9,1.0"}
LEVELS_FULL=${LEVELS_FULL:-"0.5,0.52,0.55,0.58,0.62,0.66,0.7,0.75,0.8,0.9,1.0"}
SEEDS=${SEEDS:-"0,1,2"}
log "start folds=[$FOLDS] jobs=$JOBS"
for f in $FOLDS; do
  [ -f $ROOT/truth/TRUTH_${f}_cutoff12.npz ] || $CR -q -W ${MAXWAIT:-43200} -m 4G -t 3600s -n sim-truth-$f -- $PY tools/strategy_output_simulation.py truth --fold $f --admitted $ADM --csv-dir $CSV --variants base,cost2,cost3,spike,cutoff12 --out $ROOT/truth > $ROOT/logs/truth_$f.log 2>&1 && log "truth $f DONE" || log "truth $f FAILED"
done
# job list: (fold, variant, levels)
JOBLIST=$ROOT/JOBS.txt; : > $JOBLIST
# catalog (owner 2026-10-26): base on both gates over the full ladder; every other scenario on the per-bar-cost gate over
# the 5-level ladder; the head-shape scenarios (inverted / constant / saturated) at one level
for f in $FOLDS; do
  echo "$f base contract $LEVELS_FULL" >> $JOBLIST
  echo "$f base perbar $LEVELS_FULL" >> $JOBLIST
  for v in block regime onesided asym closer_only opener_only nop_only cal08 cal12 lag1 cost2 cost3 spike missing10 missing30 cap2 cap4 cutoff12 risk05 risk2; do echo "$f $v perbar $LEVELS" >> $JOBLIST; done
  for v in inv_open const_net sat; do echo "$f $v perbar ${LEVEL_SHAPE:-0.8}" >> $JOBLIST; done
done
run_job(){ f=$1; v=$2; g=$3; lv=$4
  [ -n "${SKIP:-}" ] && grep -qx "$f $v $g" "$SKIP" && return        # jobs already in flight under an earlier driver
  $CR -q -W ${MAXWAIT:-43200} -m ${MEM:-4G} -t 43200s -n sim-$f-$v-$g -- $PY tools/strategy_output_simulation.py run --fold $f --admitted $ADM --csv-dir $CSV --truth-dir $ROOT/truth --variant $v --gate $g --levels $lv --seeds $SEEDS --out $ROOT/runs > $ROOT/logs/run_${f}_${v}_${g}.log 2>&1 && log "run $f $v $g DONE" || log "run $f $v $g FAILED (see logs/run_${f}_${v}_${g}.log)"
}
export -f run_job log; export ROOT CODE PY CR ADM CSV SEEDS MEM SKIP MAXWAIT
cat $JOBLIST | xargs -P $JOBS -L 1 bash -c 'run_job $0 $1 $2 $3'
log "sweeps end"
