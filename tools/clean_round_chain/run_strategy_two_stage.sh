#!/bin/bash
# Two-stage optimisation (owner 2026-10-26): Stage A on the hierarchical synthetic catalog (mean + CVaR-10 % of the
# owner's composite across leaves -> Pareto set + robust region), then Stage B on the real folds with the gene bounds
# restricted to the Stage-A robust box, scored once on the decision blocks. Families c3 and param_param.
# Usage: run_strategy_two_stage.sh <root> <code> [stage_b_jobs]
set -u
ROOT=$1; CODE=$2; JOBS=${3:-4}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
ADM=${ADM:-$ROOT/data/admitted}; CSV=${CSV:-$ROOT/data/sel15}
export CUDA_VISIBLE_DEVICES="" STAGE_A_WORKERS=${STAGE_A_WORKERS:-8}
log(){ echo "$(date -u +%FT%TZ) [two-stage] $*" >> $ROOT/LOG; }
mkdir -p $ROOT/hier $ROOT/runs $ROOT/logs
cd $CODE || exit 1
[ -f $ROOT/hier/CATALOG.json ] || $PY tools/strategy_hierarchy.py catalog --n-lhs ${N_LHS:-24} --out $ROOT/hier/CATALOG.json > $ROOT/logs/catalog.log 2>&1
log "catalog: $(tail -1 $ROOT/logs/catalog.log)"
for fam in ${FAMILIES:-c3 param_param}; do
  [ -f $ROOT/hier/STAGE_A_$fam.json ] && continue
  $CR -q -W ${MAXWAIT:-43200} -m ${MEM_A:-6G} -t 43200s -n stage-a-$fam${NAME_SUFFIX:-} -- $PY tools/strategy_hierarchy.py stage-a --fold ${FOLD_A:-wf2022} --admitted $ADM --csv-dir $CSV --catalog $ROOT/hier/CATALOG.json \
     --family $fam --population ${POP_A:-32} --generations ${GEN_A:-16} --seed 0 ${LEAVES:+--max-leaves $LEAVES} --out $ROOT/hier > $ROOT/logs/stage_a_$fam.log 2>&1 && log "stage A $fam DONE: $(tail -1 $ROOT/logs/stage_a_$fam.log | cut -c1-200)" || log "stage A $fam FAILED (see logs/stage_a_$fam.log)"
done
if [ -n "${SKIP_B:-}" ]; then log "two-stage end (Stage A only)"; exit 0; fi
JOBLIST=$ROOT/JOBS_B.txt; : > $JOBLIST
for f in ${FOLDS:-wf2022 wf2023}; do for fam in ${FAMILIES:-c3 param_param}; do
  [ -f $ROOT/hier/STAGE_A_$fam.json ] && echo "$f $fam" >> $JOBLIST
done; done
run_b(){ f=$1; fam=$2
  $CR -q -W ${MAXWAIT:-43200} -m ${MEM:-3G} -t 43200s -n stage-b-$f-$fam${NAME_SUFFIX:-} -- $PY tools/strategy_output_simulation.py run --fold $f --admitted $ADM --csv-dir $CSV --truth-dir $ROOT/truth --variant base --gate perbar \
     --family $fam --levels ${LEVELS_B:-0.55,0.62,0.7,0.8,0.9} --seeds ${SEEDS:-0,1,2} --bounds-from $ROOT/hier/STAGE_A_$fam.json --tag stageB_ --out $ROOT/runs > $ROOT/logs/stage_b_${f}_${fam}.log 2>&1 \
     && log "stage B $f $fam DONE" || log "stage B $f $fam FAILED (see logs/stage_b_${f}_${fam}.log)"
}
export -f run_b log; export ROOT CODE PY CR ADM CSV SEEDS MEM LEVELS_B MAXWAIT NAME_SUFFIX
[ -s $JOBLIST ] && cat $JOBLIST | xargs -P $JOBS -L 1 bash -c 'run_b $0 $1'
log "two-stage end"
