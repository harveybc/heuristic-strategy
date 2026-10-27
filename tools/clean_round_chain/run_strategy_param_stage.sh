#!/bin/bash
# Strategy parameter stage (owner order 2026-10-27), automatic after the behaviour matrix reaches 100 % effective PASS:
#   1. wait until the matrix report shows every queued cell run and zero effective failures;
#   2. build the declared optimisation subset (stratified over every level, each world with its real-cost mirror, at
#      perfect heads and at quality 0.70; seed 20261027);
#   3. DEAP (fitness = mean daily Sharpe, constraint domination by failing witnesses, population 16, 30 generations,
#      elitism 2), one capped job with up to 10 worker processes, resumable;
#   4. re-verify the best feasible point on the FULL matrix (every cell, seeds 0,1,2, perfect heads); accept it only at
#      100 % effective PASS, otherwise take the next feasible point.
# Usage: run_strategy_param_stage.sh <matrix root>   (honours <root>/STOP_PARAM_STAGE; WARM_ARCHIVE=<earlier ARCHIVE.jsonl> warm-starts generation 0)
set -u
R=$1; CODE=${CODE:-$R/code}                                     # the DEAP's own code root may differ (resume on the code it loaded); the re-verification always uses $R/code
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
export CUDA_VISIBLE_DEVICES="" BEHAVIOUR_MATRIX_ALLOWED=1
log(){ echo "$(date -u +%FT%TZ) [param-stage] $*" >> $R/LOG; }
P0=$CODE/docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json
OUT=$R/deap; mkdir -p $OUT
log "waiting for the matrix to reach 100 % effective PASS (FORCE_START=${FORCE_START:-0})"
while [ "${FORCE_START:-0}" != "1" ]; do
  [ -f $R/STOP_PARAM_STAGE ] && { log "STOP_PARAM_STAGE: exit"; exit 0; }
  ok=$($PY - "$R/report/BEHAVIOUR_MATRIX.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("yes" if d["run"] == d["cells"] and d.get("fail_effective", 1) == 0 else f"no run={d['run']}/{d['cells']} fail_effective={d.get('fail_effective')}")
except Exception as e:
    print("no", e)
PY
)
  [ "$ok" = "yes" ] && break
  sleep 600
done
log "starting DEAP (FORCE_START=${FORCE_START:-0}): building the subset"
cd $CODE
[ -f $OUT/SUBSET.json ] || $PY tools/strategy_param_deap.py subset --cells ${SUBSET_CELLS:-$R/QUEUE.json} --per-level ${PER_LEVEL:-1} --out $OUT/SUBSET.json > $R/logs/deap_subset.log 2>&1
log "subset: $(tail -1 $R/logs/deap_subset.log | cut -c1-200)"
$CR -q -W 43200 -m ${MEM_DEAP:-18G} -t 43200s -n deap-strategy-params -- $PY tools/strategy_param_deap.py run --subset $OUT/SUBSET.json --params $P0 \
   --admitted $R/data/admitted --csv-dir $R/data/sel15 --jobs ${JOBS_DEAP:-10} --out $OUT ${WARM_ARCHIVE:+--warm-archive $WARM_ARCHIVE} > $R/logs/deap_run.log 2>&1 \
   && log "DEAP DONE: $(grep '"done"' $OUT/PROGRESS.log | tail -1)" || { log "DEAP FAILED (see logs/deap_run.log)"; exit 1; }
for k in $(seq 0 ${MAX_RANK:-9}); do
  $PY tools/strategy_param_deap.py best --archive $OUT/ARCHIVE.jsonl --params $P0 --k $k --allow-infeasible --out $OUT/CANDIDATE_k$k.json > $R/logs/deap_best_k$k.log 2>&1 || { log "no feasible point at rank $k"; exit 2; }
  log "re-verifying rank $k on the full matrix: $(tail -1 $R/logs/deap_best_k$k.log | cut -c1-200)"
  VR=$R/verify_k$k                                              # its own matrix root: never shares claims with the running matrix
  mkdir -p $VR/queue $VR/logs $VR/report
  ln -sfn $R/data $VR/data; ln -sfn $R/code $VR/code
  cp $R/queue/*.json $VR/queue/
  ROOT=$VR RES=$VR/results PARAMS=$OUT/CANDIDATE_k$k.json ONESHOT=1 JOBS=${JOBS_VERIFY:-10} bash $R/matrix_driver_v2.sh > $R/logs/verify_k$k.log 2>&1
  v=$($PY -c "import json; d=json.load(open('$VR/report/BEHAVIOUR_MATRIX.json')); print('yes' if d['run']==d['cells'] and d['fail_effective']==0 else 'no %s/%s fail_effective=%s' % (d['run'], d['cells'], d['fail_effective']))")
  if [ "$v" = "yes" ]; then
    cp $OUT/CANDIDATE_k$k.json $OUT/STRATEGY_PARAMS_OPTIMISED.json
    log "ACCEPTED rank $k: full matrix 100 % effective PASS; sha256 $(sha256sum $OUT/STRATEGY_PARAMS_OPTIMISED.json | cut -c1-64)"
    exit 0
  fi
  log "rank $k REJECTED by the full matrix: $v; next feasible point"
done
log "no feasible point up to rank ${MAX_RANK:-9} keeps 100 % effective PASS"
exit 3
