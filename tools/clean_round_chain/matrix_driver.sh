#!/bin/bash
# Behaviour-matrix queue driver (owner order 2026-10-27): every 60 s it merges queue/*.json into QUEUE.json, finds the
# base cells whose results (3 seeds, cell + zero-cost twin + mirror) are not all present and that no job claims, and
# launches up to $JOBS crispdm jobs (one base cell with its twins per job, 1500M each, 12 h queue); it rewrites
# BEHAVIOUR_MATRIX.{json,md} after every loop. New queue files are picked up without a restart. Honours $ROOT/STOP
# (its own flag only); base cells listed in $ROOT/EXCLUDE_IDS belong to another host's shard and are skipped. The DONE line of a cell is read from the RESULT files (witnesses passed / total, PASS or FAIL
# per file), never from the job's stdout.
set -u
ROOT=${ROOT:-/var/tmp/behaviour-matrix-20261027}; CODE=$ROOT/code; JOBS=${JOBS:-10}; SEEDS=${SEEDS:-0,1,2}; RES=${RES:-$ROOT/results_v2}
PY=${PY:-~/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-~/.local/bin/crispdm-run}
ADM=$ROOT/data/admitted; CSV=$ROOT/data/sel15
export CUDA_VISIBLE_DEVICES=""
log(){ echo "$(date -u +%FT%TZ) [matrix] $*" >> $ROOT/LOG; }
verdict(){ $PY - "$RES" "$1" <<'PYV'
import json, glob, sys
root, cid = sys.argv[1], sys.argv[2]
out = []
for p in sorted(glob.glob(f"{root}/CELL_{cid}_s*.json") + glob.glob(f"{root}/CELL_S_{cid}_s*.json") + glob.glob(f"{root}/CELL_M_{cid}_s*.json")):
    d = json.load(open(p)); res = d.get("results") or {}
    w = [v for k, v in res.items() if isinstance(v, dict) and "pass" in v]
    bad = ",".join(k for k, v in res.items() if isinstance(v, dict) and v.get("pass") is False)
    out.append(f"{d['id']}/s{d['seed']} {sum(1 for v in w if v['pass'])}/{len(w)} {'PASS' if d.get('PASS') else 'FAIL'}" + (f" [{bad}]" if bad else "") + (f" ERROR {str(d.get('error'))[:60]}" if d.get("error") else ""))
print("; ".join(out) if out else "no result file")
PYV
}
mkdir -p $RES $ROOT/claims $ROOT/logs $ROOT/queue $ROOT/report
cd $CODE
log "driver start jobs=$JOBS seeds=$SEEDS"
while true; do
  [ -f $ROOT/STOP ] && { log "STOP flag: driver exits (running jobs finish their cell)"; break; }
  $PY - "$ROOT" "$SEEDS" "$RES" > $ROOT/PENDING.txt <<'PYEOF'
import json, glob, os, sys, time
root, seeds, res = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
cells, seen = [], set()
for q in sorted(glob.glob(f"{root}/queue/*.json")):
    for c in json.load(open(q)):
        if c["id"] not in seen:
            seen.add(c["id"]); cells.append(c)
json.dump(cells, open(f"{root}/QUEUE.json", "w"))
by_id = {c["id"]: c for c in cells}
excl = set(open(f"{root}/EXCLUDE_IDS").read().split()) if os.path.exists(f"{root}/EXCLUDE_IDS") else set()   # base cells another host's shard runs
pending = []
for c in cells:
    if c["mirror"] or c.get("sym_of") or c["id"] in excl:
        continue
    need = [(c["id"], s) for s in seeds]
    if ("S_" + c["id"]) in by_id:
        need.append(("S_" + c["id"], seeds[0]))
    if ("M_" + c["id"]) in by_id:
        need.append(("M_" + c["id"], seeds[0]))
    done = all(os.path.exists(f"{res}/CELL_{i}_s{s}.json") for i, s in need)
    claim = f"{root}/claims/{c['id']}"
    stale = os.path.exists(claim) and (time.time() - os.path.getmtime(claim)) > 4 * 3600
    if not done and (not os.path.exists(claim) or stale):
        pending.append(c["id"])
print("\n".join(pending))
PYEOF
  running=$(ls $ROOT/claims 2>/dev/null | wc -l)
  J=$(cat $ROOT/JOBS_OVERRIDE 2>/dev/null || echo $JOBS)        # the job count can be rebalanced between runs without a restart
  slots=$((J - running)); [ $slots -lt 0 ] && slots=0
  n=0
  while read -r cid && [ $n -lt $slots ]; do
    [ -z "$cid" ] && continue
    touch $ROOT/claims/$cid
    ( $CR -q -W 43200 -m ${MEM:-1500M} -t 7200s -n bm-$cid -- $PY tools/strategy_behaviour_matrix.py run --cells $ROOT/QUEUE.json --fold wf2022 --admitted $ADM --csv-dir $CSV \
         --ids $cid --seeds $SEEDS --params ${PARAMS:-$CODE/docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json} --out $RES > $ROOT/logs/cell_$cid.log 2>&1 && log "cell $cid DONE: $(verdict $cid)" \
         || log "cell $cid FAILED_JOB (see logs/cell_$cid.log): $(verdict $cid)"; rm -f $ROOT/claims/$cid ) &
    n=$((n+1))
  done < $ROOT/PENDING.txt
  $PY tools/strategy_behaviour_matrix.py report --cells $ROOT/QUEUE.json --results $RES --out ${REPORT:-$ROOT/report} > $ROOT/logs/report${REPORT_TAG:-}.log 2>&1
  pend=$(grep -c "" $ROOT/PENDING.txt)
  if [ "$pend" -eq 0 ] && [ "$running" -eq 0 ] && [ "$n" -eq 0 ]; then
    log "queue drained ($RES): $(tail -1 $ROOT/logs/report${REPORT_TAG:-}.log)"
    [ -n "${ONESHOT:-}" ] && { log "ONESHOT driver exits ($RES)"; break; }
    sleep 300
  else
    sleep 60
  fi
done
