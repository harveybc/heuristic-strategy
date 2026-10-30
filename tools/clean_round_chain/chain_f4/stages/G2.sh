#!/bin/bash
# G2 -- the pre-launch gate on the chain's code sha: the mirror property test (frozen point + PROP_N random genomes x the
# twin pairs, full_f4 swapped-reference symmetry form) -> $R/qual/PROPERTY_PASSED_<sha>. Pairs are claimed, so up to
# PROP_PARALLEL instances run (each memory-admitted).
STAGE=G2; source "$CHAIN_ROOT/lib.sh"
PF=$R/qual/PROPERTY_PASSED_$CODE_SHA; O=$W/G2; mkdir -p "$O/claims" "$R/qual"
[ -f "$PF" ] && { state done "" "$PF"; exit 0; }
one(){ for p in $PROP_PAIRS; do
         [ -f "$C/STOP" ] && return 75
         [ -f "$O/PROPERTY_$p.done" ] && continue
         mkdir "$O/claims/$p" 2>/dev/null || continue
         crun "f4chain-G2-$p" "$PROP_MEM" 14400s -- bash -c "cd '$CODE' && $PY tools/strategy_mirror_property.py --subset '$PROP_SUBSET' --pairs $p --n $PROP_N --point '$W/S1R/FROZEN_PARAMS.json' --admitted '$DATA/admitted' --csv-dir '$DATA/sel15' --out '$O/PROPERTY_$p.jsonl' > '$O/log_$p.txt' 2>&1" \
           && touch "$O/PROPERTY_$p.done"
         rmdir "$O/claims/$p" 2>/dev/null
       done; }
rm -rf "$O/claims"; mkdir -p "$O/claims"
for i in $(seq 1 "$PROP_PARALLEL"); do one & sleep 20; done
wait
stopped && exit 75
for p in $PROP_PAIRS; do [ -f "$O/PROPERTY_$p.done" ] || { reason "property test incomplete on pair $p (see $O/log_$p.txt)"; exit 1; }; done
cat $(for p in $PROP_PAIRS; do echo "$O/PROPERTY_$p.jsonl"; done) > "$O/PROPERTY_ALL.jsonl"
python3 - "$O/PROPERTY_ALL.jsonl" "$PF" "$CODE_SHA" "$O/VERDICT.json" <<'PY'
import hashlib, json, sys
src, pf, sha, vf = sys.argv[1:]
rs = [json.loads(l) for l in open(src)]
d = {"sha": sha, "runs": len(rs), "code_symmetric": sum(r["code_symmetric"] for r in rs), "per_run_w14_verdicts": sum(len(r["per_run_w14"]) for r in rs),
     "max_W14_swap_dev_sigma": max((r["W14_swap_dev_sigma"] for r in rs if r["W14_swap_dev_sigma"] is not None), default=None),
     "results_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest()}
d["verdict"] = "PASS" if rs and d["code_symmetric"] == d["runs"] and d["per_run_w14_verdicts"] == 0 else "FAIL"
open(vf, "w").write(json.dumps(d))
if d["verdict"] == "PASS":
    open(pf, "w").write(json.dumps(d) + "\n")
print(json.dumps(d))
PY
[ -f "$PF" ] || { reason "PROPERTY test FAILED: $(cat $O/VERDICT.json)"; exit 11; }
state done "$W/S1R/FROZEN_PARAMS.json" "$PF,$O/PROPERTY_ALL.jsonl"
