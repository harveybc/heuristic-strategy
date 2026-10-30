#!/bin/bash
# S1 -- the stage-1 level grid (144 points x the DEAP subset, truth rebuilt per point). Adopts the run already in
# $S1_ROOT (the armed relaunch); while its scope is active it only waits (propagating STOP to its out/STOP); otherwise it
# (re)starts the same tool on $S1_CODE, resuming TASKS.jsonl.
STAGE=S1; source "$CHAIN_ROOT/lib.sh"
D=${S1_ROOT:-$W/S1}; mkdir -p "$D/out"
done_ok(){ [ -f "$D/out/PROGRESS.log" ] && grep -q '"done": true' "$D/out/PROGRESS.log"; }
degenerate(){ [ -f "$D/out/FAILED_DEGENERATE_GEN0" ]; }
finish(){ state done "$D/out/RUN_META.json" "$D/out/TASKS.jsonl,$D/out/GATE_GEN0.json"; exit 0; }
degenerate && { reason "FAILED_DEGENERATE_GEN0: $(python3 -c "import json; print('; '.join(json.load(open('$D/out/FAILED_DEGENERATE_GEN0'))['reasons']))")"; exit 11; }
done_ok && finish
for pfx in ${S1_ADOPT_SCOPES:-}; do
  while scope_active "$pfx"; do stopped && touch "$D/out/STOP"; sleep 120; done
done
degenerate && { reason "FAILED_DEGENERATE_GEN0"; exit 11; }
done_ok && finish
stopped && exit 75
[ -f "$D/SUBSET.json" ] || cp "$SUBSET" "$D/SUBSET.json"
rm -f "$D/out/STOP"
bg_stop "$D/out/STOP" -- crun f4chain-S1 "$S1_MEM" 259200s -- bash -c "cd '$S1_CODE' && $PY tools/strategy_level_grid.py run --subset '$D/SUBSET.json' --admitted '$DATA/admitted' --csv-dir '$DATA/sel15' --jobs $S1_JOBS ${GRID_MAX_POINTS:+--max-points $GRID_MAX_POINTS} --out '$D/out' >> '$D/run.log' 2>&1"
rc=$?
degenerate && { reason "FAILED_DEGENERATE_GEN0"; exit 11; }
done_ok && finish
stopped && exit 75
reason "stage-1 grid ended rc=$rc without its done line (see $D/run.log)"
exit 1
