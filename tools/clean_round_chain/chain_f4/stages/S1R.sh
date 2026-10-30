#!/bin/bash
# S1R -- STAGE1_RESULT.json (infeasible points listed apart), the EDGE RULE (one extension per edge axis, one step
# beyond, within the sealed full_f4 bounds, at the canonical genes; a value at a sealed bound = BOUNDARY_HIT), the Deb
# re-ranking with the extension points, and the frozen level point (FROZEN_PARAMS.json).
STAGE=S1R; source "$CHAIN_ROOT/lib.sh"
D=${S1_ROOT:-$W/S1}; O=$W/S1R; mkdir -p "$O"
cd "$CODE" || exit 1
$PY tools/strategy_level_grid.py report --out "$D/out" || { reason "stage-1 report failed"; exit 1; }
cp "$D/out/STAGE1_RESULT.json" "$O/STAGE1_RESULT.json"
$PY tools/strategy_chain_f4.py edge --stage1 "$O/STAGE1_RESULT.json" --out "$O/EDGE.json" --points-out "$O/EXT_POINTS.json" || { reason "edge rule failed (no ranked point?)"; exit 11; }
n=$(python3 -c "import json; print(len(json.load(open('$O/EXT_POINTS.json'))))")
if [ "$n" -gt 0 ] && ! grep -q '"done": true' "$O/ext/PROGRESS.log" 2>/dev/null; then
  stopped && exit 75
  mkdir -p "$O/ext"; rm -f "$O/ext/STOP"
  bg_stop "$O/ext/STOP" -- crun f4chain-S1R "$S1_MEM" 86400s -- bash -c "cd '$CODE' && $PY tools/strategy_level_grid.py run --subset '$SUBSET' --params '$CANONICAL' --points-json '$O/EXT_POINTS.json' --admitted '$DATA/admitted' --csv-dir '$DATA/sel15' --jobs $S1_JOBS --out '$O/ext' >> '$O/ext.log' 2>&1"
  stopped && exit 75
  grep -q '"done": true' "$O/ext/PROGRESS.log" 2>/dev/null || { reason "edge extension run incomplete (see $O/ext.log)"; exit 1; }
fi
$PY tools/strategy_chain_f4.py freeze --stage1 "$O/STAGE1_RESULT.json" --params "$CANONICAL" --ext "$O/ext" --edge "$O/EDGE.json" --out "$O/FREEZE.json" --out-params "$O/FROZEN_PARAMS.json" || { reason "freeze failed"; exit 11; }
state done "$D/out/TASKS.jsonl,$O/ext" "$O/STAGE1_RESULT.json,$O/EDGE.json,$O/FREEZE.json,$O/FROZEN_PARAMS.json"
