#!/bin/bash
# Q -- final qualification (QUALIFICATION_SPEC v2.6) of FINAL_CHAMPION.json. Every step has its own done marker
# ($W/Q/<step>.done) and resumes from its result files: the full matrix (seeds Q_SEEDS_DEV + twins), D3 unseen seeds
# (Q_SEEDS_UNSEEN: canonical / champion / champion without reversal, paired confirm), the q-ladder over the level worlds,
# D9 size = 1 re-runs and same-genes single-horizon ablations, D4 panels (negative controls, seeded positions), D7
# closer ladder, and the D9 analysis (cluster bootstrap with Holm, trace equality, counterfactual on every early close).
STAGE=Q; source "$CHAIN_ROOT/lib.sh"
Q=$W/Q; mkdir -p "$Q"
CH=$W/FINAL_CHAMPION.json
[ -f "$CH" ] || { reason "FINAL_CHAMPION.json missing"; exit 11; }
cd "$CODE" || exit 1
step(){ [ -f "$Q/$1.done" ]; }
mark(){ touch "$Q/$1.done"; log "Q step $1 done"; }
run_matrix(){ # step queue params seeds extra
  step "$1" && return 0
  matrix_root "$Q/$1" "$2" "$3" "$4" "$Q_JOBS" "$5" "f4q-$1"; local rc=$?
  [ $rc -eq 0 ] && mark "$1"
  return $rc
}
$PY tools/strategy_chain_f4.py noreverse --params "$CH" --out "$Q/NOREVERSE.json" || exit 1
# 1. the full matrix with twins (seeds 0-2) and its tier report
run_matrix m012 "$Q_QUEUE" "$CH" "$Q_SEEDS_DEV" "" || exit $?
if ! step m012_report; then
  crun f4chain-Q-report 3G 7200s -- $PY tools/strategy_tier_report.py --results "$Q/m012/results" --out "$Q/m012/report_tier" --expect-profile full_f4 > "$Q/m012_report.log" 2>&1 || { reason "tier report failed"; exit 1; }
  mark m012_report
fi
# 2. D3 unseen confirmation seeds
run_matrix m345_canonical "$Q_QUEUE" "$CANONICAL" "$Q_SEEDS_UNSEEN" "" || exit $?
run_matrix m345_deap "$Q_QUEUE" "$CH" "$Q_SEEDS_UNSEEN" "" || exit $?
run_matrix m345_noreverse "$Q_QUEUE" "$Q/NOREVERSE.json" "$Q_SEEDS_UNSEEN" "" || exit $?
if ! step d3; then
  $PY tools/strategy_qualification.py confirm --manifest "$Q_QUEUE" --seeds-a "$Q_SEEDS_UNSEEN" --seeds-b "$Q_SEEDS_UNSEEN" --a "$Q/m345_canonical/results" --b "$Q/m345_deap/results" --out "$Q/D3_canonical_vs_deap.json" > "$Q/d3.log" 2>&1 || { reason "D3 confirm failed"; exit 1; }
  $PY tools/strategy_qualification.py confirm --manifest "$Q_QUEUE" --seeds-a "$Q_SEEDS_UNSEEN" --seeds-b "$Q_SEEDS_UNSEEN" --a "$Q/m345_noreverse/results" --b "$Q/m345_deap/results" --out "$Q/D3_noreverse_vs_deap.json" >> "$Q/d3.log" 2>&1 || { reason "D3 confirm (no reversal) failed"; exit 1; }
  mark d3
fi
# 3. the q-ladder over the level worlds, size = 1, single-horizon ablations
for q in $Q_LADDER; do
  $PY tools/strategy_chain_f4.py cells --subset "$SUBSET" --quality $q --limit ${Q_LEVEL_LIMIT:-0} --out "$Q/cells_q$q.json" > /dev/null || exit 1
  run_matrix "ladder_q$q" "$Q/cells_q$q.json" "$CH" "$Q_SEEDS_DEV" "" || exit $?
done
run_matrix size_one "$Q/cells_q1.00.json" "$CH" "$Q_SEEDS_DEV" "--size-one" || exit $?
for h in 6 12 24; do run_matrix "ablation_h$h" "$Q/cells_q1.00.json" "$CH" "$Q_SEEDS_DEV" "--menu-horizons $h" || exit $?; done
IDS=$(python3 -c "import json; print(','.join(c['id'] for c in json.load(open('$Q/cells_q1.00.json'))))")
# 4. D4 panels and D7 closer ladder on the level worlds
if ! step d4_controls; then stopped && exit 75
  crun f4chain-Q-D4c "$PANEL_MEM" 172800s -- $PY tools/strategy_qualification.py controls --cells "$Q/cells_q1.00.json" --ids "$IDS" --params "$CH" --admitted "$DATA/admitted" --csv-dir "$DATA/sel15" --out "$Q/D4_controls" > "$Q/d4c.log" 2>&1 || { reason "D4 controls failed"; exit 1; }
  mark d4_controls; fi
if ! step d4_seeded; then stopped && exit 75
  crun f4chain-Q-D4s "$PANEL_MEM" 172800s -- $PY tools/strategy_qualification.py seeded --cells "$Q/cells_q1.00.json" --ids "$IDS" --params "$CH" --admitted "$DATA/admitted" --csv-dir "$DATA/sel15" --out "$Q/D4_seeded" > "$Q/d4s.log" 2>&1 || { reason "D4 seeded failed"; exit 1; }
  mark d4_seeded; fi
if ! step d7; then stopped && exit 75
  RULE=$(python3 -c "import json; d=json.load(open('$CH')); print('%s:%d:%s' % (d['z_close'], int(round(d['k_persist'])), d['closer_filter']))")
  D7IDS=$(python3 -c "print(','.join(x + ':0' for x in '$IDS'.split(',')))")
  crun f4chain-Q-D7 "$PANEL_MEM" 172800s -- $PY tools/strategy_closer_ladder.py --cells "$Q/cells_q1.00.json" --ids "$D7IDS" --params "$CH" --admitted "$DATA/admitted" --csv-dir "$DATA/sel15" --qualities 0.85,0.70 --phis 0.587 --rules "$RULE" --out "$Q/D7" > "$Q/d7.log" 2>&1 || { reason "D7 closer ladder failed"; exit 1; }
  mark d7; fi
# 5. D9 analysis
$PY tools/strategy_chain_f4.py d9 --qroot "$Q" --out "$Q/D9.json" > "$Q/d9.log" 2>&1 || { reason "D9 analysis failed"; exit 1; }
state done "$CH,$Q_QUEUE" "$Q/m012/report_tier/TIER_REPORT.json,$Q/D3_canonical_vs_deap.json,$Q/D3_noreverse_vs_deap.json,$Q/D9.json,$Q/D4_controls,$Q/D4_seeded,$Q/D7"
