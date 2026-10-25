#!/bin/bash
# clean round v5 chain (lane L7, 2026-10-25): wait for the seeds->B chain -> re-run B_winner_multires_aux (fixed) -> seed
# ensembles + Sharpe standard errors for every finished A / A_prime / B arm -> report B+ensemble -> walk-forward units
# (2 folds x 3 arms x 5 seeds on derived fold recipes) -> ensembles on the folds -> walk-forward report. Sequential, one
# unit at a time, every unit memory-capped (10G) by the emitted queues / crispdm-run; resumable through STATE marks and
# the report steps' own output files. Nothing here asks a human.
Q=/var/tmp/expert-guided-clean-a1-20261024; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
SPEC=$Q/CLEAN_ROUND_SPEC_v5_host.json; REP=$Q/report; mkdir -p $REP $Q/STATE
log(){ echo "$(date -u +%FT%TZ) [chain-v5] $*" >> $Q/LOG; }
test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log "gate missing: refusing"; exit 3; }
# 0. wait for the previous chain's end marker (max 12 h)
for i in $(seq 1 144); do grep -q "\[chain-v4\] chain end" $Q/LOG && break; sleep 300; done
grep -q "\[chain-v4\] chain end" $Q/LOG || { log "previous chain did not end within 12 h: stopping"; exit 5; }
log "chain start: B multires_aux rerun -> ensembles -> report B -> WF units -> WF ensembles -> report WF"
cd $Q/code || exit 1
# 1. re-run the B arm that failed on the fixed multires aux path (same unit name; its STATE mark is absent)
if [ ! -f $Q/STATE/B_winner_multires_aux ]; then
  $PY tools/expert_guided_clean_round.py emit --spec $SPEC --stage B --winner-from $Q/DECISION_A_5SEED.json --arms winner_multires_aux --out $Q/queue_B_multires_aux.sh > $Q/emit_B_multires_aux.log 2>&1 \
    || { log "emit B multires_aux FAILED (see emit_B_multires_aux.log)"; exit 6; }
  bash $Q/queue_B_multires_aux.sh
fi
# 2. ensembles + SE for every finished arm of A / A_prime / B (pooled seeds for the two `both` arms); CPU, capped
ens(){ # unit recipe split periods [extra dirs...]
  local u=$1 r=$2 sp=$3 per=$4; shift 4; local extra=""; for d in "$@"; do extra="$extra --extra-arm-dir $Q/out/$d"; done
  [ -f $Q/out/$u/E3M.json ] || { log "ensemble $u skipped: no E3M.json"; return; }
  [ -f $Q/out/$u/ENSEMBLE.json ] && return
  for try in $(seq 1 36); do CUDA_VISIBLE_DEVICES="" $CR -q -m 6G -t 7200s -n ens-$u -- $PY tools/expert_guided_seed_ensemble.py --arm-dir $Q/out/$u $extra \
      --admitted $Q/data/admitted --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/$r --split-contract $sp --periods $per > $Q/out/$u.ensemble.log 2>&1 \
      && { log "ensemble $u DONE"; return; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $Q/out/$u.ensemble.log && { log "ensemble $u not admitted (try $try)"; sleep 600; } || { log "ensemble $u FAILED"; return; }
  done
}
P3=TRAIN,INNER_A,OBS_2024
for u in A_zz_hold A_zz_hold_shorter A_opp_hold A_opp_optimum A_opp_hold_shorter A_prime_multires_default A_prime_multires_deap A_prime_multires_deap_weighted B_winner_aux B_winner_multires_aux; do ens $u CONSUMER_RECIPE.json consolidated $P3; done
ens A_both_hold CONSUMER_RECIPE.json consolidated $P3 A_both_hold_seeds2
ens A_both_hold_shorter CONSUMER_RECIPE.json consolidated $P3 A_both_hold_shorter_seeds2
# 3. report B + ensembles (host copy; collect.sh on the coordinator brings it into the worktree)
$PY tools/expert_guided_wf_report.py b-ensemble --spec $SPEC --results $Q/out --decision-5seed $Q/DECISION_A_5SEED.json \
  --out-json $REP/RESULTS_B_AND_ENSEMBLE.json --out-md $REP/RESULTS_B_AND_ENSEMBLE_2026_10_25.md > $REP/report_b.log 2>&1 && log "report B+ensemble written" || log "report B+ensemble FAILED (see report/report_b.log)"
# 4. walk-forward units (emitted from the spec: fold recipes, fold split, pre-registered seeds, 10G)
$PY tools/expert_guided_clean_round.py emit --spec $SPEC --stage WF --out $Q/queue_WF.sh > $Q/emit_WF.log 2>&1 || { log "emit WF FAILED (see emit_WF.log)"; exit 7; }
bash $Q/queue_WF.sh
# 5. ensembles on the folds (validation year INNER_B scored after freeze; OBS_2024 one read) and the walk-forward report
P4=TRAIN,INNER_A,INNER_B,OBS_2024
for f in wf2022 wf2023; do for a in zz_hold both_hold both_hold_shorter; do ens WF_${f}_${a} CONSUMER_RECIPE_${f}.json $f $P4; done; done
$PY tools/expert_guided_wf_report.py walk-forward --spec $SPEC --results $Q/out \
  --out-json $REP/RESULTS_WALK_FORWARD.json --out-md $REP/RESULTS_WALK_FORWARD_2026_10_25.md > $REP/report_wf.log 2>&1 && log "report WF written" || log "report WF FAILED (see report/report_wf.log)"
log "chain end (v5)"
