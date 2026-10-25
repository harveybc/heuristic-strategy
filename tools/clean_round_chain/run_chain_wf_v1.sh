#!/bin/bash
# clean round v6 chain (lane L8, 2026-10-25): wait for the v5 chain's end marker -> install the STAGED v1 files (code,
# action contract, spec v6; verified against stage_wf_v1/STAGE.sha256) -> re-pin -> gate += WF_V1 -> localize the v6
# spec -> re-run the walk-forward report with the fixed per-fold text -> emit + run stage WF_V1 (2 folds x 2 arms x 5
# seeds, 10G, one unit at a time) -> v1 seed ensembles (every fitness variant) -> wf-v1 report + filled closure table.
# Sequential; resumable (STATE marks, report outputs); nothing here asks a human. Owner's words for WF_V1 (2026-10-25):
# "ejecuta y orquesta todo ... hay que hacerlo con toda la complejidad que se requiera".
Q=/var/tmp/expert-guided-clean-a1-20261024; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run
S=$Q/stage_wf_v1; SPEC=$Q/CLEAN_ROUND_SPEC_v6_host.json; REP=$Q/report; mkdir -p $REP $Q/STATE $Q/contracts
log(){ echo "$(date -u +%FT%TZ) [chain-v6] $*" >> $Q/LOG; }
test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log "gate missing: refusing"; exit 3; }
log "waiting for the v5 chain end marker"
for i in $(seq 1 288); do grep -q "\[chain-v5\] chain end (v5)" $Q/LOG && break; sleep 300; done
grep -q "\[chain-v5\] chain end (v5)" $Q/LOG || { log "v5 chain did not end within 24 h: stopping"; exit 5; }
log "chain start: install v1 -> WF report rerun -> WF_V1 units -> v1 ensembles -> wf-v1 report"
# 1. staged files -> package (verified first), pins replaced line by line, gate extended
( cd $S && sha256sum -c --quiet STAGE.sha256 ) || { log "staged files differ from STAGE.sha256: refusing"; exit 6; }
if [ ! -f $Q/STATE/v6_installed ]; then
  while read -r h f; do install -D -m 0644 "$S/$f" "$Q/$f" || { log "install $f FAILED"; exit 6; }; done < $S/STAGE.sha256
  cd $Q || exit 1
  cp PINS.sha256 PINS.sha256.before_v6
  while read -r h f; do grep -v "  \./$f\$" PINS.sha256 > PINS.tmp && mv PINS.tmp PINS.sha256; echo "$h  ./$f" >> PINS.sha256; done < $S/STAGE.sha256
  sha256sum -c --quiet PINS.sha256 || { log "PINS check FAILED after the v6 install"; exit 6; }
  $PY - <<'PYEOF' || { log "gate update FAILED"; exit 6; }
import json
p = "/var/tmp/expert-guided-clean-a1-20261024/LAUNCH_EXPERT_GUIDED.json"
g = json.load(open(p))
st = g.get("stage") if isinstance(g.get("stage"), list) else [g.get("stage")]
if "WF_V1" not in st:
    st.append("WF_V1")
g["stage"] = st
g.setdefault("owner_words_by_stage", {})["WF_V1"] = "ejecuta y orquesta todo ... hay que hacerlo con toda la complejidad que se requiera (2026-10-25)"
json.dump(g, open(p, "w"), indent=1)
PYEOF
  touch $Q/STATE/v6_installed
  log "v1 installed: $(wc -l < $S/STAGE.sha256) files, PINS re-verified, gate stage += WF_V1"
fi
cd $Q/code || exit 1
$PY tools/expert_guided_clean_round.py localize --spec $Q/CLEAN_ROUND_SPEC_v6.json --root $Q --out $SPEC > $Q/localize_v6.log 2>&1 || { log "localize v6 FAILED"; exit 7; }
# 2. the walk-forward report again, with the fixed per-fold verdict text (lane L8 found the TypeError)
$PY tools/expert_guided_wf_report.py walk-forward --spec $SPEC --results $Q/out --out-json $REP/RESULTS_WALK_FORWARD.json \
  --out-md $REP/RESULTS_WALK_FORWARD_2026_10_25.md > $REP/report_wf_v6.log 2>&1 && log "report WF rewritten" || log "report WF FAILED (see report/report_wf_v6.log)"
# 3. stage WF_V1
$PY tools/expert_guided_clean_round.py emit --spec $SPEC --stage WF_V1 --out $Q/queue_WF_V1.sh > $Q/emit_WF_V1.log 2>&1 || { log "emit WF_V1 FAILED (see emit_WF_V1.log)"; exit 8; }
bash $Q/queue_WF_V1.sh
# 4. v1 ensembles (every fitness variant; INNER_B scored after freeze; OBS_2024 one read for the arm's variant)
ens(){ # unit recipe split periods
  local u=$1 r=$2 sp=$3 per=$4
  [ -f $Q/out/$u/E3M.json ] || { log "ensemble $u skipped: no E3M.json"; return; }
  [ -f $Q/out/$u/ENSEMBLE.json ] && return
  for try in $(seq 1 36); do CUDA_VISIBLE_DEVICES="" $CR -q -m 6G -t 10800s -n ens-$u -- $PY tools/expert_guided_seed_ensemble.py --arm-dir $Q/out/$u \
      --admitted $Q/data/admitted --store $Q/data/store/full_pool_eurusd_h1.parquet --recipe $Q/recipe/$r --split-contract $sp --periods $per > $Q/out/$u.ensemble.log 2>&1 \
      && { log "ensemble $u DONE"; return; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $Q/out/$u.ensemble.log && { log "ensemble $u not admitted (try $try)"; sleep 600; } || { log "ensemble $u FAILED"; return; }
  done
}
P4=TRAIN,INNER_A,INNER_B,OBS_2024
for f in wf2022 wf2023; do for a in both_hold_v1 zz_hold_v1; do ens WF_V1_${f}_${a} CONSUMER_RECIPE_${f}.json $f $P4; done; done
# 5. wf-v1 report and the closure table (evidence column only)
$PY tools/expert_guided_wf_report.py wf-v1 --spec $SPEC --results $Q/out --out-json $REP/RESULTS_WF_V1.json --out-md $REP/RESULTS_WF_V1_2026_10_25.md \
  --closure-template $Q/stage_wf_v1/OUTPUT_CONTRACT_V1_CLOSURE.md --closure-out $REP/OUTPUT_CONTRACT_V1_CLOSURE.md > $REP/report_wf_v1.log 2>&1 \
  && log "report WF_V1 written" || log "report WF_V1 FAILED (see report/report_wf_v1.log)"
log "chain end (v6)"
