#!/bin/bash
# DIAGNOSTIC pipeline-control queue: strictly sequential (one heavy job at a time on the owner's desktop), resumable.
D=/var/tmp/pipeline-controls-20261026; CODE=/var/tmp/hs-controls-20261026; PY=$HOME/anaconda3/envs/trading-stack/bin/python; CR=$HOME/.local/bin/crispdm-run
SYNPIN=6d771a43e3a82b52e77447438620559ad2453af64118d6a33ad3d291d69aa080
mkdir -p $D/STATE $D/outcomes
log(){ echo "$(date -u +%FT%TZ) [queue] $*" >> $D/LOG; }
step(){ local n=$1; shift; [ -f $D/STATE/$n ] && return 0; log "begin $n"; "$@" && { touch $D/STATE/$n; log "done $n"; } || log "FAILED $n"; }
outc(){ # name datadir admitted [pin]
  cd $CODE && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 $CR -q -m 3G -t 10800s -n ctrl-outc-$1 -- $PY tools/pipeline_controls.py run-tool --pin-admitted "${4:-}" -- tools/expert_guided_c2.py outcomes \
    --admitted $3 --store $2/store/full_pool_eurusd_h1.parquet --recipe $(ls $2/recipe/*.json) --split-contract wf2022 --out $D/outcomes/OUTCOMES_ENTRY_wf2022_DIAGNOSTIC_$1.npz > $D/outcomes/$1.log 2>&1; }
U=$D/run_unit.sh
SEED=${SEED:-20261019}
step cheat_v13 $U cheat_v13 $D/DIAGNOSTIC_cheat v13 $SEED
step outc_REALBARS outc REALBARS $D/DIAGNOSTIC_cheat /var/tmp/fs-timing/admitted
step cheat_c2 $U cheat_c2 $D/DIAGNOSTIC_cheat c2 $SEED
step outc_SYNTHETIC outc SYNTHETIC $D/DIAGNOSTIC_synthetic $D/DIAGNOSTIC_synthetic/admitted $SYNPIN
step syn_v13 env ADM=$D/DIAGNOSTIC_synthetic/admitted $U syn_v13 $D/DIAGNOSTIC_synthetic v13 $SEED --pin-admitted $SYNPIN
step syn_c2 env ADM=$D/DIAGNOSTIC_synthetic/admitted OUTCOMES=$D/outcomes/OUTCOMES_ENTRY_wf2022_DIAGNOSTIC_SYNTHETIC.npz $U syn_c2 $D/DIAGNOSTIC_synthetic c2 $SEED --pin-admitted $SYNPIN
step shuf_v13 $U shuf_v13 $D/DIAGNOSTIC_cheat v13 $SEED --shuffle-labels-within-day 7
step shuf_c2 $U shuf_c2 $D/DIAGNOSTIC_cheat c2 $SEED --shuffle-labels-within-day 7
step noiseinp_v13 $U noiseinp_v13 $D/DIAGNOSTIC_noise_inputs v13 $SEED
step noiseinp_c2 $U noiseinp_c2 $D/DIAGNOSTIC_noise_inputs c2 $SEED
for s in 0.5 1 2; do
  step n${s}_c2 $U cheat_noise${s}_c2 $D/DIAGNOSTIC_cheat_noise$s c2 $SEED
  step n${s}_v13 $U cheat_noise${s}_v13 $D/DIAGNOSTIC_cheat_noise$s v13 $SEED
done
log "queue end"
