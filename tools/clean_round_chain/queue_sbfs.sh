#!/bin/bash
# SBFS with the final model (Musashi's plan, my evaluator); emitted by tools/expert_guided_sbfs.py; NOT launched by the emitter.
Q=<package-root>; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run; R=<package-root>/sbfs-c3-20261026
log(){ echo "$(date -u +%FT%TZ) [sbfs] $*" >> $Q/LOG; }
test -f $Q/LAUNCH_EXPERT_GUIDED.json || { log 'gate missing: refusing'; exit 3; }
cd $Q/code || exit 1
if [ ! -f $R/SPEC.private.json ]; then
  $PY tools/expert_guided_sbfs.py prepare --root $R --recipe $Q/recipe/CONSUMER_RECIPE.json --store $Q/data/store/full_pool_eurusd_h1.parquet --manifest $Q/data/store/MANIFEST.json --admitted $Q/data/admitted --csv-dir $Q/data/sel15 --action-contract $Q/contracts/ACTION_CONTRACT_C3.json --selector-root $Q/feature_selector --contract consolidated --python $PY --crispdm-run $CR --memory 10G || { log 'prepare FAILED'; exit 4; }
fi
# one supervisor; every E3-M fit inside runs under crispdm with the per-fit cap; resumable (tasks/, CONTROLS.json, subsets/, CLOSE.json)
$PY tools/expert_guided_sbfs.py run --root $R --evaluator-path $Q/code/app/short_horizon/expert_guided/fs_evaluator.py --selector-root $Q/feature_selector/src > $R/run.log 2>&1 && log 'sbfs run finished' || { log 'sbfs run FAILED'; exit 5; }
log 'chain end (sbfs)'
