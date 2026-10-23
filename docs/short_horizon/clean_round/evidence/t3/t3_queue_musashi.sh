#!/bin/bash
# T3 t3v2_8875abf84a63 queue for musashi (21 units). Sequential, resumable (STATE markers),
# retries while not admitted, never lowers a refused cap. A marker is not a result: verify each E3M.json by identity.
ROOT=${ROOT:?set ROOT to the package root}; PY=${PY:-$HOME/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-$HOME/.local/bin/crispdm-run}
MEM=${MEM:-4G}; GPU=${GPU:?set GPU to the CUDA index of the assigned card}
mkdir -p $ROOT/STATE $ROOT/out; log(){ echo "$(date -u +%FT%TZ) [T3] $*" >> $ROOT/LOG; }
cd $ROOT/code || exit 1
if [ ! -f $ROOT/STATE/t3_unit_01 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-01 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_01.json --out $ROOT/out/t3_unit_01 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_01.log 2>&1 && { touch $ROOT/STATE/t3_unit_01; log "t3_unit_01 DONE identity 02508df7283513c9"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_01.log && { log "t3_unit_01 not admitted (try $try)"; sleep 600; } || { log "t3_unit_01 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_03 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-03 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_03.json --out $ROOT/out/t3_unit_03 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_03.log 2>&1 && { touch $ROOT/STATE/t3_unit_03; log "t3_unit_03 DONE identity 27c0e6b7420e4ae5"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_03.log && { log "t3_unit_03 not admitted (try $try)"; sleep 600; } || { log "t3_unit_03 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_05 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-05 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_05.json --out $ROOT/out/t3_unit_05 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_05.log 2>&1 && { touch $ROOT/STATE/t3_unit_05; log "t3_unit_05 DONE identity bd7ff893d3150f73"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_05.log && { log "t3_unit_05 not admitted (try $try)"; sleep 600; } || { log "t3_unit_05 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_07 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-07 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_07.json --out $ROOT/out/t3_unit_07 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_07.log 2>&1 && { touch $ROOT/STATE/t3_unit_07; log "t3_unit_07 DONE identity 55fec7152afd38c6"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_07.log && { log "t3_unit_07 not admitted (try $try)"; sleep 600; } || { log "t3_unit_07 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_09 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-09 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_09.json --out $ROOT/out/t3_unit_09 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_09.log 2>&1 && { touch $ROOT/STATE/t3_unit_09; log "t3_unit_09 DONE identity ee7d7e22c9f09b63"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_09.log && { log "t3_unit_09 not admitted (try $try)"; sleep 600; } || { log "t3_unit_09 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_11 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-11 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_11.json --out $ROOT/out/t3_unit_11 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_11.log 2>&1 && { touch $ROOT/STATE/t3_unit_11; log "t3_unit_11 DONE identity a7d7cf8cf081fb53"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_11.log && { log "t3_unit_11 not admitted (try $try)"; sleep 600; } || { log "t3_unit_11 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_13 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-13 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_13.json --out $ROOT/out/t3_unit_13 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_13.log 2>&1 && { touch $ROOT/STATE/t3_unit_13; log "t3_unit_13 DONE identity 546731eacfbebc12"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_13.log && { log "t3_unit_13 not admitted (try $try)"; sleep 600; } || { log "t3_unit_13 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_15 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-15 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_15.json --out $ROOT/out/t3_unit_15 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_15.log 2>&1 && { touch $ROOT/STATE/t3_unit_15; log "t3_unit_15 DONE identity e54e6c3227f1c3ab"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_15.log && { log "t3_unit_15 not admitted (try $try)"; sleep 600; } || { log "t3_unit_15 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_17 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-17 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_17.json --out $ROOT/out/t3_unit_17 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_17.log 2>&1 && { touch $ROOT/STATE/t3_unit_17; log "t3_unit_17 DONE identity 77dec77ce4e25b99"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_17.log && { log "t3_unit_17 not admitted (try $try)"; sleep 600; } || { log "t3_unit_17 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_19 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-19 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_19.json --out $ROOT/out/t3_unit_19 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_19.log 2>&1 && { touch $ROOT/STATE/t3_unit_19; log "t3_unit_19 DONE identity 1ac6452a4e8de53a"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_19.log && { log "t3_unit_19 not admitted (try $try)"; sleep 600; } || { log "t3_unit_19 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_21 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-21 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_21.json --out $ROOT/out/t3_unit_21 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_21.log 2>&1 && { touch $ROOT/STATE/t3_unit_21; log "t3_unit_21 DONE identity 7b2cdf8d699b4d4c"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_21.log && { log "t3_unit_21 not admitted (try $try)"; sleep 600; } || { log "t3_unit_21 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_23 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-23 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_23.json --out $ROOT/out/t3_unit_23 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_23.log 2>&1 && { touch $ROOT/STATE/t3_unit_23; log "t3_unit_23 DONE identity 0f0f71465e692b88"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_23.log && { log "t3_unit_23 not admitted (try $try)"; sleep 600; } || { log "t3_unit_23 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_25 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-25 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_25.json --out $ROOT/out/t3_unit_25 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_25.log 2>&1 && { touch $ROOT/STATE/t3_unit_25; log "t3_unit_25 DONE identity 6f0273747e2b9cb8"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_25.log && { log "t3_unit_25 not admitted (try $try)"; sleep 600; } || { log "t3_unit_25 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_27 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-27 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_27.json --out $ROOT/out/t3_unit_27 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_27.log 2>&1 && { touch $ROOT/STATE/t3_unit_27; log "t3_unit_27 DONE identity a170ec48d3dcd4d2"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_27.log && { log "t3_unit_27 not admitted (try $try)"; sleep 600; } || { log "t3_unit_27 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_29 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-29 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_29.json --out $ROOT/out/t3_unit_29 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_29.log 2>&1 && { touch $ROOT/STATE/t3_unit_29; log "t3_unit_29 DONE identity 9367f606e808eb1e"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_29.log && { log "t3_unit_29 not admitted (try $try)"; sleep 600; } || { log "t3_unit_29 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_31 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-31 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_31.json --out $ROOT/out/t3_unit_31 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_31.log 2>&1 && { touch $ROOT/STATE/t3_unit_31; log "t3_unit_31 DONE identity f34bed7d4428e237"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_31.log && { log "t3_unit_31 not admitted (try $try)"; sleep 600; } || { log "t3_unit_31 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_33 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-33 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_33.json --out $ROOT/out/t3_unit_33 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_33.log 2>&1 && { touch $ROOT/STATE/t3_unit_33; log "t3_unit_33 DONE identity 3a73728203dbde2f"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_33.log && { log "t3_unit_33 not admitted (try $try)"; sleep 600; } || { log "t3_unit_33 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_35 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-35 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_35.json --out $ROOT/out/t3_unit_35 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_35.log 2>&1 && { touch $ROOT/STATE/t3_unit_35; log "t3_unit_35 DONE identity 85725e3373bd5712"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_35.log && { log "t3_unit_35 not admitted (try $try)"; sleep 600; } || { log "t3_unit_35 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_37 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-37 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_37.json --out $ROOT/out/t3_unit_37 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_37.log 2>&1 && { touch $ROOT/STATE/t3_unit_37; log "t3_unit_37 DONE identity 27bc8edb223c417b"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_37.log && { log "t3_unit_37 not admitted (try $try)"; sleep 600; } || { log "t3_unit_37 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_39 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-39 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_39.json --out $ROOT/out/t3_unit_39 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_39.log 2>&1 && { touch $ROOT/STATE/t3_unit_39; log "t3_unit_39 DONE identity 6c3ec04bf4c5f02b"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_39.log && { log "t3_unit_39 not admitted (try $try)"; sleep 600; } || { log "t3_unit_39 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_41 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-41 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_41.json --out $ROOT/out/t3_unit_41 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_41.log 2>&1 && { touch $ROOT/STATE/t3_unit_41; log "t3_unit_41 DONE identity d90574204ac5c827"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_41.log && { log "t3_unit_41 not admitted (try $try)"; sleep 600; } || { log "t3_unit_41 FAILED"; break; }
  done
fi
log "T3_MUSASHI_QUEUE_END"
