#!/bin/bash
# T3 t3v2_8875abf84a63 queue for satoshi (22 units). Sequential, resumable (STATE markers),
# retries while not admitted, never lowers a refused cap. A marker is not a result: verify each E3M.json by identity.
ROOT=${ROOT:?set ROOT to the package root}; PY=${PY:-$HOME/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-$HOME/.local/bin/crispdm-run}
MEM=${MEM:-4G}; GPU=${GPU:?set GPU to the CUDA index of the assigned card}
mkdir -p $ROOT/STATE $ROOT/out; log(){ echo "$(date -u +%FT%TZ) [T3] $*" >> $ROOT/LOG; }
cd $ROOT/code || exit 1
if [ ! -f $ROOT/STATE/t3_unit_00 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-00 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_00.json --out $ROOT/out/t3_unit_00 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_00.log 2>&1 && { touch $ROOT/STATE/t3_unit_00; log "t3_unit_00 DONE identity 7fd91fdb3735ce5a"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_00.log && { log "t3_unit_00 not admitted (try $try)"; sleep 600; } || { log "t3_unit_00 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_02 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-02 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_02.json --out $ROOT/out/t3_unit_02 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_02.log 2>&1 && { touch $ROOT/STATE/t3_unit_02; log "t3_unit_02 DONE identity 1a5a077d56f8c082"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_02.log && { log "t3_unit_02 not admitted (try $try)"; sleep 600; } || { log "t3_unit_02 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_04 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-04 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_04.json --out $ROOT/out/t3_unit_04 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_04.log 2>&1 && { touch $ROOT/STATE/t3_unit_04; log "t3_unit_04 DONE identity 94605149e83e394d"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_04.log && { log "t3_unit_04 not admitted (try $try)"; sleep 600; } || { log "t3_unit_04 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_06 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-06 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_06.json --out $ROOT/out/t3_unit_06 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_06.log 2>&1 && { touch $ROOT/STATE/t3_unit_06; log "t3_unit_06 DONE identity d4aa1700c57b27ed"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_06.log && { log "t3_unit_06 not admitted (try $try)"; sleep 600; } || { log "t3_unit_06 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_08 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-08 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_08.json --out $ROOT/out/t3_unit_08 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_08.log 2>&1 && { touch $ROOT/STATE/t3_unit_08; log "t3_unit_08 DONE identity 660acd79f5548e86"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_08.log && { log "t3_unit_08 not admitted (try $try)"; sleep 600; } || { log "t3_unit_08 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_10 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-10 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_10.json --out $ROOT/out/t3_unit_10 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_10.log 2>&1 && { touch $ROOT/STATE/t3_unit_10; log "t3_unit_10 DONE identity 2524c6649ff8b5f4"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_10.log && { log "t3_unit_10 not admitted (try $try)"; sleep 600; } || { log "t3_unit_10 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_12 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-12 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_12.json --out $ROOT/out/t3_unit_12 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_12.log 2>&1 && { touch $ROOT/STATE/t3_unit_12; log "t3_unit_12 DONE identity 60bc06f616b1a787"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_12.log && { log "t3_unit_12 not admitted (try $try)"; sleep 600; } || { log "t3_unit_12 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_14 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-14 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_14.json --out $ROOT/out/t3_unit_14 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_14.log 2>&1 && { touch $ROOT/STATE/t3_unit_14; log "t3_unit_14 DONE identity 9dadab8c00253550"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_14.log && { log "t3_unit_14 not admitted (try $try)"; sleep 600; } || { log "t3_unit_14 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_16 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-16 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_16.json --out $ROOT/out/t3_unit_16 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_16.log 2>&1 && { touch $ROOT/STATE/t3_unit_16; log "t3_unit_16 DONE identity 49040797da7d804e"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_16.log && { log "t3_unit_16 not admitted (try $try)"; sleep 600; } || { log "t3_unit_16 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_18 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-18 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_18.json --out $ROOT/out/t3_unit_18 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_18.log 2>&1 && { touch $ROOT/STATE/t3_unit_18; log "t3_unit_18 DONE identity 76ce2caa17441806"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_18.log && { log "t3_unit_18 not admitted (try $try)"; sleep 600; } || { log "t3_unit_18 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_20 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-20 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_20.json --out $ROOT/out/t3_unit_20 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_20.log 2>&1 && { touch $ROOT/STATE/t3_unit_20; log "t3_unit_20 DONE identity 2ebb06fbe74a1cc2"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_20.log && { log "t3_unit_20 not admitted (try $try)"; sleep 600; } || { log "t3_unit_20 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_22 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-22 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_22.json --out $ROOT/out/t3_unit_22 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_22.log 2>&1 && { touch $ROOT/STATE/t3_unit_22; log "t3_unit_22 DONE identity 0ca3457de7f05932"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_22.log && { log "t3_unit_22 not admitted (try $try)"; sleep 600; } || { log "t3_unit_22 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_24 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-24 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_24.json --out $ROOT/out/t3_unit_24 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_24.log 2>&1 && { touch $ROOT/STATE/t3_unit_24; log "t3_unit_24 DONE identity 9ab44359eaa872f8"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_24.log && { log "t3_unit_24 not admitted (try $try)"; sleep 600; } || { log "t3_unit_24 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_26 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-26 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_26.json --out $ROOT/out/t3_unit_26 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_26.log 2>&1 && { touch $ROOT/STATE/t3_unit_26; log "t3_unit_26 DONE identity 34056fd80807e5b8"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_26.log && { log "t3_unit_26 not admitted (try $try)"; sleep 600; } || { log "t3_unit_26 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_28 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-28 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_28.json --out $ROOT/out/t3_unit_28 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_28.log 2>&1 && { touch $ROOT/STATE/t3_unit_28; log "t3_unit_28 DONE identity 14b169bd0f42e823"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_28.log && { log "t3_unit_28 not admitted (try $try)"; sleep 600; } || { log "t3_unit_28 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_30 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-30 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_30.json --out $ROOT/out/t3_unit_30 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_30.log 2>&1 && { touch $ROOT/STATE/t3_unit_30; log "t3_unit_30 DONE identity 61aff45fdef5125f"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_30.log && { log "t3_unit_30 not admitted (try $try)"; sleep 600; } || { log "t3_unit_30 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_32 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-32 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_32.json --out $ROOT/out/t3_unit_32 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_32.log 2>&1 && { touch $ROOT/STATE/t3_unit_32; log "t3_unit_32 DONE identity a8f7ec369a57fe08"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_32.log && { log "t3_unit_32 not admitted (try $try)"; sleep 600; } || { log "t3_unit_32 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_34 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-34 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_34.json --out $ROOT/out/t3_unit_34 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_34.log 2>&1 && { touch $ROOT/STATE/t3_unit_34; log "t3_unit_34 DONE identity 02fb88f6ec54753b"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_34.log && { log "t3_unit_34 not admitted (try $try)"; sleep 600; } || { log "t3_unit_34 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_36 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-36 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_36.json --out $ROOT/out/t3_unit_36 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_36.log 2>&1 && { touch $ROOT/STATE/t3_unit_36; log "t3_unit_36 DONE identity f7fa03299070d641"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_36.log && { log "t3_unit_36 not admitted (try $try)"; sleep 600; } || { log "t3_unit_36 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_38 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-38 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_38.json --out $ROOT/out/t3_unit_38 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_38.log 2>&1 && { touch $ROOT/STATE/t3_unit_38; log "t3_unit_38 DONE identity 9547c6bc32cf6347"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_38.log && { log "t3_unit_38 not admitted (try $try)"; sleep 600; } || { log "t3_unit_38 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_40 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-40 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_40.json --out $ROOT/out/t3_unit_40 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_40.log 2>&1 && { touch $ROOT/STATE/t3_unit_40; log "t3_unit_40 DONE identity 056b36818ce1d3f0"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_40.log && { log "t3_unit_40 not admitted (try $try)"; sleep 600; } || { log "t3_unit_40 FAILED"; break; }
  done
fi
if [ ! -f $ROOT/STATE/t3_unit_42 ]; then
  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-42 -- $PY tools/expert_guided_e3m.py run --admitted $ROOT/data/admitted --csv-dir $ROOT/data/sel15 --store $ROOT/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list $ROOT/in/FEATURE_LIST_shortlist_strict.json --groups $ROOT/t3/groups/unit_42.json --out $ROOT/out/t3_unit_42 --variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0 > $ROOT/out/t3_unit_42.log 2>&1 && { touch $ROOT/STATE/t3_unit_42; log "t3_unit_42 DONE identity 7cd7f392af8eb50f"; break; }
    grep -q "REFUSED\|QUEUED HOST_HEADROOM" $ROOT/out/t3_unit_42.log && { log "t3_unit_42 not admitted (try $try)"; sleep 600; } || { log "t3_unit_42 FAILED"; break; }
  done
fi
log "T3_SATOSHI_QUEUE_END"
