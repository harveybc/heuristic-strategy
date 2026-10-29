#!/bin/bash
# lane F10 cross-quality arm (ruling 2026-10-29): EV heads at q_ev, hierarchy heads at q_zz; gated vs off; 24 DEAP-subset worlds x seeds 0-2; code 6ee6cf7
X=<scratchpad>/f10_cross; D=<scratchpad>/f9/diag/data; export CUDA_VISIBLE_DEVICES=""
cd $X/code
for pair in 0.62:0.80 0.62:0.90 0.70:0.90; do
  qe=${pair%%:*}; qz=${pair##*:}
  python3 -c "
import json; d=json.load(open('$X/CELLS_base.json'))
for c in d: c['quality']=float('$qe'); c['zz_quality']=float('$qz')
json.dump(d, open('$X/CELLS_ev${qe}_zz${qz}.json','w'))"
  for arm in gated off; do
    [ -f $X/STOP ] && exit 0
    o=$X/ev${qe}_zz${qz}/$arm; mkdir -p $o
    for k in 0 1; do
      ( crispdm-run -q -W 7200 -m 1450M -t 10800s -n f10-cross-$qe-$qz-$arm-$k -- python tools/strategy_behaviour_matrix.py run --cells $X/CELLS_ev${qe}_zz${qz}.json --fold wf2022 --admitted $D/admitted --csv-dir $D/sel15           --seeds 0,1,2 --chunk $k/2 --veto-arm $arm --params docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json --out $o > $X/log_${qe}_${qz}_${arm}_$k.txt 2>&1 ) &
      sleep 3
    done
    wait
    echo "$(date -u +%FT%TZ) ev$qe zz$qz $arm done $(ls $o | grep -c CELL_)" >> $X/PROGRESS
  done
done
echo ALL > $X/ALL_DONE
