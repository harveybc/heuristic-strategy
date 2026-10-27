#!/bin/bash
# Keeper of a matrix shard (runs on the coordinator, which reaches both hosts): every 10 min it copies the shard's
# result files into the main host's result directory (the main report merges them; result files are idempotent per
# cell and seed). When the worker writes RETURNED, stays unreachable for an hour, or drains its queue, it does a final
# copy and removes the main host's EXCLUDE_IDS, so the main driver runs whatever the shard did not finish.
# Usage: shard_keeper.sh <worker> <worker shard root> <main> <main root> <local relay dir> [results subdir]
set -u
W=$1; WR=$2; M=$3; MR=$4; L=$5; SUB=${6:-results_v10}
mkdir -p $L
log(){ echo "$(date -u +%FT%TZ) [shard-keeper] $*" | tee -a $L/KEEPER.log >/dev/null; ssh -o BatchMode=yes -o ConnectTimeout=15 $M "echo '$(date -u +%FT%TZ) [shard-keeper] $*' >> $MR/LOG" < /dev/null 2>/dev/null; }
miss=0
log "keeper start: $W:$WR/$SUB -> $M:$MR/$SUB"
while true; do
  if timeout 300 rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=15" $W:$WR/$SUB/ $L/$SUB/ < /dev/null 2>/dev/null; then
    miss=0
    timeout 300 rsync -a --ignore-existing -e "ssh -o BatchMode=yes -o ConnectTimeout=15" $L/$SUB/ $M:$MR/$SUB/ < /dev/null 2>/dev/null
    state=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $W "test -f $WR/RETURNED && echo returned; grep -q 'queue drained' $WR/LOG 2>/dev/null && echo drained" < /dev/null 2>/dev/null)
  else
    miss=$((miss + 1)); state=""
    [ $miss -ge 6 ] && state="unreachable"
  fi
  if [ -n "$state" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=15 $M "rm -f $MR/EXCLUDE_IDS" < /dev/null
    log "shard ended ($(echo $state | tr '\n' ' ')): $(ls $L/$SUB | grep -c CELL_) result files copied; EXCLUDE_IDS removed on the main host"
    exit 0
  fi
  sleep 600
done
