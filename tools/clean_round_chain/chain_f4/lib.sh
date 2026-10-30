#!/bin/bash
# Strategy chain F4 -- shared helpers (sourced by run_chain.sh and every stage). No kill / pkill anywhere: stops are
# STOP flags (the chain's $C/STOP, propagated to the running tool's own STOP file) or the owner's systemctl stop.
# Exit codes of a stage: 0 done, 75 stopped by STOP (resumable), 11 typed failure without retry, anything else = a
# failure the driver retries ONCE.
set -u
C=${CHAIN_ROOT:?CHAIN_ROOT not set}
source "$C/chain.env"
export CUDA_VISIBLE_DEVICES=""
CODE=${CODE:-$C/code}
CODE_SHA=$(cat "$C/CODE_SHA" 2>/dev/null || echo unknown)
W=$C/work
mkdir -p "$W" "$C/state" "$C/logs"
log(){ echo "$(date -u +%FT%TZ) [$STAGE] $*" >> "$C/logs/chain.log"; }
stopped(){ [ -f "$C/STOP" ]; }
state(){ # kind [inputs] [outputs] [reason]
  (cd "$CODE" && $PY tools/strategy_chain_f4.py state --chain "$C" --stage "$STAGE" --kind "$1" --code-sha "$CODE_SHA" --inputs "${2:-}" --outputs "${3:-}" --reason "${4:-}") >> "$C/logs/chain.log" 2>&1
}
host_free(){ python3 ~/.local/libexec/crispdm/crispdm_admission.py state | python3 -c 'import json,sys; print(json.load(sys.stdin)["host_free_for_new_bytes"])'; }
to_bytes(){ python3 -c "import sys; s=sys.argv[1].upper(); m={'K':2**10,'M':2**20,'G':2**30}; print(int(float(s[:-1])*m[s[-1]]) if s[-1] in m else int(s))" "$1"; }
wait_mem(){ # cap: wait until host_free >= cap + MARGIN (coordinator rule); returns 75 on STOP
  local need=$(( $(to_bytes "$1") + $(to_bytes "$MARGIN") ))
  while [ "$(host_free)" -lt "$need" ]; do stopped && return 75; sleep 120; done
  return 0
}
scope_active(){ # job-name prefix -> 0 when a crispdm scope with that name is active
  systemctl --user list-units --type=scope --state=active --no-legend "crispdm-$1-*" 2>/dev/null | grep -q .
}
wait_scope_gone(){ # never launch a duplicate of a job that survived a driver restart
  while scope_active "$1"; do log "scope crispdm-$1-* still active: waiting"; sleep 120; done
}
crun(){ # name cap wall -- cmd... : memory-admitted through crispdm-run only
  local name=$1 cap=$2 wall=$3; shift 4
  wait_scope_gone "$name"
  wait_mem "$cap" || return 75
  log "launch $name cap=$cap wall=$wall"
  ~/.local/bin/crispdm-run -q -W 43200 -m "$cap" -t "$wall" -n "$name" -- "$@"
}
bg_stop(){ # stopfile -- cmd... : run cmd in the background, propagate the chain STOP to the tool's own STOP file between its tasks
  local sf=$1; shift 2
  "$@" &
  local pid=$!
  while [ -e /proc/$pid ]; do stopped && touch "$sf"; sleep 30; done
  wait $pid
}
reason(){ echo "$*" > "$C/state/$STAGE.reason"; log "reason: $*"; }
must(){ # cmd... : a STOP while the command waited (memory admission) or ran is a stop (exit 75), never a failed attempt
  "$@"; local rc=$?
  [ $rc -eq 0 ] && return 0
  stopped && exit 75
  return $rc
}
matrix_root(){ # dir queue_json params seeds jobs extra_args name_prefix : one ONESHOT driver root (resumable: result files)
  local root=$1 queue=$2 params=$3 seeds=$4 jobs=$5 extra=$6 pfx=$7
  mkdir -p "$root/queue" "$root/logs" "$root/claims"
  [ -e "$root/data" ] || ln -s "$DATA" "$root/data"
  [ -e "$root/code" ] || ln -s "$CODE" "$root/code"
  cp "$queue" "$root/queue/q.json"
  stopped && { touch "$root/STOP"; return 75; }
  rm -f "$root/STOP"
  wait_mem "$(( jobs * $(to_bytes $MATRIX_JOB_MEM) ))" || return 75
  ( cd "$root" && ROOT=$root RES=$root/results JOBS=$jobs ONESHOT=1 SEEDS=$seeds MEM=$MATRIX_JOB_MEM PARAMS=$params EXTRA="$extra" NAME_PREFIX=$pfx NO_LOOP_REPORT=1 \
      bash "$CODE/tools/behaviour_matrix_driver.sh" ) > "$root/logs/driver.log" 2>&1 &
  local pid=$!
  while [ -e /proc/$pid ]; do                                      # the driver honours $root/STOP (running cells finish)
    stopped && touch "$root/STOP"
    sleep 60
  done
  stopped && return 75
  grep -q "ONESHOT driver exits" "$root/LOG" || return 1
  return 0
}
