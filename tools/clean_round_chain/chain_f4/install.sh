#!/bin/bash
# Install the strategy chain F4 into a chain root from a git checkout at a given sha (never from a working tree):
#   bash install.sh <chain_root> <repo_checkout> <sha> <chain.env> [--no-units]
# Copies the code (git archive), the chain scripts and chain.env; with units, installs and enables the systemd --user
# service and the status timer (the caller starts it). Re-running with the same sha is a no-op for the code.
set -eu
C=$1; REPO=$2; SHA=$3; ENV=$4; UNITS=${5:-}
mkdir -p "$C/code" "$C/stages" "$C/state" "$C/logs" "$C/work"
if [ "$(cat "$C/CODE_SHA" 2>/dev/null)" != "$SHA" ]; then
  rm -rf "$C/code.new"; mkdir -p "$C/code.new"
  git -C "$REPO" archive "$SHA" | tar -x -C "$C/code.new"
  [ -d "$C/code" ] && mv "$C/code" "$C/code.prev.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$C/code.new" "$C/code"
  echo "$SHA" > "$C/CODE_SHA"
fi
S=$C/code/tools/clean_round_chain/chain_f4
cp "$S/run_chain.sh" "$S/lib.sh" "$S/status.sh" "$C/"
cp "$S"/stages/*.sh "$C/stages/"
cp "$ENV" "$C/chain.env"
[ -f "$C/code/docs/short_horizon/clean_round/RUNBOOK_CADENA_ESTRATEGIA_F4.md" ] && cp "$C/code/docs/short_horizon/clean_round/RUNBOOK_CADENA_ESTRATEGIA_F4.md" "$C/RUNBOOK.md"
if [ "$UNITS" != "--no-units" ]; then
  mkdir -p ~/.config/systemd/user
  for u in strategy-chain-f4.service strategy-chain-f4-status.service strategy-chain-f4-status.timer; do
    sed "s#@CHAIN_ROOT@#$C#g" "$S/units/$u" > ~/.config/systemd/user/$u
  done
  systemctl --user daemon-reload
  systemctl --user enable strategy-chain-f4.service strategy-chain-f4-status.timer
fi
echo "installed $SHA into $C"
