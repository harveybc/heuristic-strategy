#!/bin/bash
# One-command status of the strategy chain F4 (regenerates STATUS.md / STATUS.json from files, then prints them).
C=$(cd "$(dirname "$0")" && pwd)
source "$C/chain.env"
cd "$C/code" && $PY tools/strategy_chain_f4.py status --chain "$C" --quiet && cat "$C/STATUS.md"
systemctl --user --no-pager status strategy-chain-f4.service 2>/dev/null | head -5
