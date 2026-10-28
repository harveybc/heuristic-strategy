#!/usr/bin/env python3
"""Choose features and units for E1-T / E2-T from E0-T run B (rule fixed 2026-10-20 BEFORE run B was read).

1. Capacity inside the v4 four: the arm chosen by run B's ladder (rule v2).
2. Feature set: that arm versus T2_core15, by seed-mean calibrated INNER_A log-loss on the hindsight target (same
   rows, same target); the lower wins; an exact tie goes to fewer parameters. INNER_B and 2024 are never used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e0t", required=True, help="E0T.json of run B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.e0t).read_text())
    seeds = sorted(d["students"]["T2_v4"]["hindsight"])

    def ll(arm):
        return float(np.mean([d["students"][arm]["hindsight"][s]["losses"]["INNER_A"] for s in seeds]))

    def params(arm):
        return d["architectures"][arm]["layers"][-1]["params"]
    v4 = d["ladder"]["chosen"]
    cand = [v4, "T2_core15"]
    win = min(cand, key=lambda k: (ll(k), params(k)))
    cfg = {"arm": win, "features": d["arms"][win]["features"], "units": d["arms"][win]["units"],
           "evidence": {k: {"inner_a_log_loss_seed_mean": ll(k), "params": params(k)} for k in cand},
           "ladder": d["ladder"], "rule": __doc__}
    Path(a.out).write_text(json.dumps(cfg, indent=1))
    print(json.dumps({k: cfg[k] for k in ("arm", "features", "units", "evidence")}))


if __name__ == "__main__":
    main()
