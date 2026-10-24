#!/usr/bin/env python3
"""Example profile of the multi-resolution labels (owner 2026-10-24) on the eligible flat origins of the development
periods: NOP hard / graded / negative / conflict / open long / open short, weights, fitness percentiles and the closer
labels of a position of each side held at those rows, under the declared (or given) thresholds. Labels only, nothing
fitted, nothing launched; OBS_2024 and TEST are not read."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import multires as MR  # noqa: E402
from tools.expert_guided_e1t import Ctx  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--thresholds", default="", help="JSON file with multires.Thresholds fields")
    ap.add_argument("--periods", default="TRAIN,INNER_A")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract="consolidated")
    th = MR.Thresholds.from_json(Path(a.thresholds).read_text()) if a.thresholds else MR.Thresholds()
    lab = MR.MultiResLabels(Ctx(ds), th, MR.LADDER, DG.EXPERT)
    prof = {p: lab.profile(ds, p) for p in a.periods.split(",")}
    doc = {"rule": "labels only; eligible flat origins per period; closer labels for a hypothetical position of each side; nothing fitted",
           "ladder": list(MR.LADDER), "thresholds": dataclasses.asdict(th), "periods": {p: ds.meta["periods"][p] for p in prof}, "profile": prof}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    for p, r in prof.items():
        print(p, json.dumps({"rows": r["rows"], "censored": r["censored"], **r["kinds"], **r["classes"],
                             "closer_long": r["closer"]["long"], "closer_short": r["closer"]["short"],
                             "fitness_p10_p50_p90": r["fitness_p10_p50_p90"]}), flush=True)


if __name__ == "__main__":
    main()
