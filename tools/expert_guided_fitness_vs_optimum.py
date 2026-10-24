#!/usr/bin/env python3
"""Owner 2026-10-24: fitness of every intermediate OPEN and CLOSE point relative to the OPTIMUM ZigZag leg, measured on
TRAIN for every resolution of the ladder (the math lives in ``app.short_horizon.expert_guided.multires.leg_tables``):
  f_open(b, d)  = d * (level[e] - o[b+1]) / leg   for the optimum leg [s, e] containing b; 1 = the optimal open,
                  negative when b lies in the optimum's contrary leg (open against the optimal scale)
  f_close(b, d) = d * (c[b] - level[s_d]) / leg_d for a trade opened at the optimal open of side d and closed at c[b]
Opens are each resolution's own pivot bars (where it says open); closes are its pivots of the closing kind."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import multires as MR  # noqa: E402
from app.short_horizon.expert_guided import oracle as OR  # noqa: E402


def stats(x):
    x = np.asarray(x, float)
    fin = x[np.isfinite(x)]
    if not len(fin):
        return {"n": 0, "n_total": int(len(x))}
    return {"n": int(len(fin)), "n_total": int(len(x)), "share_defined": float(len(fin) / len(x)),
            "p10": float(np.percentile(fin, 10)), "p50": float(np.percentile(fin, 50)), "p90": float(np.percentile(fin, 90)),
            "share_ge_0.5": float(np.mean(fin >= 0.5)), "share_le_0": float(np.mean(fin <= 0)),
            "share_negative_counter_leg": float(np.mean(fin < 0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--specs", default=",".join(MR.LADDER) + ",rev30")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract="consolidated")
    o, h, l, c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    v = ds.bars.index.as_unit("ns").asi8
    t0, t1 = (DG._utc_ns(x) for x in ds.meta["periods"]["TRAIN"])
    train = (v >= t0) & (v < t1)
    legs = MR.leg_tables(o, h, l, c, ds.sigma, ds.pivots)
    pb = np.array([int(p[0]) for p in ds.pivots], np.int64)
    conf = np.asarray(ds.support_end, np.int64)[pb]
    rows = {}
    for spec in a.specs.split(","):
        piv, _ = MR.resolution_pivots(h, l, spec, ds.pivots, conf, DG.EXPERT)
        y = OR.pivot_labels(len(c), piv)
        fo = [legs["f_open"][1 if y[b] == OR.LONG_CLASS else -1][b] for b in np.flatnonzero(train & (y != 0))]
        fc = [legs["f_close"][1 if k == Z.PEAK else -1][b] for b, k, _ in piv if train[b]]   # a peak closes a long, a valley a short
        rows[spec] = {"open": stats(fo), "close": stats(fc)}
        print(spec, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items()} for k, r in rows[spec].items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"definition": __doc__, "train": ds.meta["periods"]["TRAIN"], "by_resolution": rows}, indent=1))


if __name__ == "__main__":
    main()
