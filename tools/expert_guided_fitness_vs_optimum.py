#!/usr/bin/env python3
"""Owner 2026-10-24: fitness of every intermediate OPEN and CLOSE point relative to the OPTIMUM ZigZag leg.
  open fitness  f_open(b, d)  = gain captured by opening at open(b+1) on side d and closing at the optimum's next
                               favourable pivot, as a fraction of the full optimal leg (valley->peak for long);
                               1 = the optimal open; < 0 = opening against the optimal leg
  close fitness f_close(b, d) = gain of a trade opened at the optimal open of the leg containing b, closed at close(b),
                               as a fraction of the full optimal leg; 1 at the optimal close, < 1 before, falling after
Measured on TRAIN for every resolution's demonstrated opens (where the finer ZigZag says open) and closes (its pivots)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import data as DG, oracle as OR
from tools.expert_guided_e1t import Ctx
from tools.expert_guided_e3m import parse_track
import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--admitted", required=True); ap.add_argument("--csv-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract="consolidated")
cx = Ctx(ds); h, l, c, o = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE", "OPEN"))
v = ds.bars.index.as_unit("ns").asi8; t0, t1 = (DG._utc_ns(x) for x in ds.meta["periods"]["TRAIN"]); train = (v >= t0) & (v < t1)
piv = ds.pivots; pb = np.array([p[0] for p in piv]); pk = np.array([p[1] for p in piv]); pl = np.array([p[2] for p in piv])
def leg_of(b, d):
    """optimum leg [start, end] of side d containing bar b: start = last pivot <= b of the opposite kind (valley for long)"""
    i = np.searchsorted(pb, b, side="right") - 1
    if i < 0 or i + 1 >= len(pb): return None
    want_start = Z.VALLEY if d > 0 else Z.PEAK
    if pk[i] != want_start:      # b lies inside an opposite leg
        return None
    return i, i + 1
def f_open(b, d):
    lg = leg_of(b, d)
    if lg is None or b + 1 >= len(o): return np.nan
    s, e = lg; full = d * (pl[e] - pl[s]); 
    return d * (pl[e] - o[b + 1]) / full if full > 0 else np.nan
def f_close(b, d):
    lg = leg_of(b, d)
    if lg is None:               # after the optimal close: the next leg is adverse -> gain measured from the previous same-side leg
        i = np.searchsorted(pb, b, side="right") - 1
        if i < 1: return np.nan
        s, e = i - 1, i
        if pk[s] != (Z.VALLEY if d > 0 else Z.PEAK): return np.nan
    else:
        s, e = lg
    full = d * (pl[e] - pl[s])
    return d * (c[b] - pl[s]) / full if full > 0 else np.nan
rows = {}
for spec in ("3", "rev30", "rev20", "rev15", "rev10", "rev8", "rev5"):
    t = parse_track(cx, spec) if spec != "3" else None
    y = ds.y_hind if t is None else t.y
    tp = piv if t is None else (OR.reversal_pivots(h, l, float(spec[3:]) * Z.PIP)[0])
    fo = {1: [], -1: []}; fc = {1: [], -1: []}
    for b in np.flatnonzero(train & (y != 0)):
        d = 1 if y[b] == OR.LONG_CLASS else -1
        fo[d].append(f_open(int(b), d))
    for (b, k, lv) in tp:
        if not train[b]: continue
        d = 1 if k == Z.PEAK else -1          # a peak closes a long, a valley closes a short
        fc[d].append(f_close(int(b), d))
    def st(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return {"n": int(len(x)), "share_inside_optimal_leg": None, "p10": float(np.percentile(x, 10)) if len(x) else None, "p50": float(np.percentile(x, 50)) if len(x) else None,
                "p90": float(np.percentile(x, 90)) if len(x) else None, "share_ge_0.5": float(np.mean(x >= 0.5)) if len(x) else None, "share_le_0": float(np.mean(x <= 0)) if len(x) else None}
    oo = fo[1] + fo[-1]; cc = fc[1] + fc[-1]
    rows[spec] = {"open": st(oo) | {"n_total": len(oo), "share_inside_optimal_leg": float(np.mean(np.isfinite(oo))) if len(oo) else None},
                  "close": st(cc) | {"n_total": len(cc), "share_inside_optimal_leg": float(np.mean(np.isfinite(cc))) if len(cc) else None}}
    print(spec, "open", {k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in rows[spec]["open"].items()}, "close", {k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in rows[spec]["close"].items()}, flush=True)
Path(a.out).write_text(json.dumps({"definition": __doc__, "train": ds.meta["periods"]["TRAIN"], "by_resolution": rows}, indent=1))
