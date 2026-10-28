#!/usr/bin/env python3
"""Label-only profile of candidate expert signals on TRAIN (owner 2026-10-22). No model, no validation period.

1. ZigZag demonstrators SHORTER than the registered optimum (Depth 3 / Backstep 0; Deviation is inert in the
   replicated MetaTrader ZigZag): depth 3, 2, 1 -> pivots, legs per week, median / mean leg size in pips and in sigma,
   share of legs smaller than the round-trip cost.
2. A reversal-threshold ZigZag (classic definition: a pivot is fixed when price reverses by >= threshold from the
   running extreme) at 40.7 / 30 / 20 / 10 pips, for reference — the threshold that the inert Deviation never applied.
3. Hindsight opportunity at every flat origin (entry at the next open, side d): the best favourable excursion in
   sigma before the optimal expert's next pivot (capped by the forced flat and 72 bars), net of the declared cost;
   and the closer's hold gain: for a position of direction d open at bar b, the best further gain in sigma from the
   close of b until the same horizon. Percentiles fix the bounded regression heads' ranges.
"""
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

COST_SIGMA = 0.06
HORIZON = 72


def legs(piv, close_px):
    lv = np.array([v for _, _, v in piv])
    bars = np.array([i for i, _, _ in piv])
    return np.abs(np.diff(lv)), np.diff(bars)


def reversal_zigzag(h, l, thr):
    """Classic reversal ZigZag on highs/lows: alternate extremes, a new leg starts when price moves >= thr against
    the running extreme. Hindsight (labels only)."""
    piv, trend, ext_i, ext_v = [], 0, 0, h[0]
    lo_i, lo_v, hi_i, hi_v = 0, l[0], 0, h[0]
    for i in range(1, len(h)):
        if trend >= 0:
            if h[i] > hi_v:
                hi_i, hi_v = i, h[i]
            if trend == 0 and l[i] < lo_v:
                lo_i, lo_v = i, l[i]
            if hi_v - l[i] >= thr and (trend == 1 or hi_i > lo_i):
                piv.append((hi_i, Z.PEAK, hi_v)); trend, lo_i, lo_v = -1, i, l[i]
                continue
            if trend == 0 and h[i] - lo_v >= thr and lo_i > hi_i:
                piv.append((lo_i, Z.VALLEY, lo_v)); trend, hi_i, hi_v = 1, i, h[i]
        else:
            if l[i] < lo_v:
                lo_i, lo_v = i, l[i]
            if h[i] - lo_v >= thr:
                piv.append((lo_i, Z.VALLEY, lo_v)); trend, hi_i, hi_v = 1, i, h[i]
    return piv


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract="consolidated")
    b = ds.bars
    v = b.index.as_unit("ns").asi8
    t0, t1 = DG._utc_ns(ds.meta["periods"]["TRAIN"][0]), DG._utc_ns(ds.meta["periods"]["TRAIN"][1])
    tr = (v >= t0) & (v < t1)
    o, h, l, c = (b[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    sig = np.asarray(ds.sigma, float)
    weeks = tr.sum() / (24 * 5)
    out = {"period": ds.meta["periods"]["TRAIN"], "cost_sigma": COST_SIGMA, "zigzag_depth": [], "zigzag_reversal": []}
    hT, lT = h[tr], l[tr]
    for dep in (3, 2, 1):
        piv = Z.hindsight_pivots(hT, lT, Z.Params(depth=dep, deviation_pips=DG.EXPERT.deviation_pips, backstep=0))
        sz, dur = legs(piv, c)
        out["zigzag_depth"].append({"depth": dep, "pivots": len(piv), "legs_per_week": round(len(sz) / weeks, 2),
                                    "leg_pips_median": round(float(np.median(sz)) / Z.PIP, 1), "leg_pips_mean": round(float(np.mean(sz)) / Z.PIP, 1),
                                    "leg_bars_median": float(np.median(dur)), "share_legs_below_5pips": round(float((sz < 5 * Z.PIP).mean()), 3)})
    for thr in (40.7, 30.0, 20.0, 10.0):
        piv = reversal_zigzag(hT, lT, thr * Z.PIP)
        sz, dur = legs(piv, c)
        out["zigzag_reversal"].append({"threshold_pips": thr, "pivots": len(piv), "legs_per_week": round(len(sz) / weeks, 2),
                                       "leg_pips_median": round(float(np.median(sz)) / Z.PIP, 1), "leg_bars_median": float(np.median(dur))})
    # opportunity / hold gain relative to the optimal expert's next pivot (registered expert, full series pivots)
    piv_bars = np.array([i for i, _, _ in ds.pivots])
    n = len(c)
    nxt = piv_bars[np.minimum(np.searchsorted(piv_bars, np.arange(n), side="right"), len(piv_bars) - 1)]
    opp = {1: [], -1: []}
    hold = {1: [], -1: []}
    idx = np.flatnonzero(tr)[::3]                      # every third TRAIN bar (profile only)
    for bb in idx:
        if bb + 2 >= n or not np.isfinite(sig[bb + 1]) or sig[bb + 1] <= 0:
            continue
        end = int(min(nxt[bb], bb + HORIZON, n - 1))
        if end <= bb + 1:
            continue
        s = sig[bb + 1]
        hh, ll = h[bb + 1:end + 1], l[bb + 1:end + 1]
        e = o[bb + 1]
        opp[1].append((hh.max() - e) / s - COST_SIGMA)
        opp[-1].append((e - ll.min()) / s - COST_SIGMA)
        hold[1].append((hh.max() - c[bb]) / s)
        hold[-1].append((c[bb] - ll.min()) / s)
    q = lambda x: {p: round(float(np.percentile(x, p)), 3) for p in (1, 5, 25, 50, 75, 95, 99)}  # noqa: E731
    out["opportunity_sigma"] = {"long": q(opp[1]), "short": q(opp[-1]), "rows": len(opp[1])}
    out["hold_gain_sigma"] = {"long": q(hold[1]), "short": q(hold[-1])}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
