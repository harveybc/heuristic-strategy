#!/usr/bin/env python3
"""TRAIN-only profile of the per-agent hindsight signals (labels only, no model): how dense each agent's label is
for a grid of declared thresholds, how it relates to the ZigZag anticipatory label, and the closer's balance.
Writes SIGNAL_DESIGN_PROFILE.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import oracle as OR  # noqa: E402
from app.short_horizon.expert_guided import signals as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract="consolidated")
    b = ds.bars
    v = b.index.as_unit("ns").asi8
    o, h, l, c = (b[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    n = len(c)
    seg = ds.cal.assign(b.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    flat = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[i])], side="right")) - 1 if seg[i] >= 0 else n - 1 for i in range(n)])
    sig, hz = S.hindsight_signals(o, h, l, c, ds.sigma, ds.pivots, flat)
    H = np.maximum.reduce([hz["open"][1], hz["open"][-1], hz["close"]])      # the latest horizon a row may need
    t0, t1 = DG._utc_ns(ds.meta["periods"]["TRAIN"][0]), DG._utc_ns(ds.meta["periods"]["TRAIN"][1])
    end_ns = np.where(H >= 0, v[np.clip(H, 0, n - 1)] + DG.H_NS, np.iinfo(np.int64).max)
    tr = (v >= t0) & (v < t1) & (end_ns <= t1)          # TRAIN rows whose horizon is final inside TRAIN
    y = ds.y_hind
    out = {"train_rows_with_final_horizon": int(tr.sum()), "grid": [], "closer": [], "percentiles": {}}
    q = lambda x: {p: round(float(np.nanpercentile(x, p)), 3) for p in (5, 25, 50, 75, 95)}  # noqa: E731
    for k in ("opportunity", "adverse_to_peak", "bars_to_peak", "hold_gain", "hold_adverse", "next_is_favourable"):
        out["percentiles"][k] = {"long": q(sig[k][1][tr]), "short": q(sig[k][-1][tr])}
    for theta in (0.25, 0.5, 0.75, 1.0):
        for slm in (1.0, 2.0, 3.0):
            p = S.SignalParams(theta_open=theta, sl_max=slm)
            ol, osh = S.opener_labels(sig, 1, p)[tr], S.opener_labels(sig, -1, p)[tr]
            nop = S.nop_labels(sig, p)[tr]
            yl, ys = (y[tr] == OR.LONG_CLASS), (y[tr] == OR.SHORT_CLASS)
            out["grid"].append({"theta_open": theta, "sl_max": slm,
                                "share_open_long": round(float(np.nanmean(ol)), 4), "share_open_short": round(float(np.nanmean(osh)), 4),
                                "share_nop": round(float(np.nanmean(nop)), 4), "share_both_sides": round(float(np.nanmean((ol == 1) & (osh == 1))), 4),
                                "zigzag_long_share": round(float(yl.mean()), 4),
                                "p_label_long_given_zigzag_long": round(float(np.nanmean(ol[yl])), 3) if yl.any() else None,
                                "p_zigzag_long_given_label_long": round(float(yl[ol == 1].mean()), 3) if (ol == 1).any() else None})
    for theta_c in (0.05, 0.1, 0.25, 0.5):
        p = S.SignalParams(theta_close=theta_c)
        row = {"theta_close": theta_c}
        for sld in (1.0, 2.0):
            cl = S.closer_labels(sig, 1, np.full(n, sld), p)[tr]
            cs = S.closer_labels(sig, -1, np.full(n, sld), p)[tr]
            row[f"share_close_long_sl{sld:g}"] = round(float(np.nanmean(cl)), 4)
            row[f"share_close_short_sl{sld:g}"] = round(float(np.nanmean(cs)), 4)
        out["closer"].append(row)
    # at the ZigZag's own pivot bars, the closer of the right side must say CLOSE (hold_gain ~ 0)
    pb = np.array([i for i, k, _ in ds.pivots if tr[i]], int)
    kinds = np.array([k for i, k, _ in ds.pivots if tr[i]], int)
    from app.short_horizon import zigzag as Z
    hg_long_at_peak = sig["hold_gain"][1][pb[kinds == Z.PEAK]]
    hg_short_at_valley = sig["hold_gain"][-1][pb[kinds == Z.VALLEY]]
    out["at_optimal_pivots"] = {"long_hold_gain_at_peaks_p95": round(float(np.nanpercentile(hg_long_at_peak, 95)), 3),
                                "short_hold_gain_at_valleys_p95": round(float(np.nanpercentile(hg_short_at_valley, 95)), 3)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
