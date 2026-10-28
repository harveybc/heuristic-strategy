#!/usr/bin/env python3
"""TRAIN-only profile of the per-agent hindsight signals (labels only, no model) on the BUSINESS supports (auditor
2026-10-22 §4): openers / NOP on the eligible flat origins (entry gate open, window and sigma available, label final
inside TRAIN); closers on the observable bars (a position may be managed at any observable bar; the entry gate is
not imposed on the closer). Exclusions and NaN per target are counted. Writes SIGNAL_DESIGN_PROFILE.json."""
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
    t0, t1 = DG._utc_ns(ds.meta["periods"]["TRAIN"][0]), DG._utc_ns(ds.meta["periods"]["TRAIN"][1])
    fin = {}
    for d in (1, -1):                                                      # horizon final inside TRAIN, per side
        hb = hz[d]["bar"]
        end_ns = np.where(hb >= 0, v[np.clip(hb, 0, n - 1)] + DG.H_NS, np.iinfo(np.int64).max)
        fin[d] = (v >= t0) & (v < t1) & (end_ns <= t1)
    in_train = (v >= t0) & (v < t1)
    rob = ds.row_of_bar()
    row_ok = np.zeros(n, bool); row_ok[rob >= 0] = True
    elig_rows = np.zeros(len(ds.eligible), bool); elig_rows[ds.rows("TRAIN")] = True
    gate = np.zeros(n, bool); gate[rob >= 0] = elig_rows[rob[rob >= 0]]     # eligible flat origins (entry gate open)
    obs_rows = ds.observable()
    obs = np.zeros(n, bool); obs[rob >= 0] = obs_rows[rob[rob >= 0]]        # observable bars (closer support)
    tr = gate & fin[1] & fin[-1]                                            # opener rows: both sides final
    tr_close = {d: obs & in_train & fin[d] for d in (1, -1)}
    y = ds.y_hind
    out = {"supports": {"train_bars": int(in_train.sum()), "eligible_flat_origins": int((gate & in_train).sum()),
                        "opener_rows_both_horizons_final": int(tr.sum()),
                        "excluded_opener_horizon_not_final": int((gate & in_train & ~(fin[1] & fin[-1])).sum()),
                        "closer_rows_long": int(tr_close[1].sum()), "closer_rows_short": int(tr_close[-1].sum()),
                        "truncated_share_long": round(float(np.nanmean(sig["truncated"][1][tr])), 4),
                        "truncated_share_short": round(float(np.nanmean(sig["truncated"][-1][tr])), 4),
                        "nan_per_target_openers": {k: int(np.isnan(sig[k][1][tr]).sum()) for k in ("opportunity", "adverse_to_peak")}},
           "grid": [], "closer": [], "percentiles": {}}
    q = lambda x: {p: round(float(np.nanpercentile(x, p)), 3) for p in (5, 25, 50, 75, 95)}  # noqa: E731
    for k in ("opportunity", "adverse_to_peak", "bars_to_peak"):
        out["percentiles"][k] = {"long": q(sig[k][1][tr]), "short": q(sig[k][-1][tr])}
    for k in ("hold_gain", "hold_adverse"):
        out["percentiles"][k] = {"long": q(sig[k][1][tr_close[1]]), "short": q(sig[k][-1][tr_close[-1]])}
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
    for rule in ("hold_through", "optimum"):
        for theta_c in (0.05, 0.1, 0.25, 0.5):
            p = S.SignalParams(theta_close=theta_c)
            row = {"rule": rule, "theta_close": theta_c}
            for sld in (1.0, 2.0):
                cl = S.closer_labels(sig, 1, np.full(n, sld), p, rule)[tr_close[1]]
                cs = S.closer_labels(sig, -1, np.full(n, sld), p, rule)[tr_close[-1]]
                row[f"share_close_long_sl{sld:g}"] = round(float(np.nanmean(cl)), 4)
                row[f"share_close_short_sl{sld:g}"] = round(float(np.nanmean(cs)), 4)
                row[f"censored_long_sl{sld:g}"] = int(np.isnan(cl).sum())
            out["closer"].append(row)
    # at the ZigZag's own pivot bars, the closer of the right side must say CLOSE (hold_gain ~ 0)
    pb = np.array([i for i, k, _ in ds.pivots if in_train[i]], int)
    kinds = np.array([k for i, k, _ in ds.pivots if in_train[i]], int)
    from app.short_horizon import zigzag as Z
    hg_long_at_peak = sig["hold_gain"][1][pb[kinds == Z.PEAK]]
    hg_short_at_valley = sig["hold_gain"][-1][pb[kinds == Z.VALLEY]]
    out["at_optimal_pivots"] = {"note": "the horizon of a long at a peak is the NEXT peak: hold gain there is the rise after the dip; the closer label is decided by that gain and the dip against the order's SL",
                                "long_hold_gain_at_peaks_p50": round(float(np.nanpercentile(hg_long_at_peak, 50)), 3),
                                "long_hold_adverse_at_peaks_p50": round(float(np.nanpercentile(sig["hold_adverse"][1][pb[kinds == Z.PEAK]], 50)), 3),
                                "short_hold_gain_at_valleys_p50": round(float(np.nanpercentile(hg_short_at_valley, 50)), 3),
                                "share_close_long_at_peaks_sl2_theta0.1_hold_through": round(float(np.nanmean(S.closer_labels(sig, 1, np.full(n, 2.0), S.SignalParams(theta_close=0.1))[pb[kinds == Z.PEAK]])), 3),
                                "share_close_long_at_peaks_sl2_theta0.1_optimum": round(float(np.nanmean(S.closer_labels(sig, 1, np.full(n, 2.0), S.SignalParams(theta_close=0.1), "optimum")[pb[kinds == Z.PEAK]])), 3)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
