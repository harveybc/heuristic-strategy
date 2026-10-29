"""Ruling B instrument check: for every planted burst window of the W17-failing records, the declared burst amplitude
(burst_mult) vs the realized-vol ratio the generator delivers (window realized vol with the burst / the SAME path with
burst_mult = 1: identical random streams, only the multiplier differs), and what the causal gate sees (rv over
burst_window_bars scaled to 24 h / causal sigma) at the bars >= t0 + lag. Usage: burst_check.py QUEUE RES_DIR RECS OUT"""
import json, re, sys, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from tools import strategy_hierarchy as H, strategy_output_simulation as S, strategy_behaviour_matrix as BM
from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import strategy_zz as ZZ

R = "/var/tmp/behaviour-matrix-20261027/data"
cells = {c["id"]: c for c in json.load(open(sys.argv[1]))}
res = Path(sys.argv[2]); recs = [l.strip() for l in open(sys.argv[3]) if l.strip()]
cx, SCFG = S.load_fold("wf2022", R + "/admitted", R + "/sel15")
ds = cx.ds
base_sd = float(np.nanstd(np.diff(np.log(ds.bars["CLOSE"].to_numpy(float)))))
blk = H.period_bars(ds, BM.BLOCK)
lag, w = BM.BURST_DETECTION_LAG, int(ZZ.PARAMS["burst_window_bars"])
out = []
for rec in recs:
    cid, s = rec.rsplit("_s", 1); seed = int(s)
    c = cells[cid]; ms = c["ms"]
    base_id = re.sub(r"^(M_|S_)+", "", cid)
    gseed = (zlib.crc32(base_id.encode()) & 0xFFFF) + 7919 * seed
    kw = dict(mirror=bool(c.get("mirror")), mirror_tick=ZZ.LEVEL_TICK)
    bars, meta = H.synthetic_path_ms(ds.bars.index, base_sd, ms, gseed, blk, **kw)
    ms0 = json.loads(json.dumps(ms)); ms0["short"]["burst_mult"] = 1.0
    bars0, meta0 = H.synthetic_path_ms(ds.bars.index, base_sd, ms0, gseed, blk, **kw)
    assert meta0["burst_windows"] == meta["burst_windows"]
    cl, cl0 = bars["CLOSE"].to_numpy(float), bars0["CLOSE"].to_numpy(float)
    sig = Z.causal_sigma(bars)
    d2 = np.r_[0.0, np.diff(cl) ** 2]; cs = np.cumsum(d2)
    rv = np.full(len(cl), np.nan); rv[w:] = np.sqrt((cs[w:] - cs[:-w]) * 24.0 / w)
    ratio = rv / sig
    inw = np.zeros(len(cl), bool)
    for t0, t1 in meta["burst_windows"]:
        inw[t0:t1] = True
    calm = ratio[blk[0]:blk[1]][~inw[blk[0]:blk[1]]]
    rec_ = json.loads((res / f"CELL_{cid}_s{seed}.json").read_text())
    w17 = rec_["results"]["W17_no_open_in_burst"]
    tr = {int(t["entry_bar_abs"]): t for t in rec_["trades"]}
    off = int(rec_["trades"][0]["entry_bar_abs"]) - int(w17["inside"][0]["decision_bar"]) if False else None
    wins = []
    for t0, t1 in meta["burst_windows"]:
        a_ = np.sqrt(np.sum(np.diff(cl[t0 - 1:t1]) ** 2)); b_ = np.sqrt(np.sum(np.diff(cl0[t0 - 1:t1]) ** 2))
        g = ratio[t0 + lag:t1]
        wins.append({"window": [t0, t1], "declared_mult": float(ms["short"]["burst_mult"]), "delivered_rv_ratio": float(a_ / b_),
                     "gate_ratio_after_lag_max": float(np.nanmax(g)) if len(g) else None, "gate_ratio_after_lag_median": float(np.nanmedian(g)) if len(g) else None,
                     "bars_after_lag": int(len(g)), "fires_at_2.0": int(np.sum(g > 2.0)), "fires_at_1.5": int(np.sum(g > 1.5))})
    opens = []
    for x in w17["inside"]:
        db = int(x["decision_bar"])
        opens.append({"decision_bar": db, "window": x["window"], "rv_over_sigma_record": x["rv_over_sigma"], "rv_over_sigma_rebuilt": float(ratio[db])})
    out.append({"record": rec, "declared_mult": float(ms["short"]["burst_mult"]), "burst_bars": ms["short"].get("burst_bars"), "vol": ms.get("vol"),
                "calm_gate_ratio_p50_p90": [float(np.nanpercentile(calm, 50)), float(np.nanpercentile(calm, 90))], "calm_share_above_1.5": float(np.nanmean(calm > 1.5)),
                "calm_share_above_2.0": float(np.nanmean(calm > 2.0)), "windows": wins, "opens_inside": opens})
    print(rec, out[-1]["declared_mult"], [round(x["delivered_rv_ratio"], 2) for x in wins], [round(x["gate_ratio_after_lag_max"] or 0, 2) for x in wins], flush=True)
Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
