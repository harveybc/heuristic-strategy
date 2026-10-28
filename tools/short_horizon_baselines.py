#!/usr/bin/env python3
"""Cheap causal baseline predictors scored through the sealed-prediction adapter (DEVELOPMENT).

For one block size per invocation (bounded CPU): build the block table, fit each causal
predictor (refits expanding, weekly or daily, using only blocks completed before the refit),
write one prediction file per (predictor, split), seal it, load it back through the adapter
(which refuses leaks) and score it with the realistic execution and the frequency rule.
TRAIN 2020-2023 and VALIDATION 2024 only; no GPU, no deep model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.short_horizon import adapter as AD
from app.short_horizon import blocks as BL
from app.short_horizon import predictors as PR
from tools.short_horizon_oracle import SPLITS, load_inputs, utc

EXCURSIONS = ("PERSIST", "MEAN20", "EWMA", "HAR")
ORDERS = ("BASE", "LOGIT")
CADENCES = ("weekly", "daily")
HL_GRID = (2, 3, 5, 8, 13, 21, 34, 55)


def excursion_preds(t, B, name, cadence, train):
    if name == "PERSIST":
        return t["U"].shift(1).to_numpy(), t["D"].shift(1).to_numpy(), None
    if name == "MEAN20":
        return PR.mean_window_predict(t["U"], 20), PR.mean_window_predict(t["D"], 20), None
    if name == "EWMA":
        hl = PR.select_halflife(t, train, HL_GRID)
        return PR.ewma_predict(t["U"].to_numpy(), hl), PR.ewma_predict(t["D"].to_numpy(), hl), hl
    if name == "HAR":
        return PR.har_predict(t, B, "U", cadence), PR.har_predict(t, B, "D", cadence), None
    raise ValueError(name)


def order_preds(t, B, name, cadence):
    return PR.base_rate_predict(t, cadence) if name == "BASE" else PR.logit_predict(t, B, cadence)


def run(args):
    bars, cal, digest = load_inputs(args.data)
    B = args.block_hours
    blocks = BL.build_blocks(bars, cal, B)
    m = AD.Market(bars, cal, blocks, B)
    t = PR.block_table(blocks, cal, B)
    train = (t["start"] < utc("2024-01-01")).to_numpy()   # TRAIN (and its December 2019 warm-up) only
    out = Path(args.output) / f"B{B}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    specs = [("PERSISTENCE_PATH", None, None, None)]
    specs += [(f"{e}+{o}", e, o, c) for e in EXCURSIONS for o in ORDERS for c in CADENCES]
    rows = []
    for name, ex, od, cad in specs:
        tag = name if cad is None else f"{name}@{cad}"
        if ex is not None:
            u, d, hl = excursion_preds(t, B, ex, cad, train)
            p = order_preds(t, B, od, cad)
        for split_name, (a, b) in SPLITS.items():
            sel = ((t["start"] >= utc(a)) & (t["start"] < utc(b))).to_numpy()
            frame = pd.DataFrame({"block_start": [s.isoformat() for s in t["start"][sel]],
                                  "made_at": [s.isoformat() for s in t["start"][sel]]})
            if ex is None:
                for h in range(1, B + 1):
                    frame[f"p_h{h}"] = t["ref"][sel].to_numpy()
                keep = np.ones(sel.sum(), bool)
            else:
                frame["u_hat"], frame["d_hat"], frame["p_low_first"] = u[sel], d[sel], p[sel]
                keep = np.isfinite(frame[["u_hat", "d_hat", "p_low_first"]]).all(axis=1).to_numpy()
            frame = frame[keep]
            fdir = out / tag.replace("+", "_").replace("@", "_") / split_name
            fpath = fdir / "predictions.csv"
            fdir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(fpath, index=False, float_format="%.10g")
            sealed = AD.seal(fpath, fdir)
            AD.verify_seal(fpath, sealed)
            ps = AD.load_predictions(fpath, m, (utc(a), utc(b)))
            AD.verify_seal(fpath, sealed)
            for st in ("RF-M", "RF-L"):
                for cost in (1.0, 2.0):
                    r = AD.score(ps, m, (utc(a), utc(b)), st, cost)
                    r.update({"predictor": name, "excursion": ex or "path", "order": od or "path",
                              "cadence": cad or "n/a", "ewma_halflife": hl if ex == "EWMA" else None,
                              "block_hours": B, "split": split_name, "rows_predicted": int(len(frame))})
                    rows.append(r)
    # order-model reference: log-loss of the causal base rate on the same rows
    df = pd.DataFrame(rows)
    cpu = time.process_time() - t0
    df.to_csv(out / "results.csv", index=False)
    (out / "receipt.json").write_text(json.dumps({"block_hours": B, "rows": len(df), "cpu_seconds": cpu,
                                                  "data_sha256": digest}, indent=1) + "\n")
    print(json.dumps({"block_hours": B, "rows": len(df), "cpu_seconds": round(cpu, 1)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--block-hours", type=int, choices=(6, 24), required=True)
    ap.add_argument("--output", type=Path, required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
