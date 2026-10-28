"""The closer head's error autocorrelation phi (CLOSER_NOISE_PHI), measured on real hourly predictor residuals.

For every committed hourly prediction CSV of predictor (columns Target_Hk / Prediction_Hk, one row per hourly bar),
the lag-1 autocorrelation of the error e_t = Prediction_Hk - Target_Hk per horizon k; phi = the median over files at
the longest horizon present (H6). Rows dated 2025-01-01 or later are never read (TEST).

  python tools/measure_head_error_phi.py --root <predictor checkout> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GLOBS = ("examples/results/phase_1/*_1h_prediction.csv", "examples/results/phase_3_1/*_1h_prediction.csv")
TEST_START = pd.Timestamp("2025-01-01")


def lag1(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    if len(e) < 30:
        return float("nan")
    e = e - e.mean()
    return float(np.dot(e[1:], e[:-1]) / np.dot(e, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(p for g in GLOBS for p in root.glob(g))
    per = {}
    for p in files:
        df = pd.read_csv(p)
        if "DATE_TIME" in df:
            df = df[pd.to_datetime(df["DATE_TIME"]) < TEST_START]
        hs = sorted(int(c.split("_H")[1]) for c in df.columns if c.startswith("Prediction_H"))
        per[str(p.relative_to(root))] = {f"H{h}": lag1((df[f"Prediction_H{h}"] - df[f"Target_H{h}"]).to_numpy(float)) for h in hs} | {"rows": int(len(df))}
    hz = sorted({k for v in per.values() for k in v if k.startswith("H")}, key=lambda s: int(s[1:]))
    med = {h: float(np.nanmedian([v[h] for v in per.values() if h in v])) for h in hz}
    doc = {"what": "lag-1 autocorrelation of the hourly prediction error (Prediction_Hk - Target_Hk) per committed predictor CSV",
           "source": "predictor " + ", ".join(GLOBS), "n_files": len(per), "median_by_horizon": med, "phi": med[hz[-1]], "phi_horizon": hz[-1], "files": per}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("n_files", "median_by_horizon", "phi", "phi_horizon")}))


if __name__ == "__main__":
    main()
