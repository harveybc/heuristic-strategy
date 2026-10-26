#!/usr/bin/env python3
"""Derived walk-forward recipe: the ACCEPTED consumer recipe with ONLY two things changed, declared and identity-covered:
the scaler constants (re-fitted on the fold's fit block, same NaN policy and std_zero_to_one rule as the loader) and the
split contract (``wf2022`` / ``wf2023``, app/short_horizon/expert_guided/data.py). Store, list, tensor (columns, upstream
stages, consumer operations, window), representation, regularization, business and labels are copied byte for byte.
Diagnostic identity: ``scientific_release`` stays False and ``derived_from`` records the base identity + digest and the
deviation list, so the launcher can accept it for the WF stage only.

Witness (written next to the recipe): the loader through the derived recipe scores every TRAIN-period (fit block) row
with per-column mean ~ 0 and std ~ 1 for the columns whose fit std is positive (the constants are the ones in the file);
the base recipe's constants would not.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402

IDENT_KEYS = ("store", "list", "tensor", "scaler", "representation", "regularization", "split", "business", "labels")
COPIED_KEYS = ("store", "list", "tensor", "representation", "regularization", "business", "labels")
DEVIATION = ["scaler.constants_refitted_on_fold_fit_block", "split.walk_forward_fold"]


def identity(rec: dict) -> str:
    return "recipe_" + hashlib.sha256(json.dumps({k: rec[k] for k in IDENT_KEYS}, sort_keys=True).encode()).hexdigest()[:16]


def fold_constants(store_parquet: Path, base: dict, fold: str) -> dict:
    """Per-column constants of the base recipe's tensor (after its consumer ops) over the fold's fit block; the loader's
    rules: rows timestamped at availability (store bar start + 1 h), nanmean / nanstd ddof 0, std == 0 -> 1."""
    sc = DG.split_contract(fold)
    cols = base["tensor"]["columns"]
    names = [c["store_column"] for c in cols]
    raw = pd.read_parquet(store_parquet, columns=["timestamp"] + [n for n in names if n != "timestamp"])
    ts = (pd.to_datetime(raw["timestamp"], utc=True) + pd.Timedelta(hours=1)).dt.as_unit("ns").astype("int64").to_numpy()   # parquet may carry µs
    X = raw[names].to_numpy(float)
    fin = np.isfinite(X).all(1)
    first = int(np.argmax(fin))                       # the loader drops the leading rows before the first all-finite row
    X, ts = X[first:], ts[first:]
    X2 = DG.apply_consumer_ops(X, cols, names)        # DIFF / onehot_argmax / window_hours, as the loader applies them
    t0, t1 = DG._utc_ns(sc["scaler_fit"][0]), DG._utc_ns(sc["scaler_fit"][1])
    tr = (ts >= t0) & (ts < t1)
    if not tr.any():
        raise ValueError("SCALING_TRAIN_EMPTY")
    Xtr = X2[tr]
    n_fin = np.isfinite(Xtr).sum(0)
    if (n_fin == 0).any():
        raise ValueError(f"SCALING_FEATURE_ALL_NAN:{[names[j] for j in np.flatnonzero(n_fin == 0)]}")
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        raise ValueError("SCALING_NONFINITE_CONSTANT")
    std = np.where(sd > 0, sd, 1.0)
    return {"fit_period": list(sc["scaler_fit"]), "fit_rows": int(tr.sum()), "mean": mu.tolist(), "std": std.tolist(), "std_raw": sd.tolist(),
            "variance": (sd ** 2).tolist(), "rule": ["std_zero_to_one" if s <= 0 else "z" for s in sd],
            "fit_rows_per_feature": n_fin.astype(int).tolist(), "nan_share_train": (1.0 - n_fin / int(tr.sum())).tolist()}


def derive(base: dict, base_sha: str, store_parquet: Path, fold: str) -> dict:
    if fold not in DG.WALK_FORWARD_FOLDS:
        raise ValueError(f"WF_FOLD_UNKNOWN:{fold}")
    sc = DG.split_contract(fold)
    rec = {k: copy.deepcopy(base[k]) for k in COPIED_KEYS}
    rec["schema"] = base["schema"]
    rec["scientific_release"] = False
    rec["scope"] = (f"DIAGNOSTIC walk-forward fold {fold} derived from {base['identity']}: same representation, constants re-fitted on the "
                    f"fold's fit block {sc['scaler_fit'][0]}..{sc['scaler_fit'][1]}; validation year {sc['validation'][0]}..{sc['validation'][1]} "
                    "scored after freeze; never a release")
    scl = copy.deepcopy(base["scaler"])
    scl.update(fold_constants(store_parquet, base, fold))
    scl["transform"] = base["scaler"].get("transform", "fixed TRAIN z with the recipe's constants")
    scl.pop("artifact_sha256", None)
    scl["derived"] = f"constants re-fitted on the {fold} fit block with the loader's rules; stationarity decisions inherited from the base recipe"
    rec["scaler"] = scl
    rec["split"] = {"arm": fold, "train": list(sc["train"]), "validation": list(sc["validation"]), "selection_period": sc["selection"],
                    "read_once": sc["read_once"], "test_closed_from": DG.TEST_CLOSED_FROM,
                    "periods": {k: list(v) for k, v in sc["periods"].items()}, "status": dict(sc["status"]), "support_hash": None}
    rec["identity"] = identity(rec)
    rec["identity_covers"] = list(IDENT_KEYS)
    rec["derived_from"] = {"identity": base["identity"], "sha256": base_sha, "deviation": list(DEVIATION), "fold": fold}
    rec["provenance"] = {"base_provenance": copy.deepcopy(base.get("provenance")), "derived_by": "tools/expert_guided_wf_recipe.py"}
    rec["custody"] = base.get("custody")
    return rec


def witness(rec: dict, admitted: Path, store_parquet: Path, base: dict, tol: float = 0.05) -> dict:
    """Through the loader: fit-block rows of the tensor have |mean| < tol and |std - 1| < tol per column with positive fit std
    (last window slot = the row itself); the base constants fail the same check on the fold (reported, not asserted)."""
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    ds = DG.load_store(admitted, store_parquet, feats, scaling="fixed_train", contract=rec["split"]["arm"], recipe=rec)
    tr = ds.row_ok & (ds.period == "TRAIN")
    last = ds.windows[tr][:, -1, :].astype(float)
    pos = np.asarray(rec["scaler"]["std_raw"], float) > 0
    mu, sd = last.mean(0), last.std(0)
    ok = bool(np.all(np.abs(mu[pos]) < tol) and np.all(np.abs(sd[pos] - 1) < tol))
    bmean, bstd = np.asarray(base["scaler"]["mean"], float), np.asarray(base["scaler"]["std"], float)
    rmean, rstd = np.asarray(rec["scaler"]["mean"], float), np.asarray(rec["scaler"]["std"], float)
    # the base constants applied to the same rows: how far they sit from 0 / 1 on this fold
    base_last = (last * rstd + rmean - bmean) / bstd
    return {"fold": rec["split"]["arm"], "fit_rows_scored": int(tr.sum()), "columns": len(feats), "positive_std_columns": int(pos.sum()),
            "max_abs_mean": float(np.abs(mu[pos]).max()), "max_abs_std_minus_1": float(np.abs(sd[pos] - 1).max()), "tolerance": tol, "ok": ok,
            "base_constants_on_this_fold": {"max_abs_mean": float(np.abs(base_last.mean(0)[pos]).max()), "max_abs_std_minus_1": float(np.abs(base_last.std(0)[pos] - 1).max())},
            "rows_eligible": ds.meta["rows_eligible"], "support_hash": ds.meta["support_hash"],
            "columns_with_changed_constants": int(np.sum(~np.isclose(rmean, bmean) | ~np.isclose(rstd, bstd)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="the accepted CONSUMER_RECIPE.json")
    ap.add_argument("--store", required=True)
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--fold", required=True, choices=list(DG.WALK_FORWARD_FOLDS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-witness", action="store_true")
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text())
    rec = derive(base, DG.sha256(Path(a.base)), Path(a.store), a.fold)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"CONSUMER_RECIPE_{a.fold}.json"
    p.write_text(json.dumps(rec, indent=1))
    doc = {"recipe": p.name, "sha256": DG.sha256(p), "identity": rec["identity"], "derived_from": rec["derived_from"]}
    if not a.no_witness:
        doc["witness"] = witness(rec, Path(a.admitted), Path(a.store), base)
        (out / f"WF_RECIPE_{a.fold}_WITNESS.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in doc if k != "witness"} | ({"witness_ok": doc["witness"]["ok"], "support_hash": doc["witness"]["support_hash"]} if "witness" in doc else {})))


if __name__ == "__main__":
    main()
