#!/usr/bin/env python3
"""Source-to-tensor probe of the shared consumer recipe (Musashi's final joint audit A3, 2026-10-23).

Both lanes (expert-guided E3-M; SAC without expert input) must produce the SAME probe before either new policy
campaign. A probe binds, for one recipe identity:
  - the column order, the tensor order, the scaler constants and the resolved groups (digests),
  - the valid-row mask per period (which decision rows have a complete, finite window),
  - for a fixed set of probe rows (first / middle / last valid row of each period + 16 evenly spaced valid rows):
    the decision timestamp, the first and last window-slot timestamps and the exact float32 tensor values.
  emit     the reference probe through the E3-M loader (DG.load_store with the recipe: nothing measured or fitted)
  compare  --reference P --candidate Q: exact equality of every field; a permuted column, an altered scaler or wrong
           groups fail here, before any training
`probe_document(...)` is the shared definition: another consumer calls it on ITS tensor to produce the candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def h(obj) -> str:
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def probe_rows(valid: np.ndarray, period: np.ndarray, periods, n_spread: int = 16) -> np.ndarray:
    rows = []
    for p in periods:
        idx = np.flatnonzero(valid & (period == p))
        if len(idx):
            rows += [idx[0], idx[len(idx) // 2], idx[-1]]
    idx = np.flatnonzero(valid)
    rows += list(idx[np.linspace(0, len(idx) - 1, n_spread).astype(int)])
    return np.array(sorted(set(int(r) for r in rows)), int)


def probe_document(recipe: dict, row_ts_ns: np.ndarray, windows: np.ndarray, valid: np.ndarray, period: np.ndarray,
                   slot_ts_ns: np.ndarray, tensor_features: list, groups_resolved: dict, scaler_mean, scaler_std) -> dict:
    """The probe of one consumer's tensor. ``windows[row, slot, feature]`` float32; ``slot_ts_ns[row, slot]`` the
    timestamp of each slot's source row; ``valid`` rows with a complete finite window; ``period`` per row."""
    periods = [p for p in recipe["split"]["periods"]]
    pr = probe_rows(valid, period, periods)
    W = np.asarray(windows[pr], np.float32)
    return {"schema": "consumer_probe.v1", "recipe": recipe["identity"],
            "columns_sha256": h([c["store_column"] for c in recipe["tensor"]["columns"]]),
            "tensor_order_sha256": h(list(tensor_features)),
            "scaler_sha256": h({"mean": [float(x) for x in scaler_mean], "std": [float(x) for x in scaler_std]}),
            "groups_sha256": h(groups_resolved),
            "valid_mask_sha256": {p: h(np.flatnonzero(valid & (period == p)).astype(np.int64)) for p in periods},
            "valid_rows": {p: int((valid & (period == p)).sum()) for p in periods},
            "probe_rows_ts": [int(row_ts_ns[r]) for r in pr],
            "window_endpoints_ts": [[int(slot_ts_ns[r, 0]), int(slot_ts_ns[r, -1])] for r in pr],
            "tensor_sha256": h(W), "tensor_shape": list(W.shape), "tensor_values": W.round(7).tolist()}


def reference(recipe_path, admitted, store, verify=True):
    from app.short_horizon.expert_guided import data as DG
    from tools import expert_guided_clean_round as C
    rec = json.loads(Path(recipe_path).read_text())
    cols = [c["store_column"] for c in rec["tensor"]["columns"]]
    w = rec["tensor"]["window"]
    ds = DG.load_store(admitted, store, cols, scaling="fixed_train", verify=verify, contract=rec["split"]["arm"],
                       win=int(w["observations"]), stride=int(w["stride_rows"]), recipe=rec)
    tensor = list(ds.meta["input_contract"]["features"])
    idx, units, names, kinds = C.load_groups_doc(rec["representation"]["groups"], tensor)
    stride, win = int(w["stride_rows"]), int(w["observations"])
    n = len(ds.row_ts)
    i = np.arange(n)
    slot_idx = np.clip(i[:, None] - stride * np.arange(win - 1, -1, -1)[None, :], 0, n - 1)
    slot_ts = ds.row_ts[slot_idx]
    valid = ds.row_ok.copy()
    sc = ds.meta["scaling"]["features"]
    return rec, probe_document(rec, ds.row_ts, ds.windows, valid, ds.period, slot_ts, tensor,
                               {"names": names, "indices": idx, "units": units, "kinds": kinds},
                               [sc[f]["mean"] for f in tensor], [sc[f]["std"] for f in tensor])


def compare(ref: dict, cand: dict) -> list:
    errs = []
    for k in ("recipe", "columns_sha256", "tensor_order_sha256", "scaler_sha256", "groups_sha256", "valid_mask_sha256",
              "valid_rows", "probe_rows_ts", "window_endpoints_ts", "tensor_shape", "tensor_sha256"):
        if ref.get(k) != cand.get(k):
            errs.append(f"PROBE_MISMATCH:{k}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.add_argument("--recipe", required=True); e.add_argument("--admitted", required=True)
    e.add_argument("--store", required=True); e.add_argument("--out", required=True); e.add_argument("--no-verify", action="store_true")
    c = sub.add_parser("compare"); c.add_argument("--reference", required=True); c.add_argument("--candidate", required=True)
    a = ap.parse_args()
    if a.cmd == "emit":
        _, doc = reference(a.recipe, a.admitted, a.store, verify=not a.no_verify)
        Path(a.out).write_text(json.dumps(doc))
        print(json.dumps({k: doc[k] for k in ("recipe", "tensor_sha256", "valid_rows")}))
        return
    errs = compare(json.loads(Path(a.reference).read_text()), json.loads(Path(a.candidate).read_text()))
    print(json.dumps({"equal": not errs, "errors": errs}))
    sys.exit(0 if not errs else 2)


if __name__ == "__main__":
    main()
