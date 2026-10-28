#!/usr/bin/env python3
"""Derive revision 3 by retiring only px.er_1; never edit a source or fit a model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.expert_guided_consumer_recipe import need, sha

KEYS = ("store", "list", "tensor", "scaler", "representation", "regularization", "split", "business", "labels")
COLUMN = "px.er_1"
SCALER_VECTORS = ("features", "mean", "std", "std_raw", "variance", "rule", "fit_rows_per_feature", "nan_share_train")


def encoded(doc):
    return (json.dumps(doc, indent=1, allow_nan=False) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def identity(rec):
    return "recipe_" + sha_bytes(json.dumps({k: rec[k] for k in KEYS}, sort_keys=True).encode())[:16]


def derive(parent, parent_sha, observed):
    need(parent.get("schema") == "expert_guided_consumer_recipe.v2"
         and parent.get("identity") == identity(parent)
         and tuple(parent.get("identity_covers", [])) == KEYS, "PARENT_IDENTITY")
    need(len(parent_sha) == 64 and all(c in "0123456789abcdef" for c in parent_sha), "PARENT_SHA")
    cols = parent["list"]["columns"]
    need(cols.count(COLUMN) == 1 and parent["list"]["n"] == len(cols), "RETIRE_COLUMN")
    i, n = cols.index(COLUMN), len(cols)
    sc = parent["scaler"]
    need(all(len(sc[k]) == n for k in SCALER_VECTORS), "RETIRE_SCALER_LENGTH")
    need(sc["policy"] == "fixed_train" and sc["features"][i] == COLUMN
         and sc["mean"][i] == 1. and sc["std"][i] == 1.
         and sc["std_raw"][i] == 0. and sc["variance"][i] == 0.
         and parent["tensor"]["columns"][i]["consumer_ops"] == [], "RETIRE_ZERO_CHANNEL")
    values = np.asarray(observed, float)
    finite = np.isfinite(values)
    need(values.ndim == 1 and finite.any() and not np.isinf(values).any()
         and (values[finite] == 1.).all(), "RETIRE_NOT_CONSTANT")
    rec = copy.deepcopy(parent)
    keep = [j for j in range(n) if j != i]
    survivor = [cols[j] for j in keep]
    files = {"FEATURE_LIST.json": {"features": survivor}}
    rec["list"] = dict(file="FEATURE_LIST.json", sha256=sha_bytes(encoded(files["FEATURE_LIST.json"])),
                       n=n - 1, columns=survivor)
    rec["tensor"]["columns"] = [rec["tensor"]["columns"][j] for j in keep]
    rec["tensor"]["retirement"] = dict(column=COLUMN, parent_identity=parent["identity"],
        parent_recipe_sha256=parent_sha, reason="constant finite value 1; fixed-scaled value 0; no imputation",
        revision=3, fresh_support_required=True)
    for key in SCALER_VECTORS:
        rec["scaler"][key] = [sc[key][j] for j in keep]
    rec["scaler"]["n_features"] = n - 1
    stationarity = rec["scaler"]["stationarity"]
    stationarity.get("features", {}).pop(COLUMN, None)
    for key, value in list(stationarity.items()):
        if isinstance(value, list):
            stationarity[key] = [v for v in value if v != COLUMN]
    parent_scaler_sha = rec["scaler"].pop("artifact_sha256")
    rec["scaler"]["derivation"] = dict(parent_artifact_sha256=parent_scaler_sha,
        retained_constants="exact; no fit on expanded support", retired_columns=[COLUMN])
    files["SCALER.json"] = copy.deepcopy(rec["scaler"])
    rec["scaler"]["artifact_sha256"] = sha_bytes(encoded(files["SCALER.json"]))
    rep = rec["representation"]
    resolved = rep["resolved"]
    affected = [g for g, indices in enumerate(resolved["indices"]) if i in indices]
    need(len(affected) == 1, "RETIRE_GROUP_PARTITION")
    g = affected[0]
    need(len(resolved["indices"][g]) > 1 and resolved["kinds"][g] == "conv", "RETIRE_GROUP_KIND")
    delta = 3 * resolved["units"][g]
    resolved["indices"] = [[j - (j > i) for j in group if j != i] for group in resolved["indices"]]
    groups = rep["groups"]
    groups["name"] = "t3_C_without_px_er_1"
    groups["rule"] = "inherited C topology with constant channel removed; not reselected on expanded support"
    groups["groups"] = {k: [v for v in vs if v != COLUMN] for k, vs in groups["groups"].items()}
    need(all(groups["groups"].values()), "RETIRE_EMPTY_GROUP")
    files["GROUPS.json"] = copy.deepcopy(groups)
    rep["groups_file_sha256"] = sha_bytes(encoded(files["GROUPS.json"]))
    rep["params_per_module"] = {k: v - delta for k, v in rep["params_per_module"].items()}
    old_support = rec["split"].pop("support_hash", None)
    rec["split"]["support_status"] = "recompute from derived consumer; parent label support not inherited"
    rec["provenance"] = dict(parent_recipe_sha256=parent_sha, parent_identity=parent["identity"],
        parent_provenance=copy.deepcopy(parent["provenance"]), parent_support_hash=old_support,
        derivation="remove one constant scaled-zero channel; no retraining or scientific reselection",
        checkpoint_migration="not performed; SAC uses no expert weights",
        expanded_support_performance="unmeasured; no equivalence claim")
    rec["scientific_release"] = False
    rec["identity"] = identity(rec)
    rec["scope"] = "Revision 3: inherited shared representation minus px.er_1; new support, not new performance evidence"
    return rec, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "store", "manifest", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--parent-sha", required=True)
    args = parser.parse_args()
    need(sha(args.parent) == args.parent_sha, "PARENT_FILE_PIN")
    parent = json.loads(args.parent.read_text())
    need(sha(args.store) == parent["store"]["sha256"]
         and sha(args.manifest) == parent["store"]["manifest_sha256"], "RETIRE_INPUT_PINS")
    import pandas as pd
    frame = pd.read_parquet(args.store, columns=[COLUMN], filters=[
        ("timestamp", "<", pd.Timestamp(parent["split"]["test_closed_from"], tz="UTC"))])
    rec, files = derive(parent, args.parent_sha, frame[COLUMN].to_numpy())
    files["RECIPE.json"] = rec
    args.out.mkdir(parents=True, exist_ok=False)
    for name, doc in files.items():
        with (args.out / name).open("xb") as stream:
            stream.write(encoded(doc))
    ready = dict(schema="constant_recipe_derivation.v1", state="DERIVED_NOT_SCIENTIFICALLY_EVALUATED",
                 identity=rec["identity"], retired=[COLUMN], fits=0, imputations=0,
                 parent_sha256=args.parent_sha, files={name: sha(args.out / name) for name in files})
    with (args.out / "READY.json").open("xb") as stream:
        stream.write(encoded(ready))
    print(json.dumps(ready))


if __name__ == "__main__":
    main()
