#!/usr/bin/env python3
"""One identified CONSUMER RECIPE shared by both lanes (Musashi's final T3 handoff, 2026-10-23, item 3): the exact
representation both the expert-guided line and the SAC-without-expert line must consume, derived from the sealed
artifacts of the selected combination unit C and never retyped by hand.

  columns        the strict v4 list in its order, with the per-column transform that the fixed TRAIN scaler applied
                 (tensor order, `@diff` / `@log` / `@asinh` suffixes as recorded in SCALER.json);
  scaler         fixed TRAIN-only mean / std per feature (fit period, rows, ddof, constant-column rule, NaN policy,
                 stationarity decisions) copied from the scaler artifact beside C's checkpoints;
  representation groups and extractor kind per group from the combination groups file (T3 v2 decision), model widths
                 and the T2 regularization;
  business       replay execution contract, weekly activity rule, forced flat, SL + TP on every order;
  split          consolidated contract (fit, inner block, 2024 read once, TEST closed);
  identities     science commit, store / list / groups / decision / combination / result digests.
Labels are the inherited ZigZag demonstrations, training-only; hindsight signals never enter observations. The recipe
is a description for consumers, not a release: `scientific_release` stays false."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUSINESS = {
    "execution_contract": "replay (Musashi's): max 1 entry per day, leverage 1:50, rel_volume 0.10, capital 10000",
    "activity_rule": "orders_per_week >= 2.0 as the PERIOD AVERAGE (not per week); weeks_below_min_orders reported; no forced trading",
    "orders": "every order carries broker-side SL and TP computed at entry; early close is additional; integer units = f*balance*rel_volume*leverage/price",
    "forced_flat": "weekly forced flat from the admitted closure calendar; SL and TP in the same H1 bar -> SL first (conservative)",
    "agent_state": "none: no agent feedback slots in the observation (constant zeros); the env enforces every rule with its real state",
    "scaling": "fixed TRAIN-only standardization; rolling z-score forbidden; no clipping of observations",
}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_emit(a):
    spec = json.loads(Path(a.spec).read_text())
    sc = json.loads(Path(a.scaler).read_text())
    groups = json.loads(Path(a.groups).read_text())
    cols = json.loads(Path(a.list).read_text())["features"]
    man = json.loads(Path(a.manifest).read_text())
    dec = json.loads(Path(a.decision).read_text())
    comb = json.loads(Path(a.combination).read_text())
    e3m = json.loads(Path(a.c_result).read_text())
    base = lambda c: c.split("@")[0]  # noqa: E731
    tensor = list(sc["features"])
    if [base(c) for c in tensor] != [base(c) for c in cols]:
        raise SystemExit("RECIPE_SCALER_COLUMNS_DIFFER_FROM_LIST")
    if sorted(c for g in groups["groups"].values() for c in g) != sorted(cols):
        raise SystemExit("RECIPE_GROUPS_DO_NOT_PARTITION_THE_LIST")
    if comb.get("frozen_rule_decision") != "COMBINATION_SELECTED":
        raise SystemExit(f"RECIPE_COMBINATION_NOT_SELECTED:{comb.get('frozen_rule_decision')}")
    if comb.get("combined_result_sha256") != sha(a.c_result):
        raise SystemExit("RECIPE_C_RESULT_DIGEST_DIFFERS_FROM_COMBINATION_DECISION")
    kinds = groups.get("kinds", {})
    names = e3m["arch"].get("group_names") or []
    if names and names != list(groups["groups"]):
        raise SystemExit("RECIPE_C_GROUPS_DIFFER_FROM_GROUPS_FILE")
    reg = spec["regularization"]
    m = spec["model"]
    recipe = {
        "schema": "expert_guided_consumer_recipe.v1", "scientific_release": False,
        "scope": "shared representation for both lanes (expert-guided learning; SAC without expert input); E3-M selection is a proxy, not evidence that SAC benefits",
        "store": {"file": Path(spec["store"]["path"]).name, "sha256": spec["store"]["sha256"]},
        "list": {"file": Path(a.list).name, "sha256": sha(a.list), "n": len(cols), "columns": cols},
        "tensor": {"order": tensor, "n_features": len(tensor), "transforms": {c: (c.split("@", 1)[1] if "@" in c else "none") for c in tensor},
                   "window": sc["window"], "stride_note": "window slots are observed rows (session gaps never compressed)"},
        "scaler": {"policy": sc["policy"], "fit_period": sc.get("fit_period"), "fit_rows": sc.get("fit_rows"), "ddof": sc.get("ddof"),
                   "rule": sc.get("rule"), "nan_policy": sc.get("nan_policy"), "transform": sc.get("transform"), "stationarity": sc.get("stationarity"),
                   "mean": sc["mean"], "std": sc["std"], "artifact_sha256": sha(a.scaler)},
        "representation": {"groups": groups["groups"], "units": groups.get("units", {}), "kinds": {g: kinds.get(g, "conv") for g in groups["groups"]},
                           "changed_from_control": {g: k for g, k in kinds.items() if k != "conv"} | {g: "per_feature" for g in
                                                                                                      {x["group"] for x in dec["groups"] if x["chosen"] == "per_feature"}},
                           "groups_file_sha256": sha(a.groups), "model": {"units": m["units"], "depth": m["depth"], "head": m["head"], "core": m.get("core")},
                           "regularization": {"dropout": reg["dropout"], "wd": reg["wd"], "source": reg.get("source")},
                           "params_per_module": e3m["params"]["FULL"]},
        "business": BUSINESS,
        "split": e3m["split_contract"] | {"periods": e3m["meta"]["periods"], "support_hash": e3m["meta"]["support_hash"]},
        "labels": {"mode": "zigzag (inherited hindsight demonstrations, training-only)", "note": "opportunity/both labels are a separate experiment; hindsight never enters observations"},
        "identities": {"science_commit": a.science_commit, "t3_manifest": man["identity"], "t3_manifest_sha256": sha(a.manifest),
                       "t3_decision_sha256": sha(a.decision), "combination_decision_sha256": sha(a.combination),
                       "c_result_sha256": sha(a.c_result), "c_producer_config_sha256": comb.get("producer_config_sha256"),
                       "t2_regularization_decision": reg.get("source"), "selection_v4": spec.get("selection", {}).get("identity"),
                       "selection_ledger_sha256": spec.get("selection", {}).get("ledger_sha256")},
        "custody": "reports and physical copies verified by hash; OLAP ingestion / read-back NOT certified by this recipe",
    }
    recipe["identity"] = "recipe_" + hashlib.sha256(json.dumps({k: recipe[k] for k in ("store", "list", "tensor", "scaler", "representation", "split")},
                                                                sort_keys=True).encode()).hexdigest()[:16]
    Path(a.out).write_text(json.dumps(recipe, indent=1) + "\n")
    print(json.dumps({"identity": recipe["identity"], "columns": len(cols), "changed": recipe["representation"]["changed_from_control"], "out": a.out}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    for k in ("spec", "scaler", "groups", "list", "manifest", "decision", "combination", "c-result", "out", "science-commit"):
        e.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    sys.exit({"emit": cmd_emit}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
