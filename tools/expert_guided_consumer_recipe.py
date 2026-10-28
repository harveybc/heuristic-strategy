#!/usr/bin/env python3
"""Shared CONSUMER RECIPE (v2): the exact, loadable representation both lanes consume (Musashi's final joint audit
A2/A3/A4, 2026-10-23). v1 hashed whatever the caller passed; v2 DERIVES every field from the authenticated producer
chain of the selected combination unit C and refuses anything that does not match it.

  chain     C's execution receipt (RESULT.json: config identity, effective command, every sealed file hash) -> the
            producer config (identity = sha256 of its key-sorted JSON; it pins the science code) -> the sealed files
            in C's folder (E3M.json, SCALER.json, PROBE.json, the 12 checkpoints) -> the combination decision
            (COMBINATION_SELECTED, combined result digest, producer identity) and its replay (PASS, exact) -> the T3
            decision (its manifest identity) whose `combine` regenerates the groups file byte for byte.
  checks    scaler policy fixed_train, columns / tensor order / window equal to C's dataset, lengths, finite constants,
            std > 0; store, manifest and list digests equal to the spec and to C's sources; resolved groups (names,
            indices, kinds, units) and model widths equal to C's architecture; replay passed; labels = ZigZag by an
            authenticated producer.
  stages    per column: the UPSTREAM operation already present in the store (layer decision, learned producer and its
            fit scope) and the CONSUMER operations the loader applies (DIFF from the recorded stationarity rule, then
            the fixed TRAIN z) - never re-applied by another consumer.
  window    typed: slots, stride in OBSERVED rows (admitted market bars; closures occupy no slot), current completed
            bar included, elapsed-hours span measured on the store.
  business  the effective execution contracts actually used by E3-M (tools/expert_guided_e3m.effective_contracts),
            the admitted closure calendar and its offsets, execution timing; loadable, digest-covered.
The identity covers store, list, tensor (columns, stages, window), scaler, representation, regularization, split,
business and labels; provenance (receipt, decision, replay digests) is checked and recorded, not part of the identity.
`scientific_release` stays false."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPTS = [f"FULL_{m}_{s}.pt" for m in ("open_l", "open_s", "close_l", "close_s") for s in ("20261019", "20261020", "20261021")]


class RecipeError(SystemExit):
    pass


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonable(o):
    if dataclasses.is_dataclass(o):
        return jsonable(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    return o


def need(cond, code):
    if not cond:
        raise RecipeError(f"RECIPE_{code}")


def business_contract(exec_contract: str, min_orders: float, closures_sha: str, admitted_sha: str) -> dict:
    from tools.expert_guided_e3m import effective_contracts
    S, X = effective_contracts(exec_contract, min_orders)
    return {"exec_contract": exec_contract, "evaluation_and_policy": jsonable(S), "expert_demonstrations": jsonable(X),
            "calendar": {"id": "ADMITTED_CLOSURE_COMPLEMENT", "entry_delay_after_open_hours": 12, "force_flat_before_close_hours": 12,
                         "entry_cutoff_before_flat_hours": 0, "closures_sha256": closures_sha, "admitted_sha256": admitted_sha},
            "execution_timing": "decision after the completed bar b (availability = bar start + 1 h); fill at open(b + 1); SL and TP "
                                "broker-side on every order; SL and TP touched in the same bar -> SL first; forced flat has priority",
            "activity_rule": f"orders_per_week >= {min_orders} as the PERIOD AVERAGE; weeks_below_min_orders reported; no forced trading",
            "agent_state": "none: no agent feedback in the observation"}


def window_semantics(store_parquet: Path, win: int, stride: int) -> dict:
    import pandas as pd
    ts = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(store_parquet, columns=["timestamp"])["timestamp"], utc=True))
    span = (ts[stride * (win - 1):] - ts[: len(ts) - stride * (win - 1)]).total_seconds().to_numpy() / 3600.0
    return {"observations": int(win), "stride_rows": int(stride), "row_unit": "observed admitted market bar (H1); closures occupy no slot",
            "includes_current_completed_bar": True, "slot_offsets_rows": [-stride * k for k in range(win - 1, -1, -1)],
            "elapsed_hours_first_to_last_slot": {"min": float(span.min()), "median": float(np.median(span)), "max": float(span.max())},
            "note": "no interpolation over closures; not a clock-hour window"}


def upstream_stage(col: str, l1dec: dict, l2dec: dict, l2fam: dict, scope: dict | None) -> dict:
    base, _, suf = col.partition("@")
    dec = l1dec.get(base) or l2dec.get(base)
    fam = l2fam.get(base)
    rec = {"store_column": col, "decision": dec or "NONE", "applied_in_store": bool(suf), "suffix": suf or None}
    if fam in ("kalman", "gmm", "hmm", "fracdiff") or base == "act.p_hit":
        rec["learned_producer"] = fam or "activity"
        rec["fit_scope"] = scope
    return rec


def validate_replay(rep, result_sha):
    """Bind the exact module and rollout evidence, not just its PASS header."""
    need(rep.get("schema") == "t3_checkpoint_probe_replay.v1"
         and rep.get("state") == "PASS" and rep.get("count") == 1
         and rep.get("checkpoints") == 12 and rep.get("torch_cpu_threads") == 1
         and rep.get("threshold") == "exact equality; no tolerance", "REPLAY_NOT_PASSED")
    units = rep.get("units", {})
    need(set(units) == {"t3_unit_C"}, "REPLAY_UNITS")
    unit = units["t3_unit_C"]
    need(unit.get("result_sha256") == result_sha, "REPLAY_STALE_RESULT")
    modules = unit.get("modules", {})
    need(unit.get("checkpoints") == 12 and unit.get("exact") is True
         and set(modules) == {n[:-3] for n in CKPTS}
         and all(m.get(k, {}).get("exact") is True for m in modules.values()
                 for k in ("logits", "reg")), "REPLAY_MODULES")
    trading = unit.get("trading", {})
    need(rep.get("full_trading_metrics_replayed") is True and unit.get("trading_exact") is True
         and set(trading) == {"20261019", "20261020", "20261021"}
         and all(set(ps) == {"TRAIN", "INNER_A", "OBS_2024"}
                 and all(r.get("exact") is True and r.get("differences") == {} for r in ps.values())
                 for ps in trading.values()), "REPLAY_TRADING")


def cmd_emit(a):
    from tools import expert_guided_clean_round as C
    from tools import expert_guided_t3_manifest as T3
    spec = json.loads(Path(a.spec).read_text())
    cdir = Path(a.c_dir)
    e3m_p, sc_p = cdir / "E3M.json", cdir / "SCALER.json"
    rc = json.loads(Path(a.c_receipt).read_text())
    cfg = json.loads(Path(a.c_config).read_text())
    cfg_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    # --- receipt chain
    need(rc.get("config_sha256") == cfg_id, "RECEIPT_CONFIG_IDENTITY")
    hashes = rc.get("hashes", {})
    for n in ["E3M.json", "SCALER.json", "PROBE.json"] + CKPTS:
        need(n in hashes and (cdir / n).exists() and sha(cdir / n) == hashes[n], f"SEALED_FILE:{n}")
    import shlex
    job = rc.get("job", {})
    cmd = job.get("peer_command", "")
    tok = shlex.split(cmd)
    arg = lambda f: tok[tok.index(f) + 1] if f in tok else None  # noqa: E731
    need(Path(arg("--groups") or "").name == Path(a.groups).name and Path(arg("--feature-list") or "").name == Path(a.list).name, "RECEIPT_COMMAND_INPUTS")
    need(job.get("groups_sha256") == sha(a.groups), "RECEIPT_GROUPS_DIGEST")
    prod = C.authenticate_producer(a.c_config, a.science_commit, blobs=a.science_blobs)
    need(prod.get("authenticated"), "PRODUCER_NOT_AUTHENTICATED")
    # --- combination decision and replay
    comb, rep = json.loads(Path(a.combination).read_text()), json.loads(Path(a.replay).read_text())
    need(comb.get("frozen_rule_decision") == "COMBINATION_SELECTED" and comb.get("state") == "COMBINATION_SELECTED", "COMBINATION_NOT_SELECTED")
    need(comb.get("combined_result_sha256") == sha(e3m_p), "COMBINATION_RESULT_DIGEST")
    need(comb.get("producer_config_sha256") == cfg_id, "COMBINATION_PRODUCER")
    need(comb.get("checkpoint_probe_replay") == "PASS" and comb.get("full_trading_metrics_exact") is True, "COMBINATION_REPLAY_NOT_PASSED")
    validate_replay(rep, sha(e3m_p))
    # --- T3 decision -> manifest -> groups regenerated byte for byte
    man, dec = json.loads(Path(a.manifest).read_text()), json.loads(Path(a.decision).read_text())
    need(dec.get("manifest") == man.get("identity"), "MANIFEST_IDENTITY")
    regen = T3.combined_groups(man, dec, Path(a.manifest).parent / "groups" / "unit_00.json")
    need(json.dumps(regen, indent=1).encode() == Path(a.groups).read_bytes(), "GROUPS_NOT_THE_DECISION")
    # --- C's dataset: store / list / scaler / tensor / window
    e3m, scl = json.loads(e3m_p.read_text()), json.loads(sc_p.read_text())
    cols = json.loads(Path(a.list).read_text())["features"]
    need(any(arm.get("list_sha256") == sha(a.list) for st in spec["stages"].values() for arm in st), "LIST_NOT_THE_SELECTION")
    src = (e3m.get("meta", {}).get("sources") or {}).get("sha256", {})
    need(src.get("store.parquet") == spec["store"]["sha256"] == sha(a.store), "STORE_DIGEST")
    need(src.get("MANIFEST.json") == sha(a.store_manifest), "STORE_MANIFEST_DIGEST")
    need(scl.get("policy") == "fixed_train", "SCALER_POLICY")
    n = len(cols)
    tensor = list(scl.get("features") or [])
    need(len(tensor) == n == int(scl.get("n_features", -1)), "SCALER_LENGTHS")
    need([t.split("@")[0] for t in tensor] == [c.split("@")[0] for c in cols], "SCALER_COLUMNS_DIFFER_FROM_LIST")
    need(tensor == list(e3m["meta"]["input_contract"]["features"]), "SCALER_DIFFERS_FROM_C_DATASET")
    for k in ("mean", "std", "std_raw", "variance", "rule", "fit_rows_per_feature", "nan_share_train"):
        need(isinstance(scl.get(k), list) and len(scl[k]) == n, f"SCALER_FIELD:{k}")
    mean, std = np.asarray(scl["mean"], float), np.asarray(scl["std"], float)
    need(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), "SCALER_CONSTANTS_INVALID")
    fd = e3m["meta"]["scaling"]["features"]
    need(np.array_equal(mean, [fd[t]["mean"] for t in tensor]) and np.array_equal(std, [fd[t]["std"] for t in tensor]), "SCALER_DIFFERS_FROM_C_DATASET")
    need(list(scl["fit_period"]) == list(e3m["meta"]["scaling"]["train"]), "SCALER_FIT_PERIOD")
    win, stride = int(scl["window"]["observations"]), int(scl["window"]["stride_rows"])
    need((win, stride) == (int(e3m["window"]["observations"]), int(e3m["window"]["stride_rows"])), "WINDOW_DIFFERS_FROM_C")
    differenced = set((scl.get("stationarity") or {}).get("differenced") or [])
    # --- representation: resolved groups and model widths equal C's architecture
    idx, units, names, kinds = C.load_groups(a.groups, tensor)
    arch = e3m["arch"]
    need(arch.get("group_names") == names and arch.get("groups") == idx and arch.get("group_units") == units
         and (arch.get("group_kinds") or ["conv"] * len(names)) == kinds, "GROUPS_DIFFER_FROM_C_ARCHITECTURE")
    m = spec["model"]
    need((int(arch["units"]), int(arch["depth"]), int(arch["head"]), arch.get("core")) == (int(m["units"]), int(m["depth"]), int(m["head"]), m.get("core")), "MODEL_DIFFERS_FROM_C")
    reg = spec["regularization"]
    need(float(arg("--dropout") or "nan") == float(reg["dropout"]) and float(arg("--wd") or "nan") == float(reg["wd"]), "REGULARIZATION_DIFFERS_FROM_C")
    need((int(arg("--window") or -1), int(arg("--stride") or -1)) == (win, stride) and arg("--split-contract") == spec["split_contract"]
         and arg("--exec-contract") == spec["exec_contract"] and arg("--agent-state") == "none", "COMMAND_DIFFERS_FROM_RECIPE")
    # --- labels
    mode, how = C.label_mode_of(e3m, prod)
    need(C.labels_ok(mode, how), f"LABELS:{mode}:{how}")
    # --- stages and semantics
    l1 = json.loads(Path(a.layer1_manifest).read_text())
    l2 = json.loads(Path(a.layer2_manifest).read_text())
    norm = lambda d: {c: (v if isinstance(v, str) else v["decision"]) for c, v in d.items()}  # noqa: E731
    l1dec, l2dec = norm(l1["decisions"]), norm(l2["decisions"])
    l2fam = {c: (v.get("family") if isinstance(v, dict) else None) for c, v in (l2.get("columns") or {}).items()}
    scope = {"fit_window": l2.get("fit_period"), "note": "learned producers fitted on the consumer scope only"} if l2.get("fit_period") else None
    tcols = []
    for c, t in zip(cols, tensor):
        ops = ["DIFF"] if t != c and t.endswith("@diff") and c in differenced else []
        need(t == (f"{c}@diff" if ops else c), f"TENSOR_NAME:{t}")
        tcols.append({"store_column": c, "tensor_name": t, "upstream": upstream_stage(c, l1dec, l2dec, l2fam, scope),
                      "consumer_ops": ops, "then": "fixed TRAIN z with this recipe's constants"})
    adm = Path(a.admitted_dir)
    rec = {
        "schema": "expert_guided_consumer_recipe.v2", "scientific_release": False,
        "scope": "shared representation for both lanes (expert-guided learning; SAC without expert input); E3-M selection is a proxy, not evidence that SAC benefits",
        "store": {"file": Path(a.store).name, "sha256": sha(a.store), "manifest_sha256": sha(a.store_manifest)},
        "list": {"file": Path(a.list).name, "sha256": sha(a.list), "n": n, "columns": cols},
        "tensor": {"columns": tcols, "window": window_semantics(Path(a.store), win, stride)},
        "scaler": {k: scl[k] for k in ("policy", "features", "n_features", "fit_period", "fit_rows", "ddof", "mean", "std", "std_raw", "variance", "rule",
                                       "fit_rows_per_feature", "nan_share_train", "nan_policy", "transform", "stationarity")} | {"artifact_sha256": sha(sc_p)},
        "representation": {"groups": json.loads(Path(a.groups).read_text()), "resolved": {"names": names, "indices": idx, "units": units, "kinds": kinds},
                           "groups_file_sha256": sha(a.groups), "model": {"units": int(arch["units"]), "depth": int(arch["depth"]), "head": int(arch["head"]), "core": arch.get("core")},
                           "params_per_module": e3m["params"]["FULL"]},
        "regularization": {"dropout": reg["dropout"], "wd": reg["wd"], "source": reg.get("source")},
        "split": e3m["split_contract"] | {"periods": e3m["meta"]["periods"], "support_hash": e3m["meta"]["support_hash"]},
        "business": business_contract(e3m.get("exec_contract", "sac"), float(e3m.get("min_orders_per_week", 2.0)), sha(adm / "CLOSURE_INTERVALS.json"), sha(adm / "eurusd.parquet")),
        "labels": {"mode": mode, "how": how, "note": "training-only hindsight demonstrations; never an observation; opportunity/both is a separate experiment"},
    }
    ident_keys = ("store", "list", "tensor", "scaler", "representation", "regularization", "split", "business", "labels")
    rec["identity"] = "recipe_" + hashlib.sha256(json.dumps({k: rec[k] for k in ident_keys}, sort_keys=True).encode()).hexdigest()[:16]
    rec["identity_covers"] = list(ident_keys)
    rec["provenance"] = {"science_commit": a.science_commit, "c_receipt_sha256": sha(a.c_receipt), "c_producer_config_identity": cfg_id,
                         "combination_decision_sha256": sha(a.combination), "combination_replay_sha256": sha(a.replay),
                         "t3_manifest": man["identity"], "t3_manifest_sha256": sha(a.manifest), "t3_decision_sha256": sha(a.decision),
                         "c_result_sha256": sha(e3m_p), "producer": {k: prod[k] for k in ("authenticated", "emits_labels", "science_commit")}}
    rec["custody"] = "reports and physical copies verified by hash; OLAP ingestion / read-back NOT certified by this recipe"
    Path(a.out).write_text(json.dumps(jsonable(rec), indent=1) + "\n")
    print(json.dumps({"identity": rec["identity"], "columns": n, "consumer_diff": sum(1 for c in tcols if c["consumer_ops"]),
                      "upstream_transformed": sum(1 for c in tcols if c["upstream"]["applied_in_store"]), "out": a.out}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    for k in ("spec", "c-dir", "c-receipt", "c-config", "groups", "list", "manifest", "decision", "combination", "replay",
              "store", "store-manifest", "layer1-manifest", "layer2-manifest", "admitted-dir", "science-commit", "out"):
        e.add_argument(f"--{k}", required=True)
    e.add_argument("--science-blobs", default=None, help="science_blobs/<commit>/tools/expert_guided_e3m.py exported from git (hosts without the repository)")
    a = ap.parse_args()
    try:
        sys.exit({"emit": cmd_emit}[a.cmd](a) or 0)
    except RecipeError as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
