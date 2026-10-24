"""Consumer recipe v2, pinned-recipe consumption and the source-to-tensor probe (Musashi's final joint audit A2/A3/A4,
2026-10-23). A synthetic but complete producer chain: the honest derivation passes; Musashi's seven retained
mutations and the controls are rejected; the loader applies a recipe exactly; the probe catches a permuted column,
an altered scaler and wrong groups before training. No store, no fit."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import consumer_probe as CP  # noqa: E402
from tools import expert_guided_t3_manifest as T3  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402

SCIENCE = "1aaf367923fae467988b1ca87d5aa3c74321141f"
COLS = ["a", "b", "c", "d"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _blob_sha():
    r = subprocess.run(["git", "show", f"{SCIENCE}:tools/expert_guided_e3m.py"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        pytest.skip("science commit not in this clone")
    return hashlib.sha256(r.stdout).hexdigest()


def make_chain(t: Path):
    """Every file the exporter reads, mutually consistent."""
    (t / "groups").mkdir(parents=True)
    ts = pd.date_range("2020-01-01", periods=300, freq="h", tz="UTC")
    store = t / "store.parquet"
    pd.DataFrame({"timestamp": ts, **{c: np.arange(300.0) + i for i, c in enumerate(COLS)}}).to_parquet(store, index=False)
    smani = t / "MANIFEST.json"
    smani.write_text(json.dumps({"features": COLS, "file_sha256": sha(store)}))
    lst = t / "FEATURE_LIST_x.json"
    lst.write_text(json.dumps({"features": COLS}))
    (t / "groups" / "unit_00.json").write_text(json.dumps({"groups": {"g1": ["a", "b"], "g2": ["c", "d"]}, "units": {"g1": 2, "g2": 2},
                                                          "kinds": {"g1": "conv", "g2": "conv"}}))
    man = t / "T3_MANIFEST.json"
    man.write_text(json.dumps({"identity": "t3v2_test"}))
    dec = t / "T3_DECISION.json"
    dec.write_text(json.dumps({"manifest": "t3v2_test", "groups": [{"group": "g2", "chosen": "pointwise"}, {"group": "g1", "chosen": "conv"}]}))
    groups = t / "groups" / "unit_C.json"
    groups.write_text(json.dumps(T3.combined_groups(json.loads(man.read_text()), json.loads(dec.read_text()), t / "groups" / "unit_00.json"), indent=1))
    spec = t / "SPEC.json"
    spec.write_text(json.dumps({"store": {"path": str(store), "sha256": sha(store)}, "split_contract": "consolidated", "exec_contract": "sac",
                                "model": {"units": 2, "depth": 2, "head": 16, "core": None}, "regularization": {"dropout": 0.1, "wd": 0.0001, "source": "T2"},
                                "stages": {"A": [{"name": "x", "list_sha256": sha(lst)}]}}))
    cdir = t / "C"
    cdir.mkdir()
    mean, std = [0.5, 1.5, 2.5, 3.5], [1.0, 2.0, 3.0, 4.0]
    sc = {"artifact": "SCALER", "policy": "fixed_train", "features": COLS, "n_features": 4, "fit_period": ["2020-01-01", "2023-07-01"], "fit_rows": 100,
          "ddof": 0, "mean": mean, "std": std, "std_raw": std, "variance": [s * s for s in std], "rule": ["z"] * 4, "fit_rows_per_feature": [100] * 4,
          "nan_share_train": [0.0] * 4, "nan_policy": "nan-aware", "transform": "(x - mean) / std", "stationarity": {"differenced": []},
          "window": {"observations": 13, "stride_rows": 2}}
    (cdir / "SCALER.json").write_text(json.dumps(sc))
    e3m = {"meta": {"sources": {"sha256": {"store.parquet": sha(store), "MANIFEST.json": sha(smani)}}, "input_contract": {"features": COLS},
                    "scaling": {"features": {c: {"mean": m, "std": s} for c, m, s in zip(COLS, mean, std)}, "train": ["2020-01-01", "2023-07-01"]},
                    "periods": {"TRAIN": ["2020-01-01", "2023-07-01"]}, "support_hash": {"TRAIN": "t"}},
           "window": {"observations": 13, "stride_rows": 2},
           "arch": {"units": 2, "depth": 2, "head": 16, "core": None, "group_names": ["g1", "g2"], "groups": [[0, 1], [2, 3]], "group_units": [2, 2],
                    "group_kinds": ["conv", "pointwise"]},
           "params": {"FULL": {"open_l": 1}}, "split_contract": {"arm": "consolidated"}, "exec_contract": "sac", "min_orders_per_week": 2.0,
           "labels": {"mode": "zigzag"}}
    (cdir / "E3M.json").write_text(json.dumps(e3m))
    (cdir / "PROBE.json").write_text("{}")
    from tools.expert_guided_consumer_recipe import CKPTS
    for n in CKPTS:
        (cdir / n).write_bytes(n.encode())
    cfg = t / "CONFIG.json"
    cfg.write_text(json.dumps({"pins": {"<root>/code/tools/expert_guided_e3m.py": _blob_sha()}}))
    cfg_id = hashlib.sha256(json.dumps(json.loads(cfg.read_text()), sort_keys=True).encode()).hexdigest()
    cmd = ("tools/expert_guided_e3m.py run --store '<root>/data/s.parquet' --feature-list '<root>/in/FEATURE_LIST_x.json' --groups '<root>/t3/groups/unit_C.json' "
           "--dropout 0.1 --wd 0.0001 --split-contract consolidated --agent-state none --window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0")
    rc = t / "RESULT.json"
    rc.write_text(json.dumps({"config_sha256": cfg_id, "job": {"peer_command": cmd, "groups_sha256": sha(groups)},
                              "hashes": {n: sha(cdir / n) for n in ["E3M.json", "SCALER.json", "PROBE.json"] + CKPTS}}))
    comb = t / "COMB.json"
    comb.write_text(json.dumps({"state": "COMBINATION_SELECTED", "frozen_rule_decision": "COMBINATION_SELECTED", "combined_result_sha256": sha(cdir / "E3M.json"),
                                "producer_config_sha256": cfg_id, "checkpoint_probe_replay": "PASS", "full_trading_metrics_exact": True}))
    rep = t / "REPLAY.json"
    rep.write_text(json.dumps({"schema": "t3_checkpoint_probe_replay.v1", "state": "PASS", "count": 1,
                              "checkpoints": 12, "threshold": "exact equality; no tolerance", "torch_cpu_threads": 1,
                              "full_trading_metrics_replayed": True, "units": {"t3_unit_C": {
                                  "result_sha256": sha(cdir / "E3M.json"), "checkpoints": 12, "exact": True,
                                  "modules": {n[:-3]: {"logits": {"exact": True}, "reg": {"exact": True}} for n in CKPTS},
                                  "trading_exact": True, "trading": {str(s): {p: {"exact": True, "differences": {}}
                                      for p in ("TRAIN", "INNER_A", "OBS_2024")} for s in (20261019, 20261020, 20261021)}}}}))
    for name in ("L1.json", "L2.json"):
        (t / name).write_text(json.dumps({"decisions": {c: "NONE" for c in COLS}, "columns": {}}))
    adm = t / "admitted"
    adm.mkdir()
    (adm / "CLOSURE_INTERVALS.json").write_text("[]")
    (adm / "eurusd.parquet").write_bytes(b"x")
    args = {"spec": spec, "c-dir": cdir, "c-receipt": rc, "c-config": cfg, "groups": groups, "list": lst, "manifest": man, "decision": dec,
            "combination": comb, "replay": rep, "store": store, "store-manifest": smani, "layer1-manifest": t / "L1.json",
            "layer2-manifest": t / "L2.json", "admitted-dir": adm, "science-commit": SCIENCE, "out": t / "RECIPE.json"}
    return args


def emit(args):
    cmd = [sys.executable, str(ROOT / "tools" / "expert_guided_consumer_recipe.py"), "emit"]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def reseal(args):
    """Rewrite the receipt hashes after a mutation, so the SEMANTIC check (not the hash) has to catch it."""
    rc = json.loads(Path(args["c-receipt"]).read_text())
    rc["hashes"] = {n: sha(Path(args["c-dir"]) / n) for n in rc["hashes"]}
    Path(args["c-receipt"]).write_text(json.dumps(rc))
    comb = json.loads(Path(args["combination"]).read_text())
    comb["combined_result_sha256"] = sha(Path(args["c-dir"]) / "E3M.json")
    Path(args["combination"]).write_text(json.dumps(comb))
    rep = json.loads(Path(args["replay"]).read_text())
    rep["units"]["t3_unit_C"]["result_sha256"] = comb["combined_result_sha256"]
    Path(args["replay"]).write_text(json.dumps(rep))


def edit_json(p, fn):
    d = json.loads(Path(p).read_text())
    fn(d)
    Path(p).write_text(json.dumps(d))


def test_honest_derivation_passes_and_records_stages_window_and_business(tmp_path):
    args = make_chain(tmp_path)
    r = emit(args)
    assert r.returncode == 0, r.stderr
    rec = json.loads(Path(args["out"]).read_text())
    assert rec["schema"] == "expert_guided_consumer_recipe.v2" and rec["identity"].startswith("recipe_") and rec["scientific_release"] is False
    assert [c["consumer_ops"] for c in rec["tensor"]["columns"]] == [[]] * 4
    assert rec["tensor"]["window"]["row_unit"].startswith("observed admitted market bar") and rec["tensor"]["window"]["observations"] == 13
    assert rec["business"]["evaluation_and_policy"]["max_entries_per_day"] == 1 and rec["business"]["evaluation_and_policy"]["min_orders_per_week"] == 2.0
    assert rec["representation"]["resolved"]["kinds"] == ["conv", "pointwise"] and set(rec["identity_covers"]) >= {"business", "tensor", "scaler"}
    assert rec["labels"]["mode"] == "zigzag"


@pytest.mark.parametrize("name, mutate, code", [
    ("scaler_mean_changed", lambda a: edit_json(Path(a["c-dir"]) / "SCALER.json", lambda d: d["mean"].__setitem__(0, 9.0)), "SEALED_FILE"),
    ("scaler_mean_changed_resealed", lambda a: (edit_json(Path(a["c-dir"]) / "SCALER.json", lambda d: d["mean"].__setitem__(0, 9.0)), reseal(a)), "SCALER_DIFFERS_FROM_C_DATASET"),
    ("scaler_policy_rolling_resealed", lambda a: (edit_json(Path(a["c-dir"]) / "SCALER.json", lambda d: d.__setitem__("policy", "rolling_z")), reseal(a)), "SCALER_POLICY"),
    ("group_kind_changed", lambda a: edit_json(a["groups"], lambda d: d["kinds"].__setitem__("g2", "conv")), "RECEIPT_GROUPS_DIGEST"),
    ("model_depth_changed", lambda a: edit_json(a["spec"], lambda d: d["model"].__setitem__("depth", 3)), "MODEL_DIFFERS_FROM_C"),
    ("store_digest_changed", lambda a: edit_json(a["spec"], lambda d: d["store"].__setitem__("sha256", "0" * 64)), "STORE_DIGEST"),
    ("manifest_identity_changed", lambda a: edit_json(a["manifest"], lambda d: d.__setitem__("identity", "t3v2_other")), "MANIFEST_IDENTITY"),
    ("replay_failed", lambda a: edit_json(a["replay"], lambda d: d.__setitem__("state", "FAIL")), "REPLAY_NOT_PASSED"),
    ("combination_replay_failed", lambda a: edit_json(a["combination"], lambda d: d.__setitem__("checkpoint_probe_replay", "FAIL")), "COMBINATION_REPLAY_NOT_PASSED"),
    ("nonfinite_constant_resealed", lambda a: (edit_json(Path(a["c-dir"]) / "SCALER.json", lambda d: d["std"].__setitem__(1, 0.0)), reseal(a)), "SCALER_CONSTANTS_INVALID"),
    ("malformed_lengths_resealed", lambda a: (edit_json(Path(a["c-dir"]) / "SCALER.json", lambda d: d["std"].pop()), reseal(a)), "SCALER_FIELD:std"),
    ("regularization_changed", lambda a: edit_json(a["spec"], lambda d: d["regularization"].__setitem__("dropout", 0.3)), "REGULARIZATION_DIFFERS_FROM_C"),
    ("negative_control_bad_result", lambda a: Path(a["c-dir"], "E3M.json").write_text("{}"), "SEALED_FILE"),
    ("negative_control_bad_column", lambda a: Path(a["list"]).write_text(json.dumps({"features": ["a", "b", "c", "z"]})), "LIST_NOT_THE_SELECTION"),
    ("labels_opportunity_resealed", lambda a: (edit_json(Path(a["c-dir"]) / "E3M.json", lambda d: d.__setitem__("labels", {"mode": "opportunity"})), reseal(a)), "LABELS:opportunity"),
    ("groups_not_the_decision", lambda a: edit_json(a["decision"], lambda d: d["groups"].__setitem__(0, {"group": "g2", "chosen": "per_feature"})), "GROUPS_NOT_THE_DECISION"),
])
def test_inconsistent_inputs_are_rejected(tmp_path, name, mutate, code):
    args = make_chain(tmp_path)
    mutate(args)
    r = emit(args)
    assert r.returncode != 0 and f"RECIPE_{code}" in r.stderr, (name, r.stderr[-400:])


@pytest.mark.parametrize("mutation", ["stale", "missing", "inexact", "trading", "threshold"])
def test_replay_header_cannot_substitute_for_bound_exact_evidence(tmp_path, mutation):
    args = make_chain(tmp_path)
    rep = json.loads(Path(args["replay"]).read_text())
    unit = rep["units"]["t3_unit_C"]
    if mutation == "stale":
        unit["result_sha256"] = "0" * 64
    elif mutation == "missing":
        rep.pop("units")
    elif mutation == "inexact":
        next(iter(unit["modules"].values()))["logits"]["exact"] = False
    elif mutation == "trading":
        unit["trading"]["20261019"]["TRAIN"]["exact"] = False
    else:
        rep["threshold"] = "inexact"
    Path(args["replay"]).write_text(json.dumps(rep))
    r = emit(args)
    assert r.returncode != 0 and "RECIPE_REPLAY_" in r.stderr, (mutation, r.stdout, r.stderr)


def _recipe_for(cols, mean, std, ops=None, win=3, stride=1):
    ops = ops or [[]] * len(cols)
    tensor = [f"{c}@diff" if o == ["DIFF"] else c for c, o in zip(cols, ops)]
    n = len(cols)
    return {"identity": "recipe_test", "tensor": {"columns": [{"store_column": c, "tensor_name": t, "consumer_ops": o} for c, t, o in zip(cols, tensor, ops)],
                                                  "window": {"observations": win, "stride_rows": stride}},
            "scaler": {"features": tensor, "mean": mean, "std": std, "std_raw": std, "variance": [s * s for s in std], "rule": ["z"] * n,
                       "nan_share_train": [0.0] * n, "fit_rows_per_feature": [10] * n, "fit_period": ["2020-01-01", "2023-07-01"], "fit_rows": 10,
                       "ddof": 0, "nan_policy": "x", "stationarity": {"differenced": [c for c, o in zip(cols, ops) if o]}}}


def test_recipe_windows_apply_pinned_operations_and_constants_exactly():
    X = np.arange(20, dtype=float).reshape(10, 2)
    rec = _recipe_for(["p", "q"], [1.0, 2.0], [2.0, 4.0], ops=[["DIFF"], []])
    W, ok, doc, tensor = DG.recipe_windows(X, ["p", "q"], rec, stride=1, win=3)
    assert tensor == ["p@diff", "q"] and doc["recipe"]["identity"] == "recipe_test"
    z0 = (np.r_[np.nan, np.diff(X[:, 0])] - 1.0) / 2.0
    assert np.allclose(W[5, :, 0], z0[3:6].astype(np.float32)) and np.allclose(W[5, :, 1], ((X[3:6, 1] - 2.0) / 4.0).astype(np.float32))
    assert not ok[0] and not ok[2] and ok[3]                       # the DIFF NaN at row 0 and the warm-up exclude rows 0..2
    with pytest.raises(ValueError, match="RECIPE_COLUMN_ORDER_MISMATCH"):
        DG.recipe_windows(X, ["q", "p"], rec, stride=1, win=3)
    with pytest.raises(ValueError, match="RECIPE_SCALER_INVALID"):
        DG.recipe_windows(X, ["p", "q"], _recipe_for(["p", "q"], [1.0, 2.0], [0.0, 4.0]), stride=1, win=3)
    with pytest.raises(ValueError, match="RECIPE_WINDOW_MISMATCH"):
        DG.recipe_windows(X, ["p", "q"], rec, stride=2, win=3)


def _probe(rec, W, mean, std, groups, cols_order=None):
    n = W.shape[0]
    ts = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    valid = np.ones(n, bool)
    period = np.array(["TRAIN"] * n, object)
    slot = np.stack([ts - 2 * 3_600_000_000_000, ts], 1)
    return CP.probe_document(rec, ts, W, valid, period, slot, cols_order or rec["scaler"]["features"], groups, mean, std)


def test_probe_catches_a_permuted_column_an_altered_scaler_and_wrong_groups():
    rec = _recipe_for(["p", "q"], [1.0, 2.0], [2.0, 4.0])
    rec["split"] = {"periods": {"TRAIN": ["2020-01-01", "2023-07-01"]}}
    rng = np.random.default_rng(0)
    W = rng.normal(size=(40, 3, 2)).astype(np.float32)
    g = {"names": ["g"], "indices": [[0, 1]], "units": [2], "kinds": ["conv"]}
    ref = _probe(rec, W, [1.0, 2.0], [2.0, 4.0], g)
    assert CP.compare(ref, copy.deepcopy(ref)) == []
    assert "PROBE_MISMATCH:tensor_sha256" in CP.compare(ref, _probe(rec, W[:, :, ::-1].copy(), [1.0, 2.0], [2.0, 4.0], g, ["q", "p"]))
    assert "PROBE_MISMATCH:tensor_order_sha256" in CP.compare(ref, _probe(rec, W, [1.0, 2.0], [2.0, 4.0], g, ["q", "p"]))
    assert "PROBE_MISMATCH:scaler_sha256" in CP.compare(ref, _probe(rec, W, [1.0, 2.0], [2.0, 4.5], g))
    assert "PROBE_MISMATCH:groups_sha256" in CP.compare(ref, _probe(rec, W, [1.0, 2.0], [2.0, 4.0], dict(g, kinds=["pointwise"])))


def test_selection_engine_substitutes_fold_local_columns_and_keeps_one_support(tmp_path):
    from tools import pool_selection_v2 as PS
    ts = pd.date_range("2020-01-01", periods=12, freq="h", tz="UTC")
    e = PS.Engine.__new__(PS.Engine)
    e.cols = ["x", "y", "rg.gmm.p0", "rg.gmm.p1", "rg.gmm.p2", "enc.gmm.oh0", "enc.gmm.oh1", "enc.gmm.oh2"]
    M = np.ones((12, len(e.cols)))
    M[:, 2:5] = [0.2, 0.3, 0.5]
    e.M = M
    e.ds = type("DS", (), {"bars": pd.DataFrame(index=ts)})()
    sc_x = np.arange(12.0)
    sc_x[4] = np.nan                                       # the scope's transform loses one cell of support
    p = np.tile([0.6, 0.3, 0.1], (12, 1))
    f = tmp_path / "SCOPE_sel_K3_f0.parquet"
    pd.DataFrame({"timestamp": ts, "x": sc_x, "rg.gmm.p0": p[:, 0], "rg.gmm.p1": p[:, 1], "rg.gmm.p2": p[:, 2]}).to_parquet(f, index=False)
    (tmp_path / "SCOPES_REPORT.json").write_text(json.dumps({"scopes": {"sel_K3_f0": {"file": f.name, "sha256": sha(f)}}}))
    e.scopes = e.load_fold_scopes(tmp_path)
    assert e.support_mask_removed == {"x": 1} and np.isnan(e.M[4, 0])        # the pool loses the same cell: one support
    sc = e.scope_of(3, 0)
    A = e.cols_matrix([0, 1, 5, 6], sc, np.arange(12))
    assert np.array_equal(A[:, 0], sc_x, equal_nan=True) and (A[:, 1] == 1).all()
    assert (A[:, 2] == 1).all() and (A[:, 3] == 0).all()                       # one-hot follows the scope's argmax (state 0)
    assert np.array_equal(e.cols_matrix([5], None, np.arange(12))[:, 0], M[:, 5])
    with pytest.raises(ValueError, match="FOLD_SCOPE_MISSING"):
        e.scope_of(6, 1)
