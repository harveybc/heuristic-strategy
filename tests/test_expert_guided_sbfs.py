"""SBFS with the final model (owner 2026-10-26): the real-model evaluator's structural rules and receipts, and the
driver's supervision loop (controls first, floating backward search with fresh fits, bounded attempts, byte-authenticated
results, resume without refits, undefined scores never 0, decision blocks refused). The real E3-M subprocess is
replaced by a deterministic ORACLE evaluator that honours the same interface; no data, no training."""
import copy
import json
import os
from pathlib import Path

import pytest

from app.short_horizon.expert_guided import fs_evaluator as FSE

SELECTOR_SRC = Path(__file__).resolve().parents[2] / "feature_selector" / "src"
if not SELECTOR_SRC.exists():
    SELECTOR_SRC = Path.home() / "Documents" / "GitHub" / "feature_selector" / "src"
pytestmark = pytest.mark.skipif(not SELECTOR_SRC.exists(), reason="feature_selector checkout not found")

COLS = ["a_sin", "a_cos", "b", "c", "rg.p0", "rg.p1", "rg.p2", "d"]


def parent_recipe():
    cols = [{"store_column": c, "tensor_name": c, "upstream": {"decision": "NONE"}, "consumer_ops": [], "then": "z"} for c in COLS]
    n = len(COLS)
    return {"schema": "expert_guided_consumer_recipe.v2", "scientific_release": False, "scope": "test",
            "store": {"file": "store.parquet", "sha256": "0" * 64, "manifest_sha256": "1" * 64},
            "list": {"file": "L", "sha256": "2" * 64, "n": n, "columns": list(COLS)},
            "tensor": {"columns": cols, "window": {"observations": 13, "stride_rows": 2}},
            "scaler": {"policy": "fixed_train", "features": list(COLS), "n_features": n, "fit_period": ["2020-01-01", "2023-07-01"], "fit_rows": 10, "ddof": 0,
                       "mean": [0.0] * n, "std": [1.0] * n, "std_raw": [1.0] * n, "variance": [1.0] * n, "rule": ["z"] * n, "fit_rows_per_feature": [10] * n,
                       "nan_share_train": [0.0] * n, "nan_policy": "x", "transform": "z", "stationarity": {"policy": "diff", "features": {c: {"decision": "NONE"} for c in COLS}}},
            "representation": {"groups": {"name": "g", "rule": "r", "groups": {"cal": ["a_sin", "a_cos"], "px": ["b", "c"], "rg": ["rg.p0", "rg.p1", "rg.p2"], "one": ["d"]},
                                          "units": {"cal": 2, "px": 2, "rg": 2, "one": 1}, "kinds": {"cal": "pointwise", "px": "conv", "rg": "pointwise", "one": "conv"}},
                               "resolved": {"names": ["cal", "px", "rg", "one"], "indices": [[0, 1], [2, 3], [4, 5, 6], [7]], "units": [2, 2, 2, 1], "kinds": ["pointwise", "conv", "pointwise", "conv"]},
                               "model": {"units": 2, "depth": 2, "head": 16, "core": None}, "params_per_module": {"x": 1}, "groups_file_sha256": "3" * 64},
            "regularization": {"dropout": 0.3, "wd": 0.01, "source": "T2"},
            "split": {"arm": "consolidated", "test_closed_from": "2025-01-01", "periods": {"TRAIN": ["2020-01-01", "2023-07-01"], "INNER_A": ["2023-07-01", "2024-01-01"]}},
            "business": {"evaluation_and_policy": {"max_entries_per_day": 1, "min_orders_per_week": 2.0}}, "labels": {"mode": "zigzag"},
            "identity": "recipe_parent0000000", "identity_covers": list(FSE.IDENTITY_KEYS), "provenance": {}}


def test_subset_recipe_restricts_groups_removes_empty_branch_and_recomputes_fusion_width():
    p = parent_recipe()
    r = FSE.subset_recipe(p, ["b", "rg.p0", "rg.p1", "rg.p2"])
    assert r["list"]["columns"] == ["b", "rg.p0", "rg.p1", "rg.p2"] and r["list"]["n"] == 4
    assert r["representation"]["resolved"]["names"] == ["px", "rg"]                      # cal and one: empty -> branch removed
    assert r["representation"]["resolved"]["indices"] == [[0], [1, 2, 3]]                 # indices remapped to the subset order
    assert r["representation"]["fusion_width"] == 4 and r["scaler"]["n_features"] == 4
    assert r["identity"] != p["identity"] and r["provenance"]["removed_columns"] == ["a_sin", "a_cos", "c", "d"]
    assert "params_per_module" not in r["representation"]
    with pytest.raises(ValueError, match="SUBSET_ORDER"):
        FSE.subset_recipe(p, ["c", "b"])
    with pytest.raises(ValueError, match="SUBSET"):
        FSE.subset_recipe(p, ["zz"])


def test_coherent_blocks_keep_sin_cos_and_regime_triplets_together():
    blocks = FSE.coherent_blocks(COLS)
    assert ["a_sin", "a_cos"] in blocks and ["rg.p0", "rg.p1", "rg.p2"] in blocks and ["b"] in blocks and len(blocks) == 5
    import sys
    sys.path.insert(0, str(SELECTOR_SRC))
    from feature_selector.contracts import subset_view
    d = dataset_doc(parent_recipe())
    with pytest.raises(ValueError, match="PARTIAL_ENCODING_BLOCK"):
        subset_view(d, ["a_sin", "b"])


def test_refit_train_scaler_uses_train_rows_only_and_refuses_constant_columns():
    import numpy as np
    import pandas as pd
    p = parent_recipe()
    ts = pd.date_range("2019-12-31", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c: rng.normal(size=400) for c in COLS})
    frame["timestamp"] = ts
    frame.loc[ts >= pd.Timestamp("2020-06-01", tz="UTC"), "b"] = 1000.0            # rows after the fit window must not move the constants
    r = FSE.refit_train_scaler(p, frame, ("2020-01-01", "2020-06-01"))
    j = COLS.index("b")
    mask = ((ts + pd.Timedelta(hours=1)) >= pd.Timestamp("2020-01-01", tz="UTC")) & ((ts + pd.Timedelta(hours=1)) < pd.Timestamp("2020-06-01", tz="UTC"))
    assert abs(r["scaler"]["mean"][j] - frame.loc[np.asarray(mask), "b"].mean()) < 1e-12 and r["scaler"]["fit_rows"] == int(mask.sum())
    frame["d"] = 5.0
    with pytest.raises(ValueError, match="STD_ZERO_COLUMN:d"):
        FSE.refit_train_scaler(p, frame, ("2020-01-01", "2020-06-01"))


def dataset_doc(rec):
    import sys
    sys.path.insert(0, str(SELECTOR_SRC))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import expert_guided_sbfs as S
    return {"schema": "feature_selector.dataset.v1", "scope": "development", "features": rec["list"]["columns"], "groups": S.groups_for_selector(rec),
            "blocks": FSE.coherent_blocks(rec["list"]["columns"]), "support_hash": "a" * 64, "scaler_hash": "b" * 64, "representation": rec["representation"],
            "splits": {"train": ["2020-01-01", "2023-07-01"], "selection": ["2023-07-01", "2024-01-01"], "test_start": "2025-01-01"},
            "constraints": dict(S.CONSTRAINTS)}


class Oracle(FSE.ExpertGuidedModelEvaluator):
    """Deterministic stand-in for the E3-M subprocess: the score of a subset is a known function of its columns; the
    receipt has the real schema so ``verify`` (the real one) authenticates it. Counts fresh evaluations."""
    SCORES = {("a_sin", "a_cos", "b", "c", "rg.p0", "rg.p1", "rg.p2", "d"): 0.0, ("b", "c", "d"): 0.30, ("b", "d"): 0.45, ("b",): 0.10, ("d",): 0.05,
              ("c", "d"): 0.2, ("b", "c"): 0.25}
    calls = []

    def evaluate(self, request, context, params, work, base):
        self.validate(params)
        Oracle.calls.append((tuple(request["features"]), request["seed"], params["cost_mode"], params["trunk"]))
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        feats = tuple(request["features"])
        base_score = Oracle.SCORES.get(feats, 0.15 - 0.01 * len(feats))
        undefined = feats == ("c",)                                              # a subset whose policy never trades
        art = work / "E3M.json"
        art.write_text(json.dumps({"oracle": True}))
        orders = 0 if undefined else 120
        mk = lambda profit: {"profit": profit, "sharpe": (None if undefined else 1.0), "max_drawdown": 0.1, "orders": orders, "orders_per_week": (0.0 if undefined else 4.6),  # noqa: E731
                             "win_rate": (None if undefined else 0.5), "buy_orders": orders // 2, "sell_orders": orders - orders // 2,
                             "buy_win_rate": (None if undefined else 0.5), "sell_win_rate": (None if undefined else 0.5), "early_closes": (0 if undefined else 30),
                             "costs": (0.0 if params["cost_mode"] == "zero_cost" else 100.0), "valid": not undefined}
        comp = lambda: {"spread": 0.0, "slippage": 0.0, "commission": 0.0, "financing": 0.0} if params["cost_mode"] == "zero_cost" else {"spread": 40.0, "slippage": 20.0, "commission": 40.0, "financing": 0.0}  # noqa: E731
        res = {"schema": FSE.SCHEMA_RESULT, "request_sha256": FSE.digest(request), "recipe_identity": "recipe_x", "recipe_sha256": "0" * 64,
               "features": list(request["features"]), "groups": request.get("groups"), "seed": request["seed"], "support_hash": request["support_hash"],
               "scaler_hash": request["scaler_hash"], "splits": request["splits"], "model_id": params["model_id"], "training_budget": params["training_budget"],
               "action_contract": "action_x", "cost_mode": params["cost_mode"], "architecture": {"trunk": params["trunk"]}, "fresh_training": True, "test_access": False,
               "observation_contract": request["constraints"], "parameter_count": 404,
               "metrics": {"train": mk(500.0), "selection": mk(base_score * 1000)}, "cost_components": {"train": comp(), "selection": comp()},
               "primary": {"statistic": "expectancy", "block": params["selection_block"], "value": (None if undefined else base_score + 0.001 * request["seed"]),
                           "interval": None, "reason": ("ZERO_TRADES" if undefined else None), "reproduced": True},
               "train_primary": {"value": base_score + 0.2, "interval": None}, "decision_blocks_read_by_e3m": [],
               "artifacts": [{"path": str(art), "sha256": FSE.file_sha(art)}], "fits": {}}
        return self.verify(res, request, params, work)


def make_campaign(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(SELECTOR_SRC))
    from tools import expert_guided_sbfs as S
    monkeypatch.setenv("SBFS_FAST_RETRY", "1")
    root = tmp_path / "campaign"
    root.mkdir()
    rec = parent_recipe()
    (root / "RECIPE.json").write_text(json.dumps(rec))
    ac = tmp_path / "ACTION_CONTRACT.json"
    ac.write_text("{}")
    store = tmp_path / "store.parquet"
    store.write_bytes(b"store")
    hs_root = tmp_path / "hs"
    (hs_root / "tools").mkdir(parents=True)
    (hs_root / "tools" / "x.py").write_text("x = 1\n")
    sel_root = tmp_path / "selector"
    (sel_root / "src").mkdir(parents=True)
    (sel_root / "src" / "y.py").write_text("y = 1\n")
    spec = dict(schema="expert_guided_selection.v1", contract="consolidated", inputs=dict(store=str(store), manifest="m", admitted="adm", csv_dir="csv", recipe=str(root / "RECIPE.json")),
                hs_root=str(hs_root), selector_root=str(sel_root), action_contract=str(ac), action_contract_sha256=FSE.file_sha(ac), budget=S.BUDGET, search=dict(S.SEARCH, min_features=1, max_evaluations=40),
                seeds=[7, 19], python="python", crispdm_run="", memory_cap="1G", max_attempts=2, task_timeout_seconds=10, created_at="t",
                code=dict(hs=S.source_pins(hs_root, ["tools"]), selector=S.source_pins(sel_root, ["src"])), recipe_sha256=FSE.file_sha(root / "RECIPE.json"),
                dataset=dataset_doc(rec), selection_block="INNER_A")
    S.write(root / "SPEC.private.json", spec)
    S.write(root / "IDENTITY.json", dict(spec_sha256=FSE.file_sha(root / "SPEC.private.json")))
    return S, root


def test_supervise_runs_controls_then_floating_search_and_resumes_without_refits(tmp_path, monkeypatch):
    S, root = make_campaign(tmp_path, monkeypatch)
    Oracle.calls = []
    close = S.supervise(root, evaluator=Oracle(), selector_root=str(SELECTOR_SRC))
    controls = S.read(root / "CONTROLS.json")
    assert set(controls) == {"tcn_configured", "tcn_zero_cost", "u_configured"}
    assert controls["tcn_configured"]["eligible"] is True and len(controls["tcn_configured"]["seeds"]) == 2
    assert [c for c in Oracle.calls if c[2] == "zero_cost"] and all(len(c[0]) == 8 for c in Oracle.calls[:6])   # controls first, on the full set
    search = S.read(root / "SEARCH.json")
    assert search["selected"] == ["b", "d"] and search["termination"] in ("MIN_SIZE", "NO_NEW_PATH")
    assert close["state"] == "WRAPPER_PHASE_COMPLETE" and close["global_release"] is False and close["test_access"] is False
    # a cardinality-3 incumbent exists and the full set is the comparison baseline
    assert "3" in search["best_by_size"] and search["best_by_size"]["8"]["features"] == COLS
    # every (subset, seed) evaluated once; results are byte-authenticated and reused on resume
    assert len(Oracle.calls) == len(set(Oracle.calls))
    n_calls = len(Oracle.calls)
    (root / "CLOSE.json").unlink()
    S.supervise(root, evaluator=Oracle(), selector_root=str(SELECTOR_SRC))
    assert len(Oracle.calls) == n_calls                                        # resume: zero fresh fits
    # tampering with a retained result is refused
    victim = next(root.glob("tasks/*-attempt-1/RESULT.json"))
    r = S.read(victim)
    r["metrics"]["selection"]["orders"] += 1
    S.write(victim, r)
    (root / "CLOSE.json").unlink()
    with pytest.raises(ValueError, match="DIRECTION_COUNT"):
        S.supervise(root, evaluator=Oracle(), selector_root=str(SELECTOR_SRC))


def test_undefined_score_is_none_never_zero_and_not_eligible(tmp_path, monkeypatch):
    S, root = make_campaign(tmp_path, monkeypatch)
    spec = S.read(root / "SPEC.private.json")
    task = S.task_for(spec, ["c"], 7, "configured", "tcn", False)
    r = S.execute_task(root, spec, task, Oracle())
    assert r["score"] is None and r["gap"] is None and r["eligible"] is False and r["native"]["primary"]["reason"] == "ZERO_TRADES"
    agg = S.aggregate([r, r])
    assert agg["score"] is None and agg["eligible"] is False
    from feature_selector.contracts import validate_score
    validate_score(agg)


def test_decision_block_evaluation_is_refused_and_training_that_read_it_is_rejected(tmp_path, monkeypatch):
    S, root = make_campaign(tmp_path, monkeypatch)
    spec = S.read(root / "SPEC.private.json")
    task = S.task_for(spec, COLS, 7, "configured", "tcn", True)
    bad = copy.deepcopy(task["params"])
    bad["selection_block"] = "INNER_B"
    with pytest.raises(ValueError, match="DECISION_BLOCK_EVALUATION_REFUSED"):
        Oracle().validate(bad)
    ev = Oracle()
    good = ev.evaluate(task["request"], spec["dataset"], task["params"], tmp_path / "w", root)
    native = copy.deepcopy(good["native"])
    native["decision_blocks_read_by_e3m"] = ["INNER_B"]
    with pytest.raises(ValueError, match="DECISION_BLOCK_READ_BY_TRAINING"):
        ev.verify(native, task["request"], task["params"], root)
    native = copy.deepcopy(good["native"])
    native["primary"]["block"] = "OBS_2024"
    with pytest.raises(ValueError, match="DECISION_BLOCK_EVALUATION_REFUSED"):
        ev.verify(native, task["request"], task["params"], root)


def test_zero_cost_is_diagnostic_only_and_cannot_release(tmp_path, monkeypatch):
    S, root = make_campaign(tmp_path, monkeypatch)
    spec = S.read(root / "SPEC.private.json")
    p = S.params_for(spec, "zero_cost", "tcn")
    assert p["selection_use"] == "diagnostic" and Oracle().can_release(p) is False
    with pytest.raises(ValueError, match="ZERO_COST_DIAGNOSTIC_ONLY"):
        Oracle().validate({**p, "selection_use": "cost_aware"})
    assert Oracle().can_release(S.params_for(spec, "configured", "tcn")) is True


def test_interrupted_attempt_is_retained_and_bounded_retries(tmp_path, monkeypatch):
    S, root = make_campaign(tmp_path, monkeypatch)
    spec = S.read(root / "SPEC.private.json")
    task = S.task_for(spec, ["b", "d"], 7, "configured", "tcn", False)
    name = FSE.digest(task)
    (root / "tasks").mkdir()
    S.write(root / "tasks" / (name + "-attempt-1.json"), dict(state="STARTED"))      # an interrupted first attempt
    r = S.execute_task(root, spec, task, Oracle())
    assert r["eligible"] and (root / "tasks" / (name + "-attempt-2" ) / "RESULT.json").exists()
    assert S.read(root / "tasks" / (name + "-attempt-1.json"))["state"] == "STARTED"  # never overwritten

    class Broken(Oracle):
        def evaluate(self, *a, **k):
            raise RuntimeError("E3M_FAILED:1")
    task2 = S.task_for(spec, ["b", "c"], 7, "configured", "tcn", False)
    with pytest.raises(RuntimeError, match="TASK_FAILED_AFTER_BOUNDED_RETRIES"):
        S.execute_task(root, spec, task2, Broken())
    assert len(list(root.glob(f"tasks/{FSE.digest(task2)}-attempt-*.json"))) == 2


def test_exclusive_lock_and_source_pins(tmp_path, monkeypatch):
    S, root = make_campaign(tmp_path, monkeypatch)
    import fcntl
    with (root / "RUN.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="RUN_ALREADY_ACTIVE"):
            S.supervise(root, evaluator=Oracle(), selector_root=str(SELECTOR_SRC))
    spec = S.read(root / "SPEC.private.json")
    (Path(spec["hs_root"]) / "tools" / "x.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="SOURCE_PIN:hs"):
        S.verify(root)


def test_emit_writes_a_gated_queue_and_launches_nothing(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import expert_guided_sbfs as S
    out = tmp_path / "queue_sbfs.sh"
    a = type("A", (), dict(package_root="/pkg", campaign_root="/pkg/sbfs", out=str(out), contract="consolidated", action_contract_name="ACTION_CONTRACT_C3.json",
                           python="py", crispdm_run="cr", memory="10G"))()
    S.cmd_emit(a)
    q = out.read_text()
    assert "LAUNCH_EXPERT_GUIDED.json" in q and "expert_guided_sbfs.py run" in q and "--evaluator-path" in q and "core19" not in q and os.access(out, os.X_OK)


def test_entry_point_is_declared_in_packaging():
    setup = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    assert "'feature_selector.evaluators'" in setup and "expert_guided_model=app.short_horizon.expert_guided.fs_evaluator:ExpertGuidedModelEvaluator" in setup
