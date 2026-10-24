"""The clean-round launcher consumes the ACCEPTED shared recipe (Musashi's re-review 2026-10-24) and varies only
training-side treatments: it refuses every forbidden configuration (rolling z-score, agent feedback, undeclared or
recipe-divergent regularization, store / model / split drift from the recipe, arm overrides of recipe fields,
non-admitted or too-long demonstrators) and every command that is not recipe-bound; it refuses to emit while the
launch gate is closed. No data, no model fit."""
import copy
import hashlib
import json

import pytest

pytest.importorskip("torch")

from tools import expert_guided_clean_round as C  # noqa: E402

STORE = b"store-bytes"
COLS = ["a", "b", "c", "d"]


def make_recipe(store_sha):
    return {"schema": "expert_guided_consumer_recipe.v2", "scientific_release": False, "identity": "recipe_test0000000001",
            "store": {"file": "store.parquet", "sha256": store_sha, "manifest_sha256": "m"},
            "list": {"file": "FEATURE_LIST_x.json", "sha256": "l", "n": 4, "columns": COLS},
            "tensor": {"columns": [{"store_column": c, "tensor_name": c, "consumer_ops": []} for c in COLS],
                       "window": {"observations": 13, "stride_rows": 2}},
            "scaler": {"features": COLS, "mean": [0.0] * 4, "std": [1.0] * 4},
            "representation": {"groups": {"groups": {"g1": ["a", "b"], "g2": ["c", "d"]}, "units": {"g1": 2, "g2": 2}, "kinds": {"g1": "conv", "g2": "pointwise"}},
                               "resolved": {"names": ["g1", "g2"], "indices": [[0, 1], [2, 3]], "units": [2, 2], "kinds": ["conv", "pointwise"]},
                               "model": {"units": 2, "depth": 2, "head": 16, "core": None}},
            "regularization": {"dropout": 0.3, "wd": 0.01, "source": "T2"},
            "split": {"arm": "consolidated"},
            "business": {"evaluation_and_policy": {"max_entries_per_day": 1, "min_orders_per_week": 2.0}},
            "labels": {"mode": "zigzag"}}


@pytest.fixture()
def spec(tmp_path):
    store = tmp_path / "store.parquet"
    store.write_bytes(STORE)
    rec = tmp_path / "CONSUMER_RECIPE.json"
    rec.write_text(json.dumps(make_recipe(hashlib.sha256(STORE).hexdigest())))
    demo = tmp_path / "DEMONSTRATORS.json"
    demo.write_text(json.dumps({"demonstrators": [{"spec": "3", "admitted": True}, {"spec": "2", "admitted": True},
                                                  {"spec": "rev10", "admitted": True}, {"spec": "rev5", "admitted": False},
                                                  {"spec": "6", "admitted": True}]}))
    gate = tmp_path / "LAUNCH_EXPERT_GUIDED.json"
    s = {"identity": "t", "scaling": "fixed_train", "nonstationary": "diff", "split_contract": "consolidated", "exec_contract": "sac", "min_orders_per_week": 2.0,
         "store": {"path": str(store), "sha256": hashlib.sha256(STORE).hexdigest()},
         "recipe": {"path": str(rec), "sha256": C.sha(rec), "identity": "recipe_test0000000001"},
         "model": {"units": 2, "core": None, "depth": 2, "head": 16, "max_epochs": 5},
         "limits": {"max_extractor_units": 4, "max_core_width": 16, "max_head": 32},
         "regularization": {"dropout": 0.3, "wd": 0.01, "source": "T2"},
         "demonstrators": {"file": str(demo)}, "memory": "4G", "remote_root": "/r", "decision_control": "zz",
         "gates": [{"path": str(gate)}],
         "stages": {"A": [{"name": "zz", "labels": "zigzag"},
                          {"name": "opp_hold", "labels": "opportunity", "closer_rule": "hold_through"},
                          {"name": "both_experts", "labels": "both", "extra_experts": ["2", "rev10"]}],
                    "B": [{"name": "winner_aux", "aux_targets": True},
                          {"name": "winner_minimal_state", "agent_state": "minimal"}]}}
    return s, tmp_path


def test_valid_spec_passes(spec):
    s, _ = spec
    assert C.validate(s) == []


@pytest.mark.parametrize("mut, code", [
    (lambda s: s.update(scaling="rolling_z"), "SCALING_FORBIDDEN"),
    (lambda s: s.update(nonstationary="keep"), "NONSTATIONARY_POLICY_FORBIDDEN"),
    (lambda s: s.update(split_contract="dev3y"), "CONTRACT_NOT_PRINCIPAL"),
    (lambda s: s["regularization"].update(dropout=None), "REGULARIZATION_NOT_DECLARED"),
    (lambda s: s["regularization"].update(dropout=0.1, wd=1e-4), "REGULARIZATION_DIFFERS_FROM_RECIPE"),
    (lambda s: s["stages"]["A"][0].update(agent_state="full"), "AGENT_STATE_FORBIDDEN"),
    (lambda s: [a.update(agent_state="minimal") for a in s["stages"]["A"]], "STAGE_WITHOUT_NO_FEEDBACK_ARM"),
    (lambda s: s["model"].update(depth=1), "MODEL_DIFFERS_FROM_RECIPE"),
    (lambda s: s["model"].update(units=8), "MODEL_EXCEEDS_DECLARED_LIMITS"),
    (lambda s: s["store"].update(sha256="0" * 64), "STORE_DIFFERS_FROM_RECIPE"),
    (lambda s: s["recipe"].update(sha256="0" * 64), "RECIPE_DIGEST_MISMATCH"),
    (lambda s: s["recipe"].update(identity="recipe_other"), "RECIPE_IDENTITY_MISMATCH"),
    (lambda s: s["recipe"].update(path="/nonexistent/recipe.json"), "RECIPE_FILE_MISSING"),
    (lambda s: s["stages"]["A"][2].update(extra_experts=["rev5"]), "DEMONSTRATOR_NOT_ADMITTED"),
    (lambda s: s["stages"]["A"][2].update(extra_experts=["6"]), "DEMONSTRATOR_NOT_SHORTER_THAN_OPTIMUM"),
    (lambda s: s["stages"]["A"][1].update(labels="oracle"), "LABELS_MODE_UNKNOWN"),
    (lambda s: s["stages"]["A"][1].update(closer_rule="greedy"), "CLOSER_RULE_UNKNOWN"),
    (lambda s: s["stages"]["A"][0].update(list="/some/list.json"), "ARM_OVERRIDES_RECIPE"),
    (lambda s: s["stages"]["A"][0].update(window=[24, 1]), "ARM_OVERRIDES_RECIPE"),
    (lambda s: s.update(exec_contract="native"), "EXEC_CONTRACT_NOT_REPLAY"),
    (lambda s: s.update(min_orders_per_week=3.0), "MIN_ORDERS_PER_WEEK_NOT_OWNER_RULE"),
])
def test_each_forbidden_configuration_is_refused(spec, mut, code):
    s, _ = spec
    s = copy.deepcopy(s)
    mut(s)
    assert any(e.startswith(code) for e in C.validate(s)), C.validate(s)


def test_recipe_drift_is_refused(spec):
    s, t = spec
    rec = json.loads((t / "CONSUMER_RECIPE.json").read_text())
    for mutate, code in ((lambda r: r["store"].update(sha256="0" * 64), "STORE_DIFFERS_FROM_RECIPE"),
                         (lambda r: r["representation"]["model"].update(head=32), "MODEL_DIFFERS_FROM_RECIPE"),
                         (lambda r: r["split"].update(arm="dev3y"), "RECIPE_SPLIT_MISMATCH"),
                         (lambda r: r["business"]["evaluation_and_policy"].update(max_entries_per_day=4), "RECIPE_BUSINESS_CONTRACT"),
                         (lambda r: r.update(scientific_release=True), "RECIPE_SCHEMA")):
        r2 = copy.deepcopy(rec)
        mutate(r2)
        p = t / "R2.json"
        p.write_text(json.dumps(r2))
        s2 = copy.deepcopy(s)
        s2["recipe"] = {"path": str(p), "sha256": C.sha(p), "identity": r2["identity"]}
        assert any(e.startswith(code) for e in C.validate(s2)), (code, C.validate(s2))


def test_commands_are_recipe_bound_and_carry_only_treatments(spec):
    s, _ = spec
    cmd = " ".join(C.e3m_cmd(s, s["stages"]["A"][2], "OUT", "ROOT"))
    assert "--recipe ROOT/recipe/CONSUMER_RECIPE.json" in cmd and "--store ROOT/store/store.parquet" in cmd
    assert all(f not in cmd for f in C.FORBIDDEN_FLAGS) and "rolling" not in cmd
    assert "--labels both --closer-rule hold_through" in cmd and "--extra-experts 2,rev10" in cmd and "--agent-state none" in cmd
    assert "--dropout 0.3 --wd 0.01" in cmd and "--exec-contract sac --min-orders-per-week 2.0" in cmd
    aux = " ".join(C.e3m_cmd(s, s["stages"]["B"][0], "OUT", "ROOT"))
    assert "--aux-targets" in aux and "--labels zigzag" in aux


def test_emit_is_refused_while_the_launch_gate_is_closed_and_written_once_open(spec, tmp_path):
    s, t = spec
    sp = t / "SPEC.json"
    sp.write_text(json.dumps(s))
    a = type("A", (), {"spec": str(sp), "stage": "A", "winner_from": None, "out": str(t / "q.sh")})()
    with pytest.raises(SystemExit, match="GATES_CLOSED"):
        C.cmd_emit(a)
    (t / "LAUNCH_EXPERT_GUIDED.json").write_text("{}")
    C.cmd_emit(a)
    q = (t / "q.sh").read_text()
    assert q.count("expert_guided_e3m.py run") == 3 and "sha256sum -c" in q and "LAUNCH_EXPERT_GUIDED.json" in q and "RELEASE.json" not in q
    assert "--feature-list" not in q and "--groups" not in q
    s2 = copy.deepcopy(s)
    s2["scaling"] = "rolling_z"
    sp.write_text(json.dumps(s2))
    with pytest.raises(SystemExit, match="SPEC_INVALID"):
        C.cmd_emit(a)


def test_parameter_counts_are_reported_per_module_on_the_recipe(spec):
    s, _ = spec
    rec = C.load_recipe(s)
    p = C.arm_params(s["stages"]["A"][0], s, rec)
    assert set(p) == {"open_l", "open_s", "close_l", "close_s"} and all(v > 0 for v in p.values())
    q = C.arm_params(s["stages"]["B"][0], s, rec)
    assert q["open_l"] > p["open_l"] and q["close_l"] > p["close_l"]          # the aux heads add outputs only
