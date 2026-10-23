"""The clean-round runner refuses every configuration the owner forbade (rolling z-score, agent feedback, raw or
unjustified encoding columns, unseen window slots, oversized extractors, undeclared regularization, non-admitted
demonstrators) and refuses to emit while a gate is closed. No data, no model fit."""
import copy
import hashlib
import json

import pytest

pytest.importorskip("torch")

from tools import expert_guided_clean_round as C  # noqa: E402


@pytest.fixture()
def spec(tmp_path):
    store = tmp_path / "store.parquet"
    store.write_bytes(b"store-bytes")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"features": ["a", "b", "c", "d"]}))
    lst = tmp_path / "FEATURE_LIST_x.json"
    lst.write_text(json.dumps({"features": ["a", "b", "c", "d"]}))
    grp = tmp_path / "FEATURE_GROUPS_x.json"
    grp.write_text(json.dumps({"groups": {"g1": ["a", "b"], "g2": ["c", "d"]}, "units": {"g1": 2, "g2": 2}}))
    demo = tmp_path / "DEMONSTRATORS.json"
    demo.write_text(json.dumps({"demonstrators": [{"spec": "3", "admitted": True}, {"spec": "2", "admitted": True},
                                                  {"spec": "rev10", "admitted": True}, {"spec": "rev5", "admitted": False},
                                                  {"spec": "6", "admitted": True}]}))
    gate = tmp_path / "RELEASE.json"
    s = {"identity": "t", "scaling": "fixed_train", "nonstationary": "diff", "split_contract": "consolidated", "exec_contract": "sac", "min_orders_per_week": 2.0,
         "store": {"path": str(store), "sha256": hashlib.sha256(b"store-bytes").hexdigest()},
         "encodings": {"adopted": False}, "model": {"units": 2, "core": None, "depth": 2, "head": 16, "max_epochs": 5},
         "limits": {"max_extractor_units": 4, "max_core_width": 16, "max_head": 32},
         "regularization": {"dropout": 0.1, "wd": 1e-4, "source": "test"},
         "demonstrators": {"file": str(demo)}, "memory": "4G", "remote_root": "/r",
         "gates": [{"path": str(gate)}],
         "stages": {"A": [{"name": "a1", "list": str(lst), "extractor": "per_family", "groups": str(grp), "window": [24, 1],
                           "agent_state": "none"},
                          {"name": "a2", "list": str(lst), "extractor": "per_feature", "window": [13, 2], "agent_state": "none"}],
                    "B": [{"name": "b1", "list": None, "extractor": None, "window": [24, 1], "agent_state": "minimal"},
                          {"name": "b2", "list": None, "extractor": None, "window": [24, 1], "agent_state": "none", "extra_experts": ["2", "rev10"],
                           "aux_targets": True}]}}
    return s, tmp_path


def test_valid_spec_passes(spec):
    s, _ = spec
    assert C.validate(s) == []


@pytest.mark.parametrize("mut, code", [
    (lambda s: s.update(scaling="rolling_z"), "SCALING_FORBIDDEN"),
    (lambda s: s.update(nonstationary="keep"), "NONSTATIONARY_POLICY_FORBIDDEN"),
    (lambda s: s.update(split_contract="dev3y"), "CONTRACT_NOT_PRINCIPAL"),
    (lambda s: s["regularization"].update(dropout=None), "REGULARIZATION_NOT_DECLARED"),
    (lambda s: s["stages"]["A"][0].update(agent_state="full"), "AGENT_STATE_FORBIDDEN"),
    (lambda s: [a.update(agent_state="minimal") for a in s["stages"]["A"]], "STAGE_WITHOUT_NO_FEEDBACK_ARM"),
    (lambda s: s["model"].update(depth=1), "RECEPTIVE_FIELD_BELOW_WINDOW"),
    (lambda s: s["model"].update(units=8), "MODEL_EXCEEDS_DECLARED_LIMITS"),
    (lambda s: s["store"].update(sha256="0" * 64), "STORE_DIGEST_MISMATCH"),
    (lambda s: s["stages"]["B"][1].update(extra_experts=["rev5"]), "DEMONSTRATOR_NOT_ADMITTED"),
    (lambda s: s["stages"]["B"][1].update(extra_experts=["6"]), "DEMONSTRATOR_NOT_SHORTER_THAN_OPTIMUM"),
    (lambda s: s.update(exec_contract="native"), "EXEC_CONTRACT_NOT_REPLAY"),
    (lambda s: s.update(min_orders_per_week=3.0), "MIN_ORDERS_PER_WEEK_NOT_OWNER_RULE"),
])
def test_each_forbidden_configuration_is_refused(spec, mut, code):
    s, _ = spec
    s = copy.deepcopy(s)
    mut(s)
    assert any(e.startswith(code) for e in C.validate(s)), C.validate(s)


def test_raw_and_unadopted_encoding_columns_are_refused(spec):
    s, tmp = spec
    for bad in ("raw.log_close", "enc.hod.oh3"):
        (tmp / "MANIFEST.json").write_text(json.dumps({"features": ["a", "b", "c", bad]}))
        (tmp / "FEATURE_LIST_x.json").write_text(json.dumps({"features": ["a", "b", "c", bad]}))
        (tmp / "FEATURE_GROUPS_x.json").write_text(json.dumps({"groups": {"g1": ["a", "b"], "g2": ["c", bad]}}))
        assert any(e.startswith("FORBIDDEN_COLUMNS") for e in C.validate(s))


def test_groups_must_partition_the_list_and_extractors_stay_small(spec):
    s, tmp = spec
    (tmp / "FEATURE_GROUPS_x.json").write_text(json.dumps({"groups": {"g1": ["a", "b"], "g2": ["c"]}}))
    assert any(e.startswith("GROUPS_DO_NOT_PARTITION_LIST") for e in C.validate(s))
    (tmp / "FEATURE_GROUPS_x.json").write_text(json.dumps({"groups": {"g1": ["a", "b"], "g2": ["c", "d"]}, "units": {"g1": 9}}))
    assert any(e.startswith("EXTRACTOR_TOO_LARGE") for e in C.validate(s))


def test_emit_is_refused_while_a_gate_is_closed_and_written_once_open(spec, tmp_path):
    s, tmp = spec
    sp = tmp / "spec.json"
    sp.write_text(json.dumps(s))
    out = tmp / "q.sh"
    a = type("A", (), {"spec": str(sp), "stage": "A", "winner_from": None, "out": str(out)})()
    with pytest.raises(SystemExit, match="GATES_CLOSED"):
        C.cmd_emit(a)
    (tmp / "RELEASE.json").write_text("{}")
    C.cmd_emit(a)
    q = out.read_text()
    assert "--scaling fixed_train" in q and "--agent-state none" in q and "--split-contract consolidated" in q
    assert "rolling" not in q and "--agent-state full" not in q and "test -f $Q/RELEASE.json" in q
    assert "--groups" in q and "--window 24 --stride 1" in q and "--depth 2" in q and "--exec-contract sac" in q and "--min-orders-per-week 2.0" in q


def test_parameter_counts_are_reported_per_module(spec):
    s, tmp = spec
    p = C.arm_params(s["stages"]["A"][0], s, ["a", "b", "c", "d"])
    assert set(p) == {"open_l", "open_s", "close_l", "close_s"} and all(v > 0 for v in p.values())
    assert C.receptive_field(2) == 31 and C.receptive_field(1) == 15
