"""Round results tool: frozen ranking, verification of recipe/state/support/probe, and the owner's closure table; built
from synthetic E3M dicts, no data."""
import json

from tools import expert_guided_round_results as R

REC = {"identity": "recipe_x", "sha256": "ab" * 32}


def e3m(inner, train, valid=True, labels=None, sup="s1", probe=(1, 2), rec=REC, state="none"):
    tr = {}
    for i, (ia, ta) in enumerate(zip(inner, train)):
        g = lambda s: {"sharpe_daily_sqrt365": s, "max_drawdown_pct": 5.0, "orders_per_week": 3.0, "win_rate": 0.5, "trades": 10, "valid": valid}  # noqa: E731
        tr[str(2026 + i)] = {"trading": {"TRAIN": g(ta), "INNER_A": g(ia), "OBS_2024": g(-1.0)}}
    return {"trials": {"FULL": tr}, "params": {"FULL": {"a": 1, "b": 2}}, "labels": labels or {"mode": "zigzag"}, "recipe": rec,
            "agent_state": {"mode": state}, "verify": True, "meta": {"support_hash": {"INNER_A": sup}, "rows_eligible": {}},
            "probe": {"FULL_open_l_2026": {"row_ts_ns": list(probe)}}, "wall_seconds": 1.0}


def spec(tmp_path, arms):
    s = {"decision_control": "ctl", "recipe": REC, "stages": {"A": [{"name": n} for n in arms]}}
    for n, e in arms.items():
        d = tmp_path / f"A_{n}"
        d.mkdir()
        (d / "E3M.json").write_text(json.dumps(e))
    return s


def test_ranking_follows_the_frozen_rule_and_ties_go_to_the_control(tmp_path):
    s = spec(tmp_path, {"ctl": e3m([1, 1, 1], [2, 2, 2]), "t1": e3m([1, 1, 1], [3, 3, 3]), "t2": e3m([2, 2, 2], [3, 3, 3], valid=False)})
    doc = R.build(s, tmp_path, ["A"], {"A_ctl": "6G"})
    r = doc["rankings"]["A"]
    assert r["chosen"] == "ctl" and r["order"][0] == "ctl"          # t2 invalid is excluded; tie t1/ctl -> control
    assert doc["arms"][0]["memory_cap"] == "6G" and doc["verification"]["all_ok"]


def test_verification_flags_every_deviation(tmp_path):
    bad_rec = {"identity": "recipe_y", "sha256": "cd" * 32}
    s = spec(tmp_path, {"ctl": e3m([1, 1, 1], [2, 2, 2]), "t1": e3m([0, 0, 0], [1, 1, 1], rec=bad_rec, state="minimal", sup="s2", probe=(9, 9))})
    v = R.build(s, tmp_path, ["A"], {})["verification"]
    assert not v["all_ok"]
    assert {"t1:recipe_identity_ok", "t1:recipe_sha_ok", "t1:agent_state_ok", "t1:support_hash", "t1:probe_rows"} <= set(v["deviations"])


def test_closure_rows_compare_every_treatment_to_the_control_on_the_same_rows(tmp_path):
    s = spec(tmp_path, {"ctl": e3m([1, 1, 1], [2, 2, 2]), "t1": e3m([2, 2, 2], [3, 3, 3])})
    doc = R.build(s, tmp_path, ["A"], {})
    rows = doc["closure"]
    assert len(rows) == 3 and {r["period"] for r in rows} == {"TRAIN", "INNER_A", "OBS_2024"}
    inner = next(r for r in rows if r["period"] == "INNER_A")
    assert inner["chosen"] and "delta Sharpe +1.00" in inner["skill"] and inner["literature"].startswith("NOT_COMPARABLE")
    assert "never decides" in next(r for r in rows if r["period"] == "OBS_2024")["comparability"]
    md = R.markdown(doc)
    assert "Closure table" in md and "t1 **(chosen)**" in md


def test_multires_arm_reports_labels_and_weights_flag(tmp_path):
    lab = {"mode": "multires", "thresholds": {"theta_f": 0.5}, "weights_in_loss": True, "profile": {"TRAIN": {"classes": {"nop": 3}}}}
    s = spec(tmp_path, {"ctl": e3m([1, 1, 1], [2, 2, 2]), "m": e3m([0, 0, 0], [1, 1, 1], labels=lab)})
    a = next(x for x in R.build(s, tmp_path, ["A"], {})["arms"] if x["arm"] == "m")
    assert a["labels_mode"] == "multires" and a["weights_in_loss"] is True and a["labels_profile"]["TRAIN"]["classes"]["nop"] == 3
