"""Protocol v11 (audit A5 / A1-1 / A1-7): nested contracts, pooled per-trade expectancy, paired two-sided tests with
multiplicity control, deflated Sharpe / PBO, the read ledger, the pre-registered sanity gate, seed CIs with a tie band
and the member-frozen ensemble. Known-answer and refusal tests; no data, no model."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided import data as DG
from app.short_horizon.expert_guided import ensemble_freeze as EF
from app.short_horizon.expert_guided import read_ledger as RL
from app.short_horizon.expert_guided import stats as ST


# ------------------------------------------------------------------------------------------------------- contracts
def test_nested_folds_exist_are_ordered_embargoed_and_before_test():
    assert len(DG.WALK_FORWARD_NESTED) >= 10
    for k, sc in DG.WALK_FORWARD_NESTED.items():
        assert DG.nested_contract_errors(sc) == [], k
        P = sc["periods"]
        assert (pd.Timestamp(P["INNER_B"][0]) - pd.Timestamp(P["INNER_SEL"][1])).days >= DG.EMBARGO_DAYS
        assert sc["early_stop"] == "INNER_ES" and sc["calibration"] == "INNER_A" and sc["selection"] == "INNER_SEL" and sc["decision"] == "INNER_B"
        assert sc["scaler_fit"] == P["TRAIN"] and sc["feature_list"] == "FEATURE_LIST_core19.json"
    assert "wf2022" in DG.SPLIT_CONTRACTS and "wf2023" in DG.SPLIT_CONTRACTS      # yearly folds kept for comparability
    quarters = [k for k in DG.WALK_FORWARD_NESTED if "q" in k]
    assert len(quarters) == 10 and quarters[0] == "wf2021q3" and quarters[-1] == "wf2023q4"


def test_nested_contract_refusals():
    sc = json.loads(json.dumps(DG.WALK_FORWARD_NESTED["wf2022q1"]))
    sc["periods"]["INNER_SEL"] = (sc["periods"]["INNER_SEL"][0], sc["periods"]["INNER_B"][0])
    assert "NESTED_EMBARGO_TOO_SHORT" in DG.nested_contract_errors(sc)
    sc = json.loads(json.dumps(DG.WALK_FORWARD_NESTED["wf2022q1"]))
    sc["scaler_fit"] = ("2020-01-01", "2022-06-01")
    assert "NESTED_SCALER_NOT_FIT_BLOCK" in DG.nested_contract_errors(sc)
    sc["fold_local_store_required"] = False
    assert "NESTED_FEATURE_LIST_OR_STORE_NOT_FIXED" in DG.nested_contract_errors(sc)
    with pytest.raises(ValueError, match="NESTED_DECISION_OVERLAPS_OBS_2024"):
        DG._nested_contract("2024-01-01", "2024-04-01")


def test_no_label_may_cross_a_decision_block_boundary_by_construction():
    """The loader purges every label whose support ends after its period end (existing rule); the embargo rows belong
    to no period. A label ending inside the embargo therefore belongs to INNER_SEL only if its support ends before
    INNER_SEL's end, i.e. strictly before the decision block."""
    sc = DG.WALK_FORWARD_NESTED["wf2022q2"]
    emb = sc["embargo"]
    assert pd.Timestamp(emb[0]) == pd.Timestamp(sc["periods"]["INNER_SEL"][1]) and pd.Timestamp(emb[1]) == pd.Timestamp(sc["periods"]["INNER_B"][0])
    spec = DG.nested_store_rebuild_spec()
    assert spec["status"].startswith("NOT_BUILT") and spec["folds"]["wf2022q2"]["producer_fit_window"] == sc["periods"]["TRAIN"]


# ------------------------------------------------------------------------------------------------------- statistics
def test_mde_known_answer_and_table():
    assert abs(ST.mde(1.0, 0.05, 0.8, True) - 2.8016) < 1e-3          # (1.96 + 0.8416) * 1
    assert abs(ST.mde(1.0, 0.05, 0.8, False) - 2.4865) < 1e-3
    t = ST.mde_table({"wf2022": 1.0, "wf2023": 1.0})
    assert abs(t["POOLED"]["se"] - 1 / math.sqrt(2)) < 1e-9


def test_day_block_bootstrap_expectancy_known_answer_and_empty():
    tb = {f"d{i}": [0.1, 0.1] for i in range(40)}
    e = ST.day_block_bootstrap_expectancy(tb, n=200)
    assert abs(e["expectancy"] - 0.1) < 1e-12 and abs(e["low"] - 0.1) < 1e-9 and e["n_trades"] == 80
    rng = np.random.default_rng(1)
    tb = {f"d{i}": list(rng.normal(0, 1, 3)) for i in range(60)}
    e = ST.day_block_bootstrap_expectancy(tb, n=300)
    assert e["low"] < e["expectancy"] < e["high"]
    assert math.isnan(ST.day_block_bootstrap_expectancy({"d": []}, n=10)["expectancy"])


def test_paired_tests_identical_series_give_zero_delta_and_p_one():
    r = np.random.default_rng(2).normal(0.0003, 0.01, 365)
    pb = ST.paired_block_bootstrap_diff(r, r, n=200)
    assert pb["delta"] == 0.0 and pb["p_two_sided"] == 1.0 and pb["sign"] == "equal"
    h = ST.hac_sharpe_diff_test(r, r)
    assert abs(h["delta"]) < 1e-12 and h["p_two_sided"] == 1.0
    with pytest.raises(ValueError, match="PAIRED_SERIES_LENGTH_MISMATCH"):
        ST.hac_sharpe_diff_test(r, r[:-1])


def test_paired_tests_detect_a_large_shift():
    rng = np.random.default_rng(3)
    a = rng.normal(0.004, 0.01, 500)
    b = a - 0.004 + rng.normal(0, 0.001, 500)
    assert ST.hac_sharpe_diff_test(a, b)["p_two_sided"] < 0.01
    assert ST.paired_block_bootstrap_diff(a, b, n=300)["p_two_sided"] < 0.05


def test_romano_wolf_is_monotone_bounded_and_controls_the_null():
    rng = np.random.default_rng(4)
    boot = rng.normal(0, 1, (2000, 5))
    stats = np.array([5.0, 0.1, 0.0, -0.2, 4.0])
    adj = ST.romano_wolf(stats, boot)
    assert np.all((adj >= 0) & (adj <= 1)) and adj[0] < 0.01 and adj[4] < 0.01 and adj[2] > 0.5
    order = np.argsort(-np.abs(stats))
    assert np.all(np.diff(adj[order]) >= -1e-12)                    # step-down monotone


def test_deflated_sharpe_decreases_with_trials_and_pbo_on_identical_trials():
    d1 = ST.deflated_sharpe(0.1, 1, 365)
    d20 = ST.deflated_sharpe(0.1, 20, 365)
    assert d1["dsr"] > d20["dsr"] and 0 <= d20["dsr"] <= 1 and d1["sr0"] == 0.0
    rng = np.random.default_rng(5)
    R = np.repeat(rng.normal(0, 0.01, (400, 1)), 4, axis=1) + rng.normal(0, 1e-6, (400, 4))
    p = ST.pbo(R, S=8, max_combos=40)
    assert 0.0 <= p["pbo"] <= 1.0 and p["combos"] == 40
    assert math.isnan(ST.pbo(R[:10], S=8)["pbo"])


def test_holds_v2_separable_v2_and_refusals():
    good = {"expectancy": 0.05, "low": 0.01, "high": 0.09, "n_trades": 300}
    assert ST.holds_v2(good)["verdict"] == "HOLDS"
    assert ST.holds_v2({**good, "low": -0.01})["verdict"] == "DOES_NOT_HOLD"
    bad = {"unit": "u", "sanity": "FAILED_SANITY", "sanity_causes": ["zero_trades"]}
    assert ST.holds_v2(good, inputs=(bad,))["verdict"] == ST.SANITY_REFUSAL
    assert ST.separable_v2(0.5, 0.01)["verdict"] == "SEPARABLE" and ST.separable_v2(-0.5, 0.2)["verdict"] == "NOT_SEPARABLE"
    assert ST.separable_v2(0.5, 0.01, inputs=(bad,))["verdict"] == ST.SANITY_REFUSAL
    assert ST.separable_v2(-0.5, 0.2)["sign"] == "negative"
    sf = ST.sign_flip({"a": 1.0, "b": -2.0, "c": 0.5})
    assert sf["positive"] == 2 and sf["negative"] == 1 and sf["stable_sign"] is False


def test_noise_floor_and_seed_ci_tie_band():
    assert ST.noise_floor_note(3.9)["evidence"] == "NOT_EVIDENCE_BELOW_NOISE_FLOOR" and ST.noise_floor_note(4.5)["evidence"] == "ABOVE_NOISE_FLOOR"
    ci = ST.seed_mean_ci([1.0, 2.0, 3.0])
    assert abs(ci["sd"] - 1.0) < 1e-12 and ci["ddof"] == 1 and ci["low"] < 2.0 < ci["high"]
    a, b = ST.seed_mean_ci([1.0, 2.0, 3.0]), ST.seed_mean_ci([1.5, 2.5, 3.5])
    assert ST.tie_band(a, b)["tied"] is True
    assert ST.tie_band(a, ST.seed_mean_ci([10.0, 11.0, 12.0]))["tied"] is False


# ------------------------------------------------------------------------------------------------------- ledger
def test_read_ledger_counts_and_development_label(tmp_path):
    p = tmp_path / RL.LEDGER_NAME
    assert RL.status_table(p) == {}
    RL.record(p, "wf2022q1/INNER_B", "u1", "ensemble", "sharpe_lcb", "report", when="2026-10-26T00:00:00Z")
    assert RL.status_table(p)["wf2022q1/INNER_B"] == {"reads": 1, "status": RL.STATUS_HELD_OUT}
    RL.record(p, "wf2022q1/INNER_B", "u1", "seed 1", "sharpe_lcb", "audit")
    assert RL.status_table(p)["wf2022q1/INNER_B"]["status"] == RL.STATUS_DEVELOPMENT
    assert RL.block_status(0) == RL.STATUS_UNREAD
    with pytest.raises(ValueError, match="LEDGER_BLOCK_MALFORMED"):
        RL.record(p, "INNER_B", "u", "p", "v", "w")
    md = "\n".join(RL.markdown(p))
    assert "DEVELOPMENT" in md and "| wf2022q1/INNER_B | 2 |" in md


# ------------------------------------------------------------------------------------------------------- ensemble
def test_ensemble_strategy_is_an_actual_member_closest_to_the_median():
    members = {"20261019": {"genes": {"q_L": 0.05, "q_S": 0.06, "allow_reverse": True}, "nop_veto": False},
               "20261020": {"genes": {"q_L": 0.20, "q_S": 0.21, "allow_reverse": False}, "nop_veto": True},
               "20261021": {"genes": {"q_L": 0.10, "q_S": 0.11, "allow_reverse": True}, "nop_veto": False}}
    r = EF.strategy_from_members(members)
    assert r["seed"] == "20261021" and r["strategy"] is members["20261021"] and "no re-search" in r["rule"]
    with pytest.raises(ValueError, match="ENSEMBLE_GENE_NAMES_DIFFER"):
        EF.strategy_from_members({"1": {"genes": {"a": 1}}, "2": {"genes": {"b": 1}}})
    with pytest.raises(ValueError, match="ENSEMBLE_NO_MEMBERS"):
        EF.strategy_from_members({})


# ------------------------------------------------------------------------------------------------------- sanity gate
def test_sanity_gate_activity_counts_only_on_the_calibration_block(tmp_path):
    from tools import expert_guided_sanity_gate as G
    win = lambda opw: {"trades": 50, "orders_per_week": opw, "sharpe_daily_sqrt365": 1.0, "n_close_TP": 10, "n_close_SL": 10, "n_close_AGENT_CLOSE": 30,  # noqa: E731
                       "n_entries_long": 25, "n_entries_short": 25}
    u = tmp_path / "X_unit"; u.mkdir()
    (u / "E3M.json").write_text(json.dumps({"trials": {"FULL": {"1": {"trading": {"INNER_A": win(3.0), "INNER_B": win(1.5)}}}}}))
    d = G.gate_unit(u)
    assert d["status"] == "PASS" and d["decision_block_activity_invalid"] == [{"where": "seed 1", "block": "INNER_B"}]
    (u / "E3M.json").write_text(json.dumps({"trials": {"FULL": {"1": {"trading": {"INNER_A": win(1.5), "INNER_B": win(3.0)}}}}}))
    d = G.gate_unit(u)
    assert d["status"] == "FAILED_SANITY" and d["causes"][0]["cause"] == "below_2_trades_per_week" and d["causes"][0]["block"] == "INNER_A"
    assert d["activity_rule_block"] == "INNER_A" and d["preregistered_in"]


def test_spec_v11_preregisters_the_rules_verbatim():
    d = json.load(open("docs/short_horizon/clean_round/CLEAN_ROUND_SPEC.json"))
    assert d["identity"].endswith("-v11")
    pr = d["evaluation_protocol_v11"]
    assert pr["rules_verbatim"]["HOLDS"] == ST.HOLDS_RULE_V2 and pr["rules_verbatim"]["SEPARABLE"] == ST.SEPARABLE_RULE_V2
    assert pr["rules_verbatim"]["ensemble_strategy"] == EF.RULE and "below_2_trades_per_week is a cause ONLY on the calibration block" in pr["rules_verbatim"]["sanity_gate_v11"]
    assert set(pr["nested_contracts"]) == set(DG.WALK_FORWARD_NESTED) and pr["fold_local_store"]["status"].startswith("NOT_BUILT")
    assert d["revisions"][-1]["identity"] == d["identity"]
