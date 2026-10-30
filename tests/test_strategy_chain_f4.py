"""Strategy chain F4 (owner order 2026-10-29): the decision rules fixed before the chain runs -- the S1R edge rule, the
S3 re-run rule -- and the D9 instruments (cluster bootstrap, Holm, trace equality, counterfactual pooling), the size = 1
sizing mode and the single-horizon ablation mask."""
import json
import types

import numpy as np
import pytest

from app.short_horizon.expert_guided import exec_variants as XV
from app.short_horizon.expert_guided import strategy_params as SP
from tools import strategy_behaviour_matrix as BM
from tools import strategy_chain_f4 as CH


def pt(**k):
    d = {"a_sl": 1.0, "a_tp": 1.0, "b_sl": 0.0, "b_tp": 0.0, "buffer_pips": 1.0, "ref_p75": 0.0}
    d.update(k)
    return d


def test_edge_rule_interior_point_extends_only_the_two_valued_axes_and_records_sealed_bounds():
    r = CH.edge_rule(pt())
    axes = {e["axis"]: e for e in r["extensions"]}
    hits = {h["axis"] for h in r["boundary_hits"]}
    assert "a_sl" not in axes and "a_tp" not in axes                 # 1.0 is interior on {0.75, 1, 1.5}
    assert hits == {"b_sl", "b_tp"}                                   # 0 is the sealed lower bound of b_sl / b_tp: BOUNDARY_HIT
    assert axes["buffer_pips"]["to"] == 0.0 and axes["buffer_pips"]["clipped_to_bound"]   # 1 - 2 = -1 clipped to the sealed 0
    assert all(e["axis"] != "ref_p75" for e in r["edges"])           # categorical: never an edge


def test_edge_rule_outer_edges_step_one_grid_step_within_the_sealed_bounds():
    r = CH.edge_rule(pt(a_sl=0.75, a_tp=1.5, b_sl=0.5, b_tp=0.5, buffer_pips=3.0))
    to = {e["axis"]: e["to"] for e in r["extensions"]}
    assert to == {"a_sl": 0.5, "a_tp": 2.0, "b_sl": 1.0, "b_tp": 1.0, "buffer_pips": 5.0}
    for p in r["extension_points"]:                                  # one axis moved per extension point, each a valid full_f4 document
        assert sum(1 for k in p if p[k] != pt(a_sl=0.75, a_tp=1.5, b_sl=0.5, b_tp=0.5, buffer_pips=3.0)[k]) == 1
        SP.validate({**SP.default_doc(), **p})
    b = SP.bounds_of("full_f4")
    r2 = CH.edge_rule(pt(a_sl=0.75), bounds={**b, "a_sl": [0.75, 2.0]})   # a sealed bound AT the edge value: BOUNDARY_HIT, no extension
    assert "a_sl" in {h["axis"] for h in r2["boundary_hits"]} and "a_sl" not in {e["axis"] for e in r2["extensions"]}


def test_deb_sort_excludes_infeasible_and_ranks_constraint_first():
    rows = [{"n_failing_witnesses": 0, "fitness_sharpe": 5.0}, {"n_failing_witnesses": 1, "fitness_sharpe": 30.0},
            {"n_failing_witnesses": 0, "fitness_sharpe": 9.0}, {"n_failing_witnesses": 0, "fitness_sharpe": 99.0, "infeasible": True}]
    assert [r["fitness_sharpe"] for r in CH.deb_sort(rows)] == [9.0, 5.0, 30.0]


def test_s3_rule_reruns_once_from_the_best_winner_only():
    rows = [{"index": 0, "params": pt(), "paired_mean_dsharpe_vs_frozen": 0.0, "se": 0.1, "wins_by_more_than_1se": False},
            {"index": 1, "params": pt(a_sl=0.75), "paired_mean_dsharpe_vs_frozen": 0.5, "se": 0.1, "wins_by_more_than_1se": True},
            {"index": 2, "params": pt(a_sl=1.5), "paired_mean_dsharpe_vs_frozen": 0.8, "se": 0.2, "wins_by_more_than_1se": True},
            {"index": 3, "params": pt(a_tp=0.75), "paired_mean_dsharpe_vs_frozen": 0.05, "se": 0.1, "wins_by_more_than_1se": False}]
    d = CH.s3_decision({"rows": rows})
    assert d["rerun_stage2"] and d["winner"]["index"] == 2
    assert CH.s3_decision({"rows": [dict(r, wins_by_more_than_1se=False) for r in rows]}) == {"rerun_stage2": False, "winner": None}


def test_grid_compare_marks_a_win_only_beyond_one_se(tmp_path):
    from tools import strategy_level_grid as G
    pts = [pt(), pt(a_sl=0.75)]
    w = lambda k, sh: {"key": k, "sharpe": sh, "fails": [], "quality": 1.0, "activity": {"trades_per_week": 1.0, "reopen_1bar_share": 0.0}, "net_sigma_per_trade_median": 0.3, "burst_gate": None, "w17": None}  # noqa: E731
    tasks = []
    for i, gain in ((0, 0.0), (1, 0.5)):
        tasks.append({"point": i, "group": "g", "n_worlds": 4, "error": None, "per_world": [w(f"x{j}", 10.0 + j + gain + (0.01 * j if i else 0)) for j in range(4)]})
    (tmp_path / "RUN_META.json").write_text(json.dumps({"groups": 1, "points": 2, "point_list": pts}))
    (tmp_path / "TASKS.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    G.cmd_compare(types.SimpleNamespace(out=str(tmp_path), frozen=0))
    r = json.loads((tmp_path / "STAGE3_RESULT.json").read_text())
    assert r["rerun_stage2"] and r["rows"][1]["wins_by_more_than_1se"] and not r["rows"][0]["wins_by_more_than_1se"]


def test_cluster_bootstrap_and_holm():
    b = CH.cluster_bootstrap(np.full(40, 1.0) + np.linspace(-0.1, 0.1, 40))
    assert b["n"] == 40 and b["ci95"][0] > 0 and b["p"] == 0.0
    z = CH.cluster_bootstrap(np.linspace(-1, 1, 41))
    assert z["ci95"][0] < 0 < z["ci95"][1] and z["p"] > 0.5
    assert CH.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(CH.holm([float("nan"), 0.02])[0]) and CH.holm([float("nan"), 0.02])[1] == pytest.approx(0.04)


def test_trace_equality_flags_any_decision_difference():
    t = lambda e, x, d, r="TP": {"entry_bar_abs": e, "exit_bar_abs": x, "direction": d, "exit_reason": r}  # noqa: E731
    a = {("A", 0): {"trades": [t(1, 3, 1), t(5, 7, -1)]}, ("B", 0): {"trades": [t(2, 4, 1)]}}
    assert CH.trace_equality(a, {k: {"trades": list(v["trades"])} for k, v in a.items()})["verdict"] == "PASS"
    b = {("A", 0): {"trades": [t(1, 3, 1), t(5, 8, -1)]}, ("B", 0): {"trades": [t(2, 4, 1)]}}
    r = CH.trace_equality(a, b)
    assert r["verdict"] == "FAIL" and r["different"][0]["first_difference_index"] == 1


def test_counterfactual_pool_weights_by_early_closes():
    recs = {("A", 0): {"exit_counterfactual": {"BUY": {"early_closes": 10, "destructive_share": 0.1, "band_share": 0.5, "regret_vs_Ostar_cash_risk_mean": 0.2}, "SELL": {"early_closes": 0}}},
            ("B", 0): {"exit_counterfactual": {"BUY": {"early_closes": 30, "destructive_share": 0.5, "band_share": 0.1, "regret_vs_Ostar_cash_risk_mean": 0.4}, "SELL": {"early_closes": 0}}}}
    p = CH.counterfactual_pool(recs)
    assert p["BUY"]["early_closes"] == 40 and p["BUY"]["destructive_share"] == pytest.approx(0.4) and p["BUY"]["regret_vs_Ostar_cash_risk_mean"] == pytest.approx(0.35)
    assert p["SELL"]["early_closes"] == 0


def test_exit_counterfactual_classifies_each_early_close_against_holding():
    class Exact:
        def path(self, pos):
            return {"hold_net": pos.hold, "P": np.array([pos.best])}
    mk = lambda hold, best: types.SimpleNamespace(units=10, sigma=0.01, entry_price=1.0, sl=0.99, hold=hold, best=best)  # noqa: E731
    roll = {"trades": [{"entry_bar_abs": 1, "direction": 1, "exit_reason": "EARLY", "net_usd": 0.0},
                       {"entry_bar_abs": 2, "direction": 1, "exit_reason": "EARLY", "net_usd": 0.0},
                       {"entry_bar_abs": 3, "direction": 1, "exit_reason": "EARLY", "net_usd": 0.0},
                       {"entry_bar_abs": 4, "direction": 1, "exit_reason": "TP", "net_usd": 1.0}],
            "positions": {(1, 0): mk(0.01, 0.01), (2, 0): mk(-0.01, 0.0), (3, 0): mk(0.0001, 0.0)}}
    r = BM.exit_counterfactual({}, roll, Exact())
    b = r["BUY"]                                                      # d = (hold - close) / units / sigma = 0.1 (destructive), -0.1 (beneficial), 0.001 (band)
    assert b["early_closes"] == 3 and b["destructive_share"] == pytest.approx(1 / 3) and b["beneficial_share"] == pytest.approx(1 / 3) and b["band_share"] == pytest.approx(1 / 3)
    assert b["regret_vs_Ostar_cash_risk_mean"] == pytest.approx(np.mean([0.1, 0.0, 0.001]))   # (P* - net) / (|entry - SL| x units = 0.1)
    assert r["SELL"]["early_closes"] == 0


def test_unit_sizing_mode_is_declared_and_leaves_existing_identities_unchanged():
    assert "unit" in XV.SIZING_MODES
    from app.short_horizon.expert_guided import c3 as C3
    assert C3.FULL_F3.identity == "exec_709f9b1fa3512486" and C3.FULL_F4.identity == "exec_342c820a6b4bf1c6"
    import dataclasses
    u = dataclasses.replace(C3.FULL_F4, sizing_mode="unit")
    assert u.identity != C3.FULL_F4.identity


def test_level_cells_are_the_quality_below_one_worlds_without_pairs():
    sub = [{"subset_key": "A@q1.00", "id": "A", "quality": 1.0}, {"subset_key": "A@q0.70", "id": "A", "quality": 0.7},
           {"subset_key": "S_X@q1.00", "id": "S_X", "quality": 1.0, "pair": "X"}]
    cs = CH.level_cells(sub)
    assert [c["id"] for c in cs] == ["A"] and "subset_key" not in cs[0]


def test_mapped_head_delegates_the_wrapped_heads_attributes():
    from tools import strategy_qualification as Q
    bc = types.SimpleNamespace(S=1, T=2, n=10, exact=True, row=lambda b, si: (b, si))
    m = Q.MappedHead(bc, bar_map=lambda b: b + 1, swap=True)
    assert m.exact is True and m.n == 10 and m.row(3, 0) == (4, 1)
    with pytest.raises(AttributeError):
        _ = Q.MappedHead.__new__(Q.MappedHead).bc
