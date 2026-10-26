"""Stage C3 (2026-10-26): the audit register's label / target fixes. Contract sealed and validated; the stop-loss head is
refused; openers are 3-class with TIME its own class; the grid sits under the engine's effective cap and reaches
below 0.75 sigma; the reward/risk floor and the EV open rule with known answers; temperature calibration with a known
answer; per-side close thresholds; constant regression targets refused; contradictory closer keys deduplicated;
round-0 exploration never calls the expert; the early-close benefit decision; the loss-balance kwargs leave the
default fit path byte-identical. No data, no model fit."""
import copy
import json
import types
from pathlib import Path

import numpy as np
import pytest

from app.short_horizon.expert_guided import action_contract as ACT
from app.short_horizon.expert_guided import c2 as C2
from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "docs/short_horizon/clean_round/contracts"


# ----------------------------------------------------------------------------- contract
def test_committed_c3_contract_is_the_sealed_default_and_passes_every_check():
    d = json.loads((CONTRACTS / "ACTION_CONTRACT_C3.json").read_text())
    assert d == C3.default_contract() and d["architecture"]["trunk"] == "residual_tcn" and C3.validate(d) == [] and C2.encoding_audit(d)["verdict"] == "PASS"
    assert C3.is_c3(d) and not C2.is_c2(d) and d["identity"] == ACT.identity_of(d)
    for m in C3.OPENERS:
        assert d["specialists"][m]["decision"]["classes"] == ["TP_FIRST", "SL_FIRST", "TIME"]
        assert "stop_loss_head" in d["specialists"][m] and d["specialists"][m]["stop_loss_head"].startswith("none")
        assert list(d["specialists"][m]["continuous"]) == [C2.cell_name(c) for c in C3.CELLS]
    assert d["label_rules"]["teacher"] == "none" and d["constraints"]["teacher"] == "none"


@pytest.mark.parametrize("mutate, code", [
    (lambda d: d["specialists"]["open_l"]["continuous"].update({"sl_sigma": {"encoding": "bounded_sigmoid", "lo": 0.5, "hi": 3.0, "output_dim": 1, "loss": "huber", "role": "stop_loss"}}), "C3_SL_HEAD_FORBIDDEN"),
    (lambda d: d["specialists"]["open_l"]["decision"].update(classes=["NOT_TP_FIRST", "TP_FIRST"], n_classes=2, output_dim=2), "C3_CONTRACT_OPENER_CLASSES"),
    (lambda d: d["outcomes"].update(cells=[[tp, sl] for tp in (0.5, 1.0, 1.5, 2.0, 3.0) for sl in C3.GRID_SL]), "C3_CONTRACT_OUTCOMES"),
    (lambda d: d["specialists"]["open_s"].update(state_slots=["dist_sl_sigma"]), "GEOMETRY_ON_NON_CLOSER"),
    (lambda d: d["specialists"]["close_l"].update(state_slots=list(C3.GEOMETRY) + ["balance"]), "STATE_SLOT_NOT_ALLOWED"),
    (lambda d: d["strategy"].update(require_early_close=False), "C3_CONTRACT_EARLY_CLOSE_RULE"),
    (lambda d: d["strategy"]["ev_rule"]["rr_floor"].update(rr_margin_sigma=0.0), "C3_CONTRACT_EV_RULE"),
    (lambda d: d.update(loss_balance={"ce_weight": 1.0}), "C3_CONTRACT_LOSS_BALANCE"),
    (lambda d: d["label_rules"].update(closers="expert trajectory + exploration"), "C3_CONTRACT_LABEL_RULES"),
    (lambda d: d["strategy"].update(genes=dict(C2.Q_BOUNDS_13)), "C3_CONTRACT_GENES"),
    (lambda d: d.update(schema=C2.SCHEMA_G), "C3_CONTRACT_SCHEMA"),
])
def test_each_mutation_is_refused(mutate, code):
    d = copy.deepcopy(C3.default_contract())
    mutate(d)
    d = ACT.seal(dict(d, identity=None))
    assert any(e.startswith(code) for e in C3.validate(d)), C3.validate(d)


def test_grid_is_under_the_effective_cap_and_reaches_below_0_75_sigma():
    assert max(max(c) for c in C3.CELLS) <= C3.CAP_P1_SIGMA
    assert min(C3.GRID_SL) == 0.5 and min(C3.GRID_TP) == 0.5
    assert all(any(tp >= sl for (tp, sl) in C3.CELLS if sl == s) for s in C3.GRID_SL)      # an RR >= 1 cell for every SL


# ----------------------------------------------------------------------------- the EV open rule
def test_reward_risk_floor_allows_tp_ge_sl_and_refuses_tiny_tp_unless_the_net_clears_cost():
    cells = [(0.5, 2.0), (1.0, 1.0), (2.0, 0.5)]
    net = np.array([0.05, -0.1, 0.0])
    ok = C3.allowed_cells(net, cells, cost=0.1)
    assert ok.tolist() == [False, True, True]                  # (0.5, 2.0) only with net > cost + margin = 0.2
    ok2 = C3.allowed_cells(np.array([0.25, -0.1, 0.0]), cells, cost=0.1)
    assert ok2.tolist() == [True, True, True]


def test_ev_rule_known_answer_and_no_trade_when_no_cell_is_allowed():
    cells = [(1.0, 1.0), (0.5, 2.0)]
    p3 = np.array([0.6, 0.3, 0.1])
    ci, ev = C3.best_cell_ev(np.array([0.2, 0.9]), p3, cells, e_time=-0.2, cost=0.05)
    # (0.5, 2.0) allowed only with net > 0.15: 0.9 qualifies and has the higher net -> chosen; EV = 0.6*0.5 - 0.3*2.0 + 0.1*(-0.2) - 0.05
    assert ci == 1 and abs(ev - (0.3 - 0.6 - 0.02 - 0.05)) < 1e-12
    ci, ev = C3.best_cell_ev(np.array([0.2, 0.1]), p3, cells, e_time=-0.2, cost=0.05)
    assert ci == 0 and abs(ev - (0.6 - 0.3 - 0.02 - 0.05)) < 1e-12
    ci, ev = C3.best_cell_ev(np.array([np.nan, 0.1]), p3, cells, -0.2, 0.05)
    assert ci == -1 and np.isnan(ev)
    only_bad = [(0.5, 2.0)]
    ci, ev = C3.best_cell_ev(np.array([0.1]), p3, only_bad, -0.2, 0.05)
    assert ci == -1                                            # "no trade" is a valid answer


def _table(n=4, cells=((1.0, 1.0), (0.5, 0.5))):
    t = {"net": np.zeros((n, 2, len(cells))), "p3": np.tile(np.array([0.5, 0.3, 0.2]), (n, 2, 1)), "pn": np.zeros(n)}
    return t


def test_policy_opens_the_side_with_the_higher_ev_and_respects_per_side_margins_and_veto():
    cells = [(1.0, 1.0), (0.5, 0.5)]
    t = _table(cells=cells)
    t["net"][0, 0, :] = [0.3, 0.1]
    t["net"][0, 1, :] = [0.4, 0.2]
    t["p3"][0, 1, :] = [0.8, 0.1, 0.1]                           # short EV 0.8 - 0.1 - 0.02 - 0.05 = 0.63; long 0.5 - 0.3 - 0.02 - 0.05 = 0.13
    s = C3.StrategyC3(0.1, 0.1, 0.0, 0.0, 1.0, 1.0, False, None)
    pol = C3.PolicyC3(t, s, cells, (-0.2, -0.2), 0.05)
    assert pol.open_decision(0) == (E.SHORT, 1.0, 1.0)
    s2 = C3.StrategyC3(0.1, 0.7, 0.0, 0.0, 1.0, 1.0, False, None)   # the short margin above its EV: the long opens
    assert C3.PolicyC3(t, s2, cells, (-0.2, -0.2), 0.05).open_decision(0) == (E.LONG, 1.0, 1.0)
    t["pn"][0] = 0.9
    s3 = C3.StrategyC3(0.1, 0.1, 0.0, 0.0, 1.0, 1.0, False, 0.5)
    assert C3.PolicyC3(t, s3, cells, (-0.2, -0.2), 0.05).open_decision(0) is None


class _Closer:
    def __init__(self, rem, psl):
        self.rem, self.psl = rem, psl

    def value(self, r, bar, pos):
        return self.rem, self.psl


def test_closers_use_their_own_side_threshold_and_the_early_close_switch():
    t = _table()
    t["closer"] = _Closer(rem=-0.5, psl=0.3)
    s = C3.StrategyC3(9, 9, 0.2, 9.0, 1.0, 1.0, False, None)         # long closes on remaining < -0.2; short never
    pol = C3.PolicyC3(t, s, [(1.0, 1.0), (0.5, 0.5)], (0, 0), 0.0)
    assert pol.close_now(0, 0, 0, None) is True and pol.close_now(0, 1, 0, None) is False
    off = s.no_early_close()
    assert off.early_close is False and C3.PolicyC3(t, off, [(1.0, 1.0), (0.5, 0.5)], (0, 0), 0.0).close_now(0, 0, 0, None) is False
    assert C3.strategy_from_dict(off.to_dict()) == off and C2.strategy_from_dict(off.to_dict()) == off


def test_gene_map_and_strategy_from_genes_are_per_side():
    rows = np.arange(6)
    cells = [(1.0, 1.0)]
    view = {"net": np.zeros((6, 2, 1)), "p3": np.tile([0.6, 0.3, 0.1], (6, 2, 1)), "pn": np.linspace(0.1, 0.6, 6)}
    view["net"][:, 0, 0] = np.linspace(0.1, 0.6, 6)
    view["net"][:, 1, 0] = 0.2
    ref = {"rem_ref_l": np.linspace(-1, 1, 5), "rem_ref_s": np.linspace(-2, 2, 5), "psl_ref_l": np.linspace(0.1, 0.5, 5), "psl_ref_s": np.linspace(0.5, 0.9, 5)}
    gm = C3.gene_map(view, rows, ref, cells, (-0.1, -0.1), 0.05)
    g = gm.to_absolute({"open_rate_l": 0.5, "open_rate_s": 0.5, "close_rem_rate_l": 0.0, "close_rem_rate_s": 0.0, "close_p_rate_l": 0.0,
                        "close_p_rate_s": 0.0, "allow_reverse": 1.0, "nop_veto_rate": 0.5})
    st = C3.StrategyC3.from_genes(g, True)
    assert st.close_margin_l == 1.0 and st.close_margin_s == 2.0 and st.close_p_l == 0.5 and st.close_p_s == 0.9 and st.allow_reverse
    assert set(gm.rules) == set(C3.GENES) and st.early_close is True


# ----------------------------------------------------------------------------- calibration
def test_temperature_scaling_recovers_a_known_temperature_and_keeps_the_argmax():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (4000, 3))
    y = np.array([rng.choice(3, p=C3.softmax(zi / 2.0)) for zi in z])      # the data were generated at T = 2
    T = C3.fit_temperature(z, y)
    assert 1.5 <= T <= 2.7
    assert np.array_equal(np.argmax(C3.softmax(z, T), 1), np.argmax(z, 1))
    m = C3.multiclass_metrics(C3.softmax(z, T), y)
    assert m["n"] == 4000 and 0 <= m["tp_first"]["ece"] <= 1 and m["brier_multiclass"] < 1.0


# ----------------------------------------------------------------------------- targets, dedup, exploration
def test_constant_regression_target_is_refused_and_varying_targets_pass():
    D = {"open_l": (None, np.zeros(10, int), np.c_[np.full(10, 2.0), np.linspace(-1, 1, 10)], None)}
    with pytest.raises(SystemExit, match="C3_CONSTANT_TARGET"):
        C3.check_target_std(D, cells=[(1.0, 1.0), (0.5, 0.5)])
    D2 = {"open_l": (None, np.zeros(10, int), np.c_[np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], None)}
    assert len(C3.check_target_std(D2, cells=[(1.0, 1.0), (0.5, 0.5)])["open_l"]) == 2


def test_contradictory_closer_keys_are_dropped_and_duplicates_merged():
    rows = np.array([5, 5, 5, 7, 7, 9])
    y = np.array([1, 0, 1, 1, 0, 0])
    R = np.array([[0.2], [0.4], [0.6], [1.0], [-1.0], [0.5]], np.float32)
    G = np.array([[0.1, 0.2, 0.3, 1.0]] * 3 + [[0.5, 0.5, 0.5, -1.0]] * 2 + [[0.0, 0.0, 0.0, 1.0]], np.float32)
    r2, y2, R2, G2, dropped, merged = C3.dedup_closer(rows, y, R, G)
    assert dropped == 1 and merged == 2                                   # row 7 is a tie: dropped; row 5 majority 1, mean 0.4
    assert r2.tolist() == [5, 9] and y2.tolist() == [1, 0] and abs(float(R2[0, 0]) - 0.4) < 1e-6


def test_round0_exploration_never_calls_the_expert_and_draws_cells_from_the_grid():
    cells = [(0.5, 0.5), (2.0, 1.0)]
    calls = []

    def boom(*a, **k):
        raise AssertionError("the expert must never be consulted in C3 round 0")
    cx = types.SimpleNamespace(expert=boom)
    f = C3.prior_random_cells(cx, [0.5, 0.5, 0.0, 0.0], seed=1, cells=cells)
    for _ in range(50):
        a, tp, sl = f(0, 0, np.zeros(9), None)
        calls.append((tp, sl))
        assert a in (0, 1)
    assert set(calls) <= set(cells) and len(set(calls)) == 2


# ----------------------------------------------------------------------------- the early-close benefit decision
def test_early_close_benefit_lower_bound_is_positive_only_for_a_real_improvement():
    rng = np.random.default_rng(3)
    good = rng.normal(5.0, 1.0, 120)
    lb = C3.mean_diff_lower_bound(good, 5.0, 300, 1, 0.9)
    assert lb > 0
    noise = rng.normal(0.0, 10.0, 120)
    assert C3.mean_diff_lower_bound(noise, 5.0, 300, 1, 0.9) <= 0
    assert np.isnan(C3.mean_diff_lower_bound(np.zeros(3), 5.0, 10, 1, 0.9))


def test_daily_pnl_aggregates_by_exit_date():
    import pandas as pd
    t = [{"exit_time": pd.Timestamp("2022-03-01 10:00", tz="UTC"), "net_usd": 10.0}, {"exit_time": pd.Timestamp("2022-03-01 15:00", tz="UTC"), "net_usd": -4.0},
         {"exit_time": pd.Timestamp("2022-03-02 10:00", tz="UTC"), "net_usd": 1.0}]
    assert C3.daily_pnl(t) == {"2022-03-01": 6.0, "2022-03-02": 1.0}


# ----------------------------------------------------------------------------- loss balance
def test_loss_balance_kwargs_default_to_the_unchanged_fit_path():
    pytest.importorskip("torch")
    from app.short_horizon.expert_guided import learn as L
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    R = X[:, :2].astype(np.float32)
    Rm = np.ones_like(R)
    Rm[::2] = 0.0                                    # half the rows carry no regression target: masked != rows normalisation
    m1, _, i1 = L.fit(X, y, X, y, seed=1, n_cls=2, R=R, Rm=Rm, Ri=R, Rmi=Rm, hidden=(8,), max_epochs=3)
    m2, _, i2 = L.fit(X, y, X, y, seed=1, n_cls=2, R=R, Rm=Rm, Ri=R, Rmi=Rm, hidden=(8,), max_epochs=3, reg_norm="rows", reg_weight=1.0)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert np.array_equal(a.numpy(), b.numpy())
    m3, _, i3 = L.fit(X, y, X, y, seed=1, n_cls=2, R=R, Rm=Rm, Ri=R, Rmi=Rm, hidden=(8,), max_epochs=3, reg_norm="masked", reg_weight=1.0)
    assert i3["reg_norm"] == "masked" and i1["reg_norm"] == "rows"
    assert any(not np.array_equal(a.numpy(), b.numpy()) for a, b in zip(m1.state_dict().values(), m3.state_dict().values()))


def test_modules_and_bounds_for_c3():
    m = C3.modules()
    assert m["open_l"] == (3, len(C3.CELLS)) and m["close_l"] == (2, 1) and m["nop"] == (2, 0)
    assert C3.reg_bounds("open_l") == tuple(C3.NET_BOUND for _ in C3.CELLS) and C3.reg_bounds("close_s") == (C3.REM_BOUND,)


# ----------------------------------------------------------------------------- architecture inside the identity, protocol v11 hooks
def test_two_sealed_variants_carry_the_architecture_and_differ_only_by_it():
    t = json.loads((CONTRACTS / "ACTION_CONTRACT_C3.json").read_text())          # DEFAULT = residual_tcn (owner order)
    u = json.loads((CONTRACTS / "ACTION_CONTRACT_C3_U.json").read_text())        # the u-trunk control arm
    assert t == C3.default_contract() == C3.default_contract(C3.ARCH_TCN) and u == C3.default_contract(C3.ARCH_U)
    assert C3.validate(u) == [] and C3.validate(t) == [] and u["identity"] != t["identity"]
    assert u["architecture"]["trunk"] == "u" and t["architecture"]["trunk"] == "residual_tcn"
    diff = {k for k in u if u[k] != t[k]}
    assert diff == {"architecture", "identity"}


def test_architecture_mismatch_between_contract_and_run_is_refused():
    u = C3.default_contract(C3.ARCH_U)
    C3.check_architecture(u, {"units": 2})                                   # u defaults
    with pytest.raises(SystemExit, match="C3_ARCHITECTURE_MISMATCH"):
        C3.check_architecture(u, {"trunk": "residual_tcn", "fusion_units": 8, "core_blocks": 2, "readout": "flatten"})
    t = C3.default_contract(C3.ARCH_TCN)
    C3.check_architecture(t, {"trunk": "residual_tcn", "fusion_units": 8, "core_blocks": 2, "readout": "flatten"})
    with pytest.raises(SystemExit, match="C3_ARCHITECTURE_MISMATCH"):
        C3.check_architecture(t, {})


def test_blocks_fold_to_inner_a_for_the_yearly_contracts_and_name_four_for_nested():
    from app.short_horizon.expert_guided import data as DG
    b = C3.blocks_of(DG.split_contract("wf2022"))
    assert b == {"early_stop": "INNER_A", "calibration": "INNER_A", "selection": "INNER_A", "decision": "INNER_B"}
    nested = [k for k, v in DG.SPLIT_CONTRACTS.items() if isinstance(v, dict) and v.get("nested")]
    if nested:
        bn = C3.blocks_of(DG.split_contract(nested[0]))
        assert bn == {"early_stop": "INNER_ES", "calibration": "INNER_A", "selection": "INNER_SEL", "decision": "INNER_B"}


def test_launcher_refuses_a_nested_fold_unit_bound_to_the_consolidated_store():
    from tools import expert_guided_clean_round as CR
    nested = {"fold_local_store_required": True}
    rec_consolidated = {"store": {"sha256": "abc"}}
    assert CR.nested_store_error(nested, rec_consolidated, "abc", "WF/x").startswith("NESTED_FOLD_STORE_NOT_FOLD_LOCAL")
    rec_declared_but_same = {"store": {"sha256": "abc"}, "fold_local_store": True}
    assert CR.nested_store_error(nested, rec_declared_but_same, "abc", "WF/x") is not None
    rec_ok = {"store": {"sha256": "def"}, "fold_local_store": True}
    assert CR.nested_store_error(nested, rec_ok, "abc", "WF/x") is None
    assert CR.nested_store_error({"fold_local_store_required": False}, rec_consolidated, "abc", "WF/x") is None


def test_search_overfit_fields_from_a_fake_freeze(monkeypatch):
    fr = {"runs": {"without_veto": {"points": [{"genes": {"a": 1}, "veto": False}, {"genes": {"a": 2}, "veto": False}], "front": []},
                   "with_veto": {"points": [{"genes": {"a": 1}, "veto": True}], "front": []}}, "chosen": {"sharpe": 1.2}}
    monkeypatch.setattr(C3, "StrategyC3", C3.StrategyC3)
    out = C3.search_overfit(fr, None, None, None, C3.CELLS, (0, 0), 0.0, "INNER_A")
    assert out["n_trials_archived"] == 3 and out["front_points_replayed"] == 0 and "pbo" in out


# ----------------------------------------------------------------------------- execution variant, best-of-day gate, per-bar band
def test_full_f3_variant_identity_matches_lane_f3_and_the_contract_declares_it():
    from app.short_horizon.expert_guided import exec_variants as XV
    assert C3.FULL_F3.identity == "exec_709f9b1fa3512486" and XV.LEGACY.identity == "exec_5d80473b21dc61a0"
    d = C3.default_contract()
    assert d["exec_variant"]["identity"] == C3.FULL_F3.identity and d["exec_variant"]["name"] == "full_f3"
    bad = copy.deepcopy(d)
    bad["exec_variant"]["options"]["risk_pct"] = 0.02
    bad = ACT.seal(dict(bad, identity=None))
    assert any(e.startswith("C3_CONTRACT_EXEC_VARIANT") for e in C3.validate(bad))
    bad2 = ACT.seal(dict(copy.deepcopy(d), identity=None, best_of_day={"observe_bars": 0}))
    assert any(e.startswith("C3_CONTRACT_BEST_OF_DAY") for e in C3.validate(bad2))


def test_policy_call_returns_a_best_of_day_gate_over_the_scored_policy():
    from app.short_horizon.expert_guided import exec_variants as XV
    cells = [(1.0, 1.0), (0.5, 0.5)]
    t = _table(cells=cells)
    t["net"][0, 0, :] = [0.3, 0.1]
    s = C3.StrategyC3(-9, -9, 0.0, 0.0, 1.0, 1.0, False, None)
    pol = C3.PolicyC3(t, s, cells, (-0.2, -0.2), 0.05)
    g = pol(None)
    assert isinstance(g, XV.BestOfDayGate) and g.observe_bars == 1 and pol.last_gate is g
    a, tp, sl, score = pol.scored(None)(0, 0, np.zeros(9), None)
    assert a == E.LONG and (tp, sl) == (1.0, 1.0) and np.isfinite(score)
    assert pol.open_decision(0, with_score=True)[3] == score
    assert C3.PolicyC3(t, s, cells, (-0.2, -0.2), 0.05, observe_bars=0)(None).observe_bars == 0     # the audited first-of-day control


def test_cells_outside_the_per_bar_effective_band_are_not_selectable():
    cells = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
    net = np.array([0.5, 0.4, 0.9])
    assert C3.allowed_cells(net, cells, 0.0, cap=1.5).tolist() == [True, True, False]
    assert C3.allowed_cells(net, cells, 0.0, cap=1.5, lo=0.75).tolist() == [False, True, False]
    ci, ev = C3.best_cell_ev(net, [0.5, 0.3, 0.2], cells, 0.0, 0.0, cap=1.5, lo=0.75)
    assert ci == 1
    t = _table(cells=cells)
    t["net"][0, 0, :] = net
    t["cap"], t["lo"] = np.array([1.5] * 4), np.array([0.75] * 4)
    s = C3.StrategyC3(-9, 9, 0.0, 0.0, 1.0, 1.0, False, None)
    assert C3.PolicyC3(t, s, cells, (0, 0), 0.0).open_decision(0) == (E.LONG, 1.0, 1.0)


def test_early_close_rule_uses_share_and_lower_bound():
    from app.short_horizon.expert_guided import exec_variants as XV
    assert XV.early_close_ok(0.10, 5.0, C3.EARLY_CLOSE_MIN_SHARE)[0]
    assert not XV.early_close_ok(0.01, 5.0, C3.EARLY_CLOSE_MIN_SHARE)[0]
    assert not XV.early_close_ok(0.10, -1.0, C3.EARLY_CLOSE_MIN_SHARE)[0]


# ----------------------------------------------------------------------------- A1-11: the encoding decisions applied
def test_consumer_ops_onehot_and_window_known_answers_and_unknown_refused():
    from app.short_horizon.expert_guided import data as DG
    feats = ["p0", "p1", "p2", "hs"]
    X = np.array([[0.2, 0.7, 0.1, 1.0], [0.6, 0.3, 0.1, 5.0], [np.nan, 0.5, 0.5, 23.0], [0.1, 0.1, 0.8, 22.0]])
    cols = [{"store_column": "p0", "consumer_ops": [{"op": "onehot_argmax", "sources": ["p0", "p1", "p2"], "index": 0}]},
            {"store_column": "p1", "consumer_ops": [{"op": "onehot_argmax", "sources": ["p0", "p1", "p2"], "index": 1}]},
            {"store_column": "p2", "consumer_ops": [{"op": "onehot_argmax", "sources": ["p0", "p1", "p2"], "index": 2}]},
            {"store_column": "hs", "consumer_ops": [{"op": "window_hours", "hours": 2.0, "period_hours": 24.0}]}]
    Y = DG.apply_consumer_ops(X, cols, feats)
    assert Y[0, :3].tolist() == [0.0, 1.0, 0.0] and Y[1, :3].tolist() == [1.0, 0.0, 0.0] and Y[3, :3].tolist() == [0.0, 0.0, 1.0]
    assert np.isnan(Y[2, :3]).all()
    assert Y[:, 3].tolist() == [1.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="RECIPE_CONSUMER_OP_UNKNOWN"):
        DG.apply_consumer_ops(X, [{"store_column": "p0", "consumer_ops": [{"op": "bin"}]}], feats)
    assert np.array_equal(DG.apply_consumer_ops(X, [{"store_column": "p0", "consumer_ops": ["DIFF"]}], feats)[1:, 0], np.diff(X[:, 0]), equal_nan=True)


def test_c3_refuses_a_recipe_without_the_encoding_decisions():
    d = C3.default_contract()
    with pytest.raises(SystemExit, match="C3_RECIPE_ENCODING_NOT_APPLIED"):
        C3.check_recipe_encoding(d, {"identity": "x"})
    C3.check_recipe_encoding(d, {"encoding_decisions_applied": dict(C3.ENCODING_DECISIONS)})
    bad = ACT.seal(dict(copy.deepcopy(d), identity=None, encoding_decisions={"regime_gmm": "NONE"}))
    assert any(e.startswith("C3_CONTRACT_ENCODING_DECISIONS") for e in C3.validate(bad))


def test_derive_encoding_recipe_rewrites_columns_groups_and_scaler(monkeypatch, tmp_path):
    from tools import expert_guided_wf_recipe as WR
    base = {"identity": "recipe_base", "tensor": {"columns": [
        {"store_column": "rg.gmm.p0", "tensor_name": "rg.gmm.p0", "consumer_ops": []}, {"store_column": "rg.gmm.p1", "tensor_name": "rg.gmm.p1", "consumer_ops": []},
        {"store_column": "rg.gmm.p2", "tensor_name": "rg.gmm.p2", "consumer_ops": []}, {"store_column": "rg.gmm.entropy", "tensor_name": "rg.gmm.entropy", "consumer_ops": []},
        {"store_column": "ev.ny_cut.hours_since", "tensor_name": "ev.ny_cut.hours_since", "consumer_ops": []},
        {"store_column": "ev.ny_cut.hours_to", "tensor_name": "ev.ny_cut.hours_to", "consumer_ops": []}, {"store_column": "ATR", "tensor_name": "ATR", "consumer_ops": []}]},
        "list": {"n": 7, "columns": []}, "representation": {"groups": {"groups": {"gmm": ["rg.gmm.p0", "rg.gmm.p1", "rg.gmm.p2", "rg.gmm.entropy"],
                                                                                   "ev": ["ev.ny_cut.hours_since", "ev.ny_cut.hours_to"], "atr": ["ATR"]}},
                                                      "resolved": {"names": ["gmm", "ev", "atr"], "indices": [[0, 1, 2, 3], [4, 5], [6]], "units": [2, 2, 2], "kinds": ["conv", "pointwise", "conv"]}},
        "scaler": {"features": [], "n_features": 7, "mean": [0] * 7, "std": [1] * 7}, "store": {}, "regularization": {}, "split": {}, "business": {}, "labels": {}}
    monkeypatch.setattr(WR, "fold_constants", lambda store, rec, fold: {"fit_period": ["a", "b"], "fit_rows": 1, "mean": [0.0] * len(rec["tensor"]["columns"]),
                                                                        "std": [1.0] * len(rec["tensor"]["columns"]), "std_raw": [], "variance": [], "rule": [], "fit_rows_per_feature": [], "nan_share_train": []})
    rec = C3.derive_encoding_recipe(base, "sha", tmp_path / "x.parquet", "wf2022")
    names = [c["tensor_name"] for c in rec["tensor"]["columns"]]
    assert names == ["rg.gmm.onehot0", "rg.gmm.onehot1", "rg.gmm.onehot2", "rg.gmm.entropy", "ev.ny_cut.window2h", "ATR"]
    assert rec["list"]["n"] == 6 and rec["representation"]["resolved"]["indices"] == [[0, 1, 2, 3], [4], [5]]
    assert rec["encoding_decisions_applied"] == C3.ENCODING_DECISIONS and rec["identity"] != "recipe_base" and rec["derived_from"]["identity"] == "recipe_base"
    ops = {c["tensor_name"]: c["consumer_ops"] for c in rec["tensor"]["columns"]}
    assert ops["rg.gmm.onehot2"] == [{"op": "onehot_argmax", "sources": ["rg.gmm.p0", "rg.gmm.p1", "rg.gmm.p2"], "index": 2}]
    assert ops["ev.ny_cut.window2h"] == [{"op": "window_hours", "hours": 2.0, "period_hours": 24.0}]


def test_head_outputs_block_is_explicit_and_checked():
    d = C3.default_contract()
    ho = d["head_outputs"]
    assert set(ho) >= {"open_l.decision", "open_l.continuous", "nop.decision", "close_l.decision", "close_l.continuous", "gate", "early_close_switch"}
    assert ho["open_l.continuous"]["outputs"] == len(C3.CELLS) and "[-4, 4]" in ho["open_l.continuous"]["range"]
    assert ho["open_l.decision"]["classes"] == C3.CLASSES_OPEN
    bad = ACT.seal(dict(copy.deepcopy(d), identity=None, head_outputs={}))
    assert any(e.startswith("C3_CONTRACT_HEAD_OUTPUTS") for e in C3.validate(bad))
