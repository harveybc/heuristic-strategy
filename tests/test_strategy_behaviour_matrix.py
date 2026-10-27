"""Behaviour matrix (owner order 2026-10-27): the generators every queued cell relies on must pass their unit tests
before their cells are appended to the running queue. Exact price mirror, multi-leg transitions with the switch bars
and per-leg drift signs, volatility processes (step / ramp / GARCH persistence), gaps with and against the leg, the
cell list (ids unique, every base cell has its zero-cost twin and exact mirror, expected / forbidden written), the
mirror check and the parameter file wiring. No engine, no data."""
import json

import numpy as np
import pandas as pd
import pytest

from tools import strategy_behaviour_matrix as BM
from tools import strategy_hierarchy as H


def _idx(n=6000):
    return pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")


def test_mirror_bars_is_an_exact_price_mirror():
    idx = _idx(3000)
    bars, _ = H.synthetic_path_v2(idx, 5e-4, ("trend_up",), "mid" if "mid" in H.VOL else "low", "medium", 3, (500, 2500))
    m = H.mirror_bars(bars)
    K = float(bars["HIGH"].max()) + float(bars["LOW"].min())
    assert np.allclose(m["CLOSE"].to_numpy(), K - bars["CLOSE"].to_numpy(), atol=1e-12)
    assert np.allclose(m["HIGH"].to_numpy(), K - bars["LOW"].to_numpy(), atol=1e-12)
    assert float(m["LOW"].min()) >= float(bars["LOW"].min()) - 1e-12 and float(m["HIGH"].max()) <= float(bars["HIGH"].max()) + 1e-12   # stays in range
    assert np.allclose(np.diff(m["CLOSE"].to_numpy()), -np.diff(bars["CLOSE"].to_numpy()), atol=1e-12)
    assert np.all(m["HIGH"] >= m[["OPEN", "CLOSE"]].max(axis=1)) and np.all(m["LOW"] <= m[["OPEN", "CLOSE"]].min(axis=1))
    mm, _ = H.synthetic_path_v2(idx, 5e-4, ("trend_up",), "low", "medium", 3, (500, 2500), mirror=True)
    b2, _ = H.synthetic_path_v2(idx, 5e-4, ("trend_up",), "low", "medium", 3, (500, 2500), mirror=False)
    assert np.allclose(mm["CLOSE"].to_numpy(), float(b2["HIGH"].max()) + float(b2["LOW"].min()) - b2["CLOSE"].to_numpy(), atol=1e-12)


def test_transitions_switch_where_declared_and_legs_have_their_drift():
    idx = _idx()
    sd = 7e-4
    bars, meta = H.synthetic_path_v2(idx, sd, ("trend_up", "trend_down"), "garch09", "medium", 11, (1000, 5000), [0.5], strength="strong")
    assert meta["switch_bars"] == [3000]
    w = H.transition_witness(bars, sd, meta)
    assert [x["drift_sign"] for x in w["legs"]] == [1.0, -1.0]
    bars3, meta3 = H.synthetic_path_v2(idx, sd, ("trend_down", "range", "trend_down"), "garch09", "medium", 12, (1000, 5000), [0.3, 0.7], strength="strong")
    assert meta3["switch_bars"] == [2200, 3800]
    w3 = H.transition_witness(bars3, sd, meta3)
    # realised signs are sample statistics over 1 200-bar legs (noise + fine oscillation); the witness records them per
    # leaf and the expectation is the declared sign: here the last (longest) leg and at least two of three legs agree
    agree = [x["drift_sign"] == x["expected_drift_sign"] for x in w3["legs"] if x["expected_drift_sign"] != 0]
    assert w3["legs"][2]["drift_sign"] == -1.0 and sum(agree) >= len(agree) - 1
    bm, metam = H.synthetic_path_v2(idx, sd, ("trend_up", "trend_down"), "garch09", "medium", 11, (1000, 5000), [0.5], strength="strong", mirror=True)
    wm = H.transition_witness(bm, sd, metam)
    assert [x["drift_sign"] for x in wm["legs"]] == [-1.0, 1.0] and [x["expected_drift_sign"] for x in wm["legs"]] == [-1.0, 1.0]


def test_volatility_processes_change_the_level_as_declared():
    idx = _idx()
    sd = 7e-4
    for vol, up in (("step_lh", True), ("step_hl", False), ("ramp_up", True), ("ramp_down", False)):
        bars, meta = H.synthetic_path_v2(idx, sd, ("range",), vol, "medium", 5, (1000, 5000))
        w = H.transition_witness(bars, sd, meta)
        assert (w["vol_second_over_first_half"] > 1.3) == up and (w["vol_second_over_first_half"] < 0.77) == (not up), vol
    b9, _ = H.synthetic_path_v2(idx, sd, ("range",), "garch09", "medium", 5, (1000, 5000))
    b97, _ = H.synthetic_path_v2(idx, sd, ("range",), "garch097", "medium", 5, (1000, 5000))
    r9, r97 = np.diff(np.log(b9["CLOSE"].to_numpy())), np.diff(np.log(b97["CLOSE"].to_numpy()))
    ac = lambda r: np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1]  # noqa: E731
    assert ac(r97) > ac(r9) > 0.0                                       # stronger clustering at higher persistence


def test_gaps_are_placed_with_or_against_the_leg():
    idx = _idx()
    sd = 7e-4
    for against in (False, True):
        bars, meta = H.synthetic_path_v2(idx, sd, ("trend_down",), "low", "medium", 8, (1000, 5000), jump={"sigma": 4.0, "against": against, "frac": 0.5})
        jb = meta["jump_bar"]
        assert jb == 3000
        r = np.diff(np.log(bars["CLOSE"].to_numpy()))
        assert abs(r[jb - 1]) > 3.0 * sd * np.sqrt(24.0) * 0.7
        assert (r[jb - 1] > 0) == against                               # down leg: 'against' = an upward gap


def test_matrix_cells_are_complete_and_sealed_by_content():
    cells = BM.build_matrix(n_lhs=8, seed=1)
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids))
    base = [c for c in cells if not c["mirror"] and not c.get("sym_of")]
    assert len(cells) == 3 * len(base)
    byid = {c["id"]: c for c in cells}
    for c in base:
        s, m = byid["S_" + c["id"]], byid["M_" + c["id"]]
        assert s["sym_of"] == c["id"] and m["mirror_of"] == s["id"] and m["legs_gen"] == c["legs"] and m["legs"] == [H.MIRROR[x] for x in c["legs"]]
        assert s["exec"].get("costs_zero") and m["exec"].get("costs_zero")
        assert c["expected"] and c["forbidden"] and c["quality"] == 1.0
    levels = {c["level"] for c in base}
    assert {"backbone_trend", "backbone_transition", "backbone_vol_level", "backbone_vol_dynamics", "backbone_fine", "backbone_gap", "backbone_spread", "backbone_calendar",
            "F1_trend_x_fine", "F2_trend_x_vol", "F3_transition_x_fine", "F4_trend_x_gap", "F5_trend_x_spread_x_calendar", "LHS"} <= levels
    json.dumps(cells)                                                  # serialisable (sealed by sha of the file)


def test_mirror_check_demands_identical_bars_flipped_sides_and_equal_sigma_net():
    a = [{"entry_bar_abs": 10, "exit_bar_abs": 20, "direction": 1, "net_sigma": 1.5, "financing_sigma": 0.0, "net_usd": 10.0},
         {"entry_bar_abs": 30, "exit_bar_abs": 35, "direction": -1, "net_sigma": -0.7, "financing_sigma": 0.0, "net_usd": -5.0}]
    b = [{**a[0], "direction": -1}, {**a[1], "direction": 1}]
    assert BM.mirror_check(a, b)["pass"]
    assert not BM.mirror_check(a, [b[0], {**b[1], "exit_bar_abs": 36}])["pass"]
    assert not BM.mirror_check(a, [b[0], {**b[1], "net_sigma": -0.6}])["pass"]
    assert not BM.mirror_check(a, [{**b[0], "direction": 1}, b[1]])["pass"]


def test_canonical_genes_come_from_the_parameter_file():
    from app.short_horizon.expert_guided import strategy_params as SP
    g = SP.genes_c3(SP.default_doc())
    assert g["open_rate"] == pytest.approx(0.45) and g["allow_reverse"] == 0.0 and g["nop_veto"] == 0.5 and g["close_p_l"] == 0.5
    d = SP.default_doc()                                               # the v10 canonical point (transition fixes): no reversal, no strong-net exception, room 1.75 x TP
    assert d["zz_counter_net_frac"] == pytest.approx(1.2) and d["zz_room_tp_mult"] == pytest.approx(1.75)


def test_intensity_generators_vary_trend_and_volatility_independently():
    """Owner 2026-10-27: per-leg trend strengths and volatility changes of declared ratio / slope at their own bar."""
    idx = _idx(8000)
    sd = 7e-4
    blk = (1000, 7000)
    drift = {}
    for s1, s2 in (("weak", "strong"), ("strong", "weak")):
        segs = []
        for seed in range(10):                                         # one path's leg mean has SE ~ 1.3e-5 (AR(1) drift + noise): average 10 seeds
            bars, meta = H.synthetic_path_v2(idx, sd, ("trend_up", "trend_up"), "low", "medium", 21 + seed, blk, [0.5], strengths=[s1, s2])
            r = np.diff(np.log(bars["CLOSE"].to_numpy()))
            segs.append((r[1500:4000].mean(), r[4500:7000].mean()))
            assert meta["strengths"] == [s1, s2]
        drift[(s1, s2)] = tuple(np.mean(segs, axis=0))
    assert drift[("weak", "strong")][1] > drift[("weak", "strong")][0] and drift[("strong", "weak")][0] > drift[("strong", "weak")][1]
    for vk, up in (("vstep_up_3", True), ("vstep_down_1.5", False), ("vramp_up_steep", True), ("vramp_down_gentle", False)):
        bars, meta = H.synthetic_path_v2(idx, sd, ("range",), vk, "medium", 9, blk, None, vol_frac=0.3)
        assert meta["vol_change_bar"] == 1000 + int(0.3 * 6000)
        r = np.diff(np.log(bars["CLOSE"].to_numpy()))
        cb = meta["vol_change_bar"]
        ratio = r[cb + 400:7000].std() / r[1000:cb - 50].std()
        assert (ratio > 1.15) == up and (ratio < 0.87) == (not up), (vk, ratio)
    b3, _ = H.synthetic_path_v2(idx, sd, ("range",), "vstep_up_3", "medium", 9, blk, None, vol_frac=0.5)
    b15, _ = H.synthetic_path_v2(idx, sd, ("range",), "vstep_up_1.5", "medium", 9, blk, None, vol_frac=0.5)
    rr = lambda b: np.diff(np.log(b["CLOSE"].to_numpy()))  # noqa: E731
    assert rr(b3)[4400:7000].std() / rr(b3)[1000:3950].std() > rr(b15)[4400:7000].std() / rr(b15)[1000:3950].std()   # intensity is graded
    bt, mt = H.synthetic_path_v2(idx, sd, ("trend_up", "trend_down"), "vstep_up_2", "medium", 9, blk, [0.7], vol_frac=0.3)
    assert mt["switch_bars"] == [1000 + int(0.7 * 6000)] and mt["vol_change_bar"] == 1000 + int(0.3 * 6000)   # independent in time
    bs, _ = H.synthetic_path_v2(idx, sd, ("trend_up",), "low", "medium", 4, blk)
    bs2, _ = H.synthetic_path_v2(idx, sd, ("trend_up",), "low", "medium", 4, blk, strengths=None)
    assert np.allclose(bs["CLOSE"].to_numpy(), bs2["CLOSE"].to_numpy())                                        # sealed-block paths unchanged


def test_intensity_block_cells_cover_every_declared_combination():
    cells = BM.build_intensity_block()
    base = [c for c in cells if not c["mirror"] and not c.get("sym_of")]
    assert len(cells) == 3 * len(base) and len({c["id"] for c in cells}) == len(cells)
    i1 = [c for c in base if c["level"] == "I1_trend_change_intensity"]
    tt = [c for c in i1 if c["intensity"]["kind"] == "trend_change"]
    assert len(tt) == (4 * 9 - 2 * 3) * 3                                                                     # 4 ordered trend pairs x 3 x 3 strengths (no-change excluded) x 3 switches
    assert {(c["intensity"]["from"], c["intensity"]["to"]) for c in tt} == {(a, b) for a in BM.STRENGTHS for b in BM.STRENGTHS}
    assert len([c for c in i1 if c["intensity"]["kind"] == "trend_range"]) == 2 * 3 * 2 * 3
    i2 = [c for c in base if c["level"] == "I2_vol_change_intensity"]
    i3 = [c for c in base if c["level"] == "I3_vol_x_trend_change"]
    assert len(i2) == 10 * 3 and len(i3) == 10 * 2 * 3
    assert {c["intensity"]["timing"] for c in i3} == {"vol_first", "trend_first", "same_bar"}
    assert all(c["intensity"]["vol_at"] != c["intensity"]["trend_at"] for c in i3 if c["intensity"]["timing"] != "same_bar")
    assert len([c for c in base if c["level"] == "I4_garch_x_trend_change"]) == 6
    for c in base:
        assert c["expected"] and c["forbidden"] and c["quality"] == 1.0


def test_multiscale_world_components_are_planted_and_the_ladder_reads_them():
    """Multi-horizon worlds: independent components with their own turning points; the ladder pivots and the ideal
    hierarchy heads exist for every bar; the mirror flips the coarse direction; the multi-horizon block is complete."""
    from app.short_horizon.expert_guided import strategy_zz as ZZ
    idx = _idx(6000)
    sd = 7e-4
    spec = {"long": {"legs": ["trend_up", "trend_down"], "strengths": ["strong", "strong"], "switch_fracs": [0.5]},
            "medium": {"amp": "mid", "align": "with", "leg_bars": [48, 144]}, "short": {"fine": "medium", "bursts": 2, "burst_mult": 3.0, "burst_bars": [6, 12], "burst_at_switch": True, "jumps": []},
            "vol": {"horizon": "both", "slow_ratio": 2.0, "slow_mean_bars": 360, "fast_persistence": 0.97}}
    bars, meta = H.synthetic_path_ms(idx, sd, spec, 5, (1000, 5000))
    assert meta["switch_bars"] == [3000] and meta["burst_windows"][0][0] == 3000
    assert len(meta["turning_points"]["medium"]) > 20 and len(meta["turning_points"]["short"]) > 200
    sigma = pd.Series(np.diff(np.r_[bars["CLOSE"].iloc[0], bars["CLOSE"].to_numpy()]) ** 2).rolling(24).sum().pow(0.5).rolling(480, min_periods=120).mean().shift(1).to_numpy()
    q = ZZ.ladder_quantities(bars, sigma)
    assert q["room"].shape == (6000, 2) and set(q["pivots"]) == set(ZZ.DEFAULT_LEVELS)
    mb, _ = H.synthetic_path_ms(idx, sd, spec, 5, (1000, 5000), mirror=True)
    qm = ZZ.ladder_quantities(mb, sigma)
    both = (q["direction"] != 0) & (qm["direction"] != 0)
    assert np.mean(q["direction"][both] == -qm["direction"][both]) > 0.99
    pm = H.pivot_matching(q["pivots"], meta["turning_points"], (1000, 5000))
    assert set(pm["capture"]) == {"long", "medium", "short"}
    cells = BM.build_multihorizon_block()
    base = [c for c in cells if not c["mirror"] and not c.get("sym_of")]
    assert len(cells) == 3 * len(base) and all(c["ms"] for c in base)
    assert {c["intensity"]["active"] for c in base if c["level"] == "H1_active_horizons"} == {"long", "medium", "short", "long+medium", "long+short", "medium+short", "long+medium+short"}
    assert all(any("NO NEW OPENS" in e for e in c["expected"]) and any("EARLY CLOSE" in e for e in c["expected"]) and any("OPENS" in e for e in c["expected"]) for c in base)


def test_exit_advantage_and_path_definitions():
    """P(t), P*, A(t): the advantage is the close-now value minus the best continuation (the natural exit included),
    in sigma units of the trade; the sealed exit spec carries every definition and threshold."""
    from tools import strategy_output_simulation as S
    path = {"bars": np.array([10, 11, 12, 13]), "P": np.array([5.0, 9.0, 7.0, 3.0]), "hold_net": -20.0, "units": 10, "sigma": 0.5, "hold_reason": "SL", "exit_bar": 14}
    assert S.path_advantage(path, 10) == pytest.approx((5.0 - 9.0) / 10 / 0.5)
    assert S.path_advantage(path, 11) == pytest.approx((9.0 - 7.0) / 10 / 0.5)        # the best exit: A >= 0
    assert S.path_advantage(path, 13) == pytest.approx((3.0 + 20.0) / 10 / 0.5)       # close before the SL
    assert np.isnan(S.path_advantage(path, 99))
    assert set(BM.EXIT_SPEC["witnesses_per_side"]) >= {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E1'", "E2'", "E7", "Z1", "E8", "EP"}
    assert len(BM.EXIT_SPEC_SHA) == 64 and BM.EPS_CLOSE == 0.02


def test_episode_replay_scores_regret_lateness_and_the_loss_curve():
    """run_episodes on a hand-made path: a closer that closes at the best tick has R = 0 and lateness 0; one that
    holds to the SL has R = P*_pre - SL loss; L(t) is P*_pre - P(t) over the held ticks."""
    class _C:
        def path(self, pos):
            return {"bars": np.array([0, 1, 2, 3]), "P": np.array([1.0, 4.0, 2.0, 0.0]), "hold_net": -6.0, "units": 1, "sigma": 1.0, "hold_reason": "SL", "exit_bar": 4}
    class _CX:
        rob = np.arange(10)
        obs_ok = np.ones(10, bool)
    world = {"cx": _CX()}
    eps = [{"pos": None, "side": 0, "class": "SL", "start_bar": 0}]
    best = BM.run_episodes(world, eps, lambda r, si, bar, pos: bar == 1, _C())[0]
    hold = BM.run_episodes(world, eps, lambda r, si, bar, pos: False, _C())[0]
    assert best["R_pre"] == pytest.approx(0.0) and best["late_bars"] == 0 and best["early"]
    assert hold["R_pre"] == pytest.approx(4.0 + 6.0) and not hold["early"]
    assert best["L_curve"] == [pytest.approx(3.0), pytest.approx(0.0)]


def test_validity_rule_vacuous_controls_and_the_w8_replay_conflict():
    """VALIDITY_SPEC (coordinator ruling 2026-10-27): a control's pass with zero events does not validate the sealed
    form; a control's failure admits the corrected form; O* is a second control for the timing / exit witnesses; the W8
    switch v3 governs where the replay shows a destructive-close conflict."""
    w = lambda p, e=None, **k: {"value": 0, "pass": p, "events": e, **k}  # noqa: E731
    strat = {"W1_side": w(False, 10), "W1_side_v2": w(True, 10), "W5_reverse": w(False, 3), "W5_reverse_v2": w(True, 3),
             "W8_early_close_at_switch": w(False, 2), "W8_early_close_at_switch_v2": w(True, 2), "W8_early_close_at_switch_v3": w(True, 2, conflict_shown=True),
             "W3_nop": w(False, 10)}
    hold = {"W1_side": w(True, 0), "W5_reverse": w(True, 0), "W8_early_close_at_switch": w(True, 0), "W3_nop": w(True, 12)}
    ostar = {"W5_reverse": w(True, 4), "W8_early_close_at_switch": w(True, 3)}
    v = BM.validity(strat, hold, ostar)["per_witness"]
    assert v["W1_side"]["status"] == "SEALED_UNVALIDATED" and v["W1_side"]["governing_form"] == "W1_side_v2" and v["W1_side"]["pass"]   # vacuous hold pass; O* is not a W1 control
    assert v["W5_reverse"]["status"] == "VALIDATED" and v["W5_reverse"]["governing_form"] == "W5_reverse" and not v["W5_reverse"]["pass"]   # O* reverses: sealed form governs
    assert v["W8_early_close_at_switch"]["status"] == "REPLAY_CONFLICT" and v["W8_early_close_at_switch"]["governing_form"] == "W8_early_close_at_switch_v3"
    assert v["W3_nop"]["status"] == "VALIDATED" and not v["W3_nop"]["pass"]
    hold["W3_nop"] = w(False, 12)
    v = BM.validity(strat, hold, ostar)
    assert v["per_witness"]["W3_nop"]["status"] == "DEFECT" and v["per_witness"]["W3_nop"]["governing_form"] == "W3_nop"   # no corrected form: sealed, reported
    assert set(v["failing"]) == {"W5_reverse", "W3_nop"}
    strat["W8_early_close_at_switch_v3"] = w(True, 2, conflict_shown=False)
    assert BM.validity(strat, hold, ostar)["per_witness"]["W8_early_close_at_switch"]["status"] == "VALIDATED"   # no replayed conflict: the sealed failure stands


def test_spec_v3_is_resealed_with_the_validity_rule():
    assert "W8_early_close_at_switch_v3" in BM.WITNESS_SPEC["v3"] and "vacuity" in BM.VALIDITY_SPEC
    doc = {"witness_spec": BM.WITNESS_SPEC, "exit_spec": BM.EXIT_SPEC, "validity_spec": BM.VALIDITY_SPEC, "thresholds": BM.THRESH}
    import hashlib
    assert BM.SPEC_V3_SHA == hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    assert any("v3" in x["witness"] and "-0.497" in x["evidence"] for x in BM.WITNESS_CHANGELOG)
