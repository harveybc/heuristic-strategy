"""Behaviour matrix (owner order 2026-10-27): the generators every queued cell relies on must pass their unit tests
before their cells are appended to the running queue. Exact price mirror, multi-leg transitions with the switch bars
and per-leg drift signs, volatility processes (step / ramp / GARCH persistence), gaps with and against the leg, the
cell list (ids unique, every base cell has its zero-cost twin and exact mirror, expected / forbidden written), the
mirror check and the parameter file wiring. No engine, no data."""
import json
import math

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


def test_validity_v2_reproduces_musashi_gm01_witnesses_with_the_corrected_outputs():
    """Musashi's three executed counterexamples (agent-multi 660511a0, tools/audit_guided_mechanism_design.py,
    GUIDED_COUNTEREXAMPLES.json) under the v1 rule: PASS on an empty set; a zero-event control selected the corrected
    passing form; one failing + one passing control selected DEFECT and the corrected form. Under v2 (GM01): an empty
    set is NOT_EVALUABLE (never PASS); a vacuous control leaves the SEALED form governing (strict, flagged, counted
    separately); the single designated control decides, and only non-vacuously."""
    old = {"pass": False, "events": 1}
    new = {"pass": True, "events": 1}
    witnesses = {"W": old, "W_v2": new}
    empty = BM.validity({}, {}, None)
    assert empty["PASS"] is False and empty["evaluated"] == 0 and not empty["per_witness"]
    vacuous = BM.validity(witnesses, {"W": {"pass": True, "events": 0}}, None)
    w = vacuous["per_witness"]["W"]
    assert vacuous["PASS"] is False and vacuous["unvalidated"] == ["W"] and vacuous["failing"] == ["W"]
    assert w["status"] == "SEALED_UNVALIDATED" and w["governing_form"] == "W" and w["pass"] is False and w["designated_control"] == "hold_oracle"
    old_timing = BM.TIMING_EXIT_WITNESSES
    try:
        BM.TIMING_EXIT_WITNESSES = ("W",)                                 # W a timing witness: O* is its ONE designated control
        conflicting = BM.validity(witnesses, {"W": old}, {"W": new})
        w = conflicting["per_witness"]["W"]
        assert w["status"] == "VALIDATED" and w["governing_form"] == "W" and w["pass"] is False and w["designated_control"] == "O*" and w["failed_by"] == []
        assert conflicting["PASS"] is False and conflicting["defects"] == []
        defect = BM.validity(witnesses, {"W": new}, {"W": old})           # the designated control (O*) fails non-vacuously: DEFECT, v2 governs
        assert defect["per_witness"]["W"]["status"] == "DEFECT" and defect["per_witness"]["W"]["governing_form"] == "W_v2" and defect["PASS"] is True
        vac_fail = BM.validity(witnesses, {"W": new}, {"W": {"pass": False, "events": 0}})   # a vacuous failure admits nothing
        assert vac_fail["per_witness"]["W"]["status"] == "SEALED_UNVALIDATED" and vac_fail["PASS"] is False
    finally:
        BM.TIMING_EXIT_WITNESSES = old_timing
    # the v1 rule, retired, still shows Musashi's three outputs (the delta the re-score reports)
    assert BM.validity_v1_retired({}, {}, None)["PASS"] is True
    assert BM.validity_v1_retired(witnesses, {"W": {"pass": True, "events": 0}}, None)["per_witness"]["W"]["governing_form"] == "W_v2"


def test_validity_v2_expected_witnesses_not_evaluable_and_the_w8_replay_conflict():
    """VALIDITY_SPEC v2: an expected witness the run did not produce is NOT_EVALUABLE and fails the run; the W8 switch
    REPLAY_CONFLICT comes only from the strategy's own replay (conflict_shown) and reports the replayed margins."""
    w = lambda p, e=None, **k: {"value": 0, "pass": p, "events": e, **k}  # noqa: E731
    strat = {"W1_side": w(False, 10), "W1_side_v2": w(True, 10), "W5_reverse": w(False, 3), "W5_reverse_v2": w(True, 3),
             "W8_early_close_at_switch": w(False, 2), "W8_early_close_at_switch_v2": w(True, 2),
             "W8_early_close_at_switch_v3": {"value": {"non_early_exits": [{"best_A_of_an_earlier_close_sigma": -0.4}], "Ostar_same_trade_misses": []}, "pass": True, "events": 2, "conflict_shown": True},
             "W3_nop": w(False, 10)}
    hold = {"W1_side": w(True, 0), "W5_reverse": w(True, 0), "W8_early_close_at_switch": w(True, 0), "W3_nop": w(True, 12)}
    ostar = {"W5_reverse": w(True, 4), "W8_early_close_at_switch": w(True, 3)}
    v = BM.validity(strat, hold, ostar, ["W1_side", "W5_reverse", "W8_early_close_at_switch", "W3_nop", "W9_gap_loss_bounded"])
    pw = v["per_witness"]
    assert pw["W1_side"]["status"] == "SEALED_UNVALIDATED" and pw["W1_side"]["governing_form"] == "W1_side" and not pw["W1_side"]["pass"]   # vacuous hold: SEALED governs, strict
    assert pw["W5_reverse"]["status"] == "VALIDATED" and not pw["W5_reverse"]["pass"]   # O* (the one designated control) reverses non-vacuously
    assert pw["W8_early_close_at_switch"]["status"] == "REPLAY_CONFLICT" and pw["W8_early_close_at_switch"]["governing_form"] == "W8_early_close_at_switch_v3"
    assert pw["W8_early_close_at_switch"]["replay"]["non_early_exits"][0]["best_A_of_an_earlier_close_sigma"] == -0.4
    assert pw["W9_gap_loss_bounded"]["status"] == "NOT_EVALUABLE" and not pw["W9_gap_loss_bounded"]["pass"] and v["not_evaluable"] == ["W9_gap_loss_bounded"]
    assert pw["W3_nop"]["status"] == "VALIDATED" and not pw["W3_nop"]["pass"] and v["PASS"] is False
    assert set(v["failing"]) == {"W1_side", "W5_reverse", "W3_nop", "W9_gap_loss_bounded"}
    hold["W3_nop"] = w(False, 12)
    assert BM.validity(strat, hold, ostar)["per_witness"]["W3_nop"]["status"] == "DEFECT"       # no corrected form: the sealed form governs, reported
    strat["W8_early_close_at_switch_v3"]["conflict_shown"] = False
    assert BM.validity(strat, hold, ostar)["per_witness"]["W8_early_close_at_switch"]["status"] == "VALIDATED"   # no replayed conflict: the sealed failure stands


def test_expected_witness_ids_follow_the_cell_spec():
    base = {"legs": ["trend_up"], "exec": {}, "vol": "normal"}
    ids = BM.expected_witness_ids(base)
    assert "W1_side" in ids and "W8_early_close_at_switch" not in ids and "W13_tpsl_track_sigma" not in ids and "W14_mirror" not in ids
    ids = BM.expected_witness_ids({"legs": ["trend_up", "trend_down"], "exec": {"spread": "x5", "cutoff_h": 12, "missing_runs": 2}, "vol": "step_lh", "mirror_of": "x"},
                                  {"meta": {"jump_bar": 100, "burst_windows": [(1, 2)]}})
    for k in ("W8_early_close_at_switch", "W8_old_direction_after_lag", "W9_gap_loss_bounded", "W10_spread_window", "W11_cutoff", "W12_missing", "W13_tpsl_track_sigma", "W17_no_open_in_burst"):
        assert k in ids
    assert "W14_mirror" not in ids                                      # a pair witness: judged after both twins ran, never per run
    nw = BM.pair_witness_not_evaluable("twin missing")
    assert nw["pass"] is False and nw["status"] == "NOT_EVALUABLE"


def test_judge_pairs_marks_a_split_pair_not_evaluable_and_passes_an_exact_mirror():
    from tools import strategy_param_deap as D
    t = [{"entry_bar_abs": 10, "exit_bar_abs": 20, "direction": 1, "net_sigma": 1.0, "financing_sigma": 0.0}]
    m = [{"entry_bar_abs": 10, "exit_bar_abs": 20, "direction": -1, "net_sigma": 1.0, "financing_sigma": 0.0}]
    res = [{"fails": [], "_trades": t, "_pair": "X", "_role": "S"}, {"fails": [], "_trades": m, "_pair": "X", "_role": "M"}]
    D.judge_pairs(res)
    assert res[0]["fails"] == [] and res[1]["fails"] == [] and "_trades" not in res[0]
    lone = [{"fails": [], "_trades": m, "_pair": "Y", "_role": "M"}]
    D.judge_pairs(lone)
    assert lone[0]["fails"] == ["W14_mirror_NOT_EVALUABLE"]


def test_spec_v3_is_resealed_with_the_validity_rule():
    assert "W8_early_close_at_switch_v3" in BM.WITNESS_SPEC["v3"] and "vacuity" in BM.VALIDITY_SPEC
    doc = {"witness_spec": BM.WITNESS_SPEC, "exit_spec": BM.EXIT_SPEC, "validity_spec": BM.VALIDITY_SPEC, "thresholds": BM.THRESH}
    import hashlib
    assert BM.SPEC_V3_SHA == hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    assert any("v3" in x["witness"] and "-0.497" in x["evidence"] for x in BM.WITNESS_CHANGELOG)


def test_w8_switch_v3_c_prime_replays_ostar_on_the_same_trade():
    """(c') coordinator ruling 2026-10-27: O*'s exit rule replayed on each strategy trade alive in the window; a trade O*
    would close inside the window must have been early-closed by the strategy there (or earlier at A >= -EPS_CLOSE)."""
    import types
    lag = BM.LAG
    sw = 1000
    # one LONG trade alive across the switch; its path peaks at bar sw + 5 (O* closes there: A >= -EPS_CLOSE)
    bars = np.arange(990, sw + 40)
    P = np.where(bars <= sw + 5, (bars - 990) * 1.0, (sw + 5 - 990) - (bars - sw - 5) * 2.0)
    path = {"bars": bars, "P": P.astype(float), "hold_net": float(P[-1] - 5), "units": 1, "sigma": 1.0}
    closer = types.SimpleNamespace(path=lambda pos: path)
    world = {"meta": {"switch_bars": [sw]}}
    trade = lambda reason, xb: {"entry_bar_abs": 990, "exit_bar_abs": xb, "direction": 1, "exit_reason": reason}  # noqa: E731
    pos = {(990, 0): object()}
    held = {"trades": [trade("TIME", sw + 40)], "positions": pos, "closes": []}
    r = BM.w8_switch_v3(world, held, None, closer)
    assert not r["pass"] and len(r["value"]["Ostar_same_trade_misses"]) == 1      # O* closes at sw + 5, the strategy held: a miss
    closed = {"trades": [trade("EARLY", sw + 6)], "positions": pos, "closes": [(0, sw + 5, 0, 0.0, 0.0, True)]}
    r = BM.w8_switch_v3(world, closed, None, closer)
    assert r["pass"] and not r["value"]["Ostar_same_trade_misses"] and r["value"]["trades_alive_in_window"] == 1
    late = {"trades": [trade("EARLY", sw + lag + 3)], "positions": pos, "closes": [(0, sw + lag + 2, 0, 0.0, 0.0, True)]}
    assert not BM.w8_switch_v3(world, late, None, closer)["pass"]                 # closed only after the window: a miss


def test_w4_v3_is_admitted_only_where_ostar_fails_w4_v2_non_vacuously():
    """W4_v3' governs only where O* (the designated control) fails BOTH the sealed W4 and W4 v2 non-vacuously; where O*
    passes v2, the corrected form is v2 (coordinator ruling 2026-10-27, VALIDITY_SPEC v2)."""
    w = lambda p, e=None, **k: {"value": 0, "pass": p, "events": e, **k}  # noqa: E731
    strat = {"W4_early_close_justified": w(False, 20), "W4_early_close_justified_v2": w(False, 20), "W4_early_close_justified_v3": w(True, 20)}
    hold = {"W4_early_close_justified": w(True, 5)}
    ostar_fails_v2 = {"W4_early_close_justified": w(False, 20), "W4_early_close_justified_v2": w(False, 20)}
    ostar_passes_v2 = {"W4_early_close_justified": w(False, 20), "W4_early_close_justified_v2": w(True, 20)}
    ostar_vacuous = {"W4_early_close_justified": w(False, 0), "W4_early_close_justified_v2": w(False, 0)}
    a = BM.validity(strat, hold, ostar_fails_v2)["per_witness"]["W4_early_close_justified"]
    b = BM.validity(strat, hold, ostar_passes_v2)["per_witness"]["W4_early_close_justified"]
    c = BM.validity(strat, hold, ostar_vacuous)["per_witness"]["W4_early_close_justified"]
    assert a["status"] == "DEFECT" and a["governing_form"] == "W4_early_close_justified_v3" and a["pass"]
    assert b["status"] == "DEFECT" and b["governing_form"] == "W4_early_close_justified_v2" and not b["pass"]
    assert c["status"] == "SEALED_UNVALIDATED" and c["governing_form"] == "W4_early_close_justified" and not c["pass"]   # a vacuous O* admits nothing


def test_w4_v3_economic_bound_tolerates_only_cheap_and_rare_destructive_closes():
    """GM02: a close with remaining > EPS_CLOSE passes W4 v3' only if remaining <= EPS_ECON and such closes are <= 5 %
    of the early closes; the distance to a barrier is a diagnostic, never the criterion."""
    assert BM.EPS_ECON == 0.05 and BM.EPS_CLOSE < BM.EPS_ECON
    assert "economic" in BM.WITNESS_SPEC["v3_W4"]["W4_early_close_justified_v3"] and any("GM02" in x["witness"] or "GM02" in x["defect"] for x in BM.WITNESS_CHANGELOG)
    import inspect
    src = inspect.getsource(BM.witnesses)
    assert "c[3] <= EPS_ECON" in src and "share_bad <= 0.05 and not costly" in src and "knife_edge_diagnostic" in src


def test_bc_interp_is_bilinear_and_clamped():
    from tools import strategy_output_simulation as S
    Sg, Tg = np.array([0.1, 0.5, 1.0]), np.array([0.2, 1.0, 3.0])
    g = np.add.outer(2.0 * Sg, 3.0 * Tg)                                # linear in both distances: bilinear interpolation is exact
    assert S.bc_interp(g, Sg, Tg, 2.0, 0.75) == pytest.approx(2 * 0.75 + 3 * 2.0)
    assert S.bc_interp(g, Sg, Tg, 9.0, 5.0) == pytest.approx(2 * 1.0 + 3 * 3.0)        # clamped to the grid


def test_hierarchy_heads_degrade_with_quality_and_the_causal_control_lags():
    from app.short_horizon.expert_guided import strategy_zz as ZZ
    from scipy.stats import norm
    n = 20000
    d = np.where((np.arange(n) // 500) % 2 == 0, 1.0, -1.0)
    q = {"direction": d, "flip_age": np.zeros(n), "counter": np.zeros((n, 2), np.int8), "fresh": np.zeros((n, 2), np.int8),
         "nop_hard": np.zeros(n, bool), "room": np.ones((n, 2))}
    assert ZZ.degrade_quantities(q, 1.0, 1) is q
    rho = math.sin(math.pi * 0.2)
    out = ZZ.degrade_quantities(q, rho, 1)
    err = float(np.mean(out["direction"] != d))
    assert abs(err - norm.cdf(-rho / math.sqrt(1 - rho * rho))) < 0.08                # persistent errors: the rate matches in expectation
    room = np.arange(n, dtype=float) % 37
    out2 = ZZ.degrade_quantities({**q, "room": np.stack([room, room], axis=1)}, rho, 2)
    assert abs(np.corrcoef(out2["room"][:, 0], room)[0, 1] - rho) < 0.1                 # the room regression carries correlation rho
    # causal: a V (down then up) confirms the up leg only after the reversal threshold
    idx = pd.date_range("2021-01-04", periods=400, freq="h", tz="UTC")
    c = np.r_[np.linspace(1.20, 1.10, 200), np.linspace(1.10, 1.20, 200)]
    bars = pd.DataFrame({"OPEN": c, "HIGH": c + 1e-4, "LOW": c - 1e-4, "CLOSE": c}, index=idx)
    cq = ZZ.causal_quantities(bars, np.full(400, 1e-3), levels=("rev200",), roles={"coarse": ("rev200",), "middle": (), "fine": ()})
    up = np.flatnonzero(cq["direction"] > 0)
    assert len(up) and up[0] > 200 and up[0] <= 200 + 50                              # 200 pips from the valley take ~40 bars here


def test_every_proposed_open_is_placed_or_counted_and_ties_are_not_deferred():
    """W14 (M_LHS_48): the best-of-day gate deferred an open whose score tied the day's best within float rounding
    (0.7499999999999943 vs 0.75) in the mirror only, and the drop showed in no engine count."""
    from app.short_horizon.expert_guided import exec_variants as XV
    import types
    env = types.SimpleNamespace(v=np.arange(10), business_day=lambda k: 0, entry_gate=lambda b: (True, "OK"))
    seq = iter([(1, 1.0, 1.0, 0.75), (2, 1.0, 1.0, 0.7499999999999943), (1, 1.0, 1.0, 0.70)])
    g = XV.BestOfDayGate(lambda bar, r, st, e: next(seq), observe_bars=0)
    flat = np.array([0])
    placed = [g(b, b, flat, env)[0] for b in range(3)]
    assert placed[:2] == [1, 2] and placed[2] == 0                                   # the tie is placed; a real lower score is deferred
    assert g.stats["placed"] + g.stats["deferred_below_best"] + g.stats["deferred_observing"] + g.stats["below_threshold"] == 3   # every proposal counted


def test_closer_head_is_market_only_and_its_noise_grows_with_the_lead():
    """The closer's output contract (C3 closer_head): the head row takes no position argument, the strategy's
    advantage never reads the trade's own path, and the quality noise grows with the lead (sqrt(h / 24))."""
    import inspect
    from tools import strategy_output_simulation as S
    from app.short_horizon.expert_guided import c3 as C3
    assert list(inspect.signature(S.BCHead.row).parameters) == ["self", "b", "si"]
    import ast
    import textwrap
    def code_of(f):
        fn = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
        return ast.unparse([n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))])   # the code, not the docstring
    code = code_of(S.SimCloser.advantage_full) + code_of(S.SimCloser.head_core)
    assert "self.path(" not in code and "path_advantage" not in code and "math.sqrt(h / H_REF_BARS)" in code and "_ar_noise" in code
    assert "self.head_core(" in code_of(S.SimCloser.advantage_full)
    cal = code_of(S.SimCloser.calibration_pairs)                       # the filter's calibration reads the CALIBRATION block only
    assert "CAL_BLOCK" in cal and "BLOCK ==" not in cal and "self.path(" not in cal
    assert S.H_REF_BARS == 24 and "first-passage" in C3.default_contract()["closer_head"]["name"]


def test_close_rule_at_quality_one_reproduces_the_perfect_head_rule_and_persistence_counts_ticks():
    """At q1.00 the head returns s_hat = 0, so A_hat - z_close x s_hat >= -margin is A >= -margin for every z_close and,
    with k_persist = 1, the decision of the perfect-head rule on every tick; k_persist > 1 needs that many consecutive
    true ticks of the same position (a false tick resets the count)."""
    import types
    from app.short_horizon.expert_guided import strategy_zz as ZZ
    from app.short_horizon.expert_guided import env as E

    class Stub:
        def __init__(self, a, s):
            self.a, self.s = list(a), list(s)

        def advantage(self, r, bar, pos):
            return self.a[bar]

        def advantage_full(self, r, bar, pos):
            return (self.a[bar], self.s[bar]) + (0.0,) * 5 + (1.0,)

        def value(self, r, bar, pos):
            return 0.0, 0.0

    def run(a, s, menu=False):
        from app.short_horizon.expert_guided import menu as M
        pol = types.SimpleNamespace(s=types.SimpleNamespace(early_close=True), t={"closer": Stub(a, s)}, positions={}, closes=[],
                                    cells=list(M.MENU_CELLS) if menu else [(1.0, 1.0)])   # v2.2: the rule reads the policy's profile (menu = full_f4)
        pol._menu = types.MethodType(ZZ.PolicyC3Z._menu, pol)
        return [ZZ.PolicyC3Z.close_now(pol, 0, 0, b, pos) for b in range(len(a))]

    pos = E.Position(1, 1000, 10, 1.1, 1.09, 1.12, 0.001, 0, {})
    a = [0.3, -0.5, -0.01, 0.0, -0.03, 0.2, -0.019, -2.0]
    old = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(closer_mode=1.0, closer_adv_margin=0.02)
        for z in (0.0, 1.0, 3.0):
            for f in (0.0, 1.0):                                      # the closer filter reads an exact head (s_hat = 0) as is
                ZZ.PARAMS.update(z_close=z, k_persist=1, closer_filter=f)
                assert run(a, [0.0] * len(a)) == [x >= -0.02 for x in a]
        ZZ.PARAMS.update(closer_filter=0.0)
        ZZ.PARAMS.update(z_close=1.0, k_persist=2)
        assert run(a, [0.0] * len(a)) == [False, False, False, True, False, False, True, False]
        ZZ.PARAMS.update(z_close=1.0, k_persist=1)
        assert run(a, [0.5] * len(a)) == [x - 0.5 >= -0.02 for x in a]
        assert run(a, [0.0] * len(a), menu=True) == [x > 0 for x in a]      # full_f4 (ruling C): ties HOLD, the margin is inert
        assert run(a, [0.1] * len(a), menu=True) == [x - 0.1 > 0 for x in a]
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)


def test_closer_filter_is_estimated_on_calibration_pairs_and_its_posterior_averages_the_noise():
    """CloserFilterModel.estimate recovers the observation model of a head output A_hat = alpha + beta A + eps (eps
    AR(1)) from (bar, A_hat, s_hat, h, A_true) calibration series; the Kalman posterior of a constant A over a run of
    noisy outputs is closer to A than the last output alone."""
    from app.short_horizon.expert_guided import strategy_zz as ZZ
    rng = np.random.default_rng(3)
    phi, beta, alpha, sd = 0.6, 0.5, -0.1, 0.3
    series = []
    for _ in range(4):
        n = 1500
        A = np.cumsum(rng.normal(0, 0.05, n)) * 0.2 + rng.normal(0, 1.0)
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = phi * e[t - 1] + math.sqrt(1 - phi * phi) * sd * rng.standard_normal()
        e[0] = sd * rng.standard_normal()
        Ah = alpha + beta * A + e
        series.append(np.column_stack([np.arange(n), Ah, np.full(n, 0.1), np.full(n, 12.0), A]))
    m = ZZ.CloserFilterModel.estimate(series)
    a_, b_, s_ = m.obs(12.0)
    assert abs(m.phi - phi) < 0.05 and abs(b_ - beta) < 0.05 and abs(a_ - alpha) < 0.1 and abs(s_ - sd) < 0.03
    A_true, st, est = -0.4, {}, None
    errs_last, errs_post = [], []
    for trial in range(200):
        st = {}
        eps = sd * rng.standard_normal()
        for t in range(30):
            eps = phi * eps + math.sqrt(1 - phi * phi) * sd * rng.standard_normal()
            y = alpha + beta * A_true + eps
            est, _ = ZZ.filter_step(st, 0, m, y, 12.0, 1.0, t)
        errs_last.append(abs((y - alpha) / beta - A_true))
        errs_post.append(abs(est - A_true))
    assert np.mean(errs_post) < np.mean(errs_last)


def test_compact_ladder_levels_change_no_ladder_quantity():
    """ZZ.compact_levels (memory, 2026-10-28): narrow exact dtypes and dropped unread arrays leave every ladder and causal
    quantity bit-identical."""
    from app.short_horizon.expert_guided import strategy_zz as ZZ
    rng = np.random.default_rng(5)
    n = 3000
    c = 1.1 + np.cumsum(rng.normal(0, 4e-4, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 2e-4, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 2e-4, n))
    bars = pd.DataFrame({"OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c}, index=_idx(n))
    sig = np.full(n, 6e-4)
    lv = ZZ.ladder_levels(bars, sig)
    cp = ZZ.compact_levels(lv)
    assert sum(a.nbytes for r in cp["arr"].values() for a in r.values()) < 0.6 * sum(a.nbytes for r in lv["arr"].values() for a in r.values())
    for fn in (ZZ.ladder_quantities, ZZ.causal_quantities):
        a, b = fn(bars, sig, cached=lv), fn(bars, sig, cached=cp)
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                assert v.dtype == b[k].dtype and np.array_equal(v, b[k], equal_nan=v.dtype.kind == "f"), k


def test_cost_view_equals_the_array_expression_bit_for_bit():
    from tools import strategy_output_simulation as S
    rng = np.random.default_rng(9)
    sp = rng.uniform(0, 3e-4, 5000)
    idx = rng.integers(0, 5000, 700)
    for adds in ((2e-5, 3.5e-5), (3.5e-5,)):
        arr = sp / 2
        for a in adds:
            arr = arr + a
        v = S._CostView(sp, *adds)
        assert np.array_equal(v[idx], arr[idx]) and v[17] == arr[17] and np.array_equal(v[np.arange(10, 40)], arr[10:40])


def test_generation0_degeneracy_gate():
    """The standing pre-launch gate (coordinator order 2026-10-28): after generation 0 the search stops on its own when
    every candidate has the same failing count, a witness fails in every candidate in every world where it is evaluated,
    or a NOT_EVALUABLE verdict is in every candidate; shared failures in only some worlds are reported, not a stop."""
    from tools import strategy_param_deap as D
    worlds = [("A@q1.00", 1.0), ("M_X@q1.00", 1.0), ("M_Y@q1.00", 1.0), ("B@q0.70", 0.7), ("C@q0.70", 0.7)]

    def rec(nf, fails):
        return {"n_failing_witnesses": nf, "per_world": [{"key": k, "quality": q, "fails": list(fails.get(k, []))} for k, q in worlds]}

    w14 = [rec(10 + i, {"M_X@q1.00": ["W14_mirror"], "M_Y@q1.00": ["W14_mirror"], "A@q1.00": ["W7_best_cell"] if i % 2 else []}) for i in range(4)]
    d = D.degenerate_gen0(w14)
    assert d["degenerate"] and any(r.startswith("(b) W14_mirror") for r in d["reasons"])
    same_count_other_sets = [rec(7, {"A@q1.00": ["W7_best_cell"] if i else ["W5_reverse"]}) for i in range(4)]
    d = D.degenerate_gen0(same_count_other_sets)
    assert not d["degenerate"] and d["same_count_logged_only"] == {"count": 7, "distinct_failing_sets": 2}
    same = [rec(7, {"A@q1.00": ["W7_best_cell"], "B@q0.70": ["D7_E4_BUY"]}) for i in range(4)]
    d = D.degenerate_gen0(same)
    assert d["degenerate"] and d["reasons"][0].startswith("(a)")
    zero = [rec(0, {}) for i in range(4)]
    assert not D.degenerate_gen0(zero)["degenerate"]                    # every candidate feasible: never a stop
    shared = [rec(10 + i, {"B@q0.70": ["D7_regret_BUY"], "C@q0.70": ["D7_regret_BUY"] if i % 2 else []}) for i in range(4)]
    d = D.degenerate_gen0(shared)
    assert not d["degenerate"] and d["every_candidate_some_worlds"] == [{"witness": "D7_regret_BUY", "worlds_failing_in_every_candidate": 1, "worlds_evaluated": 2}]
    ne = [rec(3 + i, {"M_X@q1.00": ["W14_mirror_NOT_EVALUABLE"]}) for i in range(3)]
    assert D.degenerate_gen0(ne)["degenerate"]
    healthy = [rec(i, {"A@q1.00": ["W7_best_cell"]} if i % 2 else {}) for i in range(5)]
    assert not D.degenerate_gen0(healthy)["degenerate"]
    assert not D.degenerate_gen0(healthy[:1])["degenerate"]


def test_warm_start_reuses_genomes_only_and_reevaluates_every_point(tmp_path):
    """Run 4b's restart (coordinator order 2026-10-28): an earlier ARCHIVE seeds generation 0 with GENOMES only; every
    warm point is evaluated afresh by the pool, no cached fitness or violation count is carried over."""
    from tools import strategy_param_deap as D
    from app.short_horizon.expert_guided import strategy_params as SP
    base = SP.default_doc()
    old = tmp_path / "OLD_ARCHIVE.jsonl"
    with open(old, "w") as fh:
        for i, v in enumerate((0.45, 0.30, 0.20)):
            fh.write(json.dumps({"generation": 0, "params": {k: base[k] for k in D.GENOME} | {"open_rate_l": v}, "fitness_sharpe": 99.0 + i, "n_failing_witnesses": 1000}) + "\n")
    init = D.warm_population(old, base, 3)
    assert all(isinstance(x, list) and len(x) == len(D.GENOME) and all(isinstance(u, float) for u in x) for x in init)

    class FakePool:
        calls = 0

        def evaluate(self, doc):
            FakePool.calls += 1
            nf = FakePool.calls                                       # fresh, distinct values: never the archive's 1000 / 99+
            return {"fitness_sharpe": 1.0 + nf / 10, "n_failing_witnesses": nf, "feasible": False, "per_world": [{"key": "A@q1.00", "quality": 1.0, "fails": ["W7_best_cell"] * nf}, {"key": "B@q1.00", "quality": 1.0, "fails": []}],
                    "wall_sec": 0.0, "worker_maxrss_mb": []}

    out = tmp_path / "deap"
    out.mkdir()
    res = D.run_search(FakePool(), base, out, population=3, generations=0, seed=1, log=lambda *_: None, init=init)
    recs = [json.loads(x) for x in (out / "ARCHIVE.jsonl").read_text().splitlines()]
    assert FakePool.calls == 3 and len(recs) == 3 and all(r["generation"] == 0 for r in recs)
    assert sorted(r["n_failing_witnesses"] for r in recs) == [1, 2, 3] and all(r["fitness_sharpe"] < 2.0 for r in recs)
    assert res["best"]["n_failing_witnesses"] == 1 and not (out / "FAILED_DEGENERATE_GEN0").exists()


def test_trade_summary_record_schema_carries_sigma_per_trade_win_rate_and_starting_balance():
    """Standing rule 2026-10-29: every record stores net sigma per trade (mean and median), the win rate and the
    per-world starting balance, plus the executed SL / TP pips and holding bars (p10 / p50 / p90); reports show the
    median net per family, never the compounded mean."""
    sig = np.full(100, 0.002)

    def t(eb, xb, net):
        return {"entry_bar_abs": eb, "exit_bar_abs": xb, "net_usd": net, "units": 1000, "entry_price": 1.1, "stop_loss": 1.095, "take_profit": 1.11}

    s = BM.trade_summary_from([t(10, 22, 4.0), t(30, 36, -2.0), t(50, 80, 10.0)], sig, 10_000.0)
    assert s["n"] == 3 and s["starting_balance_usd"] == 10_000.0
    assert abs(s["net_sigma_per_trade_mean"] - 2.0) < 1e-12 and abs(s["net_sigma_per_trade_median"] - 2.0) < 1e-12   # (4 - 2 + 10) / 3 / 1000 / 0.002
    assert abs(s["win_rate"] - 2 / 3) < 1e-12 and abs(s["sl_pips_p10_p50_p90"][1] - 50.0) < 1e-9 and abs(s["tp_pips_p10_p50_p90"][1] - 100.0) < 1e-9
    assert s["hold_bars_p10_p50_p90"][1] == 12.0
    assert BM.trade_summary_from([], sig, 10_000.0) == {"n": 0, "starting_balance_usd": 10_000.0}
    from tools import strategy_closer_ladder_report as CLR
    side = {"E4": 0.0, "regret_norm": 0.0, "mix": {"early": 0.0}, "early_close_value_norm": 0.0, "sl_destined_closed_early": 0, "closes": 0}
    run = {"sharpe": 1.0, "net_usd": 7.0, "trades": 3, "mean_hold_bars": 2.0, "trade_summary": s, "BUY": dict(side), "SELL": dict(side)}
    bayes = {"sharpe": 1.0, "net_usd": 5.0, "BUY": dict(side), "SELL": dict(side)}
    v = CLR.verdicts([{"cell": "x", "seed": 0, "rows": [{"quality": 1.0, "phi": 0.0, "bayes": bayes, "rules": {"1.0:1": run}, "strategy": run}]}], "1.0:1")
    m = v["summary"]["q1.00_phi0"]["mean"]
    assert m["net_usd_median"] == 7.0 and m["bayes_net_usd_median"] == 5.0 and abs(m["net_sigma_per_trade"] - 2.0) < 1e-12
    assert "median net USD per family" in CLR.markdown(v)
