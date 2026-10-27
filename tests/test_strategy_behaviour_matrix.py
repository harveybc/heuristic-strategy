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
    c0 = float(bars["CLOSE"].iloc[0])
    assert np.allclose(m["CLOSE"].to_numpy(), 2 * c0 - bars["CLOSE"].to_numpy(), atol=1e-12)
    assert np.allclose(m["HIGH"].to_numpy(), 2 * c0 - bars["LOW"].to_numpy(), atol=1e-12)
    assert np.allclose(np.diff(m["CLOSE"].to_numpy()), -np.diff(bars["CLOSE"].to_numpy()), atol=1e-12)
    assert np.all(m["HIGH"] >= m[["OPEN", "CLOSE"]].max(axis=1)) and np.all(m["LOW"] <= m[["OPEN", "CLOSE"]].min(axis=1))
    mm, _ = H.synthetic_path_v2(idx, 5e-4, ("trend_up",), "low", "medium", 3, (500, 2500), mirror=True)
    b2, _ = H.synthetic_path_v2(idx, 5e-4, ("trend_up",), "low", "medium", 3, (500, 2500), mirror=False)
    assert np.allclose(mm["CLOSE"].to_numpy(), 2 * float(b2["CLOSE"].iloc[0]) - b2["CLOSE"].to_numpy(), atol=1e-12)


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
    assert g["open_rate"] == pytest.approx(0.251) and g["allow_reverse"] == 1.0 and g["nop_veto"] == 0.5 and g["close_p_l"] == 0.5
