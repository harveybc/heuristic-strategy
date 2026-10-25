"""Stage C2G (owner 2026-10-25, "Sí, como brazo extra"): C2 plus ONE change, the closers alone also read the geometry
of their own open order. The C2G contract and its state-slot refusals (every other slot, every other specialist), the
geometry with known answers, the on-demand closer evaluator (per order, cached, restricted, ensembled exactly as the
per-row table path), the policy and the exit tables with geometry, C2 left byte-identical, the launcher's WF_C2G
block and emission, the audit's buckets and the C2G report rules. No data, no model fit."""
import copy
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import calendar as CAL
from app.short_horizon.expert_guided import action_contract as ACT
from app.short_horizon.expert_guided import c2 as C2
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import outcomes as OC

HOUR = pd.Timedelta(hours=1)
SIG = 0.0010
ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "docs/short_horizon/clean_round/contracts"


# ----------------------------------------------------------------------------- contract
def test_committed_c2g_contract_is_the_sealed_default_and_differs_from_c2_only_by_the_geometry():
    g = json.loads((CONTRACTS / "ACTION_CONTRACT_C2G.json").read_text())
    c = json.loads((CONTRACTS / "ACTION_CONTRACT_C2.json").read_text())
    assert g == C2.default_contract_geometry() and C2.validate(g) == [] and C2.encoding_audit(g)["verdict"] == "PASS"
    assert c == C2.default_contract() and C2.validate(c) == []
    spec = json.loads((ROOT / "docs/short_horizon/clean_round/CLEAN_ROUND_SPEC.json").read_text())
    assert spec["walk_forward_c2"]["action_contract"]["identity"] == c["identity"] and spec["walk_forward_c2g"]["action_contract"]["identity"] == g["identity"]
    assert C2.is_c2(g) and C2.is_c2g(g) and C2.is_c2(c) and not C2.is_c2g(c)
    assert g["closer_order_geometry"]["order"] == list(C2.GEOMETRY) and g["closer_order_geometry"]["applies_to"] == ["close_l", "close_s"]
    assert [g["specialists"][m]["state_slots"] for m in C2.SPECIALISTS] == [[], [], [], list(C2.GEOMETRY), list(C2.GEOMETRY)]
    # everything but the declared C2G additions is C2's contract
    strip = lambda d: {k: v for k, v in d.items() if k not in ("schema", "identity", "design", "closer_order_geometry", "specialists", "constraints")}  # noqa: E731
    assert strip(g) == strip(c)
    assert {m: {k: v for k, v in g["specialists"][m].items() if k != "state_slots"} for m in C2.SPECIALISTS} == c["specialists"]


def _sealed(fn, base=None):
    d = copy.deepcopy(base or C2.default_contract_geometry())
    fn(d)
    d["identity"] = None
    return C2.validate(ACT.seal(d))


@pytest.mark.parametrize("mut, code", [
    (lambda d: d["specialists"]["close_l"]["state_slots"].append("balance"), "STATE_SLOT_NOT_ALLOWED:close_l:balance"),
    (lambda d: d["specialists"]["close_s"]["state_slots"].append("equity"), "STATE_SLOT_NOT_ALLOWED:close_s:equity"),
    (lambda d: d["specialists"]["close_l"]["state_slots"].append("pnl_history"), "STATE_SLOT_NOT_ALLOWED:close_l:pnl_history"),
    (lambda d: d["specialists"]["close_l"]["state_slots"].append("entries_today_share"), "STATE_SLOT_NOT_ALLOWED:close_l:entries_today_share"),
    (lambda d: d["specialists"]["close_s"]["state_slots"].append("agent_outputs"), "STATE_SLOT_NOT_ALLOWED:close_s:agent_outputs"),
    (lambda d: d["specialists"]["close_s"]["state_slots"].append("can_enter"), "STATE_SLOT_NOT_ALLOWED:close_s:can_enter"),
    (lambda d: d["specialists"]["close_l"]["state_slots"].append("unrealized_sigma"), "STATE_SLOT_NOT_ALLOWED:close_l:unrealized_sigma"),
    (lambda d: d["specialists"]["open_l"].__setitem__("state_slots", ["dist_sl_sigma"]), "GEOMETRY_ON_NON_CLOSER:open_l:dist_sl_sigma"),
    (lambda d: d["specialists"]["open_s"].__setitem__("state_slots", ["side"]), "GEOMETRY_ON_NON_CLOSER:open_s:side"),
    (lambda d: d["specialists"]["nop"].__setitem__("state_slots", ["hours_held"]), "GEOMETRY_ON_NON_CLOSER:nop:hours_held"),
    (lambda d: d["specialists"]["open_l"].__setitem__("state_slots", ["balance"]), "STATE_SLOT_NOT_ALLOWED:open_l:balance"),
    (lambda d: d["specialists"]["close_l"].__setitem__("state_slots", ["dist_sl_sigma", "side"]), "C2G_CLOSER_GEOMETRY:close_l"),
    (lambda d: d["closer_order_geometry"].__setitem__("order", ["side", "dist_sl_sigma", "dist_tp_sigma", "hours_held"]), "C2G_GEOMETRY_BLOCK"),
    (lambda d: d["closer_order_geometry"].__setitem__("applies_to", ["close_l", "close_s", "open_l"]), "C2G_GEOMETRY_BLOCK"),
    (lambda d: d["closer_order_geometry"].__setitem__("hours_scale", 1.0), "C2G_GEOMETRY_BLOCK"),
    (lambda d: d["constraints"].__setitem__("closer_inputs", "recipe_window"), "C2G_CONSTRAINTS"),
    (lambda d: d["constraints"].__setitem__("agent_state", "full"), "C2_CONTRACT_CONSTRAINTS"),
])
def test_c2g_refuses_every_other_state_slot_and_geometry_outside_the_closers(mut, code):
    errs = _sealed(mut)
    assert any(e.startswith(code) for e in errs), (code, errs)


def test_a_c2_contract_carries_no_state_slot_and_no_geometry_block():
    errs = _sealed(lambda d: d["specialists"]["close_l"].__setitem__("state_slots", list(C2.GEOMETRY)), C2.default_contract())
    assert any(e.startswith("STATE_SLOT_NOT_ALLOWED:close_l:dist_sl_sigma") for e in errs)
    errs = _sealed(lambda d: d.__setitem__("closer_order_geometry", {"order": list(C2.GEOMETRY)}), C2.default_contract())
    assert any(e.startswith("STATE_SLOT_NOT_ALLOWED:closer_order_geometry") for e in errs)


def test_c2g_encoding_audit_still_refuses_one_continuous_output_cut_into_classes():
    d = C2.default_contract_geometry()
    d["specialists"]["close_l"]["decision"] = {"encoding": "tanh_intervals", "bins": [-1, -0.5, 0, 0.5, 1], "n_classes": 5, "classes": list("abcde"),
                                               "output_dim": 1, "loss": "mse"}
    errs = C2.encoding_audit(d)["errors"]
    assert any(e.startswith("CATEGORICAL_FROM_CONTINUOUS:close_l") for e in errs) and any(e.startswith("HEAD_DIM_MISMATCH:close_l") for e in errs)


# ----------------------------------------------------------------------------- geometry
def test_order_geometry_known_answer_long_and_short():
    v = pd.date_range("2022-01-03", periods=10, freq="h", tz="UTC").as_unit("ns").asi8
    close = np.full(10, 1.0)
    pos = types.SimpleNamespace(direction=1, entry_bar=2, sl=0.99, tp=1.02, sigma=0.01)
    g = C2.order_geometry(pos, [2, 5], close, v)
    assert g.shape == (2, 4) and g.dtype == np.float32
    assert np.allclose(g[1], [1.0, 2.0, 4.0 / 24.0, 1.0], atol=1e-6)        # decided after bar 5: 4 h since the fill bar opened
    assert np.allclose(g[0], [1.0, 2.0, 1.0 / 24.0, 1.0], atol=1e-6)        # the fill bar itself: 1 h
    short = types.SimpleNamespace(direction=-1, entry_bar=2, sl=1.015, tp=0.97, sigma=0.01)
    assert np.allclose(C2.order_geometry(short, [5], close, v)[0], [1.5, 3.0, 4.0 / 24.0, -1.0], atol=1e-6)


def test_x_geo_layout_is_window_then_zero_state_then_geometry():
    Xw = np.arange(12, dtype=np.float32).reshape(4, 3)
    G = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32)
    X = C2.X_geo(Xw, [1, 3], 9, G)
    assert X.shape == (2, 3 + 9 + 4)
    assert np.array_equal(X[:, :3], Xw[[1, 3]]) and not X[:, 3:12].any() and np.array_equal(X[:, 12:], G)


# ----------------------------------------------------------------------------- the on-demand closer evaluator
class StubCloser:
    """A closer whose logit margin is the input's dist_sl column and whose remaining value is dist_tp / 10."""

    def __call__(self, X):
        X = np.asarray(X, float)
        z = -X[:, -4]                                          # nearer the SL -> higher P(SL first)
        lg = np.c_[np.zeros(len(X)), z]
        return lg, (X[:, -3] / 10.0)[:, None]


def stub_cx(n=30):
    idx = pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")
    close = 1.0 + 0.001 * np.arange(n)
    bars = pd.DataFrame({"OPEN": close, "HIGH": close, "LOW": close, "CLOSE": close}, index=idx)
    ds = types.SimpleNamespace(bars=bars, row_bar=np.arange(n), period=np.array(["INNER_A"] * n))
    return types.SimpleNamespace(ds=ds, Xw=np.zeros((n, 2), np.float32), rob=np.arange(n))


def members(k=1):
    cal = {m: {"a": 1.0 + 0.5 * i, "b": 0.1 * i} for i, m in enumerate(C2.CLOSERS)}
    return [({m: StubCloser() for m in C2.CLOSERS}, {m: {"a": c["a"] * (j + 1), "b": c["b"] - 0.05 * j} for m, c in cal.items()}) for j in range(k)]


def direct(cx, mb, pos, r, ens_cal=None):
    G = C2.order_geometry(pos, [r], cx.ds.bars["CLOSE"].to_numpy(float), cx.ds.bars.index.as_unit("ns").asi8)
    X = C2.X_geo(cx.Xw, [r], 9, G)
    m = "close_l" if pos.direction > 0 else "close_s"
    ps, rm = [], []
    for pols, cal in mb:
        z, rg = ACT.module_outputs(pols[m], X)
        ps.append(ACT.platt(z, cal[m]["a"], cal[m]["b"])[0])
        rm.append(rg[0, 0])
    p = float(np.mean(ps))
    if ens_cal is not None:
        p = float(ACT.platt(ACT.logit(np.array([p])), ens_cal[m]["a"], ens_cal[m]["b"])[0])
    return float(np.mean(rm)), p


def test_closer_values_equal_a_direct_computation_and_are_cached_per_order():
    cx = stub_cx()
    mb = members(1)
    cg = C2.CloserGeo(cx, mb, 9)
    pos = types.SimpleNamespace(direction=1, entry_bar=3, sl=0.995, tp=1.02, sigma=0.002)
    for r in (3, 7, 12):
        assert cg.value(r, r, pos) == pytest.approx(direct(cx, mb, pos, r), abs=1e-12)
    assert len(cg.cache) == 1 and len(next(iter(cg.cache.values()))) == 1     # one order, one batched segment
    other = types.SimpleNamespace(direction=-1, entry_bar=5, sl=1.01, tp=0.99, sigma=0.002)
    assert cg.value(9, 9, other) == pytest.approx(direct(cx, mb, other, 9), abs=1e-12) and len(cg.cache) == 2
    with pytest.raises(ValueError, match="C2_ROW_OUTSIDE_TABLE"):              # a row before the order was filled
        cg.value(2, 2, types.SimpleNamespace(direction=1, entry_bar=4, sl=0.99, tp=1.02, sigma=0.002))
    with pytest.raises(ValueError, match="C2G_ROW_BAR_MISMATCH"):
        cg.value(8, 9, pos)


def test_restricted_view_refuses_rows_outside_and_shares_the_numbers():
    cx = stub_cx()
    cg = C2.CloserGeo(cx, members(1), 9)
    keep = np.zeros(30, bool)
    keep[:10] = True
    rv = cg.restricted(keep)
    pos = types.SimpleNamespace(direction=1, entry_bar=3, sl=0.995, tp=1.02, sigma=0.002)
    assert rv.value(5, 5, pos) == cg.value(5, 5, pos)
    with pytest.raises(ValueError, match="C2_ROW_OUTSIDE_TABLE"):
        rv.value(15, 15, pos)
    t = C2.restrict({"pn": np.zeros(30), "closer": cg}, np.arange(10))
    with pytest.raises(ValueError, match="C2_ROW_OUTSIDE_TABLE"):
        t["closer"].value(15, 15, pos)


def test_ensemble_closer_equals_mean_of_calibrated_then_one_platt_refit():
    cx = stub_cx()
    mb = members(3)
    tables = [{**{f: np.zeros((30, 2)) for f in ("rem", "psl", "ptp")}, "net": np.zeros((30, 2, len(OC.CELLS))), "pn": np.zeros(30),
               "closer": C2.CloserGeo(cx, [m_], 9)} for m_ in mb]
    ens = C2.mean_tables(tables)
    assert len(ens["closer"].members) == 3
    pos = types.SimpleNamespace(direction=-1, entry_bar=4, sl=1.012, tp=0.995, sigma=0.002)
    assert ens["closer"].value(8, 8, pos) == pytest.approx(direct(cx, mb, pos, 8), abs=1e-12)
    cal = {m: {"a": 0.7, "b": -0.2} for m in C2.SPECIALISTS}
    back = C2.calibrate_table(C2.to_raw(ens), cal)                          # the audit's ensemble path
    assert back["closer"].value(8, 8, pos) == pytest.approx(direct(cx, mb, pos, 8, cal), abs=1e-12)
    G = C2.order_geometry(pos, [8], cx.ds.bars["CLOSE"].to_numpy(float), cx.ds.bars.index.as_unit("ns").asi8)
    z = ens["closer"].raw_logit_rows("close_s", np.array([8]), G)
    assert float(z[0]) == pytest.approx(float(ACT.logit(np.array([direct(cx, mb, pos, 8)[1]]))[0]), abs=1e-9)


# ----------------------------------------------------------------------------- policy: C2 unchanged, C2G reads the order
def table(n=3):
    nc = len(OC.CELLS)
    t = {"net": np.full((n, 2, nc), -1.0), "ptp": np.full((n, 2), 0.5), "pn": np.full(n, 0.1), "rem": np.full((n, 2), 0.5), "psl": np.full((n, 2), 0.1)}
    t["net"][0, 1, 7] = 0.8
    return t


class FixedCloser:
    def __init__(self, rem, psl):
        self.rem, self.psl, self.calls = rem, psl, []

    def value(self, r, bar, pos):
        self.calls.append((r, bar, pos))
        return self.rem, self.psl

    def restricted(self, keep):
        return self


def test_policy_without_a_closer_key_is_c2s_and_with_one_reads_the_open_order():
    st_long = np.zeros(9)
    st_long[0] = 1
    s = C2.StrategyC2(0.2, 0.1, 0.6, False, None)
    env = types.SimpleNamespace(pos="the-open-order")
    t = table()
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, env)[0] == E.HOLD        # C2: per-row table values
    t["rem"][1, 0] = -0.3
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, env)[0] == E.CLOSE
    t = table()
    t["rem"][:], t["psl"][:] = np.nan, np.nan                                # C2G tables carry no per-row closer columns
    fc = FixedCloser(0.5, 0.1)
    t["closer"] = fc
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, env)[0] == E.HOLD
    assert fc.calls == [(1, 1, "the-open-order")]
    fc.psl = 0.7
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, env)[0] == E.CLOSE
    fc.psl, fc.rem = 0.1, -0.3
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, env)[0] == E.CLOSE


def test_exit_prob_asks_for_the_position_only_under_c2g():
    cx = types.SimpleNamespace(rob=np.arange(3))
    t = table()
    f = C2.exit_prob(cx, t)
    assert not getattr(f, "wants_position", False) and f(1, np.r_[1, np.zeros(8)]) == t["psl"][1, 0]
    t["closer"] = FixedCloser(0.0, 0.42)
    g = C2.exit_prob(cx, t)
    assert g.wants_position and g(1, np.zeros(9), "pos") == 0.42


# ----------------------------------------------------------------------------- exit tables with geometry (engine)
def synthetic():
    stamps, t = [], pd.Timestamp("2020-01-05 21:00", tz="UTC")
    for _ in range(3):
        stamps += [t + k * HOUR for k in range(120)]
        t = stamps[-1] + 49 * HOUR
    n = len(stamps)
    bars = pd.DataFrame({"OPEN": 1.1, "HIGH": 1.1, "LOW": 1.1, "CLOSE": 1.1}, index=pd.DatetimeIndex(stamps, name="DATE_TIME"))
    segs = CAL.derive_segments(bars.index, max_gap_hours=24)
    cal = CAL.load_calendar({"calendar_id": "TEST", "version": "t1", "segments": segs, "entry_delay_after_open_hours": 12,
                             "force_flat_before_close_hours": 12, "entry_cutoff_before_flat_hours": 0})
    sig, sr = np.full(n, SIG), np.full(n, 3 * SIG)
    probe = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + HOUR, E.EnvConfig())
    b = int(np.searchsorted(probe.v, cal.segments[0].operational_start.value)) - 1 + 5
    m, new = b + 4, 1.1 - 4 * SIG
    bars.iloc[m] = [1.1, 1.1, new, new]
    for k in range(m + 1, n):
        bars.iloc[k] = [new, new, new, new]
    ds = types.SimpleNamespace(bars=bars, cal=cal, sigma=sig, sigma_rule=sr)
    return types.SimpleNamespace(ds=ds), b


def test_exit_table_geometry_is_additive_and_matches_the_order():
    cx, b = synthetic()
    env = OC.make_env(cx, E.EnvConfig(), end=cx.ds.bars.index[-1] + HOUR)
    env.step(b, E.LONG, 1.5, 2.0)
    pos = env.pos
    visits = [(10, b + 1, pos), (11, b + 2, pos)]
    end = cx.ds.bars.index[-1] + HOUR
    plain = OC.exit_table(cx, E.EnvConfig(), visits, end=end)
    gt = OC.exit_table(cx, E.EnvConfig(), visits, end=end, geometry_fn=lambda p, bar: C2.order_geometry(
        p, [bar], cx.ds.bars["CLOSE"].to_numpy(float), cx.ds.bars.index.as_unit("ns").asi8)[0])
    assert "geo" not in plain and set(gt) == set(plain) | {"geo"}
    assert all(np.array_equal(plain[k], gt[k]) for k in plain)              # the C2 columns are untouched
    assert gt["geo"].shape == (2, 4) and np.allclose(gt["geo"][:, 0], 2.0, atol=1e-6) and np.allclose(gt["geo"][:, 1], 1.5, atol=1e-6)
    assert gt["geo"][1, 2] > gt["geo"][0, 2] and np.all(gt["geo"][:, 3] == 1.0)


def test_exit_sets_unchanged_and_exit_sets_geo_aligned():
    idx = pd.date_range("2022-01-01", periods=10, freq="h", tz="UTC")
    cx = types.SimpleNamespace(ds=types.SimpleNamespace(bars=pd.DataFrame(index=idx), period=np.array(["TRAIN"] * 5 + ["INNER_A"] * 5),
                                                        meta={"periods": {"TRAIN": ("2022-01-01", "2022-01-01 05:00"), "INNER_A": ("2022-01-01 05:00", "2022-01-02")}}))
    xt = {"row": np.array([0, 1, 2, 3, 6]), "bar": np.array([0, 1, 2, 3, 6]), "side": np.array([0, 1, 0, 0, 0]),
          "remaining": np.array([0.1, 0.2, np.nan, 0.4, 0.5]), "sl_first": np.array([1, 0, 1, 0, 1]),
          "exit_bar": np.array([1, 2, 3, 8, 7]), "censored": np.zeros(5, bool), "geo": np.arange(20, dtype=np.float32).reshape(5, 4)}
    r, y, R = OC.exit_sets(cx, xt, "TRAIN", 0)
    assert r.tolist() == [0] and y.tolist() == [1] and R[:, 0].tolist() == pytest.approx([0.1])  # row 2 NaN, row 3 resolves late
    r2, y2, R2, G = C2.exit_sets_geo(cx, xt, "TRAIN", 0)
    assert r2.tolist() == r.tolist() and y2.tolist() == y.tolist() and np.array_equal(R2, R) and np.array_equal(G, xt["geo"][[0]])
    empty = C2.exit_sets_geo(cx, {k: v[:0] for k, v in xt.items()}, "TRAIN", 0)
    assert [len(x) for x in empty] == [0, 0, 0, 0] and empty[3].shape == (0, 4)


def test_saved_sets_carry_the_geometry_and_rebuild_the_closer_input(tmp_path):
    named = {"INNER_A": {"close_l": (np.array([1, 2]), np.array([0, 1]), np.zeros((2, 1)), np.ones((2, 4))),
                         "open_l": (np.array([0]), np.array([1]), np.zeros((1, 12)), None), "nop": (np.array([3]), np.array([0]), None, None)}}
    C2.save_sets(tmp_path / "s.npz", named)
    back = C2.load_sets(tmp_path / "s.npz")["INNER_A"]
    assert "G" in back["close_l"] and "G" not in back["open_l"] and "R" not in back["nop"]
    cx = types.SimpleNamespace(Xw=np.zeros((4, 3), np.float32))
    D = C2.sets_from_saved(cx, back, 9)
    assert D["close_l"][0].shape == (2, 3 + 9 + 4) and np.all(D["close_l"][0][:, -4:] == 1) and D["open_l"][0].shape == (1, 3 + 9)


# ----------------------------------------------------------------------------- E3-M networks
def test_c2g_closer_networks_are_wider_and_c2_networks_unchanged():
    pytest.importorskip("torch")
    from app.short_horizon.expert_guided import temporal as T
    n9 = T.TemporalPolicyNet(4, 3, 2, n_reg=1, n_state=9)
    n13 = T.TemporalPolicyNet(4, 3, 2, n_reg=1, n_state=9 + C2.N_GEO)
    x = np.zeros((2, 4 * 3 + 13), np.float32)
    lg, rg = T.TorchPolicy(n13)(x)
    assert lg.shape == (2, 2) and rg.shape == (2, 1) and n13.n_state == 13 and n9.n_state == 9


# ----------------------------------------------------------------------------- launcher
def wf_specs():
    pc2 = CONTRACTS / "ACTION_CONTRACT_C2.json"
    pg = CONTRACTS / "ACTION_CONTRACT_C2G.json"
    base = {"arms": [], "folds": ["wf2022", "wf2023"], "seeds": "1,2", "fitness_variants": list(ACT.FITNESS), "rule": "r", "early_close_rule": "e",
            "fitness_rule": "f", "outcomes": {"file_pattern": "OUTCOMES_ENTRY_<fold>.npz"}}
    c2 = dict(base, action_contract={"path": str(pc2), "sha256": ACT.sha256_file(pc2), "identity": json.loads(pc2.read_text())["identity"]},
              arms=[{"name": "zz_feat_c2", "base": "zz_hold", "strategy_fitness": "sharpe_lcb"}],
              baseline={"zz_feat_c2": {"wf": "WF_<fold>_zz_hold", "wf_v1": "WF_V1_<fold>_zz_hold_v1"}})
    g = dict(base, action_contract={"path": str(pg), "sha256": ACT.sha256_file(pg), "identity": json.loads(pg.read_text())["identity"]},
             arms=[{"name": "zz_feat_c2g", "base": "zz_hold", "strategy_fitness": "sharpe_lcb"}],
             baseline={"zz_feat_c2g": {"c2": "WF_C2_<fold>_zz_feat_c2", "wf": "WF_<fold>_zz_hold", "wf_v1": "WF_V1_<fold>_zz_hold_v1"}})
    return {"stages": {"A": [{"name": "zz_hold"}]}, "walk_forward": {"folds": {"wf2022": {}, "wf2023": {}}}, "walk_forward_v1": {"folds": ["wf2022", "wf2023"]},
            "walk_forward_c2": c2, "walk_forward_c2g": g}


def test_launcher_wf_c2g_block():
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = wf_specs()
    assert C.walk_forward_c2_errors(s) == [] and C.walk_forward_c2g_errors(s) == []
    pc2 = CONTRACTS / "ACTION_CONTRACT_C2.json"
    for mut, code in ((lambda s: s["walk_forward_c2g"]["baseline"]["zz_feat_c2g"].pop("c2"), "WF_C2G_NO_C2_BASELINE"),
                      (lambda s: s["walk_forward_c2g"]["baseline"]["zz_feat_c2g"].__setitem__("c2", "WF_C2_<fold>_other"), "WF_C2G_NO_C2_BASELINE"),
                      (lambda s: s["walk_forward_c2g"].__setitem__("action_contract", dict(s["walk_forward_c2"]["action_contract"])), "WF_C2G_CONTRACT_NOT_C2G"),
                      (lambda s: s["walk_forward_c2g"]["action_contract"].__setitem__("sha256", "0" * 64), "WF_C2G_CONTRACT_DIGEST_MISMATCH"),
                      (lambda s: s["walk_forward_c2g"]["arms"][0].__setitem__("labels", "multires"), "WF_C2G_ARM_HAS_TEACHER_TREATMENT"),
                      (lambda s: s["walk_forward_c2g"]["arms"][0].__setitem__("agent_state", "minimal"), "WF_C2G_AGENT_STATE_FORBIDDEN"),
                      (lambda s: s["walk_forward_c2g"].__setitem__("rule", ""), "WF_C2G_RULE_NOT_PREREGISTERED"),
                      (lambda s: s["walk_forward_c2"].__setitem__("folds", ["wf2022"]), "WF_C2G_FOLD_NOT_IN_WF_C2")):
        s2 = copy.deepcopy(s)
        mut(s2)
        assert any(e.startswith(code) for e in C.walk_forward_c2g_errors(s2)), (code, C.walk_forward_c2g_errors(s2))
    assert pc2.exists()


def emit_spec(tmp_path):
    s = wf_specs()
    s.update(identity="t", remote_root="/r", memory="4G", gates=[{"path": str(tmp_path / "LAUNCH_EXPERT_GUIDED.json")}], scaling="fixed_train",
             nonstationary="diff", split_contract="consolidated", exec_contract="sac", min_orders_per_week=2.0,
             recipe={"path": "/x/CONSUMER_RECIPE.json", "sha256": "0" * 64, "identity": "recipe_t"},
             store={"path": "/x/store.parquet", "sha256": "0" * 64}, model={"units": 2, "core": None, "depth": 2, "head": 16, "max_epochs": 5},
             regularization={"dropout": 0.3, "wd": 0.01})
    s["stages"]["A"] = [{"name": "zz_hold", "labels": "zigzag"}]
    s["walk_forward"]["folds"] = {f: {"recipe": {"path": f"/x/CONSUMER_RECIPE_{f}.json", "sha256": "1" * 64}} for f in ("wf2022", "wf2023")}
    return s


def test_emit_wf_c2g_units_and_wf_c2_text_unchanged(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = emit_spec(tmp_path)
    out_g, out_c = tmp_path / "g.sh", tmp_path / "c.sh"
    C.emit_walk_forward_c2(s, types.SimpleNamespace(out=str(out_g)), "walk_forward_c2g", C.WF_C2G_STAGE)
    C.emit_walk_forward_c2(s, types.SimpleNamespace(out=str(out_c)))
    g, c = out_g.read_text(), out_c.read_text()
    assert g.count("STATE/WF_C2G_wf2022_zz_feat_c2g ]") == 1 and g.count("STATE/WF_C2G_wf2023_zz_feat_c2g ]") == 1
    assert "--action-contract" in g and "ACTION_CONTRACT_C2G.json" in g and "[WF_C2G]" in g and 'log "STAGE_WF_C2G_DONE"' in g
    assert g.count("--seeds 1,2") == 2 and g.count("--outcomes $Q/outcomes/OUTCOMES_ENTRY_wf20") == 2
    assert "stage WF_C2 (action contract" in c and "[WF_C2]" in c and 'log "STAGE_WF_C2_DONE"' in c and "WF_C2_wf2022_zz_feat_c2" in c
    assert "C2G" not in c
    # WF_C2 emission is the text of the C2 stage with the stage name swapped (the template is shared, nothing else differs)
    assert c.replace("WF_C2", "STG").replace("zz_feat_c2", "ARM").replace("ACTION_CONTRACT_C2.json", "CT") .replace(
        s["walk_forward_c2"]["action_contract"]["identity"], "ID").replace(s["walk_forward_c2"]["action_contract"]["sha256"], "SHA") == \
        g.replace("WF_C2G", "STG").replace("zz_feat_c2g", "ARM").replace("ACTION_CONTRACT_C2G.json", "CT").replace(
        s["walk_forward_c2g"]["action_contract"]["identity"], "ID").replace(s["walk_forward_c2g"]["action_contract"]["sha256"], "SHA")


def test_localize_moves_the_c2g_contract_into_the_package(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = emit_spec(tmp_path)
    s.pop("walk_forward_v1")
    s["demonstrators"] = {"file": "/x/DEMONSTRATORS.json"}
    loc = C.localize(s, "/root")
    assert loc["walk_forward_c2g"]["action_contract"]["path"] == "/root/contracts/ACTION_CONTRACT_C2G.json"
    assert loc["walk_forward_c2"]["action_contract"]["path"] == "/root/contracts/ACTION_CONTRACT_C2.json"


def test_committed_spec_v9_preregisters_wf_c2g():
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = json.loads((ROOT / "docs/short_horizon/clean_round/CLEAN_ROUND_SPEC.json").read_text())
    w = s["walk_forward_c2g"]
    assert s["identity"].endswith("-v9") and w["stage"] == "WF_C2G" and w["owner_words"] == "Sí, como brazo extra (2026-10-25)"
    assert w["arms"][0]["name"] == "zz_feat_c2g" and w["baseline"]["zz_feat_c2g"]["c2"] == "WF_C2_<fold>_zz_feat_c2"
    assert w["action_contract"]["identity"] == json.loads((CONTRACTS / "ACTION_CONTRACT_C2G.json").read_text())["identity"]
    assert w["action_contract"]["sha256"] == ACT.sha256_file(CONTRACTS / "ACTION_CONTRACT_C2G.json")
    assert C.walk_forward_c2g_errors(s) == []


# ----------------------------------------------------------------------------- audit buckets and report rules
def test_closer_by_bucket_known_answer():
    from tools import expert_guided_early_close_audit as A
    lab = [{"dist_sl_sigma": d} for d in (0.2, 0.3, 0.7, 1.2, 2.5, 4.0)]
    y1 = np.array([1, 0, 1, 0, 0, 0])
    fired = np.array([True, False, False, True, False, False])
    pe = np.array([0.9, 0.2, 0.6, 0.5, 0.1, 0.05])
    b = A.closer_by_bucket(lab, pe, y1, fired)
    assert [x["rows"] for x in b] == [2, 1, 1, 0, 1, 1] and b[0]["dist_sl_sigma"] == [None, 0.5] and b[-1]["dist_sl_sigma"] == [3.0, None]
    assert b[0]["recall_sl_first"] == 1.0 and b[0]["precision_sl_first"] == 1.0 and b[0]["auc_sl_first"] == 1.0
    assert b[1]["recall_sl_first"] == 0.0 and np.isnan(b[2]["recall_sl_first"]) and b[2]["precision_sl_first"] == 0.0


def test_c2g_contrast_and_report_rules(tmp_path):
    from tools import expert_guided_c2 as R
    g = {"wf2022": {"value_usd": 50.0, "recall": 0.30, "auc": 0.7, "cut_winner_share": 0.3},
         "wf2023": {"value_usd": -5.0, "recall": 0.10, "auc": 0.6, "cut_winner_share": 0.4}}
    c = {"wf2022": {"value_usd": 10.0, "recall": 0.20, "auc": 0.6, "cut_winner_share": 0.4},
         "wf2023": {"value_usd": 5.0, "recall": 0.12, "auc": 0.6, "cut_winner_share": 0.4}}
    x = R.contrast_early_close(g, c)
    assert x["wf2022"]["verdict"] == "C2G_BETTER" and x["wf2023"]["verdict"] == "C2G_WORSE" and x["wf2022"]["delta_value_usd"] == 40.0
    assert R.contrast_early_close({"wf2022": g["wf2022"]}, c)["wf2023"]["verdict"] == "NOT_MEASURED"
    spec = {"identity": "t", "walk_forward_c2g": {"action_contract": {}, "rule": "r", "early_close_rule": "e", "fitness_rule": "f", "folds": ["wf2022", "wf2023"],
                                                  "arms": [{"name": "zz_feat_c2g"}],
                                                  "baseline": {"zz_feat_c2g": {"c2": "WF_C2_<fold>_zz_feat_c2", "wf": "WF_<fold>_zz_hold"}}}}
    doc = R.build_report_c2g(spec, tmp_path, tmp_path)
    rec = doc["arms"]["zz_feat_c2g"]
    assert rec["holds"]["verdict"] == "NOT_MEASURED" and rec["early_close"]["verdict"] == "NOT_MEASURED"
    assert rec["folds"]["wf2022"]["status"] == "MISSING" and "C2G" in R.report_markdown_c2g(doc)
