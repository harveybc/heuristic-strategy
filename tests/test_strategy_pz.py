"""PZ, the pragmatic strategy (docs/short_horizon/clean_round/ESTRATEGIA_PRAGMATICA_PZ_2026_10_30.md): labels, heads,
orders, the opening and close rules, and the additivity of the hooks (C3Z full_f3 / full_f4 / LEGACY untouched)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import horizon_levels as HL
from app.short_horizon.expert_guided import strategy_pz as PZ
from app.short_horizon.expert_guided import strategy_zz as ZZ


def _bars(o, h, l, c):
    idx = pd.date_range("2022-01-03", periods=len(o), freq="h", tz="UTC")
    return pd.DataFrame({"OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c}, index=idx)


def _mt(pivots_by_h: dict, conf_by_h: dict | None = None):
    out = {}
    for name, (d, b, h) in ZZ.MT_HORIZONS.items():
        piv = pivots_by_h.get(h, [])
        conf = (conf_by_h or {}).get(h, [p[0] + 1 for p in piv])
        out[name] = {"pivots": piv, "conf": conf, "h": h}
    return out


def test_contract_constants():
    assert PZ.N_MIN_OUTPUTS == 24
    assert PZ.HORIZONS == (3, 6, 12, 24)
    assert len(PZ.GENES) == 10
    g = PZ.validate_genes({})
    assert g == {**PZ.CANONICAL, "k_persist": 1}
    with pytest.raises(PZ.PZError):
        PZ.validate_genes({"alpha": 0.3})
    with pytest.raises(PZ.PZError):
        PZ.validate_genes({"a_sl": 2.5})
    with pytest.raises(PZ.PZError):
        PZ.validate_genes({"nope": 1})
    assert abs(PZ.rho_of(0.70) - math.sin(math.pi * 0.2)) < 1e-12 and PZ.rho_of(1.0) == 1.0


def _toy():
    # bars 0..9: valley at 2 (low 0.9), peak at 6 (high 1.3), valley at 8 (low 1.0)
    o = np.array([1.10, 1.05, 1.00, 1.05, 1.10, 1.20, 1.25, 1.15, 1.05, 1.10])
    c = np.r_[o[1:], 1.12]
    h = np.maximum(o, c) + 0.01
    l = np.minimum(o, c) - 0.01
    l[2], h[6], l[8] = 0.90, 1.30, 1.00
    piv = [(2, Z.VALLEY, 0.90), (6, Z.PEAK, 1.30), (8, Z.VALLEY, 1.00)]
    return _bars(o, h, l, c), piv


def test_labels_exact_on_a_toy_path():
    bars, piv = _toy()
    lab = PZ.expert_labels(bars, _mt({h: piv for h in PZ.HORIZONS}))
    o, hi, lo = bars["OPEN"].to_numpy(), bars["HIGH"].to_numpy(), bars["LOW"].to_numpy()
    j = 1
    # decision after bar 0: next pivot is the valley at 2 -> p_up 0, tau 2
    assert lab["p_up"][0, j] == 0.0 and lab["tau"][0, j] == 2.0
    # buy: next PEAK at 6: fav = 1.30 - open[1]; adv = open[1] - min(low[1..6]) = open[1] - 0.90
    assert lab["fav"][0, j, 0] == pytest.approx(1.30 - o[1])
    assert lab["adv"][0, j, 0] == pytest.approx(o[1] - 0.90)
    # sell: next VALLEY at 2: fav = open[1] - 0.90; adv = max(high[1..2]) - open[1]
    assert lab["fav"][0, j, 1] == pytest.approx(o[1] - 0.90)
    assert lab["adv"][0, j, 1] == pytest.approx(max(hi[1], hi[2]) - o[1])
    # decision after bar 2: the next pivot is the peak at 6 (the valley at 2 is not after the decision)
    assert lab["p_up"][2, j] == 1.0 and lab["tau"][2, j] == 4.0
    assert lab["adv"][2, j, 0] == pytest.approx(o[3] - min(lo[3:7]))
    # after bar 8 nothing follows: censored, never fabricated
    assert np.isnan(lab["p_up"][8, j]) and np.isnan(lab["fav"][8, j, 0]) and np.isnan(lab["tau"][9, j])
    assert np.all(lab["adv"][np.isfinite(lab["adv"])] >= 0)


def test_unconfirmed_pivot_is_censored():
    bars, piv = _toy()
    lab = PZ.expert_labels(bars, _mt({h: piv for h in PZ.HORIZONS}, {h: [3, 7, -1] for h in PZ.HORIZONS}))
    assert np.isnan(lab["p_up"][7, 0])                                    # the valley at 8 never confirms
    assert np.isnan(lab["fav"][7, 0, 1])


def test_labels_mirror_symmetric():
    bars, piv = _toy()
    k = 2.4
    mb = _bars(k - bars["OPEN"].to_numpy(), k - bars["LOW"].to_numpy(), k - bars["HIGH"].to_numpy(), k - bars["CLOSE"].to_numpy())
    mpiv = [(b, -kind, k - p) for b, kind, p in piv]
    a = PZ.expert_labels(bars, _mt({h: piv for h in PZ.HORIZONS}))
    m = PZ.expert_labels(mb, _mt({h: mpiv for h in PZ.HORIZONS}))
    fin = np.isfinite(a["p_up"])
    assert np.array_equal(fin, np.isfinite(m["p_up"])) and np.allclose(a["p_up"][fin], 1 - m["p_up"][fin])
    assert np.allclose(a["fav"][..., 0], m["fav"][..., 1], equal_nan=True)
    assert np.allclose(a["adv"][..., 0], m["adv"][..., 1], equal_nan=True)


def _rand_labels(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return {"p_up": (rng.random((n, 4)) < 0.5).astype(float), "tau": rng.integers(1, 40, (n, 4)).astype(float),
            "fav": rng.gamma(2.0, 0.002, (n, 4, 2)), "adv": rng.gamma(1.5, 0.001, (n, 4, 2))}


def test_heads_perfect_and_ladder():
    lab = _rand_labels()
    cal = np.zeros(3000, bool)
    cal[:1000] = True
    p = PZ.heads(lab, 1.0, 0, cal)
    assert p["mode"] == "perfect" and np.array_equal(p["p_up"], lab["p_up"]) and all(np.array_equal(p["fav"][q], lab["fav"]) for q in PZ.QUANTS)
    h7 = PZ.heads(lab, 0.70, 3, cal)
    h7b = PZ.heads(lab, 0.70, 3, cal)
    assert np.array_equal(h7["p_up"], h7b["p_up"], equal_nan=True)            # reproducible per seed
    assert np.all((h7["p_up"] > 0) & (h7["p_up"] < 1))
    assert np.all(h7["adv"][0.75] >= 0)
    rep = PZ.head_report(lab, h7, ~cal, cal)
    # residual quantiles from CAL: the 90 % interval covers about 0.90 out of sample on an exchangeable toy
    for k in ("fav", "adv", "tau"):
        assert 0.85 <= rep[k]["coverage_90"] <= 0.95
        assert 0.0 < rep[k]["mae_over_naive"] < 1.0
    assert rep["p_up"]["brier"] < rep["p_up"]["naive_brier"]
    r1 = PZ.head_report(lab, p, ~cal, cal)
    assert r1["fav"]["coverage_90"] == 1.0 and r1["fav"]["mae_q50"] == 0.0


def test_ar1_error_persistence():
    lab = _rand_labels(n=20000, seed=1)
    lab["fav"][:] = 0.002                                                   # a constant label: the head error is the noise itself
    lab["fav"][::2] = 0.003
    h = PZ.heads(lab, 0.70, 5, np.ones(20000, bool))
    e = h["pred"]["fav"][:, 0, 0] - lab["fav"][:, 0, 0]
    e = e - e.mean()
    # the noise part rho k sd u has lag-1 autocorrelation phi; the label part alternates (-1 lag-1): the residual's sign is not tested,
    # only that the AR(1) stream drives it: u reconstructed from the conditional-mean form
    rho = PZ.rho_of(0.70)
    y = lab["fav"][:, 0, 0]
    mu, sd = y.mean(), y.std()
    u = (h["pred"]["fav"][:, 0, 0] - mu - rho * rho * (y - mu)) / (rho * math.sqrt(1 - rho * rho) * sd)
    ac = np.corrcoef(u[1:], u[:-1])[0, 1]
    assert abs(ac - PZ.PHI) < 0.03


def test_orders_rounding_and_validity(monkeypatch):
    monkeypatch.setattr(PZ, "DESIGN_VERSION", "1.0")                       # the v1.0 history (offset-free TP), pinned
    bars, piv = _toy()
    n = len(bars)
    hd = {"fav": {0.5: np.full((n, 4, 2), 0.00123)}, "adv": {0.75: np.full((n, 4, 2), 0.00051)}}
    hd["fav"][0.5][3, :, 1] = -0.001                                        # a negative favourable excursion: no sell order
    o = PZ.orders(bars, np.full(n, 0.005), np.arange(n), hd, {"a_sl": 1.0, "a_tp": 1.0, "buffer_pips": 1.0})
    q = bars["CLOSE"].to_numpy()
    lv = HL.levels(1, q[0], 0.00123, 0.00051, 0.0, {"a_sl": 1, "b_sl": 0, "buffer_pips": 1, "a_tp": 1, "b_tp": 0}, ZZ.LEVEL_TICK)
    assert o["tp"][0, 0, 0] == lv["tp"] and o["sl"][0, 0, 0] == lv["sl"]
    assert o["tp"][0, 0, 0] <= q[0] + 0.00123 + 1e-12 and o["sl"][0, 0, 0] <= q[0] - 0.00061 + 1e-12   # TP toward, SL away (buy)
    assert np.isnan(o["tp"][3, 1, 2]) and not o["mask"][3, 1, 2] and o["mask"][3, 0, 2]
    o6 = PZ.orders(bars, np.full(n, 0.005), np.arange(n), hd, {}, horizons=(6,))
    assert np.isnan(o6["tp"][:, :, [0, 2, 3]]).all() and np.isfinite(o6["tp"][0, :, 1]).all()


class _Cl:
    def value(self, r, bar, pos):
        return 0.0, 0.0


def _table(n=4, genes=None, horizons=PZ.HORIZONS):
    t = {"pz_genes": PZ.validate_genes(genes or {}), "pz_horizons": horizons, "pz_p_up": np.full((n, 4), 0.5),
         "pz_fav25": np.zeros((n, 4, 2)), "pz_fav75": np.zeros((n, 4, 2)), "pz_tau50": np.full((n, 4), 5.0),
         "pz_tp": np.full((n, 2, 4), np.nan), "pz_sl": np.full((n, 2, 4), np.nan), "pz_quote": np.full(n, 1.1), "pz_cost": np.full(n, 0.0001), "closer": _Cl()}
    return t


def _pol(t):
    from app.short_horizon.expert_guided import menu as M
    return PZ.PolicyPZ(t, PZ.strategy_object(t["pz_genes"]), cells=M.MENU_CELLS)


def test_open_rule_scores_and_side_symmetry():
    t = _table()
    # row 0: buy h6 TP 30 pips SL 10 pips p 0.6; sell h12 TP 20 SL 10 p 0.7 (p_up 0.3)
    t["pz_p_up"][0] = [0.5, 0.6, 0.3, 0.5]
    t["pz_tp"][0, 0, 1], t["pz_sl"][0, 0, 1] = 1.1030, 1.0990
    t["pz_tp"][0, 1, 2], t["pz_sl"][0, 1, 2] = 1.0980, 1.1010
    pol = _pol(t)
    g_b = (0.6 * 0.0030 - 0.4 * 0.0010 - 0.0001) / 0.0010
    g_s = (0.7 * 0.0020 - 0.3 * 0.0010 - 0.0001) / 0.0010
    out = pol.open_decision(0, with_score=True)
    assert out[0] == E.LONG and out[1] == 1.1030 and out[3] == pytest.approx(g_b) and g_b > g_s
    assert pol.attr[-1]["h"] == 6 and pol.log[-1][3] and pol.log[-1][4]
    # theta above both: NOP
    t2 = dict(t, pz_genes=PZ.validate_genes({"theta": 1.0}), pz_p_up=np.full((4, 4), 0.5))
    t2["pz_p_up"][0] = [0.5, 0.4, 0.5, 0.5]                               # scores 0.5 (buy) and 0.4 (sell) < theta 1
    assert _pol(t2).open_decision(0) is None
    # exact mirror (sell side = the buy's mirror): an exact side tie is a NOP
    t3 = _table()
    t3["pz_p_up"][0] = [0.5, 0.5, 0.5, 0.5]
    t3["pz_tp"][0, 0, 1], t3["pz_sl"][0, 0, 1] = 1.1030, 1.0990
    t3["pz_tp"][0, 1, 1], t3["pz_sl"][0, 1, 1] = 1.0970, 1.1010
    pz3 = _pol(t3)
    assert pz3.open_decision(0) is None and pz3.attr[-1]["blocked"] == "side_tie"


def test_horizon_restriction_and_alpha():
    t = _table(horizons=(6,))
    t["pz_p_up"][0] = [0.9, 0.6, 0.5, 0.5]
    t["pz_tp"][0, 0, 0], t["pz_sl"][0, 0, 0] = 1.1050, 1.0990            # h3: better, but outside the PZ-6h menu
    t["pz_tp"][0, 0, 1], t["pz_sl"][0, 0, 1] = 1.1030, 1.0990
    assert _pol(t).open_decision(0)[1] == 1.1030
    ta = _table(genes={"alpha": 1.0, "tau0": 1.0})
    ta["pz_p_up"][0] = [0.6, 0.6, 0.5, 0.5]
    ta["pz_tp"][0, 0, 0], ta["pz_sl"][0, 0, 0] = 1.1030, 1.0990
    ta["pz_tp"][0, 0, 1], ta["pz_sl"][0, 0, 1] = 1.1030, 1.0990
    ta["pz_tau50"][0] = [20.0, 2.0, 5.0, 5.0]                              # same order, the 6 h level reaches its pivot faster
    p = _pol(ta)
    p.open_decision(0)
    assert p.attr[-1]["h"] == 6


def _pos(d=1, tp=1.1030, sl=1.0990, sigma=0.005):
    return E.Position(d, 1000, 0, 1.1, sl, tp, sigma, 0, {})


def test_close_rule_hold_urgent_and_persistence(monkeypatch):
    monkeypatch.setattr(PZ, "DESIGN_VERSION", "1.0")                       # the v1.0 urgent rule, pinned
    t = _table()
    t["pz_tp"][0, 0, 1], t["pz_sl"][0, 0, 1] = 1.1030, 1.0990
    t["pz_p_up"][0] = [0.5, 0.9, 0.5, 0.5]
    pol = _pol(t)
    pol.open_decision(0)                                                   # remembers h_p = 6 for this order
    pos = _pos()
    t["pz_p_up"][1] = [0.9, 0.9, 0.5, 0.5]
    assert pol.close_now(1, 0, 1, pos) is False                            # HOLD = 0.9 x 30 - 0.1 x 10 pips > 0
    t["pz_p_up"][2] = [0.9, 0.1, 0.5, 0.5]
    assert pol.close_now(2, 0, 2, pos) is True                             # HOLD < 0
    t["pz_p_up"][3] = [0.1, 0.9, 0.5, 0.5]
    assert pol.close_now(3, 0, 3, pos) is True                             # the finer level (3 h) turns against: urgent (p_urg 0.25)
    assert pol.closes[-1][7] and not pol.closes[-1][6]
    # O*'s tie band: HOLD at band - cost_close closes (a tie closes), just above holds
    t5 = _table(n=2)
    pol5 = _pol(t5)
    band, cc = PZ.EPS_CLOSE * 0.005, 0.5 * 0.0001
    # buy p'=0.5 with up = down + 2 (band - cc): HOLD = band - cc
    down = 0.0010
    up = down + 2 * (band - cc) - 1e-9                                     # HOLD = band - cc (minus float slack)
    t5["pz_p_up"][1] = [0.5, 0.5, 0.5, 0.5]
    assert pol5.close_now(1, 0, 1, _pos(tp=1.1 + up, sl=1.1 - down)) is True
    assert pol5.close_now(1, 0, 1, _pos(tp=1.1 + up + 1e-6, sl=1.1 - down)) is False
    # persistence k = 2: two consecutive ticks
    tk = _table(genes={"k_persist": 2})
    pk = _pol(tk)
    tk["pz_p_up"][1] = [0.5, 0.1, 0.5, 0.5]
    tk["pz_p_up"][2] = [0.5, 0.1, 0.5, 0.5]
    ps = _pos()
    assert pk.close_now(1, 0, 1, ps) is False and pk.close_now(2, 0, 2, ps) is True


def test_close_rule_mirror():
    t = _table()
    t["pz_p_up"][1] = [0.5, 0.2, 0.5, 0.5]
    tm = _table()
    tm["pz_p_up"][1] = [0.5, 0.8, 0.5, 0.5]
    tm["pz_quote"][:] = 1.1
    a = _pol(t).close_now(1, 0, 1, _pos(1, 1.1030, 1.0990))
    b = _pol(tm).close_now(1, 1, 1, _pos(-1, 1.0970, 1.1010))
    assert a == b is True


def test_trivial_never_closes_early():
    t = _table()
    t["triv_tp"], t["triv_sl"] = np.full((4, 2), np.nan), np.full((4, 2), np.nan)
    t["triv_tp"][0], t["triv_sl"][0] = [1.1020, 1.0980], [1.0980, 1.1020]
    t["pz_p_up"][0, 1] = 0.7
    from app.short_horizon.expert_guided import menu as M
    p = PZ.PolicyTrivial(t, PZ.strategy_object({}), cells=M.MENU_CELLS)
    assert p.open_decision(0) == (E.LONG, 1.1020, 1.0980)
    t["pz_p_up"][1, 1] = 0.5
    assert p.open_decision(1) is None
    assert p.close_now(1, 0, 1, _pos()) is False


def test_hooks_default_to_c3z():
    from tools import strategy_behaviour_matrix as BM
    from tools import strategy_hierarchy as H
    from tools import strategy_output_simulation as S
    import inspect
    assert BM.STRATEGY == "c3z" and BM.policy_class() is ZZ.PolicyC3Z and H.MENU_ORDERS_FN is None
    assert inspect.signature(S.build_truth_menu).parameters["orders_fn"].default is None
    assert inspect.signature(BM.build_episodes).parameters["Pol"].default is None


def test_truth_orders_fn_matches_view_orders(monkeypatch):
    monkeypatch.setattr(PZ, "DESIGN_VERSION", "1.0")
    bars, piv = _toy()
    lab = PZ.expert_labels(bars)
    n = len(bars)
    sig = np.full(n, 0.005)
    f = PZ.truth_orders_fn(PZ.CANONICAL)
    a = f(bars, sig, np.arange(n), {})
    b = PZ.orders(bars, sig, np.arange(n), PZ.heads(lab, 1.0, 0, np.zeros(n, bool)), PZ.CANONICAL)
    assert np.array_equal(a["tp"], b["tp"], equal_nan=True) and np.array_equal(a["mask"], b["mask"])


def test_pz_deap_genome_and_staging():
    from tools import strategy_pz_deap as PD
    g = PZ.validate_genes({})
    x = PD.encode(g)
    assert len(x) == len(PD.GENOME2) == 7 and all(0.0 <= u <= 1.0 for u in x)
    d = PD.decode(x, g)
    assert all(abs(d[k] - g[k]) < 1e-9 for k in PZ.GENES)                  # the canonical point round-trips (it is in generation 0)
    lo = PD.decode([0.0] * 7, g)
    hi = PD.decode([1.0] * 7, g)
    assert lo["alpha"] == 0.0 and hi["alpha"] == 1.0 and lo["k_persist"] == 1 and hi["k_persist"] == 3
    assert hi["theta"] == PZ.BOUNDS["theta"][1] and lo["rho"] == PZ.BOUNDS["rho"][0]
    assert all(d_[k] == g[k] for d_ in (lo, hi) for k in PD.LEVEL)           # stage 2 never moves the level genes
    pts = PD.level_points()
    assert len(pts) == 18 and {"a_sl": 1.0, "a_tp": 1.0, "buffer_pips": 1.0} in pts
    ext = PD.edge_extensions({"a_sl": 0.75, "a_tp": 1.0, "buffer_pips": 3.0})
    assert {"a_sl": 0.5, "a_tp": 1.0, "buffer_pips": 3.0} in ext and {"a_sl": 0.75, "a_tp": 1.0, "buffer_pips": 5.0} in ext and len(ext) == 2
    assert PD.edge_extensions({"a_sl": 1.0, "a_tp": 1.0, "buffer_pips": 1.0}) == [{"a_sl": 1.0, "a_tp": 1.0, "buffer_pips": 0.0}]
    sub = [{"subset_key": "A"}, {"subset_key": "S_X@q1", "pair": "X"}, {"subset_key": "M_X@q1", "pair": "X"}]
    gs = PD.groups_of(sub)
    assert len(gs) == 2 and any(len(g_) == 2 for g_ in gs)                   # a pair is evaluated by one worker


class _Prof:
    limit_through = 1e-4

    def spread_at(self, t, o):
        return 2e-5


class _Var:
    short_trigger_on_ask = True


class _Cfg:
    profile = _Prof()
    variant = _Var()


class _Env:
    def __init__(self, n):
        self.cfg = _Cfg()
        self.v = np.arange(n, dtype=np.int64) * 3600 * 10 ** 9
        self.mo = np.zeros(1, np.int64)
        self.seg_of = np.zeros(n, np.int64)


def test_v11_offsets_and_tp_distance():
    assert PZ.DESIGN_VERSION == "1.1"
    env = _Env(10)
    xo = PZ.exit_offsets(env, np.arange(10))
    assert np.allclose(xo[:, 0], 1e-4) and np.allclose(xo[:, 1], 1.2e-4)    # long exits at the bid; short at the ask (spread)
    bars, piv = _toy()
    n = len(bars)
    hd = {"fav": {0.5: np.full((n, 4, 2), 0.00123)}, "adv": {0.75: np.full((n, 4, 2), 0.00051)}}
    with pytest.raises(PZ.PZError):
        PZ.orders(bars, np.full(n, 0.005), np.arange(n), hd, PZ.CANONICAL)  # v1.1 refuses an order without its exit offset
    o = PZ.orders(bars, np.full(n, 0.005), np.arange(n), hd, PZ.CANONICAL, exit_offset=xo)
    q = bars["CLOSE"].to_numpy()
    assert o["tp"][0, 0, 1] <= q[0] + 0.00123 - 1e-4 + 1e-12 and o["tp"][0, 0, 1] >= q[0] + 0.00123 - 1e-4 - ZZ.LEVEL_TICK - 1e-12
    assert o["tp"][0, 1, 1] >= q[0] - 0.00123 + 1.2e-4 - 1e-12                    # a sell's TP sits above the predicted valley by through + spread
    hd["fav"][0.5][2] = 0.00005                                                   # a predicted excursion smaller than the offset: no order
    o2 = PZ.orders(bars, np.full(n, 0.005), np.arange(n), hd, PZ.CANONICAL, exit_offset=xo)
    assert not o2["mask"][2].any()


def test_v11_perfect_order_at_a_pivot_is_fillable():
    """The ruling's test: with perfect outputs the TP of the order whose leg ends at a pivot fills when the price goes
    through it (v1.0's TP sat at the extreme and never filled: 0 TP exits in 32 231)."""
    from app.short_horizon import execution as X
    bars, piv = _toy()
    n = len(bars)
    o_, h_, l_ = (bars[k].to_numpy() for k in ("OPEN", "HIGH", "LOW"))
    c_ = bars["CLOSE"].to_numpy()
    assert np.allclose(o_[1:], c_[:-1])                                             # the toy's next open = the close (fav anchor = quote)
    lab = PZ.expert_labels(bars, _mt({h: piv for h in PZ.HORIZONS}))
    hd = PZ.heads(lab, 1.0, 0, np.zeros(n, bool))
    env = _Env(n)
    xo = PZ.exit_offsets(env, np.arange(n))
    g = dict(PZ.CANONICAL, buffer_pips=0.0, a_sl=2.0)
    o = PZ.orders(bars, np.full(n, 0.005), np.arange(n), hd, g, exit_offset=xo)
    b, j = 2, 1                                                                     # after the valley at 2: buy, next peak at 6 (high 1.30)
    tp, sl = o["tp"][b, 0, j], o["sl"][b, 0, j]
    assert np.isfinite(tp)
    hits = [X._protect(1, tp, sl, o_[k], h_[k], l_[k], env.cfg.profile.limit_through) for k in range(b + 1, 7)]
    assert any(x is not None and x[0] == "TP" for x in hits)
    o10 = dict(o)                                                                   # the v1.0 TP (at the extreme) on the same bar never fills
    tp10 = c_[b] + lab["fav"][b, j, 0]
    assert all(X._protect(1, tp10, sl, o_[k], h_[k], l_[k], env.cfg.profile.limit_through) is None or X._protect(1, tp10, sl, o_[k], h_[k], l_[k], env.cfg.profile.limit_through)[0] != "TP"
               for k in range(b + 1, 7))


def test_v11_urgent_close_needs_the_reach_to_sl():
    t = _table()
    t["pz_fav50"] = np.zeros((4, 4, 2))
    pol = _pol(t)
    pos = _pos()                                                                    # long, P = 1.1, SL 1.0990: 10 pips to the SL
    t["pz_p_up"][1] = [0.1, 0.9, 0.5, 0.5]                                         # the 3 h level against, the 6 h level in favour (HOLD > 0)
    t["pz_fav50"][1, 0, 1] = 0.0005                                                 # the fine reversal reaches 5 pips: not the SL
    assert pol.close_now(1, 0, 1, pos) is False
    t["pz_fav50"][2, 0, 1] = 0.0011                                                 # 11 pips >= the 10 pips left to the SL
    t["pz_p_up"][2] = [0.1, 0.9, 0.5, 0.5]
    assert pol.close_now(2, 0, 2, pos) is True and pol.closes[-1][7]               # reaches the SL: urgent
    t0 = dict(t, pz_genes=PZ.validate_genes({"p_urg": 0.0}))
    assert _pol(t0).close_now(2, 0, 2, pos) is False                              # p_urg = 0: the urgent close is off
