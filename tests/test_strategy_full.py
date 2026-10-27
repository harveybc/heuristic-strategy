"""The FULL parametric upper layer (owner 2026-10-26): gene bounds per family pair, the explicit TP/SL and volume
formulas with their clipping, v_min refusing the order, the gene map copying the coefficient genes, strategy round
trips, the volume-forwarding best-of-day gate and the volume-aware rollout. No data, no fit."""
import dataclasses

import numpy as np
import pytest

from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import strategy_full as SF


def test_gene_bounds_per_family():
    g = SF.gene_bounds("cell", "risk")
    assert set(g) == set(C3.GENES_Q) | {"observe_bars"}
    g = SF.gene_bounds("param", "param")
    assert {"tp_a_l", "tp_b_s", "sl_c_l", "sl_e_s", "v_alpha_l", "v_delta_s", "v_min"} <= set(g)
    assert g["tp_a_l"] == [0.25, 2.0] and g["v_min"] == [0.0, 0.5]


def _strategy(tp="param", vol="param", **over):
    g = {"open_margin": -9, "close_margin_l": 9, "close_margin_s": 9, "close_p_l": 2, "close_p_s": 2, "allow_reverse": 0.0, "nop_veto": 9,
         "observe_bars": 0.0, "tp_a_l": 1.0, "tp_b_l": 0.5, "sl_c_l": 1.0, "sl_e_l": 1.0, "tp_a_s": 0.75, "tp_b_s": 0.0, "sl_c_s": 0.75, "sl_e_s": 0.0,
         "v_alpha_l": 0.5, "v_beta_l": 0.5, "v_gamma_l": 0.5, "v_delta_l": 0.5, "v_alpha_s": 0.2, "v_beta_s": 0.0, "v_gamma_s": 0.0, "v_delta_s": 0.0, "v_min": 0.1}
    g.update(over)
    return SF.StrategyFull.from_genes(g, False, tp, vol, 0.01)


def _table(n=4):
    t = {"net": np.full((n, 2, len(C3.CELLS)), 0.3), "p3": np.zeros((n, 2, 3)), "pn": np.full(n, 0.2), "cap": np.full(n, 2.4), "lo": np.full(n, 0.3)}
    t["p3"][:, 0, :] = [0.7, 0.2, 0.1]          # long: P(TP) 0.7, P(SL) 0.2
    t["p3"][:, 1, :] = [0.3, 0.5, 0.2]          # short
    return t


def test_param_tp_sl_formulas_and_clipping():
    s = _strategy()
    pol = SF.PolicyFull(_table(), s, C3.CELLS, (0.0, 0.0), 0.1)
    tp, sl, ev = pol.levels(0, 0, None, None)
    assert tp == pytest.approx(1.0 + 0.5 * 0.7) and sl == pytest.approx(1.0 + 1.0 * 0.2)
    assert ev == pytest.approx(0.7 * tp - 0.2 * sl + 0.1 * 0.0 - 0.1)
    # clipping to the band and the business floors
    s2 = _strategy(tp_a_l=3.5, sl_c_l=0.1, sl_e_l=0.0)
    tp, sl, _ = SF.PolicyFull(_table(), s2, C3.CELLS, (0.0, 0.0), 0.1).levels(0, 0, None, None)
    assert tp == pytest.approx(2.4) and sl == pytest.approx(0.5)          # cap 2.4 (band), SL floor 0.5 (business)
    # the cell family reproduces C3's choice
    sc = _strategy(tp="cell")
    tpc, slc, evc = SF.PolicyFull(_table(), sc, C3.CELLS, (0.0, 0.0), 0.1).levels(0, 0, None, None)
    ci, ev = C3.best_cell_ev(_table()["net"][0, 0], _table()["p3"][0, 0], C3.CELLS, 0.0, 0.1, cap=2.4, lo=0.3)
    assert (tpc, slc) == tuple(C3.CELLS[ci]) and evc == pytest.approx(ev)


def test_param_volume_formula_and_v_min():
    s = _strategy()
    pol = SF.PolicyFull(_table(), s, C3.CELLS, (0.0, 0.0), 0.1)
    v = pol.volume(0, 0, 1.2, None, None)
    assert v == pytest.approx(min(1.0, 0.5 + 0.5 * 0.7 - 0.5 * 0.2 - 0.5 * 0.2))
    # short side: alpha 0.2 only -> v 0.2 >= v_min 0.1 -> order; with v_min 0.3 the short order is refused
    o = pol.open_decision_full(0, None, None)
    assert o is not None and o[1] == E.LONG and 0 < o[4] <= 1.0
    s3 = _strategy(v_min=0.3, open_margin=9.0)            # both sides blocked by the pooled margin -> no order
    assert SF.PolicyFull(_table(), s3, C3.CELLS, (0.0, 0.0), 0.1).open_decision_full(0, None, None) is None


def test_strategy_round_trip_and_gene_map_copies_coefficients():
    s = _strategy()
    d = s.to_dict()
    s2 = SF.strategy_from_dict(d)
    assert s2 == s and s2.no_early_close().base.early_close is False
    bounds = SF.gene_bounds("param", "param")
    rules = {k: (k, None, "copy") for k in bounds if k not in SF.THRESHOLD_GENES}
    from app.short_horizon.expert_guided import action_contract as ACT
    gm = ACT.GeneMap({"x": np.arange(10.0)}, {"open_margin": ("open_rate", "x", "upper_rate"), **rules})
    out = gm.to_absolute({"open_rate": 0.1, "tp_a_l": 1.3, "v_min": 0.2, "observe_bars": 2.0})
    assert out["tp_a_l"] == 1.3 and out["v_min"] == 0.2 and out["observe_bars"] == 2.0 and out["open_margin"] == pytest.approx(np.quantile(np.arange(10.0), 0.9))


class _Env:
    def __init__(self):
        self.v = np.arange(100) * 3_600_000_000_000
        self.calls = []

    def business_day(self, k):
        return k // 24

    def entry_gate(self, b):
        return True, ""


def test_gate_forwards_volume_and_observes():
    act = lambda bar, r, st, env: (E.LONG, 1.0, 1.0, 0.6, 1.0 + bar * 0.0)  # noqa: E731
    g = SF.BestOfDayGateV(act, observe_bars=1)
    env = _Env()
    flat = np.zeros(9)
    a1 = g(0, 0, flat, env)
    a2 = g(1, 1, flat, env)
    assert a1[0] == 0 and a2[0] == E.LONG and a2[3] == 0.6
    pos = np.zeros(9)
    pos[0] = 1
    assert g(2, 2, pos, env) == (E.LONG, 1.0, 1.0, 0.6)      # positioned bars pass through untouched
