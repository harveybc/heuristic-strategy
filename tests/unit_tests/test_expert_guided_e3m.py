"""E3-M modular specialists: partition labels, composition and arbiter arithmetic (no data, no training)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools import expert_guided_e3m as M  # noqa: E402

FLAT = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], np.float32)
LONG = np.array([1, 0.5, 0.2, 1.5, 2.0, 1, 1, 0.25, 1], np.float32)
SHORT = np.array([-1, -0.3, 0.1, 1.0, 2.5, 1, 1, 0.25, 1], np.float32)


def test_partition_labels_cover_every_module_exactly_once():
    assert M.labels_for("FULL", FLAT, E.LONG) == {"open_l": (1, True), "open_s": (0, False)}
    assert M.labels_for("FULL", LONG, E.SHORT) == {"close_l": (M.EXIT, False)}        # reverse = exit for the closer
    assert M.labels_for("FULL", SHORT, E.HOLD) == {"close_s": (0, False)}
    assert M.labels_for("TASK", FLAT, E.SHORT) == {"opener": (E.SHORT, True)}
    assert M.labels_for("TASK", LONG, E.CLOSE) == {"closer": (M.EXIT, False)}
    assert M.labels_for("DIR", FLAT, E.LONG) == {"long": (1, True), "short": (0, False)}
    assert M.labels_for("DIR", SHORT, E.LONG) == {"short": (2, False)}
    assert M.labels_for("JOINT", LONG, E.CLOSE) == {"joint": (E.CLOSE, False)}
    for v, mods in M.MODULES.items():
        for st, a in ((FLAT, E.LONG), (FLAT, E.HOLD), (LONG, E.HOLD), (SHORT, E.CLOSE)):
            assert set(M.labels_for(v, st, a)) <= set(mods)


class Const:
    def __init__(self, logits, reg=None):
        self.lg, self.rg = np.asarray(logits, np.float32), reg

    def __call__(self, x):
        n = len(x)
        return np.tile(self.lg, (n, 1)), (None if self.rg is None else np.tile(np.asarray(self.rg, np.float32), (n, 1)))


def test_arbiters_and_composition_on_full_variant():
    pols = {"open_l": Const([0.0, 2.0], [1.0, 2.0]),      # p(open long) = sigmoid(2) ~ 0.88, tp 1, sl 2
            "open_s": Const([2.0, 0.0], [1.0, 2.0]),      # p(open short) ~ 0.12
            "close_l": Const([0.0, 3.0]),                 # exit
            "close_s": Const([3.0, 0.0])}                 # hold
    c = M.Composed("FULL", pols, "prob", tau=0.5)
    x = np.zeros(13 * 15 + 9, np.float32)
    pL, tpL, slL, pS, tpS, slS = c.openers(x)
    assert pL > 0.85 and pS < 0.15 and tpL == 1.0 and slL == 2.0
    assert c.arbitrate(pL, tpL, slL, pS, tpS, slS) == E.LONG
    assert M.Composed("FULL", pols, "prob", tau=0.95).arbitrate(pL, tpL, slL, pS, tpS, slS) == E.HOLD
    g = M.Composed("FULL", pols, "expected_gain")
    assert g.arbitrate(0.9, 1.0, 2.0, 0.1, 1.0, 2.0) == E.LONG            # 0.9 - 0.2 - cost > 0
    assert g.arbitrate(0.5, 1.0, 2.0, 0.5, 1.0, 2.0) == E.HOLD            # 0.5 - 1.0 - cost < 0
    r = M.Composed("FULL", pols, "gain_over_risk")
    assert r.arbitrate(0.9, 1.0, 2.0, 0.95, 1.0, 10.0) == E.LONG          # short has higher gain but 5x the risk

    class CX:
        Xw = np.zeros((1, 13 * 15), np.float32)
    f = c(CX())
    assert f(0, 0, SHORT, None)[0] == E.HOLD                              # closer_s says hold
    a, tp, sl = f(0, 0, LONG, None)                                       # closer_l exits, arbiter wants LONG again
    assert a == E.CLOSE
    assert f(0, 0, FLAT, None) == (E.LONG, 1.0, 2.0)
    assert M.flat_state(LONG)[:5].sum() == 0 and M.flat_state(LONG)[5:].tolist() == LONG[5:].tolist()


def test_exit_label_sl_before_tp_and_forced_flat():
    class CX:
        h = np.array([1.0, 1.0, 1.02, 1.0, 1.0, 1.0])
        l = np.array([1.0, 1.0, 1.00, 0.97, 1.0, 1.0])
        c = np.array([1.0, 1.0, 1.01, 0.98, 0.99, 0.995])
    cx = CX()
    assert M.exit_label(cx, 1, (1, 1.0, 0.98, 1.015, 6)) == 0        # bar 2 touches TP before bar 3 touches SL
    assert M.exit_label(cx, 1, (1, 1.0, 0.98, 1.03, 6)) == 1         # TP never reached, SL at bar 3
    assert M.exit_label(cx, 1, (-1, 1.0, 1.015, 0.98, 6)) == 1       # short: bar 2 high 1.02 >= sl first
    assert M.exit_label(cx, 1, (1, 1.0, 0.95, 1.05, 6)) == 1         # nothing hit, negative at forced flat
    assert M.exit_label(cx, 1, (-1, 1.0, 1.05, 0.95, 6)) == 0        # short ends positive at forced flat
