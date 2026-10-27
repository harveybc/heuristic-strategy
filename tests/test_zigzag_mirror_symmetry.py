"""Owner rule 2026-10-27: the production ZigZags are symmetric under price negation — zigzag(K - path) equals the
mirror of zigzag(path) EXACTLY (same bars, peak <-> valley, levels mirrored) on random paths, for the reversal
ZigZag at every ladder threshold and for the expert MetaTrader ZigZag. The same implementations are used by the
behaviour matrix, the labels and production; the legacy first-pivot rules are kept only to measure the change."""
import numpy as np
import pytest

from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import oracle as OR


def _path(seed, n=3000, sd=7e-4, drift=0.0):
    rng = np.random.default_rng(seed)
    c = 1.10 * np.exp(np.cumsum(drift + sd * rng.standard_normal(n)))
    o = np.r_[c[0], c[:-1]]
    intr = np.abs(rng.standard_normal(n)) * sd * 0.6
    return np.maximum(o, c) * (1 + intr), np.minimum(o, c) * (1 - intr)


def _mirror(piv, K):
    return [(int(b), -int(k), K - float(v)) for b, k, v in piv]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("pips", [5, 8, 10, 15, 20, 60, 100, 200, 400, 800])
def test_reversal_zigzag_is_exactly_mirror_symmetric(seed, pips):
    h, l = _path(seed, drift=(seed % 3 - 1) * 3e-5)
    K = 2.0 * float(h.max())
    p1, c1 = OR.reversal_pivots(h, l, pips * Z.PIP)
    p2, c2 = OR.reversal_pivots(K - l, K - h, pips * Z.PIP)
    m = _mirror(p1, K)
    assert [(b, k) for b, k, _ in p2] == [(b, k) for b, k, _ in m]
    assert np.allclose([v for _, _, v in p2], [v for _, _, v in m], atol=1e-12, rtol=0)
    assert c1 == c2
    assert all(c > b for (b, _, _), c in zip(p1, c1))                       # every pivot confirmed strictly after its bar


@pytest.mark.parametrize("seed", range(12))
def test_expert_metatrader_zigzag_is_exactly_mirror_symmetric(seed):
    h, l = _path(100 + seed, n=2500)
    K = 2.0 * float(h.max())
    p = Z.Params(depth=3, deviation_pips=40.7, backstep=0)
    p1 = Z.hindsight_pivots(h, l, p)
    p2 = Z.hindsight_pivots(K - l, K - h, p)
    m = _mirror(p1, K)
    assert [(b, k) for b, k, _ in p2] == [(b, k) for b, k, _ in m]
    assert np.allclose([v for _, _, v in p2], [v for _, _, v in m], atol=1e-12, rtol=0)
    for a, b in zip(p1, p1[1:]):
        assert a[1] != b[1]                                                  # alternation


def test_the_change_is_confined_to_the_first_pivots():
    """Against the legacy rules, only the pivots up to the first confirmation can differ."""
    for seed in range(8):
        h, l = _path(200 + seed)
        new, conf = OR.reversal_pivots(h, l, 20 * Z.PIP)
        old, _ = OR.reversal_pivots_legacy(h, l, 20 * Z.PIP)
        tail_new = [x[0] for x in new[2:]]
        assert set(tail_new) <= {x[0] for x in old}
        p = Z.Params(depth=3, deviation_pips=40.7, backstep=0)
        a, b = Z.hindsight_pivots(h, l, p), Z.hindsight_pivots_legacy(h, l, p)
        assert [x[0] for x in a[2:]] == [x[0] for x in b[-len(a[2:]):]] if len(a) > 2 else True
