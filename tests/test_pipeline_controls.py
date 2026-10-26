"""DIAGNOSTIC pipeline controls (tools/pipeline_controls.py): the leaked feature, the within-day label shuffle and the
synthetic generator do what the report says they do (small synthetic inputs only)."""
import numpy as np
import pandas as pd

from tools import pipeline_controls as PC


def test_first_touch_up_down_neither_and_ambiguous():
    c = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    h = np.array([1.0, 1.2, 1.0, 1.0, 1.0, 1.0])
    l = np.array([1.0, 1.0, 0.8, 1.0, 1.0, 1.0])
    sig = np.full(6, 0.1)
    ffb = np.full(6, 5)
    out = PC.first_touch(h, l, c, sig, ffb)
    assert out[0] == 1.0          # bar 1 touches +0.1 first
    assert out[1] == -1.0         # bar 2 touches -0.1 first
    assert out[2] == 0.0          # nothing touched afterwards
    h2, l2 = h.copy(), l.copy()
    l2[1] = 0.8                   # both levels inside bar 1 -> ambiguous -> 0
    assert PC.first_touch(h2, l2, c, sig, ffb)[0] == 0.0
    sig_nan = sig.copy()
    sig_nan[1] = np.nan
    assert PC.first_touch(h, l, c, sig_nan, ffb)[0] == 0.0


def test_first_touch_stops_at_forced_flat():
    c = np.ones(6)
    h = np.array([1.0, 1.0, 1.0, 1.3, 1.0, 1.0])
    l = np.ones(6)
    out = PC.first_touch(h, l, c, np.full(6, 0.1), np.array([2, 2, 5, 5, 5, 5]))
    assert out[0] == 0.0 and out[2] == 1.0


def test_permute_within_keeps_groups():
    g = np.array([0, 0, 0, 1, 1, 2])
    perm = PC._permute_within(g, np.random.default_rng(0))
    assert sorted(perm.tolist()) == list(range(6))
    assert (g[perm] == g).all()


def test_synthetic_prices_plant_the_drift():
    ts = pd.date_range("2021-01-04", periods=20000, freq="h", tz="UTC")
    real = 1.1 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 1e-3, len(ts))))
    o, h, l, c, f, gen = PC.synth_prices(ts, real, seed=3, kappa=0.3, half_life=12.0)
    assert (h >= np.maximum(o, c) - 1e-15).all() and (l <= np.minimum(o, c) + 1e-15).all()
    r = np.log(c / o)
    assert np.corrcoef(f[:-1], r[1:])[0, 1] > 0.1     # the factor at b predicts the return of b + 1
    assert abs(np.std(f) - 1) < 0.15
    assert gen["kappa"] == 0.3


def test_auc_basic():
    assert PC.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert np.isnan(PC.auc([0.1, 0.2], [1, 1]))
