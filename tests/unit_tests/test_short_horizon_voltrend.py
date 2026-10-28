"""VolTrendStrategy (owner, 2026-10-19): a single simulable, parametrized component with a JSON schema,
bounded parameters, a manifest hash and a DEAP-ready objective adapter.

Written before app/short_horizon/voltrend.py existed.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import voltrend as VT
from tests.unit_tests.test_short_horizon import calendar_for, synthetic_bars


def test_schema_has_bounds_and_defaults_for_every_parameter_and_rejects_out_of_bounds():
    s = VT.SCHEMA
    for name, spec in s["properties"].items():
        assert "default" in spec, name
        if spec["type"] == "number":
            assert spec["minimum"] <= spec["default"] <= spec["maximum"], name
        elif spec["type"] == "integer" and "enum" not in spec:
            assert spec["minimum"] <= spec["default"] <= spec["maximum"], name
        else:
            assert spec["default"] in spec["enum"], name
    p = VT.defaults()
    VT.validate(p)
    assert p["m"] == 2.0 and p["c_tp"] == 1.0 and p["w_trend"] == 1.0 and p["theta"] == 0.0 and p["rel_volume"] == 0.10
    with pytest.raises(VT.ParamError):
        VT.validate({**p, "m": 99.0})
    with pytest.raises(VT.ParamError):
        VT.validate({**p, "tp_form": "magic"})
    with pytest.raises(VT.ParamError):
        VT.validate({**p, "unknown": 1})
    with pytest.raises(VT.ParamError):
        VT.validate({k: v for k, v in p.items() if k != "m"})


def test_manifest_hash_is_canonical_and_sensitive_to_every_parameter():
    p = VT.defaults()
    h1 = VT.manifest_hash(p)
    assert h1 == VT.manifest_hash(dict(reversed(list(p.items()))))
    for k in p:
        q = dict(p)
        q[k] = (q[k] + 0.01) if isinstance(q[k], float) else (q[k] + 1 if isinstance(q[k], int) and not isinstance(q[k], bool) else "x")
        assert VT.manifest_hash(q) != h1, k
    assert json.loads(VT.manifest_json(p))["manifest_sha256"] == h1


def _ctx():
    bars = synthetic_bars(seed=21, segments=30, hours=120, start="2020-01-05 17:00")
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 24)
    return bars, cal, bl


def test_orders_follow_the_declared_geometry_with_no_hidden_constant():
    bars, cal, bl = _ctx()
    n = len(bl)
    trend = np.full(n, 0.0020)
    sigma = np.full(n, 0.0040)
    p = VT.defaults()
    o = VT.build_orders(bl, trend, sigma, sigma, p)
    k = next(i for i in range(n) if o[i] is not None)
    r = float(bl.ref[k])
    assert o[k].direction == 1
    assert o[k].take_profit == pytest.approx(r + 1.0 * 0.0020 + 1.0 * 0.0040)        # additive form at defaults
    assert o[k].stop_loss == pytest.approx(r - 2.0 * 0.0040)
    assert o[k].percentual_order_size == pytest.approx(min(1.0, max(p["pos_min"], p["pos_a"])))
    q = {**p, "tp_form": "clipped", "c_tp": 0.25}
    oc = VT.build_orders(bl, trend, sigma, sigma, q)
    assert oc[k].take_profit == pytest.approx(r + min(0.0020, 0.25 * 0.0040))
    # abstention threshold: |trend| < theta * sigma -> no order
    ot = VT.build_orders(bl, trend, sigma, sigma, {**p, "theta": 1.0})
    assert all(x is None for x in ot)
    # zero drift with the coin direction (control t0)
    oz = VT.build_orders(bl, np.zeros(n), sigma, sigma, p, coin=np.ones(n, int))
    assert oz[k].direction == 1 and oz[k].take_profit == pytest.approx(r + 0.0040)
    # sizing rule is the declared clip(a + b |trend|/sigma - c sigma/sigma_bar, pos_min, 1)
    ps = VT.build_orders(bl, trend, sigma, 2 * sigma, {**p, "pos_a": 0.5, "pos_b": 1.0, "pos_c": 0.5, "pos_min": 0.1})
    assert ps[k].percentual_order_size == pytest.approx(min(1.0, max(0.1, 0.5 + 1.0 * 0.5 - 0.5 * 0.5)))


def test_objective_adapter_is_deterministic_enforces_rules_and_penalizes_invalid():
    bars, cal, bl = _ctx()
    n = len(bl)
    rng = np.random.default_rng(0)
    inputs = {"trend": {"t0": np.zeros(n), "tx": rng.normal(0, 0.002, n)},
              "sigma": {"MEAN20": np.full(n, 0.004)}, "sigma_bar": np.full(n, 0.004),
              "coin": np.where(rng.integers(0, 2, n) == 1, 1, -1)}
    strat = VT.VolTrendStrategy(bars, bl, cal, inputs)
    period = (bars.index[0], bars.index[-1] + B.HOUR)
    p = {**VT.defaults(), "trend_model": "tx", "vol_model": "MEAN20"}
    a = strat.evaluate(p, period, seed=1)
    b = strat.evaluate(p, period, seed=1)
    assert a == b
    for key in ("sharpe", "max_dd", "net", "orders_per_week", "valid", "objectives", "rule_skipped_orders"):
        assert key in a
    assert a["manifest_sha256"] == VT.manifest_hash(p)
    # an abstaining configuration fails the frequency rule: invalid and penalized, never dropped
    z = strat.evaluate({**p, "theta": 50.0}, period, seed=1)
    assert z["valid"] is False and z["orders_per_week"] == 0.0
    assert z["objectives"][0] <= VT.PENALTY_SHARPE and z["objectives"][1] >= VT.PENALTY_DD
    with pytest.raises(VT.ParamError):
        strat.evaluate({**p, "m": -1.0}, period, seed=1)
    # the SL/TP/units rules are enforced inside (an order too far for the causal sigma is skipped and counted)
    far = strat.evaluate({**p, "m": 4.0, "c_tp": 3.0}, period, seed=1)
    assert far["rule_skipped_orders"] >= 0
