"""Tests for app/policies/business_pair_policy.py (I6-DMO strategy lane).

The module is loaded by explicit file path, exactly as the predictor adapter
loads the pinned blob, so ``app/__init__`` and ``app/policies/__init__`` (which
imports ``trading_contracts``) are never executed for the pure-policy tests.
Only the parity test imports the legacy policy module.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "app" / "policies" / "business_pair_policy.py"


def _load():
    name = "hs_business_pair_policy_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bp = _load()

ENTRY, TP_LONG, SL_LONG = 1.10000, 1.10100, 1.09800
TP_SHORT, SL_SHORT = 1.09900, 1.10200


def head(family, h, value, reason="PREDICTION_UNAVAILABLE"):
    if value is None:
        return bp.HeadForecast(family=family, horizon_hours=h, available=False,
                               value=None, reason=reason)
    return bp.HeadForecast(family=family, horizon_hours=h, available=True, value=value)


def short_path(values):
    return [head("short", h, v) for h, v in zip(bp.SHORT_HORIZONS_HOURS, values)]


def flat_long(value=ENTRY):
    return [head("long", h, value) for h in bp.LONG_HORIZONS_HOURS]


# Audit construction (I6-N addendum counterexample_v2): S1 = TP at 2 h, SL at 12 h.
S1_LONG = [1.10120, 1.10060, 1.10040, 1.10020, 1.10000, 1.09750]
S2_LONG = list(reversed(S1_LONG))  # SL at 2 h, TP at 12 h
S1_SHORT = [2 * ENTRY - v for v in S1_LONG]  # mirror: TP (<=1.099) at 2 h, SL (>=1.102) at 12 h
S2_SHORT = list(reversed(S1_SHORT))


def test_family_horizons_are_the_contract_values():
    assert bp.SHORT_HORIZONS_HOURS == (2, 3, 4, 5, 6, 12)
    assert bp.LONG_HORIZONS_HOURS == (18, 24, 36, 48, 60, 72)
    assert isinstance(bp.RULES_VERSION, str) and bp.RULES_VERSION


@pytest.mark.parametrize("direction,tp,sl,s1,s2", [
    ("long", TP_LONG, SL_LONG, S1_LONG, S2_LONG),
    ("short", TP_SHORT, SL_SHORT, S1_SHORT, S2_SHORT),
])
def test_reversed_hit_order_changes_the_decision(direction, tp, sl, s1, s2):
    d1 = bp.early_close_decision(direction=direction, take_profit=tp, stop_loss=sl,
                                 heads=short_path(s1) + flat_long())
    d2 = bp.early_close_decision(direction=direction, take_profit=tp, stop_loss=sl,
                                 heads=short_path(s2) + flat_long())
    assert (d1.action, d1.reason, d1.first_crossing_horizon_hours) == ("HOLD", "TP_FIRST", 2)
    assert (d2.action, d2.reason, d2.first_crossing_horizon_hours) == ("EARLY_CLOSE", "SL_FIRST", 2)
    assert sorted(s1) == sorted(s2)  # same multiset of values: min/max cannot separate them


def test_order_is_numeric_horizon_order_across_both_families():
    # TP only in the long family at 18 h; SL in the short family at 12 h -> SL first.
    heads = [head("long", 18, TP_LONG + 0.001)] + short_path(
        [ENTRY, ENTRY, ENTRY, ENTRY, ENTRY, SL_LONG - 0.001])
    heads += [head("long", h, ENTRY) for h in (24, 36, 48, 60, 72)]
    random.Random(7).shuffle(heads)  # input order is irrelevant
    d = bp.early_close_decision(direction="long", take_profit=TP_LONG, stop_loss=SL_LONG,
                                heads=heads)
    assert (d.action, d.reason, d.first_crossing_horizon_hours,
            d.first_crossing_family) == ("EARLY_CLOSE", "SL_FIRST", 12, "short")
    assert [c[1] for c in d.consulted] == [2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72]


def test_long_family_can_decide_when_short_family_has_no_crossing():
    heads = short_path([ENTRY] * 6) + [head("long", 18, SL_LONG - 1e-4)] + \
        [head("long", h, TP_LONG + 1e-4) for h in (24, 36, 48, 60, 72)]
    d = bp.early_close_decision(direction="long", take_profit=TP_LONG, stop_loss=SL_LONG,
                                heads=heads)
    assert (d.action, d.reason, d.first_crossing_horizon_hours) == ("EARLY_CLOSE", "SL_FIRST", 18)


@pytest.mark.parametrize("direction,tp,sl", [("long", TP_LONG, SL_LONG),
                                             ("short", TP_SHORT, SL_SHORT)])
def test_exact_barrier_equality_counts_as_a_touch(direction, tp, sl):
    d_sl = bp.early_close_decision(direction=direction, take_profit=tp, stop_loss=sl,
                                   heads=[head("short", 2, sl), head("short", 3, tp)])
    d_tp = bp.early_close_decision(direction=direction, take_profit=tp, stop_loss=sl,
                                   heads=[head("short", 2, tp), head("short", 3, sl)])
    assert (d_sl.action, d_sl.reason) == ("EARLY_CLOSE", "SL_FIRST")
    assert (d_tp.action, d_tp.reason) == ("HOLD", "TP_FIRST")
    assert bp.BARRIER_EQUALITY_RULE == "INCLUSIVE_TOUCH"


def test_no_crossing_holds_and_no_available_heads_holds_with_distinct_reason():
    d = bp.early_close_decision(direction="long", take_profit=TP_LONG, stop_loss=SL_LONG,
                                heads=short_path([ENTRY] * 6) + flat_long())
    assert (d.action, d.reason, d.first_crossing_horizon_hours) == (
        "HOLD", "NO_SAMPLED_CROSSING", None)
    empty = [head("short", h, None) for h in bp.SHORT_HORIZONS_HOURS]
    d0 = bp.early_close_decision(direction="long", take_profit=TP_LONG, stop_loss=SL_LONG,
                                 heads=empty)
    assert (d0.action, d0.reason) == ("HOLD", "NO_AVAILABLE_HEADS")


def test_missing_head_is_skipped_never_filled():
    # The 2 h head would be the SL crossing, but it is unavailable: it must not be
    # replaced by any value (current price or otherwise); the 3 h TP decides.
    heads = [head("short", 2, None), head("short", 3, TP_LONG + 1e-4),
             head("short", 4, SL_LONG - 1e-4)]
    d = bp.early_close_decision(direction="long", take_profit=TP_LONG, stop_loss=SL_LONG,
                                heads=heads)
    assert (d.action, d.reason, d.first_crossing_horizon_hours) == ("HOLD", "TP_FIRST", 3)
    assert (("short", 2) not in [(c[0], c[1]) for c in d.consulted])
    assert d.skipped == (("short", 2, "PREDICTION_UNAVAILABLE"),)


def test_head_record_contract_is_enforced():
    with pytest.raises(bp.PolicyError) as e:
        bp.HeadForecast(family="short", horizon_hours=2, available=True, value=float("nan"))
    assert e.value.code == "NON_FINITE_AVAILABLE_VALUE"
    with pytest.raises(bp.PolicyError) as e:
        bp.HeadForecast(family="short", horizon_hours=2, available=False, value=1.1,
                        reason="X")
    assert e.value.code == "UNAVAILABLE_HEAD_CARRIES_VALUE"
    with pytest.raises(bp.PolicyError) as e:
        bp.HeadForecast(family="short", horizon_hours=2, available=False, value=None)
    assert e.value.code == "UNAVAILABLE_HEAD_WITHOUT_REASON"
    with pytest.raises(bp.PolicyError) as e:
        bp.HeadForecast(family="short", horizon_hours=18, available=True, value=1.1)
    assert e.value.code == "HORIZON_NOT_IN_FAMILY"
    with pytest.raises(bp.PolicyError) as e:
        bp.HeadForecast(family="short", horizon_hours=True, available=True, value=1.1)
    assert e.value.code == "HORIZON_NOT_IN_FAMILY"


def test_duplicate_heads_and_invalid_barriers_are_refused():
    with pytest.raises(bp.PolicyError) as e:
        bp.early_close_decision(direction="long", take_profit=TP_LONG, stop_loss=SL_LONG,
                                heads=[head("short", 2, ENTRY), head("short", 2, ENTRY)])
    assert e.value.code == "DUPLICATE_HEAD"
    with pytest.raises(bp.PolicyError) as e:
        bp.early_close_decision(direction="long", take_profit=SL_LONG, stop_loss=TP_LONG,
                                heads=[head("short", 2, ENTRY)])
    assert e.value.code == "INVALID_BARRIERS"
    with pytest.raises(bp.PolicyError) as e:
        bp.early_close_decision(direction="short", take_profit=SL_SHORT, stop_loss=TP_SHORT,
                                heads=[head("short", 2, ENTRY)])
    assert e.value.code == "INVALID_BARRIERS"
    with pytest.raises(bp.PolicyError) as e:
        bp.early_close_decision(direction="flat", take_profit=TP_LONG, stop_loss=SL_LONG,
                                heads=[head("short", 2, ENTRY)])
    assert e.value.code == "INVALID_DIRECTION"


PARAMS = dict(pip_cost=0.00001, rel_volume=0.02, min_order_volume=10_000.0,
              max_order_volume=1_000_000.0, leverage=100.0, profit_threshold=5.0,
              min_drawdown_pips=10.0, tp_multiplier=0.9, sl_multiplier=2.0,
              lower_rr_threshold=0.5, upper_rr_threshold=2.0, max_trades_per_5days=3,
              min_valid_long_heads=2)


def test_entry_uses_long_family_only_and_refuses_short_heads():
    params = bp.EntryParameters(**PARAMS)
    longs = [head("long", h, v) for h, v in zip(bp.LONG_HORIZONS_HOURS,
                                                  [1.1003, 1.1008, 1.1012, 1.1006, 1.0998, 1.1001])]
    d = bp.entry_decision(current_price=ENTRY, long_heads=longs, params=params)
    assert d.action == "OPEN_LONG" and d.geometry is not None
    assert [c[0] for c in d.consulted] == ["long"] * 6
    with pytest.raises(bp.PolicyError) as e:
        bp.entry_decision(current_price=ENTRY, long_heads=longs + [head("short", 2, 1.2)],
                          params=params)
    assert e.value.code == "SHORT_HEAD_IN_ENTRY"


def test_entry_minimum_valid_long_heads_abstains_and_never_fills():
    params = bp.EntryParameters(**PARAMS)
    one = [head("long", 18, 1.1050)] + [head("long", h, None) for h in (24, 36, 48, 60, 72)]
    d = bp.entry_decision(current_price=ENTRY, long_heads=one, params=params)
    assert (d.action, d.reason, d.geometry) == ("ABSTAIN", "INSUFFICIENT_LONG_HEADS", None)
    two = [head("long", 18, 1.1050), head("long", 72, 1.1040)] + \
        [head("long", h, None) for h in (24, 36, 48, 60)]
    d2 = bp.entry_decision(current_price=ENTRY, long_heads=two, params=params)
    assert d2.action == "OPEN_LONG"
    # The missing heads were not replaced by the current price: min over the two
    # available values is 1.1040, so drawdown is the floor (10 pips), not 0.
    assert math.isclose(d2.geometry.ideal_drawdown_pips, 10.0)
    assert math.isclose(d2.geometry.ideal_profit_pips, (1.1050 - ENTRY) / 0.00001)


def test_entry_is_unchanged_by_short_family_but_early_close_is_not():
    params = bp.EntryParameters(**PARAMS)
    longs = [head("long", h, v) for h, v in zip(bp.LONG_HORIZONS_HOURS,
                                                  [1.1003, 1.1008, 1.1012, 1.1006, 1.0998, 1.1001])]
    a = bp.entry_decision(current_price=ENTRY, long_heads=longs, params=params)
    b = bp.entry_decision(current_price=ENTRY, long_heads=list(reversed(longs)), params=params)
    assert a == b  # entry is defined by min/max of the longs (order-free by contract)
    tp, sl = a.geometry.take_profit_price, a.geometry.stop_loss_price
    hold = bp.early_close_decision(direction="long", take_profit=tp, stop_loss=sl,
                                   heads=short_path([ENTRY] * 6) + longs)
    close = bp.early_close_decision(direction="long", take_profit=tp, stop_loss=sl,
                                    heads=short_path([sl - 1e-5] + [ENTRY] * 5) + longs)
    assert hold.action == "HOLD" and close.action == "EARLY_CLOSE"


def test_order_units_formula_matches_legacy_and_entry_params_validate():
    params = bp.EntryParameters(**PARAMS)
    assert bp.compute_order_units(reward_risk_ratio=3.0, available_cash=10_000.0,
                                  params=params) == 20_000.0  # cash cap 10000*0.02*100
    assert bp.compute_order_units(reward_risk_ratio=0.1, available_cash=1e9,
                                  params=params) == 10_000.0
    with pytest.raises(bp.PolicyError) as e:
        bp.EntryParameters(**{**PARAMS, "min_valid_long_heads": 0})
    assert e.value.code == "PARAMETER_INVALID"


def test_entry_geometry_is_bitwise_equal_to_the_legacy_policy():
    legacy = pytest.importorskip("app.policies.prediction_entry_exit",
                                 reason="legacy policy needs trading_contracts")
    legacy_params = legacy.PredictionEntryExitParameters(
        **{k: v for k, v in PARAMS.items() if k != "min_valid_long_heads"})
    params = bp.EntryParameters(**PARAMS)
    rng = random.Random(6102026)
    checked = 0
    for _ in range(500):
        price = 1.1 + rng.uniform(-0.01, 0.01)
        values = [price + rng.uniform(-0.004, 0.004) for _ in bp.LONG_HORIZONS_HOURS]
        old = legacy.calculate_entry_geometry(current_price=price,
                                              long_horizon_predictions=values,
                                              params=legacy_params)
        new = bp.entry_decision(
            current_price=price,
            long_heads=[head("long", h, v) for h, v in zip(bp.LONG_HORIZONS_HOURS, values)],
            params=params)
        if old is None:
            assert new.action == "ABSTAIN" and new.reason == "NO_ENTRY_EDGE"
        else:
            assert new.action == ("OPEN_LONG" if old.direction == "long" else "OPEN_SHORT")
            g = new.geometry
            assert (g.take_profit_price, g.stop_loss_price, g.reward_risk_ratio,
                    g.ideal_profit_pips, g.ideal_drawdown_pips) == (
                old.take_profit_price, old.stop_loss_price, old.reward_risk_ratio,
                old.ideal_profit_pips, old.ideal_drawdown_pips)
            checked += 1
        assert bp.compute_order_units(reward_risk_ratio=1.3, available_cash=price * 9000,
                                      params=params) == legacy.compute_legacy_order_size(
            reward_risk_ratio=1.3, available_cash=price * 9000, params=legacy_params)
    assert checked > 50


def test_module_is_pure_stdlib():
    import ast
    tree = ast.parse(MODULE_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import would require the app package"
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "math", "dataclasses", "typing"}, imported
