"""S2 absolute-level interface (business contract, owner decision 2026-10-29): the vendored TRAIN profiles and shared
cases are digest-pinned; our re-implemented primitives reproduce Musashi's CASES (agent-multi 82862801) exactly with
HIS synthetic constants injected; his refusal / funding / gap controls hold as fixtures; invalid inputs become NOP with a
reason, never an abort; the engine's absolute mode applies no sigma bounds and no band, clips nothing, keeps the broker
minimum distance and never inflates a TP."""
import hashlib
import math
import random

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import execution as X
from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import exec_variants as XV
from app.short_horizon.expert_guided import horizon_levels as HL
from tools import strategy_output_simulation as S
from tests.test_exec_variants import synth, run

HIS = {"balance": 10_000.0, "tick": 1e-5, "min_distance": 0.5e-4, "slippage_market": 0.2e-4, "slippage_stop": 0.5e-4, "commission_per_lot_side": 3.5, "lot_units": 100_000.0, "funding": 0.0}
HIS_COST = HIS["slippage_market"] + HIS["slippage_stop"] + 2 * HIS["commission_per_lot_side"] / HIS["lot_units"]
SIZING50 = X.Sizing(rel_volume=0.1, leverage=50.0, basis="margin")


def _args(**changes):
    a = dict(direction=1, entry_quote=1.1, liquidation_quote=1.0999, balance=10_000.0, tp_reference=0.003, mae_reference=0.001, sigma=0.001,
             genes=HL.coefficients(), sizing=SIZING50, tick=1e-5, min_distance=0.5e-4, quality=1.0, loss_cost_per_unit=0.00014, funding_reserve_per_unit=0.0)
    a.update(changes)
    return a


def test_vendored_profiles_and_cases_are_pinned_and_the_references_read_his_table():
    assert hashlib.sha256(HL.PROFILES_PATH.read_bytes()).hexdigest() == HL.PROFILES_SHA256
    assert hashlib.sha256(HL.CASES_PATH.read_bytes()).hexdigest() == HL.CASES_SHA256
    prof = HL.load_profiles()
    assert prof["schema"] == "zigzag_train_risk_profiles.v1" and prof["train"] == ["2020-01-01", "2024-01-01"]
    r = HL.references(24, 1, "median", prof)
    assert abs(r["tp_reference"] - 42.1e-4) < 1e-9 and abs(r["mae_reference"] - 16.0e-4) < 1e-9 and r["zigzag"] == {"backstep": 11, "depth": 12, "deviation_pips": 25.0}
    assert HL.references(3, -1, "p75", prof)["tp_reference"] > HL.references(3, -1, "median", prof)["tp_reference"]
    with pytest.raises(HL.LevelInputError):
        HL.references(5, 1, "median", prof)
    assert abs(HL.sigma_h(0.0048, 6) - 0.0048 * math.sqrt(0.25)) < 1e-15


def test_parity_with_his_shared_cases_on_a_stratified_sample():
    """Cross-lane parity: >= 500 of his 3456 cases, stratified by (horizon, direction), replayed through OUR primitives
    with HIS constants: SL / TP / q / units / margin / modeled loss identical (integers exact, floats 1e-9)."""
    cases = HL.load_cases()
    strata = {}
    for c in cases:
        strata.setdefault((c["horizon"], c["direction"]), []).append(c)
    rng = random.Random(20261029)
    sample = [c for k in sorted(strata) for c in rng.sample(strata[k], 70)]
    assert len(sample) >= 500 and len(strata) == 8
    prof = HL.load_profiles()
    for c in sample:
        cell, o = c["cell"], c["order"]
        d = 1 if c["direction"] == "buy" else -1
        ref = HL.references(c["horizon"], d, cell["reference"], prof)
        entry = 1.1
        out = HL.propose(direction=d, entry_quote=entry, liquidation_quote=entry - d * cell["spread_pips"] * 1e-4, balance=HIS["balance"],
                         tp_reference=ref["tp_reference"], mae_reference=ref["mae_reference"], sigma=cell["sigma_pips"] * 1e-4,
                         genes=HL.coefficients(rho=cell["rho"], a_sl=cell["a_sl"], a_tp=cell["a_tp"], buffer_pips=cell["buffer_pips"], b_sl=o["genes"]["b_sl"], b_tp=o["genes"]["b_tp"]),
                         sizing=SIZING50, tick=HIS["tick"], min_distance=HIS["min_distance"], quality=o["quality"], loss_cost_per_unit=HIS_COST, funding_reserve_per_unit=HIS["funding"])
        assert out["status"] == o["status"] == "ORDER", c
        assert out["units"] == o["units"] and isinstance(out["units"], int)
        for k in ("sl", "tp", "q", "margin", "estimated_loss", "loss_per_unit", "raw_sl_distance", "raw_tp_distance", "risk_budget", "loss_at_q1"):
            assert abs(out[k] - o[k]) <= 1e-9, (k, c)


def test_his_geometry_mirror_and_gene_directions():
    a = HL.levels(1, 1.1, 0.003, 0.001, 0.001, HL.coefficients(), 1e-5)
    b = HL.levels(-1, 1.1, 0.003, 0.001, 0.001, HL.coefficients(), 1e-5)
    assert a["sl"] == pytest.approx(1.0989) and a["tp"] == pytest.approx(1.103) and b["sl"] == pytest.approx(1.1011) and b["tp"] == pytest.approx(1.097)
    for name, field, rel in (("a_sl", "sl", "less"), ("b_sl", "sl", "less"), ("buffer_pips", "sl", "less"), ("a_tp", "tp", "greater"), ("b_tp", "tp", "less")):
        g = HL.coefficients()
        g[name] += 0.5
        c = HL.levels(1, 1.1, 0.003, 0.001, 0.001, g, 1e-5)
        assert (c[field] < a[field]) if rel == "less" else (c[field] > a[field])
    t = HL.levels(1, 1.100003, 0.003007, 0.001001, 0.001, HL.coefficients(), 1e-5)   # tick rounding is conservative
    assert t["sl"] <= 1.100003 - 0.001101 and t["tp"] <= 1.100003 + 0.003007


def test_his_eighteen_refusal_funding_and_gap_controls():
    reports = []
    for d in (1, -1):
        base = _args(direction=d, liquidation_quote=1.1 - d * 1e-4, genes=HL.coefficients(rho=0.001))
        fixtures = [("zero_quality", dict(quality=0.0), "NOP"), ("zero_budget", dict(genes=HL.coefficients(rho=0.0)), "NOP"),
                    ("eroded_tp", dict(genes=HL.coefficients(b_tp=4)), "NOP"), ("spread_through_sl", dict(liquidation_quote=1.1 - d * 0.01), "NOP"),
                    ("below_minimum", dict(sizing=X.Sizing(rel_volume=0.1, leverage=50.0, min_units=100_000)), "NOP"),
                    ("zero_mae_unbuffered", dict(mae_reference=0.0, genes=HL.coefficients(buffer_pips=0.0)), "NOP"),
                    ("zero_mae_buffered", dict(mae_reference=0.0, liquidation_quote=1.1 - d * 0.00002), "ORDER")]
        for name, ch, expected in fixtures:
            assert HL.propose(**(base | ch))["status"] == expected, (d, name)
            reports.append((d, name))
        baseline = HL.propose(**base)
        funded = HL.propose(**(base | dict(funding_reserve_per_unit=0.0005)))
        assert funded["q"] < baseline["q"] and funded["estimated_loss"] <= funded["risk_budget"]
        reports.append((d, "funding_reduces_q"))
        gap_price = baseline["sl"] - d * 0.01                            # a 100-pip gap through the SL: the modeled loss is NOT a bound (W9 stays)
        hit = X._protect(d, baseline["tp"], baseline["sl"], gap_price, gap_price, gap_price)
        gap_loss = baseline["units"] * (d * (base["entry_quote"] - hit[1]) + base["loss_cost_per_unit"])
        assert hit[0] == "SL" and gap_loss > baseline["risk_budget"]
        reports.append((d, "gap_exceeds_nominal_budget"))
    assert len(reports) == 18
    a = HL.propose(**_args(genes=HL.coefficients(rho=0.01)))
    assert a["q"] == 1 and 999 < a["margin"] <= 1000 and a["loss_per_unit"] == pytest.approx(0.00124)
    b = HL.propose(**_args(quality=0.3, genes=HL.coefficients(rho=0.01)))
    assert b["q"] == pytest.approx(0.3) and 299 < b["margin"] <= 300
    assert HL.propose(**_args(liquidation_quote=1.0999))["loss_per_unit"] == HL.propose(**_args(liquidation_quote=1.0998))["loss_per_unit"]   # spread not debited twice
    assert HL.propose(**_args(liquidation_quote=1.098))["reason"] == "LEVELS_NOT_EXECUTABLE"
    wide = HL.coefficients(rho=0.001)
    wide["a_sl"] = 2
    assert HL.propose(**_args(genes=wide))["q"] < HL.propose(**_args(genes=HL.coefficients(rho=0.001)))["q"]


@pytest.mark.parametrize("change", [dict(direction=0), dict(quality=1.1), dict(loss_cost_per_unit=-1), dict(sigma=float("nan")), dict(sigma=-1.0),
                                    dict(sigma=float("inf")), dict(tick=0), dict(funding_reserve_per_unit=-1), dict(balance=0), dict(liquidation_quote=1.2)])
def test_invalid_inputs_raise_in_the_primitive_and_become_a_counted_nop_in_the_consumer(change):
    with pytest.raises(HL.LevelInputError):
        HL.propose(**_args(**change))
    out = HL.safe_propose(**_args(**change))
    assert out["status"] == "NOP" and out["reason"].startswith("INVALID_INPUT:") and out["units"] == 0


def test_bad_genes_and_the_sizing_contract_are_refused_or_turned_into_nop():
    with pytest.raises(HL.LevelInputError):
        HL.coefficients(surprise=1)
    assert HL.safe_propose(**_args(genes={"a_sl": -1.0}))["reason"].startswith("INVALID_INPUT:INVALID_RISK_GENE")
    with pytest.raises(HL.LevelInputError, match="BUSINESS"):
        HL.propose(**_args(sizing=X.Sizing(rel_volume=0.1, leverage=100.0)))


def _absolute_policy(levels_by_bar):
    def act(b, r, st, env):
        if env.pos is not None:
            return E.HOLD, 1.0, 1.0
        tp, sl = levels_by_bar(b, env)
        return E.LONG, tp, sl
    return act


def test_engine_absolute_mode_applies_no_bounds_no_band_clips_nothing_and_keeps_the_broker_minimum():
    bars, cal, sig, sr = synth(weeks=1)
    f4 = XV.ExecVariant(reverse_counts_as_one_entry=True, sizing_mode="risk", risk_pct=0.01, short_trigger_on_ask=True, leverage=50.0, no_daily_cap=True, level_mode="absolute_pips")
    cfg = S.with_variant(E.EnvConfig(level_policy="clip"), f4)
    big = 6.0 * sig[0]                                                 # 6 sigma: beyond the sigma mode's 3-sigma bound and the 1.5 x sigma_rule band

    def wide(b, env):
        r = float(env.c[b])
        return r + big, r - big
    env, m = run(bars, cal, sig, sr, cfg, _absolute_policy(wide))
    assert env.counts["entries"] > 0 and env.counts["levels_clipped"] == 0 and m["n_levels_clipped"] == 0
    t = env.trades[0]
    assert abs((t["take_profit"] - float(env.c[t["entry_bar_abs"] - 1])) - big) < 1e-12 and abs((float(env.c[t["entry_bar_abs"] - 1]) - t["stop_loss"]) - big) < 1e-12   # the contracted levels, untouched
    # the sigma mode clips the same request: the absolute mode is the only way past the bounds
    cfg_sigma = S.with_variant(E.EnvConfig(level_policy="clip"), XV.ExecVariant(reverse_counts_as_one_entry=True, sizing_mode="risk", risk_pct=0.01, short_trigger_on_ask=True, leverage=50.0, no_daily_cap=True))
    env_s, _ = run(bars, cal, sig, sr, cfg_sigma, lambda b, r, st, env: ((E.LONG if env.pos is None else E.HOLD), 6.0, 6.0))
    assert env_s.counts["levels_clipped"] > 0

    def too_close(b, env):
        r = float(env.c[b])
        return r + 0.2e-4, r - 0.2e-4                                 # inside the broker minimum: refused, never clipped outward
    env2, _ = run(bars, cal, sig, sr, cfg, _absolute_policy(too_close))
    assert env2.counts["entries"] == 0 and env2.counts["refused_LEVELS"] > 0 and env2.counts["levels_clipped"] == 0

    def ask_side(b, env):                                              # a short's SL must clear the ASK-side liquidation quote too
        r = float(env.c[b])
        return r - 30e-4, r + 0.6e-4
    sh = lambda b, r, st, env: ((E.SHORT if env.pos is None else E.HOLD),) + ask_side(b, env)  # noqa: E731
    env3, _ = run(bars, cal, sig, sr, cfg, sh)
    assert env3.counts["entries"] == 0 and env3.counts["refused_LEVELS"] > 0
    assert XV.LEGACY.identity == "exec_5d80473b21dc61a0" and C3.FULL_F3.identity == "exec_709f9b1fa3512486"
    assert "level_mode" not in C3.FULL_F3.document() and C3.FULL_F4.document()["level_mode"] == "absolute_pips"
