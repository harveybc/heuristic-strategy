"""S3 risk-budget sizing (business contract): in the engine, under FULL_F4, every entry's units are floor(q x units_max)
with q = c x min(1, rho x balance / L1), L1 = units_max x (SL distance + our cost per unit + funding reserve); the modeled
loss never exceeds rho x the balance at entry; the in-engine numbers equal the horizon_levels primitive fed the same
inputs; rho = 0 refuses every entry with a counted reason; the audited defaults are untouched."""
import math

import numpy as np

from app.short_horizon import execution as X
from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import exec_variants as XV
from app.short_horizon.expert_guided import horizon_levels as HL
from tools import strategy_output_simulation as S
from tests.test_exec_variants import synth


def _run_capturing(bars, cal, sig, sr, cfg, act):
    env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + __import__("pandas").Timedelta(hours=1), cfg)
    sized = []
    for b in range(env.first, env.last):
        n0 = env.counts["entries"]
        st = env.state(b)
        a, tp, sl = act(b, b, st, env)
        bal = env.balance
        env.step(b, a, tp, sl)
        if env.counts["entries"] > n0:
            sized.append({"bar": b, "balance": bal, "ref": float(env.c[b]), "liq": env.liquidation_quote(b, 1), **env.last_sizing})
    return env, env.result(), sized


def _policy(d_sl=30e-4, d_tp=60e-4, hold_bars=6):
    def act(b, r, st, env):
        if env.pos is not None:
            return (E.CLOSE if b - env.pos.entry_bar >= hold_bars else E.HOLD), 1.0, 1.0   # many independent entries
        ref = float(env.c[b])
        return E.LONG, ref + d_tp, ref - d_sl
    return act


def test_risk_budget_units_in_the_engine_match_the_primitive_and_respect_the_budget():
    bars, cal, sig, sr = synth(weeks=2)
    cfg = S.with_variant(E.EnvConfig(level_policy="clip"), C3.FULL_F4)
    assert cfg.variant.sizing_mode == "risk_budget" and cfg.sizing.leverage == 50.0 and cfg.max_entries_per_day is None
    env, m, sized = _run_capturing(bars, cal, sig, sr, cfg, _policy())
    assert len(env.trades) >= 5 and len(sized) == len(env.trades)
    prof = cfg.profile
    cost = prof.slippage_market + prof.slippage_stop + 2 * prof.commission_per_lot_side / prof.lot_units
    for t, s_ in zip(env.trades, sized):
        umax = cfg.sizing.units_max(s_["balance"], s_["ref"])
        assert t["units"] == int(math.floor(s_["q"] * umax)) and t["units"] >= 1
        # coordinator ruling 2026-10-29 (DA26, Musashi's rule): under FULL_F4 the funding reserve is the known financing cost
        # per unit to the forced flat (it was the constant 0 when this test was written in S3); the long side here pays swap
        reserve = env.funding_reserve_to_flat(int(t["entry_bar_abs"]), 1)
        loss = t["units"] * (abs(s_["ref"] - t["stop_loss"]) + cost + reserve)
        assert loss <= 0.005 * s_["balance"] + 1e-9                    # the modeled loss within rho x the balance at entry
        assert abs(s_["loss_per_unit"] - (abs(s_["ref"] - t["stop_loss"]) + cost + reserve)) < 1e-15
        # the same q from the shared sizing primitive on the same inputs (our constants, the engine's SL distance)
        sized = HL.size_from_risk(s_["balance"], 0.1, 0.005 * s_["balance"], umax * (abs(s_["ref"] - t["stop_loss"]) + cost + reserve), 1.0)
        assert abs(sized["q"] - s_["q"]) < 1e-12 and t["units"] == int(math.floor(sized["q"] * umax))
        assert abs(t["margin_usd"] - t["units"] * t["entry_price"] / 50.0) < 1e-9
    assert env.counts["levels_clipped"] == 0 and m["n_levels_clipped"] == 0


def test_zero_budget_refuses_every_entry_with_a_counted_reason_and_defaults_are_untouched():
    bars, cal, sig, sr = synth(weeks=1)
    v0 = XV.ExecVariant(reverse_counts_as_one_entry=True, sizing_mode="risk_budget", rho=0.0, short_trigger_on_ask=True, leverage=50.0, no_daily_cap=True, level_mode="absolute_pips")
    env, m, sized = _run_capturing(bars, cal, sig, sr, S.with_variant(E.EnvConfig(level_policy="clip"), v0), _policy())
    assert env.counts["entries"] == 0 and env.counts["refused_UNITS"] > 0 and not sized
    rb = XV.risk_budget_units(10_000.0, 1.1, 30e-4, 1.4e-4, 0.0, 0.0, 1.0, X.Sizing(rel_volume=0.1, leverage=50.0))
    assert rb["units"] is None and rb["reason"] == "ZERO_BUDGET_OR_QUALITY"
    half = XV.risk_budget_units(10_000.0, 1.1, 30e-4, 1.4e-4, 0.0, 0.005, 0.5, X.Sizing(rel_volume=0.1, leverage=50.0))
    full = XV.risk_budget_units(10_000.0, 1.1, 30e-4, 1.4e-4, 0.0, 0.005, 1.0, X.Sizing(rel_volume=0.1, leverage=50.0))
    assert half["units"] == int(math.floor(0.5 * full["q"] * X.Sizing(rel_volume=0.1, leverage=50.0).units_max(10_000.0, 1.1)))
    assert XV.LEGACY.identity == "exec_5d80473b21dc61a0" and C3.FULL_F3.identity == "exec_709f9b1fa3512486"
    assert E.EnvConfig().sizing.leverage == 100.0 and E.EnvConfig().max_entries_per_day == 4 and XV.LEGACY.sizing_mode == "notional_share"
