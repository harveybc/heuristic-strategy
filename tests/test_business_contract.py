"""S1 of the shared business contract (owner decision 2026-10-29, Musashi's packet agent-multi 82862801): the vendored
contract is byte-pinned; the engine supports no daily cap natively; the FULL_F4 variant applies leverage 50 and no cap
through with_variant; every earlier execution identity and the engine defaults are byte-identical."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import execution as X
from app.short_horizon.expert_guided import business_contract as BC
from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import exec_variants as XV
from tools import strategy_output_simulation as S
from tests.test_exec_variants import synth, run


def test_contract_is_vendored_byte_for_byte_and_pinned():
    raw = BC.BUSINESS_CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BC.BUSINESS_CONTRACT_SHA256 == "73f32e43005da04ae44bcfffb576624513e8040aa8df3f06c521be5b7495b036"
    p = BC.profile()
    assert p == {"leverage": 50.0, "rel_volume": 0.1, "max_entries_per_day": None, "entry_wait_after_session_open_hours": 12.0,
                 "forced_flat_before_session_close_hours": 12.0, "max_simultaneous_positions": 1, "early_close_enabled": True,
                 "schema": "mechanism_synthesis.business_contract.v1", "sha256": BC.BUSINESS_CONTRACT_SHA256}
    with pytest.raises(BC.BusinessContractError):
        BC.load(expected_sha256="0" * 64)


def test_earlier_identities_and_engine_defaults_are_unchanged():
    assert XV.LEGACY.identity == "exec_5d80473b21dc61a0" and C3.FULL_F3.identity == "exec_709f9b1fa3512486"
    assert "leverage" not in XV.LEGACY.document() and "no_daily_cap" not in C3.FULL_F3.document()
    cfg = E.EnvConfig()
    assert cfg.max_entries_per_day == 4 and cfg.sizing.leverage == 100.0 and cfg.sizing.rel_volume == 0.10
    f4 = C3.FULL_F4
    assert f4.leverage == 50.0 and f4.no_daily_cap and f4.entry_cutoff_hours_before_flat is None and not f4.is_legacy()
    assert f4.document()["leverage"] == 50.0 and f4.document()["no_daily_cap"] is True and C3.EXEC_VARIANTS["full_f4"] is f4
    with pytest.raises(ValueError):
        XV.ExecVariant(leverage=200.0)


def test_with_variant_applies_leverage_and_no_daily_cap_only_for_the_contract_profile():
    base = E.EnvConfig()
    assert S.with_variant(base, XV.LEGACY) == base
    f3 = S.with_variant(base, C3.FULL_F3)
    assert f3.max_entries_per_day == 4 and f3.sizing.leverage == 100.0 and f3.variant is C3.FULL_F3
    f4 = S.with_variant(base, C3.FULL_F4)
    assert f4.max_entries_per_day is None and f4.sizing.leverage == 50.0 and f4.sizing.rel_volume == 0.10 and f4.variant is C3.FULL_F4


def _open_every_bar(b, r, st, env):
    return (E.LONG if env.pos is None else E.HOLD), 1.0, 1.0


def _close_and_reopen(b, r, st, env):
    """A close every bar while positioned, a new long every flat bar: as many entries per day as the engine allows."""
    return (E.CLOSE if env.pos is not None else E.LONG), 1.0, 1.0


def test_no_daily_cap_admits_more_than_four_entries_a_day_and_the_cap_still_refuses_the_fifth():
    bars, cal, sig, sr = synth(weeks=1)
    capped = E.EnvConfig(level_policy="clip", max_entries_per_day=4)
    env_c, m_c = run(bars, cal, sig, sr, capped, _close_and_reopen)
    assert env_c.counts["refused_DAILY_CAP"] > 0 and m_c["max_entries_per_day"] <= 4 and m_c["valid"] in (True, False)
    free = E.EnvConfig(level_policy="clip", max_entries_per_day=None)
    env_f, m_f = run(bars, cal, sig, sr, free, _close_and_reopen)
    assert env_f.counts["refused_DAILY_CAP"] == 0 and m_f["max_entries_per_day"] > 4 and env_f.counts["entries"] > env_c.counts["entries"]
    st = env_f.state(env_f.first + 30)
    assert st[7] == 0.0                                               # the daily-cap share of the state is 0 under no cap
    assert m_f["valid_weekly_rule"] is True


def test_contract_leverage_halves_the_unit_cap_and_doubles_the_margin_per_unit():
    bars, cal, sig, sr = synth(weeks=1)
    base = E.EnvConfig(level_policy="clip")
    f4 = S.with_variant(base, C3.FULL_F4)
    env_b, _ = run(bars, cal, sig, sr, base, _open_every_bar)
    env_4, _ = run(bars, cal, sig, sr, f4, _open_every_bar)
    tb, t4 = env_b.trades[0], env_4.trades[0]
    assert t4["entry_bar_abs"] == tb["entry_bar_abs"]
    assert abs(t4["margin_usd"] - t4["units"] * t4["entry_price"] / 50.0) < 1e-9 and abs(tb["margin_usd"] - tb["units"] * tb["entry_price"] / 100.0) < 1e-9
    assert t4["units"] <= X.Sizing(rel_volume=0.10, leverage=50.0).units_max(10_000.0, t4["entry_price"])
