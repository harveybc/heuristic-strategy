"""Early-close audit (owner 2026-10-25: early close is indispensable): engine counterfactuals with known answers.

A long opened on flat bars and closed by the agent; afterwards the price either falls through the SL (the close
SAVED the trade) or rises through the TP (the close CUT a winner). The counterfactual is the engine's own hold of the
same position; the no-early-close replay of the same decisions must end exactly where the counterfactual ends."""
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import calendar as CAL
from app.short_horizon.expert_guided import env as E
from tools import expert_guided_early_close_audit as A

HOUR = pd.Timedelta(hours=1)
SIG = 0.0010


def bars_with(after_close):
    """3 segments x 120 h; flat 1.1000 around the trade, then one move bar (``after_close``: 'down' through the SL,
    'up' through the TP), then flat at the new level."""
    rng = np.random.default_rng(3)
    stamps, t = [], pd.Timestamp("2020-01-05 21:00", tz="UTC")
    for _ in range(3):
        stamps += [t + k * HOUR for k in range(120)]
        t = stamps[-1] + 49 * HOUR
    n = len(stamps)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[1.10, close[:-1]]
    high = np.maximum(open_, close) + 0.0003
    low = np.minimum(open_, close) - 0.0003
    df = pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=pd.DatetimeIndex(stamps, name="DATE_TIME"))
    return df


def setup(after_close):
    bars = bars_with(after_close)
    segs = CAL.derive_segments(bars.index, max_gap_hours=24)
    cal = CAL.load_calendar({"calendar_id": "TEST", "version": "t1", "segments": segs, "entry_delay_after_open_hours": 12,
                             "force_flat_before_close_hours": 12, "entry_cutoff_before_flat_hours": 0})
    probe = E.BarEnv(bars, cal, np.full(len(bars), SIG), np.full(len(bars), 3 * SIG), bars.index[0], bars.index[-1] + HOUR, E.EnvConfig())
    s = cal.segments[0]
    b = int(np.searchsorted(probe.v, s.operational_start.value)) - 1 + 5          # entry decision bar
    lvl = 1.1000
    for k in range(b - 2, b + 40):                                                # flat around the trade
        bars.iloc[k] = [lvl, lvl, lvl, lvl]
    m = b + 9                                                                     # the move bar, after the close at b + 5
    if after_close == "down":
        bars.iloc[m] = [lvl, lvl, lvl - 4 * SIG, lvl - 4 * SIG]
        new = lvl - 4 * SIG
    else:
        bars.iloc[m] = [lvl, lvl + 4 * SIG, lvl, lvl + 4 * SIG]
        new = lvl + 4 * SIG
    for k in range(m + 1, b + 40):
        bars.iloc[k] = [new, new, new, new]
    sigma, sr = np.full(len(bars), SIG), np.full(len(bars), 3 * SIG)
    mk = lambda: E.BarEnv(bars, cal, sigma, sr, bars.index[0], bars.index[-1] + HOUR, E.EnvConfig())  # noqa: E731
    return mk, b


def scripted(b):
    def decide(bar, st, env):
        if bar == b and env.pos is None:
            return E.LONG, 1.5, 2.0
        if bar == b + 5 and env.pos is not None:
            return E.CLOSE, 1.5, 2.0
        return E.HOLD, 1.5, 2.0
    return decide


def exit_prob(b):
    return lambda bar, st: 0.9 if bar == b + 5 else 0.1


@pytest.mark.parametrize("move, cls, cf_reason, sign", [("down", "SAVED", "SL", 1), ("up", "CUT_WINNER", "TP", -1)])
def test_counterfactual_hold_classifies_the_agent_close(move, cls, cf_reason, sign):
    mk, b = setup(move)
    au = A.audit_replay(mk, None, scripted(b), exit_prob(b))
    assert len(au["trades"]) == 1 and au["trades"][0]["close_reason"] == "AGENT_CLOSE"
    c = au["closes"][0]
    assert c["cf_reason"] == cf_reason
    d_sig = (c["actual_net"] - c["cf_net"]) / c["sigma_usd"]
    assert sign * d_sig > 1.0                                                     # ~2 sigma past the SL, ~1.5 sigma TP minus the exit-cost difference
    ne = A.audit_replay(mk, None, A.no_early_close(scripted(b)), None)["metrics"]
    assert math.isclose(ne["net_profit_usd"], c["cf_net"], abs_tol=1e-9)          # the executable replay ends where the counterfactual does
    s = A.summarize(au, 10_000.0, ne)
    assert s["classes"][cls] == 1 and s["agent_closes"] == 1 and s["early_close_share"] == 1.0
    assert sign * s["early_close_value_usd"] > 0 and s["early_close_value_usd_ci95"][0] == s["early_close_value_usd_ci95"][1]
    lab = [r for r in au["pos_bars"] if "sl_first" in r]
    assert len(lab) == 5 and all(r["sl_first"] == (1 if move == "down" else 0) for r in lab)
    assert s["closer"]["fired_rate"] == pytest.approx(1 / 5) and s["direction_balance"]["long"]["entries"] == 1
    assert s["direction_balance"]["short"]["entries"] == 0 and s["no_early_close_policy"]["agent_closes"] == 0
    assert s["flags"] == ([] if move == "down" else ["HARMFUL_EARLY_CLOSE"])


def test_close_now_counterfactual_equals_the_actual_close():
    mk, b = setup("down")
    au = A.audit_replay(mk, None, scripted(b), exit_prob(b))
    r = [x for x in au["pos_bars"] if x["bar"] == b + 5][0]
    assert r["close_now_net"] == pytest.approx(au["closes"][0]["actual_net"], abs=1e-9) and r["fired"]


def test_no_agent_close_is_flagged_degenerate():
    mk, b = setup("down")
    au = A.audit_replay(mk, None, A.no_early_close(scripted(b)), None)
    s = A.summarize(au, 10_000.0)
    assert s["agent_closes"] == 0 and "DEGENERATE_NO_EARLY_CLOSE" in s["flags"]


def test_auc_bootstrap_classify_and_labels_known_answers():
    assert A.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert math.isnan(A.auc([0.1, 0.2], [1, 1]))
    assert A.boot_sum_ci([2.0] * 7) == [14.0, 14.0]
    assert A.classify(0.04) == "NEUTRAL" and A.classify(0.06) == "SAVED" and A.classify(-0.06) == "CUT_WINNER"
    assert A.sl_first({"close_reason": "SL", "gross_usd": 5.0}) == 1 and A.sl_first({"close_reason": "TP", "gross_usd": -1.0}) == 0
    assert A.sl_first({"close_reason": "FORCED_FLAT", "gross_usd": -0.1}) == 1 and A.sl_first({"close_reason": "FORCED_FLAT", "gross_usd": 0.1}) == 0


def test_replay_must_reproduce_the_run():
    m = {"sharpe_daily_sqrt365": 1.0, "net_profit_usd": 5.0, "max_drawdown_pct": 2.0, "trades": 3, "n_entries": 3, "n_close_TP": 1,
         "n_close_SL": 1, "n_close_FORCED_FLAT": 0, "n_close_AGENT_CLOSE": 1, "n_close_REVERSAL": 0}
    assert A.check_reproduces(m, dict(m), "x")["reproduces_run"]
    with pytest.raises(SystemExit, match="REPLAY_DOES_NOT_REPRODUCE_RUN"):
        A.check_reproduces(m, dict(m, n_close_AGENT_CLOSE=2), "x")
    with pytest.raises(SystemExit, match="REPLAY_DOES_NOT_REPRODUCE_RUN"):
        A.check_reproduces(m, dict(m, sharpe_daily_sqrt365=1.0 + 1e-6), "x")


def test_side_sharpe_of_an_absent_side_is_zero():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    tr = [{"direction": 1, "exit_time": pd.Timestamp("2020-01-02 10:00", tz="UTC"), "net_usd": 10.0}]
    assert A.side_sharpe(tr, -1, idx, 10_000.0) == 0.0 and A.side_sharpe(tr, 1, idx, 10_000.0) > 0


# ----------------------------------------------------------------------------- the pre-registered early-close selection rule
def test_require_early_close_excludes_never_closing_points_and_keeps_them_in_the_archive():
    from app.short_horizon.expert_guided import action_contract as ACT
    from tests.test_expert_guided_action_contract import comp, contract, table
    c = contract()
    c["strategy"]["optimizer"]["final"] = {"population": 16, "generations": 6}

    def score(view, s):                                     # the best Sharpe sits where the closer never fires
        never = s.close_threshold > 0.6
        return comp(sharpe=2.0 + (1.0 if never else 0.0) + 0.1 * s.open_threshold, se=0.4, dd=3.0 - s.open_threshold,
                    agent_closes=0 if never else 25)
    t = table(rows=6)
    off = ACT.freeze_strategy(t, [0, 1, 2], score, c, seed=5)
    assert off["chosen"]["agent_closes"] == 0 and "early_close_rule" not in off       # rule off: the v1 behaviour (installed)
    c["strategy"]["require_early_close"] = True
    on = ACT.freeze_strategy(t, [0, 1, 2], score, c, seed=5)
    assert on["chosen"]["agent_closes"] > 0 and on["strategy"]["close_threshold"] <= 0.6 and "early_close_rule" in on
    run = on["runs"]["without_veto"]
    assert run["degenerate_no_early_close_points"] > 0 and any(p["agent_closes"] == 0 for p in run["points"])   # archived, not selectable
    assert all(p["agent_closes"] > 0 for p in run["front_early_close"])
    c2 = contract()                                                                     # every point degenerate -> INVALID with the reason
    c2["strategy"]["optimizer"]["final"] = {"population": 8, "generations": 2}
    c2["strategy"]["require_early_close"] = True
    fr = ACT.freeze_strategy(t, [0, 1, 2], lambda view, s: comp(agent_closes=0), c2, seed=1)
    assert fr["strategy"] is None and fr["valid"] is False and fr["note"].startswith("DEGENERATE_NO_EARLY_CLOSE")


def test_next_stage_contract_v1_1_carries_the_rule_and_the_installed_v1_is_unchanged():
    import json
    from pathlib import Path
    from app.short_horizon.expert_guided import action_contract as ACT
    d = Path(__file__).resolve().parents[1] / "docs" / "short_horizon" / "clean_round" / "contracts"
    v1, v11 = json.loads((d / "ACTION_CONTRACT_v1.json").read_text()), json.loads((d / "ACTION_CONTRACT_v1_1.json").read_text())
    assert ACT.validate(v1) == [] and ACT.validate(v11) == []
    assert v1["identity"] == "action_490039ad1bf73dbd" and "require_early_close" not in v1["strategy"]
    assert v11["strategy"]["require_early_close"] is True and v11["derived_from"]["identity"] == v1["identity"]
    assert ACT.sha256_file(d / "ACTION_CONTRACT_v1.json") == v11["derived_from"]["sha256"]
