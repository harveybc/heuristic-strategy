"""Multi-resolution ZigZag labels scored against the optimum (owner 2026-10-24): fitness signs and bounds, counter-leg
negativity, NOP hard / graded, the closer rule, availability / censoring, the demonstration policy acting only on
available labels, the thresholds JSON round-trip and the E3-M wiring. Synthetic bars and pivots; no data, no fit."""
import json

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import multires as MR

H = 3_600_000_000_000
# a 12-bar path: valley at 2 (0.9988), peak at 6 (1.0032), valley at 10 (0.9968); closes rise 2 -> 6 then fall 6 -> 10
C = np.array([1.0000, 0.9995, 0.9990, 1.0000, 1.0010, 1.0020, 1.0030, 1.0020, 1.0005, 0.9990, 0.9970, 0.9980])
O = np.r_[C[0], C[:-1]]
HI, LO = C + 0.0002, C - 0.0002
SIGMA = np.full(12, 0.0010)
PIV = [(2, Z.VALLEY, 0.9988), (6, Z.PEAK, 1.0032), (10, Z.VALLEY, 0.9968)]
CONF = [4, 8, 11]
V = pd.date_range("2021-01-04", periods=12, freq="h", tz="UTC").as_unit("ns").asi8


def tables(period_end_bar=11, resolutions=None, conf=CONF):
    """One period ending at bar start + 1 h of ``period_end_bar``; every bar is its own row."""
    res = resolutions if resolutions is not None else {"3": (PIV, conf)}
    row_end = np.full(12, V[period_end_bar] + H, np.int64)
    return MR.Tables.build(O, HI, LO, C, SIGMA, PIV, conf, res, V, np.arange(12), row_end)


def test_open_fitness_is_one_at_the_optimal_open_late_entries_lower_and_counter_leg_negative():
    L = MR.leg_tables(O, HI, LO, C, SIGMA, PIV)
    leg = 1.0032 - 0.9988
    # decision at the valley bar 2, fill at o[3] = c[2] = 0.9990: almost the whole leg remains
    assert np.isclose(L["f_open"][1][2], (1.0032 - 0.9990) / leg)
    assert L["f_open"][1][2] > L["f_open"][1][4] > L["f_open"][1][5] > 0          # later entries capture less
    assert np.all(np.abs(np.nan_to_num(L["f_open"][1])) <= 1.0 + 1e-12)
    # a SHORT inside the rising leg [2, 6] opens against the optimum: negative, = -(adverse move to the peak)/leg
    assert np.isclose(L["f_open"][-1][3], -(1.0032 - O[4]) / leg) and L["f_open"][-1][3] < 0
    # the leg in sigma and the adverse excursion of the long entry at o[3] until the peak (lows never below the entry)
    assert np.isclose(L["leg_sigma"][2], leg / 0.0010)
    assert np.isclose(L["mae"][1][2], (O[3] - LO[3:7].min()) / leg) and np.isnan(L["mae"][-1][2])
    assert L["leg_end"][2] == 1 and L["leg_end"][7] == 2 and L["leg_end"][11] == -1


def test_close_fitness_relative_to_the_optimal_open_before_and_after_the_optimal_close():
    L = MR.leg_tables(O, HI, LO, C, SIGMA, PIV)
    leg = 1.0032 - 0.9988
    f, rem = L["f_close"][1], L["remaining"][1]
    assert np.isclose(f[6], (1.0030 - 0.9988) / leg) and f[6] < 1.0           # the bar close never reaches the intrabar extreme
    assert f[4] < f[6] and np.isclose(f[4] + rem[4], 1.0)                      # before the close: gain + remaining = the leg
    assert f[8] < f[6] and rem[8] < 0                                          # after the optimal close the remaining gain is negative
    assert L["close_end"][1][4] == 1 and L["close_end"][1][8] == 1             # both legs refer to the peak at index 1
    # a short held at bar 8 is measured from the peak (its optimal open) towards the valley at 10
    assert np.isclose(L["f_close"][-1][8], (1.0032 - 1.0005) / (1.0032 - 0.9968))


def test_labels_nop_hard_graded_negative_and_open_with_tp_in_sigma():
    tb = tables()
    th = MR.Thresholds()
    assert MR.origin_label(tb, 4, th)[3] == "nop_hard"                        # no resolution votes at bar 4
    cls, w, tp, kind, f = MR.origin_label(tb, 2, th)
    assert (cls, kind) == (MR.OPEN_L, "open") and np.isclose(w, f) and np.isclose(tp, f * tb.legs["leg_sigma"][2])
    # a tiny leg/sigma threshold violation -> graded NOP with weight 1 - f
    cls, w, _, kind, f = MR.origin_label(tb, 2, MR.Thresholds(theta_vol=10.0))
    assert (cls, kind) == (MR.NOP, "nop_graded") and np.isclose(w, 1 - f)
    assert MR.origin_label(tb, 2, MR.Thresholds(theta_f=0.999))[3] == "nop_graded"
    assert MR.origin_label(tb, 2, MR.Thresholds(theta_mae=-1.0))[3] == "nop_graded"
    # a finer resolution that calls SHORT at bar 3 (inside the rising optimum leg) is a negative example
    fine = [(3, Z.PEAK, 1.0002), (5, Z.VALLEY, 1.0018)]
    tb2 = tables(resolutions={"3": (PIV, CONF), "fine": (fine, [4, 6])})
    cls, w, _, kind, f = MR.origin_label(tb2, 3, th)
    assert (cls, kind) == (MR.NOP, "negative") and f < 0 and np.isclose(w, min(1.0, -f))
    # both sides voted at one bar -> conflict, graded
    both = [(2, Z.PEAK, 1.0), (4, Z.VALLEY, 0.99)]
    tb3 = tables(resolutions={"3": (PIV, CONF), "x": (both, [3, 5])})
    assert MR.origin_label(tb3, 2, th)[3] == "conflict"


def test_closer_rule_remaining_or_taken_fitness():
    tb = tables()
    th = MR.Thresholds()
    assert MR.close_label(tb, 4, 1, th)[0] == E.HOLD                           # mid-leg: plenty remaining, f < 0.9
    assert MR.close_label(tb, 6, 1, th)[0] == E.CLOSE                          # at the peak bar: f_close >= 0.9
    assert MR.close_label(tb, 8, 1, th)[0] == E.CLOSE                          # after the peak: remaining < 0
    assert MR.close_label(tb, 4, 1, MR.Thresholds(theta_close=0.9))[0] == E.CLOSE
    assert MR.close_label(tb, 4, 1, MR.Thresholds(f_close_take=0.1))[0] == E.CLOSE


def test_availability_waits_for_every_confirmation_and_censors_otherwise():
    # the leg [2, 6] is final when the peak is confirmed (bar 8): a period ending at bar 7 censors bar 2
    assert tables(period_end_bar=8).open_available(2, 2) and not tables(period_end_bar=7).open_available(2, 2)
    # a slower ladder member whose label support at bar 2 lies at bar 10 delays availability to 10
    slow = [(2, Z.VALLEY, 0.9988), (6, Z.PEAK, 1.0032)]
    tb = tables(period_end_bar=9, resolutions={"3": (PIV, CONF), "slow": (slow, [10, 11])})
    assert not tb.open_available(2, 2) and tables(period_end_bar=10, resolutions={"3": (PIV, CONF), "slow": (slow, [10, 11])}).open_available(2, 2)
    # a censored pivot (conf -1) censors every bar depending on it; closers need only the leg end
    assert not tables(conf=[4, -1, 11]).open_available(2, 2) and not tables(conf=[4, -1, 11]).close_available(4, 2, 1)
    assert tables(period_end_bar=8).close_available(4, 4, 1) and not tables(period_end_bar=7).close_available(4, 4, 1)
    assert not tables().open_available(11, 11)                                 # no leg end after the last pivot


def test_demonstration_policy_acts_only_on_available_labels():
    th = MR.Thresholds()
    flat = np.zeros(9, np.float32); flat[-1] = 1.0
    gated = np.zeros(9, np.float32)
    long_ = np.zeros(9, np.float32); long_[0] = 1.0
    pol = MR.demonstration_policy(tables(period_end_bar=11), th)
    a, tp, sl = pol(2, 2, flat, None)
    assert a == E.LONG and sl == MR.SL_DEMO and MR.TP_BOUNDS[0] <= tp <= MR.TP_BOUNDS[1]
    assert pol(2, 2, gated, None)[0] == E.HOLD                                 # entry gate closed
    assert pol(4, 4, flat, None)[0] == E.HOLD                                  # NOP hard
    assert pol(6, 6, long_, None)[0] == E.CLOSE and pol(4, 4, long_, None)[0] == E.HOLD
    censored = MR.demonstration_policy(tables(period_end_bar=7), th)
    assert censored(2, 2, flat, None)[0] == E.HOLD and censored(6, 6, long_, None)[0] == E.HOLD


def test_thresholds_json_round_trip_and_unknown_field_refused():
    th = MR.Thresholds(theta_f=0.6, min_votes=2)
    assert MR.Thresholds.from_json(th.to_json()) == th
    assert MR.Thresholds.from_json("{}") == MR.Thresholds()
    with pytest.raises(ValueError, match="MULTIRES_THRESHOLDS_UNKNOWN"):
        MR.Thresholds.from_json(json.dumps({"theta_open": 0.5}))


def test_e3m_wiring_declares_multires_without_changing_the_defaults():
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as CR
    from tools import expert_guided_e3m as M
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c"])
    assert a.labels == "zigzag" and a.multires_thresholds == ""
    a2 = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--labels", "multires"])
    assert a2.labels == "multires" and "multires" in M.DROP_CENSORED_MODES and "both" not in M.DROP_CENSORED_MODES
    assert "multires" in CR.LABEL_MODES and CR.label_mode_of({"labels": {"mode": "multires"}})[1] == CR.DECLARED
    assert not CR.labels_ok("multires", CR.DECLARED)                          # the T3 decision still demands the inherited ZigZag
