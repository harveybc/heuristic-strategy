"""Hindsight per-agent signals: exact values on a hand-made path, horizon purge, label rules."""
import numpy as np

from app.short_horizon.expert_guided import signals as S


def _path():
    # closes: 1.0000 .. a rise to 1.0030 at bar 4, dip to 0.9990 at bar 2, then fall to 0.9970 at bar 7
    c = np.array([1.0000, 1.0005, 0.9990, 1.0020, 1.0030, 1.0010, 0.9990, 0.9970, 0.9980, 0.9985])
    o = np.r_[c[0], c[:-1]]                    # open = previous close
    h, l = c + 0.0002, c - 0.0002
    sigma = np.full(len(c), 0.0010)            # 10 pips
    return o, h, l, c, sigma


def test_opportunity_adverse_and_hold_values_on_a_known_path():
    o, h, l, c, sigma = _path()
    p = S.SignalParams(max_bars=72, cost_sigma=0.0)
    piv = [(2, -1, 0.9988), (4, 1, 1.0032), (7, -1, 0.9968)]        # valley, peak, valley
    sig, H = S.hindsight_signals(o, h, l, c, sigma, piv, flat_bar=np.full(len(c), 50), p=p)   # no forced flat inside the data
    # decision after bar 0, entry at o[1] = c[0] = 1.0000; long horizon = next PEAK = bar 4, short = next VALLEY = bar 2
    assert H[1]["bar"][0] == 4 and H[-1]["bar"][0] == 2 and H[1]["reason"][0] == "pivot" and sig["truncated"][1][0] == 0
    assert np.isclose(sig["opportunity"][1][0], (1.0032 - 1.0000) / 0.001)        # high of bar 4
    assert np.isclose(sig["adverse_to_peak"][1][0], (1.0000 - 0.9988) / 0.001)    # low of bar 2 before the peak
    assert sig["bars_to_peak"][1][0] == 4
    assert np.isclose(sig["opportunity"][-1][0], (1.0000 - 0.9988) / 0.001)       # short: best is the dip at bar 2
    assert np.isclose(sig["adverse_to_peak"][-1][0], (1.0007 - 1.0000) / 0.001)   # high of bar 1 before that dip
    # the closer of a long at bar 0 shares the opener's horizon (peak at 4): further gain 3.2 sigma, dip 1.2 before it
    assert np.isclose(sig["hold_gain"][1][0], (1.0032 - 1.0000) / 0.001) and np.isclose(sig["hold_adverse"][1][0], (1.0000 - 0.9988) / 0.001)
    # decision after bar 4 (the peak): the next peak lies beyond the data -> horizon truncated at the end, nothing to gain
    assert H[1]["reason"][4] == "end" and sig["truncated"][1][4] == 1 and sig["hold_gain"][1][4] == 0.0
    assert np.isnan(sig["opportunity"][1][len(c) - 1])                            # no horizon at the end


def test_labels_follow_the_declared_rules_and_purge():
    o, h, l, c, sigma = _path()
    p = S.SignalParams(max_bars=72, cost_sigma=0.0, theta_open=1.0, sl_max=2.0, theta_close=0.5)
    piv = [(2, -1, 0.9988), (4, 1, 1.0032), (7, -1, 0.9968)]
    sig, H = S.hindsight_signals(o, h, l, c, sigma, piv, flat_bar=np.full(len(c), 9), p=p)
    ol, os_ = S.opener_labels(sig, 1, p), S.opener_labels(sig, -1, p)
    assert ol[0] == 1 and os_[0] == 1          # long reaches 3.2 sigma by the peak; short 1.2 by the valley
    nop = S.nop_labels(sig, p)
    assert nop[0] == 0 and np.isnan(nop[len(c) - 1])
    cl = S.closer_labels(sig, 1, np.full(len(c), 2.0), p)
    assert cl[4] == 1 and cl[2] == 0 and cl[0] == 0   # at the peak: close; after and before the valley: hold (the peak lies ahead, dip survivable)
    assert ol[0] == 1 and cl[0] == 0                   # opener and closer of the same side agree at the same bar
    # the SL rule: with a 0.5-sigma SL the 1.2-sigma dip before the peak would stop the long first -> close
    assert S.closer_labels(sig, 1, np.full(len(c), 0.5), p)[0] == 1
    # the kind of a pivot beyond the window never matters: moving the far valley to a peak leaves bar 0 untouched
    sig2, _ = S.hindsight_signals(o, h, l, c, sigma, [(2, -1, 0.9988), (4, 1, 1.0032), (7, 1, 0.9968)], flat_bar=np.full(len(c), 9), p=p)
    assert S.closer_labels(sig2, 1, np.full(len(c), 2.0), p)[0] == cl[0] and np.isclose(sig2["hold_gain"][1][0], sig["hold_gain"][1][0])


def test_forced_flat_caps_the_horizon():
    o, h, l, c, sigma = _path()
    sig, H = S.hindsight_signals(o, h, l, c, sigma, [(7, 1, 0.9972)], flat_bar=np.full(len(c), 3))
    assert H[1]["bar"][0] == 3 and H[1]["reason"][0] == "flat" and sig["truncated"][1][0] == 1
    assert np.isclose(sig["opportunity"][1][0], (1.0022 - 1.0000) / 0.001 - 0.06)
    # auditor's witness 2: with the window truncated at bar 3, changing the kind of the pivot at bar 7 changes nothing
    sig2, _ = S.hindsight_signals(o, h, l, c, sigma, [(7, -1, 0.9972)], flat_bar=np.full(len(c), 3))
    for k in ("opportunity", "hold_gain", "hold_adverse"):
        assert np.isclose(sig2[k][1][0], sig[k][1][0])
    assert S.closer_labels(sig2, 1, np.full(len(c), 2.0))[0] == S.closer_labels(sig, 1, np.full(len(c), 2.0))[0]


def test_optimum_closer_rule_closes_when_an_adverse_pivot_lies_inside_the_window_and_is_censored_otherwise():
    o, h, l, c, sigma = _path()
    p = S.SignalParams(max_bars=72, cost_sigma=0.0, theta_close=0.05)
    piv = [(2, -1, 0.9988), (4, 1, 1.0032), (7, -1, 0.9968)]
    sig, H = S.hindsight_signals(o, h, l, c, sigma, piv, flat_bar=np.full(len(c), 50), p=p)
    sl = np.full(len(c), 2.0)
    hold, opt = S.closer_labels(sig, 1, sl, p, "hold_through"), S.closer_labels(sig, 1, sl, p, "optimum")
    assert hold[0] == 0 and opt[0] == 1        # bar 0: the valley at 2 (adverse) lies inside the long's window -> optimum closes
    assert hold[2] == 0 and opt[2] == 0        # bar 2: next pivot inside is the peak (favourable) -> both hold
    assert sig["next_adverse_inside"][1][0] == 1 and sig["next_adverse_inside"][1][2] == 0
    # window truncated before any pivot: the optimum rule is censored (NaN), hold_through still decides
    sig2, _ = S.hindsight_signals(o, h, l, c, sigma, [(7, 1, 0.9972)], flat_bar=np.full(len(c), 3), p=p)
    assert np.isnan(S.closer_labels(sig2, 1, sl, p, "optimum")[0]) and np.isfinite(S.closer_labels(sig2, 1, sl, p)[0])
    import pytest
    with pytest.raises(ValueError, match="CLOSER_RULE_UNKNOWN"):
        S.closer_labels(sig, 1, sl, p, "bogus")


def test_one_bar_horizon_is_a_valid_window_and_end_of_data_has_none():
    """Auditor's edge cases (2026-10-22): a horizon of exactly one bar (the favourable pivot or the cap right after the
    decision bar) is defined over that single bar; the last bars, with no bar after them, carry no signal."""
    o, h, l, c, sigma = _path()
    p = S.SignalParams(max_bars=72, cost_sigma=0.0)
    sig, H = S.hindsight_signals(o, h, l, c, sigma, [(1, 1, 1.0007), (2, -1, 0.9988)], flat_bar=np.full(len(c), 50), p=p)
    assert H[1]["bar"][0] == 1 and H[1]["reason"][0] == "pivot"                       # the peak is the very next bar
    assert np.isclose(sig["opportunity"][1][0], (1.0007 - 1.0000) / 0.001) and sig["bars_to_peak"][1][0] == 1
    assert np.isclose(sig["adverse_to_peak"][1][0], max(1.0000 - 1.0003, 0.0) / 0.001)
    sig2, H2 = S.hindsight_signals(o, h, l, c, sigma, [(7, 1, 0.9972)], flat_bar=np.full(len(c), 50), p=S.SignalParams(max_bars=1, cost_sigma=0.0))
    assert H2[1]["reason"][0] == "cap" and sig2["truncated"][1][0] == 1 and np.isclose(sig2["opportunity"][1][0], (1.0007 - 1.0000) / 0.001)
    assert np.isnan(sig2["next_adverse_inside"][1][0])                                  # no pivot inside a 1-bar window
    assert np.isnan(sig["opportunity"][1][len(c) - 1]) and np.isnan(sig["opportunity"][1][len(c) - 2])
