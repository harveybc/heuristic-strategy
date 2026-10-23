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
    sig, H = S.hindsight_signals(o, h, l, c, sigma, piv, flat_bar=np.full(len(c), 9), p=p)
    # decision after bar 0, entry at o[1] = c[0] = 1.0000; long horizon = next PEAK = bar 4, short = next VALLEY = bar 2
    assert H["open"][1][0] == 4 and H["open"][-1][0] == 2 and H["close"][0] == 2
    assert np.isclose(sig["opportunity"][1][0], (1.0032 - 1.0000) / 0.001)        # high of bar 4
    assert np.isclose(sig["adverse_to_peak"][1][0], (1.0000 - 0.9988) / 0.001)    # low of bar 2 before the peak
    assert sig["bars_to_peak"][1][0] == 4
    assert np.isclose(sig["opportunity"][-1][0], (1.0000 - 0.9988) / 0.001)       # short: best is the dip at bar 2
    assert np.isclose(sig["adverse_to_peak"][-1][0], (1.0007 - 1.0000) / 0.001)   # high of bar 1 before that dip
    # closer horizon from bar 0 = first pivot of any kind = the valley at bar 2 (adverse for a long)
    assert np.isclose(sig["hold_gain"][1][0], (1.0007 - 1.0000) / 0.001) and np.isclose(sig["hold_adverse"][1][0], (1.0000 - 0.9988) / 0.001)
    assert sig["next_is_favourable"][1][0] == 0 and sig["next_is_favourable"][-1][0] == 1
    # decision after bar 2 (the valley): next pivot is the peak at 4 -> a long still has the optimum ahead
    assert sig["next_is_favourable"][1][2] == 1 and np.isclose(sig["hold_gain"][1][2], (1.0032 - 0.9990) / 0.001)
    # decision after bar 4 (the peak): next pivot is the valley at 7 -> a long must close
    assert sig["next_is_favourable"][1][4] == 0 and sig["hold_gain"][1][4] == 0.0
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
    assert cl[4] == 1 and cl[2] == 0 and cl[0] == 1   # at the peak: close; after the valley: hold; before the valley: close (next pivot adverse)
    # the SL rule: a short at bar 0 has its optimum ahead (valley at 2) but the high of bar 1 is 0.7 sigma against it:
    # with a 2-sigma SL it holds, with a 0.5-sigma SL the stop would be hit first -> close
    assert S.closer_labels(sig, -1, np.full(len(c), 2.0), p)[0] == 0
    assert S.closer_labels(sig, -1, np.full(len(c), 0.5), p)[0] == 1


def test_forced_flat_caps_the_horizon():
    o, h, l, c, sigma = _path()
    sig, H = S.hindsight_signals(o, h, l, c, sigma, [(7, 1, 0.9972)], flat_bar=np.full(len(c), 3))
    assert H["open"][1][0] == 3 and np.isclose(sig["opportunity"][1][0], (1.0022 - 1.0000) / 0.001 - 0.06)
