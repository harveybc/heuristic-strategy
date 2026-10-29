"""S5 (b): Musashi's separating fixture, NATIVE engine (quotes, costs, deadline, execution order), as a unit test of the
market-only closer D1 (BCHead + bc_from_curves, A = -BC, close when A >= -EPS_CLOSE): quote 100, path 98 -> 104;
A (SL 99 / TP 106) -> CLOSE, B (SL 97 / TP 104) -> HOLD; the mirrored pair (short on the negated path); (b2) A with
realistic costs stays CLOSE; (b3) a planted A' with SL 99.95 flips to HOLD exactly when the market-close cost exceeds
the stop-exit cost by more than d_SL - EPS_CLOSE, the threshold computed from the engine's own cost convention and
asserted on both sides. S5 (a): the fixed-delay selection on synthetic episode paths."""
import dataclasses
import types

import numpy as np
import pandas as pd

from app.short_horizon import execution as X
from app.short_horizon.expert_guided import env as E
from tools import strategy_behaviour_matrix as BM
from tools import strategy_output_simulation as S
from tests.test_exec_variants import synth


def _planted(mirror=False, prof=None):
    """One synthetic week; at the decision bar the close is 100, the next open 100, then 6 bars down to 98, 8 bars up
    to 104, then flat; sigma = 1 price unit so sigma distances are price distances."""
    bars, cal, _, _ = synth(weeks=1)
    n = len(bars)
    c = np.full(n, 100.0)
    b = 40                                                             # inside the session, after the operational start
    down = np.linspace(100.0, 98.0, 7)[1:]
    up = np.linspace(98.0, 104.0, 9)[1:]
    path = np.r_[down, up]
    c[b + 1: b + 1 + len(path)] = path
    c[b + 1 + len(path):] = 104.0
    o = np.r_[c[0], c[:-1]]
    o[b + 1] = 100.0
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    if mirror:
        o, c, hi, lo = 200.0 - o, 200.0 - c, 200.0 - lo, 200.0 - hi
    bars = pd.DataFrame({"OPEN": o, "HIGH": hi, "LOW": lo, "CLOSE": c}, index=bars.index)
    sig = np.full(n, 1.0)
    cfg = E.EnvConfig(level_policy="clip", profile=prof if prof is not None else E.executable_profile())
    env = E.BarEnv(bars, cal, sig, np.full(n, 10.0), bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)
    cx = types.SimpleNamespace(rob=np.arange(n), obs_ok=np.ones(n, bool))
    return env, cx, b


def _advantage(env, cx, b, side, d_tp, d_sl):
    head = S.BCHead(env, cx)
    cv = head.row(b, side)[4]
    bc = S.bc_from_curves(cv, d_tp, d_sl)
    return -float(bc)


def test_separating_fixture_native_engine_and_its_mirror():
    for mirror, side in ((False, 0), (True, 1)):
        env, cx, b = _planted(mirror=mirror)
        a_A = _advantage(env, cx, b, side, 6.0, 1.0)                  # A: SL 99 / TP 106 -> the SL is hit on the way to 98; the best later close (next open 99.67) still loses: closing now wins
        a_B = _advantage(env, cx, b, side, 4.0, 3.0)                  # B: SL 97 / TP 104 -> the TP at 104 is reached: holding beats closing now
        assert a_A >= -BM.EPS_CLOSE and 0.3 < a_A < 0.4, (mirror, a_A)   # A = -(best continuation) = +0.333 (the discretionary close at 99.67 beats the SL at 99)
        assert a_B < -BM.EPS_CLOSE and a_B < -3.5, (mirror, a_B)
    # the mirror is EXACT under symmetric financing (the head's curves are float32 channels: 1e-6); with the executable
    # profile's asymmetric swaps (-7 / +2.5 per lot per rollover) the pair differs by that declared business asymmetry only
    sym = dataclasses.replace(E.executable_profile(), swap_long_per_lot=0.0, swap_short_per_lot=0.0)
    env, cx, b = _planted(prof=sym)
    envm, cxm, _ = _planted(mirror=True, prof=sym)
    for d_tp, d_sl in ((6.0, 1.0), (4.0, 3.0)):
        assert abs(_advantage(env, cx, b, 0, d_tp, d_sl) - _advantage(envm, cxm, b, 1, d_tp, d_sl)) < 1e-6
    env, cx, b = _planted()
    envm, cxm, _ = _planted(mirror=True)
    gap = abs(_advantage(env, cx, b, 0, 4.0, 3.0) - _advantage(envm, cxm, b, 1, 4.0, 3.0))
    prof = E.executable_profile()
    assert 0 < gap <= 2 * (abs(prof.swap_long_per_lot) + abs(prof.swap_short_per_lot)) / prof.lot_units + 1e-6


def test_fixture_A_with_realistic_costs_stays_close_and_the_planted_A_prime_flips_at_the_engine_threshold():
    env, cx, b = _planted(prof=E.executable_profile())
    assert _advantage(env, cx, b, 0, 6.0, 1.0) >= -BM.EPS_CLOSE       # (b2) realistic costs: still CLOSE
    # (b3) A': SL at 99.95 (d_SL = 0.05), hit on the first bar. Closing now exits at the bid and pays the market-close
    # cost; holding exits at the stop and pays the stop-exit cost; spread and commission are paid either way, so only the
    # slippage difference decides: BC = -d_SL + (cost_market - cost_stop). The closer HOLDS iff A = -BC < -EPS_CLOSE, i.e.
    # iff cost_market - cost_stop > d_SL + EPS_CLOSE: the sealed tie band EPS_CLOSE = 0.02 sigma makes the engine's
    # threshold 0.07, not the bare 0.05 (a hold must beat the close by more than EPS to be taken). Both sides asserted.
    d_sl = 0.05
    threshold = d_sl + BM.EPS_CLOSE
    base = E.executable_profile()
    for delta, want_hold in ((threshold + 0.01, True), (threshold - 0.01, False), (0.06, False), (0.10, True), (0.0, False)):
        prof = dataclasses.replace(base, slippage_market=base.slippage_stop + delta)   # market-close cost exceeds the stop-exit cost by delta
        env, cx, b = _planted(prof=prof)
        a = _advantage(env, cx, b, 0, 6.0, d_sl)
        assert (a < -BM.EPS_CLOSE) == want_hold, (delta, a)
        envm, cxm, _ = _planted(mirror=True, prof=prof)
        assert (_advantage(envm, cxm, b, 1, 6.0, d_sl) < -BM.EPS_CLOSE) == want_hold, ("mirror", delta)


def test_fixed_delay_selection_on_calibration_paths():
    # three episodes: the optimum exit is at tick 2 (then the path decays to the stop); passive hold loses; immediate close gives 0
    paths = [{"P": np.array([0.0, 2.0, 5.0, 3.0, 1.0, -1.0]), "hold_net": -10.0, "risk": 10.0},
             {"P": np.array([1.0, 1.0, 4.0, 2.0]), "hold_net": -10.0, "risk": 10.0},
             {"P": np.array([0.0, 3.0]), "hold_net": 6.0, "risk": 10.0}]   # terminated natively (TP) before tick 2: keeps its own result
    sel = BM.select_fixed_delay(paths, menu=(0, 1, 2, 3, None))
    reg = sel["regrets"]
    assert sel["delay"] == 2 and abs(reg[2] - ((5 - 5) + (4 - 4) + (6 - 6)) / 3 / 10) < 1e-12
    assert reg["hold"] > reg[2] and reg[0] > reg[2] and sel["episodes"] == 3
    only_hold = BM.select_fixed_delay([{"P": np.array([-1.0, -2.0]), "hold_net": 5.0, "risk": 10.0}], menu=(0, 1, None))
    assert only_hold["delay"] is None                                 # passive hold wins when every early close loses
    Pol = BM.fixed_delay_policy(2)
    assert Pol.delay == 2 and issubclass(Pol, BM.ZZ.PolicyC3Z)
