from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided.data import calendar_from_closures
from app.short_horizon.expert_guided.env import BarEnv, EnvConfig, HOLD, LONG, SHORT, CLOSE
from app.short_horizon.execution import RealisticProfile, Sizing


def make(action=LONG, sigma=.001, profile=None, cost_mode=None):
    ts = pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
    bars = pd.DataFrame(dict(OPEN=1.1, HIGH=1.10001, LOW=1.09999, CLOSE=1.1), index=ts)
    cal = calendar_from_closures([['2020-01-03T22:00Z', '2020-01-05T22:00Z'],
                                 ['2020-01-10T22:00Z', '2020-01-12T22:00Z']])
    cfg = EnvConfig(sizing=Sizing(leverage=50), profile=profile or RealisticProfile())
    if cost_mode is not None:
        cfg = replace(cfg, cost_mode=cost_mode)
    e = BarEnv(bars, cal, np.full(120, sigma), np.full(120, sigma * 2), ts[0],
               ts[-1] + pd.Timedelta(hours=1), cfg)
    assert e.step(12, action)['entry']
    return e


def set_bar(e, k, o, high=None, low=None, close=None):
    for name, value in [('o', o), ('h', high if high is not None else o),
                        ('l', low if low is not None else o), ('c', close if close is not None else o)]:
        values = getattr(e, name).copy()
        values[k] = value
        setattr(e, name, values)
    e.bars.iloc[k] = [e.o[k], e.h[k], e.l[k], e.c[k]]


@pytest.mark.parametrize('action', [LONG, SHORT])
@pytest.mark.parametrize('last,multiplier', [(25, 1), (73, 5)])
def test_open_financing_and_final_reconciliation(action, last, multiplier):
    e = make(action)
    p = e.pos
    rate = e.cfg.profile.swap_long_per_lot if action == LONG else e.cfg.profile.swap_short_per_lot
    due = -rate * p.units / e.cfg.profile.lot_units * multiplier
    for b in range(13, last):
        e.step(b, HOLD)
    expected = e.cfg.capital - sum(p.entry_cost.values()) - due
    assert e.equity_at(last) == pytest.approx(expected)
    assert e.equity_at(last) == pytest.approx(expected)  # read is idempotent
    e._close(last, e.c[last], False, 'AGENT_CLOSE')
    t = e.trades[0]
    assert t['financing_usd'] == pytest.approx(due)
    assert sum(x['amount_usd'] for x in t['financing_events']) == pytest.approx(due)
    result = e.result()
    assert result['end_equity_usd'] == pytest.approx(e.balance)
    assert result['equity'].loc[e.bars.index[last] + pd.Timedelta(hours=1)] == pytest.approx(e.balance)
    roll = e.bars.index[24]
    assert result['equity'].loc[roll] == pytest.approx(
        e.cfg.capital - sum(p.entry_cost.values()) + rate * p.units / e.cfg.profile.lot_units)


@pytest.mark.parametrize('at_open', [False, True])
def test_known_fill_spread_uses_event_time(at_open):
    e = make()
    e._close(22, e.c[22], at_open, 'AGENT_CLOSE')
    t = e.trades[0]
    prof = e.cfg.profile
    expected = t['units'] / 2 * (prof.spread_at(t['entry_time'].value, e.mo[0]) +
                                prof.spread_at(t['exit_time'].value, e.mo[0]))
    assert t['spread_usd'] == pytest.approx(expected)
    assert t['exit_clock'] == ('OPEN' if at_open else 'CLOSE')


def test_intrabar_clock_is_declared_not_exact():
    e = make()
    e._close(22, e.pos.sl, False, 'SL')
    assert e.trades[0]['exit_clock'] == 'INTRABAR_END_PROXY_COST_AT_OPEN'


@pytest.mark.parametrize('action', [LONG, SHORT])
@pytest.mark.parametrize('gap', [True, False])
def test_margin_liquidates_before_wide_stop(action, gap):
    e = make(action, sigma=.2)
    p = e.pos
    margin = p.units * p.entry_price / e.cfg.sizing.leverage * e.cfg.sizing.margin_closeout
    level = p.entry_price + p.direction * (margin - e.balance + sum(p.entry_cost.values())) / p.units
    worse = level - p.direction * .001
    assert min(p.sl, p.tp) < worse < max(p.sl, p.tp)
    set_bar(e, 14, worse if gap else p.entry_price,
            high=max(p.entry_price, worse), low=min(p.entry_price, worse), close=worse)
    e.step(13, HOLD)
    assert e.pos is None
    t = e.trades[0]
    assert t['close_reason'] == 'MARGIN_CLOSEOUT'
    assert t['exit_price'] == pytest.approx(worse if gap else level)
    assert e.counts['close_MARGIN_CLOSEOUT'] == 1


def test_nearer_stop_wins_over_margin():
    e = make()
    p = e.pos
    set_bar(e, 14, p.entry_price, low=.7)
    e.step(13, HOLD)
    assert e.trades[0]['close_reason'] == 'SL'


def test_open_protection_precedes_discretionary_close():
    e = make()
    set_bar(e, 14, e.pos.sl - .001)
    e.step(13, CLOSE)
    assert e.trades[0]['close_reason'] == 'SL'


@pytest.mark.parametrize('action', [LONG, SHORT])
def test_gap_fill_rejects_minimum_distance_without_moving_levels(action):
    e = make(action)
    e.reset()
    direction = 1 if action == LONG else -1
    tp, sl, _ = e._levels(12, direction, 1.5, 2.)
    set_bar(e, 13, tp - direction * 1e-6)
    info = e.step(12, action)
    assert not info['entry'] and e.pos is None
    assert info['refused'] == 'FILL_DISTANCE'
    assert not e.trades


def test_equity_is_close_timestamped():
    e = make()
    m = e.result()
    assert m['equity'].index.equals(e.bars.index + pd.Timedelta(hours=1))
    pd.testing.assert_series_equal(m['daily_equity'],
        m['equity'].groupby(m['equity'].index.normalize()).last(), check_freq=False)


@pytest.mark.parametrize('action', [LONG, SHORT])
def test_zero_cost_keeps_execution_contract(action):
    e = make(action, cost_mode='zero_cost')
    p = e.pos
    for b in range(13, 26):
        e.step(b, HOLD)
    e.step(26, CLOSE)
    t = e.trades[0]
    for key in ['spread_usd', 'slippage_usd', 'commission_usd', 'financing_usd']:
        assert t[key] == 0
    assert t['net_usd'] == t['gross_usd'] == 0
    assert e.cfg.profile.limit_through == RealisticProfile().limit_through
    assert e.cfg.profile.entry_delay_bars == RealisticProfile().entry_delay_bars
    control = make(action)
    assert (p.units, p.sl, p.tp, p.entry_price) == (
        control.pos.units, control.pos.sl, control.pos.tp, control.pos.entry_price)
    m = e.result()
    assert m['cost_mode'] == 'zero_cost'
    assert m['accounting_version'] == 'event_costs_v2'


def test_unknown_cost_mode_rejected():
    with pytest.raises(ValueError, match='COST_MODE'):
        make(cost_mode='free_but_unspecified')


def test_rollover_can_trigger_close_time_margin_liquidation():
    e = make(profile=replace(RealisticProfile(),swap_long_per_lot=-21_000.))
    for b in range(13, 22):
        e.step(b,HOLD)
    assert e.pos is not None
    e.step(22,HOLD)
    assert e.pos is None
    assert e.trades[0]['close_reason']=='MARGIN_CLOSEOUT'
    assert e.trades[0]['exit_clock']=='CLOSE'


def test_gap_liquidation_does_not_reverse_in_same_bar():
    e = make()
    set_bar(e,14,e.pos.sl-.001)
    e.step(13,SHORT)
    assert e.pos is None and e.counts['entries']==1


def test_financing_tampering_is_rejected():
    e = make()
    e._close(25,e.c[25],False,'AGENT_CLOSE')
    e.trades[0]['financing_events'][0]['amount_usd']+=1
    with pytest.raises(ValueError,match='FINANCING_EVENTS_MISMATCH'):
        e.result()
