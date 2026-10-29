from dataclasses import replace, fields

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv, AbsoluteOrder, FillContext
from app.short_horizon.expert_guided.env import EnvConfig, HOLD, LONG, SHORT, CLOSE
from app.short_horizon.expert_guided.data import calendar_from_closures
from app.short_horizon.execution import Sizing, RealisticProfile


def make():
    ts = pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
    bars = pd.DataFrame(dict(OPEN=1.1, HIGH=1.10001, LOW=1.09999, CLOSE=1.1), index=ts)
    cal = calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                                ['2020-01-10T22:00Z','2020-01-12T22:00Z']], cutoff=0)
    p = replace(RealisticProfile(), spread_schedule=((0,24,.0002),), weekly_open_spread=.0002)
    cfg = EnvConfig(sizing=Sizing(leverage=50), profile=p,
                    execution_price_mode='bid_ask_v3', max_entries_per_day=None)
    return AbsoluteBarEnv(bars, cal, np.full(120,.001), np.full(120,.002),
                          ts[0], ts[-1]+pd.Timedelta(hours=1), cfg, tick=.00001)


def advance(e, until):
    while e.next_decision < until:
        e.step(e.next_decision, HOLD)


def plan(c):
    return AbsoluteOrder(tp=1.1+c.direction*.01, sl=1.1-c.direction*.01,
                         q=.2, risk_budget=200., funding_reserve_per_unit=0.)


@pytest.mark.parametrize('action,d', [(LONG,1), (SHORT,-1)])
def test_absolute_levels_survive_and_close_costs_reconcile(action,d):
    e = make(); advance(e,11)
    seen = []
    def builder(c):
        seen.append(c)
        return plan(c)
    assert e.step(11,action,entry_plan=builder)['entry']
    p = e.pos
    assert p.tp == 1.1+d*.01 and p.sl == 1.1-d*.01
    assert e.counts['levels_clipped'] == 0
    assert isinstance(seen[0],FillContext)
    assert set(f.name for f in fields(FillContext)) == {'direction','balance','entry_quote','liquidation_quote','decision_ns','fill_ns'}
    assert e.step(12,CLOSE)['close'] == 'AGENT_CLOSE'
    t = e.trades[0]
    assert t['percentual_order_size'] == .2
    assert t['net_usd'] == pytest.approx(d*p.units*(t['exit_price']-p.entry_price)-t['slippage_usd']-t['commission_usd']-t['financing_usd'])
    advance(e,119)
    assert e.result()['end_equity_usd'] == pytest.approx(e.balance)


def test_daily_cap_removed_without_implicit_reversal():
    e=make(); advance(e,11)
    for _ in range(6):
        assert e.step(e.next_decision,LONG,entry_plan=plan)['entry']
        e.step(e.next_decision,CLOSE)
    assert max(e.entries_day.values()) == 6
    e.step(e.next_decision,LONG,entry_plan=plan)
    position=e.pos
    out=e.step(e.next_decision,SHORT,entry_plan=plan)
    assert out['refused'] == 'POSITION_OCCUPIED' and e.pos is position
    assert e.counts['close_REVERSAL'] == 0


def test_calendar_boundary_and_last_hour_not_silently_removed():
    e=make(); advance(e,10)
    assert e.step(10,LONG,entry_plan=plan)['refused'] == 'BEFORE_OPERATIONAL_START'
    assert e.step(11,LONG,entry_plan=plan)['entry']
    e.step(12,CLOSE); advance(e,106)
    out=e.step(106,SHORT,entry_plan=plan)
    assert out['entry'] and out['close'] == 'FORCED_FLAT'
    assert e.trades[-1]['exit_time'].value == e.ff[0]
    assert e.step(107,LONG,entry_plan=plan)['refused'] == 'ENTRY_CUTOFF'


@pytest.mark.parametrize('action,d', [(LONG,1), (SHORT,-1)])
def test_gap_sl_has_priority_over_early_close(action,d):
    e=make(); advance(e,11); e.step(11,action,entry_plan=plan)
    price=e.pos.sl-d*.001
    for name in ('o','h','l','c'):
        a=getattr(e,name).copy(); a[13]=price; setattr(e,name,a)
    assert e.step(12,CLOSE)['close'] == 'SL'


@pytest.mark.parametrize('changes,reason', [
    (dict(q=0), 'ABSOLUTE_ORDER'), (dict(q=1.1), 'ABSOLUTE_ORDER'),
    (dict(risk_budget=.01), 'RISK_BUDGET'), (dict(tp=1.100001), 'PRICE_TICK'),
    (dict(sl=1.101), 'GAP_BEYOND_LEVEL')])
def test_bad_levels_size_and_budget_refused(changes,reason):
    e=make(); advance(e,11)
    out=e.step(11,LONG,entry_plan=lambda c:replace(plan(c),**changes))
    assert not out['entry'] and out['refused'] == reason


def test_unknown_skipped_or_duplicate_decisions_and_partial_result_rejected():
    e=make()
    with pytest.raises(ValueError,match='SEQUENTIAL'): e.step(11,HOLD)
    with pytest.raises(ValueError,match='ACTION'): e.step(0,4)
    e.step(0,HOLD)
    with pytest.raises(ValueError,match='SEQUENTIAL'): e.step(0,HOLD)
    with pytest.raises(ValueError,match='INCOMPLETE'): e.result()
    with pytest.raises(ValueError,match='NO_AGENT_STATE'): e.state(0)


def test_future_prices_are_not_in_callback_context():
    a,b=make(),make()
    b.h=b.h.copy(); b.h[12:]+=.1
    contexts=[]
    for e in (a,b):
        advance(e,11)
        def capture(c):
            contexts.append(c)
            return plan(c)
        e.step(11,LONG,entry_plan=capture)
    assert contexts[0] == contexts[1]


@pytest.mark.parametrize('action,d', [(LONG,1),(SHORT,-1)])
def test_same_market_requires_position_specific_exit(action,d):
    import copy
    outcomes=[]
    for distance in (.001,.01):
        e=make(); advance(e,11)
        def builder(c):
            return replace(plan(c), sl=round(1.1-d*distance,5))
        e.step(11,action,entry_plan=builder)
        # Same future in both positions: an adverse excursion, then a recovery.
        for k,price in ((13,1.1),(14,1.1-d*.002),(15,1.1+d*.005)):
            for name in ('o','h','l','c'):
                a=getattr(e,name).copy(); a[k]=price; setattr(e,name,a)
        immediate=copy.deepcopy(e)
        immediate.step(12,CLOSE)
        close_now=immediate.trades[0]['net_usd']
        e.step(12,HOLD); e.step(13,HOLD)
        if e.pos is not None:
            e.step(14,CLOSE)
        outcomes.append((close_now,e.trades[0]['net_usd'],e.trades[0]['close_reason']))
    assert outcomes[0][0] > outcomes[0][1] and outcomes[0][2]=='SL'
    assert outcomes[1][0] < outcomes[1][1] and outcomes[1][2]=='AGENT_CLOSE'


def test_missing_open_session_hour_is_rejected_not_compressed():
    e=make()
    bars=e.bars.drop(e.bars.index[20])
    with pytest.raises(ValueError,match='UNOBSERVED'):
        AbsoluteBarEnv(bars,e.cal,np.full(len(bars),.001),np.full(len(bars),.002),
                       bars.index[0],bars.index[-1]+pd.Timedelta(hours=1),e.cfg,tick=.00001)


def test_broker_minimum_is_not_rounded_up_and_next_order_uses_current_balance():
    e=make(); advance(e,11)
    out=e.step(11,LONG,entry_plan=lambda c:replace(plan(c),q=1e-8))
    assert out['refused']=='UNITS' and e.pos is None
    contexts=[]
    def builder(c):
        contexts.append(c)
        return plan(c)
    e.step(12,LONG,entry_plan=builder); e.step(13,CLOSE)
    e.step(14,SHORT,entry_plan=builder)
    assert contexts[1].balance < contexts[0].balance
    assert contexts[1].balance == e.balance
