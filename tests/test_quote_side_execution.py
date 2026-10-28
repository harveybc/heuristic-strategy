from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided.data import calendar_from_closures
from app.short_horizon.expert_guided.env import BarEnv, EnvConfig, LONG, SHORT, HOLD, CLOSE
from app.short_horizon.execution import RealisticProfile, Sizing


def make(side=LONG, spread=.0002, zero=False):
    ts=pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
    bars=pd.DataFrame(dict(OPEN=1.1,HIGH=1.10001,LOW=1.09999,CLOSE=1.1),index=ts)
    cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                               ['2020-01-10T22:00Z','2020-01-12T22:00Z']])
    prof=replace(RealisticProfile(),spread_schedule=((0,24,spread),),weekly_open_spread=spread)
    cfg=EnvConfig(profile=prof,sizing=Sizing(leverage=50),execution_price_mode='bid_ask_v3',
                  cost_mode='zero_cost' if zero else 'configured')
    e=BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),cfg)
    assert e.step(12,side)['entry']
    return e


def set_bar(e,k,value,high=None,low=None):
    for name,x in [('o',value),('h',value if high is None else high),
                   ('l',value if low is None else low),('c',value)]:
        a=getattr(e,name).copy();a[k]=x;setattr(e,name,a)
    e.bars.iloc[k]=[e.o[k],e.h[k],e.l[k],e.c[k]]


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
def test_quotes_size_cash_and_mark_reconcile(side,d):
    e=make(side);p=e.pos;u=p.units
    assert p.entry_price==pytest.approx(1.1+d*.0001)
    assert u==math.floor(10000*.1*50/p.entry_price)
    expected=10000-u*.0002-p.entry_cost['slippage']-p.entry_cost['commission']
    mark=e.equity_at(13)
    assert mark==pytest.approx(expected)
    e.step(13,CLOSE);t=e.trades[0]
    assert t['exit_price']==pytest.approx(1.1-d*.0001)
    quote_pnl=d*u*(t['exit_price']-t['entry_price'])
    assert t['quote_gross_usd']==pytest.approx(quote_pnl)
    assert t['gross_usd']==pytest.approx(0,abs=1e-8)
    assert t['net_usd']==pytest.approx(quote_pnl-t['slippage_usd']-t['commission_usd']-t['financing_usd'])
    result=e.result()
    assert result['equity'].iloc[13]==pytest.approx(mark)
    assert result['end_equity_usd']==pytest.approx(e.balance)
    assert t['turnover_usd']==pytest.approx(u*(t['entry_price']+t['exit_price']))


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
def test_spread_crosses_stop_even_when_mid_does_not(side,d):
    e=make(side);sl=e.pos.sl
    mid=sl+d*.00005
    set_bar(e,14,mid)
    e.step(13,HOLD)
    t=e.trades[0]
    assert t['close_reason']=='SL'
    assert t['exit_price']==pytest.approx(mid-d*.0001)


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
def test_target_not_triggered_by_mid_alone(side,d):
    e=make(side);tp=e.pos.tp
    set_bar(e,14,tp+d*.00005)
    e.step(13,HOLD)
    assert e.pos is not None
    set_bar(e,15,tp+d*.00015)
    e.step(14,HOLD)
    assert e.trades[0]['close_reason']=='TP'
    assert e.trades[0]['exit_price']==pytest.approx(tp+d*.00005)


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
def test_minimum_distance_is_enforced_on_exit_side(side,d):
    e=make(side);e.reset()
    # Mid-to-stop 0.1 pip exceeds neither spread nor broker minimum after clipping.
    e.cfg=replace(e.cfg,sl_bounds=(.0001,3.))
    info=e.step(12,side,tp_mult=1.,sl_mult=.0001)
    assert not info['entry'] and info['refused'] in ('FILL_DISTANCE','GAP_BEYOND_LEVEL')
    assert not e.trades


@pytest.mark.parametrize('side',[LONG,SHORT])
def test_zero_cost_quote_mode_has_no_spread_debit(side):
    e=make(side,zero=True)
    assert e.pos.entry_price==1.1
    e.step(13,CLOSE)
    assert e.trades[0]['net_usd']==0
    assert e.result()['end_equity_usd']==10000


def test_unknown_quote_mode_rejected():
    with pytest.raises(ValueError,match='EXECUTION_PRICE_MODE'):
        EnvConfig(execution_price_mode='anything')


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
def test_intrabar_stop_uses_quote_and_reconciles(side,d):
    e=make(side);p=e.pos
    set_bar(e,14,1.1,high=p.sl-.00005 if d<0 else 1.1,
            low=p.sl+.00005 if d>0 else 1.1)
    e.step(13,HOLD)
    t=e.trades[0]
    assert t['close_reason']=='SL' and t['exit_price']==p.sl
    assert t['net_usd']==pytest.approx(d*p.units*(p.sl-p.entry_price)-t['slippage_usd']-t['commission_usd'])
    assert e.result()['end_equity_usd']==pytest.approx(e.balance)


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
@pytest.mark.parametrize('bar,total_spread',[(24,.0004),(25,.0002)])
def test_funding_and_variable_spread_marks_reconcile(side,d,bar,total_spread):
    e=make(side);p=e.pos
    e.cfg=replace(e.cfg,profile=replace(e.cfg.profile,spread_schedule=((0,2,.0006),(2,24,.0002))))
    for b in range(13,bar):e.step(b,HOLD)
    mark=e.equity_at(bar)
    rate=e.cfg.profile.swap_long_per_lot if d>0 else e.cfg.profile.swap_short_per_lot
    due=-rate*p.units/e.cfg.profile.lot_units
    assert mark==pytest.approx(10000-p.units*total_spread-p.entry_cost['slippage']-p.entry_cost['commission']-due)
    e.step(bar,CLOSE)
    t=e.trades[0]
    assert t['financing_usd']==pytest.approx(due)
    assert t['net_usd']==pytest.approx(d*p.units*(t['exit_price']-p.entry_price)-t['slippage_usd']-t['commission_usd']-due)
    assert e.result()['equity'].iloc[bar]==pytest.approx(mark)


@pytest.mark.parametrize('side,d',[(LONG,1),(SHORT,-1)])
def test_quote_margin_barrier_and_open_priority(side,d):
    e=make(side)
    p=e.pos
    # Isolate margin liquidation with a distant stop, without changing quote accounting.
    p.sl=p.entry_price-d*.8
    level=e._margin_level(14)
    mid=level-d*.001+d*.0001
    set_bar(e,14,mid)
    e.step(13,CLOSE)
    assert e.trades[0]['close_reason']=='MARGIN_CLOSEOUT'
    assert e.trades[0]['exit_price']==pytest.approx(mid-d*.0001)
    assert e.result()['end_equity_usd']==pytest.approx(e.balance)
