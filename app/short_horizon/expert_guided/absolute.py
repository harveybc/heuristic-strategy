"""Opt-in absolute orders on the native quote/cost/protection state machine."""
from dataclasses import dataclass
import math

import numpy as np

from .env import BarEnv, HOLD, LONG, SHORT, CLOSE, H_NS


@dataclass(frozen=True)
class FillContext:
    direction: int
    balance: float
    entry_quote: float
    liquidation_quote: float
    decision_ns: int
    fill_ns: int


@dataclass(frozen=True)
class AbsoluteOrder:
    tp: float
    sl: float
    q: float
    risk_budget: float
    funding_reserve_per_unit: float


class AbsoluteBarEnv(BarEnv):
    """Templates resolve at the next open, never receive future intrabar prices.

    The caller captures decision-time references and sigma in a pure entry_plan.
    No daily cap, implicit reversal, level clipping or agent-state observation.
    """
    def __init__(self, *args, tick, **kwargs):
        if not math.isfinite(tick) or tick <= 0:
            raise ValueError('PRICE_TICK')
        self.tick = tick
        super().__init__(*args, **kwargs)
        c = self.cfg
        if (c.execution_price_mode != 'bid_ask_v3' or c.max_entries_per_day is not None
                or c.sizing.leverage != 50 or c.sizing.rel_volume != .1 or c.sizing.basis != 'margin'):
            raise ValueError('ABSOLUTE_BUSINESS_CONTRACT')
        if (not np.all(self.op-self.mo == 12*H_NS) or not np.array_equal(self.ec, self.ff)
                or any(s.market_close.value-s.force_flat.value != 12*H_NS for s in self.cal.segments)):
            raise ValueError('ABSOLUTE_CALENDAR_CONTRACT')
        if (not np.isfinite(self.bars[['OPEN','HIGH','LOW','CLOSE']].to_numpy()).all()
                or np.any(np.minimum(self.o,self.l) <= 0)
                or np.any(self.h < np.maximum.reduce([self.o,self.l,self.c]))
                or np.any(self.l > np.minimum.reduce([self.o,self.h,self.c]))
                or np.any(np.diff(self.v) <= 0) or (self.seg_of < 0).any()
                or np.any((np.diff(self.seg_of) == 0) & (np.diff(self.v) != H_NS))):
            raise ValueError('UNOBSERVED_OPEN_SESSION_GAP_OR_INVALID_BARS')

    def reset(self):
        first = super().reset()
        self.next_decision = first
        self._entry_plan = None
        self.preparation = None
        for reason in ('ABSOLUTE_ORDER','PRICE_TICK','RISK_BUDGET','POSITION_OCCUPIED'):
            self.counts['refused_'+reason] = 0
        return first

    def state(self, b):
        raise ValueError('NO_AGENT_STATE_IN_ABSOLUTE_INTERFACE')

    def entry_gate(self, b):
        k = b+1
        if k > self.last or self.seg_of[k] != self.seg_of[b]:
            return False, 'SESSION_END'
        si = self.seg_of[k]
        if self.v[k] < self.op[si]:
            return False, 'BEFORE_OPERATIONAL_START'
        if self.v[k] >= self.ff[si]:
            return False, 'ENTRY_CUTOFF'
        if self.balance <= 0:
            return False, 'RUIN'
        return True, 'OK'

    def _prepare_entry(self, b, direction, tp_mult, sl_mult):
        k = b+1
        ctx = FillContext(direction, float(self.balance), self.entry_fill_price(k,direction),
                          self.quote_price(k,self.o[k],-direction), int(self.v[b])+H_NS, int(self.v[k]))
        order = self._entry_plan(ctx)
        if (not isinstance(order, AbsoluteOrder)
                or not all(math.isfinite(v) for v in (order.tp,order.sl,order.q,order.risk_budget,order.funding_reserve_per_unit))
                or min(order.tp,order.sl,order.risk_budget) <= 0 or not 0 < order.q <= 1
                or order.funding_reserve_per_unit < 0):
            return None, 'ABSOLUTE_ORDER'
        if any(not math.isclose(v/self.tick, round(v/self.tick), abs_tol=1e-7, rel_tol=0) for v in (order.tp,order.sl)):
            return None, 'PRICE_TICK'
        if any(direction*(order.tp-p) <= 0 or direction*(p-order.sl) <= 0 for p in (ctx.entry_quote,ctx.liquidation_quote)):
            return None, 'GAP_BEYOND_LEVEL'
        if min(abs(v-p) for v in (order.tp,order.sl) for p in (ctx.entry_quote,ctx.liquidation_quote)) < self.cfg.rules.min_distance-1e-12:
            return None, 'FILL_DISTANCE'
        units = self.cfg.sizing.order_units(ctx.balance, ctx.entry_quote, order.q, risk_accepts_minimum=False)
        if units is None:
            return None, 'UNITS'
        p = self.cfg.profile
        cost = p.slippage_market+p.slippage_stop+2*p.commission_per_lot_side/p.lot_units+order.funding_reserve_per_unit
        risk = units*(direction*(ctx.entry_quote-order.sl)+cost)
        margin = self.cfg.sizing.required_margin(units,ctx.entry_quote)
        if risk > order.risk_budget+1e-9 or margin > .1*ctx.balance+1e-9:
            return None, 'RISK_BUDGET'
        self.preparation = dict(tp=order.tp,sl=order.sl,q=order.q,units=units,modeled_loss=risk,
                                margin=margin,risk_budget=order.risk_budget,fill_ns=ctx.fill_ns)
        return (order.tp,order.sl,units,False,order.q), 'OK'

    def step(self, b, action, *, entry_plan=None):
        if b != self.next_decision:
            raise ValueError('SEQUENTIAL_DECISIONS_REQUIRED')
        if action not in (HOLD,LONG,SHORT,CLOSE):
            raise ValueError('UNKNOWN_ACTION')
        if (action in (LONG,SHORT) and not callable(entry_plan)) or (action in (HOLD,CLOSE) and entry_plan is not None):
            raise ValueError('ENTRY_PLAN_CONTRACT')
        occupied = action in (LONG,SHORT) and self.pos is not None
        self._entry_plan = entry_plan
        self.preparation = None
        try:
            info = super().step(b,HOLD if occupied else action)
        finally:
            self._entry_plan = None
        if occupied:
            self.counts['refused_POSITION_OCCUPIED'] += 1
            info['refused'] = 'POSITION_OCCUPIED'
        if self.preparation is not None:
            info['absolute_order'] = self.preparation.copy()
        self.next_decision += 1
        return info

    def flatten_at_end(self):
        if self.next_decision != self.last:
            raise ValueError('INCOMPLETE_ABSOLUTE_REPLAY')
        return super().flatten_at_end()
