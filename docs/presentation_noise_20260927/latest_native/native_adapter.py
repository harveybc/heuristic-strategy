"""Execute the existing plugin, without replacing its entry or exit decisions."""
import argparse
import hashlib
import math
from pathlib import Path
import time

import backtrader as bt
import numpy as np
import pandas as pd

from tools.corrected_price_noise_sweep import ExplicitCosts, margin_order_size, masked_mae
from tools.price_noise_sweep import ROOT, COLUMNS, OFFSETS, digest, write_json, equity_statistics
from app.plugins.plugin_long_short_predictions import Plugin
from app.policies.prediction_entry_exit import compute_legacy_order_size


def row_digest(values):
    return hashlib.sha256(np.asarray(values,dtype="<f8").tobytes()).hexdigest()


class MarginBroker(bt.brokers.BackBroker):
    def _try_exec_market(self,order,popen,phigh,plow):
        if order.info.get("reason")=="ENTRY" and order.data.datetime[0]>order.created.dt:
            if self.positions[order.data]:
                raise RuntimeError("Entry requires flat state")
            original=abs(order.executed.remsize)
            size=margin_order_size(min(self.cash,order.info.decision_balance),.05,100,popen,original)
            order.addinfo(original_requested_units=original,fill_balance=self.cash)
            signed=math.copysign(size,order.created.size)
            order.size=order.created.size=order.executed.remsize=signed
        return super()._try_exec_market(order,popen,phigh,plow)


class RecordedNative(Plugin.HeuristicStrategy):
    def __init__(self,prediction_frame,last_origin_position,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Input adapter only: exact float64 arrays, no second lossy CSV parse.
        self.pred_df=prediction_frame.copy()
        self.last_origin_position=last_origin_position
        self.order_entry_price=None
        self.equity,self.fills,self.closed_trades,self.decisions=[],[],[],[]
        self.native_calls=0
        self.rejections=0
        self.finalizing=False

    def next(self):
        dt=self.data.datetime.datetime()
        self.equity.append((dt,float(self.broker.getvalue())))
        if len(self)-1>self.last_origin_position:
            if self.position:
                self.finalizing=True
                self.close()
            return
        row=self.pred_df.loc[dt].to_numpy() if dt in self.pred_df.index else None
        self.decisions.append(dict(time=str(dt),stage="native",position=float(self.position.size),
            direction=self.current_direction,tp=self.current_tp,sl=self.current_sl,
            prediction_sha256=None if row is None else row_digest(row)))
        self.native_calls+=1
        super().next()

    def compute_size(self,rr):
        cap=compute_legacy_order_size(reward_risk_ratio=rr,available_cash=math.inf,params=self._policy_params())
        return margin_order_size(self.broker.getvalue(),.05,100,self.data.close[0],cap)

    def _tag(self,order):
        if order is None:
            return order
        reason="FINAL_LIQUIDATION" if self.finalizing else ("NATIVE_EXIT" if self.position else "ENTRY")
        order.addinfo(reason=reason,decision_balance=float(self.broker.getvalue()))
        return order

    def buy(self,*args,**kwargs):
        return self._tag(super().buy(*args,**kwargs))

    def sell(self,*args,**kwargs):
        return self._tag(super().sell(*args,**kwargs))

    def notify_order(self,order):
        if order.status==order.Completed:
            self.fills.append(dict(time=str(bt.num2date(order.executed.dt)),created=str(bt.num2date(order.created.dt)),
                reason=order.info.reason,order_type=order.getordername(),requested_price=float(order.created.price),
                tp=self.current_tp,sl=self.current_sl,decision_balance_usd=order.info.decision_balance,
                fill_balance_usd=order.info.get("fill_balance"),requested_units=order.info.get("original_requested_units"),
                entry_margin_usd=abs(order.executed.size)*order.executed.price/100 if order.info.reason=="ENTRY" else None,
                size=float(order.executed.size),price=float(order.executed.price),
                commission_usd=abs(order.executed.size)*7/100000,friction_usd=abs(order.executed.size)*.00010,
                all_costs_with_swap_usd=float(order.executed.comm)))
        if order.status in [order.Margin,order.Rejected,order.Expired]:
            self.rejections+=1
        super().notify_order(order)

    def notify_trade(self,trade):
        if trade.isclosed:
            self.closed_trades.append(dict(open=str(bt.num2date(trade.dtopen)),close=str(bt.num2date(trade.dtclose)),
                gross_usd=float(trade.pnl),net_usd=float(trade.pnlcomm),costs_usd=float(trade.commission)))
        super().notify_trade(trade)

    def stop(self):
        if self.position or self.broker.get_orders_open():
            raise RuntimeError("Unsettled final position/order")


def execute(frame,predicted,last_position,out,strategy_class=RecordedNative):
    # Constructor expects a file; supply only its header, then inject exact rows.
    path=out/"prediction_schema.csv"
    predicted.iloc[:0].to_csv(path,index_label="DATE_TIME")
    params={k:v for k,v in Plugin.plugin_params.items() if k not in {"spread_pips","commission_per_lot","slippage_pips"}}
    params.update(rel_volume=.05,pred_file=str(path))
    engine=bt.Cerebro(stdstats=False)
    engine.setbroker(MarginBroker(shortcash=False))
    costs=ExplicitCosts()
    engine.broker.addcommissioninfo(costs)
    engine.broker.setcash(10000)
    engine.adddata(bt.feeds.PandasData(dataname=frame.iloc[:last_position+3].rename(columns=str.lower)))
    engine.addstrategy(strategy_class,prediction_frame=predicted,last_origin_position=last_position,**params)
    return engine.run()[0],engine,costs


def run_cell(frame,origins,target,positions,predictions,out,cell,margin_fraction=.05):
    if margin_fraction!=.05:
        raise ValueError("Frozen five percent margin required")
    out.mkdir(parents=True,exist_ok=False)
    predicted=pd.DataFrame(predictions,index=origins,columns=COLUMNS)
    cpu,wall=time.process_time(),time.monotonic()
    result,engine,costs=execute(frame,predicted,int(positions[-1]),out)
    cpu,wall=time.process_time()-cpu,time.monotonic()-wall
    equity=pd.Series([x[1] for x in result.equity],index=pd.DatetimeIndex([x[0] for x in result.equity]))
    stats,daily=equity_statistics(equity)
    gross=math.fsum(t["gross_usd"] for t in result.closed_trades)
    commission=math.fsum(f["commission_usd"] for f in result.fills)
    friction=math.fsum(f["friction_usd"] for f in result.fills)
    net=math.fsum(t["net_usd"] for t in result.closed_trades)
    if abs(gross-commission-friction-costs.swap_debits-net)>1e-5 or abs(net-stats["profit_usd"])>1e-5:
        raise RuntimeError("Native cash/trade/cost reconciliation failed")
    if result.rejections:
        raise RuntimeError("Native orders rejected")
    daily.to_csv(out/"daily_equity.csv",index_label="date",header=["equity_usd"])
    pd.DataFrame(result.fills,columns=["time","created","reason","order_type","requested_price","tp","sl",
        "decision_balance_usd","fill_balance_usd","requested_units","entry_margin_usd","size","price",
        "commission_usd","friction_usd","all_costs_with_swap_usd"]).to_csv(out/"fills.csv",index=False)
    pd.DataFrame(result.closed_trades,columns=["open","close","gross_usd","net_usd","costs_usd"]).to_csv(out/"trades.csv",index=False)
    write_json(out/"decisions.json",result.decisions)
    record=dict(cell=cell,**stats,n_origins=len(origins),
        mae_by_horizon=dict(zip(map(str,OFFSETS),map(float,masked_mae(predictions,target)))),
        n_by_horizon=dict(zip(map(str,OFFSETS),map(int,np.isfinite(target).sum(axis=0)))),
        n_closed_trades=len(result.closed_trades),n_rejected_orders=result.rejections,
        commission_usd=commission,friction_usd=friction,swap_usd=costs.swap_debits,
        final_position=0,max_margin_fraction=.05,native_calls=result.native_calls,exit_variant=result.exit_variant,
        cpu_seconds=cpu,wall_seconds=wall,artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    write_json(out/"result.json",record)
    return record


if __name__=="__main__":
    from tools.conditional_noise_sweep import run
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=ROOT/"tests/data/phase_2_3_base_d3.csv")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot",action="store_true")
    args=parser.parse_args()
    args.native_plugin=True
    run(args)
