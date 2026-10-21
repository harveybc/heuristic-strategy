"""Controlled specialization study using the existing temporal BC/DAgger engine."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from app.short_horizon.expert_guided import data as DG, env as E
from tools import expert_guided_e3m as M
from tools.expert_guided_e1t import Ctx, cat

ARMS = ('JOINT16', 'JOINT_MATCHED', 'FULL')
SEEDS = (20261019, 20261020, 20261021, 20261022, 20261023)
RATES = (.02, .03, .04, .06, .08, .12, .15, .18)


def cells():
    return [dict(id=f'{arm.lower()}-{seed}', arm=arm, seed=seed) for seed in SEEDS for arm in ARMS]


def business_config():
    return E.EnvConfig(max_entries_per_day=1, min_orders_per_week=0.,
                       sizing=E.X.Sizing(rel_volume=.1, leverage=50.))


def modules(arm):
    return M.MODULES['FULL' if arm in ('FULL','FULL_MEAN','FULL_GAP') else 'JOINT']


def network(arm, name, head=None):
    from app.short_horizon.expert_guided.temporal import TemporalPolicyNet
    nc, nr = modules(arm)[name]
    return TemporalPolicyNet(15, 13, nc, n_reg=nr, n_state=9, units=2,
                             branch=2, core=2, depth=1, head=head or 16, dropout=.1)


def architecture_counts():
    count = lambda arm, name, h: sum(p.numel() for p in network(arm, name, h).parameters())
    full = sum(count('FULL', n, 16) for n in modules('FULL'))
    h = min(range(1, 257), key=lambda h: (abs(count('JOINT16', 'joint', h)-full), h))
    return {a: dict(head=h if a == 'JOINT_MATCHED' else 16,
                    total=sum(count(a, n, h if a == 'JOINT_MATCHED' else 16) for n in modules(a)))
            for a in ARMS}


class Policy:
    def __init__(self, arm, pols, tau, *, exit_mode='learned', direction=0, argmax=False):
        self.arm, self.pols, self.tau = arm, pols, tau
        self.exit_mode, self.direction, self.argmax = exit_mode, direction, argmax
        self.base = M.Composed('FULL' if arm in ('FULL','FULL_MEAN','FULL_GAP') else 'JOINT', pols, 'prob', tau)

    def action(self, window, st):
        if len(st) != 9 or not np.isfinite(st).all() or st[0] not in (-1, 0, 1):
            raise ValueError('MODULAR_STATE_INVALID')
        x = np.r_[window, st].astype(np.float32)
        if st[0] != 0:
            close = self.exit_mode == 'learned' and self.base.exit_now(x, int(st[0]))
            return (E.CLOSE if close else E.HOLD), 1.5, 2.
        if st[-1] <= .5:
            return E.HOLD, 1.5, 2.
        pL, tpL, slL, pS, tpS, slS = self.base.openers(x)
        if self.argmax:
            P, _ = self.base._probs('joint', x)
            d = int(np.argmax(P))
            if d not in (E.LONG, E.SHORT):
                return E.HOLD, 1.5, 2.
        else:
            # Direction ablation suppresses the other direction, without retuning.
            d = self.base.arbitrate(pL if self.direction != -1 else -1., tpL, slL,
                                    pS if self.direction != 1 else -1., tpS, slS)
        if d == E.HOLD:
            return E.HOLD, 1.5, 2.
        return d, float(tpL if d == E.LONG else tpS), float(slL if d == E.LONG else slS)

    def callback(self, cx):
        return lambda bar, r, st, env: self.action(cx.Xw[r], st)


def label_within_period(cx, info, period_end):
    if info is None:
        return True
    ffb = info[-1]
    return 0 < ffb <= len(cx.v) and int(cx.v[ffb-1])+DG.H_NS <= period_end


def opening_label_within_period(cx, bar, period_end):
    # The TP and DAgger leg use the NEXT pivot, even when bar is itself a pivot.
    pivots=np.asarray([p[0] for p in cx.ds.pivots],dtype=np.int64)
    j=int(np.searchsorted(pivots,bar,side='right'))
    if j>=len(pivots):return False
    confirmation=cx.ds.conf[j]
    return (confirmation is not None and 0<=confirmation<len(cx.v)
            and int(cx.v[confirmation])+DG.H_NS<=period_end)


def rollout(cx, period, action, cfg=None, training=False):
    """Inference never uses label support; training rows also respect future label bounds."""
    cfg = cfg or business_config()
    a, b = DG.PERIODS[period]
    env = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, cfg)
    end = DG._utc_ns(b)
    vis = []
    for bar in range(env.first, env.last):
        r, st = cx.rob[bar], env.state(bar)
        act, tp, sl = E.HOLD, 1.5, 2.
        if r >= 0 and cx.obs_ok[r]:
            act, tp, sl = action(bar, r, st, env)
            if training and cx.lab_ok[r] and (st[0] != 0 or st[-1] > .5):
                p = env.pos
                info = None if p is None else (p.direction, p.entry_price, p.sl, p.tp,
                      int(np.searchsorted(cx.v, env.ff[int(env.seg_of[bar])])))
                if (label_within_period(cx, info, end)
                        and (info is not None or opening_label_within_period(cx,bar,end))):
                    vis.append((r, bar, st, act, tp, sl, info))
        # Even the teacher/exploration obeys the same close-only positioned routing.
        if st[0] != 0 and act in (E.LONG, E.SHORT):
            d = 1 if act == E.LONG else -1
            act = E.CLOSE if d != st[0] else E.HOLD
        env.step(bar, act, tp, sl)
    return env.result(), vis, env.trades


def scalar_metrics(result, trades):
    out = E.summary(result)
    out['profit_pct'] = out['net_profit_usd']/100.
    out['activity'] = 'NO_TRADES' if not trades else 'TRADING'
    if not trades:
        out['sharpe_daily_sqrt365'] = None
        out['sortino_daily_sqrt365'] = None
    for direction, label in ((1,'buy'), (-1,'sell')):
        selected = [t for t in trades if t['direction'] == direction]
        out[label+'_trades'] = len(selected)
        out[label+'_net_usd'] = sum(t['net_usd'] for t in selected)
        out[label+'_win_fraction'] = sum(t['net_usd']>0 for t in selected)/len(selected) if selected else None
    equity = np.asarray(result['daily_equity'], dtype=float)
    # Account returns a Series; include the initial capital in daily growth.
    vals = np.r_[10000., equity]
    out['mean_daily_log_growth'] = float(np.log(vals[1:]/vals[:-1]).mean()) if (vals>0).all() and len(vals)>1 else None
    return out


def choose_threshold(rows):
    def key(r):
        s = r['sharpe']
        score = float(s) if s is not None and math.isfinite(s) else (0. if r['entries']==0 else -math.inf)
        return score, -r['entries'], r['tau']
    return max(rows, key=key)


def calibration_scores(cx, policy):
    a,b = DG.PERIODS['INNER_A']
    env = E.BarEnv(cx.ds.bars,cx.ds.cal,cx.ds.sigma,cx.ds.sigma_rule,a,b,business_config())
    scores=[]
    for bar in range(env.first, env.last):
        r,st=cx.rob[bar],env.state(bar)
        if r>=0 and cx.obs_ok[r] and st[-1]>.5:
            pL,_,_,pS,_,_=policy.base.openers(np.r_[cx.Xw[r],st].astype(np.float32))
            scores.append(max(pL,pS))
        env.step(bar,E.HOLD)
    if not scores:
        raise ValueError('CALIBRATION_SUPPORT_EMPTY')
    return np.asarray(scores)


def calibrate(cx, arm, pols, exit_mode='learned'):
    scores=calibration_scores(cx,Policy(arm,pols,.5))
    records=[]
    for rate in RATES:
        tau=float(np.quantile(scores,1-rate))
        p=Policy(arm,pols,tau,exit_mode=exit_mode)
        result,_,trades=rollout(cx,'INNER_A',p.callback(cx))
        met=scalar_metrics(result,trades)
        records.append(dict(rate=rate,tau=tau,sharpe=met['sharpe_daily_sqrt365'],
                            entries=met['n_entries'],net=met['net_profit_usd']))
    return choose_threshold(records), records


def run_cell(spec, cell, out):
    import torch
    from app.short_horizon.expert_guided import learn as L, temporal as T
    from tools.modular_campaign import write, digest, read, clean
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):
        raise ValueError('ASSIGNED_5090_REQUIRED')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.time()
    def progress(**kw):
        write(out/'PROGRESS.json',dict(cell=cell,updated=time.time(),elapsed=time.time()-started,**kw))
    progress(stage='DATA_VERIFICATION')
    ds=DG.load(spec['admitted'],spec['csv_dir'])
    cx=Ctx(ds)
    cx.h,cx.l,cx.c=(ds.bars[k].to_numpy(float) for k in ('HIGH','LOW','CLOSE'))
    counts=architecture_counts();counts['FULL_MEAN']=counts['FULL_GAP']=counts['FULL']
    arm=cell['arm']; seed=cell['seed']; head=counts[arm]['head']
    gap_penalty=.25 if arm=='FULL_GAP' else 0.
    effective=dict(schema='modular_controlled.v1',cell=cell,architecture=counts[arm],
        runtime=spec['runtime'],business=asdict(business_config()),periods=DG.PERIODS,
        input_pins=DG.PINS,features=DG.FEATURES,window=DG.WIN,stride=DG.STRIDE,
        labels='ZigZag v2 opening; SL-before-TP early exit',rounds=3,
        max_epochs=1000,patience=10,min_delta=1e-8,lr=.001,weight_decay=.0001,dropout=.1,
        batch=256,monitor='.5*(TRAIN + INNER_A inference losses)',rates=RATES,
        device=torch.cuda.get_device_name(0),torch_version=torch.__version__,gap_penalty=gap_penalty)
    if gap_penalty:
        effective['monitor'] += ' + .25*max(0,INNER_A_loss-TRAIN_loss)'
    write(out/'CONFIG.json',effective)
    write(out/'DATA_REPORT.json',ds.meta)
    cfg=business_config()
    expert_cfg=replace(cfg,level_policy='refuse',tp_bounds=(1e-6,1e3),sl_bounds=(1e-6,1e3))
    demos={p:rollout(cx,p,cx.expert,expert_cfg,training=True)[1] for p in ('TRAIN','INNER_A')}
    prior=np.bincount([v[3] for v in demos['TRAIN']],minlength=4).astype(float)
    if not prior.sum(): raise ValueError('DEMONSTRATIONS_EMPTY')
    prior/=prior.sum()
    explore={p:[v for v in rollout(cx,p,cx.prior_random(prior,0),training=True)[1] if v[6] is not None]
             for p in ('TRAIN','INNER_A')}
    variant='FULL' if arm in ('FULL','FULL_MEAN','FULL_GAP') else 'JOINT'
    sets={p:M.build_sets(variant,demos[p]+explore[p],cx,True) for p in demos}
    pols={}; histories=[]
    for rnd in range(4):
        if rnd:
            comp=Policy(arm,pols,.5)
            comp.tau=float(np.quantile(calibration_scores(cx,comp),1-.06)); comp.base.tau=comp.tau
            for p in sets:
                progress(stage='DAGGER_COLLECTION',round=rnd,period=p)
                vis=rollout(cx,p,comp.callback(cx),training=True)[1]
                for name,values in M.build_sets(variant,vis,cx,False).items():
                    sets[p][name]=cat(sets[p][name],values) if name in sets[p] else values
        for name,(nc,nr) in modules(arm).items():
            tr,va=sets['TRAIN'].get(name),sets['INNER_A'].get(name)
            if tr is None or va is None or len(np.unique(tr[1]))<2:
                if name not in pols:raise ValueError('EMPTY_INITIAL_MODULE:'+name)
                histories.append(dict(round=rnd,module=name,kept=True));continue
            progress(stage='FIT',round=rnd,module=name,train_rows=len(tr[1]),monitor_rows=len(va[1]))
            freq=np.maximum(np.bincount(tr[1],minlength=nc)/len(tr[1]),1e-3)
            w=1/np.sqrt(freq);w/=w[tr[1]].mean()
            model,hist,info=L.fit(tr[0],tr[1],va[0],va[1],seed=seed,n_cls=nc,
                R=tr[2] if nr else None,Rm=tr[3] if nr else None,
                Ri=va[2] if nr else None,Rmi=va[3] if nr else None,class_weight=w,
                dropout=.1,lr=.001,wd=.0001,batch=256,max_epochs=1000,patience=10,min_delta=1e-8,
                device='cuda',net=lambda:network(arm,name,head),gap_penalty=gap_penalty)
            pol=T.TorchPolicy(model)
            path=out/f'round{rnd}-{name}.pt'
            torch.save(model.state_dict(),path)
            clone=network(arm,name,head)
            clone.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
            copied=T.TorchPolicy(clone)
            for x,y in zip(pol(tr[0][:16]),copied(tr[0][:16])):
                if x is not None and not np.array_equal(x,y):raise ValueError('RELOAD_PARITY_FAILED')
            pols[name]=pol
            record=dict(round=rnd,module=name,info=info,history=hist,checkpoint=path.name,
                        checkpoint_sha256=digest(path),reload_bitwise=True,
                        rows={p:len(sets[p][name][1]) for p in sets},
                        data_sha256={p:hashlib.sha256(b''.join(x.tobytes() for x in sets[p][name])).hexdigest() for p in sets})
            write(out/f'round{rnd}-{name}.json',record);histories.append(record)
            torch.cuda.empty_cache()
    progress(stage='CALIBRATION')
    best,trials=calibrate(cx,arm,pols)
    write(out/'FROZEN.json',dict(cell=cell,best=best,trials=trials,config_sha256=digest(out/'CONFIG.json'),
                               frozen_at=time.time(),selected_on='2023_H1'))
    policies={'combined':Policy(arm,pols,best['tau'])}
    if variant=='FULL':
        fixed,fixed_trials=calibrate(cx,arm,pols,exit_mode='fixed')
        write(out/'FIXED_EXIT_CALIBRATION.json',dict(best=fixed,trials=fixed_trials))
        policies.update(fixed_exit=Policy(arm,pols,best['tau'],exit_mode='fixed'),
            fixed_exit_calibrated=Policy(arm,pols,fixed['tau'],exit_mode='fixed'),
            buy_only=Policy(arm,pols,best['tau'],direction=1),sell_only=Policy(arm,pols,best['tau'],direction=-1))
    else:policies['argmax']=Policy(arm,pols,best['tau'],argmax=True)
    results={}
    for label,policy in policies.items():
        results[label]={}
        for period in DG.PERIODS:
            progress(stage='EVALUATION',control=label,period=period)
            result,_,trades=rollout(cx,period,policy.callback(cx))
            met=scalar_metrics(result,trades)
            if met['max_entries_per_day']>1:raise ValueError('BUSINESS_DAILY_LIMIT')
            results[label][period]=met
            write(out/f'{label}-{period}-trades.json',clean(trades))
            eq=result['daily_equity']
            write(out/f'{label}-{period}-equity.json',dict(time=[str(t) for t in eq.index],equity=eq.tolist()))
    summary=dict(cell=cell,metrics=results,architecture=counts[arm],wall_seconds=time.time()-started,
                 frozen_sha256=digest(out/'FROZEN.json'),config_sha256=digest(out/'CONFIG.json'),
                 gpu_peak_bytes=torch.cuda.max_memory_allocated(),completed_at=time.time())
    write(out/'RESULT.json',clean(summary))
    files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ('PROGRESS.json','CLOSE.json','worker.log')}
    write(out/'CLOSE.json',dict(cell=cell,spec_sha256=spec['spec_sha256'],files=files))
    progress(stage='COMPLETED')


def main():
    from tools.modular_campaign import read, load_spec
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cell',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    spec=load_spec(Path(a.root));cell=next(c for c in spec['cells'] if c['id']==a.cell)
    try:
        run_cell(spec,cell,Path(a.out))
    except ValueError as exc:
        from tools.modular_campaign import write
        write(Path(a.out)/'ERROR.json',dict(type=type(exc).__name__,reason=str(exc)))
        # Scientific identity / business / numerical violations are not a retryable fit.
        raise SystemExit(78)

if __name__=='__main__':main()
