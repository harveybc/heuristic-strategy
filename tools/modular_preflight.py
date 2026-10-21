"""Bounded real-input, publication-schema and synthetic GPU admission witness."""
import argparse
from pathlib import Path
import time
import numpy as np
from tools import modular_controlled as C, modular_campaign as Q


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    root=Path(a.root);spec=Q.load_spec(root);start=time.time()
    ds=C.DG.load(spec['admitted'],spec['csv_dir']);cx=C.Ctx(ds)
    cx.h,cx.l,cx.c=(ds.bars[k].to_numpy(float) for k in ('HIGH','LOW','CLOSE'))
    cfg=C.business_config();expert=C.replace(cfg,level_policy='refuse',tp_bounds=(1e-6,1e3),sl_bounds=(1e-6,1e3))
    counts={};prior=None
    for period in ('TRAIN','INNER_A'):
        _,demo,_=C.rollout(cx,period,cx.expert,expert,training=True)
        if prior is None:
            prior=np.bincount([v[3] for v in demo],minlength=4).astype(float);prior/=prior.sum()
        _,explore,_=C.rollout(cx,period,cx.prior_random(prior,0),training=True)
        demo += [v for v in explore if v[6] is not None]
        counts[period]={}
        for arm in ('JOINT','FULL'):
            data=C.M.build_sets(arm,demo,cx,True)
            counts[period][arm]={m:dict(rows=len(d[1]),classes=np.bincount(d[1]).tolist()) for m,d in data.items()}
            if set(data)!=set(C.M.MODULES[arm]) or any(len(np.unique(d[1]))<2 for d in data.values()):
                raise ValueError('INITIAL_PARTITION_EMPTY:'+period+':'+arm)
    met,_,trades=C.rollout(cx,'INNER_A',lambda *a:(C.E.HOLD,1.5,2.))
    no_trade=C.scalar_metrics(met,trades)
    if no_trade['net_profit_usd']!=0 or no_trade['sharpe_daily_sqrt365'] is not None:raise ValueError('ABSTENTION_CONTROL')
    gpu=None
    if a.gpu:
        import torch
        from app.short_horizon.expert_guided import learn as L,temporal as T
        if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):raise ValueError('GPU_ASSIGNMENT')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(701);x=rng.normal(size=(64,204)).astype(np.float32);y=np.tile([0,1,2,3],16)
        model,hist,info=L.fit(x,y,x[:16],y[:16],seed=701,n_cls=4,device='cuda',max_epochs=2,batch=16,
                            gap_penalty=.25 if spec.get('study')=='monitor_gap' else 0.,
                            net=lambda:C.network('JOINT16','joint'))
        pol=T.TorchPolicy(model);path=root/'GPU_SMOKE.pt';torch.save(model.state_dict(),path)
        clone=C.network('JOINT16','joint');clone.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
        other=T.TorchPolicy(clone)
        assert all(np.array_equal(v,w) for v,w in zip(pol(x),other(x)))
        gpu=dict(epochs=len(hist),reload_bitwise=True,synthetic=True,peak_bytes=torch.cuda.max_memory_allocated())
    Q.write(root/'PREFLIGHT.json',dict(state='PASS',seconds=time.time()-start,counts=counts,
        architecture=C.architecture_counts(),no_trade=no_trade,gpu=gpu,inputs=spec['inputs']))

if __name__=='__main__':main()
