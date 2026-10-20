"""Owner-approved specialist contrast: independent behavioral witnesses."""
from types import SimpleNamespace
import numpy as np
import pytest
from tools import modular_controlled as C
from app.short_horizon.expert_guided import env as E

class Const:
    def __init__(self, logits, reg=None):
        self.logits, self.reg = logits, reg
    def __call__(self, x):
        return np.tile(self.logits, (len(x), 1)), None if self.reg is None else np.tile(self.reg, (len(x), 1))

def full():
    return dict(open_l=Const([0., 3.], [1., 2.]), open_s=Const([3., 0.], [1., 2.]),
                close_l=Const([0., 3.]), close_s=Const([3., 0.]))

def state(pos=0, allowed=1):
    return np.array([pos, .1, .2, 1., 2., .4, .5, .0, allowed], np.float32)

def test_only_correct_specialist_is_called_and_no_reversal():
    p=full(); p['open_l']=lambda x: (_ for _ in ()).throw(AssertionError('opener while positioned'))
    f=C.Policy('FULL', p, .5).action(np.zeros(195), state(1))
    assert f[0]==E.CLOSE
    assert C.Policy('FULL', p, .5).action(np.zeros(195), state(-1))[0]==E.HOLD

def test_flat_gate_and_threshold_apply_to_joint_and_full():
    joint={'joint':Const([0.,3.,0.,0.], [1.,2.])}
    for variant,pols in [('JOINT16',joint),('FULL',full())]:
        assert C.Policy(variant,pols,.5).action(np.zeros(195),state())[0]==E.LONG
        assert C.Policy(variant,pols,.999).action(np.zeros(195),state())[0]==E.HOLD
        assert C.Policy(variant,pols,.5).action(np.zeros(195),state(allowed=0))[0]==E.HOLD
        assert C.Policy(variant,pols,.5,exit_mode='fixed').action(np.zeros(195),state(1))[0]==E.HOLD

def test_capacity_matches_before_outcomes():
    counts=C.architecture_counts()
    assert counts['JOINT16']['total']==610
    assert counts['FULL']['total']==2236
    assert abs(counts['JOINT_MATCHED']['total']-2236)<10
    assert counts['FULL']['head']==16

def test_schedule_and_business_are_frozen():
    cells=C.cells()
    assert len(cells)==15 and len({x['id'] for x in cells})==15
    assert [x['seed'] for x in cells[::3]]==[20261019,20261020,20261021,20261022,20261023]
    cfg=C.business_config()
    assert cfg.max_entries_per_day==1 and cfg.sizing.leverage==50
    assert cfg.min_orders_per_week==0 and cfg.sizing.rel_volume==.1

def test_threshold_tie_prefers_abstention_without_fake_sharpe():
    rows=[{'tau':.2,'sharpe':None,'entries':0}, {'tau':.9,'sharpe':None,'entries':0}]
    assert C.choose_threshold(rows)['tau']==.9
    rows.append({'tau':.3,'sharpe':.4,'entries':8})
    assert C.choose_threshold(rows)['tau']==.3

def test_exit_target_support_cannot_cross_role_end():
    cx=SimpleNamespace(v=np.arange(10,dtype=np.int64)*3600000000000)
    assert C.label_within_period(cx, (1,1.,.9,1.1,5), 5*3600000000000)
    assert not C.label_within_period(cx, (1,1.,.9,1.1,6), 5*3600000000000)

def test_fixed_and_direction_controls_keep_mandatory_protections():
    c=C.Policy('FULL',full(),.5,direction=-1)
    assert c.action(np.zeros(195),state())[0]==E.HOLD
    c=C.Policy('FULL',full(),.5,exit_mode='fixed')
    a,tp,sl=c.action(np.zeros(195),state())
    assert a==E.LONG and tp>0 and sl>0

def test_invalid_state_rejected():
    with pytest.raises(ValueError,match='STATE'):
        C.Policy('FULL',full(),.5).action(np.zeros(195),state(pos=2))


def test_next_pivot_tp_is_purged_even_when_current_pivot_is_confirmed():
    ds=SimpleNamespace(pivots=[(2,1,1.),(6,-1,1.1)],conf=[3,8])
    cx=SimpleNamespace(ds=ds,v=np.arange(12,dtype=np.int64)*3600000000000)
    assert not C.opening_label_within_period(cx,2,7*3600000000000)
    assert C.opening_label_within_period(cx,2,9*3600000000000)


def test_native_fit_cpu_and_reload_use_mean_monitor(tmp_path):
    import torch
    from app.short_horizon.expert_guided import learn as L, temporal as T
    rng=np.random.default_rng(40)
    x=rng.normal(size=(32,204)).astype(np.float32);y=np.tile([0,1,2,3],8)
    model,hist,info=L.fit(x,y,x[:12],y[:12],seed=7,n_cls=4,batch=16,max_epochs=2,
                        device='cpu',net=lambda:C.network('JOINT16','joint'))
    assert len(hist)==2 and all(abs(h['score']-.5*(h['train_loss']+h['inner_loss']))<1e-12 for h in hist)
    p=tmp_path/'weights.pt';torch.save(model.state_dict(),p)
    clone=C.network('JOINT16','joint');clone.load_state_dict(torch.load(p,weights_only=True))
    for a,b in zip(T.TorchPolicy(model)(x),T.TorchPolicy(clone)(x)):
        assert np.array_equal(a,b)


def test_rollout_business_masks_and_inference_do_not_use_labels():
    from tests.unit_tests.test_expert_guided_env import _env
    bars,cal,env=_env(segments=2,max_entries_per_day=1,min_orders_per_week=0.)
    n=len(bars)
    ds=SimpleNamespace(bars=bars,cal=cal,sigma=env.sigma,sigma_rule=env.sigma_rule)
    cx=SimpleNamespace(ds=ds,rob=np.arange(n),obs_ok=np.ones(n,bool))
    # This attribute raises if inference ever accesses retrospective label eligibility.
    class Forbidden:
        def __getitem__(self,key):raise AssertionError('future label touched in inference')
    cx.lab_ok=Forbidden()
    from unittest.mock import patch
    with patch.dict(C.DG.PERIODS,{'SYNTHETIC':(str(bars.index[0]),str(bars.index[-1]+__import__('pandas').Timedelta(hours=1)))}):
        result,_,trades=C.rollout(cx,'SYNTHETIC',lambda *a:(E.LONG,1.5,2.))
    assert result['max_entries_per_day']<=1 and len(trades)>0
    for t in trades:
        assert t['stop_loss']<t['entry_price']<t['take_profit']
        seg=cal.segments[t['segment']]
        assert t['entry_time']>=seg.operational_start and t['exit_time']<=seg.force_flat
    met=C.scalar_metrics(result,trades)
    assert met['buy_trades']+met['sell_trades']==len(trades)
    assert met['buy_net_usd']+met['sell_net_usd']==pytest.approx(met['net_profit_usd'])
