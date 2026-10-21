import numpy as np
import pytest
from app.short_horizon.expert_guided import learn as L
from tools import modular_campaign as Q, modular_controlled as C


def test_penalized_monitor_and_reference():
    assert L.monitor_value(.4,.6,0.)==pytest.approx(.5)
    assert L.monitor_value(.4,.6,.25)==pytest.approx(.55)
    assert L.monitor_value(.6,.4,.25)==pytest.approx(.5)
    for bad in (-1,float('nan'),True):
        with pytest.raises(ValueError):L.monitor_value(.4,.6,bad)


def test_native_fit_monitor_checkpoint_and_reload():
    import torch
    x=np.random.default_rng(42).normal(size=(32,204)).astype(np.float32);y=np.tile([0,1],16)
    for penalty in (0.,.25):
        model,hist,info=L.fit(x,y,x[:12],y[:12],seed=7,n_cls=2,batch=16,max_epochs=3,
            device='cpu',gap_penalty=penalty,net=lambda:C.network('FULL_MEAN','open_l',head=16))
        assert all(h['score']==pytest.approx(L.monitor_value(h['train_loss'],h['inner_loss'],penalty)) for h in hist)
        assert info['best_epoch']==min(hist,key=lambda h:h['score'])['epoch']


def test_two_selectors_five_shared_seeds_and_identical_architecture():
    cs=Q.schedule('monitor_gap')
    assert len(cs)==10
    assert {c['arm'] for c in cs}=={'FULL_MEAN','FULL_GAP'}
    assert [c['seed'] for c in cs[::2]]==[20261019,20261020,20261021,20261022,20261023]
    assert C.modules('FULL_MEAN')==C.modules('FULL_GAP')==C.modules('FULL')
    assert sum(p.numel() for p in C.network('FULL_MEAN','open_l').parameters())==576


def test_zero_penalty_preserves_predecessor_training_bitwise():
    import subprocess,types,torch
    old=types.ModuleType('previous_native_learn')
    text=subprocess.check_output(['git','show','f6ddd53:app/short_horizon/expert_guided/learn.py'],text=True)
    exec(compile(text,'previous_native_learn.py','exec'),old.__dict__)
    rng=np.random.default_rng(32);x=rng.normal(size=(32,204)).astype(np.float32);y=np.tile([0,1],16)
    reg=np.ones((32,2),np.float32);mask=np.ones(32,np.float32)
    args=dict(seed=8,n_cls=2,batch=16,max_epochs=3,device='cpu',R=reg,Rm=mask,
              Ri=reg[:12],Rmi=mask[:12],net=lambda:C.network('FULL','open_l'))
    first,h1,i1=old.fit(x,y,x[:12],y[:12],**args)
    second,h2,i2=L.fit(x,y,x[:12],y[:12],gap_penalty=0.,**args)
    assert h1==h2
    assert i1['best_epoch']==i2['best_epoch']
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())


def test_source_gate_is_required_and_digest_bound(tmp_path):
    p=tmp_path/'gate.json'
    Q.write(p,dict(passed=True,admitted_identity_unchanged=True,source_pins={'admitted':'a'*64}))
    spec=dict(study='monitor_gap',data_gate=str(p),data_gate_sha256=Q.digest(p),
              admitted=str(tmp_path),inputs={str(tmp_path/'eurusd.parquet'):'a'*64})
    Q.data_gate(spec)
    Q.write(p,{'passed':True})
    with pytest.raises(ValueError,match='IDENTITY'):Q.data_gate(spec)
