"""Finite queue recovery and publication witnesses; no GPU or network."""
import json
from pathlib import Path
import pytest
from tools import modular_campaign as Q


def fixture(tmp_path,monkeypatch):
    cells=Q.schedule()[:3]
    spec=dict(cells=cells,max_attempts=2,spec_sha256='a'*64)
    Q.write(tmp_path/'STATE.json',dict(spec_sha256='a'*64,cells=[],state='READY'))
    monkeypatch.setattr(Q,'load_spec',lambda root:spec)
    monkeypatch.setattr(Q,'schedule',lambda:cells)
    return spec


def close(spec,cell,out):
    result=dict(cell=cell,metrics={'combined':{'TRAIN':{'net_profit_usd':1.}}},
                architecture={'total':610},config_sha256='b'*64,wall_seconds=1.)
    Q.write(out/'RESULT.json',result)
    Q.write(out/'CLOSE.json',dict(cell=cell,spec_sha256=spec['spec_sha256'],files={'RESULT.json':Q.digest(out/'RESULT.json')}))
    return 0


def test_all_cells_advance_and_completed_never_refit(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);calls=[]
    def run(s,c,o):calls.append(c['id']);return close(s,c,o)
    for _ in range(5):Q.work(tmp_path,execute=run,ready=lambda *a:'READY')
    assert len(calls)==3
    assert Q.read(tmp_path/'SCIENTIFIC_COMPLETE.json')['state']=='SCIENTIFIC_COMPLETE'
    assert Q.read(tmp_path/'STATUS.json')['complete']==3


def test_crash_after_result_adopts_and_failures_are_bounded(tmp_path,monkeypatch):
    spec=fixture(tmp_path,monkeypatch)
    # Two attempts fail, then unrelated cells still run automatically.
    def run(s,c,o):return 5 if c==spec['cells'][0] else close(s,c,o)
    for _ in range(6):Q.work(tmp_path,execute=run,ready=lambda *a:'READY')
    state=Q.read(tmp_path/'STATE.json')
    assert state['cells'][0]['state']=='FAILED_EXHAUSTED'
    assert len(state['cells'][0]['attempts'])==2
    assert [r['state'] for r in state['cells'][1:]]==['CLOSED','CLOSED']
    assert Q.read(tmp_path/'SCIENTIFIC_COMPLETE.json')['state']=='COMPLETE_WITH_FAILURES'


def test_resource_wait_does_not_consume_attempt(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    Q.work(tmp_path,execute=lambda *a:pytest.fail('must not fit'),ready=lambda *a:'WAIT_RAM')
    state=Q.read(tmp_path/'STATE.json')
    assert not state['cells'][0]['attempts'] and state['state']=='WAIT_RAM'
    assert Q.read(tmp_path/'STATUS.json')['eta_seconds'] is None


def test_tampered_terminal_rejected(tmp_path):
    cell=Q.schedule()[0];close({'spec_sha256':'a'*64},cell,tmp_path)
    Q.write(tmp_path/'RESULT.json',{'altered':True})
    with pytest.raises(ValueError,match='ARTIFACT'):Q.verify_close(tmp_path,cell,'a'*64)


def test_completed_child_adopted_after_parent_crash(tmp_path,monkeypatch):
    spec=fixture(tmp_path,monkeypatch)
    cell=spec['cells'][0];out=tmp_path/'runs'/cell['id']/'a1';out.mkdir(parents=True)
    close(spec,cell,out)
    Q.write(tmp_path/'STATE.json',dict(spec_sha256='a'*64,state='RUNNING',cells=[
        dict(cell=cell,state='RUNNING',attempts=[dict(out=str(out),state='RUNNING')])]))
    calls=[]
    Q.work(tmp_path,execute=lambda s,c,o:(calls.append(c),close(s,c,o))[1],ready=lambda *a:'READY')
    assert calls==[spec['cells'][1]]
    assert Q.read(tmp_path/'STATE.json')['cells'][0]['state']=='CLOSED'


def test_atomic_json_rejects_unserializable_without_leaving_temp(tmp_path):
    p=tmp_path/'result.json';Q.write(p,{'valid':1})
    with pytest.raises(TypeError):Q.write(p,{'bad':object()})
    assert Q.read(p)=={'valid':1} and list(tmp_path.iterdir())==[p]


def test_aggregation_uses_shared_seeds_not_best_seed():
    from tools.modular_analysis import aggregate
    def result(arm,seed,profit):
        return dict(cell=dict(arm=arm,seed=seed,id=arm+str(seed)),metrics={'combined':{
            'TRAIN':{'net_profit_usd':0.,'mean_daily_log_growth':.002},
            'OBS_2024':{'net_profit_usd':profit,'mean_daily_log_growth':.001}}})
    out=aggregate([result('JOINT16',20261019,-10),result('FULL',20261019,20),
                   result('FULL',20261022,-100)])
    both=next(g for g in out['groups'] if g['arm']=='FULL' and g['period']=='OBS_2024' and g['cohort']=='all_five')
    assert both['metrics']['net_profit_usd']['mean']==-40
    delta=next(r for r in out['paired_differences'] if r['period']=='OBS_2024')
    assert delta['metrics']['net_profit_usd']==30
    assert len(out['paired_differences'])==2
