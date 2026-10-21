import pytest
from tools.modular_campaign import write,digest
from tools.monitor_curve_audit import audit


def test_curve_tradeoff_and_integrity(tmp_path):
    root=tmp_path/'runs/full-1';root.mkdir(parents=True)
    curve=root/'round3-open_l.json'
    write(curve,dict(module='open_l',info=dict(best_epoch=2,stop_reason='patience',epochs=2),history=[
        dict(epoch=1,score=.5,train_loss=.4,inner_loss=.6,gap_inner_minus_train=.2),
        dict(epoch=2,score=.45,train_loss=.2,inner_loss=.7,gap_inner_minus_train=.5)]))
    write(root/'CLOSE.json',dict(cell=dict(seed=1),files={curve.name:digest(curve)}))
    r=audit(tmp_path,tmp_path/'analysis')
    assert r['selected_later_than_inner_minimum']==1
    assert r['rows'][0]['inner_loss_difference']==pytest.approx(.1)
    write(curve,{'tampered':True})
    with pytest.raises(ValueError,match='IDENTITY'):audit(tmp_path,tmp_path/'analysis')
