import csv
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

P=Path(__file__).parents[2]/'app/policies/business_prediction_csv.py'


def module():
    spec=importlib.util.spec_from_file_location('business_csv',P)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(path,hs,**changes):
    row=dict(DATE_TIME='2024-01-08T12:00:00+00:00',origin_price='1.1',model_id='checkpoint')
    for h in hs:row.update({f'prediction_h{h}':'1.101',f'available_h{h}':'1'})
    row.update(changes)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    return row


def test_exact_origin_price_alignment_and_no_fill(tmp_path):
    m=module();a=tmp_path/'s.csv';b=tmp_path/'l.csv'
    write(a,m.FAMILIES['short'],prediction_h2='',available_h2='0');write(b,m.FAMILIES['long'])
    t=datetime(2024,1,8,12,tzinfo=timezone.utc)
    pair=m.load_pair(a,b,{t:1.1})
    assert pair['short'][t][2] is None and pair['long'][t][18]==1.101
    assert t-timedelta(hours=1) not in pair['short']
    with pytest.raises(ValueError,match='ORIGIN_PRICE_MISMATCH'):m.load_pair(a,b,{t:1.2})


@pytest.mark.parametrize('changes',[
    {'DATE_TIME':'2024-01-08T12:00:00'}, {'DATE_TIME':'2024-01-08T12:30:00+00:00'},
    {'available_h2':'true'}, {'available_h2':'0'}, {'prediction_h2':'nan'},
    {'prediction_h2':''}, {'model_id':''}])
def test_malformed_rows_are_refused(tmp_path,changes):
    m=module();p=tmp_path/'s.csv';write(p,m.FAMILIES['short'],**changes)
    with pytest.raises(ValueError):m.load_family(p,'short')


def test_duplicate_origins_and_missing_horizon_refused(tmp_path):
    m=module();p=tmp_path/'s.csv';write(p,m.FAMILIES['short'])
    p.write_text(p.read_text()+p.read_text().splitlines()[-1]+'\n')
    with pytest.raises(ValueError,match='DUPLICATE_OR_UNSORTED'):m.load_family(p,'short')
    write(p,(2,3,4,5,6))
    with pytest.raises(ValueError,match='CSV_COLUMNS'):m.load_family(p,'short')


def test_market_gate_forces_close_without_predictions_and_reopens_after_12h():
    m=module();o=datetime(2024,1,7,22,tzinfo=timezone.utc);c=o+timedelta(days=5)
    k=o+timedelta(days=3)
    assert m.market_gate(o+timedelta(hours=11),o,c,k,False)=='NO_ENTRY'
    assert m.market_gate(o+timedelta(hours=12),o,c,k,False)=='ENTRY_ALLOWED'
    assert m.market_gate(k,o,c,k,False)=='NO_ENTRY'
    assert m.market_gate(c-timedelta(hours=12),o,c,k,True)=='FORCE_CLOSE'
    assert m.market_gate(c+timedelta(hours=24),o,c,k,True)=='FORCE_CLOSE'
