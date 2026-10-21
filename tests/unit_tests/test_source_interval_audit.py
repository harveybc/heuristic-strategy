import numpy as np
import pandas as pd
import pytest
from tools import source_interval_audit as A


def test_hole_does_not_mean_a_different_price_or_hourly_return():
    idx=pd.date_range('2023-03-06',periods=120,freq='h',tz='UTC')
    rng=np.random.default_rng(1)
    a=pd.Series(np.exp(np.cumsum(rng.normal(0,.001,len(idx)))),index=idx)
    h=a.drop(idx[[10,20,21,40,50]])
    result=A.compare_close(a,h)
    assert result['close_mismatch_rows']==0
    assert result['missing_in_candidate']==5
    assert result['unequal_return_interval_rows']>0
    assert result['consecutive_hour_return_corr']==pytest.approx(1.)
    assert result['legacy_unmatched_interval_return_corr']<.999999


def test_real_price_error_is_reported_and_duplicates_refused():
    idx=pd.date_range('2023-03-01',periods=100,freq='h',tz='UTC')
    a=pd.Series(1+np.arange(100)*1e-4,index=idx);b=a.copy();b.iloc[40]+=.005
    assert A.compare_close(a,b)['close_mismatch_rows']==1
    with pytest.raises(ValueError,match='DUPLICATE'):
        A.hourly_returns(pd.concat([a,a.iloc[:1]]))


def test_clock_conversion_europe_us_dst_mismatch():
    wall=pd.to_datetime(['2023-03-15 08:00','2023-03-29 08:00'])
    utc=A.lake_clock_to_utc(wall)
    assert list(utc.hour)==[13,12]
    ny=pd.DatetimeIndex(wall).tz_localize('America/New_York').tz_convert('UTC')
    assert utc[0]!=ny[0] and utc[1]==ny[1]


def test_shifted_series_fails_exact_price_alignment():
    idx=pd.date_range('2023-01-01',periods=100,freq='h',tz='UTC')
    a=pd.Series(np.linspace(1,1.1,100),index=idx);b=a.copy();b.index+=pd.Timedelta(hours=1)
    assert A.compare_close(a,b)['close_mismatch_rows']==99


def test_future_perturbation_keeps_past_hourly_returns():
    idx=pd.date_range('2023-01-01',periods=100,freq='h',tz='UTC')
    a=pd.Series(np.linspace(1,1.1,100),index=idx);b=a.copy();b.iloc[50:]*=3
    assert np.array_equal(A.hourly_returns(a).iloc[1:50],A.hourly_returns(b).iloc[1:50])
