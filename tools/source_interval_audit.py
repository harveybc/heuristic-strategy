"""Read-only OHLC and return-interval audit of a pinned HistData candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tools.modular_campaign import digest, write

HOUR=pd.Timedelta(hours=1)


def lake_clock_to_utc(labels):
    t=pd.DatetimeIndex(pd.to_datetime(labels))
    wall=t.tz_convert('UTC').tz_localize(None) if t.tz is not None else t
    return (wall+5*HOUR).tz_localize('Europe/London',ambiguous='NaT',nonexistent='NaT').tz_convert('UTC')


def hourly_returns(series):
    if series.index.has_duplicates:raise ValueError('DUPLICATE_SOURCE_TIMESTAMP')
    if not series.index.is_monotonic_increasing:raise ValueError('UNSORTED_SOURCE_TIMESTAMP')
    return np.log(series).diff().where(series.index.to_series().diff().eq(HOUR))


def corr(frame):
    frame=frame.dropna()
    return float(frame.iloc[:,0].corr(frame.iloc[:,1])) if len(frame)>2 else None


def compare_close(admitted,candidate):
    ar,hr=hourly_returns(admitted),hourly_returns(candidate)
    j=pd.concat([admitted.rename('admitted'),candidate.rename('candidate')],axis=1,join='inner').dropna()
    valid=pd.concat([ar.rename('a'),hr.rename('h')],axis=1,join='inner').dropna()
    legacy=pd.concat([np.log(admitted).diff().rename('a'),np.log(candidate).diff().rename('h')],axis=1,join='inner').dropna()
    d=(j.admitted-j.candidate).abs()
    return dict(common_rows=len(j),close_mismatch_rows=int((d>1e-12).sum()),
        missing_in_candidate=len(admitted.index.difference(candidate.index)),
        close_p95_abs_diff_pips=float(d.quantile(.95)/1e-4) if len(d) else None,
        close_max_abs_diff_pips=float(d.max()/1e-4) if len(d) else None,
        legacy_unmatched_interval_return_corr=corr(legacy),consecutive_hour_return_corr=corr(valid),
        consecutive_hour_pairs=len(valid),unequal_return_interval_rows=len(legacy.index.difference(valid.index)))


def audit(admitted_path,hist_path,output):
    admitted_path,hist_path,output=map(Path,(admitted_path,hist_path,output))
    pins={"admitted":digest(admitted_path),"histdata":digest(hist_path)}
    if pins!={'admitted':'c80847bb149c7af1ad50e42301b5cd70608e6c37702ce8cc6a7b9eb70f36d2ca',
              'histdata':'c6628375b2872f16dcdbd6b1b90ed8e68bc5bdd852efdef01f2515b8712d4e2e'}:
        raise ValueError('AUDIT_SOURCE_PIN_MISMATCH')
    start,end=pd.Timestamp('2020-01-01',tz='UTC'),pd.Timestamp('2025-01-01',tz='UTC')
    a=pd.read_parquet(admitted_path,filters=[('timestamp','>=',start),('timestamp','<',end)]).set_index('timestamp').sort_index()
    # Source labels are false-UTC wall time. Exclude even source rows whose actual
    # UTC time could enter 2025, before reading price payloads.
    h=pd.read_parquet(hist_path,filters=[('datetime','>=',start),('datetime','<',end-5*HOUR)])
    h.index=lake_clock_to_utc(h.datetime);h=h[h.index.notna()].sort_index()
    if h.index.has_duplicates:raise ValueError('DUPLICATE_RELOCALIZED_TIMESTAMP')
    h=h[(h.index>=start)&(h.index<end)]
    whole=compare_close(a.BidClose,h.close)
    months={};differences=[]
    for month,group in a.groupby(a.index.strftime('%Y-%m')):
        subset=h[h.index.strftime('%Y-%m')==month]
        months[month]=compare_close(group.BidClose,subset.close)
    common=a.index.intersection(h.index)
    for ac,hc in [('BidOpen','open'),('BidHigh','high'),('BidLow','low'),('BidClose','close')]:
        delta=(a.loc[common,ac]-h.loc[common,hc]).abs()
        for t,v in delta[delta>1e-12].items():
            differences.append(dict(timestamp=t.isoformat(),field=hc,admitted=float(a.loc[t,ac]),
                histdata=float(h.loc[t,hc]),abs_diff_pips=float(v/1e-4),source_id=a.loc[t,'source_id']))
    half=a[(a.index>='2023-01-01')&(a.index<'2023-07-01')]
    shared=half.index.intersection(h.index)
    bad=[r for r in differences if '2023-01-01'<=r['timestamp']<'2023-07-01']
    report=dict(schema='source_interval_audit.v1',scope='DEVELOPMENT_2020_2024',source_pins=pins,
        clock_rule='EU_DST_UTC_MINUS_5 cohort-specific observed identity',whole=whole,monthly=months,
        ohlc_differences=differences,half_2023=dict(admitted_rows=len(half),common_rows=len(shared),
            candidate_missing_rows=len(half.index.difference(h.index)),ohlc_mismatch_cells=len(bad)),
        disposition='REPORTED_2023H1_PRICE_DISAGREEMENT_REFUTED_ON_COMMON_OHLC' if not bad else 'INVESTIGATE_2023H1_PRICE_DIFFERENCES',
        mechanism='Independent diff before alignment compares unequal intervals after missing HistData bars',
        accepted_inputs_changed=False,interpolation=False,test_price_rows_read=False,
        limits='Agreement does not independently certify the common upstream quotes; isolated OHLC mismatches retained for scoped review')
    output.mkdir(parents=True,exist_ok=True);write(output/'SOURCE_AUDIT.json',report)
    write(output/'DATA_GATE.json',dict(source_pins=pins,passed=not bad,
        admitted_identity_unchanged=True,scope='No demonstrated 2023H1 defect in consumed admitted inputs/target'))
    print(json.dumps(dict(disposition=report['disposition'],half_2023=report['half_2023'],
        whole=whole,remaining_ohlc_mismatch_cells=len(differences))))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--admitted',required=True);p.add_argument('--histdata',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.admitted,a.histdata,a.output)

if __name__=='__main__':main()
