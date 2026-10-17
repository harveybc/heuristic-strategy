"""Strict origin-aligned price forecasts for the mandatory business policy.

This is NOT the positional legacy CSV loader. Unavailable heads remain None;
there is no shifting, interpolation, forward fill or fallback price. The
caller must apply market_gate on every observed bar, even without forecasts.
"""
import csv
import math
from datetime import datetime, timedelta, timezone

SCHEMA='heuristic_strategy.business_prediction_csv.v1'
FAMILIES={'short':(2,3,4,5,6,12),'long':(18,24,36,48,60,72)}


def instant(text):
    t=datetime.fromisoformat(text.replace('Z','+00:00'))
    if t.tzinfo is None or t.utcoffset() is None or t.minute or t.second or t.microsecond:
        raise ValueError('INVALID_ORIGIN_TIME')
    return t.astimezone(timezone.utc)


def price(text):
    value=float(text)
    if not math.isfinite(value) or value<=0:raise ValueError('INVALID_PRICE')
    return value


def load_family(path,family):
    hs=FAMILIES[family];expected={'DATE_TIME','origin_price','model_id'}
    expected.update(f'{kind}_h{h}' for h in hs for kind in ('prediction','available'))
    out={};previous=None
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames)!=len(expected) or set(reader.fieldnames)!=expected:
            raise ValueError('CSV_COLUMNS')
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('CSV_ROW_SHAPE')
            t=instant(row['DATE_TIME'])
            if previous is not None and t<=previous:raise ValueError('DUPLICATE_OR_UNSORTED_ORIGIN')
            previous=t
            if not row['model_id'].strip():raise ValueError('MISSING_CHECKPOINT_ID')
            values={}
            for h in hs:
                flag=row[f'available_h{h}'];v=row[f'prediction_h{h}']
                if flag not in ('0','1') or (flag=='0' and v!='') or (flag=='1' and v==''):
                    raise ValueError('INVALID_HEAD_AVAILABILITY')
                values[h]=price(v) if flag=='1' else None
            out[t]=dict(predictions=values,origin_price=price(row['origin_price']),model_id=row['model_id'])
    if not out:raise ValueError('EMPTY_FORECAST_CSV')
    return out


def load_pair(short_path,long_path,observed_close):
    families={name:load_family(path,name) for name,path in (('short',short_path),('long',long_path))}
    for rows in families.values():
        for t,row in rows.items():
            if t not in observed_close:raise ValueError('ORIGIN_NOT_OBSERVED')
            if row['origin_price']!=observed_close[t]:raise ValueError('ORIGIN_PRICE_MISMATCH')
    return {name:{t:r['predictions'] for t,r in rows.items()} for name,rows in families.items()}


def market_gate(t,market_open,market_close,entry_cutoff,position_open,*,
                entry_delay_after_open_hours=12,force_flat_before_close_hours=12):
    """Calendar decision precedes prediction use; a position never waits for a CSV."""
    instants=(t,market_open,market_close,entry_cutoff)
    if any(v.tzinfo is None or v.utcoffset() is None for v in instants):raise ValueError('NAIVE_CALENDAR')
    offsets=(entry_delay_after_open_hours,force_flat_before_close_hours)
    if any(type(v) is not int or v<0 for v in offsets):raise ValueError('INVALID_BUSINESS_OFFSETS')
    a=market_open+timedelta(hours=entry_delay_after_open_hours)
    b=market_close-timedelta(hours=force_flat_before_close_hours)
    if not market_open<a<b<market_close:raise ValueError('INVALID_OPERATIONAL_WEEK')
    if position_open and t>=b:return 'FORCE_CLOSE'
    if position_open:return 'MANAGE' if a<=t<b else 'NO_SIGNAL'
    return 'ENTRY_ALLOWED' if a<=t<min(entry_cutoff,b) else 'NO_ENTRY'
