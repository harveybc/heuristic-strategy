"""Derived summaries of authenticated specialist results; never selects a model."""
import statistics

METRICS=('net_profit_usd','profit_pct','sharpe_daily_sqrt365','max_drawdown_pct',
         'n_entries','trades','win_rate','profit_factor','mean_daily_log_growth')


def aggregate(results):
    rows=[]
    for result in results:
        for control,periods in result['metrics'].items():
            for period,values in periods.items():
                rows.append(dict(arm=result['cell']['arm'],seed=result['cell']['seed'],
                                 control=control,period=period,metrics=values))
    groups=[]
    for arm,control,period in sorted({(r['arm'],r['control'],r['period']) for r in rows}):
        for cohort in ('original_three','new_two','all_five'):
            subset=[r for r in rows if (r['arm'],r['control'],r['period'])==(arm,control,period)
                and (cohort=='all_five' or (r['seed']<=20261021)==(cohort=='original_three'))]
            if not subset:continue
            measures={}
            for key in METRICS:
                vals=[r['metrics'].get(key) for r in subset if type(r['metrics'].get(key)) in (int,float)]
                measures[key]=dict(n=len(vals),mean=statistics.mean(vals) if vals else None,
                    std=statistics.stdev(vals) if len(vals)>1 else None,minimum=min(vals) if vals else None,
                    maximum=max(vals) if vals else None)
            groups.append(dict(arm=arm,control=control,period=period,cohort=cohort,
                               seeds=[r['seed'] for r in subset],metrics=measures))
    pairs=[]
    reference='FULL_MEAN' if any(r['arm']=='FULL_MEAN' for r in rows) else 'JOINT16'
    baseline={(r['seed'],r['period']):r for r in rows if r['arm']==reference and r['control']=='combined'}
    for r in rows:
        if r['arm']==reference or r['control']!='combined':continue
        base=baseline.get((r['seed'],r['period']))
        if base is None:continue
        delta={k:r['metrics'][k]-base['metrics'][k] for k in METRICS
            if type(r['metrics'].get(k)) in (int,float) and type(base['metrics'].get(k)) in (int,float)}
        pairs.append(dict(arm=r['arm'],seed=r['seed'],period=r['period'],difference_vs=reference,metrics=delta))
    gaps=[]
    for r in results:
        for control,periods in r['metrics'].items():
            tr=periods.get('TRAIN',{}).get('mean_daily_log_growth')
            va=periods.get('OBS_2024',{}).get('mean_daily_log_growth')
            gaps.append(dict(**r['cell'],control=control,train_minus_2024_daily_growth=
                             tr-va if tr is not None and va is not None else None))
    return dict(schema='modular_analysis.v1',completed_fits=len(results),reference=reference,groups=groups,paired_differences=pairs,
                daily_growth_gaps=gaps,scope='DEVELOPMENT; seeds are not independent market histories')


def markdown(summary):
    lines=['# Controlled modular specialists','',summary['scope'],'',
           'Partial until all declared fits close. Profit starts from USD 10000; Sharpe is daily annualized sqrt365.',
           'No-trade Sharpe is null. All seeds retained; no best-seed selection.','',
           '| Arm/control | Period | Cohort | Seeds | Profit USD | Sharpe | Max DD % |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    def fmt(v):return 'null' if v is None else f'{v:.4f}'
    for g in summary['groups']:
        m=g['metrics']
        lines.append(f"| {g['arm']}/{g['control']} | {g['period']} | {g['cohort']} | {len(g['seeds'])} | "
                     f"{fmt(m['net_profit_usd']['mean'])} | {fmt(m['sharpe_daily_sqrt365']['mean'])} | {fmt(m['max_drawdown_pct']['mean'])} |")
    return '\n'.join(lines)+'\n'
