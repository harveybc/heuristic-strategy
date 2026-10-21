"""Read authenticated closed learning curves without refitting or reselecting weights."""
import argparse
import json
from pathlib import Path
from tools.modular_campaign import read, write, digest


def audit(root,out):
    root,out=Path(root),Path(out);rows=[]
    for run in sorted((root/'runs').glob('full-*')):
        close=read(run/'CLOSE.json')
        for f in sorted(run.glob('round3-*.json')):
            if digest(f)!=close['files'][f.name]:raise ValueError('CURVE_IDENTITY_MISMATCH')
            body=read(f);history=body['history'];best=min(history,key=lambda h:h['score'])
            inner=min(history,key=lambda h:h['inner_loss'])
            if best['epoch']!=body['info']['best_epoch']:raise ValueError('CURVE_CHECKPOINT_MISMATCH')
            rows.append(dict(seed=close['cell']['seed'],module=body['module'],curve_sha256=digest(f),
                selected_epoch=best['epoch'],inner_minimum_epoch=inner['epoch'],
                selected_train_loss=best['train_loss'],selected_inner_loss=best['inner_loss'],
                minimum_inner_loss=inner['inner_loss'],inner_loss_difference=best['inner_loss']-inner['inner_loss'],
                selected_gap=best['gap_inner_minus_train'],minimum_inner_epoch_gap=inner['gap_inner_minus_train'],
                stop_reason=body['info']['stop_reason'],epochs=body['info']['epochs']))
    report=dict(schema='monitor_curve_audit.v1',modules=len(rows),
        selected_later_than_inner_minimum=sum(r['selected_epoch']>r['inner_minimum_epoch'] for r in rows),
        rows=rows,refit=False,checkpoint_reselected=False,
        interpretation='Checkpoint objective tradeoff observed; not proof of effect on full-year trading')
    write(out/'CURVE_AUDIT.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=audit(a.root,a.out);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))

if __name__=='__main__':main()
