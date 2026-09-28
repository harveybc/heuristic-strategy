"""Paired conditional price-noise sweeps; execution reused without changes."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from tools.corrected_price_noise_sweep import ROOT, elapsed_targets, masked_mae, run_cell
from tools.price_noise_sweep import OFFSETS, SEEDS, digest, write_json

RATIOS = [0,.05,.1,.15,.2,.3,.4,.5,.6,.75,.9,.95,1,1.05,1.1,1.25,1.5,1.75,2,2.5,3]
SCENARIOS = ["long_naive", "short_naive", "long_ideal", "short_ideal"]


def pair_for(scenario, ratio):
    if scenario not in SCENARIOS or not math.isfinite(ratio) or ratio < 0:
        raise ValueError("Unknown scenario or invalid noise ratio")
    fixed = 1 if scenario.endswith("naive") else 0
    return (fixed, ratio) if scenario.startswith("long") else (ratio, fixed)


def build_design(ratios, seeds):
    if not ratios or len(set(ratios)) != len(ratios) or not {0,1}.issubset(ratios):
        raise ValueError("Unique grid including zero and one required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique seeds required")
    jobs, curves = {}, []
    for scenario in SCENARIOS:
        for ratio in ratios:
            short, long = pair_for(scenario, ratio)
            for seed in seeds:
                cell = f"seed{seed}_c{short:g}_l{long:g}"
                job = dict(cell=cell, seed=seed, short_ratio=short, long_ratio=long)
                jobs.setdefault(cell, job)
                curves.append(dict(job, scenario=scenario, ratio=ratio))
    return list(jobs.values()), curves


def calibrate(naive, draws):
    return {h: naive[h] / (math.fsum(math.fsum(abs(x) for x in z[:,j])/len(z)
                                    for z in draws)/len(draws))
            for h,j in [("6",5),("144",11)]}


def make_predictions(target, z, scales, short_ratio, long_ratio):
    if target.shape != z.shape or target.shape[1] != 12:
        raise ValueError("Twelve matching forecast columns required")
    if not np.isfinite(z).all() or not np.isfinite([short_ratio,long_ratio]).all() or min(short_ratio,long_ratio) < 0:
        raise ValueError("Invalid noise")
    result = target.copy()
    result[:,:6] += scales["6"]*short_ratio*z[:,:6]
    result[:,6:] += scales["144"]*long_ratio*z[:,6:]
    return result


def run(args):
    native = getattr(args,"native_plugin",False)
    execute_cell = run_cell
    if native:
        from tools.native_plugin_noise_sweep import run_cell as execute_cell
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.data, parse_dates=["DATE_TIME"], index_col="DATE_TIME", float_precision="round_trip")
    origins, target, positions = elapsed_targets(frame)
    current = np.repeat(frame.CLOSE.iloc[positions].to_numpy()[:,None],12,axis=1)
    current[~np.isfinite(target)] = np.nan
    naive = dict(zip(map(str,OFFSETS),map(float,masked_mae(current,target))))
    draws = [np.random.default_rng(s).standard_normal(target.shape) for s in SEEDS]
    scales = calibrate(naive,draws)
    ratios, seeds = ([0,1],[42]) if args.pilot else (RATIOS,SEEDS)
    jobs, curves = build_design(ratios,seeds)
    for src,name in [(args.data,"input.csv"),(Path(__file__),"runner.py"),
                     (ROOT/"tools/corrected_price_noise_sweep.py","execution.py"),
                     (ROOT/"tools/price_noise_sweep.py","shared_helpers.py"),
                     (ROOT/"app/policies/prediction_entry_exit.py","legacy_policy.py"),
                     (ROOT/"docs/noise_price_sweep/CONDITIONAL_PLAN.md","PLAN.md")]:
        shutil.copy2(src,out/name)
    np.save(out/"targets.npy",target,allow_pickle=False)
    pd.DataFrame({"DATE_TIME":origins,"source_row":positions}).to_csv(out/"origins.csv",index=False)
    for seed,z in zip(SEEDS,draws):
        np.save(out/f"standard_normal_seed{seed}.npy",z,allow_pickle=False)
    write_json(out/"curves.json",curves)
    if native:
        for src,name in [(ROOT/"tools/native_plugin_noise_sweep.py","native_adapter.py"),
                         (ROOT/"app/plugins/plugin_long_short_predictions.py","native_plugin.py"),
                         (ROOT/"docs/noise_price_sweep/NATIVE_PLUGIN_PLAN.md","NATIVE_PLAN.md")]:
            shutil.copy2(src,out/name)
    manifest = dict(kind="CONDITIONAL_PRICE_NOISE_SWEEP_V4",pilot=args.pilot,
        n_bars=len(frame),n_origins=len(origins),origin_first=str(origins[0]),origin_last=str(origins[-1]),
        liquidation_last=str(frame.index[positions[-1]+2]),horizons_hours=OFFSETS.tolist(),
        n_by_horizon=dict(zip(map(str,OFFSETS),map(int,np.isfinite(target).sum(axis=0)))),
        naive_mae=naive,sigma_by_endpoint=scales,sigma_ref_measured_mean_H144=scales["144"],
        seeds=seeds,ratios=ratios,scenarios=SCENARIOS,
        execution=dict(initial_usd=10000,max_margin_fraction=.05,leverage=100,
                       commission_per_lot_side_usd=7,fixed_friction_price_per_side=.00010,
                       swap_per_lot_day_usd=10,forecast_rule="ORDERED_FIRST_PASSAGE",
                       entry_uses="long family",exit_uses="all available horizons, chronological",
                       ambiguous_bar="SL_FIRST_UNLESS_OPEN_ALREADY_CROSSED_TP"),
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    write_json(out/"manifest.json",manifest)
    if native:
        from app.plugins.plugin_long_short_predictions import Plugin
        manifest["kind"]="CONDITIONAL_NATIVE_PLUGIN_V5"
        manifest["execution"].update(forecast_rule="NATIVE_PLUGIN_NEXT",exit_variant=Plugin.plugin_params["exit_variant"],
            exit_uses="native configurable early exit; TP/SL checked at CLOSE",ambiguous_bar="NO_INTRABAR_BARRIERS",
            plugin_params=dict(Plugin.plugin_params,rel_volume=.05))
        write_json(out/"manifest.json",manifest)
    records = []
    start = time.process_time()
    for job in jobs + [dict(cell="persistence",seed=None,short_ratio=None,long_ratio=None)]:
        if time.process_time()-start > (3600 if native else 900):
            raise RuntimeError("CPU budget exhausted; partial evidence retained")
        pred = current if job["seed"] is None else make_predictions(target,draws[SEEDS.index(job["seed"])],
            scales,job["short_ratio"],job["long_ratio"])
        r = execute_cell(frame,origins,target,positions,pred,out/job["cell"],job["cell"],margin_fraction=.05)
        r.update(job,kind="persistence" if job["seed"] is None else "conditional")
        write_json(out/job["cell"]/"result.json",r)
        records.append(r)
        write_json(out/"results.json",records)
        print(json.dumps({"done":len(records),"total":len(jobs)+1,"cell":r["cell"],"profit":r["profit_usd"]}),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in {"artifacts","mae_by_horizon"}} |
                 {f"mae_h{h}":v for h,v in r["mae_by_horizon"].items()} for r in records]).to_csv(out/"sweep.csv",index=False)
    write_json(out/"completion.json",dict(status="COMPLETE",cells=len(records),
        cpu_seconds=time.process_time()-start,manifest_sha256=digest(out/"manifest.json"),
        results_sha256=digest(out/"results.json"),sweep_sha256=digest(out/"sweep.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=ROOT/"tests/data/phase_2_3_base_d3.csv")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot",action="store_true")
    run(parser.parse_args())
