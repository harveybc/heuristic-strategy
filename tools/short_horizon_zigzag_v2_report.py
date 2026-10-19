#!/usr/bin/env python3
"""ZigZag NSGA-II v2 finalize + report (DEVELOPMENT).

  evaluate  read every chain's checkpoint, form the union front per optimization, report bound hits,
            read 2023 (JOINT development) for the fronts and 2024 (PREVIOUSLY OBSERVED development) for
            the chosen points only, with a read ledger; (b) Sharpe/lag for every A4 front point; fill-path
            cost (b+2 vs b+1); v1 vs v2 confirmation lag; demonstrations v2 -> <run>/FINAL.json
  report    figures (Pareto fronts with hypervolume convergence insets) and tables -> docs/.../zigzag/v2
Run `evaluate` in chunks with --only to stay inside the bounded-CPU rule.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon import zigzag_opt as ZO  # noqa: E402
from tools.short_horizon_oracle import load_inputs  # noqa: E402
from tools.short_horizon_price_path_report import GRID, INK, PURPLE, RUST, TEAL, es  # noqa: E402
from tools.short_horizon_zigzag_nsga2 import OPTS, space_for  # noqa: E402

V1 = {"depth": 3, "deviation_pips": 36.3, "deviation_points": 363.0, "backstep": 2}
K_GRID = (0.5, 1.0, 1.5, 2.0)
SEEDS = (1, 2, 3)


def nondominated(rows, nobj):
    keep = []
    for r in rows:
        o = ZO.to_min(r["obj"])
        dom = False
        for q in rows:
            if q is r:
                continue
            oq = ZO.to_min(q["obj"])
            if all(a <= b for a, b in zip(oq, o)) and any(a < b for a, b in zip(oq, o)):
                dom = True
                break
        if not dom:
            keep.append(r)
    return keep


def load_chains(run, opt, wide=False):
    kind, cap = OPTS[opt]
    space = space_for(kind, wide)
    nobj = 2 if kind == "A" else 3
    base = Path(run) / (opt + ("_wide" if wide else ""))
    chains, rows = [], []
    for seed in SEEDS:
        ck = base / f"seed{seed}" / "checkpoint.json"
        if not ck.exists():
            continue
        st = json.loads(ck.read_text())
        done = json.loads((base / f"seed{seed}" / "DONE").read_text()) if (base / f"seed{seed}" / "DONE").exists() else None
        hv = st["hv"]
        last20 = abs(hv[-1] - hv[-21]) / hv[-21] if len(hv) > 21 and hv[-21] > 0 else math.nan
        chains.append({"seed": seed, "generations": st["generation"], "hv": hv, "hv_change_last20": last20,
                       "evals": st["evals"], "sims": st["sims"], "done": done})
        seed_rows = []
        for g, f in zip(st["population"], st["fitness"]):
            p, _ = space.decode(g)
            seed_rows.append({"p": p, "obj": list(f), "seed": seed})
        for r in nondominated(seed_rows, nobj):
            r["seed_front"] = True
            rows.append(r)
    uniq = {}
    for r in rows:
        uniq.setdefault(json.dumps(r["p"], sort_keys=True), r)
    union = nondominated(list(uniq.values()), nobj)
    # Many parameter sets give the SAME objective vector (MetaTrader's Deviation is nearly inert: it only
    # gates the current bar's distance to the Depth-window extreme). Collapse ties to one representative
    # and report a bound hit only when EVERY tied parameter set sits on that bound.
    groups = {}
    for r in union:
        groups.setdefault(tuple(round(x, 6) for x in r["obj"]), []).append(r)
    reps = []
    for _, rs in sorted(groups.items()):
        rs.sort(key=lambda r: (r["p"]["depth"], r["p"]["backstep"], r["p"]["deviation_pips"], r["p"].get("k_tp", 0.0)))
        rep = dict(rs[0])
        common = set(space.on_bound(rs[0]["p"]))
        for r in rs[1:]:
            common &= set(space.on_bound(r["p"]))
        rep["on_bound"] = [h for h in space.on_bound(rs[0]["p"]) if h in common]
        rep["n_tied"] = len(rs)
        rep["tied_ranges"] = {k: [min(r["p"][k] for r in rs), max(r["p"][k] for r in rs)]
                              for k in ("depth", "deviation_pips", "backstep", "k_tp") if k in rs[0]["p"]}
        reps.append(rep)
    return chains, reps, space


def evaluate(args):
    run = Path(args.run)
    out = run / "FINAL.json"
    final = json.loads(out.read_text()) if out.exists() else {}
    led = run / "READ_LEDGER.jsonl"
    bars, cal, digest = load_inputs(args.data)
    ev = ZO.Evaluator(bars, cal)
    t0 = time.process_time()
    for opt in args.only:
        wide = opt.endswith("_wide")
        base_opt = opt.replace("_wide", "")
        kind, cap = OPTS[base_opt]
        chains, union, space = load_chains(run, base_opt, wide)
        if not chains:
            continue
        pts = []
        ZO.ledger(led, "VALIDATION_2023", "JOINT_DEVELOPMENT", f"{opt} union front validation", n_points=len(union))
        for r in union:
            p = r["p"]
            if kind == "A":
                d, _ = ev.eval_A(p, "DECISION_2020_2022", cap=cap)
                v, _ = ev.eval_A(p, "VALIDATION_2023", cap=cap)
            else:
                d, _ = ev.eval_B(p, "DECISION_2020_2022", cap=cap)
                v, _ = ev.eval_B(p, "VALIDATION_2023", cap=cap)
            row = {"p": p, "obj": r["obj"], "seed": r["seed"], "on_bound": r["on_bound"], "dec": d, "val": v,
                   "n_tied": r["n_tied"], "tied_ranges": r["tied_ranges"]}
            if base_opt == "A4":
                ks = {k: ev.eval_B(p, "DECISION_2020_2022", k_tp=k)[0] for k in K_GRID}
                okk = {k: m for k, m in ks.items() if m["valid"]} or ks
                kb = max(okk, key=lambda k: okk[k]["sharpe"])
                vb, _ = ev.eval_B(p, "VALIDATION_2023", k_tp=kb)
                row["b"] = {"k_tp": kb, "dec": ks[kb], "val": vb}
            pts.append(row)
        acc = [x for x in pts if x["dec"]["valid"] and x["val"]["valid"]]
        chosen = max(acc or pts, key=lambda x: x["val"]["sharpe"])
        ZO.ledger(led, "OBSERVED_2024", "PREVIOUSLY_OBSERVED_DEVELOPMENT", f"{opt} chosen point, single read", n_points=1)
        if kind == "A":
            obs, _ = ev.eval_A(chosen["p"], "OBSERVED_2024", cap=cap)
            lev = {L: ev.eval_A(chosen["p"], "OBSERVED_2024", cap=cap, leverage=float(L))[0]["sharpe"] for L in (10, 50)}
        else:
            obs, _ = ev.eval_B(chosen["p"], "OBSERVED_2024", cap=cap)
            lev = {}
        final[opt] = {"chains": chains, "front": pts, "chosen": {"p": chosen["p"], "dec": chosen["dec"], "val": chosen["val"],
                                                              "obs2024": obs, "obs2024_leverage_sharpe": lev},
                      "bound_hits": [x["on_bound"] for x in pts if x["on_bound"]],
                      "space": {"lower": space.lower(), "upper": space.upper()}}
        out.write_text(json.dumps(final, indent=1, default=str))
        print(json.dumps({"opt": opt, "front": len(pts), "bound_hits": final[opt]["bound_hits"], "chosen": chosen["p"],
                          "cpu": round(time.process_time() - t0, 1)}))


def extras(args):
    """Fill-path cost, v1 vs v2 confirmation lag, demonstrations v2."""
    run = Path(args.run)
    final = json.loads((run / "FINAL.json").read_text())
    led = run / "READ_LEDGER.jsonl"
    bars, cal, digest = load_inputs(args.data)
    ev = ZO.Evaluator(bars, cal)
    h, l = ev.h, ev.l
    rows = []
    A4, B4 = pick(final, "A4"), pick(final, "B4")
    b4 = final[B4]["chosen"]["p"]
    a4 = final[A4]["chosen"]["p"]
    a4_kb = next(x["b"]["k_tp"] for x in final[A4]["front"] if x["p"] == a4)
    ZO.ledger(led, "VALIDATION_2023", "JOINT_DEVELOPMENT", "fill-path cost on chosen points and v1", n_points=3)
    for name, p, k in (("B4 chosen", b4, None), ("A4 chosen, its (b)", a4, a4_kb), ("v1 expert, its (b)", V1, 2.0)):
        for period in ("DECISION_2020_2022", "VALIDATION_2023"):
            new, _ = ev.eval_B(p, period, k_tp=k)
            bar2, _ = ev.eval_B(p, period, k_tp=k, entry_delay=1)          # executable blocks, extra bar
            old, _ = ev.eval_B(p, period, k_tp=k, executable=False)        # pinned v1 path: b+2, no reopen rule
            rows.append({"case": name, "period": period, "p": p, "k_tp": k if k is not None else p.get("k_tp"),
                         "sharpe_b1": new["sharpe"], "sharpe_b2_same_rules": bar2["sharpe"], "sharpe_v1_path": old["sharpe"],
                         "net_b1": new["net"], "net_b2_same_rules": bar2["net"], "net_v1_path": old["net"],
                         "trades_b1": new["trades"], "trades_v1_path": old["trades"],
                         "refused_reopen_b1": new["refused_before_operational_start"]})
    # confirmation lag v1 definition vs v2 (knowable) on the v1 parameters, 2020-2022 pivots
    zp = Z.Params(3, 36.3, 2)
    piv = Z.hindsight_pivots(h, l, zp)
    a, b = (pd.Timestamp(x, tz="UTC").value for x in ZO.PERIODS["DECISION_2020_2022"])
    idx = ev.idx_ns
    sel = [j for j, (i, _, _) in enumerate(piv) if a <= idx[i] < b]
    fast = Z.confirmation_bars_fast(h, l, piv, zp)
    nxt_def = [piv[j + 1][0] + zp.backstep if j + 1 < len(piv) else -1 for j in range(len(piv))]
    lag_v2 = [fast[j] - piv[j][0] for j in sel if fast[j] >= 0]
    lag_v1 = [nxt_def[j] - piv[j][0] for j in sel if nxt_def[j] >= 0]
    lags = {"v1_definition_next_final_pivot_plus_backstep": {"median": float(np.median(lag_v1)), "mean": float(np.mean(lag_v1)),
                                                              "p90": float(np.quantile(lag_v1, 0.9))},
            "v2_knowable_first_opposite_candidate_plus_backstep": {"median": float(np.median(lag_v2)), "mean": float(np.mean(lag_v2)),
                                                                   "p90": float(np.quantile(lag_v2, 0.9))}, "pivots": len(sel)}
    # demonstrations v2 (chosen A4 expert, executable path)
    ZO.ledger(led, "VALIDATION_2023", "JOINT_DEVELOPMENT", "demonstrations v2 trades 2023", n_points=1)
    ZO.ledger(led, "OBSERVED_2024", "PREVIOUSLY_OBSERVED_DEVELOPMENT", "demonstrations v2 trades 2024", n_points=1)
    trades = []
    for period in ZO.PERIODS:
        _, res = ev.eval_A(a4, period)
        trades += res.trades
    man = Z.write_demonstrations(run / "demonstrations_v2", trades, Z.Params(a4["depth"], a4["deviation_pips"], a4["backstep"]),
                                 digest, {"profile": "RealisticProfile with entry_delay_bars = 0 on the executable block path "
                                                     "(decision after the completed pivot bar, fill at the next open); ASSUMED_NOT_BROKER_VERIFIED",
                                          "sizing": "rel_volume 0.10, percentual_order_size 1.0, leverage 100", "sl": "entry -/+ 2 x causal 24 h vol",
                                          "tp": "next hindsight pivot level -/+ 1 pip", "daily_cap": 4},
                                 extra={"version": "v2", "supersedes": "v1 sha256 352ddf6a55300864e11abe790b8bad396742abe4fd4d2aaef1012127ba544f76 (kept)",
                                        "deviation_points": a4["deviation_points"], "pip": ZO.PIP, "points_per_pip": ZO.POINTS_PER_PIP,
                                        "periods": {k: [v, ZO.PERIOD_STATUS[k]] for k, v in ZO.PERIODS.items()},
                                        "data_file": "market_data/forex/g10/eurusd/1h.parquet",
                                        "calendar": "config/short_horizon/eurusd_h1_calendar.dev.json"})
    final["extras"] = {"fill_path_cost": rows, "confirmation_lag_v1_vs_v2": lags, "demonstrations_v2": man}
    (run / "FINAL.json").write_text(json.dumps(final, indent=1, default=str))
    print(json.dumps({"lags": lags, "demo_sha": man["sha256"], "rows": man["rows"]}))


def pick(final, opt):
    """The widened rerun supersedes the first run of the same optimization when it exists."""
    return opt + "_wide" if (opt + "_wide") in final else opt


# ------------------------------------------------------------------ report
CAP_COL = {1: PURPLE, 2: "#1f4e9c", 4: TEAL}


def figure(final, pub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(22, 7.4), facecolor="white")
    ax = axs[0]
    for opt0, cap in (("A1", 1), ("A2", 2), ("A4", 4)):
        opt = pick(final, opt0)
        if opt not in final:
            continue
        f = final[opt]["front"]
        xs = [x["dec"]["max_dd_pct"] for x in f]
        ys = [x["dec"]["sharpe"] for x in f]
        ax.scatter(xs, ys, color=CAP_COL[cap], s=55, edgecolors=INK, linewidths=0.5, zorder=4,
                   label=f"(A) tope {cap} órdenes/día: {len(f)} vectores del frente" + (" (cotas ampliadas)" if opt.endswith("_wide") else ""))
        hit = [(x["dec"]["max_dd_pct"], x["dec"]["sharpe"]) for x in f if x["on_bound"]]
        if hit:
            ax.scatter(*zip(*hit), marker="x", color=RUST, s=90, zorder=5, label="punto sobre una cota" if cap == 4 else None)
        c = final[opt]["chosen"]
        ax.scatter([c["dec"]["max_dd_pct"]], [c["dec"]["sharpe"]], marker="*", s=300, facecolors="none", edgecolors=CAP_COL[cap],
                   linewidths=2, zorder=6)
    ax.set_xlabel("Máxima caída (% del capital) · 2020-2022", color=INK)
    ax.set_ylabel("Sharpe anualizado · 2020-2022", color=INK)
    ax.set_title("(A) Experto con retrospectiva: frentes por tope diario (estrella = elegido por 2023)", fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, edgecolor="#d0d0d0")
    ins = ax.inset_axes([0.08, 0.08, 0.38, 0.32])
    for opt0, cap in (("A1", 1), ("A2", 2), ("A4", 4)):
        for ch in final.get(pick(final, opt0), {}).get("chains", []):
            hv = np.array(ch["hv"], float)
            ins.plot(hv / hv[-1], color=CAP_COL[cap], linewidth=0.9)
    ins.axvline(80, color=RUST, linestyle=":", linewidth=1)
    ins.set_title("hipervolumen / final por generación", fontsize=7)
    ins.tick_params(labelsize=6)
    # B
    ax = axs[1]
    B4 = pick(final, "B4")
    if B4 in final:
        f = final[B4]["front"]
        sc = ax.scatter([x["dec"]["max_dd_pct"] for x in f], [x["dec"]["sharpe"] for x in f],
                        c=[x["dec"]["lag_median"] for x in f], cmap="viridis_r", s=60, edgecolors=INK, linewidths=0.5, zorder=4)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("retardo mediano de confirmación (barras)", color=INK)
        hit = [(x["dec"]["max_dd_pct"], x["dec"]["sharpe"]) for x in f if x["on_bound"]]
        if hit:
            ax.scatter(*zip(*hit), marker="x", color=RUST, s=90, zorder=5, label="punto sobre una cota")
        c = final[B4]["chosen"]
        ax.scatter([c["dec"]["max_dd_pct"]], [c["dec"]["sharpe"]], marker="*", s=300, facecolors="none", edgecolors=RUST,
                   linewidths=2, zorder=6, label="elegido por Sharpe 2023")
        ax.axhline(0, color="#6b7378", linewidth=1)
        ins = ax.inset_axes([0.60, 0.08, 0.36, 0.30])
        for ch in final[B4]["chains"]:
            hv = np.array(ch["hv"], float)
            ins.plot(hv / hv[-1], color=TEAL, linewidth=0.9)
        ins.axvline(80, color=RUST, linestyle=":", linewidth=1)
        ins.set_title("hipervolumen / final", fontsize=7)
        ins.tick_params(labelsize=6)
        ax.legend(fontsize=7.5, loc="upper left", frameon=True, edgecolor="#d0d0d0")
    ax.set_xlabel("Máxima caída (% del capital) · 2020-2022", color=INK)
    ax.set_ylabel("Sharpe anualizado · 2020-2022", color=INK)
    ax.set_title("(B) ZigZag confirmado causalmente (decisión tras la barra, ejecución en la siguiente apertura)", fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    # A4 front: (a) vs (b)
    ax = axs[2]
    f = final.get(pick(final, "A4"), {}).get("front", [])
    if f:
        sc = ax.scatter([x["dec"]["sharpe"] for x in f], [x["b"]["dec"]["sharpe"] for x in f], c=[x["b"]["dec"]["lag_median"] for x in f],
                        cmap="viridis_r", s=60, edgecolors=INK, linewidths=0.5)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("retardo mediano de confirmación (barras)", color=INK)
        ax.axhline(0, color="#6b7378", linewidth=1)
    ax.set_xlabel("(a) Sharpe del experto con retrospectiva · 2020-2022", color=INK)
    ax.set_ylabel("(b) Sharpe de la versión confirmada (mismos parámetros) · 2020-2022", color=INK)
    ax.set_title("Cada punto del frente (A): su versión confirmada y su retardo", fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    fig.suptitle("NSGA-II v2 sobre (Depth, Deviation, Backstep) · población 64 · 3 semillas · genes logarítmicos · "
                 "HINDSIGHT_NON_CAUSAL_EXPERT en (A) · DESARROLLO (2023 desarrollo conjunto; 2024 ya observado)", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(pub / "zigzag_v2_frentes_pareto.png", dpi=120)
    fig.savefig(pub / "zigzag_v2_frentes_pareto.pdf")
    plt.close(fig)


def fmt_p(p):
    s = f"D {p['depth']} / Dev {es(p['deviation_pips'], 1)} pips ({es(p['deviation_points'], 0)} pts) / B {p['backstep']}"
    return s + (f" / k_tp {es(p['k_tp'], 2)}" if "k_tp" in p else "")


def tables(final):
    md = ["# ZigZag NSGA-II v2 — tables (generated)\n",
          "DEVELOPMENT. 2020-2022 decision; 2023 JOINT development (it chooses the expert); 2024 PREVIOUSLY OBSERVED development "
          "(read in v1; read once more here per chosen point, see READ_LEDGER.jsonl). pip = 0,0001; 1 pip = 10 MetaTrader points. "
          "Executable path: decision after the completed bar, fill at the next open, no entry in the first 12 h after a session reopens.\n"]
    md.append("## Convergence (hypervolume of the first front per generation; converged = < 1 % change over the last 20 generations after ≥ 80)\n")
    md.append("| Optimization | Seed | Generations | Stop | HV gen 0 | HV final | Δ last 20 gens | Evaluations | Simulations |")
    md.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for opt, v in final.items():
        if opt == "extras":
            continue
        for ch in v["chains"]:
            md.append(f"| {opt} | {ch['seed']} | {ch['generations']} | {(ch['done'] or {}).get('reason', 'RUNNING')} | {es(ch['hv'][0], 1)} | "
                      f"{es(ch['hv'][-1], 1)} | {es(100 * ch['hv_change_last20'], 3)} % | {ch['evals']} | {ch['sims']} |")
    md.append("")
    for opt, v in final.items():
        if opt == "extras":
            continue
        kind = "B" if opt.startswith("B") else "A"
        md.append(f"## Front {opt} (union of 3 seeds, nondominated on 2020-2022; {len(v['front'])} points; bound hits: "
                  f"{sum(1 for x in v['front'] if x['on_bound'])})\n")
        hdr = "| Parameters (representative) | Tied sets (Deviation range, pips) | Dec. Sharpe | Dec. DD % | Dec. orders/wk |" + (" Dec. lag median |" if kind == "B" else "") + \
              " 2023 Sharpe | 2023 orders/wk | 2023 valid |" + (" (b) k_tp | (b) Sharpe 2020-22 | (b) lag median | (b) Sharpe 2023 |" if opt.startswith("A4") else "") + " On bound (all tied sets) |"
        md.append(hdr)
        md.append("|" + "---|" * (hdr.count("|") - 1))
        for x in sorted(v["front"], key=lambda r: -r["dec"]["sharpe"]):
            tr = x.get("tied_ranges", {}).get("deviation_pips", [x["p"]["deviation_pips"]] * 2)
            line = f"| {fmt_p(x['p'])} | {x.get('n_tied', 1)} ({es(tr[0], 1)}–{es(tr[1], 1)}) | {es(x['dec']['sharpe'], 2)} | {es(x['dec']['max_dd_pct'], 2)} | {es(x['dec']['orders_per_week'], 1)} |"
            if kind == "B":
                line += f" {es(x['dec']['lag_median'], 1)} |"
            line += f" {es(x['val']['sharpe'], 2)} | {es(x['val']['orders_per_week'], 1)} | {'sí' if x['val']['valid'] else 'no'} |"
            if "b" in x:
                bb = x["b"]
                line += f" {es(bb['k_tp'], 1)} | {es(bb['dec']['sharpe'], 2)} | {es(bb['dec']['lag_median'], 1)} | {es(bb['val']['sharpe'], 2)} |"
            line += f" {', '.join(x['on_bound']) or '—'} |"
            md.append(line)
        c = v["chosen"]
        o = c["obs2024"]
        md.append(f"\nChosen (best 2023 Sharpe among points valid on 2020-2022 and 2023): **{fmt_p(c['p'])}** — 2020-2022 Sharpe "
                  f"{es(c['dec']['sharpe'], 2)}, 2023 Sharpe {es(c['val']['sharpe'], 2)}; 2024 (previously observed development, single read): "
                  f"Sharpe {es(o['sharpe'], 2)}, max DD {es(o['max_dd_pct'], 2)} %, {es(o['orders_per_week'], 1)} orders/wk, valid {'sí' if o['valid'] else 'no'}"
                  + (f", 1:10 / 1:50 Sharpe {es(c['obs2024_leverage_sharpe']['10'], 2)} / {es(c['obs2024_leverage_sharpe']['50'], 2)}"
                     if c.get("obs2024_leverage_sharpe") else "") + ".\n")
    ex = final.get("extras")
    if ex:
        md.append("## Fill-path cost: decision after the completed bar, fill at b + 1 vs the pinned v1 path (b + 2)\n")
        md.append("| Case | Period | k_tp | Sharpe b+1 (v2 rules) | Sharpe b+2 (same rules) | Sharpe pinned v1 path | Net b+1 | Net b+2 | Net v1 path | Trades b+1 / v1 | Refused (reopen) |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for r in ex["fill_path_cost"]:
            md.append(f"| {r['case']} | {r['period']} | {es(r['k_tp'], 2)} | {es(r['sharpe_b1'], 2)} | {es(r['sharpe_b2_same_rules'], 2)} | "
                      f"{es(r['sharpe_v1_path'], 2)} | {es(r['net_b1'], 0)} | {es(r['net_b2_same_rules'], 0)} | {es(r['net_v1_path'], 0)} | "
                      f"{r['trades_b1']} / {r['trades_v1_path']} | {r['refused_reopen_b1']} |")
        lg = ex["confirmation_lag_v1_vs_v2"]
        md.append(f"\n## Confirmation lag, v1 definition vs v2 (v1 parameters D3/36,3/B2, {lg['pivots']} pivots 2020-2022)\n")
        md.append("| Definition | Median | Mean | p90 |")
        md.append("|---|---:|---:|---:|")
        for k, d in lg.items():
            if isinstance(d, dict):
                md.append(f"| {k} | {es(d['median'], 1)} | {es(d['mean'], 2)} | {es(d['p90'], 1)} |")
        dm = ex["demonstrations_v2"]
        md.append(f"\n## Demonstrations v2\n\n`demonstrations_v2/expert_trades.csv` — {dm['rows']} trades, sha256 `{dm['sha256']}`, label "
                  f"**{dm['label']}**, params {dm['params']} ({es(dm['deviation_points'], 0)} points); supersedes {dm['supersedes']}.\n")
    return "\n".join(md) + "\n"


def report(args):
    run = Path(args.run)
    pub = Path(args.publish)
    pub.mkdir(parents=True, exist_ok=True)
    final = json.loads((run / "FINAL.json").read_text())
    figure(final, pub)
    (pub / "TABLAS_ZIGZAG_V2.md").write_text(tables(final))
    shutil.copy2(run / "FINAL.json", pub / "FINAL.json")
    shutil.copy2(run / "READ_LEDGER.jsonl", pub / "READ_LEDGER.jsonl")
    if (run / "demonstrations_v2").exists():
        (pub / "demonstrations_v2").mkdir(exist_ok=True)
        for n in ("expert_trades.csv", "MANIFEST.json"):
            shutil.copy2(run / "demonstrations_v2" / n, pub / "demonstrations_v2" / n)
    print("report written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["evaluate", "extras", "report"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data")
    ap.add_argument("--only", nargs="+", default=["A1", "A2", "A4", "B4"])
    ap.add_argument("--publish", default=str(ROOT / "docs/short_horizon/results/zigzag/v2"))
    a = ap.parse_args()
    {"evaluate": evaluate, "extras": extras, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
