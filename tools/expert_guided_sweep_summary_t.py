#!/usr/bin/env python3
"""Capacity (complexification) sweep summary for the temporal ladder: E1-T and E2-T per units u.

Reads <root>/u<U>/{e1t/E1T.json, e2t/E2T.json} (plus an optional baseline dir for u=2), and writes
SWEEP_SUMMARY.json, SWEEP_SUMMARY.md (Spanish) and the capacity curves (Sharpe and net profit versus trainable
parameters, TRAIN / VAL A / VAL B, seed means with min-max bars). No 2024 is read by any sweep arm.

Overfitting marker (reported, never used to choose): for each family, the first u at which the seed-mean TRAIN
Sharpe rises versus the previous u while the VAL A Sharpe falls. Choice rule (fixed 2026-10-20 before any sweep
arm was read): per family, u = argmax seed-mean VAL A Sharpe among u whose 3 seeds are frequency-valid in VAL A;
ties -> smaller u. VAL B is the confirmation; 2024 is read once later, only for the chosen u.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PER = ("TRAIN", "INNER_A", "INNER_B")
PN = {"TRAIN": "TRAIN", "INNER_A": "VAL A", "INNER_B": "VAL B"}
KEYS = ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct", "trades", "orders_per_week")


def stats(ms):
    out = {}
    for p in PER:
        for k in KEYS:
            v = [m[p][k] for m in ms if p in m and m[p].get(k) is not None and np.isfinite(m[p][k])]
            out[f"{p}_{k}"] = [float(np.mean(v)), float(np.min(v)), float(np.max(v))] if v else None
        out[f"{p}_all_valid"] = all(m.get(p, {}).get("valid") for m in ms)
    return out


def load(u, d):
    rows = {}
    e1 = d / "e1t" / "E1T.json"
    if e1.exists():
        j = json.loads(e1.read_text())
        rows["E1-T BC+DAgger"] = {"params": j["model"].get("params"),
                                 **stats([r["rounds"][-1]["trading"] for r in j["seeds"].values()])}
    e2 = d / "e2t" / "E2T.json"
    if e2.exists():
        j = json.loads(e2.read_text())
        for arm, name in (("SAC", "E2-T SAC"), ("SAC_DEMO", "E2-T SAC+demos")):
            ms = [r["trading"] for r in j["runs"].values() if r["arm"] == arm]
            if ms:
                rows[name] = {"params": j.get("params_actor_plus_critic"), **stats(ms)}
    return rows


def params_e2(u, nf=15, ns=9, act=3):
    trunk = nf * u * 4 + (3 * nf * u * u + u) + (3 * u * u + u)
    d = u + ns
    actor = trunk + (d * u + u) + 2 * (u * act + act)
    critic = trunk + 2 * ((d + act) * u + u + u + 1)
    return actor + critic


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline-u2", default="", help="dir holding e1t/ and e2t/ of the u=2 run")
    ap.add_argument("--units", default="2,4,8,16,32,64")
    ap.add_argument("--nfeat", type=int, default=15)
    a = ap.parse_args()
    root = Path(a.root)
    res = {}
    for u in [int(x) for x in a.units.split(",")]:
        d = Path(a.baseline_u2) if (u == 2 and a.baseline_u2) else root / f"u{u}"
        r = load(u, d)
        for k in r:
            if k.startswith("E2") and r[k]["params"] is None:
                r[k]["params"] = params_e2(u, a.nfeat)
        if r:
            res[u] = r
    fams = sorted({f for r in res.values() for f in r})
    choice, onset = {}, {}
    for f in fams:
        us = [u for u in sorted(res) if f in res[u]]
        ok = [u for u in us if res[u][f]["INNER_A_all_valid"] and res[u][f]["INNER_A_sharpe_daily_sqrt365"]]
        choice[f] = max(ok, key=lambda u: (res[u][f]["INNER_A_sharpe_daily_sqrt365"][0], -u)) if ok else None
        onset[f] = None
        for u0, u1 in zip(us, us[1:]):
            a0, a1 = res[u0][f], res[u1][f]
            if None in (a0["TRAIN_sharpe_daily_sqrt365"], a1["TRAIN_sharpe_daily_sqrt365"], a0["INNER_A_sharpe_daily_sqrt365"],
                        a1["INNER_A_sharpe_daily_sqrt365"]):
                continue
            if a1["TRAIN_sharpe_daily_sqrt365"][0] > a0["TRAIN_sharpe_daily_sqrt365"][0] and \
                    a1["INNER_A_sharpe_daily_sqrt365"][0] < a0["INNER_A_sharpe_daily_sqrt365"][0]:
                onset[f] = u1
                break
    out = {"rule": __doc__, "results": {str(u): r for u, r in res.items()}, "choice_on_inner_a": choice,
           "overfitting_onset_marker": onset, "units_done": sorted(res)}
    (root / "SWEEP_SUMMARY.json").write_text(json.dumps(out, indent=1))
    L = ["# Barrido de complejidad del tronco temporal (E1-T y E2-T), DESARROLLO", "",
         "Medias de 3 semillas (mín–máx entre paréntesis para el Sharpe). Sin lectura de 2024. Variables: las 15 de "
         "Musashi. Elección por VAL A; VAL B confirma.", ""]
    for f in fams:
        L += [f"## {f}", "", "| u | Parámetros | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Brecha TRAIN − VAL A | "
              "Ganancia VAL A | Ganancia VAL B | DD % VAL B | Órdenes/sem VAL B | Válida VAL A |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for u in sorted(res):
            r = res[u].get(f)
            if not r:
                continue

            def s(p, k="sharpe_daily_sqrt365", rng=True):
                v = r[f"{p}_{k}"]
                if v is None:
                    return "—"
                return f"{v[0]:.2f} ({v[1]:.2f}…{v[2]:.2f})" if rng else f"{v[0]:,.1f}".replace(",", " ")
            gap = (r["TRAIN_sharpe_daily_sqrt365"][0] - r["INNER_A_sharpe_daily_sqrt365"][0]) \
                if r["TRAIN_sharpe_daily_sqrt365"] and r["INNER_A_sharpe_daily_sqrt365"] else None
            L.append(f"| {u} | {r['params']} | {s('TRAIN')} | {s('INNER_A')} | {s('INNER_B')} | "
                     f"{'—' if gap is None else f'{gap:.2f}'} | {s('INNER_A', 'net_profit_usd', False)} | "
                     f"{s('INNER_B', 'net_profit_usd', False)} | {s('INNER_B', 'max_drawdown_pct', False)} | "
                     f"{s('INNER_B', 'orders_per_week', False)} | {'sí' if r['INNER_A_all_valid'] else 'NO'} |")
        L += ["", f"Elección por VAL A: u = {choice[f]}. Marcador de inicio de sobreajuste (TRAIN sube y VAL A baja): "
              f"u = {onset[f]}.", ""]
    (root / "SWEEP_SUMMARY.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, max(len(fams), 1), figsize=(5.2 * max(len(fams), 1), 4.4), squeeze=False)
    for ax, f in zip(axes[0], fams):
        us = [u for u in sorted(res) if f in res[u]]
        x = [res[u][f]["params"] for u in us]
        for p, c in zip(PER, ("tab:blue", "tab:orange", "tab:green")):
            v = [res[u][f][f"{p}_sharpe_daily_sqrt365"] for u in us]
            m = [np.nan if q is None else q[0] for q in v]
            lo = [0 if q is None else q[0] - q[1] for q in v]
            hi = [0 if q is None else q[2] - q[0] for q in v]
            ax.errorbar(x, m, yerr=[lo, hi], marker="o", capsize=3, color=c, label=PN[p])
        for u, xx in zip(us, x):
            ax.annotate(f"u={u}", (xx, ax.get_ylim()[0]), fontsize=7, ha="center", va="bottom")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Parámetros entrenables (escala log)")
        ax.set_ylabel("Sharpe diario (√365), media de semillas")
        ax.set_title(f)
        ax.legend(fontsize=8)
    fig.suptitle("Curva de capacidad: desempeño contra complejidad del modelo (desarrollo)")
    fig.tight_layout()
    fig.savefig(root / "curva_capacidad_sharpe.png", dpi=150)
    fig.savefig(root / "curva_capacidad_sharpe.pdf")
    print(json.dumps({"units_done": sorted(res), "choice": choice, "onset": onset}))


if __name__ == "__main__":
    main()
