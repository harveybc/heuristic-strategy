#!/usr/bin/env python3
"""Final tables and figures for the temporal ladder (E0-T run B, E1-T, E2-T), in Spanish.

Per policy and period: net profit (USD), Sharpe (daily, sqrt 365), max drawdown (% and USD), orders, orders per week,
win rate, profit factor, frequency validity; TRAIN versus validation (2023H1 = VAL A, 2023H2 = VAL B) differences;
2024 single read; references under identical accounting; Musashi-contract sensitivity (<= 1/day, 1:50).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

P = ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")
PN = {"TRAIN": "TRAIN 2020–22", "INNER_A": "VAL A 2023S1", "INNER_B": "VAL B 2023S2", "OBS_2024": "2024 (una lectura)"}


def f(x, n=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{n}f}".replace(",", " ")


def row(name, m):
    if not m:
        return None
    return (f"| {name} | {f(m.get('net_profit_usd'), 0)} | {f(m.get('sharpe_daily_sqrt365'))} | {f(m.get('max_drawdown_pct'), 1)} | "
            f"{m.get('trades', '—')} | {f(m.get('orders_per_week'), 1)} | {f(100 * m['win_rate'], 1) if m.get('win_rate') is not None else '—'} | "
            f"{f(m.get('profit_factor'))} | {'sí' if m.get('valid') else 'NO'} |")


HDR = ("| Política | Ganancia neta USD | Sharpe | Máx. DD % | Órdenes | Órdenes/sem | Acierto % | Profit factor | Válida |\n"
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def mean_of(ms, k):
    v = [m.get(k) for m in ms if m and m.get(k) is not None and np.isfinite(m.get(k))]
    return float(np.mean(v)) if v else None


def policies(e1, e2):
    """name -> {period: [summary per seed]}"""
    out = {}
    if e1:
        for s, rec in e1["seeds"].items():
            fr = rec["rounds"][-1]
            for p in P:
                if p in fr["trading"]:
                    out.setdefault("E1-T BC+DAgger r3", {}).setdefault(p, []).append(fr["trading"][p])
    if e2:
        for key, rec in e2["runs"].items():
            name = "E2-T SAC+demos" if rec["arm"] == "SAC_DEMO" else "E2-T SAC"
            for p in P:
                if p in rec["trading"]:
                    out.setdefault(name, {}).setdefault(p, []).append(rec["trading"][p])
    return out


def references(e2):
    out = {}
    if not e2:
        return out
    names = {"expert_demonstrations": "Experto retrospectivo (techo, no causal)", "expert_confirmed": "Experto confirmado (causal)",
             "train_prior_random": "Aleatoria con el prior de TRAIN", "no_trade": "No operar",
             "expert_representable_in_action_space": "Experto representable en el espacio de acción"}
    for p, c in e2["references"].items():
        for k, v in c.items():
            if k in names:
                out.setdefault(names[k], {})[p] = v if isinstance(v, list) else [v]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e0t", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--e1t", default="")
    ap.add_argument("--e2t", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(a.select).read_text())
    e1 = json.loads(Path(a.e1t).read_text()) if a.e1t and Path(a.e1t).exists() else None
    e2 = json.loads(Path(a.e2t).read_text()) if a.e2t and Path(a.e2t).exists() else None
    pol, ref = policies(e1, e2), references(e2)
    L = ["# RL guiado por el experto ZigZag con el tronco temporal mínimo: resultados finales", "",
         "DESARROLLO, nada operable. TRAIN 2020–2022; validación = 2023 (VAL A = S1, VAL B = S2); 2024 = desarrollo ya "
         "observado, leído una sola vez después de congelar; TEST 2025 cerrado. Contrato: SL y TP efectivos en cada orden, "
         "unidades enteras con rel_volume 0,10 y apalancamiento 1:100, ≤ 4 entradas/día, ≥ 3 órdenes/semana, costos "
         "realistas (spread, slippage, comisión, swap) ASUMIDOS, no verificados con el broker. Capital 10 000 USD por periodo.", "",
         "Cómo leer la ganancia: se capitaliza (el tamaño crece con el balance a 1:100), así que una política ganadora en "
         "TRAIN (3 años) produce cifras astronómicas, y TRAIN (3 años) no es comparable en USD con VAL A/B (6 meses cada "
         "una) ni con 2024 (1 año). Para comparar TRAIN contra validación, use el Sharpe (anualizado), el drawdown y "
         "las órdenes por semana; la ganancia solo indica el signo y el orden de magnitud dentro de un mismo periodo.", "",
         f"Modelo elegido (regla fijada antes de leer la corrida B): **{sel['arm']}**, variables **{sel['features']}**, "
         f"**u = {sel['units']}** unidades por componente.", "",
         "Evidencia de la elección (pérdida INNER_A, media de semillas): " +
         "; ".join(f"{k}: {v['inner_a_log_loss_seed_mean']:.4f} ({v['params']} parámetros)" for k, v in sel["evidence"].items()), ""]
    if e1:
        L += [f"E1-T: arquitectura {e1['model'].get('architecture')}, {e1['model'].get('params')} parámetros; compuerta: "
              f"{e1.get('gate_result', {}).get('decision')}.", ""]
    if e2:
        L += [f"E2-T: SAC (SB3), extractor temporal + net_arch {e2['design']['network'].get('net_arch')}; presupuesto "
              f"{e2['design']['budget_transitions']} transiciones; 3 semillas por brazo.", ""]
    L += ["## 1. Métricas por semilla y periodo", ""]
    for name, per in pol.items():
        L += [f"### {name}", "", HDR]
        for p in P:
            for i, m in enumerate(per.get(p, [])):
                L.append(row(f"{PN[p]} · semilla {i + 1}", m))
        L.append("")
    L += ["## 2. TRAIN contra validación (media de semillas)", "",
          "| Política | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Δ Sharpe (VAL media − TRAIN) | Ganancia TRAIN | Ganancia VAL A | "
          "Ganancia VAL B | DD % TRAIN | DD % VAL B | Órdenes/sem TRAIN | Órdenes/sem VAL B | Sharpe 2024 | Ganancia 2024 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, per in list(pol.items()) + list(ref.items()):
        g = {p: per.get(p, []) for p in P}
        st, sa, sb = (mean_of(g[p], "sharpe_daily_sqrt365") for p in ("TRAIN", "INNER_A", "INNER_B"))
        d = None if None in (st, sa, sb) else 0.5 * (sa + sb) - st
        L.append(f"| {name} | {f(st)} | {f(sa)} | {f(sb)} | {f(d)} | {f(mean_of(g['TRAIN'], 'net_profit_usd'), 0)} | "
                 f"{f(mean_of(g['INNER_A'], 'net_profit_usd'), 0)} | {f(mean_of(g['INNER_B'], 'net_profit_usd'), 0)} | "
                 f"{f(mean_of(g['TRAIN'], 'max_drawdown_pct'), 1)} | {f(mean_of(g['INNER_B'], 'max_drawdown_pct'), 1)} | "
                 f"{f(mean_of(g['TRAIN'], 'orders_per_week'), 1)} | {f(mean_of(g['INNER_B'], 'orders_per_week'), 1)} | "
                 f"{f(mean_of(g['OBS_2024'], 'sharpe_daily_sqrt365'))} | {f(mean_of(g['OBS_2024'], 'net_profit_usd'), 0)} |")
    L.append("")
    L += ["## 3. Sensibilidad con el contrato de Musashi (≤ 1 entrada/día, 1:50), políticas congeladas", "", HDR]
    for src, name in ((e1, "E1-T BC+DAgger r3"), (e2, None)):
        if not src:
            continue
        if name:
            for s, per in src.get("musashi_contract_sensitivity", {}).items():
                for p in P:
                    L.append(row(f"{name} · {PN[p]} · semilla {s}", per.get(p)))
        else:
            for key, rec in src["runs"].items():
                for p in P:
                    L.append(row(f"{key} · {PN[p]}", rec.get("musashi_contract_sensitivity", {}).get(p)))
    L.append("")
    if e2:
        L += ["## 4. Parada temprana de E2-T", "", "| Corrida | Parada | Transiciones del mejor checkpoint | Puntuación |",
              "| --- | --- | --- | --- |"]
        for key, rec in e2["runs"].items():
            L.append(f"| {key} | {rec['stop']} | {rec['best_transitions']} | {f(rec['best_score'])} |")
        L.append("")
    (out / "RESULTADOS_RL_TEMPORAL.md").write_text("\n".join(x for x in L if x is not None) + "\n")
    # figure: Sharpe and profit by period, TRAIN vs validation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(pol) + [k for k in ref if "techo" not in k]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        per = pol.get(n) or ref.get(n)
        for j, (k, lab) in enumerate((("sharpe_daily_sqrt365", "Sharpe diario (√365)"), ("net_profit_usd", "Ganancia neta (USD)"))):
            ys = [mean_of(per.get(p, []), k) for p in P]
            ax[j].bar(np.arange(len(P)) + i * w, [0 if y is None else y for y in ys], w, label=n)
            ax[j].set_ylabel(lab)
    for j in range(2):
        ax[j].axhline(0, color="k", lw=0.8)
        ax[j].set_xticks(np.arange(len(P)) + 0.4 - w / 2)
        ax[j].set_xticklabels([PN[p] for p in P], fontsize=8)
    ax[1].legend(fontsize=7, loc="best")
    fig.suptitle("TRAIN contra validación, media de 3 semillas (desarrollo)")
    fig.tight_layout()
    fig.savefig(out / "rl_temporal_train_vs_validacion.png", dpi=150)
    fig.savefig(out / "rl_temporal_train_vs_validacion.pdf")
    print("written", out)


if __name__ == "__main__":
    main()
