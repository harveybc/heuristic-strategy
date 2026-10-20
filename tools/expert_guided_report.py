#!/usr/bin/env python3
"""Spanish figures and tables for the expert-guided ladder (E0, E1, E2) from the sealed run JSONs.

  e0  <run dir with E0.json> -> E0_TABLAS.md, e0_brecha_imitabilidad.{png,pdf}, e0_curvas_entrenamiento.png,
      e0_prauc_exceso.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference categorical order (validated palette), fixed per entity, never by rank
C = {"expert": "#2a78d6", "expert_ctp": "#6250d6", "confirmed": "#eb6834", "student_h": "#1baf7a",
     "student_c": "#eda100", "prior": "#e87ba4", "sac": "#008300", "bc": "#1baf7a", "dagger": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e8e8e6"
PERIODS = ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")
P_ES = {"TRAIN": "TRAIN 2020–2022", "INNER_A": "INNER_A 2023-S1", "INNER_B": "INNER_B 2023-S2",
        "OBS_2024": "2024 (observado, 1 lectura)"}
REPO = "https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/"


def es(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if abs(v) >= 1e7:
        return f"{v:.2e}".replace(".", ",")
    s = f"{v:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)


def e0_policies(d):
    seeds = list(d["students"]["hindsight"])
    rows = [("Experto retrospectivo (TP = pivote siguiente)", "expert", lambda p: [d["comparators"][p]["expert_hindsight_tp_pivot"]]),
            ("Experto retrospectivo, TP causal 1,5σ", "expert_ctp", lambda p: [d["comparators"][p]["expert_hindsight_timing_causal_tp"]]),
            ("Experto confirmado (reactivo)", "confirmed", lambda p: [d["comparators"][p]["expert_confirmed"]]),
            ("Estudiante del experto retrospectivo", "student_h", lambda p: [d["students"]["hindsight"][s]["trading"][p] for s in seeds]),
            ("Estudiante del experto confirmado", "student_c", lambda p: [d["students"]["confirmed"][s]["trading"][p] for s in seeds]),
            ("Aleatoria con la previa de TRAIN", "prior", lambda p: d["comparators"][p]["train_prior_random"])]
    return rows


def fig_gap(d, out):
    rows = e0_policies(d)
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), sharey=False)
    for ax, p in zip(axes, PERIODS):
        style(ax)
        for i, (name, key, get) in enumerate(rows):
            ms = get(p)
            vals = np.array([m["sharpe"] if m["sharpe"] is not None else np.nan for m in ms], float)
            mu = np.nanmean(vals)
            ax.bar(i, mu, width=0.72, color=C[key], edgecolor="white", linewidth=2)
            if len(vals) > 1:
                ax.scatter(np.full(len(vals), i), vals, s=36, color="white", edgecolor=INK, linewidth=1, zorder=3)
            ax.annotate(es(mu), (i, mu), xytext=(0, 4 if mu >= 0 else -14), textcoords="offset points", ha="center",
                        fontsize=9.5, color=INK)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([str(i + 1) for i in range(len(rows))], fontsize=10)
        ax.set_title(P_ES[p], fontsize=12, color=INK)
        if p == "TRAIN":
            ax.set_ylabel("Sharpe neto (diario, √365)", fontsize=12, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for _, k, _ in rows]
    fig.legend(handles, [f"{i + 1}. {n}" for i, (n, _, _) in enumerate(rows)], loc="lower center", ncol=3, fontsize=10.5,
               frameon=False)
    gb = d["comparators"]
    ib = lambda k: gb["INNER_B"][k]["sharpe"]  # noqa: E731
    sh = np.mean([d["students"]["hindsight"][s]["trading"]["INNER_B"]["sharpe"] for s in d["students"]["hindsight"]])
    fig.suptitle("E0 · Brecha de imitabilidad del experto ZigZag (3 / 40,7 pips / 0) desde las 15 entradas del núcleo admitido\n"
                 f"INNER_B: experto {es(ib('expert_hindsight_tp_pivot'))} · confirmado {es(ib('expert_confirmed'))} · estudiante "
                 f"{es(sh)} (brecha experto − estudiante {es(ib('expert_hindsight_tp_pivot') - sh)}); puntos = semillas; costos realistas, SL/TP en cada orden, ≤ 4 órdenes/día",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"e0_brecha_imitabilidad.{ext}", dpi=160)
    plt.close(fig)


def fig_curves(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (t, title) in zip(axes, (("hindsight", "Objetivo retrospectivo (anticipatorio)"),
                                     ("confirmed", "Objetivo confirmado (reactivo)"))):
        style(ax)
        for j, (s, r) in enumerate(d["students"][t].items()):
            h = r["history"]
            ep = [x["epoch"] for x in h]
            ax.plot(ep, [x["train_loss"] for x in h], color=C["expert"], linewidth=2, alpha=0.55 + 0.15 * j,
                    label="TRAIN (inferencia)" if j == 0 else None)
            ax.plot(ep, [x["inner_loss"] for x in h], color=C["confirmed"], linewidth=2, alpha=0.55 + 0.15 * j,
                    label="INNER_A" if j == 0 else None)
            ax.axvline(r["fit"]["best_epoch"], color=MUTED, linewidth=1, linestyle=":")
        pr = d["train_prior"][t]
        ax.set_title(f"{title}: pérdida por época (3 semillas)", fontsize=12, color=INK)
        ax.set_xlabel("Época", fontsize=11, color=INK)
        ax.set_ylabel("Entropía cruzada (nats)", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=10)
        ax.text(0.98, 0.95, f"previa TRAIN {', '.join(es(x, 3) for x in pr)}", transform=ax.transAxes, ha="right",
                va="top", fontsize=9.5, color=MUTED)
    fig.suptitle("E0 · Curvas TRAIN e INNER_A (monitor del dueño = media de ambas, paciencia 10; línea punteada = mejor época)",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "e0_curvas_entrenamiento.png", dpi=160)
    plt.close(fig)


def fig_prauc(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, (t, title, key) in zip(axes, (("hindsight", "Estudiante del experto retrospectivo", "student_h"),
                                          ("confirmed", "Estudiante del experto confirmado", "student_c"))):
        style(ax)
        x = 0
        ticks, labels = [], []
        for p in PERIODS:
            for dirn in ("LONG", "SHORT"):
                vals = [r["scores"][p]["calibrated"][dirn]["pr_auc_minus_prevalence"] for r in d["students"][t].values()]
                ax.bar(x, np.mean(vals), width=0.7, color=C[key] if dirn == "LONG" else C["expert"], edgecolor="white", linewidth=2)
                ax.scatter([x] * len(vals), vals, s=30, color="white", edgecolor=INK, zorder=3)
                if p in ("INNER_B", "OBS_2024"):
                    lo = np.mean([r["scores"][p]["calibrated"][dirn]["boot_excess_lo95"] for r in d["students"][t].values()])
                    hi = np.mean([r["scores"][p]["calibrated"][dirn]["boot_excess_hi95"] for r in d["students"][t].values()])
                    ax.plot([x, x], [lo, hi], color=INK, linewidth=1.5)
                ticks.append(x)
                labels.append(f"{p.replace('_', ' ')}\n{'largo' if dirn == 'LONG' else 'corto'}")
                x += 1
            x += 0.6
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_ylabel("PR-AUC − prevalencia", fontsize=11, color=INK)
    fig.suptitle("E0 · Ranking por dirección sobre la tasa base (NONE incluido en la base); barras de error = IC 95 % bootstrap por semanas ISO",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "e0_prauc_exceso.png", dpi=160)
    plt.close(fig)


def trade_row(name, ms):
    def mean(k):
        v = [m.get(k) for m in ms if m.get(k) is not None]
        v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
        return float(np.mean(v)) if v else None
    cost = sum((mean(k) or 0.0) for k in ("cost_spread", "cost_slippage", "cost_commission", "cost_financing"))
    return (f"| {name} | {es(mean('net'))} | {es(mean('sharpe'))} | {es(mean('sortino'))} | {es(mean('max_dd_pct'))} | "
            f"{es(mean('trades'), 0)} | {es(mean('exit_tp'), 0)} / {es(mean('exit_sl'), 0)} / {es(mean('exit_time'), 0)} | "
            f"{es(100 * (mean('win_rate') or 0), 1)} | {es(cost)} ({es(mean('cost_spread'))} / {es(mean('cost_slippage'))} / "
            f"{es(mean('cost_commission'))} / {es(mean('cost_financing'))}) | {es(mean('orders_per_week'))} | "
            f"{sum(bool(m.get('valid')) for m in ms)}/{len(ms)} |")


def tables(d, out, figs):
    L = ["# E0 · Imitabilidad del experto ZigZag v2 desde las 15 entradas del núcleo admitido (generado)", "",
         "DESARROLLO. TRAIN 2020–2022 ajusta; INNER_A (2023-S1) monitorea, calibra la temperatura y elige el umbral; "
         "INNER_B (2023-S2) se puntúa después de congelar; 2024 es desarrollo ya observado, leído una sola vez después de "
         "congelar. TEST 2025 cerrado. Fuente: raíz admitida c80847bb (UTC inicio de barra, disponible +1 h); entradas: "
         "CSV preparados de Musashi (contrato 37e06192, selección 553fc047), 13 observaciones paso 2, z-score causal 256.",
         "", "Ejecución de toda política: decisión tras la barra completa, llenado en la apertura siguiente, SL = 2σ, "
         "TP = 1,5σ (causal) o pivote siguiente (solo el experto retrospectivo), unidades = ⌊1,0 × 0,10 × saldo × 100 / precio⌋, "
         "sin entradas 12 h tras reapertura, plano 12 h antes del cierre, ≤ 4 órdenes por día hábil, costos RealisticProfile "
         "(spread por hora, deslizamiento, comisión, swap). Sharpe = diario, √365, días calendario incluidos los cierres. "
         "Órdenes/semana sobre TODAS las semanas ISO con filas elegibles (más estricto que el estudio v2). Cada operación = "
         "1 entrada + 1 cierre; cierres por TP / SL / tiempo (plano forzado o siguiente señal).", ""]
    for f in figs:
        L.append(f"![{f}]({REPO}docs/short_horizon/results/expert_guided/e0/{f}) — [{f}]({REPO}docs/short_horizon/results/expert_guided/e0/{f})")
    L.append("")
    g = d["gate_result"]
    L += ["## Compuerta pre-registrada", "", f"- G1 (log-loss calibrada < previa TRAIN en INNER_B): **{'sí' if g['G1'] else 'no'}** "
          f"(por semilla {g['G1_per_seed']}, media {es(1000 * g['G1_mean_log_loss_minus_prior'], 1)} milinats)",
          f"- G2 (PR-AUC − prevalencia, IC 95 % bootstrap semanal > 0, ambas direcciones): **{'sí' if g['G2'] else 'no'}** ({g['G2_per_seed']})",
          f"- Decisión: **{g['decision']}**", ""]
    for p in PERIODS:
        L += [f"## Negociación · {P_ES[p]}", "",
              "| Política | Neto USD | Sharpe | Sortino | DD máx % | Operaciones | Cierres TP / SL / tiempo | Acierto % | Costos USD (spread / desl. / com. / swap) | Órdenes/sem | Válidas |",
              "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|"]
        for name, _, get in e0_policies(d):
            L.append(trade_row(name + (" (media 3 semillas)" if len(get(p)) > 1 else ""), get(p)))
        L.append("| Sin operar | 0 | 0 | 0 | 0 | 0 | — | — | 0 | 0 | 0/1 |")
        L.append("")
    L += ["## Clasificación (probabilidades calibradas con temperatura en INNER_A)", "",
          "| Objetivo | Semilla | Periodo | log-loss − previa (milinats) | PR-AUC largo (prev.) | PR-AUC corto (prev.) | IC95 exceso largo | IC95 exceso corto | Precisión/recall largo @umbral | Precisión/recall corto @umbral |",
          "|---|---|---|---:|---|---|---|---|---|---|"]
    for t in ("hindsight", "confirmed"):
        for s, r in d["students"][t].items():
            for p in PERIODS:
                sc = r["scores"][p]["calibrated"]
                at = r["scores"][p]["at_threshold"]
                ci = lambda k: (f"[{es(sc[k]['boot_excess_lo95'], 3)}; {es(sc[k]['boot_excess_hi95'], 3)}]"  # noqa: E731
                                if "boot_excess_lo95" in sc[k] else "—")
                L.append(f"| {'retrospectivo' if t == 'hindsight' else 'confirmado'} | {s} | {p} | "
                         f"{es(1000 * sc['log_loss_minus_prior'], 1)} | {es(sc['LONG']['pr_auc'], 3)} ({es(sc['LONG']['prevalence'], 3)}) | "
                         f"{es(sc['SHORT']['pr_auc'], 3)} ({es(sc['SHORT']['prevalence'], 3)}) | {ci('LONG')} | {ci('SHORT')} | "
                         f"{es(at['LONG']['precision'], 3)} / {es(at['LONG']['recall'], 3)} | {es(at['SHORT']['precision'], 3)} / {es(at['SHORT']['recall'], 3)} |")
    L += ["", "## Entrenamiento (regla del dueño)", "", "| Objetivo | Semilla | Épocas | Mejor época | Causa | Pérdida TRAIN | Pérdida INNER_A | Brecha INNER−TRAIN | Temperatura | Tasa de señal elegida (INNER_A) |",
          "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for t in ("hindsight", "confirmed"):
        for s, r in d["students"][t].items():
            h = r["history"][r["fit"]["best_epoch"] - 1]
            L.append(f"| {t} | {s} | {r['fit']['epochs']} | {r['fit']['best_epoch']} | {r['fit']['stop_reason']} | {es(h['train_loss'], 4)} | "
                     f"{es(h['inner_loss'], 4)} | {es(h['gap_inner_minus_train'], 4)} | {es(r['temperature'], 2)} | {es(100 * r['threshold']['rate'], 0)} % |")
    L += ["", "## Población y exclusiones (por motivo; las máscaras se solapan)", "", "| Motivo | TRAIN | INNER_A | INNER_B | 2024 |", "|---|---:|---:|---:|---:|"]
    for k, v in d["exclusions"].items():
        L.append(f"| {k} | {v['TRAIN']} | {v['INNER_A']} | {v['INNER_B']} | {v['OBS_2024']} |")
    L.append(f"| **elegibles** | {d['meta']['rows_eligible']['TRAIN']} | {d['meta']['rows_eligible']['INNER_A']} | "
             f"{d['meta']['rows_eligible']['INNER_B']} | {d['meta']['rows_eligible']['OBS_2024']} |")
    (out / "E0_TABLAS.md").write_text("\n".join(L) + "\n")


def e0(a):
    run = Path(a.run)
    d = json.loads((run / "E0.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_gap(d, out)
    fig_curves(d, out)
    fig_prauc(d, out)
    tables(d, out, ["e0_brecha_imitabilidad.png", "e0_curvas_entrenamiento.png", "e0_prauc_exceso.png"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["e0"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"e0": e0}[a.stage](a)


if __name__ == "__main__":
    main()
