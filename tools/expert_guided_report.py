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


# ---------------------------------------------------------------------------------------------- E1
RND_ES = ("BC", "DAgger 1", "DAgger 2", "DAgger 3")


def _sh(m):
    v = m.get("sharpe_daily_sqrt365")
    return np.nan if v is None else v


def e1_fig_rounds(d, out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    cB = d["comparators"]
    for ax, p in zip(axes[:3], PERIODS[:3]):
        style(ax)
        for j, (s, r) in enumerate(d["seeds"].items()):
            ys = [_sh(rr["trading"][p]) for rr in r["rounds"]]
            ax.plot(range(len(ys)), ys, color=C["dagger"], linewidth=2, alpha=0.5 + 0.2 * j, marker="o", markersize=8,
                    label="Política BC → DAgger (3 semillas)" if j == 0 else None)
        ax.axhline(_sh(cB[p]["expert_confirmed"]), color=C["confirmed"], linestyle="--", linewidth=2, label="Experto confirmado")
        ax.axhline(np.mean([_sh(x) for x in cB[p]["train_prior_random"]]), color=C["prior"], linestyle=":", linewidth=2,
                   label="Aleatoria previa TRAIN (media)")
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(RND_ES, fontsize=10)
        ax.set_title(f"{P_ES[p]}\nexperto {es(_sh(cB[p]['expert_demonstrations']))} · oráculo {es(_sh(cB[p]['oracle_leg_following']))}",
                     fontsize=11.5, color=INK)
        if p == "TRAIN":
            ax.set_ylabel("Sharpe neto (diario, √365)", fontsize=12, color=INK)
            ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax = axes[3]
    style(ax)
    p = "OBS_2024"
    vals = [_sh(r["rounds"][-1]["trading"][p]) for r in d["seeds"].values()]
    items = [("DAgger 3", np.mean(vals), C["dagger"], vals), ("Confirmado", _sh(cB[p]["expert_confirmed"]), C["confirmed"], None),
             ("Previa", np.mean([_sh(x) for x in cB[p]["train_prior_random"]]), C["prior"],
              [_sh(x) for x in cB[p]["train_prior_random"]]), ("Sin operar", 0.0, MUTED, None)]
    for i, (n, v, col, pts) in enumerate(items):
        ax.bar(i, v, color=col, width=0.7, edgecolor="white", linewidth=2)
        if pts:
            ax.scatter([i] * len(pts), pts, s=34, color="white", edgecolor=INK, zorder=3)
        ax.annotate(es(v), (i, v), xytext=(0, 4 if v >= 0 else -14), textcoords="offset points", ha="center", fontsize=9.5)
    ax.axhline(0, color=INK, linewidth=1)
    ax.margins(y=0.15)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[0] for x in items], fontsize=9.5)
    ax.set_title(f"{P_ES[p]}\nexperto {es(_sh(cB[p]['expert_demonstrations']))} · oráculo {es(_sh(cB[p]['oracle_leg_following']))}",
                 fontsize=11.5, color=INK)
    g = d["gate_result"]
    fig.suptitle("E1 · Clonación de comportamiento + DAgger en el entorno ejecutable (SL/TP en la acción, ≤ 4 entradas/día, costos realistas)\n"
                 f"Compuerta: I1 {'sí' if g['I1'] else 'no'} · I2 {'sí' if g['I2'] else 'no'} → {g['decision']}; "
                 f"INNER_B DAgger 3 = {es(g['inner_b_sharpe_mean'])} vs confirmado {es(g['confirmed'])} y previa {es(g['prior_random_mean'])}",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"e1_sharpe_por_ronda.{ext}", dpi=160)
    plt.close(fig)


def e1_fig_imitation(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (key, title, ylab) in zip(axes, (("ll", "Imitación en los estados que visita la propia política",
                                               "log-loss − previa TRAIN (nats)"),
                                              ("opw", "Frecuencia de operación", "Entradas por semana"))):
        style(ax)
        for k, p in enumerate(PERIODS[:3]):
            col = (C["expert"], C["confirmed"], C["dagger"])[k]
            for j, r in enumerate(d["seeds"].values()):
                ys = [rr["imitation"][p]["log_loss_minus_prior"] if key == "ll" else rr["trading"][p]["orders_per_week"]
                      for rr in r["rounds"]]
                ax.plot(range(len(ys)), ys, color=col, linewidth=2, alpha=0.45 + 0.2 * j, marker="o", markersize=7,
                        label=P_ES[p] if j == 0 else None)
        if key == "opw":
            ax.axhline(3, color=INK, linestyle=":", linewidth=1.5, label="mínimo 3/semana")
        ax.axhline(0, color=INK, linewidth=1) if key == "ll" else None
        ax.set_xticks(range(4))
        ax.set_xticklabels(RND_ES)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_ylabel(ylab, fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9.5)
        if key == "ll":
            ax.set_yscale("symlog", linthresh=0.1)
    fig.suptitle("E1 · DAgger mejora la imitación ronda a ronda, pero la política opera ~15 veces/semana y pierde por costos",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "e1_imitacion_por_ronda.png", dpi=160)
    plt.close(fig)


def e1_fig_curves(d, out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=False)
    for k, ax in enumerate(axes):
        style(ax)
        for j, r in enumerate(d["seeds"].values()):
            h = r["rounds"][k]["history"]
            ep = [x["epoch"] for x in h]
            ax.plot(ep, [x["train_loss"] for x in h], color=C["expert"], linewidth=2, alpha=0.5 + 0.2 * j,
                    label="TRAIN (inferencia)" if j == 0 else None)
            ax.plot(ep, [x["inner_loss"] for x in h], color=C["confirmed"], linewidth=2, alpha=0.5 + 0.2 * j,
                    label="INNER_A" if j == 0 else None)
            ax.axvline(r["rounds"][k]["fit"]["best_epoch"], color=MUTED, linestyle=":", linewidth=1)
        ax.set_title(RND_ES[k], fontsize=12, color=INK)
        ax.set_xlabel("Época", fontsize=11, color=INK)
        if k == 0:
            ax.set_ylabel("Pérdida ponderada (CE + Huber SL/TP)", fontsize=11, color=INK)
            ax.legend(frameon=False, fontsize=9.5)
    fig.suptitle("E1 · Curvas TRAIN e INNER_A por ronda (monitor = media de ambas, paciencia 10; punteada = mejor época)",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "e1_curvas_entrenamiento.png", dpi=160)
    plt.close(fig)


def env_row(name, ms):
    def mean(k):
        v = [m.get(k) for m in ms if m.get(k) is not None]
        v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
        return float(np.mean(v)) if v else None
    cost = sum((mean(k) or 0.0) for k in ("cost_spread_usd", "cost_slippage_usd", "cost_commission_usd", "cost_financing_usd"))
    return (f"| {name} | {es(mean('net_profit_usd'))} | {es(mean('sharpe_daily_sqrt365'))} | {es(mean('sortino_daily_sqrt365'))} | "
            f"{es(mean('max_drawdown_pct'))} | {es(mean('n_entries'), 0)} / {es(mean('n_closes'), 0)} | "
            f"{es(mean('n_close_TP'), 0)} / {es(mean('n_close_SL'), 0)} / {es(mean('n_close_AGENT_CLOSE'), 0)} / "
            f"{es(mean('n_close_REVERSAL'), 0)} / {es(mean('n_close_FORCED_FLAT'), 0)} | {es(100 * (mean('win_rate') or 0), 1)} | "
            f"{es(mean('gross_usd'))} | {es(cost)} ({es(mean('cost_spread_usd'))} / {es(mean('cost_slippage_usd'))} / "
            f"{es(mean('cost_commission_usd'))} / {es(mean('cost_financing_usd'))}) | {es(mean('orders_per_week'))} | "
            f"{es(mean('n_refused_DAILY_CAP'), 0)} | {sum(bool(m.get('valid')) for m in ms)}/{len(ms)} |")


def e1_tables(d, out, figs):
    L = ["# E1 · Clonación de comportamiento + DAgger-lite desde el experto ZigZag v2 (generado)", "",
         "DESARROLLO. Mismos periodos, datos y reglas de negocio que E0, ahora en el entorno barra a barra "
         "(`app/short_horizon/expert_guided/env.py`): acción {mantener, largo, corto, cerrar} + múltiplos σ de TP y SL "
         "(siempre fijados; recortados a la banda efectiva), decisión tras la barra completa y llenado en la apertura "
         "siguiente, cerrar-y-revertir = 1 cierre + 1 entrada, ≤ 4 entradas por día hábil, sin entradas 12 h tras "
         "reapertura, plano forzado 12 h antes del cierre, unidades = ⌊1,0 × 0,10 × saldo × 100 / precio⌋, costos "
         "RealisticProfile. Ronda 0 = BC sobre la trayectoria propia del experto (demostraciones); rondas 1–3 = la política "
         "actúa y el oráculo retrospectivo (`LegOracle`) reetiqueta cada estado visitado. Política final = ronda 3 "
         "(fijada de antemano). 2024 leído una vez tras congelar.", "",
         "Paridad del entorno: el experto reproducido en este entorno da el mismo Sharpe que el simulador por bloques del "
         "estudio v2 (INNER_A 18,58 / INNER_B 20,24 / 2024 18,07 en ambos).", ""]
    for f in figs:
        L.append(f"![{f}]({REPO}docs/short_horizon/results/expert_guided/e1/{f}) — [{f}]({REPO}docs/short_horizon/results/expert_guided/e1/{f})")
    g = d["gate_result"]
    L += ["", "## Compuerta pre-registrada", "",
          f"- I1 (log-loss de imitación en sus propios estados INNER_B < previa TRAIN suavizada): **{'sí' if g['I1'] else 'no'}** {g['I1_per_seed']}",
          f"- I2 (Sharpe INNER_B media de semillas > confirmado y > aleatoria previa): **{'sí' if g['I2'] else 'no'}** "
          f"({es(g['inner_b_sharpe_mean'])} vs {es(g['confirmed'])} y {es(g['prior_random_mean'])}); supera no operar: {'sí' if g['beats_no_trade'] else 'no'}",
          f"- Decisión: **{g['decision']}**", ""]
    hdr = ("| Política | Neto USD | Sharpe | Sortino | DD máx % | Entradas / cierres | Cierres TP / SL / agente / reversión / plano forzado | "
           "Acierto % | Bruto USD | Costos USD (spread / desl. / com. / swap) | Entradas/sem | Rechazos por tope diario | Válidas |")
    for p in PERIODS:
        cB = d["comparators"][p]
        L += [f"## {P_ES[p]}", "", hdr, "|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|"]
        L.append(env_row("Experto (demostraciones)", [cB["expert_demonstrations"]]))
        L.append(env_row("Oráculo de tramo (reetiquetador)", [cB["oracle_leg_following"]]))
        L.append(env_row("Experto confirmado", [cB["expert_confirmed"]]))
        if p != "OBS_2024":
            L.append(env_row("BC ronda 0 (media 3 semillas)", [r["rounds"][0]["trading"][p] for r in d["seeds"].values()]))
        L.append(env_row("DAgger ronda 3 (media 3 semillas)", [r["rounds"][-1]["trading"][p] for r in d["seeds"].values()]))
        for s, r in d["seeds"].items():
            L.append(env_row(f"  DAgger ronda 3, semilla {s}", [r["rounds"][-1]["trading"][p]]))
        L.append(env_row("Aleatoria previa TRAIN (media 3 semillas)", cB["train_prior_random"]))
        L.append(env_row("Sin operar", [cB["no_trade"]]))
        L.append("")
    L += ["## Entrenamiento e imitación por ronda", "",
          "| Semilla | Ronda | Filas TRAIN | Épocas | Mejor | Causa | Pérdida TRAIN | Pérdida INNER_A | Brecha | log-loss − previa INNER_B | Acierto INNER_B | Etiquetas INNER_B (M/L/C/Cerrar) | Predichas |",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for s, r in d["seeds"].items():
        for rr in r["rounds"]:
            h = rr["history"][rr["fit"]["best_epoch"] - 1]
            im = rr["imitation"]["INNER_B"]
            L.append(f"| {s} | {RND_ES[rr['round']]} | {rr['rows_train']} | {rr['fit']['epochs']} | {rr['fit']['best_epoch']} | "
                     f"{rr['fit']['stop_reason']} | {es(h['train_loss'], 4)} | {es(h['inner_loss'], 4)} | {es(h['gap_inner_minus_train'], 4)} | "
                     f"{es(im['log_loss_minus_prior'], 3)} | {es(im['accuracy'], 3)} | {im['label_counts']} | {im['pred_counts']} |")
    (out / "E1_TABLAS.md").write_text("\n".join(L) + "\n")


def e1(a):
    d = json.loads((Path(a.run) / "E1.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e1_fig_rounds(d, out)
    e1_fig_imitation(d, out)
    e1_fig_curves(d, out)
    e1_tables(d, out, ["e1_sharpe_por_ronda.png", "e1_imitacion_por_ronda.png", "e1_curvas_entrenamiento.png"])


STAGES = {"e0": e0, "e1": e1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["e0", "e1"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
