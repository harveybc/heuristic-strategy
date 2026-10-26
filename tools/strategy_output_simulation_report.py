"""Report of the strategy simulation sweeps (tools/strategy_output_simulation.py): pooled tables per variant x quality,
break-even quality per head family, robustness grid, gene sensitivity, and the PNG curves (one axis per figure, fixed
categorical colours, oracle ceiling and chance floor on every panel). Writes SWEEP_*.csv/json, PNGs and the Markdown
fragments the design documents include."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = {"base": "#2a78d6", "block": "#d1730a", "regime": "#1f8f4e", "onesided": "#7a4ecf", "closer_only": "#b4206b", "opener_only": "#5f6b73",
           "cost2": "#d1730a", "cost3": "#b4206b", "spike": "#1f8f4e", "missing10": "#7a4ecf"}
ORDER = ["base", "block", "regime", "onesided", "closer_only", "opener_only", "cost2", "cost3", "spike", "missing10"]
METRICS = [("dec_sharpe", "Sharpe anualizado (año de decisión)"), ("dec_net", "Neto USD (año de decisión, 10 000 iniciales)"),
           ("dec_exp", "Expectativa neta por trade (σ)"), ("dec_opw", "Órdenes por semana"), ("dec_dd", "Drawdown máximo %"),
           ("dec_early_close_value", "Valor del cierre temprano USD (con − sin)"), ("dec_win", "Tasa de acierto"),
           ("buy_win", "Acierto compras"), ("sell_win", "Acierto ventas")]


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, level): means over seeds x folds, the min lower CI of the expectancy, counts of refusals / failed checks."""
    rows = []
    for (v, lv), g in df.groupby(["variant", "level"]):
        ok = g[g["refused"].isna()]
        r = {"variant": v, "level": lv, "n": int(len(g)), "refused": int(g["refused"].notna().sum()),
             "checks_failed": int(ok["checks_failed"].fillna(0).sum()), "early_close_disabled": int(ok["early_close_disabled"].sum()),
             "auc_dec": float(ok["auc_dec"].mean()) if len(ok) else np.nan, "mae_over_naive_dec": float(ok["mae_over_naive_dec"].mean()) if len(ok) else np.nan,
             "spearman_dec": float(ok["spearman_dec"].mean()) if len(ok) else np.nan}
        for k, _ in METRICS:
            r[k] = float(ok[k].mean()) if len(ok) and ok[k].notna().any() else np.nan
            r[k + "_min"] = float(ok[k].min()) if len(ok) and ok[k].notna().any() else np.nan
            r[k + "_max"] = float(ok[k].max()) if len(ok) and ok[k].notna().any() else np.nan
        r["dec_exp_lo_min"] = float(ok["dec_exp_lo"].min()) if len(ok) and ok["dec_exp_lo"].notna().any() else np.nan
        r["dec_exp_lo_mean"] = float(ok["dec_exp_lo"].mean()) if len(ok) and ok["dec_exp_lo"].notna().any() else np.nan
        r["share_sharpe_positive"] = float((ok["dec_sharpe"] > 0).mean()) if len(ok) else np.nan
        r["dec_trades"] = float(ok["dec_trades"].mean()) if len(ok) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["variant", "level"]).reset_index(drop=True)


def break_even(p: pd.DataFrame, variant: str, col: str = "dec_exp_lo_mean") -> dict:
    """The lowest quality level at which the mean lower CI of the per-trade expectancy is > 0 AND stays > 0 at every
    higher level of the ladder (monotone break-even); None when no level qualifies."""
    g = p[p["variant"] == variant].sort_values("level")
    lv = g["level"].to_numpy()
    x = g[col].to_numpy(float)
    ok = np.isfinite(x) & (x > 0)
    be = None
    for i in range(len(lv)):
        if ok[i] and ok[i:].all():
            be = float(lv[i])
            break
    first = float(lv[np.argmax(ok)]) if ok.any() else None
    return {"variant": variant, "break_even_level": be, "first_positive_level": first, "levels": lv.tolist(), "lower_ci": [None if not math.isfinite(v) else float(v) for v in x],
            "oracle_ceiling": {k: (None if not math.isfinite(float(g[k].iloc[-1])) else float(g[k].iloc[-1])) for k in ("dec_sharpe", "dec_net", "dec_exp")} if len(g) else {},
            "chance_floor": {k: (None if not math.isfinite(float(g[k].iloc[0])) else float(g[k].iloc[0])) for k in ("dec_sharpe", "dec_net", "dec_exp")} if len(g) else {}}


def plot_curves(p: pd.DataFrame, out: Path, variants: list, metric: str, label: str, xcol: str = "level", xlabel: str = "AUC de las cabezas binarias (= Spearman de las regresiones)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    ax.set_facecolor("#fcfcfb")
    for v in variants:
        g = p[p["variant"] == v].sort_values(xcol)
        if not len(g):
            continue
        x = g[xcol].to_numpy(float)
        y = g[metric].to_numpy(float)
        ax.plot(x, y, color=PALETTE.get(v, "#52514e"), linewidth=2, marker="o", markersize=5, label=v)
        if metric + "_min" in g and metric + "_max" in g:
            ax.fill_between(x, g[metric + "_min"].to_numpy(float), g[metric + "_max"].to_numpy(float), color=PALETTE.get(v, "#52514e"), alpha=0.12, linewidth=0)
    base = p[p["variant"] == "base"].sort_values("level")
    if len(base):
        ax.axhline(float(base[metric].iloc[-1]), color="#0b0b0b", linewidth=1, linestyle="--", label="techo: oráculo (calidad 1.0, base)")
        ax.axhline(float(base[metric].iloc[0]), color="#52514e", linewidth=1, linestyle=":", label="suelo: azar (calidad 0.5, base)")
    if metric in ("dec_sharpe", "dec_net", "dec_exp", "dec_early_close_value"):
        ax.axhline(0.0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.grid(True, color="#e7e6e2", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    name = f"{metric}_vs_{xcol}.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def md_table(p: pd.DataFrame, variant: str) -> str:
    g = p[p["variant"] == variant].sort_values("level")
    h = "| calidad (AUC / ρ) | AUC medida (decisión) | MAE/naive neto | Sharpe | neto USD | expect. σ [IC inf. medio] | órdenes/sem | DD % | acierto compras / ventas | cierre temprano USD | fallos de chequeo | rechazos | n |"
    s = [h, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    f = lambda x, d=2: ("—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}")  # noqa: E731
    for _, r in g.iterrows():
        s.append(f"| {r['level']:.2f} | {f(r['auc_dec'], 3)} | {f(r['mae_over_naive_dec'], 2)} | {f(r['dec_sharpe'])} | {f(r['dec_net'], 0)} | {f(r['dec_exp'], 3)} [{f(r['dec_exp_lo_mean'], 3)}] | "
                 f"{f(r['dec_opw'])} | {f(r['dec_dd'], 1)} | {f(r['buy_win'])} / {f(r['sell_win'])} | {f(r['dec_early_close_value'], 0)} | {int(r['checks_failed'])} | {int(r['refused'])} | {int(r['n'])} |")
    return "\n".join(s)


def gene_table(runs: list) -> str:
    rows = []
    for r in runs:
        for s in r["per_seed"]:
            st = s.get("strategy_effective") or s.get("strategy")
            if not st:
                continue
            rows.append({"variante": r["variant"], "calidad": r["level"], "semilla": s["seed"], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}})
    if not rows:
        return "(sin estrategias congeladas)"
    d = pd.DataFrame(rows)
    cols = list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def sensitivity_table(runs: list) -> str:
    rows = []
    for r in runs:
        for s in r["per_seed"]:
            for g in (s.get("sensitivity") or []):
                rows.append({"variante": r["variant"], "calidad": r["level"], "semilla": s["seed"], **g})
    if not rows:
        return "(sin análisis de sensibilidad)"
    d = pd.DataFrame(rows)
    agg = d.groupby("gene").agg(d_sharpe_minus=("sharpe_minus_delta", "mean"), d_sharpe_plus=("sharpe_plus_delta", "mean"),
                                d_opw_minus=("opw_minus_delta", "mean"), d_opw_plus=("opw_plus_delta", "mean"), n=("gene", "size")).reset_index()
    out = ["| gen (cuantil) | Δ Sharpe (−0.1) | Δ Sharpe (+0.1) | Δ órdenes/sem (−0.1) | Δ órdenes/sem (+0.1) | n |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in agg.iterrows():
        out.append(f"| {r['gene']} | {r['d_sharpe_minus']:+.2f} | {r['d_sharpe_plus']:+.2f} | {r['d_opw_minus']:+.2f} | {r['d_opw_plus']:+.2f} | {int(r['n'])} |")
    return "\n".join(out)


def write(df: pd.DataFrame, runs: list, out: Path, docs: str):
    p = pooled(df)
    p.to_csv(out / "SWEEP_POOLED.csv", index=False)
    variants = [v for v in ORDER if v in set(p["variant"])]
    figs = {}
    for metric, label in METRICS:
        figs[metric] = plot_curves(p, out, variants, metric, label)
    figs["mae"] = plot_curves(p, out, ["base"], "dec_sharpe", "Sharpe (año de decisión)", xcol="mae_over_naive_dec", xlabel="MAE / MAE naive de la regresión del neto (año de decisión)")
    figs["mae_net"] = plot_curves(p, out, ["base"], "dec_net", "Neto USD (año de decisión)", xcol="mae_over_naive_dec", xlabel="MAE / MAE naive de la regresión del neto (año de decisión)")
    be = {v: break_even(p, v) for v in variants}
    (out / "BREAK_EVEN.json").write_text(json.dumps(be, indent=1))
    frag = ["## Tablas por variante (media sobre semillas y folds; año de decisión congelado)", ""]
    for v in variants:
        frag += [f"### Variante `{v}`", "", md_table(p, v), ""]
    frag += ["## Umbral de rentabilidad por familia", "", "| variante | nivel de equilibrio (IC inferior medio > 0 y se mantiene) | primer nivel positivo | techo oráculo Sharpe / neto / expect. | suelo azar Sharpe / neto / expect. |",
             "| --- | ---: | ---: | --- | --- |"]
    for v, b in be.items():
        oc, cf = b["oracle_ceiling"], b["chance_floor"]
        g = lambda d, k: ("—" if d.get(k) is None else f"{d[k]:.2f}")  # noqa: E731
        frag.append(f"| {v} | {('—' if b['break_even_level'] is None else f'{b[chr(98)+chr(114)+chr(101)+chr(97)+chr(107)+chr(95)+chr(101)+chr(118)+chr(101)+chr(110)+chr(95)+chr(108)+chr(101)+chr(118)+chr(101)+chr(108)]:.2f}')} | "
                    f"{('—' if b['first_positive_level'] is None else f'{b[chr(102)+chr(105)+chr(114)+chr(115)+chr(116)+chr(95)+chr(112)+chr(111)+chr(115)+chr(105)+chr(116)+chr(105)+chr(118)+chr(101)+chr(95)+chr(108)+chr(101)+chr(118)+chr(101)+chr(108)]:.2f}')} | "
                    f"{g(oc, 'dec_sharpe')} / {g(oc, 'dec_net')} / {g(oc, 'dec_exp')} | {g(cf, 'dec_sharpe')} / {g(cf, 'dec_net')} / {g(cf, 'dec_exp')} |")
    frag += ["", "## Genes congelados por configuración", "", gene_table(runs), "", "## Sensibilidad de los genes (±0.1 en el cuantil, bloque de selección)", "", sensitivity_table(runs), ""]
    frag += ["## Figuras", ""] + [f"- [{label}]({docs}/{figs[m]})" for m, label in METRICS] + [f"- [Sharpe vs MAE/naive]({docs}/{figs['mae']})", f"- [Neto vs MAE/naive]({docs}/{figs['mae_net']})", ""]
    (out / "SWEEP_FRAGMENT.md").write_text("\n".join(frag))
    return p, be, figs
