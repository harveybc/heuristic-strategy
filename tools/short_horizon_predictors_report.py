#!/usr/bin/env python3
"""Table of cheap causal baseline predictors scored through the sealed adapter (DEVELOPMENT)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.short_horizon_price_path_report import es  # noqa: E402

SPLIT_ES = {"VALIDATION_2024": "VALIDACIÓN 2024", "TRAIN_2020_2023": "ENTRENAMIENTO 2020-2023"}


def f(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else es(v, nd)


def build(run: Path, pub: Path):
    df = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in sorted(run.glob("B*/results.csv"))],
                   ignore_index=True)
    pub.mkdir(parents=True, exist_ok=True)
    df.to_csv(pub / "predictores_resultados.csv", index=False, float_format="%.8g")
    seals = {}
    for p in sorted(run.glob("B*/*/*/SEAL.json")):
        seals["/".join(p.parts[-4:-1])] = json.loads(p.read_text())["sha256"]
    (pub / "predictores_sellos.json").write_text(json.dumps(seals, indent=1) + "\n")
    cpu = sum(json.loads(p.read_text())["cpu_seconds"] for p in run.glob("B*/receipt.json"))
    lines = ["# Predictores base causales — tabla (generado)\n",
             "DESARROLLO; estudio de predictores baratos, no es un modelo entrenado con selección de variables. "
             "Cada archivo de predicciones se selló (sha256, `predictores_sellos.json`) antes de puntuarlo con el "
             "adaptador. Costes realistas ×1 (×2 en el CSV), regla de frecuencia (≥ 3 oper./semana, ≤ 4 órdenes/día). "
             "MAE U/D ×naïve = respecto a NAIVE_MEAN20 en las mismas filas; MAE de trayectoria ×naïve = respecto a "
             "persistencia (trayectoria reconstruida desde Û, D̂ y el orden con forma monótona). "
             f"CPU total de ajuste y puntuación: {cpu:.1f} s.\n"]
    for B, (hp, hs) in ((24, (6, 24)), (6, (3, 6))):
        for split in ("VALIDATION_2024", "TRAIN_2020_2023"):
            d = df[(df.block_hours == B) & (df.split == split) & (df.cost_mult == 1.0)]
            if d.empty:
                continue
            base_ll = d[(d.order == "BASE")].groupby("cadence").log_loss.first()
            lines.append(f"## B = {B} h · {SPLIT_ES[split]}\n")
            lines.append(f"| Predictor | Reajuste | MAE U ×naïve | MAE D ×naïve | Acierto orden | Log-loss (base) | "
                         f"Tray. ×naïve h{hp} | h{hs} | media | RF-M neto USD / Sharpe / oper.sem. / válido | "
                         "RF-L neto USD / Sharpe / oper.sem. / válido |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for (pred, cad), g in d.groupby(["predictor", "cadence"], sort=False):
                r = g.set_index("strategy")
                a = r.loc["RF-M"]
                bl = base_ll.get(cad, np.nan)
                cells = [pred, cad, f(a.mae_u_ratio_mean20), f(a.mae_d_ratio_mean20), f(a.direction_accuracy, 3),
                         f"{f(a.log_loss, 4)} ({f(bl, 4)})", f(a.path_mae_ratio_hp), f(a.path_mae_ratio_hs),
                         f(a.path_mae_ratio_mean)]
                for st in ("RF-M", "RF-L"):
                    x = r.loc[st]
                    cells.append(f"{es(x.net_profit_usd, 0)} / {es(x.sharpe_daily_sqrt365, 2)} / "
                                 f"{es(x.trades_per_week, 1)} / {'sí' if x.valid else 'no'}")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    (pub / "PREDICTORES_TABLA.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(df), "cpu_seconds": cpu}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--publish", type=Path, default=ROOT / "docs/short_horizon/results/predictors")
    a = ap.parse_args()
    build(a.run, a.publish)


if __name__ == "__main__":
    main()
