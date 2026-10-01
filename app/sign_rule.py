"""Pre-declared SIGN rule and model-free viability diagnostics (coordinator ruling, FX).

SIGN rule: at a forecast origin, position = sign of the predicted cumulative return at the
declared horizon (any magnitude), held for that horizon's bar count (``n_rows_h<h>``).
The next decision comes only when the hold ends, so trades never overlap. There is no
threshold, no grid and no selection on validation.

Diagnostics, per non-overlapping trade, close to close and per unit: mean gross edge in
price units and in z (sign · (R − n·mu) / sigma), hit rate, turnover, and the **break-even
round-trip cost**, i.e. the cost per round trip at which the mean net edge is zero. Compare
it with the declared cost profile: a break-even far below any realistic spread means NOT
economically viable.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_cost_profile(path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    profile = json.loads(raw)
    if profile.get("schema") != "heuristic_strategy.cost_profile.v1":
        raise ValueError("unsupported cost profile")
    return dict(profile, sha256=hashlib.sha256(raw).hexdigest(), path=str(path))


def solve_mu_sigma(row: Mapping[str, Any], h_a: int, h_b: int) -> tuple[float, float]:
    """logret_hat_h = sigma * y_hat_z_h + n_rows_h * mu, solved from two horizons of one row."""
    ya, yb = float(row[f"y_hat_z_h{h_a}"]), float(row[f"y_hat_z_h{h_b}"])
    ra, rb = float(row[f"logret_hat_h{h_a}"]), float(row[f"logret_hat_h{h_b}"])
    na, nb = float(row[f"n_rows_h{h_a}"]), float(row[f"n_rows_h{h_b}"])
    det = ya * nb - yb * na
    if det == 0:
        raise ValueError("degenerate row for solving mu and sigma")
    sigma = (ra * nb - rb * na) / det
    mu = (ya * rb - yb * ra) / det
    return sigma, mu


def sign_targets(bars: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]], horizon: int):
    """Per-bar targets and the non-overlapping trade list of the SIGN rule."""
    targets = [0] * len(bars)
    trades, i = [], 0
    while i < len(bars):
        row = rows.get(bars[i]["time"])
        if row is None:
            i += 1
            continue
        pred = float(row[f"logret_hat_h{horizon}"])
        n = int(row.get(f"n_rows_h{horizon}", horizon))
        sign = (pred > 0) - (pred < 0)
        if sign == 0 or n < 1 or i + n >= len(bars):
            i += 1
            continue
        for k in range(i, i + n):
            targets[k] = sign
        trades.append({"origin": i, "exit": i + n, "sign": sign})
        i += n
    return targets, trades


def sign_diagnostics(bars, rows, horizon: int, *, sigma: float, mu: float) -> dict[str, Any]:
    _, trades = sign_targets(bars, rows, horizon)
    if not trades:
        return {"trades": 0, "status": "NOT_AVAILABLE", "reason": "no trade"}
    price, logs, z, frac = [], [], [], []
    for t in trades:
        c0, c1 = float(bars[t["origin"]]["close"]), float(bars[t["exit"]]["close"])
        n = t["exit"] - t["origin"]
        r = math.log(c1 / c0)
        price.append(t["sign"] * (c1 - c0))
        frac.append(t["sign"] * (c1 - c0) / c0)
        logs.append(t["sign"] * r)
        z.append(t["sign"] * (r - n * mu) / sigma)
    k = len(trades)
    mean_price = sum(price) / k
    return {"trades": k, "horizon": horizon, "turnover_units": 2 * k,
            "hit_rate": sum(1 for v in logs if v > 0) / k,
            "mean_gross_edge_price": mean_price, "mean_gross_edge_fraction": sum(frac) / k,
            "mean_gross_edge_logret": sum(logs) / k, "mean_gross_edge_z": sum(z) / k,
            "break_even_round_trip_price": mean_price,
            "break_even_round_trip_fraction": sum(frac) / k,
            "edges_price": price}
