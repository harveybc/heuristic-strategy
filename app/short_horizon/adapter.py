"""Sealed-prediction adapter: score external forecasts through range-first (DEVELOPMENT).

Input: CSV or parquet keyed by ``block_start`` (t0, UTC ISO) with ``made_at`` (when the
forecast was issued, must be <= t0) and either
  (a) a price path ``p_h1 .. p_hB`` (B = block hours), or
  (b) excursions ``u_hat``, ``d_hat`` (price units from the decision price r; negative when a
      block is expected to gap past r, as realised excursions sometimes do) and an
      order probability ``p_low_first`` in [0, 1] or a label ``order`` (LOW_FIRST/HIGH_FIRST).
The file is sealed (sha256) before it is read for scoring. Rows outside the split,
duplicated keys, forecasts issued after t0, keys that are not a block start of the
configured grid, and non-finite values are refused. Missing blocks are abstentions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import analysis as A
from . import blocks as BL
from . import execution as X
from . import strategies as S

AXES = {24: (6, 24), 6: (3, 6)}
CAPITAL = 10_000.0
PIP = 1e-4


class PredictionError(ValueError):
    pass


@dataclass
class Market:
    bars: pd.DataFrame
    calendar: object
    blocks: object
    hours: int
    _cache: dict = field(default_factory=dict)

    def path_targets(self):
        if "T" not in self._cache:
            self._cache["T"] = S.comparator_targets(self.bars, self.blocks, tuple(range(1, self.hours + 1)))
        return self._cache["T"]

    def naive(self, kind):
        if kind not in self._cache:
            self._cache[kind] = S.naive_forecasts(self.blocks, kind)
        return self._cache[kind]


@dataclass
class PredictionSet:
    mode: str
    P: Optional[np.ndarray]
    u: np.ndarray
    d: np.ndarray
    p_low: np.ndarray
    has: np.ndarray
    sha256: str = ""


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(path, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"file": Path(path).name, "sha256": _sha(path), "bytes": Path(path).stat().st_size,
           "sealed_at": pd.Timestamp.now(tz="UTC").isoformat()}
    (out / "SEAL.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def verify_seal(path, seal_doc) -> None:
    if _sha(path) != seal_doc["sha256"]:
        raise PredictionError("prediction file changed after sealing")


def _read(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, float_precision="round_trip")


def load_predictions(path, m: Market, split) -> PredictionSet:
    df = _read(path)
    for c in ("block_start", "made_at"):
        if c not in df.columns:
            raise PredictionError(f"missing column {c}")
    t0 = pd.to_datetime(df["block_start"], utc=True)
    made = pd.to_datetime(df["made_at"], utc=True)
    if t0.duplicated().any():
        raise PredictionError("duplicated block_start keys")
    if (made > t0).any():
        raise PredictionError("a prediction was issued after its block start t0")
    a, b = (pd.Timestamp(x) for x in split)
    if ((t0 < a) | (t0 >= b)).any():
        raise PredictionError("block_start outside the requested split")
    in_split = m.blocks.in_split(split)
    index = {m.blocks.start[k].value: k for k in np.flatnonzero(in_split)}
    keys = [index.get(t.value) for t in t0]
    if any(k is None for k in keys):
        raise PredictionError("block_start is not an eligible block of the configured grid")
    n = len(m.blocks)
    has = np.zeros(n, bool)
    has[keys] = True
    u, d, p = (np.full(n, np.nan) for _ in range(3))
    P = None
    path_cols = [f"p_h{h}" for h in range(1, m.hours + 1)]
    if path_cols[0] in df.columns:
        if not set(path_cols) <= set(df.columns):
            raise PredictionError("path mode needs p_h1..p_hB")
        vals = df[path_cols].to_numpy(float)
        T = m.path_targets()[keys]
        if not np.isfinite(vals[np.isfinite(T)]).all():
            raise PredictionError("non-finite predicted price where a bar exists")
        P = np.full((n, m.hours), np.nan)
        P[keys] = vals
        mode = "path"
    elif {"u_hat", "d_hat"} <= set(df.columns):
        uu, dd = df["u_hat"].to_numpy(float), df["d_hat"].to_numpy(float)
        if not (np.isfinite(uu).all() and np.isfinite(dd).all()):
            raise PredictionError("u_hat/d_hat must be finite")
        if "p_low_first" in df.columns:
            pp = df["p_low_first"].to_numpy(float)
            if not np.isfinite(pp).all() or (pp < 0).any() or (pp > 1).any():
                raise PredictionError("p_low_first must be in [0, 1]")
        elif "order" in df.columns:
            if not df["order"].isin(["LOW_FIRST", "HIGH_FIRST"]).all():
                raise PredictionError("order must be LOW_FIRST or HIGH_FIRST")
            pp = (df["order"] == "LOW_FIRST").astype(float).to_numpy()
        else:
            raise PredictionError("excursion mode needs p_low_first or order")
        u[keys], d[keys], p[keys] = uu, dd, pp
        mode = "excursion"
    else:
        raise PredictionError("neither a path (p_h*) nor excursions (u_hat, d_hat) were supplied")
    return PredictionSet(mode, P, u, d, p, has, _sha(path))


def reconstruct_path(ref, U, D, ft, B: int) -> np.ndarray:
    """Monotone piecewise-linear path: r -> first extreme at h1 = round(B/3) -> second extreme at
    h2 = round(2B/3) -> flat to B. Its max/min/order equal (r + U, r - D, order) exactly."""
    ref, U, D, ft = (np.asarray(x, float) for x in (ref, U, D, ft))
    h1 = max(1, int(round(B / 3)))
    h2 = max(h1 + 1, int(round(2 * B / 3)))
    first = np.where(ft == BL.LOW_FIRST, ref - D, ref + U)
    second = np.where(ft == BL.LOW_FIRST, ref + U, ref - D)
    h = np.arange(1, B + 1, dtype=float)
    w1 = np.clip(h / h1, 0, 1)
    w2 = np.clip((h - h1) / (h2 - h1), 0, 1)
    path = np.where(h[None, :] <= h1, ref[:, None] + w1 * (first - ref)[:, None],
                    first[:, None] + w2 * (second - first)[:, None])
    return path


def score(ps: PredictionSet, m: Market, split, strategy: str, cost_mult: float = 1.0, cap: int = 4,
          min_per_week: float = 3.0, sizing: Optional[X.Sizing] = None) -> dict:
    """``sizing``: owner sizing (units from rel_volume of the current balance); None keeps the earlier
    1 %-risk reference sizing."""
    b = m.blocks
    rows = b.in_split(split) & ps.has
    eligible = b.in_split(split)
    T = m.path_targets()
    if ps.mode == "path":
        P = ps.P
        H_hat, L_hat, ft_hat = S.path_to_range(P)
        prob = np.full(len(b), np.nan)
    else:
        prob = ps.p_low
        ft_hat = np.where(prob >= 0.5, BL.LOW_FIRST, BL.HIGH_FIRST)
        H_hat, L_hat = b.ref + ps.u, b.ref - ps.d
        P = reconstruct_path(b.ref, np.nan_to_num(ps.u), np.nan_to_num(ps.d), ft_hat, m.hours)
        P[~ps.has] = np.nan
    naive_path = np.where(np.isfinite(T), b.ref[:, None], np.nan)
    pm, nm = A.path_mae(np.where(rows[:, None], P, np.nan), T, rows), A.path_mae(naive_path, T, rows)
    hp, hs = AXES[m.hours]
    U_true, D_true = b.H - b.ref, b.ref - b.L
    Uh, Dh = H_hat - b.ref, b.ref - L_hat
    nU, nD, _ = m.naive("MEAN20")
    pU, pD, _ = m.naive("PERSIST")
    ok = rows & np.isfinite(Uh) & np.isfinite(nU) & np.isfinite(pU)
    clear = ok & (b.ft != BL.AMBIGUOUS)
    y = (b.ft[clear] == BL.LOW_FIRST).astype(float)
    called = clear & (ft_hat != BL.AMBIGUOUS)
    out = {"mode": ps.mode, "strategy": strategy, "cost_mult": cost_mult, "sha256": ps.sha256,
           "coverage": float(rows.sum() / eligible.sum()) if eligible.any() else 0.0,
           "path_mae_hp": float(pm["per_h"][hp - 1]), "path_mae_hs": float(pm["per_h"][hs - 1]),
           "path_mae_mean": pm["mae"],
           "path_mae_ratio_hp": float(pm["per_h"][hp - 1] / nm["per_h"][hp - 1]),
           "path_mae_ratio_hs": float(pm["per_h"][hs - 1] / nm["per_h"][hs - 1]),
           "path_mae_ratio_mean": pm["mae"] / nm["mae"],
           "mae_u_pips": float(np.mean(np.abs(Uh[ok] - U_true[ok])) / PIP) if ok.any() else np.nan,
           "mae_d_pips": float(np.mean(np.abs(Dh[ok] - D_true[ok])) / PIP) if ok.any() else np.nan,
           "mae_u_ratio_mean20": float(np.mean(np.abs(Uh[ok] - U_true[ok])) / np.mean(np.abs(nU[ok] - U_true[ok])))
           if ok.any() else np.nan,
           "mae_d_ratio_mean20": float(np.mean(np.abs(Dh[ok] - D_true[ok])) / np.mean(np.abs(nD[ok] - D_true[ok])))
           if ok.any() else np.nan,
           "mae_u_ratio_persist": float(np.mean(np.abs(Uh[ok] - U_true[ok])) / np.mean(np.abs(pU[ok] - U_true[ok])))
           if ok.any() else np.nan,
           "mae_r_pips": float(np.mean(np.abs((Uh + Dh)[ok] - (U_true + D_true)[ok])) / PIP) if ok.any() else np.nan,
           "mae_r_ratio_mean20": float(np.mean(np.abs((Uh + Dh)[ok] - (U_true + D_true)[ok]))
                                       / np.mean(np.abs((nU + nD)[ok] - (U_true + D_true)[ok]))) if ok.any() else np.nan,
           "mae_r_ratio_persist": float(np.mean(np.abs((Uh + Dh)[ok] - (U_true + D_true)[ok]))
                                        / np.mean(np.abs((pU + pD)[ok] - (U_true + D_true)[ok]))) if ok.any() else np.nan,
           # accuracy over blocks where the forecast makes a call (a flat path makes none)
           "direction_accuracy": float(np.mean(ft_hat[called] == b.ft[called])) if called.any() else np.nan,
           "log_loss": float(np.mean(-(y * np.log(np.clip(prob[clear], 1e-6, 1)) +
                                       (1 - y) * np.log(np.clip(1 - prob[clear], 1e-6, 1)))))
           if clear.any() and np.isfinite(prob[clear]).all() else np.nan}
    profile = X.RealisticProfile().scaled(cost_mult)
    orders = S.range_first_orders(strategy, b, H_hat, L_hat, ft_hat, profile, S.RangeFirstParams(), CAPITAL)
    orders = [o if rows[k] else None for k, o in enumerate(orders)]
    sigma = S.causal_block_sigma(b)
    orders, rule_skipped = S.apply_order_rules(orders, b, sigma, X.OrderRules())
    res = X.run_backtest(m.bars, b, orders, profile, m.calendar, CAPITAL, split, profile=profile,
                         max_orders_per_day=cap, ruin_floor=0.0, order_rules=X.OrderRules(), sigma_block=sigma,
                         sizing=sizing)
    out["rule_skipped_orders"] = rule_skipped
    mm = res.metrics
    out.update({k: mm[k] for k in ("net_profit_usd", "end_equity_usd", "max_drawdown_usd", "sharpe_daily_sqrt365",
                                    "sortino_daily_sqrt365", "trades", "win_rate", "profit_factor",
                                    "trades_per_week", "max_orders_per_day", "weeks_with_eligible_blocks")})
    out["ruined"] = bool(mm["ruined"])
    out["leverage"], out["rel_volume"] = mm.get("leverage"), mm.get("rel_volume")
    out["max_exposure_x_balance"], out["margin_call_events"] = mm.get("max_exposure_x_balance"), mm.get("margin_call_events")
    out["valid"] = bool(mm["trades_per_week"] >= min_per_week and mm["max_orders_per_day"] <= cap and not mm["ruined"])
    out["valid_4_per_week"] = bool(mm["trades_per_week"] >= 4.0 and mm["max_orders_per_day"] <= cap
                                   and not mm["ruined"])
    return out


def main(argv=None):
    from . import calendar as CAL
    from . import data as D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--data", type=Path, required=True, help="lake EURUSD 1h parquet")
    ap.add_argument("--calendar", type=Path, default=Path("config/short_horizon/eurusd_h1_calendar.dev.json"))
    ap.add_argument("--block-hours", type=int, choices=(6, 24), required=True)
    ap.add_argument("--split", choices=("TRAIN_2020_2023", "VALIDATION_2024"), required=True)
    ap.add_argument("--strategy", choices=("RF-M", "RF-L"), default="RF-M")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    splits = {"TRAIN_2020_2023": ("2020-01-01", "2024-01-01"), "VALIDATION_2024": ("2024-01-01", "2025-01-01")}
    split = tuple(pd.Timestamp(x, tz="UTC") for x in splits[a.split])
    sealed = seal(a.predictions, a.out)
    bars = D.load_ohlc(a.data, "2019-12-01", "2025-01-01")
    cal = CAL.load_calendar(json.loads(a.calendar.read_text()))
    m = Market(bars, cal, BL.build_blocks(bars, cal, a.block_hours), a.block_hours)
    verify_seal(a.predictions, sealed)
    ps = load_predictions(a.predictions, m, split)
    verify_seal(a.predictions, sealed)
    res = score(ps, m, split, a.strategy, a.cost_mult)
    res.update({"split": a.split, "block_hours": a.block_hours, "data_sha256": D.file_sha256(a.data),
                "label": "DEVELOPMENT_SEALED_PREDICTION_SCORE"})
    (a.out / "RESULT.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
