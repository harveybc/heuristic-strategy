#!/usr/bin/env python3
"""Direction-free strategies V1/V2 (+ range-first with a coin-flip first touch) versus the
error of the VOLATILITY predictor (U_hat, D_hat). DEVELOPMENT oracle/noise study.

Realistic execution (f093e6e profile), owner sizing (rel_volume 0.10 x percentual_order_size 1.0
x current balance x leverage / price, margin basis), SL/TP business rule (effective levels,
pre-filtered), daily cap 4, stop-out at ruin, frequency rule applied in the report.
Leverage scenarios 10 / 50 / 100 at x1 costs, and 100 at x2. TRAIN 2020-2023 and VALIDATION
2024; the multiplier sensitivity grid runs on 2023 only. TEST closed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.short_horizon import blocks as BL
from app.short_horizon import execution as X
from app.short_horizon import noise as N
from app.short_horizon import strategies as S
from app.short_horizon import vstrategies as V
from tools.short_horizon_oracle import load_inputs, utc

LABEL = "DEVELOPMENT_ORACLE_NOISE_STUDY_NOT_A_MODEL_RESULT"
SPLITS = {"TRAIN_2020_2023": ("2020-01-01", "2024-01-01"), "VALIDATION_2024": ("2024-01-01", "2025-01-01")}
INNER = {"INNER_2023": ("2023-01-01", "2024-01-01")}
STRATS = ("V1", "V2", "RFM_COIN", "RFL_COIN")
SEEDS = (42, 43, 44, 45, 46)
IID = (0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006)
A1W = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
A3 = tuple((w, 1 - w) for w in A1W) + ((0.0, 1.5), (0.0, 2.0))   # (weight, noise scale x 0.5 x naive MAE)
A1_EPS = 0.00005
SCEN = ((10.0, 1.0), (50.0, 1.0), (100.0, 1.0), (100.0, 2.0))
CAPITAL = 10_000.0


class Study:
    def __init__(self, bars, cal, B):
        self.bars, self.cal, self.B = bars, cal, B
        self.blocks = BL.build_blocks(bars, cal, B)
        b = self.blocks
        self.U, self.D = b.H - b.ref, b.ref - b.L
        self.nU, self.nD, _ = S.naive_forecasts(b, "MEAN20")
        self.pU, self.pD, _ = S.naive_forecasts(b, "PERSIST")
        self.sigma = S.causal_block_sigma(b)
        self.anchor = np.array([V.causal_anchor(bars, b, k) if b.status[k] == BL.OK else np.nan
                                for k in range(len(b))])
        self.coin = N.ambiguous_coin(len(b))
        train = np.array([b.status[k] == BL.OK and b.start[k] < utc("2024-01-01") for k in range(len(b))])
        ok = train & np.isfinite(self.nU)
        self.mae_naive_u = float(np.mean(np.abs(self.nU[ok] - self.U[ok])))
        self.mae_naive_d = float(np.mean(np.abs(self.nD[ok] - self.D[ok])))
        self.masks = {}

    def draws(self, seed):
        rng = np.random.default_rng([seed, self.B, 31])
        n = len(self.blocks)
        return rng.standard_normal(n), rng.standard_normal(n)

    def forecast(self, family, level, seed):
        z1, z2 = self.draws(seed)
        if family == "iid":
            return self.U + level * z1, self.D + level * z2
        if family == "a1r":
            return (V.blend_excursion(self.U, self.nU, level, A1_EPS, z1),
                    V.blend_excursion(self.D, self.nD, level, A1_EPS, z2))
        if family == "a3r":
            w, scale = A3[int(level)]
            return (V.blend_excursion(self.U, self.nU, w, 0.5 * scale * self.mae_naive_u, z1),
                    V.blend_excursion(self.D, self.nD, w, 0.5 * scale * self.mae_naive_d, z2))
        raise ValueError(family)

    def orders(self, strategy, Uh, Dh, rules, v1p=V.V1Params(), v2p=V.V2Params()):
        b, out, skipped = self.blocks, [], 0
        prof = X.RealisticProfile()
        for k in range(len(b)):
            o = None
            if b.status[k] == BL.OK and np.isfinite(Uh[k]) and np.isfinite(Dh[k]) and np.isfinite(self.sigma[k]):
                if strategy == "V1":
                    o = V.v1_order(b, k, Uh[k], Dh[k], self.sigma[k], v1p)
                elif strategy == "V2":
                    o = V.v2_order(self.bars, b, k, Uh[k], Dh[k], self.sigma[k], v2p, anchor=self.anchor[k])
                else:
                    o = S.range_first_order("RF-M" if strategy == "RFM_COIN" else "RF-L", float(b.ref[k] + Uh[k]),
                                            float(b.ref[k] - Dh[k]), int(self.coin[k]), float(b.ref[k]), prof,
                                            S.RangeFirstParams(), CAPITAL)
            if o is not None:
                try:
                    if hasattr(o, "legs"):
                        V.check_oco(o, float(self.sigma[k]), rules)
                    else:
                        X.check_order(o, float(b.ref[k]), float(self.sigma[k]), rules)
                except X.OrderRuleError:
                    o, skipped = None, skipped + 1
            out.append(o)
        return out, skipped

    def accuracy(self, split, Uh, Dh):
        if split not in self.masks:
            a, b = split
            self.masks[split] = self.blocks.in_split((utc(a), utc(b)))
        m = self.masks[split] & np.isfinite(Uh) & np.isfinite(self.nU) & np.isfinite(self.pU)
        R, Rh = self.U + self.D, Uh + Dh
        nR, pR = self.nU + self.nD, self.pU + self.pD
        f = lambda x, y: float(np.mean(np.abs(x[m] - y[m])))  # noqa: E731
        return {"mae_u": f(Uh, self.U), "mae_d": f(Dh, self.D), "mae_r": f(Rh, R),
                "naive_mae_u": f(self.nU, self.U), "naive_mae_d": f(self.nD, self.D), "naive_mae_r": f(nR, R),
                "persist_mae_u": f(self.pU, self.U), "persist_mae_d": f(self.pD, self.D), "persist_mae_r": f(pR, R)}

    def backtest(self, orders, split, lev, cost, rules):
        a, b = split
        prof = X.RealisticProfile().scaled(cost)
        return X.run_backtest(self.bars, self.blocks, orders, prof, self.cal, CAPITAL, (utc(a), utc(b)),
                              profile=prof, max_orders_per_day=4, ruin_floor=0.0, order_rules=rules,
                              sigma_block=self.sigma, sizing=X.Sizing(rel_volume=0.10, leverage=lev))


KEEP = ("net_profit_usd", "end_equity_usd", "max_drawdown_usd", "max_drawdown_pct", "sharpe_daily_sqrt365",
        "sortino_daily_sqrt365", "trades", "win_rate", "profit_factor", "trades_per_week", "max_orders_per_day",
        "weeks_with_eligible_blocks", "eligible_blocks", "ruined", "share_exit_tp", "share_exit_sl",
        "share_exit_time", "share_exit_early", "tp_rarely_hit", "margin_call_events", "max_margin_used_pct",
        "max_exposure_x_balance", "below_min_units", "margin_refused_orders", "rel_volume", "leverage",
        "rel_volume_basis", "unfilled_orders", "capped_orders", "ruin_skipped_orders")


def worst_week(res):
    eq = res.daily_equity
    w = eq.groupby(pd.DatetimeIndex(eq.index).tz_localize(None).to_period("W")).last()
    r = w.pct_change().dropna()
    return float(r.min()) if len(r) else 0.0


def cells_main():
    fam = [("iid", x) for x in IID] + [("a1r", x) for x in A1W] + [("a3r", float(j)) for j in range(len(A3))]
    return [dict(B=B, strategy=s, family=f, level=x, seed=seed)
            for B, s, (f, x), seed in itertools.product((24, 6), STRATS, fam, SEEDS)]


def cells_sensitivity():
    v1 = [("V1", dict(k_tp=a, k_trail=b)) for a in (0.3, 0.4, 0.45) for b in (0.25, 0.35, 0.5)]
    v2 = [("V2", dict(m_sl=a, b_band=b)) for a in (1.5, 2.0, 3.0) for b in (0.6, 0.8, 1.0)]
    return [dict(B=B, strategy=s, params=p, family="a3r", level=float(j), seed=seed)
            for B, (s, p), j, seed in itertools.product((24, 6), v1 + v2, (0, 4, 7), SEEDS)]


def run(args):
    bars, cal, digest = load_inputs(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in args.chunk.split("/"))
    if args.inner:   # RF-M coin-direction curves on the inner year 2023 (reference for 2023 predictor markers)
        base = [c for c in cells_main() if c["strategy"] == "RFM_COIN" and c["family"] in ("a1r", "a3r")]
        todo, kind = base[i::n], "inner"
    else:
        todo = (cells_sensitivity() if args.sensitivity else cells_main())[i::n]
        kind = "sens" if args.sensitivity else "main"
    tag = kind + f"_{i}of{n}"
    target = out / f"cells_{tag}.csv"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    rules = X.OrderRules()
    studies = {B: Study(bars, cal, B) for B in sorted({c["B"] for c in todo})}
    rows, t0 = [], time.process_time()
    for c in todo:
        st = studies[c["B"]]
        Uh, Dh = st.forecast(c["family"], c["level"], c["seed"])
        params = c.get("params", {})
        v1p = V.V1Params(**params) if c["strategy"] == "V1" else V.V1Params()
        v2p = V.V2Params(**params) if c["strategy"] == "V2" else V.V2Params()
        orders, skipped = st.orders(c["strategy"], Uh, Dh, rules, v1p, v2p)
        splits = INNER if (args.sensitivity or args.inner) else SPLITS
        scen = ((100.0, 1.0),) if (args.sensitivity or args.inner) else SCEN
        for name, split in splits.items():
            acc = st.accuracy(split, Uh, Dh)
            for lev, cost in scen:
                res = st.backtest(orders, split, lev, cost, rules)
                row = {"block_hours": c["B"], "strategy": c["strategy"], "family": c["family"], "level": c["level"],
                       "seed": c["seed"], "split": name, "leverage_scenario": lev, "cost_mult": cost,
                       "params": json.dumps(params, sort_keys=True), "rule_skipped_orders": skipped,
                       "worst_week_return": worst_week(res), "label": LABEL, **acc}
                row.update({k: res.metrics[k] for k in KEEP})
                rows.append(row)
    pd.DataFrame(rows).to_csv(target, index=False)
    (out / f"receipt_{tag}.json").write_text(json.dumps({"chunk": args.chunk, "cells": len(todo), "rows": len(rows),
                                                         "cpu_seconds": time.process_time() - t0,
                                                         "data_sha256": digest}, indent=1) + "\n")
    print(json.dumps({"tag": tag, "rows": len(rows), "cpu": round(time.process_time() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk", default="0/1")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--inner", action="store_true", help="RF-M coin-direction A1r/A3r curves on 2023 only")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
