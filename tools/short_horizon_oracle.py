#!/usr/bin/env python3
"""Short-horizon range-first oracle + Gaussian-noise frontier (S1). DEVELOPMENT only.

Successor of run_oracle_ceiling.py / sweep_noise.py for the (high, low,
first-touch) target family: same method (fixed oracle, monotone noise grid,
accuracy reported next to profit), but no HTTP service, no sibling-repo oracle
and no writes outside --output. TRAIN 2020-2023 and VALIDATION 2024 only; the
loader refuses TEST rows. No GPU, no model, no trading.

Stages (each is deterministic and recomputes blocks from the pinned inputs):
  calendar  derive the versioned session calendar JSON from observed bars
  pilot     time a handful of cells (cost evidence for chunking)
  run       --grid {baselines,rf,comparator,costs} --chunk i/n -> cells_<grid>_<i>.csv
  merge     concatenate chunk CSVs into cells.csv (refuses missing/duplicate cells)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.short_horizon import analysis as A
from app.short_horizon import blocks as B
from app.short_horizon import calendar as CAL
from app.short_horizon import data as D
from app.short_horizon import execution as X
from app.short_horizon import noise as N
from app.short_horizon import strategies as S

LAKE_RELPATH = "market_data/forex/g10/eurusd/1h.parquet"
LAKE_SHA256 = "c6628375b2872f16dcdbd6b1b90ed8e68bc5bdd852efdef01f2515b8712d4e2e"
LOAD_START, LOAD_END = "2019-12-01", "2025-01-01"
SPLITS = {"TRAIN_2020_2023": ("2020-01-01", "2024-01-01"),
          "VALIDATION_2024": ("2024-01-01", "2025-01-01")}
CALENDAR = ROOT / "config/short_horizon/eurusd_h1_calendar.dev.json"
BLOCK_HOURS = (6, 24)
VARIANTS = ("RF-M", "RF-L")
SEEDS = (42, 43, 44)
REL_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
ABS_PIPS = (1, 2, 3, 5, 7, 10, 15, 20, 30)
P_FLIP = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
CMP_REL = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
CMP_ABS = (1, 2, 3, 5, 7, 10, 15, 20)
BLEND_W = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
COST_MULTS = (0.0, 0.5, 1.5, 2.0, 3.0)
COST_BLEND_W = (1.0, 0.7, 0.5, 0.3)
COST_P = (0.0, 0.10, 0.20, 0.30)
# Owner follow-up 2026-10-19: business-day RF-L degradation on VALIDATION 2024 only.
VAL_SEEDS = (42, 43, 44, 45, 46)
# Extended once after the first pass found no zero crossing up to 120 pips (owner: "up to where
# performance is clearly lost"); common random numbers keep every earlier cell identical.
VAL_RANGE_PIPS = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0,
                  160.0, 200.0, 300.0, 400.0, 600.0)
VAL_P = tuple(round(0.05 * k, 2) for k in range(11))
VAL_JOINT_PIPS = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 200.0, 400.0)
VAL_JOINT_P = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
VAL_COSTS = (1.0, 2.0)
# Coordinator follow-up 2026-10-19: participation, RF-M range sweep, realistic sweeps (val24x).
VAL_FT_TARGETS = tuple(round(1 - 0.05 * k, 2) for k in range(11))
VAL_FIXED_FT = (0.6, 0.7)
# Owner redesign 2026-10-19: strategy consumes only a predicted hourly price path h = 1..B.
PATH_SIGMAS = (0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005,
               0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06)
# Owner realism pass 2026-10-19 (pathr): model-like error families and realistic execution.
A1_W = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
A1_EPS = 0.00005          # small i.i.d. term in A1 (0.5 pip)
A2_STEPS = (0.0, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.005)
A3_REF_STEP = 0.0002      # A3: random-walk step grows as the blend loses skill
A3_PAIRS = tuple((w, round(A3_REF_STEP * (1 - w), 8)) for w in A1_W) + ((0.0, 0.0004), (0.0, 0.0008), (0.0, 0.0016))
ORDER_CAP_PER_DAY = 4
CAPITAL = 10_000.0
PIP = 1e-4
LABEL = "DEVELOPMENT_ORACLE_NOISE_STUDY_NOT_A_MODEL_RESULT"


def utc(x):
    return pd.Timestamp(x, tz="UTC")


def load_inputs(data_path):
    digest = D.file_sha256(data_path)
    if digest != LAKE_SHA256:
        raise SystemExit(f"lake file digest {digest} != pinned {LAKE_SHA256}")
    bars = D.load_ohlc(data_path, LOAD_START, LOAD_END)
    cal = CAL.load_calendar(json.loads(CALENDAR.read_text()))
    return bars, cal, digest


def derive_calendar(data_path):
    digest = D.file_sha256(data_path)
    bars = D.load_ohlc(data_path, LOAD_START, LOAD_END)
    doc = {"calendar_id": "EURUSD_H1_VENDOR_SESSIONS_DEV",
           "version": "2026-10-18.v1",
           "description": ("DEVELOPMENT proxy for a published broker schedule. Segments = maximal runs "
                           "of observed bars with no gap > 24 h; open = first bar label, close = last "
                           "bar label + 1 h. Vendor bar clock as labelled (clock semantics under repair). "
                           "Final 2024 segment truncated at the split boundary (conservative)."),
           "derivation": {"rule": "OBSERVED_BAR_GAP_GT_24H", "max_gap_hours": 24,
                          "source": LAKE_RELPATH, "source_sha256": digest,
                          "window": [LOAD_START, LOAD_END]},
           "entry_delay_after_open_hours": 12,
           "force_flat_before_close_hours": 12,
           "entry_cutoff_before_flat_hours": 0,
           "segments": CAL.derive_segments(bars.index, max_gap_hours=24)}
    CALENDAR.parent.mkdir(parents=True, exist_ok=True)
    CALENDAR.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {CALENDAR.relative_to(ROOT)}: {len(doc['segments'])} segments")


class Context:
    """Blocks, labels, draws and split masks for one block size (deterministic)."""

    def __init__(self, bars, cal, hours):
        self.bars, self.cal, self.hours = bars, cal, hours
        self.blocks = B.build_blocks(bars, cal, hours)
        n = len(self.blocks)
        self.ft_oracle = self.blocks.ft_oracle()
        self.range = self.blocks.H - self.blocks.L
        self.horizons = S.comparator_horizons(hours)
        self.paths = S.comparator_targets(bars, self.blocks, self.horizons)
        self.draws = {s: N.draws(s, n, len(self.horizons)) for s in sorted(set(SEEDS) | set(VAL_SEEDS))}
        self.masks = {name: self.blocks.in_split((utc(a), utc(b))) for name, (a, b) in SPLITS.items()}
        self.naive = {k: S.naive_range(self.blocks, k) for k in ("PERSIST", "MEAN20")}

    def path_targets(self):
        """Realised closes of the bars labelled t0 + h - 1h, h = 1..B (NaN when absent)."""
        if not hasattr(self, "_path"):
            self._path = S.comparator_targets(self.bars, self.blocks, tuple(range(1, self.hours + 1)))
        return self._path

    def path_noise(self, seed):
        cache = self.__dict__.setdefault("_path_noise", {})
        if seed not in cache:
            cache[seed] = N.path_draws(seed, len(self.blocks), self.hours)
        return cache[seed]

    def range_forecast(self, family, level, seed):
        """(H_hat, L_hat) before the first-touch flip."""
        b = self.blocks
        if family == "blend":
            nH, nL, _ = self.naive["MEAN20"]
            return N.blend(b.H, nH, level), N.blend(b.L, nL, level)
        d = self.draws[seed]
        sig = self.sigma(family, level)
        return b.H + sig * d["z_high"], b.L + sig * d["z_low"]

    def path_forecast(self, family, level, seed):
        if family == "blend":
            return N.blend(self.paths, self.blocks.ref[:, None], level)
        return N.apply_path_noise(self.paths, self.draws[seed], self.sigma(family, level))

    def sigma(self, family, level):
        if family == "rel":
            return level * np.nan_to_num(self.range)
        if family == "abs":
            return np.full(len(self.blocks), level * PIP)
        raise ValueError(family)


def accuracy(ctx, mask, H_hat, L_hat, ft_hat):
    b = ctx.blocks
    ok = mask & np.isfinite(H_hat) & np.isfinite(L_hat)
    clear = mask & (b.ft != B.AMBIGUOUS)
    nH, nL, _ = ctx.naive["MEAN20"]
    nok = ok & np.isfinite(nH) & np.isfinite(nL)
    naive = float((np.mean(np.abs(nH[nok] - b.H[nok])) + np.mean(np.abs(nL[nok] - b.L[nok]))) / 2 / PIP) \
        if nok.any() else np.nan
    model = float((np.mean(np.abs(H_hat[nok] - b.H[nok])) + np.mean(np.abs(L_hat[nok] - b.L[nok]))) / 2 / PIP) \
        if nok.any() else np.nan
    return {"naive_mean20_mae_pips": naive, "range_mae_ratio_to_naive": model / naive if naive else np.nan,
            "mae_high_pips": float(np.mean(np.abs(H_hat[ok] - b.H[ok])) / PIP) if ok.any() else np.nan,
            "mae_low_pips": float(np.mean(np.abs(L_hat[ok] - b.L[ok])) / PIP) if ok.any() else np.nan,
            "first_touch_accuracy": float(np.mean(ft_hat[clear] == b.ft[clear])) if clear.any() else np.nan,
            "forecast_coverage": float(ok.sum() / mask.sum()) if mask.any() else np.nan,
            "ambiguous_blocks": int((mask & (b.ft == B.AMBIGUOUS)).sum()),
            "mean_true_range_pips": float(np.mean(ctx.range[mask]) / PIP)}


def path_accuracy(ctx, mask, preds):
    t = ctx.paths
    out = {}
    for j, h in enumerate(ctx.horizons):
        ok = mask & np.isfinite(t[:, j])
        out[f"mae_h{h}_pips"] = float(np.mean(np.abs(preds[ok, j] - t[ok, j])) / PIP)
        out[f"naive_mae_h{h}_pips"] = float(np.mean(np.abs(ctx.blocks.ref[ok] - t[ok, j])) / PIP)
        move = t[ok, j] - ctx.blocks.ref[ok]
        pred = preds[ok, j] - ctx.blocks.ref[ok]
        nz = move != 0
        out[f"dir_acc_h{h}"] = float(np.mean(np.sign(pred[nz]) == np.sign(move[nz]))) if nz.any() else np.nan
    return out


def evaluate(ctx, orders, costs, extra_acc=None, segment_hold=None, splits=None):
    rows = []
    for name, (a, b) in SPLITS.items():
        if splits is not None and name not in splits:
            continue
        split = (utc(a), utc(b))
        if segment_hold is None:
            res = X.run_backtest(ctx.bars, ctx.blocks, orders, costs, ctx.cal, CAPITAL, split)
        else:
            res = X.run_segment_hold(ctx.bars, ctx.cal, segment_hold, costs, CAPITAL, split, ctx.blocks)
        if res.final_position_units != 0:
            raise AssertionError("not flat at split end")
        row = {"split": name, **res.metrics}
        if extra_acc is not None:
            row.update(extra_acc(ctx.masks[name]))
        rows.append((row, res))
    return rows


def cells(grid):
    if grid == "baselines":
        names = ["NO_TRADE", "BLOCK_BUY", "BLOCK_SELL", "SEGMENT_BUY", "SEGMENT_SELL",
                 "NAIVE_PERSIST_RF-M", "NAIVE_PERSIST_RF-L", "NAIVE_MEAN20_RF-M", "NAIVE_MEAN20_RF-L",
                 "COMPARATOR_NAIVE_PERSISTENCE", "COMPARATOR_ORACLE"]
        return [dict(grid=grid, block_hours=h, strategy=n, family="none", level=0.0, p_flip=0.0,
                     seed=0, cost_mult=m) for h in BLOCK_HOURS for n in names for m in (1.0,) + COST_MULTS]
    if grid == "rf":
        lv = [("rel", x) for x in REL_LEVELS] + [("abs", float(x)) for x in ABS_PIPS] + \
            [("blend", x) for x in BLEND_W]
        return [dict(grid=grid, block_hours=h, strategy=v, family=f, level=x, p_flip=p, seed=s, cost_mult=1.0)
                for h, v, (f, x), p, s in itertools.product(BLOCK_HOURS, VARIANTS, lv, P_FLIP, SEEDS)]
    if grid == "comparator":
        lv = [("rel", x) for x in CMP_REL] + [("abs", float(x)) for x in CMP_ABS] + \
            [("blend", x) for x in BLEND_W]
        return [dict(grid=grid, block_hours=h, strategy="COMPARATOR", family=f, level=x, p_flip=0.0, seed=s,
                     cost_mult=1.0) for h, (f, x), s in itertools.product(BLOCK_HOURS, lv, SEEDS)]
    if grid == "costs":
        return [dict(grid=grid, block_hours=h, strategy=v, family="blend", level=x, p_flip=p, seed=s, cost_mult=m)
                for h, v, m, x, p, s in itertools.product(BLOCK_HOURS, VARIANTS, COST_MULTS, COST_BLEND_W, COST_P, SEEDS)]
    if grid == "val24":
        out = [dict(grid=grid, sweep="baseline", block_hours=24, strategy=n, family="none", level=0.0,
                    p_flip=0.0, seed=0, cost_mult=m) for n in ("NO_TRADE", "NAIVE_MEAN20_RF-L") for m in VAL_COSTS]
        sweeps = [("range", x, 0.0) for x in VAL_RANGE_PIPS] + [("firsttouch", 0.0, p) for p in VAL_P] + \
            [("joint", x, p) for x in VAL_JOINT_PIPS for p in VAL_JOINT_P]
        out += [dict(grid=grid, sweep=sw, block_hours=24, strategy="RF-L", family="abs", level=x, p_flip=p,
                     seed=s, cost_mult=m) for (sw, x, p), s, m in itertools.product(sweeps, VAL_SEEDS, VAL_COSTS)]
        return out
    if grid == "pathr":
        fam_levels = {"iid": PATH_SIGMAS, "a1": A1_W, "a2": A2_STEPS, "a3": tuple(float(j) for j in range(len(A3_PAIRS)))}
        out = [dict(grid=grid, sweep=prof, block_hours=h, strategy="PERSISTENCE", family="path", level=0.0,
                    p_flip=0.0, seed=0, cost_mult=m)
               for h in BLOCK_HOURS for prof, m in (("legacy", 1.0), ("realistic", 1.0), ("realistic", 2.0))]
        out += [dict(grid=grid, sweep="legacy", block_hours=h, strategy=st, family="iid", level=x, p_flip=0.0,
                     seed=s, cost_mult=1.0)
                for h, st, x, s in itertools.product(BLOCK_HOURS, ("RF-M", "RF-L"), PATH_SIGMAS, VAL_SEEDS)]
        out += [dict(grid=grid, sweep="realistic", block_hours=h, strategy=st, family=f, level=x, p_flip=0.0,
                     seed=s, cost_mult=m)
                for h, st, f, s, m in itertools.product(BLOCK_HOURS, ("RF-M", "RF-L"), fam_levels, VAL_SEEDS, VAL_COSTS)
                for x in fam_levels[f]]
        # owner variant: at most 2 orders per business day (only binding for B = 6)
        out += [dict(grid=grid, sweep="realistic_cap2", block_hours=6, strategy=st, family=f, level=x, p_flip=0.0,
                     seed=s, cost_mult=1.0, order_cap=2)
                for st, f, s in itertools.product(("RF-M", "RF-L"), fam_levels, VAL_SEEDS) for x in fam_levels[f]]
        return out
    if grid == "pathx":
        out = [dict(grid=grid, block_hours=h, strategy="PERSISTENCE", family="path", level=0.0, p_flip=0.0,
                    seed=0, cost_mult=m) for h in BLOCK_HOURS for m in VAL_COSTS]
        out += [dict(grid=grid, block_hours=h, strategy=st, family="path", level=x, p_flip=0.0, seed=s,
                     cost_mult=m)
                for h, st, x, s, m in itertools.product(BLOCK_HOURS, ("RF-M", "RF-L"), PATH_SIGMAS, VAL_SEEDS,
                                                        VAL_COSTS)]
        return out
    if grid == "val24x":
        out = [dict(grid=grid, sweep="baseline", block_hours=24, strategy=n, family="none", level=0.0,
                    p_flip=0.0, ft_target=None, seed=0, cost_mult=m)
               for n in ("NO_TRADE", "NAIVE_MEAN20_RF-L", "NAIVE_MEAN20_RF-M") for m in VAL_COSTS]
        specs = []
        for st in ("RF-M", "RF-L"):  # same participation accounting for both executions
            tag = st[-1].lower()
            specs += [(f"rf{tag}_range", st, "abs", x, 1.0) for x in VAL_RANGE_PIPS]
            specs += [(f"rf{tag}_firsttouch", st, "abs", 0.0, a) for a in VAL_FT_TARGETS]
        for st in ("RF-L", "RF-M"):
            specs += [("ft_at_naive_range", st, "blend", 0.0, a) for a in VAL_FT_TARGETS]
            specs += [("ft_at_08_range", st, "blend", 0.2, a) for a in VAL_FT_TARGETS]
            specs += [("range_abs_at_ft", st, "abs", x, a) for x in VAL_RANGE_PIPS for a in VAL_FIXED_FT]
            specs += [("range_blend_at_ft", st, "blend", w, a) for w in BLEND_W for a in VAL_FIXED_FT]
        out += [dict(grid=grid, sweep=sw, block_hours=24, strategy=st, family=f, level=x, p_flip=0.0,
                     ft_target=a, seed=s, cost_mult=m)
                for (sw, st, f, x, a), s, m in itertools.product(specs, VAL_SEEDS, VAL_COSTS)]
        return out
    raise ValueError(grid)


def run_cell(ctx, cell):
    costs = X.Costs().scaled(cell["cost_mult"])
    p = S.RangeFirstParams()
    st = cell["strategy"]
    if cell["grid"] == "pathx":
        return run_path_cell(ctx, cell, costs, p)
    if cell["grid"] == "pathr":
        return run_pathr_cell(ctx, cell, p)
    if cell["grid"] in ("val24", "val24x"):
        return run_val24_cell(ctx, cell, costs, p)
    if st in ("RF-M", "RF-L"):
        H_hat, L_hat = ctx.range_forecast(cell["family"], cell["level"], cell["seed"])
        flip = ctx.draws[cell["seed"]]["u_flip"] < cell["p_flip"]
        ft_hat = np.where(flip, -ctx.ft_oracle, ctx.ft_oracle)
        orders = S.range_first_orders(st, ctx.blocks, H_hat, L_hat, ft_hat, costs, p, CAPITAL)
        return evaluate(ctx, orders, costs, lambda m: accuracy(ctx, m, H_hat, L_hat, ft_hat))
    if st == "COMPARATOR" or st == "COMPARATOR_ORACLE":
        if st == "COMPARATOR":
            preds = ctx.path_forecast(cell["family"], cell["level"], cell["seed"])
        else:
            preds = ctx.paths
        orders = S.comparator_orders(ctx.blocks, preds, costs, CAPITAL)
        return evaluate(ctx, orders, costs, lambda m: path_accuracy(ctx, m, preds))
    if st == "COMPARATOR_NAIVE_PERSISTENCE":
        preds = np.repeat(ctx.blocks.ref[:, None], len(ctx.horizons), axis=1)
        preds = np.where(np.isfinite(ctx.paths), preds, np.nan)
        orders = S.comparator_orders(ctx.blocks, preds, costs, CAPITAL)
        return evaluate(ctx, orders, costs, lambda m: path_accuracy(ctx, m, preds))
    if st in ("SEGMENT_BUY", "SEGMENT_SELL"):
        return evaluate(ctx, None, costs, segment_hold=1 if st == "SEGMENT_BUY" else -1)
    if st.startswith("NAIVE_"):
        _, kind, variant = st.split("_")
        H_hat, L_hat, ft_hat = ctx.naive[kind]
        orders = S.range_first_orders(variant, ctx.blocks, H_hat, L_hat, ft_hat, costs, p, CAPITAL)
        return evaluate(ctx, orders, costs, lambda m: accuracy(ctx, m, H_hat, L_hat, ft_hat))
    base = S.baseline_orders(ctx.blocks, costs, p, CAPITAL)
    return evaluate(ctx, base[st], costs)


def run_val24_cell(ctx, cell, costs, p):
    """Business-day RF-L on VALIDATION 2024 only; achieved metrics on the same rows."""
    only = ("VALIDATION_2024",)
    st = cell["strategy"]
    variant = st[-4:]
    target = cell.get("ft_target")
    if st in ("RF-L", "RF-M"):
        H_hat, L_hat = ctx.range_forecast(cell["family"] if cell["grid"] == "val24x" else "abs",
                                          cell["level"], cell["seed"])
        u = ctx.draws[cell["seed"]]["u_flip"]
        if target is not None and np.isfinite(target):
            eligible = ctx.masks["VALIDATION_2024"] & (ctx.blocks.ft != B.AMBIGUOUS)
            ft_hat = N.flip_to_accuracy(ctx.ft_oracle, eligible, u, target)
            conf = float(target)
        else:
            ft_hat = np.where(u < cell["p_flip"], -ctx.ft_oracle, ctx.ft_oracle)
            conf = 1.0 - cell["p_flip"]
    elif st in ("NAIVE_MEAN20_RF-L", "NAIVE_MEAN20_RF-M"):
        H_hat, L_hat, ft_hat = ctx.naive["MEAN20"]
        conf = 1.0
    elif st == "NO_TRADE":
        return evaluate(ctx, [None] * len(ctx.blocks), costs, splits=only)
    else:
        raise ValueError(st)
    orders = S.range_first_orders(variant, ctx.blocks, H_hat, L_hat, ft_hat, costs, p, CAPITAL)

    def acc(mask):
        out = accuracy(ctx, mask, H_hat, L_hat, ft_hat)
        out["brier_calibrated"] = A.calibrated_brier(ft_hat[mask], ctx.blocks.ft[mask], conf)
        out["stated_confidence"] = conf
        return out
    return evaluate(ctx, orders, costs, acc, splits=only)


def run_path_cell(ctx, cell, costs, p):
    """Range-first driven only by a predicted price path; both splits; achieved price MAE."""
    T = ctx.path_targets()
    naive = np.where(np.isfinite(T), ctx.blocks.ref[:, None], np.nan)
    if cell["strategy"] == "PERSISTENCE":
        preds, variant = naive, "RF-M"
    else:
        preds, variant = T + cell["level"] * ctx.path_noise(cell["seed"]), cell["strategy"]
    H_hat, L_hat, ft_hat = S.path_to_range(preds)
    orders = S.range_first_orders(variant, ctx.blocks, H_hat, L_hat, ft_hat, costs, p, CAPITAL)

    def acc(mask):
        m, nv = A.path_mae(preds, T, mask), A.path_mae(naive, T, mask)
        b = ctx.blocks
        clear = mask & (b.ft != B.AMBIGUOUS)
        ok = mask & np.isfinite(H_hat)
        out = {"path_mae": m["mae"], "naive_path_mae": nv["mae"],
               "path_mae_ratio_to_naive": m["mae"] / nv["mae"] if nv["mae"] else np.nan,
               "path_prices_scored": m["count"],
               "derived_high_mae_pips": float(np.mean(np.abs(H_hat[ok] - b.H[ok])) / PIP) if ok.any() else np.nan,
               "derived_low_mae_pips": float(np.mean(np.abs(L_hat[ok] - b.L[ok])) / PIP) if ok.any() else np.nan,
               "derived_first_touch_accuracy": float(np.mean(ft_hat[clear] == b.ft[clear])) if clear.any() else np.nan}
        for j in range(ctx.hours):
            out[f"path_mae_h{j + 1}"] = float(m["per_h"][j])
            out[f"naive_mae_h{j + 1}"] = float(nv["per_h"][j])
        return out
    return evaluate(ctx, orders, costs, acc)


def path_forecast_family(ctx, family, level, seed):
    """Predicted price path for one error family (DEVELOPMENT oracle/noise constructions)."""
    T, ref = ctx.path_targets(), ctx.blocks.ref
    if family == "iid":
        return T + level * ctx.path_noise(seed)
    z1 = ctx.path_noise(seed)
    if family == "a1":
        return N.blend_path(T, ref, level, A1_EPS, z1)
    z2 = ctx.__dict__.setdefault("_rw_noise", {}).setdefault(seed, N.path_draws(seed, len(ctx.blocks), ctx.hours, 1))
    if family == "a2":
        return np.where(np.isfinite(T), T + N.random_walk_error(z2, level), np.nan)
    if family == "a3":
        w, step = A3_PAIRS[int(level)]
        return np.where(np.isfinite(T), N.blend_path(T, ref, w, 0.0, z1) + N.random_walk_error(z2, step), np.nan)
    raise ValueError(family)


def run_pathr_cell(ctx, cell, p):
    """Price-path range-first with the legacy or the realistic execution profile and the daily order cap."""
    T = ctx.path_targets()
    naive = np.where(np.isfinite(T), ctx.blocks.ref[:, None], np.nan)
    m = cell["cost_mult"]
    if cell["sweep"] == "legacy":
        costs, profile = X.Costs().scaled(m), None
    else:
        profile = X.RealisticProfile().scaled(m)
        costs = profile
    if cell["strategy"] == "PERSISTENCE":
        preds, variant = naive, "RF-M"
    else:
        preds, variant = path_forecast_family(ctx, cell["family"], cell["level"], cell["seed"]), cell["strategy"]
    H_hat, L_hat, ft_hat = S.path_to_range(preds)
    orders = S.range_first_orders(variant, ctx.blocks, H_hat, L_hat, ft_hat, costs, p, CAPITAL)
    cap = cell.get("order_cap", ORDER_CAP_PER_DAY)

    def acc(mask):
        a, nv = A.path_mae(preds, T, mask), A.path_mae(naive, T, mask)
        out = {"path_mae": a["mae"], "naive_path_mae": nv["mae"],
               "path_mae_ratio_to_naive": a["mae"] / nv["mae"] if nv["mae"] else np.nan}
        for j in range(ctx.hours):
            out[f"path_mae_h{j + 1}"] = float(a["per_h"][j])
            out[f"naive_mae_h{j + 1}"] = float(nv["per_h"][j])
        return out
    rows = []
    for name, (a, b) in SPLITS.items():
        res = X.run_backtest(ctx.bars, ctx.blocks, orders, costs, ctx.cal, CAPITAL, (utc(a), utc(b)),
                             profile=profile, max_orders_per_day=cap)
        rows.append(({"split": name, **res.metrics, **acc(ctx.masks[name])}, res))
    return rows


def cell_key(c):
    return (c["grid"], c["block_hours"], c["strategy"], c["family"], c["level"], c["p_flip"], c["seed"], c["cost_mult"])


def run_grid(args, bars, cal, digest):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in args.chunk.split("/"))
    todo = cells(args.grid)[i::n]
    target = out / f"cells_{args.grid}_{i}of{n}.csv"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    ctxs = {h: Context(bars, cal, h) for h in sorted({c["block_hours"] for c in todo})}
    rows, curves = [], {}
    t0 = time.process_time()
    for c in todo:
        for row, res in run_cell(ctxs[c["block_hours"]], c):
            rows.append({**c, **row, "label": LABEL})
            if args.save_equity:
                key = "|".join(map(str, cell_key(c) + ((c["sweep"],) if "sweep" in c else ()) +
                                   ((c["ft_target"],) if "ft_target" in c else ()) + (row["split"],)))
                curves[key] = res.daily_equity.to_numpy(np.float32)
                curves[key + "|t0"] = np.array([res.daily_equity.index[0].value], dtype=np.int64)
    cpu = time.process_time() - t0
    pd.DataFrame(rows).to_csv(target, index=False)
    if curves:
        np.savez_compressed(out / f"equity_{args.grid}_{i}of{n}.npz", **curves)
    receipt = {"grid": args.grid, "chunk": args.chunk, "cells": len(todo), "rows": len(rows),
               "cpu_seconds": cpu, "data_sha256": digest, "label": LABEL}
    (out / f"receipt_{args.grid}_{i}of{n}.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt))


def merge(args):
    out = Path(args.output)
    frames = [pd.read_csv(p) for p in sorted(out.glob("cells_*of*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    grids = args.grids.split(",")
    per_cell = {g: (1 if g in ("val24", "val24x") else len(SPLITS)) for g in grids}  # pathx: both splits
    expected = {g: len(cells(g)) * per_cell[g] for g in grids}
    got = df.groupby("grid").size().to_dict()
    if got != expected:
        raise SystemExit(f"incomplete population {got} != {expected}")
    keys = ["grid", "block_hours", "strategy", "family", "level", "p_flip", "seed", "cost_mult", "split"] + \
        [k for k in ("sweep", "ft_target") if k in df.columns]
    if df.duplicated(keys).any():
        raise SystemExit("duplicate cells")
    df.to_csv(out / "cells.csv", index=False)
    cpu = sum(json.loads(p.read_text())["cpu_seconds"] for p in out.glob("receipt_*.json"))
    print(f"merged {len(df)} rows; total backtest CPU {cpu:.1f} s")


def pilot(bars, cal):
    for h in BLOCK_HOURS:
        t = time.process_time()
        ctx = Context(bars, cal, h)
        prep = time.process_time() - t
        ok = {k: int(v.sum()) for k, v in ctx.masks.items()}
        t = time.process_time()
        r = run_cell(ctx, dict(grid="rf", block_hours=h, strategy="RF-L", family="rel", level=0.0,
                               p_flip=0.0, seed=42, cost_mult=1.0))
        per = time.process_time() - t
        print(json.dumps({"block_hours": h, "blocks": len(ctx.blocks), "evaluated": ok,
                          "prep_cpu_s": round(prep, 3), "cell_cpu_s": round(per, 3),
                          "oracle_rf_l_net_usd": {row["split"]: round(row["net_profit_usd"], 2) for row, _ in r}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calendar", "pilot", "run", "merge"])
    ap.add_argument("--data", type=Path, help=f"lake file {LAKE_RELPATH}")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--grid", choices=["baselines", "rf", "comparator", "costs", "val24", "val24x", "pathx", "pathr"])
    ap.add_argument("--grids", default="baselines,rf,comparator,costs", help="merge: grids expected")
    ap.add_argument("--chunk", default="0/1")
    ap.add_argument("--save-equity", action="store_true")
    args = ap.parse_args()
    if args.stage == "calendar":
        return derive_calendar(args.data)
    if args.stage == "merge":
        return merge(args)
    bars, cal, digest = load_inputs(args.data)
    if args.stage == "pilot":
        return pilot(bars, cal)
    run_grid(args, bars, cal, digest)


if __name__ == "__main__":
    main()
