"""Action-consequence (outcome) tables for stage C2 of the causal-ladder design (2026-10-25,
``docs/short_horizon/clean_round/CAUSAL_LADDER_DESIGN_2026_10_25.md``).

Our orders do not move the price, so the consequence of OUR action at any bar is identified exactly by replaying it
on the price path that happened: P(outcome | x, do(action)) needs no model and no assumption. Every number here comes
from the engine itself (``env.BarEnv.step``: fills at the next open, costs, effective SL/TP levels, intrabar stop-first
protection, forced flat, rollover financing); nothing about prices, levels or costs is reimplemented.

Entry table (every eligible flat row, both sides, every (TP, SL) cell of the declared grid, in causal sigma):
  net_sigma   net result per unit / causal sigma of the fill bar (every cost included; the per-trade sigma of
              ``action_contract.trade_returns_sigma``)
  hit         0 = TP, 1 = SL, 2 = TIME (forced flat)
  mae, mfe    adverse / favourable excursion over the bars entry..exit (inclusive), in the same sigma
  duration    exit bar - entry bar
  exit_bar    the resolution bar; a label is AVAILABLE in a period only when v[exit_bar] + 1 h <= the period end and
              the trade was not cut by the end of the simulated window (censored rows are dropped, never filled)
Exit table (every positioned row of a visitation): the same open position held until the engine closes it versus
closed now at the next open; ``remaining`` = (net(hold) - net(close now)) per unit / the position's sigma (> 0: holding
pays), ``sl_first`` = 1 when holding ends at the SL or negative at the forced flat (the early-close audit's label).

Hindsight enters ONLY as labels; nothing here is an observation. The simulated window ends at TEST_START (TEST closed).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from . import env as E

GRID_TP = (0.5, 1.0, 1.5, 2.0)            # sigma multiples (inside the env bounds [0.25, 3])
GRID_SL = (0.75, 1.5, 2.5)                # sigma multiples (inside the env bounds [0.5, 3])
CELLS = tuple((tp, sl) for tp in GRID_TP for sl in GRID_SL)
REF_CELL = (1.5, 1.5)                     # the cell of the binary "TP before SL" head (declared, not chosen)
HIT_TP, HIT_SL, HIT_TIME = 0, 1, 2
HIT_OF_REASON = {"TP": HIT_TP, "SL": HIT_SL, "FORCED_FLAT": HIT_TIME}
TEST_START = "2025-01-01"
SIDES = (1, -1)                           # index 0 = long, 1 = short


def cell_index(cell) -> int:
    return CELLS.index(tuple(float(x) for x in cell))


def _clean(env: E.BarEnv):
    env.pos, env.trades, env.decisions, env.entries_day = None, [], [], {}
    env.balance = env.cfg.capital


def _excursions(env: E.BarEnv, t: dict, sig: float):
    k0, k1 = int(t["entry_bar_abs"]), int(t["exit_bar_abs"])
    hi, lo, p = float(np.max(env.h[k0:k1 + 1])), float(np.min(env.l[k0:k1 + 1])), float(t["entry_price"])
    if int(t["direction"]) > 0:
        return max(0.0, p - lo) / sig, max(0.0, hi - p) / sig
    return max(0.0, hi - p) / sig, max(0.0, p - lo) / sig


def _hold_to_close(env: E.BarEnv, j: int) -> bool:
    """HOLD from decision bar j until the engine closes the open position; False when the simulated window ended
    first (the trade is then censored: closed by ``flatten_at_end``, never used as a label)."""
    while env.pos is not None:
        if j >= env.last:
            env.flatten_at_end()
            return False
        env.step(j, E.HOLD)
        j += 1
    return True


def simulate_entry(env: E.BarEnv, b: int, d: int, tp_mult: float, sl_mult: float) -> dict:
    """An entry decided after bar b (side d, TP/SL multiples) on a clean account, held until the engine closes it."""
    _clean(env)
    if not (env.first <= b < env.last):
        return {"refused": "OUTSIDE_WINDOW"}
    info = env.step(b, E.LONG if d > 0 else E.SHORT, tp_mult, sl_mult)
    if not info["entry"]:
        return {"refused": info["refused"] or "NO_ENTRY"}
    complete = True if env.pos is None else _hold_to_close(env, b + 1)
    t = env.trades[-1]
    sig = float(env.sigma[int(t["entry_bar_abs"])])
    mae, mfe = _excursions(env, t, sig)
    return {"refused": None, "net_sigma": float(t["net_usd"]) / max(int(t["units"]), 1) / sig, "hit": HIT_OF_REASON[t["close_reason"]],
            "mae": mae, "mfe": mfe, "duration": int(t["exit_bar_abs"]) - int(t["entry_bar_abs"]), "exit_bar": int(t["exit_bar_abs"]),
            "censored": not complete}


def make_env(cx, cfg, start=None, end=TEST_START) -> E.BarEnv:
    a = start if start is not None else cx.ds.bars.index[0]
    return E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, end, cfg)


def entry_rows(cx, env: E.BarEnv) -> np.ndarray:
    """Dataset rows whose bar is inside the window, observable, and whose entry gate is open on a clean account."""
    out = []
    for b in range(env.first, env.last):
        r = int(cx.rob[b])
        if r < 0 or not cx.obs_ok[r]:
            continue
        _clean(env)
        if env.entry_gate(b)[0]:
            out.append(r)
    return np.asarray(out, np.int64)


def entry_table(cx, cfg, cells=CELLS, end=TEST_START, rows=None, progress=None) -> dict:
    """The entry outcome table over the dataset rows (NaN where not eligible or refused). Arrays (n_rows, 2, n_cells)."""
    env = make_env(cx, cfg, end=end)
    n, nc = len(cx.ds.row_bar), len(cells)
    rows = entry_rows(cx, env) if rows is None else np.asarray(rows, np.int64)
    t = {k: np.full((n, 2, nc), np.nan) for k in ("net_sigma", "mae", "mfe", "duration")}
    t["hit"] = np.full((n, 2, nc), -1, np.int8)
    t["exit_bar"] = np.full((n, 2, nc), -1, np.int64)
    t["censored"] = np.zeros((n, 2, nc), bool)
    t["eligible"] = np.zeros(n, bool)
    refused = {}
    for i, r in enumerate(rows):
        b = int(cx.ds.row_bar[r])
        t["eligible"][r] = True
        for si, d in enumerate(SIDES):
            for ci, (tp, sl) in enumerate(cells):
                o = simulate_entry(env, b, d, tp, sl)
                if o["refused"]:
                    refused[o["refused"]] = refused.get(o["refused"], 0) + 1
                    continue
                for k in ("net_sigma", "mae", "mfe", "duration", "hit", "exit_bar", "censored"):
                    t[k][r, si, ci] = o[k]
        if progress and i % 2000 == 0:
            progress(i, len(rows))
    t["cells"] = np.asarray(cells, float)
    t["refused"] = refused
    t["stamp"] = stamp(cx, cfg, cells, end)
    return t


def stamp(cx, cfg, cells, end) -> dict:
    """What the table is a function of (checked on load): bars, row map, sigma, grid, window end, cost profile."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cx.ds.bars.index.as_unit("ns").asi8).tobytes())
    for k in ("OPEN", "HIGH", "LOW", "CLOSE"):
        h.update(np.ascontiguousarray(cx.ds.bars[k].to_numpy(float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(cx.ds.row_bar, np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.nan_to_num(np.asarray(cx.ds.sigma, float), nan=-1.0)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(cx.obs_ok, bool)).tobytes())
    h.update(json.dumps({"cells": [list(c) for c in cells], "end": str(end), "cfg": repr(cfg)}, sort_keys=True).encode())
    return {"sha256": h.hexdigest(), "cells": [list(c) for c in cells], "end": str(end)}


def save_table(path, t: dict):
    arrays = {k: v for k, v in t.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays, meta=np.frombuffer(json.dumps({"refused": t["refused"], "stamp": t["stamp"]}).encode(), np.uint8))


def load_table(path, cx=None, cfg=None) -> dict:
    z = np.load(path, allow_pickle=False)
    t = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(bytes(z["meta"]).decode())
    t.update(meta)
    if cx is not None and cfg is not None:
        exp = stamp(cx, cfg, [tuple(c) for c in t["cells"].tolist()], meta["stamp"]["end"])
        if exp["sha256"] != meta["stamp"]["sha256"]:
            raise SystemExit("C2_OUTCOMES_STAMP_MISMATCH: the outcome table was built on different bars / rows / sigma / grid / costs")
    return t


def period_end_ns(cx, period: str) -> int:
    from . import data as DG
    return int(DG._utc_ns(cx.ds.meta["periods"][period][1]))


def available(cx, exit_bar, censored, period: str) -> np.ndarray:
    """Label availability in ``period``: resolution bar + 1 h <= the period end, not censored by the window."""
    eb = np.asarray(exit_bar, np.int64)
    v = cx.ds.bars.index.as_unit("ns").asi8
    ok = (eb >= 0) & ~np.asarray(censored, bool)
    t_end = np.where(ok, v[np.clip(eb, 0, len(v) - 1)] + E.H_NS, np.iinfo(np.int64).max)
    return ok & (t_end <= period_end_ns(cx, period))


def entry_sets(cx, t: dict, period: str, side_index: int, ref=REF_CELL):
    """(rows, y = TP before SL at the reference cell, R = net grid, all cells available) of one side in ``period``."""
    rows = np.flatnonzero(t["eligible"] & (cx.ds.period == period))
    ok = np.all(available(cx, t["exit_bar"][rows, side_index, :], t["censored"][rows, side_index, :], period), axis=1)
    ok &= np.all(np.isfinite(t["net_sigma"][rows, side_index, :]), axis=1)
    rows = rows[ok]
    ci = cell_index(ref)
    y = (t["hit"][rows, side_index, ci] == HIT_TP).astype(np.int64)
    return rows, y, t["net_sigma"][rows, side_index, :].astype(np.float32)


# ----------------------------------------------------------------------------- exits
def hold_value(env: E.BarEnv, pos: E.Position, b: int) -> dict:
    """The open ``pos`` at decision bar b: held until the engine closes it vs closed now (fill at the next open)."""
    _clean(env)
    env.pos = dataclasses.replace(pos)
    complete = _hold_to_close(env, b)
    th = env.trades[-1]
    _clean(env)
    env.pos = dataclasses.replace(pos)
    env.step(b, E.CLOSE)
    tc = env.trades[-1]
    _clean(env)
    u = max(int(pos.units), 1)
    sl_first = 1 if th["close_reason"] == "SL" else (0 if th["close_reason"] == "TP" else int(th["gross_usd"] < 0))
    return {"remaining": (float(th["net_usd"]) - float(tc["net_usd"])) / u / float(pos.sigma), "sl_first": sl_first,
            "exit_bar": int(th["exit_bar_abs"]), "censored": not complete, "hold_reason": th["close_reason"]}


def visit_positions(cx, period, act, cfg):
    """Replay ``act`` on ``period`` (same loop as ``Ctx.rollout``); every observable row where a position is open at the
    decision yields (row, bar, a copy of the position)."""
    a, b = cx.ds.meta["periods"][period] if isinstance(period, str) else period
    env = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, cfg)
    out = []
    for bar in range(env.first, env.last):
        r = int(cx.rob[bar])
        st = env.state(bar)
        if r >= 0 and cx.obs_ok[r]:
            if env.pos is not None:
                out.append((r, bar, dataclasses.replace(env.pos)))
            action, tp, sl = act(bar, r, st, env)
        else:
            action, tp, sl = E.HOLD, 1.5, 2.0
        env.step(bar, action, tp, sl)
    return out


def exit_table(cx, cfg, visits, end=TEST_START) -> dict:
    """Per visit: side index, row, remaining value of HOLD, sl_first, resolution bar, censored."""
    env = make_env(cx, cfg, end=end)
    rec = {k: [] for k in ("row", "bar", "side", "remaining", "sl_first", "exit_bar", "censored")}
    for r, bar, pos in visits:
        if not (env.first <= bar < env.last):
            continue
        o = hold_value(env, pos, bar)
        rec["row"].append(r)
        rec["bar"].append(bar)
        rec["side"].append(0 if pos.direction > 0 else 1)
        for k in ("remaining", "sl_first", "exit_bar", "censored"):
            rec[k].append(o[k])
    return {"row": np.asarray(rec["row"], np.int64), "bar": np.asarray(rec["bar"], np.int64), "side": np.asarray(rec["side"], np.int64),
            "remaining": np.asarray(rec["remaining"], float), "sl_first": np.asarray(rec["sl_first"], np.int64),
            "exit_bar": np.asarray(rec["exit_bar"], np.int64), "censored": np.asarray(rec["censored"], bool)}


def exit_sets(cx, xt: dict, period: str, side_index: int):
    """(rows, y = SL first, R = remaining) of the available exit visits of one side in ``period``."""
    m = (xt["side"] == side_index) & (cx.ds.period[xt["row"]] == period) if len(xt["row"]) else np.zeros(0, bool)
    if not len(m):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 1), np.float32)
    m &= available(cx, xt["exit_bar"], xt["censored"], period) & np.isfinite(xt["remaining"])
    return xt["row"][m], xt["sl_first"][m].astype(np.int64), xt["remaining"][m][:, None].astype(np.float32)


def concat_exit_tables(*ts) -> dict:
    ts = [t for t in ts if t is not None and len(t["row"])]
    if not ts:
        return {k: np.zeros(0, np.int64 if k in ("row", "bar", "side", "sl_first", "exit_bar") else (bool if k == "censored" else float))
                for k in ("row", "bar", "side", "remaining", "sl_first", "exit_bar", "censored")}
    return {k: np.concatenate([t[k] for t in ts]) for k in ts[0]}
