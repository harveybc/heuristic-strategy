#!/usr/bin/env python3
"""Early-close audit of a finished E3-M unit (owner 2026-10-25: early close is an indispensable business behaviour; a
model that detects a learned pattern saying the SL will come before the TP must close). Replays each seed's frozen
policy, and the seed ensemble when ENSEMBLE.json exists, with the run's own env and replay contract; the replay must
reproduce the run's metrics exactly (asserted). Then, with the engine's own fills, levels and costs (``env.BarEnv``,
never re-implemented here):

  every agent close (AGENT_CLOSE or REVERSAL)   counterfactual = the SAME position (units, SL, TP, entry costs) held
                                                from the close decision bar with HOLD until TP, SL or forced flat;
                                                delta = actual net - counterfactual net (USD; in sigma = delta /
                                                (units x the position's causal sigma)); SAVED delta > 0 (the close
                                                avoided a worse outcome), CUT_WINNER delta < 0, NEUTRAL |delta| < 0.05
                                                sigma; early-close value = sum of deltas, bootstrap 95 % CI over trades
  every positioned, observable decision bar     label SL_FIRST = holding from this bar ends at the SL (or negative at
                                                the forced flat; the engine's protection order: open gap, stop before
                                                take-profit) -- constant over a position's bars, because a position
                                                alive at bar b has not been touched yet; label CLOSE_BETTER = closing at
                                                the next open nets more than holding; the closer's exit probability
                                                (legacy: softmax of the side's closer; v1: the calibrated pCL / pCS) ->
                                                AUC against both labels; fired = the policy closed (CLOSE or reversal)
                                                -> precision / recall against SL_FIRST
  the no-early-close policy                     the same frozen policy with every voluntary exit replaced by HOLD
                                                (CLOSE -> HOLD, opposite open while positioned -> HOLD), replayed in the
                                                engine (executable; its entries differ where positions now last
                                                longer) plus the trade-level sum (actual trades with every agent close
                                                replaced by its counterfactual; overlapping, NOT executable, reported)
  direction balance                             long / short entries, net and Sharpe per side (daily P&L by exit date /
                                                capital, sqrt(365), the period's calendar days), share of positioned
                                                decision bars per side

Flags (pre-registered 2026-10-25 before any audit number was read, CLEAN_ROUND_SPEC v7 ``early_close_rule``):
DEGENERATE_NO_EARLY_CLOSE = zero agent closes; HARMFUL_EARLY_CLOSE = the early-close value's 95 % CI entirely < 0
(the closer cuts winners: a defect to fix, never a reason to remove early closing). Reads no TEST row; trains nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import stats as ST  # noqa: E402

AGENT_REASONS = ("AGENT_CLOSE", "REVERSAL")
NEUTRAL_SIGMA = 0.05
BOOT_N, BOOT_SEED = 2000, 20261025
RULE = ("PRE-REGISTERED 2026-10-25 before any audit number was read: a selectable strategy must close early (agent closes > 0 "
        "on the calibration block), otherwise DEGENERATE_NO_EARLY_CLOSE and not selectable (owner's business rule); the "
        "early-close value is a REQUIRED reported metric for every arm; a closer whose early-close value 95 % CI is entirely "
        "< 0 is HARMFUL_EARLY_CLOSE (it cuts winners): a defect to fix, not a reason to remove early closing")


# ----------------------------------------------------------------------------- engine counterfactuals
def cf_hold(cf_env: E.BarEnv, pos: E.Position, b: int) -> dict:
    """The same position held from decision bar ``b`` with HOLD until the engine closes it (TP / SL / forced flat)."""
    cf_env.trades = []
    cf_env.pos = dataclasses.replace(pos)
    j = b
    while cf_env.pos is not None:
        if j >= cf_env.last:
            cf_env.flatten_at_end()
            break
        cf_env.step(j, E.HOLD)
        j += 1
    return cf_env.trades[-1]


def cf_close_now(cf_env: E.BarEnv, pos: E.Position, b: int) -> dict:
    """The same position closed by the agent at decision bar ``b`` (fill at the next open, market costs)."""
    cf_env.trades = []
    cf_env.pos = dataclasses.replace(pos)
    cf_env.step(b, E.CLOSE)
    return cf_env.trades[-1]


def sl_first(rec: dict) -> int:
    """1 when holding ends at the SL or negative at the forced flat (the closer's label), 0 at the TP / positive flat."""
    if rec["close_reason"] == "SL":
        return 1
    if rec["close_reason"] == "TP":
        return 0
    return int(rec["gross_usd"] < 0)


def classify(delta_sigma: float) -> str:
    if abs(delta_sigma) < NEUTRAL_SIGMA:
        return "NEUTRAL"
    return "SAVED" if delta_sigma > 0 else "CUT_WINNER"


def boot_sum_ci(x, n=BOOT_N, seed=BOOT_SEED, alpha=0.05):
    """Percentile bootstrap of the SUM over trades (resampling trades with replacement)."""
    x = np.asarray(x, float)
    if not len(x):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    s = x[rng.integers(0, len(x), size=(n, len(x)))].sum(axis=1)
    return [float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))]


def auc(score, y) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN when one class is absent."""
    s, y = np.asarray(score, float), np.asarray(y, int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------- the audited replay
def audit_replay(make_env, bar_range, decide, exit_prob=None) -> dict:
    """One engine pass with counterfactuals. ``make_env()`` returns a fresh BarEnv on the period (called twice: the
    real pass and the counterfactual scratch env). ``decide(bar, st, env)`` -> (action, tp, sl) or None when the bar is
    not observable (HOLD, policy not consulted). ``exit_prob(bar, st)`` -> the closer's exit probability on a
    positioned observable bar (None: not reported). Returns the env result, the trade list and per-trade / per-bar
    audit records."""
    env, cf = make_env(), make_env()
    closes, bars_rec, pos_bars = [], [], []
    cur_pos = None
    first, last = bar_range if bar_range is not None else (env.first, env.last)
    for bar in range(first, last):
        st = env.state(bar)
        p = env.pos
        out = decide(bar, st, env)
        observable = out is not None
        action, tp, sl = out if observable else (E.HOLD, 1.5, 2.0)
        if p is not None and observable:
            pe = None if exit_prob is None else (exit_prob(bar, st, p) if getattr(exit_prob, "wants_position", False) else exit_prob(bar, st))
            fired = action == E.CLOSE or (action in (E.LONG, E.SHORT) and (1 if action == E.LONG else -1) != p.direction)
            cn = cf_close_now(cf, p, bar)
            pos_bars.append({"bar": bar, "side": int(p.direction), "p_exit": None if pe is None else float(pe), "fired": bool(fired),
                             "close_now_net": float(cn["net_usd"]), "entry_bar": int(p.entry_bar),
                             "dist_sl_sigma": float(p.direction * (env.c[bar] - p.sl) / p.sigma) if p.sigma > 0 else float("nan")})
        if p is not None:
            bars_rec.append(int(p.direction))
        n_before = len(env.trades)
        env.step(bar, action, tp, sl)
        if p is not None and len(env.trades) > n_before:          # the position that was open at this decision ended
            rec = env.trades[n_before]
            if rec["close_reason"] in AGENT_REASONS:
                h = cf_hold(cf, p, bar)
                closes.append({"trade_index": n_before, "decision_bar": bar, "side": int(p.direction), "reason": rec["close_reason"],
                               "actual_net": float(rec["net_usd"]), "cf_net": float(h["net_usd"]), "cf_reason": h["close_reason"],
                               "sigma_usd": float(p.units * p.sigma), "units": int(p.units)})
                hold_rec = h
            else:
                hold_rec = rec
            lab = sl_first(hold_rec)
            hold_net = float(hold_rec["net_usd"])
            for r in pos_bars:
                if r["entry_bar"] == p.entry_bar and "sl_first" not in r:
                    r["sl_first"] = lab
                    r["close_better"] = int(r["close_now_net"] > hold_net)
                    r["hold_net"] = hold_net
    m = env.result()
    at_bar = {(r["entry_bar"], r["bar"]): r for r in pos_bars}
    for c in closes:                                              # the counterfactual machinery's own checks
        a = env.trades[c["trade_index"]]
        assert a["close_reason"] == c["reason"]
        r = at_bar.get((a["entry_bar_abs"], c["decision_bar"]))
        if r is not None and abs(r["close_now_net"] - c["actual_net"]) > 1e-9:
            raise AssertionError(f"CLOSE_NOW_COUNTERFACTUAL_DIFFERS_FROM_ACTUAL:{c['decision_bar']}:{r['close_now_net']}:{c['actual_net']}")
    return {"metrics": m, "trades": list(env.trades), "closes": closes, "pos_bars": pos_bars, "positioned_sides": bars_rec,
            "decision_bars": last - first}


def side_sharpe(trades, side, daily_index, capital):
    """Sharpe (daily, sqrt(365)) of one side's P&L: each trade's net on the calendar day of its exit, over the period's
    calendar days (the engine's daily grid), divided by the initial capital (declared approximation of a per-side book)."""
    idx = pd.DatetimeIndex(daily_index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    pnl = np.zeros(len(idx))
    for t in trades:
        if t["direction"] == side:
            d = pd.Timestamp(t["exit_time"])
            d = (d.tz_convert("UTC").tz_localize(None) if d.tzinfo is not None else d).normalize()
            k = int(idx.searchsorted(d, side="right")) - 1
            if k >= 0:
                pnl[k] += t["net_usd"]
    return ST.sharpe(pnl / float(capital))


def summarize(a: dict, capital: float, no_ec: dict | None = None) -> dict:
    """Report numbers of one audited replay (and its no-early-close replay)."""
    m, trades, closes, pb = a["metrics"], a["trades"], a["closes"], a["pos_bars"]
    n_exit = len(trades)
    by_reason = {}
    for t in trades:
        by_reason[t["close_reason"]] = by_reason.get(t["close_reason"], 0) + 1
    deltas = np.array([c["actual_net"] - c["cf_net"] for c in closes], float)
    dsig = np.array([(c["actual_net"] - c["cf_net"]) / c["sigma_usd"] if c["sigma_usd"] > 0 else np.nan for c in closes], float)
    cls = [classify(x) for x in dsig]
    out = {"trades": n_exit, "exits_by_reason": by_reason, "agent_closes": len(closes),
           "early_close_share": (len(closes) / n_exit) if n_exit else float("nan"),
           "classes": {k: cls.count(k) for k in ("SAVED", "CUT_WINNER", "NEUTRAL")},
           "class_shares": {k: (cls.count(k) / len(cls)) if cls else float("nan") for k in ("SAVED", "CUT_WINNER", "NEUTRAL")},
           "early_close_value_usd": float(deltas.sum()) if len(deltas) else 0.0, "early_close_value_usd_ci95": boot_sum_ci(deltas),
           "early_close_value_sigma": float(np.nansum(dsig)) if len(dsig) else 0.0, "early_close_value_sigma_ci95": boot_sum_ci(dsig[np.isfinite(dsig)]),
           "cf_reason_of_agent_closes": {k: sum(1 for c in closes if c["cf_reason"] == k) for k in ("TP", "SL", "FORCED_FLAT")}}
    for side, nm in ((1, "long"), (-1, "short")):
        d = np.array([c["actual_net"] - c["cf_net"] for c in closes if c["side"] == side], float)
        out[f"early_close_value_usd_{nm}"] = float(d.sum()) if len(d) else 0.0
        out[f"early_close_value_usd_{nm}_ci95"] = boot_sum_ci(d)
        out[f"agent_closes_{nm}"] = int(len(d))
    # closer quality on every positioned observable bar
    lab = [r for r in pb if "sl_first" in r]
    y1 = np.array([r["sl_first"] for r in lab], int)
    y2 = np.array([r["close_better"] for r in lab], int)
    fired = np.array([r["fired"] for r in lab], bool)
    pe = np.array([np.nan if r["p_exit"] is None else r["p_exit"] for r in lab], float)
    tp_ = int((fired & (y1 == 1)).sum())
    out["closer"] = {"positioned_observable_bars": len(lab), "sl_first_rate": float(y1.mean()) if len(y1) else float("nan"),
                     "close_better_rate": float(y2.mean()) if len(y2) else float("nan"),
                     "fired_rate": float(fired.mean()) if len(fired) else float("nan"),
                     "precision_sl_first": (tp_ / int(fired.sum())) if fired.sum() else float("nan"),
                     "recall_sl_first": (tp_ / int((y1 == 1).sum())) if (y1 == 1).any() else float("nan"),
                     "auc_sl_first": auc(pe, y1), "auc_close_better": auc(pe, y2)}
    out["closer_by_dist_sl_sigma"] = closer_by_bucket(lab, pe, y1, fired)
    # direction balance
    di = pd.DatetimeIndex(m["daily_equity"].index) if hasattr(m["daily_equity"], "index") else None
    sides = np.array(a["positioned_sides"], int)
    bal = {}
    for side, nm in ((1, "long"), (-1, "short")):
        ts = [t for t in trades if t["direction"] == side]
        bal[nm] = {"entries": len(ts), "net_usd": float(sum(t["net_usd"] for t in ts)),
                   "sharpe_daily_sqrt365": side_sharpe(trades, side, di, capital) if di is not None else float("nan"),
                   "positioned_bar_share": float((sides == side).sum() / a["decision_bars"]) if a["decision_bars"] else float("nan")}
    out["direction_balance"] = bal
    out["actual"] = {"sharpe": float(m["sharpe_daily_sqrt365"]), "net_usd": float(m["net_profit_usd"]), "trades": int(m["trades"]),
                     "orders_per_week": float(m["orders_per_week"]), "max_dd_pct": float(m["max_drawdown_pct"])}
    out["trade_level_hold_sum_usd"] = float(sum(t["net_usd"] for t in trades) - deltas.sum()) if n_exit else 0.0
    out["trade_level_hold_note"] = "actual trades with every agent close replaced by its counterfactual hold; positions overlap: NOT executable"
    if no_ec is not None:
        mn = no_ec
        out["no_early_close_policy"] = {"sharpe": float(mn["sharpe_daily_sqrt365"]), "net_usd": float(mn["net_profit_usd"]), "trades": int(mn["trades"]),
                                        "orders_per_week": float(mn["orders_per_week"]), "max_dd_pct": float(mn["max_drawdown_pct"]),
                                        "agent_closes": int(mn.get("n_close_AGENT_CLOSE", 0) + mn.get("n_close_REVERSAL", 0)),
                                        "note": "the same frozen policy with every voluntary exit replaced by HOLD (executable, engine replay)"}
    ci = out["early_close_value_usd_ci95"]
    out["flags"] = []
    if len(closes) == 0:
        out["flags"].append("DEGENERATE_NO_EARLY_CLOSE")
    if len(closes) and math.isfinite(ci[1]) and ci[1] < 0:
        out["flags"].append("HARMFUL_EARLY_CLOSE")
    return out


DIST_SL_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)        # sigma of the order itself (C2G lane, 2026-10-25)


def closer_by_bucket(lab: list, pe, y1, fired) -> list:
    """Closer quality split by the order's signed distance to its SL in its own sigma at the decision (where does the
    order geometry help?): per bucket rows, SL-first rate, fired rate, precision, recall and AUC of the closer score."""
    d = np.array([r.get("dist_sl_sigma", np.nan) for r in lab], float)
    edges = (-np.inf,) + DIST_SL_EDGES[1:] + (np.inf,)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        n = int(m.sum())
        yy, ff = y1[m], fired[m]
        tp_ = int((ff & (yy == 1)).sum())
        out.append({"dist_sl_sigma": [None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi)], "rows": n,
                    "sl_first_rate": float(yy.mean()) if n else float("nan"), "fired_rate": float(ff.mean()) if n else float("nan"),
                    "precision_sl_first": (tp_ / int(ff.sum())) if n and ff.sum() else float("nan"),
                    "recall_sl_first": (tp_ / int((yy == 1).sum())) if n and (yy == 1).any() else float("nan"),
                    "auc_sl_first": auc(pe[m], yy) if n else float("nan")})
    return out


def no_early_close(decide):
    """The same decisions with every voluntary exit replaced by HOLD."""
    def f(bar, st, env):
        out = decide(bar, st, env)
        if out is None:
            return None
        a, tp, sl = out
        p = env.pos
        if p is not None and (a == E.CLOSE or (a in (E.LONG, E.SHORT) and (1 if a == E.LONG else -1) != p.direction)):
            return E.HOLD, 1.5, 2.0
        return out
    return f


# ----------------------------------------------------------------------------- policies from saved artifacts
def make_decide(cx, policy_fn):
    """Ctx.rollout's observability rule around a ``policy(cx)`` closure f(bar, r, st, env)."""
    def decide(bar, st, env):
        r = cx.rob[bar]
        if r >= 0 and cx.obs_ok[r]:
            return policy_fn(bar, r, st, env)
        return None
    return decide


def legacy_exit_prob(cx, comp, mask):
    from tools.expert_guided_e3m import EXIT

    def f(bar, st):
        r = cx.rob[bar]
        x = np.r_[cx.Xw[r], E.masked_state(st, mask)].astype(np.float32)
        name = "close_l" if int(st[0]) > 0 else "close_s"
        return comp._probs(name, x)[0][EXIT]
    return f


def v1_exit_prob(cx, table):
    def f(bar, st):
        r = cx.rob[bar]
        return table["pCL" if int(st[0]) > 0 else "pCS"][r]
    return f


METRIC_KEYS_FLOAT = ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct")
METRIC_KEYS_INT = ("trades", "n_entries", "n_close_TP", "n_close_SL", "n_close_FORCED_FLAT", "n_close_AGENT_CLOSE", "n_close_REVERSAL")


def check_reproduces(m: dict, ref: dict, what: str) -> dict:
    """Exact reproduction of the run's figures (floats to 1e-9, counts exactly); raises on any mismatch."""
    diffs = {}
    for k in METRIC_KEYS_FLOAT:
        if k in ref and ref[k] is not None and not (isinstance(ref[k], float) and math.isnan(ref[k])):
            if abs(float(m[k]) - float(ref[k])) > 1e-9:
                diffs[k] = [float(m[k]), float(ref[k])]
    for k in METRIC_KEYS_INT:
        if k in ref and ref[k] is not None:
            if int(m[k]) != int(ref[k]):
                diffs[k] = [int(m[k]), int(ref[k])]
    if diffs:
        raise SystemExit(f"REPLAY_DOES_NOT_REPRODUCE_RUN:{what}:{json.dumps(diffs)}")
    return {"reproduces_run": True, "checked": [k for k in METRIC_KEYS_FLOAT + METRIC_KEYS_INT if k in ref]}


def audit_unit(dirs: list, ds, periods: list, ensemble_path: Path | None) -> dict:
    """Every seed of the unit (``dirs``: the arm dir and any seed-extension dirs) and its ensemble, on ``periods``."""
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import Composed, effective_contracts
    from tools import expert_guided_seed_ensemble as SE
    from app.short_horizon.expert_guided import action_contract as ACT
    e3ms = [(Path(d), json.loads((Path(d) / "E3M.json").read_text())) for d in dirs]
    e0 = e3ms[0][1]
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    n_feat, win, NS = ds.windows.shape[2], ds.windows.shape[1], len(E.STATE_NAMES)
    SCFG, _ = effective_contracts(e0.get("exec_contract", "sac"), e0.get("min_orders_per_week"))
    capital = float(SCFG.capital)
    from app.short_horizon.expert_guided import c2 as C2
    c2 = C2.is_c2((e0.get("action_contract") or {}).get("contract", {}))     # stage C2 / C2G (consequence learning)
    c2g = C2.is_c2g((e0.get("action_contract") or {}).get("contract", {}))   # C2G: closers also read their order's geometry
    v1 = bool(e0.get("action_contract")) and not c2
    doc = {"dirs": [str(d) for d in dirs], "v1": v1, "c2": c2, **({"c2g": True} if c2g else {}), "split_contract": ds.meta["split_contract"], "periods": periods,
           "recipe": e0.get("recipe"), "rule": RULE, "per_seed": {}, "ensemble": {}, "verification": {}}

    def run(policy_fn, exit_prob, p, ref, what):
        a, b = cx.ds.meta["periods"][p]
        mk = lambda: E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, SCFG)  # noqa: E731
        dec = make_decide(cx, policy_fn)
        au = audit_replay(mk, None, dec, exit_prob)
        doc["verification"][what] = check_reproduces(au["metrics"], ref, what)
        ne = audit_replay(mk, None, no_early_close(dec), None)["metrics"]
        return summarize(au, capital, ne)

    seeds = []
    tables = []
    if v1:
        AC = e0["action_contract"]["contract"]
        cost = ACT.row_cost_sigma(SCFG.profile.round_trip_price(), ds.sigma, ds.row_bar, AC["strategy"]["cost_sigma"])
        rows_all = np.flatnonzero(cx.obs_ok)
    for d, e3m in e3ms:
        for s, tr in e3m["trials"]["FULL"].items():
            seeds.append(s)
            doc["per_seed"][s] = {}
            if c2:
                pols = {m: SE.load_module(d, e3m, m, s, n_feat, win) for m in C2.SPECIALISTS}
                t = C2.seed_table(pols, cx, NS, tr["action_contract"]["calibration"], geo=c2g)
                tables.append(t)
                st_s = C2.strategy_of(tr["action_contract"]["strategy"])
                if st_s is None:
                    doc["per_seed"][s] = {"note": "NO_FROZEN_STRATEGY (the run froze no valid strategy): nothing to replay"}
                    continue
                pf, ep = C2.PolicyC2(t, st_s)(cx), C2.exit_prob(cx, t)
                doc["per_seed"][s]["strategy"] = st_s.to_dict()
            elif v1:
                pols = {m: SE.load_module(d, e3m, m, s, n_feat, win) for m in ACT.SPECIALISTS}
                t = ACT.calibrate_table(ACT.raw_table(pols, cx.Xw, rows_all, NS, cost), tr["action_contract"]["calibration"])
                tables.append(t)
                st_s = ACT.strategy_of(tr["action_contract"]["strategy"])
                if st_s is None:
                    doc["per_seed"][s] = {"note": "NO_FROZEN_STRATEGY (the run froze no valid strategy): nothing to replay"}
                    continue
                pf, ep = ACT.ContractPolicy(t, st_s)(cx), v1_exit_prob(cx, t)
                doc["per_seed"][s]["strategy"] = st_s.to_dict()
            else:
                mask = E.state_mask(e3m["agent_state"]["mode"])
                pols = {m: SE.load_module(d, e3m, m, s, n_feat, win) for m in SE.MODULES}
                ar = tr["arbiter"]
                comp = Composed("FULL", pols, ar["arbiter"], ar.get("tau"), mask=mask)
                pf, ep = comp(cx), legacy_exit_prob(cx, comp, mask)
            for p in periods:
                ref = tr["trading"].get(p)
                if ref is None:
                    continue
                doc["per_seed"][s][p] = run(pf, ep, p, ref, f"{s}/{p}")
    doc["seeds"] = seeds
    if ensemble_path is not None and ensemble_path.exists():
        en = json.loads(ensemble_path.read_text())
        if [str(x) for x in en["seeds"]] != seeds:
            doc["ensemble_note"] = f"ENSEMBLE_SEEDS_DIFFER:{en['seeds']} vs {seeds}: ensemble not audited"
        elif c2:
            ens_t = C2.calibrate_table(C2.to_raw(C2.mean_tables(tables)), en["ensemble_calibration"])
            primary = en["arbiter"]["primary_fitness"]
            for fname, bf in en["by_fitness"].items():
                if bf["strategy"] is None:
                    doc["ensemble"][fname] = {"note": "NO_VALID_STRATEGY on the calibration block: nothing to replay"}
                    continue
                stf = C2.strategy_from_dict(bf["strategy"])
                pf, ep = C2.PolicyC2(ens_t, stf)(cx), C2.exit_prob(cx, ens_t)
                doc["ensemble"][fname] = {"strategy": bf["strategy"], "primary": fname == primary}
                for p in periods:
                    w = bf["windows"].get(p)
                    if w is None:
                        continue
                    doc["ensemble"][fname][p] = run(pf, ep, p, w["metrics"], f"ensemble/{fname}/{p}")
        elif v1:
            ens = ACT.mean_tables(tables)
            ens_raw = {f: v.copy() for f, v in ens.items()}
            for m, f in ACT.PROB_OF.items():
                ens_raw[f] = np.where(np.isfinite(ens[f]), ACT.logit(np.nan_to_num(ens[f], nan=0.5)), np.nan)
            ens_t = ACT.calibrate_table(ens_raw, en["ensemble_calibration"])
            primary = en["arbiter"]["primary_fitness"]
            for fname, bf in en["by_fitness"].items():
                if bf["strategy"] is None:
                    doc["ensemble"][fname] = {"note": "NO_VALID_STRATEGY on the calibration block: nothing to replay"}
                    continue
                stf = ACT.strategy_from_dict(bf["strategy"])
                pf, ep = ACT.ContractPolicy(ens_t, stf)(cx), v1_exit_prob(cx, ens_t)
                doc["ensemble"][fname] = {"strategy": bf["strategy"], "primary": fname == primary}
                for p in periods:
                    w = bf["windows"].get(p)
                    if w is None:
                        continue
                    doc["ensemble"][fname][p] = run(pf, ep, p, w["metrics"], f"ensemble/{fname}/{p}")
        else:
            mask = E.state_mask(e0["agent_state"]["mode"])
            of = {s: (d, e3m) for d, e3m in e3ms for s in e3m["trials"]["FULL"]}
            pols = {s: {m: SE.load_module(of[s][0], of[s][1], m, s, n_feat, win) for m in SE.MODULES} for s in seeds}
            ens = {m: SE.ProbEnsemble([pols[s][m] for s in seeds]) for m in SE.MODULES}
            arb = en["arbiter"]
            comp = Composed("FULL", ens, arb["arbiter"], arb.get("tau"), mask=mask)
            doc["ensemble"]["legacy"] = {"arbiter": arb, "primary": True}
            for p in periods:
                w = en["ensemble"].get(p)
                if w is None:
                    continue
                doc["ensemble"]["legacy"][p] = run(comp(cx), legacy_exit_prob(cx, comp, mask), p, w["metrics"], f"ensemble/{p}")
    doc["verification"]["all_replays_reproduce_run"] = all(v.get("reproduces_run") for v in doc["verification"].values() if isinstance(v, dict))
    return doc


def _row(unit, who, period, x):
    c, b = x["closer"], x["direction_balance"]
    ne = x.get("no_early_close_policy") or {}
    ci = x["early_close_value_usd_ci95"]
    f = lambda v, n=2: "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{n}f}"  # noqa: E731
    cs = x["class_shares"]
    return (f"| {unit} | {who} | {period} | {x['trades']} | {f(100 * x['early_close_share'], 0)} % | "
            f"{f(100 * cs['SAVED'], 0)} / {f(100 * cs['CUT_WINNER'], 0)} / {f(100 * cs['NEUTRAL'], 0)} | "
            f"{f(x['early_close_value_usd'], 0)} [{f(ci[0], 0)}, {f(ci[1], 0)}] | {f(x['early_close_value_sigma'], 1)} | "
            f"{f(x['actual']['sharpe'])} / {f(ne.get('sharpe'))} | {f(c['auc_sl_first'])} / {f(c['auc_close_better'])} | "
            f"{f(c['precision_sl_first'])} / {f(c['recall_sl_first'])} | {b['long']['entries']} / {b['short']['entries']} | "
            f"{f(b['long']['net_usd'], 0)} / {f(b['short']['net_usd'], 0)} | {', '.join(x['flags']) or '—'} |")


def write_report(paths, out_json, out_md, title):
    """One table over audited units: every seed and every ensemble policy, per period."""
    docs = {Path(p).stem.replace("EARLY_CLOSE_", ""): json.loads(Path(p).read_text()) for p in paths}
    md = [f"# {title}", "", RULE, "",
          "Columns: trades; early-close share (agent closes + reversals / exits); SAVED / CUT_WINNER / NEUTRAL shares of the agent "
          "closes; early-close value = sum(actual net - engine counterfactual hold net) USD [bootstrap 95 % CI over trades]; the same "
          "in sigma; Sharpe actual / no-early-close policy (same frozen policy, voluntary exits -> HOLD, engine replay); closer exit-"
          "probability AUC vs SL_FIRST / vs CLOSE_BETTER on every positioned observable bar; precision / recall of the fired closes "
          "vs SL_FIRST; long / short entries; long / short net USD; flags. Every replay reproduces its run exactly (asserted).", "",
          "| unit | policy | period | trades | early close | SAVED / CUT / NEUTRAL % | value USD [95 % CI] | value σ | Sharpe actual / no early close | AUC SL-first / close-better | precision / recall | long / short entries | long / short net | flags |",
          "| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- |"]
    for u, d in docs.items():
        for name, ens in d.get("ensemble", {}).items():
            for p in d["periods"]:
                if isinstance(ens, dict) and p in ens:
                    md.append(_row(u, f"ensemble{'' if name == 'legacy' else ' ' + name}{' (primary)' if ens.get('primary') and name != 'legacy' else ''}", p, ens[p]))
        for s, ps in d.get("per_seed", {}).items():
            for p in d["periods"]:
                if p in ps:
                    md.append(_row(u, f"seed {s}", p, ps[p]))
        if d.get("ensemble_note"):
            md.append(f"| {u} | ensemble | — | {d['ensemble_note']} |")
    allv = {u: d["verification"]["all_replays_reproduce_run"] for u, d in docs.items()}
    md += ["", f"Reproduction of every run: {json.dumps(allv)}"]
    Path(out_json).write_text(json.dumps({"rule": RULE, "units": docs}, indent=1, default=float))
    Path(out_md).write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm-dir", required=True)
    r.add_argument("--extra-arm-dir", action="append", default=[])
    r.add_argument("--admitted", required=True)
    r.add_argument("--store", required=True)
    r.add_argument("--recipe", required=True)
    r.add_argument("--split-contract", required=True)
    r.add_argument("--periods", required=True)
    r.add_argument("--ensemble", default=None, help="ENSEMBLE.json (default: <arm-dir>/ENSEMBLE.json when present)")
    r.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("--units", nargs="+", required=True)
    q.add_argument("--out-json", required=True)
    q.add_argument("--out-md", required=True)
    q.add_argument("--title", required=True)
    a = ap.parse_args()
    if a.cmd == "report":
        write_report(a.units, a.out_json, a.out_md, a.title)
        return
    from app.short_horizon.expert_guided import data as DG
    periods = [p.strip() for p in a.periods.split(",") if p.strip()]
    if any(p.upper().startswith("TEST") for p in periods):
        raise SystemExit("TEST_IS_CLOSED")
    rec = json.loads(Path(a.recipe).read_text())
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    ds = DG.load_store(a.admitted, a.store, feats, scaling="fixed_train", contract=a.split_contract, recipe=rec)
    ens = Path(a.ensemble) if a.ensemble else Path(a.arm_dir) / "ENSEMBLE.json"
    doc = audit_unit([a.arm_dir] + a.extra_arm_dir, ds, periods, ens if ens.exists() else None)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"out": a.out, "seeds": doc["seeds"], "reproduce": doc["verification"]["all_replays_reproduce_run"]}))


if __name__ == "__main__":
    main()
