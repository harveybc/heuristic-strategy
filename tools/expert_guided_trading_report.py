#!/usr/bin/env python3
"""Full trading detail and correctness checks of a finished E3-M unit (owner 2026-10-25: "dame los resultados del
trading con todos los detalles ... para que veamos si están bien o hay que hacer cambios"; a wrong strategy makes every
comparison meaningless). Replays every seed's frozen policy and the seed ensemble's primary fitness variant with the
early-close audit's machinery (``expert_guided_early_close_audit.audit_unit``), which asserts that each replay
reproduces the run exactly, and reads the ENGINE's own trade records (``env.BarEnv``; nothing re-simulated here except
the protection re-derivation of the correctness checks, which calls the engine's own ``execution._protect``).

Per unit x policy (seed / ensemble) x period:
  activity      orders, orders/week over the engine's decision weeks (mean + 90 % weekly-bootstrap CI), max per
                business day, weeks with < 2 orders, refused entries by reason
  direction     % buy / % sell; per side: win rate (net > 0), average win / loss, payoff, profit factor,
                expectancy (USD and sigma), net USD, Sharpe of the side's daily P&L
  exits         count / % by reason (TP, SL, TIME = forced flat, AGENT_CLOSE, REVERSAL), per side, win rate and mean
                net per reason
  holding       hours per side (median, p10, p90)
  levels        TP / SL distance from the fill in sigma (the position's causal sigma) and in pips, reward/risk,
                share of entries whose levels the engine clipped
  excursions    MAE / MFE in sigma over the bars the position was open (high/low; declared bar-level
                approximation), share of winners that were more than 1 sigma under water
  costs         spread, slippage, commission, financing (USD) and their share of gross |P&L|; gross vs net
  equity        net USD, return % of the initial balance, Sharpe, Sortino, max DD %, longest losing streak,
                monthly P&L, best / worst trade
  timing        entries by UTC hour and by weekday
  early close   the audit's summary (share, early-close value with CI, closer AUC / recall)
  correctness   business rules and engine contract, PASS / FAIL with counts and examples (see ``CHECKS``)
Reads no TEST row; trains nothing; writes one JSON per unit, a per-trade CSV.gz per policy x period, and reports.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools import expert_guided_early_close_audit as EC  # noqa: E402

PIP = 1e-4
BOOT_N, BOOT_SEED = 2000, 20261025
H_NS = 3600 * 10 ** 9
EXIT_REASONS = ("TP", "SL", "FORCED_FLAT", "AGENT_CLOSE", "REVERSAL")
EXIT_LABEL = {"FORCED_FLAT": "TIME"}
CHECKS = ("sl_and_tp_on_every_order", "sl_tp_on_the_correct_side", "units_follow_the_sizing_rule", "rel_volume_in_0.10_0.20",
          "leverage_at_most_1_50", "integer_positive_units", "at_most_max_entries_per_day", "no_overlapping_positions",
          "entry_fill_at_next_open", "exit_fill_price_matches_reason", "entry_inside_operational_window",
          "closed_before_the_forced_flat", "costs_positive_on_every_trade", "protection_rederived_matches_engine",
          "net_equals_gross_minus_costs")


# ----------------------------------------------------------------------------- per-trade table
def trade_frame(trades: list, ev) -> pd.DataFrame:
    """One row per engine trade with derived columns; ``ev`` = a BarEnv (or a duck-typed view) on the period."""
    rows = []
    for t in trades:
        k, x = int(t["entry_bar_abs"]), int(t["exit_bar_abs"])
        d = int(t["direction"])
        sig = float(ev.sigma[k])
        ent, tp, sl = float(t["entry_price"]), float(t["take_profit"]), float(t["stop_loss"])
        exit_ns = pd.Timestamp(t["exit_time"]).value
        last = x if exit_ns > int(ev.v[x]) else x - 1             # an exit at the open of bar x: the bar was not held
        last = max(last, k)
        hi, lo = float(np.max(ev.h[k:last + 1])), float(np.min(ev.l[k:last + 1]))
        mae = (ent - lo) / sig if d > 0 else (hi - ent) / sig
        mfe = (hi - ent) / sig if d > 0 else (ent - lo) / sig
        u = int(t["units"])
        cost = float(t["spread_usd"] + t["slippage_usd"] + t["commission_usd"])
        rows.append({"entry_time": pd.Timestamp(t["entry_time"]), "exit_time": pd.Timestamp(t["exit_time"]),
                     "entry_bar": k, "exit_bar": x, "side": "BUY" if d > 0 else "SELL", "direction": d, "units": u,
                     "entry_price": ent, "exit_price": float(t["exit_price"]), "take_profit": tp, "stop_loss": sl,
                     "exit_reason": EXIT_LABEL.get(t["close_reason"], t["close_reason"]), "close_reason": t["close_reason"],
                     "ambiguous": bool(t.get("ambiguous", False)), "gross_usd": float(t["gross_usd"]), "net_usd": float(t["net_usd"]),
                     "spread_usd": float(t["spread_usd"]), "slippage_usd": float(t["slippage_usd"]),
                     "commission_usd": float(t["commission_usd"]), "financing_usd": float(t["financing_usd"]),
                     "trading_cost_usd": cost, "entry_cost_usd": float(t["entry_cost_usd"]), "exit_cost_usd": float(t["exit_cost_usd"]),
                     "sigma_entry": sig, "net_sigma": float(t["net_usd"]) / (u * sig) if u * sig > 0 else float("nan"),
                     "hold_hours": (exit_ns - pd.Timestamp(t["entry_time"]).value) / H_NS,
                     "dist_tp_sigma": abs(tp - ent) / sig, "dist_sl_sigma": abs(ent - sl) / sig,
                     "dist_tp_pips": abs(tp - ent) / PIP, "dist_sl_pips": abs(ent - sl) / PIP,
                     "reward_risk": abs(tp - ent) / abs(ent - sl) if ent != sl else float("nan"),
                     "mae_sigma": float(mae), "mfe_sigma": float(mfe), "rel_volume": float(t["rel_volume"]),
                     "percentual_order_size": float(t["percentual_order_size"]), "margin_usd": float(t["margin_usd"])})
    return pd.DataFrame(rows)


def _q(x, qs=(10, 50, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {f"p{q}": None for q in qs}


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def side_stats(df: pd.DataFrame, side: str, capital: float, daily_index) -> dict:
    s = df[df["side"] == side] if len(df) else df
    n = int(len(s))
    if not n:
        return {"trades": 0}
    w, lo = s[s["net_usd"] > 0]["net_usd"], s[s["net_usd"] <= 0]["net_usd"]
    gw, gl = float(w.sum()), float(-lo.sum())
    return {"trades": n, "share_of_entries": n / len(df), "win_rate": float(len(w) / n),
            "avg_win_usd": _f(float(w.mean())) if len(w) else None, "avg_loss_usd": _f(float(lo.mean())) if len(lo) else None,
            "payoff_ratio": _f(float(w.mean() / -lo.mean())) if len(w) and len(lo) and lo.mean() < 0 else None,
            "profit_factor": _f(gw / gl) if gl > 0 else None, "expectancy_usd": float(s["net_usd"].mean()),
            "expectancy_sigma": _f(float(s["net_sigma"].mean())), "net_usd": float(s["net_usd"].sum()),
            "sharpe_daily_sqrt365": _f(EC.side_sharpe([{"direction": 1 if side == "BUY" else -1, "exit_time": t, "net_usd": v}
                                                       for t, v in zip(s["exit_time"], s["net_usd"])], 1 if side == "BUY" else -1,
                                                      daily_index, capital)) if daily_index is not None else None,
            "hold_hours": _q(s["hold_hours"])}


def exit_stats(df: pd.DataFrame) -> dict:
    out = {}
    n = len(df)
    for r in ("TP", "SL", "TIME", "AGENT_CLOSE", "REVERSAL"):
        s = df[df["exit_reason"] == r] if n else df
        out[r] = {"count": int(len(s)), "share": (len(s) / n) if n else None,
                  "buy": int((s["side"] == "BUY").sum()) if len(s) else 0, "sell": int((s["side"] == "SELL").sum()) if len(s) else 0,
                  "win_rate": float((s["net_usd"] > 0).mean()) if len(s) else None, "mean_net_usd": float(s["net_usd"].mean()) if len(s) else None}
    return out


def weekly_activity(df: pd.DataFrame, ev, metrics: dict, min_per_week: float = 2.0) -> dict:
    """Entries per ISO week over the engine's decision weeks (decisions b in [first, last): fill bar b + 1)."""
    dec_v = ev.v[np.arange(ev.first, ev.last) + 1]
    weeks = sorted({tuple(pd.Timestamp(int(t), tz="UTC").isocalendar()[:2]) for t in dec_v})
    per = {w: 0 for w in weeks}
    for t in df["entry_bar"] if len(df) else []:
        w = tuple(pd.Timestamp(int(ev.v[int(t)]), tz="UTC").isocalendar()[:2])
        per[w] = per.get(w, 0) + 1
    x = np.array([per[w] for w in weeks], float)
    lo, hi = boot_mean_ci(x, 0.10)
    refused = {k[len("n_refused_"):]: int(v) for k, v in metrics.items() if k.startswith("n_refused_")}
    return {"orders": int(len(df)), "weeks": len(weeks), "orders_per_week_mean": float(x.mean()) if len(x) else 0.0,
            "orders_per_week_ci90": [lo, hi], "engine_orders_per_week": _f(metrics.get("orders_per_week")),
            "weeks_below_2": int((x < min_per_week).sum()), "weeks_below_2_share": float((x < min_per_week).mean()) if len(x) else None,
            "weeks_with_zero": int((x == 0).sum()), "max_entries_per_business_day": int(metrics.get("max_entries_per_day", 0)),
            "refused_entries": refused, "ignored_same_direction": int(metrics.get("n_ignored_same_direction", 0)),
            "levels_clipped": int(metrics.get("n_levels_clipped", 0)),
            "levels_clipped_share": (metrics.get("n_levels_clipped", 0) / len(df)) if len(df) else None}


def boot_mean_ci(x, alpha=0.10, n=BOOT_N, seed=BOOT_SEED):
    x = np.asarray(x, float)
    if not len(x):
        return None, None
    if np.all(x == x[0]):
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def longest_losing_streak(net) -> int:
    best = cur = 0
    for v in net:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def equity_stats(df: pd.DataFrame, metrics: dict, capital: float) -> dict:
    de = metrics.get("daily_equity")
    monthly = {}
    if isinstance(de, pd.Series) and len(de):
        idx = pd.DatetimeIndex(de.index)
        s = pd.Series(np.asarray(de, float), index=idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx)
        first = float(capital)
        for per, g in s.groupby(s.index.to_period("M")):
            monthly[str(per)] = float(g.iloc[-1] - first)
            first = float(g.iloc[-1])
    net = float(metrics.get("net_profit_usd", 0.0))
    return {"net_profit_usd": net, "return_pct": 100 * net / capital, "sharpe_daily_sqrt365": _f(metrics.get("sharpe_daily_sqrt365")),
            "sortino_daily_sqrt365": _f(metrics.get("sortino_daily_sqrt365")), "max_drawdown_pct": _f(metrics.get("max_drawdown_pct")),
            "win_rate": float((df["net_usd"] > 0).mean()) if len(df) else None,
            "profit_factor": _f(float(df[df.net_usd > 0].net_usd.sum() / -df[df.net_usd <= 0].net_usd.sum()))
            if len(df) and (df.net_usd <= 0).any() and df[df.net_usd <= 0].net_usd.sum() < 0 else None,
            "expectancy_usd": float(df["net_usd"].mean()) if len(df) else None,
            "expectancy_sigma": _f(float(df["net_sigma"].mean())) if len(df) else None,
            "longest_losing_streak": longest_losing_streak(df["net_usd"]) if len(df) else 0,
            "best_trade_usd": float(df["net_usd"].max()) if len(df) else None, "worst_trade_usd": float(df["net_usd"].min()) if len(df) else None,
            "monthly_pnl_usd": monthly}


def cost_stats(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    g = float(df["gross_usd"].abs().sum())
    out = {k: float(df[f"{k}_usd"].sum()) for k in ("spread", "slippage", "commission", "financing")}
    out.update({"gross_usd": float(df["gross_usd"].sum()), "net_usd": float(df["net_usd"].sum()),
                "trading_cost_usd": float(df["trading_cost_usd"].sum()),
                "cost_share_of_gross_abs": (float(df["trading_cost_usd"].sum()) / g) if g > 0 else None})
    return out


def level_stats(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    return {k: _q(df[k]) for k in ("dist_tp_sigma", "dist_sl_sigma", "dist_tp_pips", "dist_sl_pips", "reward_risk")}


def excursion_stats(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    w = df[df["net_usd"] > 0]
    return {"mae_sigma": _q(df["mae_sigma"], (50, 90)), "mfe_sigma": _q(df["mfe_sigma"], (50, 90)),
            "winners_under_water_gt_1sigma_share": float((w["mae_sigma"] > 1.0).mean()) if len(w) else None}


def timing_stats(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    t = pd.DatetimeIndex(df["entry_time"]).tz_convert("UTC")
    return {"by_utc_hour": {int(h): int(n) for h, n in pd.Series(t.hour).value_counts().sort_index().items()},
            "by_weekday": {str(d): int(n) for d, n in pd.Series(t.day_name()).value_counts().items()}}


# ----------------------------------------------------------------------------- correctness
def correctness(df: pd.DataFrame, ev, cfg, capital: float) -> dict:
    """Every business rule / engine-contract check, PASS / FAIL with the number of offending trades and examples."""
    res = {c: {"checked": 0, "failed": 0, "examples": []} for c in CHECKS}

    def chk(name, ok, i, why=""):
        r = res[name]
        r["checked"] += 1
        if not ok:
            r["failed"] += 1
            if len(r["examples"]) < 5:
                r["examples"].append({"trade": int(i), "why": why})
    if not len(df):
        return {c: {**v, "status": "NOT_APPLICABLE_NO_TRADES"} for c, v in res.items()}
    df = df.sort_values("entry_bar").reset_index(drop=True)
    sz = cfg.sizing
    through = float(cfg.profile.limit_through)
    balance = float(capital)
    prev_exit = None
    per_day = {}
    for i, t in df.iterrows():
        d, k, x = int(t.direction), int(t.entry_bar), int(t.exit_bar)
        si = int(ev.seg_of[k])
        chk("sl_and_tp_on_every_order", math.isfinite(t.stop_loss) and math.isfinite(t.take_profit) and t.stop_loss > 0 and t.take_profit > 0, i)
        chk("sl_tp_on_the_correct_side", (t.stop_loss < t.entry_price < t.take_profit) if d > 0 else (t.take_profit < t.entry_price < t.stop_loss), i,
            f"sl={t.stop_loss} entry={t.entry_price} tp={t.take_profit}")
        exp_u = sz.units(balance, float(ev.c[k - 1]), float(t.percentual_order_size))
        chk("units_follow_the_sizing_rule", int(t.units) == int(exp_u), i, f"units={t.units} expected={exp_u} balance={balance:.2f}")
        chk("rel_volume_in_0.10_0.20", 0.10 - 1e-12 <= t.rel_volume <= 0.20 + 1e-12, i, f"rel_volume={t.rel_volume}")
        chk("leverage_at_most_1_50", float(sz.leverage) <= 50.0 + 1e-12, i, f"leverage={sz.leverage}")
        chk("integer_positive_units", float(t.units) == int(t.units) and int(t.units) >= 1, i)
        day = ev.business_day(k)
        per_day[day] = per_day.get(day, 0) + 1
        chk("at_most_max_entries_per_day", per_day[day] <= cfg.max_entries_per_day, i, f"day={day} n={per_day[day]}")
        chk("no_overlapping_positions", prev_exit is None or pd.Timestamp(t.entry_time) >= prev_exit, i,
            f"entry={t.entry_time} previous exit={prev_exit}")
        chk("entry_fill_at_next_open", abs(t.entry_price - float(ev.o[k])) < 1e-12 and pd.Timestamp(t.entry_time).value == int(ev.v[k]), i,
            f"entry={t.entry_price} open[k]={ev.o[k]}")
        if t.close_reason in ("AGENT_CLOSE", "REVERSAL"):
            ok = abs(t.exit_price - float(ev.o[x])) < 1e-12
        elif t.close_reason == "FORCED_FLAT":
            ok = abs(t.exit_price - float(ev.c[x])) < 1e-12
        elif t.close_reason == "TP":
            ok = abs(t.exit_price - t.take_profit) < 1e-12 or abs(t.exit_price - float(ev.o[x])) < 1e-12
        else:
            ok = abs(t.exit_price - t.stop_loss) < 1e-12 or abs(t.exit_price - float(ev.o[x])) < 1e-12
        chk("exit_fill_price_matches_reason", ok, i, f"{t.close_reason} exit={t.exit_price}")
        vk = int(ev.v[k])
        chk("entry_inside_operational_window", int(ev.seg_of[k - 1]) == si and vk >= int(ev.op[si]) and vk < int(ev.ec[si]) and vk + H_NS < int(ev.ff[si]),
            i, f"entry={t.entry_time}")
        chk("closed_before_the_forced_flat", pd.Timestamp(t.exit_time).value <= int(ev.ff[si]) and int(ev.seg_of[x]) == si, i, f"exit={t.exit_time}")
        chk("costs_positive_on_every_trade", t.spread_usd > 0 and t.commission_usd > 0 and t.trading_cost_usd > 0, i)
        # the engine's own protection, re-derived bar by bar from the fill to the exit
        ok, why = True, ""
        stop_at = x if t.close_reason in ("TP", "SL") else (x - 1 if t.close_reason in ("AGENT_CLOSE", "REVERSAL") else x)
        for j in range(k, stop_at + 1):
            hit = X._protect(d, t.take_profit, t.stop_loss, ev.o[j], ev.h[j], ev.l[j], through)
            if j < x or t.close_reason not in ("TP", "SL"):
                if hit is not None:
                    ok, why = False, f"protection {hit[0]} on bar {j} before the recorded exit {t.close_reason} at {x}"
                    break
            else:
                if hit is None or hit[0] != t.close_reason or abs(float(hit[1]) - t.exit_price) > 1e-12:
                    ok, why = False, f"recorded {t.close_reason}@{t.exit_price} vs re-derived {hit}"
        chk("protection_rederived_matches_engine", ok, i, why)
        chk("net_equals_gross_minus_costs", abs(t.net_usd - (t.gross_usd - t.entry_cost_usd - t.exit_cost_usd)) <= 1e-9 * max(1.0, abs(t.gross_usd)), i)
        balance += float(t.net_usd)
        prev_exit = pd.Timestamp(t.exit_time)
    for v in res.values():
        v["status"] = "PASS" if v["failed"] == 0 else "FAIL"
    return res


def detail(df: pd.DataFrame, ev, metrics: dict, cfg, capital: float, ec_summary: dict) -> dict:
    de = metrics.get("daily_equity")
    di = pd.DatetimeIndex(de.index) if isinstance(de, pd.Series) else None
    n = len(df)
    cor = correctness(df, ev, cfg, capital)
    return {"activity": weekly_activity(df, ev, metrics),
            "direction": {"buy_share": float((df.side == "BUY").mean()) if n else None, "sell_share": float((df.side == "SELL").mean()) if n else None,
                          "BUY": side_stats(df, "BUY", capital, di), "SELL": side_stats(df, "SELL", capital, di)},
            "exits": exit_stats(df), "levels": level_stats(df), "excursions": excursion_stats(df), "costs": cost_stats(df),
            "equity": equity_stats(df, metrics, capital), "timing": timing_stats(df),
            "holding_hours": {"all": _q(df["hold_hours"]) if n else None},
            "early_close": {k: ec_summary.get(k) for k in ("early_close_share", "agent_closes", "classes", "class_shares", "early_close_value_usd",
                                                           "early_close_value_usd_ci95", "early_close_value_sigma", "closer", "no_early_close")},
            "correctness": cor, "correctness_pass": sum(1 for v in cor.values() if v.get("status") == "PASS"),
            "correctness_fail": sum(1 for v in cor.values() if v.get("status") == "FAIL")}


# ----------------------------------------------------------------------------- one unit
def run_unit(dirs: list, ds, periods: list, ensemble_path, csv_dir: Path, unit: str) -> dict:
    from tools.expert_guided_e3m import effective_contracts
    e0 = json.loads((Path(dirs[0]) / "E3M.json").read_text())
    SCFG, _ = effective_contracts(e0.get("exec_contract", "sac"), e0.get("min_orders_per_week"))
    capital = float(SCFG.capital)
    out = {"unit": unit, "config": {"capital": capital, "max_entries_per_day": SCFG.max_entries_per_day,
                                    "rel_volume": SCFG.sizing.rel_volume, "leverage": SCFG.sizing.leverage,
                                    "level_policy": SCFG.level_policy, "min_orders_per_week": SCFG.min_orders_per_week},
           "policies": {}}
    csv_dir.mkdir(parents=True, exist_ok=True)

    def on_run(what, p, au, mk, ec_summary):
        ev = mk()
        df = trade_frame(au["trades"], ev)
        who = what.rsplit("/", 1)[0]
        out["policies"].setdefault(who, {})[p] = detail(df, ev, au["metrics"], SCFG, capital, ec_summary)
        with gzip.open(csv_dir / f"{unit}__{who.replace('/', '_')}__{p}.csv.gz", "wt") as fh:
            df.to_csv(fh, index=False)
    doc = EC.audit_unit(dirs, ds, periods, ensemble_path, on_run=on_run, only_primary=True)
    out["verification"] = doc["verification"]
    out["seeds"] = doc["seeds"]
    out["ensemble_primary"] = next((k for k, v in doc["ensemble"].items() if isinstance(v, dict) and v.get("primary")), None)
    out["notes"] = {k: v for k, v in doc.items() if k.endswith("_note")}
    return out


# ----------------------------------------------------------------------------- report
def _p(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f} %"


def _n(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def summary_row(unit, who, p, x):
    a, dr, ex, eq, ec = x["activity"], x["direction"], x["exits"], x["equity"], x["early_close"]
    b, s = dr["BUY"], dr["SELL"]
    exits = " / ".join(f"{(ex[r]['share'] or 0) * 100:.0f}" for r in ("TP", "SL", "TIME", "AGENT_CLOSE", "REVERSAL"))
    ci = ec.get("early_close_value_usd_ci95") or [None, None]
    ecv = "—" if ec.get("early_close_value_usd") is None else f"{ec['early_close_value_usd']:.0f} [{_n(ci[0], 0)}, {_n(ci[1], 0)}]"
    return (f"| {unit} | {who} | {p} | {a['orders']} | {_n(a['orders_per_week_mean'])} | {_p(dr['buy_share'])} | {_p(b.get('win_rate'))} | "
            f"{_p(s.get('win_rate'))} | {_p(eq['win_rate'])} | {_n(eq['profit_factor'])} | {_n(eq['expectancy_sigma'], 3)} | "
            f"{eq['net_profit_usd']:.0f} | {_n(eq['sharpe_daily_sqrt365'])} | {_n(eq['max_drawdown_pct'], 1)} | {exits} | {ecv} | "
            f"{x['correctness_pass']} / {x['correctness_fail']} |")


SUMMARY_HEAD = ("| unit | policy | period | orders | orders/wk | % buy | % buy winning | % sell winning | win rate | profit factor | "
                "expectancy σ | net USD | Sharpe | max DD % | exits % TP / SL / TIME / AGENT / REV | early-close value USD [95 % CI] | checks PASS / FAIL |\n"
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")


def section(unit, who, p, x) -> list:
    a, dr, ex, eq = x["activity"], x["direction"], x["exits"], x["equity"]
    L = [f"#### {unit} — {who} — {p}", ""]
    L.append(f"- **Activity**: {a['orders']} orders over {a['weeks']} decision weeks; {_n(a['orders_per_week_mean'])} /wk "
             f"(90 % CI {_n(a['orders_per_week_ci90'][0])}–{_n(a['orders_per_week_ci90'][1])}; engine {_n(a['engine_orders_per_week'])}); "
             f"weeks < 2: {a['weeks_below_2']} ({_p(a['weeks_below_2_share'])}), weeks with 0: {a['weeks_with_zero']}; max per business day "
             f"{a['max_entries_per_business_day']}; refused: " + ", ".join(f"{k} {v}" for k, v in a["refused_entries"].items() if v) +
             f"; levels clipped {a['levels_clipped']} ({_p(a['levels_clipped_share'])})")
    L += ["", "| side | trades | share | win rate | avg win USD | avg loss USD | payoff | profit factor | expectancy USD | expectancy σ | net USD | Sharpe | hold h p10 / p50 / p90 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for sd in ("BUY", "SELL"):
        s = dr[sd]
        if not s.get("trades"):
            L.append(f"| {sd} | 0 | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        hh = s["hold_hours"]
        L.append(f"| {sd} | {s['trades']} | {_p(s['share_of_entries'])} | {_p(s['win_rate'])} | {_n(s['avg_win_usd'], 1)} | {_n(s['avg_loss_usd'], 1)} | "
                 f"{_n(s['payoff_ratio'])} | {_n(s['profit_factor'])} | {_n(s['expectancy_usd'], 1)} | {_n(s['expectancy_sigma'], 3)} | {s['net_usd']:.0f} | "
                 f"{_n(s['sharpe_daily_sqrt365'])} | {_n(hh['p10'], 0)} / {_n(hh['p50'], 0)} / {_n(hh['p90'], 0)} |")
    L += ["", "| exit | count | share | buy | sell | win rate | mean net USD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in ("TP", "SL", "TIME", "AGENT_CLOSE", "REVERSAL"):
        e = ex[r]
        L.append(f"| {r} | {e['count']} | {_p(e['share'])} | {e['buy']} | {e['sell']} | {_p(e['win_rate'])} | {_n(e['mean_net_usd'], 1)} |")
    lv, xc, co = x["levels"], x["excursions"], x["costs"]
    if lv:
        L += ["", "- **Levels** (p10 / p50 / p90): TP " + " / ".join(_n(lv['dist_tp_sigma'][k]) for k in ("p10", "p50", "p90")) + " σ = "
              + " / ".join(_n(lv['dist_tp_pips'][k], 1) for k in ("p10", "p50", "p90")) + " pips; SL "
              + " / ".join(_n(lv['dist_sl_sigma'][k]) for k in ("p10", "p50", "p90")) + " σ = "
              + " / ".join(_n(lv['dist_sl_pips'][k], 1) for k in ("p10", "p50", "p90")) + " pips; reward/risk "
              + " / ".join(_n(lv['reward_risk'][k]) for k in ("p10", "p50", "p90")),
              f"- **Excursions**: MAE p50 / p90 {_n(xc['mae_sigma']['p50'])} / {_n(xc['mae_sigma']['p90'])} σ; MFE p50 / p90 "
              f"{_n(xc['mfe_sigma']['p50'])} / {_n(xc['mfe_sigma']['p90'])} σ; winners more than 1 σ under water: {_p(xc['winners_under_water_gt_1sigma_share'])}",
              f"- **Costs**: spread {co['spread']:.0f}, slippage {co['slippage']:.0f}, commission {co['commission']:.0f}, financing {co['financing']:.0f} USD; "
              f"gross {co['gross_usd']:.0f} → net {co['net_usd']:.0f} USD; trading costs = {_p(co['cost_share_of_gross_abs'])} of gross |P&L|"]
    L.append(f"- **Equity**: net {eq['net_profit_usd']:.0f} USD ({_n(eq['return_pct'], 1)} % of the initial balance); Sharpe {_n(eq['sharpe_daily_sqrt365'])}, "
             f"Sortino {_n(eq['sortino_daily_sqrt365'])}, max DD {_n(eq['max_drawdown_pct'], 1)} %; win rate {_p(eq['win_rate'])}, profit factor "
             f"{_n(eq['profit_factor'])}; longest losing streak {eq['longest_losing_streak']}; best / worst trade {_n(eq['best_trade_usd'], 0)} / {_n(eq['worst_trade_usd'], 0)} USD")
    if eq["monthly_pnl_usd"]:
        L.append("- **Monthly P&L USD**: " + ", ".join(f"{k} {v:+.0f}" for k, v in eq["monthly_pnl_usd"].items()))
    tm = x["timing"]
    if tm:
        L.append("- **Entries by UTC hour**: " + ", ".join(f"{h}h {n}" for h, n in tm["by_utc_hour"].items()))
        L.append("- **Entries by weekday**: " + ", ".join(f"{d} {n}" for d, n in tm["by_weekday"].items()))
    ec = x["early_close"]
    cl = ec.get("closer") or {}
    L.append(f"- **Early close**: {_p(ec.get('early_close_share'))} of exits ({ec.get('agent_closes')}); classes {ec.get('class_shares')}; value "
             f"{_n(ec.get('early_close_value_usd'), 0)} USD CI {ec.get('early_close_value_usd_ci95')}; closer AUC(SL first) {_n(cl.get('auc_sl_first'))}, "
             f"precision {_n(cl.get('precision_sl_first'))}, recall {_n(cl.get('recall_sl_first'))}")
    L += ["", "| check | status | checked | failed | example |", "| --- | --- | ---: | ---: | --- |"]
    for c in CHECKS:
        v = x["correctness"][c]
        L.append(f"| {c} | {v['status']} | {v['checked']} | {v['failed']} | {(v['examples'][0]['why'] if v['examples'] else '')[:120]} |")
    return L + [""]


def write_report(unit_paths: list, out_json: str, out_md: str, title: str):
    units = [json.loads(Path(p).read_text()) for p in unit_paths]
    fails = []
    rows, secs = [], []
    for u in units:
        ens = [k for k in u["policies"] if k.startswith("ensemble")]
        for who in ens + [k for k in u["policies"] if not k.startswith("ensemble")]:
            for p, x in u["policies"][who].items():
                for c, v in x["correctness"].items():
                    if v.get("status") == "FAIL":
                        fails.append(f"| {u['unit']} | {who} | {p} | {c} | {v['failed']} / {v['checked']} | {(v['examples'][0]['why'] if v['examples'] else '')[:140]} |")
                if who.startswith("ensemble") and p == "INNER_B":
                    rows.append(summary_row(u["unit"], who, p, x))
    md = [f"# {title}", "",
          "Replays of the frozen policies (each seed and the seed ensemble's primary fitness variant), verified to reproduce the "
          "runs exactly; numbers from the engine's own trade records. INNER_A = calibration block, INNER_B = decision year (frozen), "
          "OBS_2024 = the single previously observed read (never decides). TEST closed. Generated by `tools/expert_guided_trading_report.py`.", ""]
    if fails:
        md += ["## <span style=\"color:red\">CORRECTNESS FAILURES — strategy / engine defects</span>", "",
               "| unit | policy | period | check | failed / checked | example |", "| --- | --- | --- | --- | --- | --- |"] + fails + [""]
    else:
        md += ["## Correctness: every check PASSES on every unit, policy and period", ""]
    md += ["## Summary — seed ensemble (primary fitness), decision year INNER_B", "", SUMMARY_HEAD] + rows + [""]
    md += ["## Summary — every policy and period", "", SUMMARY_HEAD]
    for u in units:
        for who, per in u["policies"].items():
            for p, x in per.items():
                md.append(summary_row(u["unit"], who, p, x))
    md += ["", "## Detail", ""]
    for u in units:
        md += [f"### {u['unit']}", "", f"Config: {json.dumps(u['config'])}; replays reproduce the run: "
               f"{u['verification'].get('all_replays_reproduce_run')}; ensemble primary fitness: {u.get('ensemble_primary')}", ""]
        order = [k for k in u["policies"] if k.startswith("ensemble")] + [k for k in u["policies"] if not k.startswith("ensemble")]
        for who in order:
            for p, x in u["policies"][who].items():
                md += section(u["unit"], who, p, x)
    Path(out_md).write_text("\n".join(md) + "\n")
    Path(out_json).write_text(json.dumps({"title": title, "units": units, "correctness_failures": len(fails)}, indent=1, default=float))


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
    r.add_argument("--ensemble", default=None)
    r.add_argument("--csv-dir", required=True)
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
    unit = Path(a.arm_dir).name
    doc = run_unit([a.arm_dir] + a.extra_arm_dir, ds, periods, ens if ens.exists() else None, Path(a.csv_dir), unit)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float))
    fails = sum(x["correctness_fail"] for per in doc["policies"].values() for x in per.values())
    print(json.dumps({"out": a.out, "seeds": doc["seeds"], "reproduce": doc["verification"].get("all_replays_reproduce_run"), "correctness_fail_checks": fails}))


if __name__ == "__main__":
    main()
