#!/usr/bin/env python3
"""Causal event-window strategy: oracle -> noise study and the real surprise-sign rule (DEVELOPMENT).

Inputs are the two feature-eng test fixtures (FIXTURE_RESEARCH_ONLY: Investing.com calendar export and
an hourly EUR/USD file with an unverified clock). They are read in place; per-event rows are written
only under the run directory (outside the repo); the repo receives hashes and aggregates.

Stages
  prep    read both fixtures, detect the bar clock from the NFP jump, build the USD/EUR universe with the
          standardized surprise and post-release windows -> <run>/events.parquet, PREP.json
  oracle  ideal knowledge degraded along four axes (drift pips, direction accuracy, vol x MEAN20,
          half-life bars), 5 seeds, decision 2011-2017 and confirmation 2018-2020 -> cells_*.csv
  real    surprise-sign rule with fitted drift / vol / half-life; grid chosen on 2011-2017, confirmed
          once on 2018-2020; arms NEXT_BAR, LATENCY_0 (oracle bound) and STRADDLE; spread and
          leverage sensitivity -> real.csv, families.csv, contributions.csv, CHOICE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import calendar as CAL  # noqa: E402
from app.short_horizon import events as EV  # noqa: E402
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon import noise as N  # noqa: E402
from app.short_horizon.blocks import OK  # noqa: E402

CAPITAL = 10_000.0
MAX_W = 12
INSET = 1.0e-4            # TP inset (1 pip), ASSUMED
MIN_DIST = 0.5e-4
PERIODS = {"DECISION_2011_2017": ("2011-01-01", "2018-01-01"), "CONFIRM_2018_2020": ("2018-01-01", "2020-05-01")}
DECISION_END = pd.Timestamp("2018-01-01", tz="UTC")
SEEDS = (0, 1, 2, 3, 4)
AXES = {"drift_pips": [0, 2, 5, 10, 15, 20, 30, 40],
        "direction_accuracy": [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5],
        "vol_x_mean20": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        "half_life_bars": [0, 1, 2, 4, 8],
        "entry_latency_bars": [1, 0]}   # 1 = next bar open (achievable); 0 = release-bar open (oracle bound)
GRID = {"m": [1.0, 2.0, 3.0], "q": [0.0, 0.25, 0.5], "k": [1, 2, 3]}
EVENT_SPREADS = {"normal": 0.0, "5pips": 5e-4, "10pips": 10e-4}
LEVERAGES = (10.0, 50.0, 100.0)
# NEXT_BAR_UNGATED is a DIAGNOSTIC only: the causal take/skip gate switched off (every family traded with
# the sign of its decision-period slope, significant or not)
ARMS = ("NEXT_BAR", "LATENCY_0", "STRADDLE", "NEXT_BAR_UNGATED")
KEEP = ("net_profit_usd", "max_drawdown_usd", "max_drawdown_pct", "sharpe_daily_sqrt365", "trades", "win_rate",
        "profit_factor", "trades_per_week", "max_orders_per_day", "weeks_with_eligible_blocks", "eligible_blocks",
        "ruined", "share_exit_tp", "share_exit_sl", "share_exit_time", "margin_call_events", "max_margin_used_pct",
        "unfilled_orders", "capped_orders", "ruin_skipped_orders", "orders_submitted", "refused_entries")


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ prep
def prep(args):
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    cal_df = EV.parse_calendar_fixture(args.calendar)
    bars_lab = EV.parse_hourly_fixture(args.prices)
    nfp = cal_df[(cal_df.family == "Nonfarm Payrolls") & (cal_df.currency == "USD")]
    winter = nfp[nfp.release_utc.dt.hour == 13]
    clock = EV.detect_clock_offset(bars_lab, winter.release_utc)
    bars = EV.bars_to_utc(bars_lab, clock["offset_hours"])
    bars.to_parquet(out / "bars_utc.parquet")
    uni = EV.universe(cal_df, impacts=tuple(args.impacts))
    uni = EV.surprise(uni)
    w = EV.event_windows(bars, uni, max_window=MAX_W)
    w = w[w.entry_bar >= 0].reset_index(drop=True)
    w.to_parquet(out / "events.parquet")
    hi = cal_df[cal_df.currency.isin(["USD", "EUR"]) & (cal_df.impact == "High")]
    by_year = hi.groupby([hi.release_utc.dt.year, "currency"]).size().unstack(fill_value=0)
    numeric = uni.groupby([uni.release_utc.dt.year, "currency"]).size().unstack(fill_value=0)
    fam = uni.groupby(["currency", "family"]).size().sort_values(ascending=False)
    # NFP POC comparison: release-bar |move| vs same label hour one day earlier
    o, c = bars.OPEN, bars.CLOSE
    mv = (c - o).abs()
    rb = w[w.family == "Nonfarm Payrolls"].release_bar.to_numpy(int)
    prev = [bars.index.get_indexer([bars.index[k] - 24 * CAL.HOUR])[0] for k in rb]
    pairs = [(mv.iloc[k], mv.iloc[p]) for k, p in zip(rb, prev) if p >= 0]
    ratio = float(np.mean([a for a, _ in pairs]) / np.mean([b for _, b in pairs])) if pairs else math.nan
    info = {"label": EV.FIXTURE_LABEL, "calendar_sha256": sha(args.calendar), "prices_sha256": sha(args.prices),
            "calendar_rows": int(len(cal_df)), "calendar_span": [str(cal_df.release_utc.min()), str(cal_df.release_utc.max())],
            "bars": int(len(bars)), "bars_span": [str(bars.index[0]), str(bars.index[-1])],
            "clock": clock, "exporter_offset_switches": EV.EXPORTER_OFFSET_SWITCHES,
            "impacts": list(args.impacts), "universe_rows": int(len(uni)), "universe_with_windows": int(len(w)),
            "high_usd_eur_by_year": by_year.to_dict(), "numeric_universe_by_year": numeric.to_dict(),
            "families_top": {f"{c}:{f}": int(n) for (c, f), n in fam.head(40).items()},
            "nfp_release_bar_vol_ratio_vs_previous_day": ratio, "nfp_events_with_windows": int(len(rb))}
    (out / "PREP.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps({"universe": len(uni), "windows": len(w), "clock": clock, "nfp_vol_ratio": ratio}))


# ------------------------------------------------------------------ shared context
class Ctx:
    def __init__(self, run: Path):
        self.run = run
        self.bars = pd.read_parquet(run / "bars_utc.parquet")
        self.w = pd.read_parquet(run / "events.parquet")
        segs = CAL.derive_segments(self.bars.index, max_gap_hours=24)
        self.cal = CAL.load_calendar({"calendar_id": "EURUSD_H1_FIXTURE_EVENT_SESSIONS_DEV", "version": "2026-10-19.v1",
                                      "segments": segs, "entry_delay_after_open_hours": 1,
                                      "force_flat_before_close_hours": 1, "entry_cutoff_before_flat_hours": 1})
        self.release_bars = EV.release_bar_labels(self.w.release_utc)
        n = len(self.w)
        self.hl_true = self.w.half_life.to_numpy(float)
        self.avail = self.w.bars_available.to_numpy(int)
        self.vol_naive_cache = {}

    def col(self, name, win):
        """Per-row value of ``name_{win_i}`` for a per-row window array."""
        win = np.minimum(np.clip(np.asarray(win, int), 1, MAX_W), np.maximum(self.avail, 1))
        M = np.stack([self.w[f"{name}_{h}"].to_numpy(float) for h in range(1, MAX_W + 1)], axis=1)
        return M[np.arange(len(win)), win - 1]

    def dedupe(self, priority):
        """One order per entry bar: keep the row with the largest priority."""
        keep = np.zeros(len(self.w), bool)
        df = pd.DataFrame({"e": self.w.entry_bar.to_numpy(int), "p": np.nan_to_num(priority, nan=-np.inf)})
        idx = df.sort_values(["e", "p"], ascending=[True, False]).drop_duplicates("e").index
        keep[idx] = True
        return keep

    def size(self, sigma_hat, surprise=None):
        ref = EV.rolling_prior_mean(self.w.assign(_v=sigma_hat), "_v", 20)
        p = np.clip(ref / sigma_hat, 0.25, 1.0)
        if surprise is not None:
            p = p * np.clip(0.5 + 0.25 * np.abs(surprise), 0.5, 1.0)
        return np.where(np.isfinite(p), p, np.nan)

    def backtest(self, blocks, orders, period, lev=100.0, event_spread=5e-4, cost=1.0):
        prof = EV.EventProfile(event_spread=event_spread, release_bars=self.release_bars).scaled(cost)
        sigma = EV.causal_event_sigma(blocks)
        a, b = PERIODS[period]
        rules = X.OrderRules()
        checked, skipped = [], 0
        for k, o in enumerate(orders):
            if o is None or blocks.status[k] != OK or not np.isfinite(sigma[k]):
                checked.append(None)
                continue
            try:
                if hasattr(o, "legs"):
                    from app.short_horizon.vstrategies import check_oco
                    check_oco(o, float(sigma[k]), rules)
                else:
                    X.check_order(o, float(blocks.ref[k]), float(sigma[k]), rules)
                checked.append(o)
            except X.OrderRuleError:
                checked.append(None)
                skipped += 1
        res = X.run_backtest(self.bars, blocks, checked, prof, self.cal, CAPITAL, (utc(a), utc(b)), profile=prof,
                             max_orders_per_day=4, ruin_floor=0.0, order_rules=rules, sigma_block=sigma,
                             sizing=X.Sizing(rel_volume=0.10, leverage=lev))
        m = {k: res.metrics.get(k) for k in KEEP}
        m["rule_skipped_orders"] = skipped
        m["valid"] = bool(m["trades_per_week"] >= 3.0 and m["max_orders_per_day"] <= 4 and not m["ruined"])
        return m, res


# ------------------------------------------------------------------ oracle -> noise
def oracle_cells(ctx: Ctx, seeds, k_hl=2, m_sl=2.0):
    w = ctx.w
    n = len(w)
    rows = []
    ts_true = np.clip(np.round(k_hl * ctx.hl_true), 1, MAX_W).astype(int)
    drift_true = ctx.col("drift", ts_true)
    vol_true = ctx.col("realized_vol", ts_true)
    naive_drift_mae = float(np.nanmean(np.abs(drift_true)))
    vol_naive = EV.rolling_prior_mean(w.assign(_v=vol_true), "_v", 20)
    naive_vol_mae = float(np.nanmean(np.abs(vol_naive - vol_true)))
    keep = ctx.dedupe(np.abs(drift_true))
    for seed in seeds:
        rng = np.random.default_rng([seed, 2026_10_19, 7])
        z1, z2, z3, u = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n), rng.uniform(size=n)
        for axis, levels in AXES.items():
            for level in levels:
                d_hat, direction, sig_hat, ts = drift_true.copy(), np.sign(drift_true).astype(int), vol_true.copy(), ts_true
                if axis == "drift_pips":
                    d_hat = drift_true + level * 1e-4 * math.sqrt(math.pi / 2) * z1
                    direction = np.sign(d_hat).astype(int)
                elif axis == "direction_accuracy":
                    elig = keep & np.isfinite(drift_true) & (np.abs(drift_true) - INSET >= MIN_DIST)
                    direction = N.flip_to_accuracy(direction, elig, u, level)
                elif axis == "vol_x_mean20":
                    sig_hat = np.maximum(vol_true + level * naive_vol_mae * math.sqrt(math.pi / 2) * z2, 0.2 * vol_true)
                elif axis == "half_life_bars":
                    hl = np.clip(np.round(ctx.hl_true + level * math.sqrt(math.pi / 2) * z3), 1, MAX_W)
                    ts = np.clip(np.round(k_hl * hl), 1, MAX_W).astype(int)
                elif axis == "entry_latency_bars" and level == 0:
                    # oracle bound: the jump of the release bar is captured too
                    o_rel = ctx.bars.OPEN.to_numpy(float)[w.release_bar.to_numpy(int)]
                    d_hat = drift_true + (ctx.bars.CLOSE.to_numpy(float)[w.release_bar.to_numpy(int)] - o_rel) \
                        + (ctx.bars.OPEN.to_numpy(float)[w.entry_bar.to_numpy(int)] - ctx.bars.CLOSE.to_numpy(float)[w.release_bar.to_numpy(int)])
                    direction = np.sign(d_hat).astype(int)
                tp = np.abs(d_hat) - INSET
                sl = m_sl * sig_hat
                size = ctx.size(sig_hat)
                direction = np.where(keep, direction, 0)
                if axis == "entry_latency_bars" and level == 0:
                    blocks = EV.latency_zero_blocks(ctx.bars, ctx.cal, w, ts)
                else:
                    blocks = EV.event_blocks(ctx.bars, ctx.cal, w, ts)
                orders = EV.directional_orders(blocks, direction, tp, sl, size)
                for period in PERIODS:
                    m, _ = ctx.backtest(blocks, orders, period)
                    a, b = (utc(x) for x in PERIODS[period])
                    sel = (w.release_utc >= a) & (w.release_utc < b) & keep & np.isfinite(drift_true)
                    acc = float(np.mean(direction[sel] == np.sign(drift_true[sel]))) if sel.any() else math.nan
                    m.update({"seed": seed, "axis": axis, "level": level, "period": period,
                              "drift_mae_pips": float(np.nanmean(np.abs(d_hat - drift_true)[sel]) * 1e4),
                              "naive_drift_mae_pips": float(np.nanmean(np.abs(drift_true[sel])) * 1e4),
                              "direction_accuracy": acc,
                              "vol_mae_x_mean20": float(np.nanmean(np.abs(sig_hat - vol_true)[sel]) / naive_vol_mae),
                              "half_life_mae_bars": float(np.nanmean(np.abs(ts / k_hl - ctx.hl_true)[sel]))})
                    rows.append(m)
    return pd.DataFrame(rows)


def oracle(args):
    ctx = Ctx(Path(args.run))
    i, n = (int(x) for x in args.chunk.split("/"))
    seeds = [s for j, s in enumerate(SEEDS) if j % n == i]
    t0 = time.process_time()
    df = oracle_cells(ctx, seeds)
    df.to_csv(ctx.run / f"cells_{i}of{n}.csv", index=False)
    print(json.dumps({"chunk": args.chunk, "rows": len(df), "cpu_seconds": time.process_time() - t0}))


# ------------------------------------------------------------------ real rule
def family_fit(ctx: Ctx, k):
    """Per family: window from the decision-period mean half-life, slope / p / sign on that window."""
    w = ctx.w
    dec = w[w.release_utc < DECISION_END]
    hl = dec.groupby("family").half_life.mean()
    est = []
    for fam, h in hl.items():
        win = int(np.clip(round(k * h), 1, MAX_W))
        e = EV.fit_families(w[w.family == fam], window=win, decision_end=DECISION_END)
        if fam in e.index:
            r = e.loc[fam].to_dict()
            r.update({"family": fam, "window": win, "mean_half_life_decision": float(h)})
            est.append(r)
    return pd.DataFrame(est).set_index("family")


def real_inputs(ctx: Ctx, est, m, q, gated=True):
    w = ctx.w
    fam = w.family.to_numpy()
    tradable = np.array([bool(est.tradable.get(f, False)) if gated else int(est.sign.get(f, 0)) != 0 for f in fam])
    sign = np.array([int(est.sign.get(f, 0)) for f in fam])
    slope = np.array([float(est.slope.get(f, np.nan)) for f in fam])
    win = np.array([int(est.window.get(f, 1)) for f in fam])
    s = w.surprise.to_numpy(float)
    dec = (w.release_utc < DECISION_END).to_numpy() & tradable & np.isfinite(s)
    thr = float(np.quantile(np.abs(s[dec]), q)) if dec.any() and q > 0 else 0.0
    vol_w = ctx.col("realized_vol", win)
    sig_hat = EV.expanding_family_mean(w.assign(_v=vol_w), "_v")
    direction = np.where(tradable & np.isfinite(s) & (np.abs(s) >= thr), np.sign(s) * sign, 0).astype(int)
    tp = np.abs(slope * s) - INSET
    sl = m * sig_hat
    size = ctx.size(sig_hat, s)
    keep = ctx.dedupe(np.where(direction != 0, np.abs(s), -1.0))
    direction = np.where(keep, direction, 0)
    return {"direction": direction, "tp": tp, "sl": sl, "size": size, "win": win, "sig_hat": sig_hat, "thr": thr,
            "drift_true": ctx.col("drift", win), "keep": keep, "tradable": tradable}


def arm_orders(ctx: Ctx, est, inp, arm):
    w = ctx.w
    if arm in ("NEXT_BAR", "NEXT_BAR_UNGATED"):
        blocks = EV.event_blocks(ctx.bars, ctx.cal, w, inp["win"])
        return blocks, EV.directional_orders(blocks, inp["direction"], inp["tp"], inp["sl"], inp["size"])
    if arm == "LATENCY_0":
        blocks = EV.latency_zero_blocks(ctx.bars, ctx.cal, w, inp["win"])
        return blocks, EV.directional_orders(blocks, inp["direction"], inp["tp"], inp["sl"], inp["size"])
    if arm == "STRADDLE":
        blocks = EV.straddle_blocks(ctx.bars, ctx.cal, w, inp["win"])
        jump = EV.expanding_family_mean(w.assign(_v=np.abs(w.jump)), "_v")
        mad = np.array([float(est.mean_abs_drift.get(f, np.nan)) for f in w.family])
        keep = ctx.dedupe(np.where(np.isfinite(jump), jump, -1.0))
        size = np.where(keep, inp["size"] / np.clip(0.5 + 0.25 * np.abs(np.nan_to_num(w.surprise)), 0.5, 1.0), np.nan)
        return blocks, EV.straddle_orders(blocks, 0.5 * jump, np.maximum(mad - INSET, 0), inp["sl"], size)
    raise ValueError(arm)


def real(args):
    ctx = Ctx(Path(args.run))
    rows, fams = [], {}
    t0 = time.process_time()
    for k in GRID["k"]:
        est = family_fit(ctx, k)
        fams[k] = est
        for m in GRID["m"]:
            for q in GRID["q"]:
                for arm in ARMS:
                    inp = real_inputs(ctx, est, m, q, gated=arm != "NEXT_BAR_UNGATED")
                    blocks, orders = arm_orders(ctx, est, inp, arm)
                    r, _ = ctx.backtest(blocks, orders, "DECISION_2011_2017")
                    r.update({"stage": "decision", "arm": arm, "k": k, "m": m, "q": q, "event_spread": "5pips",
                              "leverage": 100.0, "abstain_threshold": inp["thr"]})
                    rows.append(r)
    dec = pd.DataFrame(rows)
    choice = {}
    for arm in ARMS:
        d = dec[(dec.arm == arm) & dec.valid]
        if d.empty:
            d = dec[dec.arm == arm]
        best = d.sort_values("sharpe_daily_sqrt365", ascending=False).iloc[0]
        choice[arm] = {"k": int(best.k), "m": float(best.m), "q": float(best.q), "decision_sharpe": float(best.sharpe_daily_sqrt365),
                       "decision_valid": bool(best.valid)}
    (ctx.run / "CHOICE.json").write_text(json.dumps(choice, indent=1) + "\n")
    contrib = []
    for arm, ch in choice.items():
        est = fams[ch["k"]]
        inp = real_inputs(ctx, est, ch["m"], ch["q"], gated=arm != "NEXT_BAR_UNGATED")
        blocks, orders = arm_orders(ctx, est, inp, arm)
        for period in PERIODS:
            for sp_name, sp in EVENT_SPREADS.items():
                for lev in LEVERAGES:
                    if (sp_name != "5pips" and lev != 100.0):
                        continue
                    r, res = ctx.backtest(blocks, orders, period, lev=lev, event_spread=sp)
                    a, b = (utc(x) for x in PERIODS[period])
                    sel = (ctx.w.release_utc >= a) & (ctx.w.release_utc < b) & (inp["direction"] != 0)
                    acc = float(np.mean(inp["direction"][sel] == np.sign(inp["drift_true"][sel]))) if sel.any() else math.nan
                    r.update({"stage": "confirm" if period.startswith("CONFIRM") else "decision_chosen", "arm": arm,
                              "period": period, "k": ch["k"], "m": ch["m"], "q": ch["q"], "event_spread": sp_name,
                              "leverage": lev, "direction_accuracy": acc,
                              "drift_mae_pips": float(np.nanmean(np.abs(inp["tp"] + INSET - np.abs(inp["drift_true"]))[sel]) * 1e4)
                              if sel.any() else math.nan,
                              "naive_drift_mae_pips": float(np.nanmean(np.abs(inp["drift_true"][sel])) * 1e4) if sel.any() else math.nan,
                              "vol_mae_x_mean20": float(np.nanmean(np.abs(inp["sig_hat"] - ctx.col("realized_vol", inp["win"]))[sel])
                                                        / np.nanmean(np.abs(EV.rolling_prior_mean(ctx.w.assign(_v=ctx.col("realized_vol", inp["win"])), "_v", 20)
                                                                            - ctx.col("realized_vol", inp["win"]))[sel])) if sel.any() else math.nan})
                    rows.append(r)
                    if sp_name == "5pips" and lev == 100.0:
                        fam = ctx.w.family.to_numpy()
                        for t in res.trades:
                            contrib.append({"arm": arm, "period": period, "family": fam[t["block"]], "net_usd": t["net_usd"],
                                            "exit": t["exit_reason"]})
    pd.DataFrame(rows).to_csv(ctx.run / "real.csv", index=False)
    fam_tab = fams[choice["NEXT_BAR"]["k"]].reset_index()
    fam_tab.to_csv(ctx.run / "families.csv", index=False)
    c = pd.DataFrame(contrib)
    if not c.empty:
        c.groupby(["arm", "period", "family"]).agg(trades=("net_usd", "size"), net_usd=("net_usd", "sum"),
                                                   tp=("exit", lambda s: (s == "TP").sum()),
                                                   sl=("exit", lambda s: (s == "SL").sum())).reset_index() \
            .to_csv(ctx.run / "contributions.csv", index=False)
    print(json.dumps({"choice": choice, "rows": len(rows), "cpu_seconds": time.process_time() - t0}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "oracle", "real"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--calendar")
    ap.add_argument("--prices")
    ap.add_argument("--impacts", nargs="+", default=["High"])
    ap.add_argument("--chunk", default="0/1")
    a = ap.parse_args()
    {"prep": prep, "oracle": oracle, "real": real}[a.stage](a)


if __name__ == "__main__":
    main()
