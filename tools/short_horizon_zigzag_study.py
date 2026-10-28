#!/usr/bin/env python3
"""ZigZag hindsight-expert study (DEVELOPMENT): MetaTrader ZigZag pivots as a NON-CAUSAL expert, the
causal-confirmed counterpart, the DEAP NSGA-II parameter front, the timing-noise study, a learned
causal pivot detector and the sealed demonstration dataset for the learning-from-demonstrations lane.

Stages (all on the lake EUR/USD H1 file, pinned digest; TEST rows refused by the loader)
  nsga      NSGA-II over (Depth, Deviation, Backstep) on 2020-2022, hindsight version (a); front validated
            on 2023, chosen point confirmed once on 2024 -> front.csv, CHOICE.json
  confirmed version (b) with the SAME parameters: pivots timestamped at their confirmation bar, TP = k * sigma
            with k chosen on 2020-2022; lag distribution -> confirmed.csv, LAGS.json
  noise     hindsight pivots delayed by d bars and jittered -> Sharpe vs pivot-timing error -> noise.csv
  detector  learned causal pivot detector (small MLP, torch, GPU if allowed) on causal lake features;
            trained 2020-2022, threshold/k chosen 2023, confirmed once 2024; sealed predictions -> detector.csv
  demos     sealed expert-trade CSV for the chosen parameters -> demonstrations/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon.blocks import OK  # noqa: E402
from tools.short_horizon_oracle import load_inputs  # noqa: E402

CAPITAL = 10_000.0
PERIODS = {"DECISION_2020_2022": ("2020-01-01", "2023-01-01"), "VALIDATION_2023": ("2023-01-01", "2024-01-01"),
           "CONFIRM_2024": ("2024-01-01", "2025-01-01")}
GENES = (("depth", 3, 60), ("deviation_pips", 1, 60), ("backstep", 1, 10))
K_TP = (0.5, 1.0, 1.5, 2.0)
M_SL = 2.0
DELAYS = (0, 1, 2, 3, 5, 8, 12)
JITTERS = (0, 1, 2, 4)
SEEDS = (0, 1, 2, 3, 4)
KEEP = ("net_profit_usd", "max_drawdown_usd", "max_drawdown_pct", "sharpe_daily_sqrt365", "trades", "win_rate",
        "trades_per_week", "max_orders_per_day", "ruined", "share_exit_tp", "share_exit_sl", "share_exit_time",
        "capped_orders", "unfilled_orders")


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def per(name):
    a, b = PERIODS[name]
    return utc(a), utc(b)


class Ctx:
    def __init__(self, data):
        self.bars, self.cal, self.digest = load_inputs(data)
        self.h, self.l = self.bars.HIGH.to_numpy(float), self.bars.LOW.to_numpy(float)
        self.sigma = Z.causal_sigma(self.bars)
        self.sigma_rule = Z.causal_range_sigma(self.bars)
        self.prof = X.RealisticProfile()
        self._piv = {}

    def pivots(self, p: Z.Params):
        key = (p.depth, p.deviation_pips, p.backstep)
        if key not in self._piv:
            self._piv[key] = Z.hindsight_pivots(self.h, self.l, p)
        return self._piv[key]

    def run(self, signals, pivots, period, mode="hindsight", k_tp=1.0, m_sl=M_SL, lev=100.0):
        blocks = Z.signal_blocks(self.bars, self.cal, signals)
        orders = Z.reversal_orders(blocks, signals, pivots, self.sigma, mode=mode, k_tp=k_tp, m_sl=m_sl,
                                   sigma_rule=self.sigma_rule)
        sig_blk = self.sigma_rule[np.minimum(blocks.bar_lo, len(self.bars) - 1)]
        checked, skipped = [], 0
        for k, o in enumerate(orders):
            if o is None:
                checked.append(None)
                continue
            try:
                X.check_order(o, float(blocks.ref[k]), float(sig_blk[k]), X.OrderRules())
                checked.append(o)
            except X.OrderRuleError:
                checked.append(None)
                skipped += 1
        res = X.run_backtest(self.bars, blocks, checked, self.prof, self.cal, CAPITAL, per(period), profile=self.prof,
                             max_orders_per_day=4, ruin_floor=0.0, order_rules=X.OrderRules(), sigma_block=sig_blk,
                             sizing=X.Sizing(rel_volume=0.10, leverage=lev))
        m = {k: res.metrics[k] for k in KEEP}
        m["rule_skipped_orders"] = skipped
        m["valid"] = bool(m["trades_per_week"] >= 3.0 and m["max_orders_per_day"] <= 4 and not m["ruined"])
        m["period"] = period
        return m, res


def params_of(row):
    return Z.Params(int(row["depth"]), float(row["deviation_pips"]), int(row["backstep"]))


# ------------------------------------------------------------------ NSGA-II (hindsight, version a)
def nsga(args):
    from deap import base, creator, tools
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args.data)
    if not hasattr(creator, "FitZZ"):
        creator.create("FitZZ", base.Fitness, weights=(1.0, -1.0))
        creator.create("IndZZ", list, fitness=creator.FitZZ)
    lo, hi = [g[1] for g in GENES], [g[2] for g in GENES]
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=lo, up=hi, eta=15.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=lo, up=hi, eta=15.0, indpb=0.5)
    tb.register("select", tools.selNSGA2)

    def decode(ind):
        return Z.Params(int(round(min(max(ind[0], lo[0]), hi[0]))), float(round(min(max(ind[1], lo[1]), hi[1]), 1)),
                        int(round(min(max(ind[2], lo[2]), hi[2]))))

    cache = {}

    def evaluate(ind):
        p = decode(ind)
        if p not in cache:
            piv = ctx.pivots(p)
            m, _ = ctx.run(Z.signals_from_pivots(piv), piv, "DECISION_2020_2022")
            cache[p] = (m["sharpe_daily_sqrt365"] if m["valid"] else -10.0, m["max_drawdown_pct"] if m["valid"] else 100.0)
        return cache[p]

    ck = run / "nsga_checkpoint.json"
    rng = random.Random(args.seed)
    if ck.exists():
        st = json.loads(ck.read_text())
        pop = [creator.IndZZ(g) for g in st["population"]]
        for ind, f in zip(pop, st["fitness"]):
            ind.fitness.values = tuple(f)
        pop = tb.select(pop, len(pop))
        gen0, history = st["generation"], st["history"]
        rng = random.Random(args.seed + 1000 * gen0)
    else:
        pop = [creator.IndZZ([rng.uniform(a, b) for a, b in zip(lo, hi)]) for _ in range(args.pop)]
        for ind in pop:
            ind.fitness.values = evaluate(ind)
        pop = tb.select(pop, len(pop))
        gen0, history = 0, []
    t0 = time.process_time()
    for gen in range(gen0 + 1, gen0 + args.generations + 1):
        off = [creator.IndZZ(list(i)) for i in tools.selTournamentDCD(pop, len(pop))]
        for a, b in zip(off[::2], off[1::2]):
            if rng.random() < 0.9:
                tb.mate(a, b)
            tb.mutate(a), tb.mutate(b)
        for ind in off:
            ind.fitness.values = evaluate(ind)
        pop = tb.select(pop + off, args.pop)
        history.append({"generation": gen, "best_sharpe": max(i.fitness.values[0] for i in pop),
                        "min_dd": min(i.fitness.values[1] for i in pop)})
        ck.write_text(json.dumps({"generation": gen, "population": [list(map(float, i)) for i in pop],
                                  "fitness": [list(i.fitness.values) for i in pop], "history": history}, indent=1))
        if time.process_time() - t0 > args.budget_cpu_seconds:
            break
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    rows, seen = [], set()
    for ind in front:
        p = decode(ind)
        if p in seen:
            continue
        seen.add(p)
        piv = ctx.pivots(p)
        d, _ = ctx.run(Z.signals_from_pivots(piv), piv, "DECISION_2020_2022")
        v, _ = ctx.run(Z.signals_from_pivots(piv), piv, "VALIDATION_2023")
        rows.append({"depth": p.depth, "deviation_pips": p.deviation_pips, "backstep": p.backstep, "pivots": len(piv),
                     **{f"dec_{k}": d[k] for k in ("sharpe_daily_sqrt365", "max_drawdown_pct", "net_profit_usd", "trades_per_week", "valid")},
                     **{f"val_{k}": v[k] for k in ("sharpe_daily_sqrt365", "max_drawdown_pct", "net_profit_usd", "trades_per_week", "valid")}})
    fr = pd.DataFrame(rows).sort_values("dec_sharpe_daily_sqrt365", ascending=False)
    fr["frequency_rule"] = np.where(fr.dec_valid & fr.val_valid, "ACCEPTED", "REJECTED")
    fr.to_csv(run / "front.csv", index=False)
    ok = fr[fr.frequency_rule == "ACCEPTED"]
    ch = (ok if not ok.empty else fr).sort_values("val_sharpe_daily_sqrt365", ascending=False).iloc[0]
    p = params_of(ch)
    piv = ctx.pivots(p)
    conf, res = ctx.run(Z.signals_from_pivots(piv), piv, "CONFIRM_2024")
    lev = {int(L): ctx.run(Z.signals_from_pivots(piv), piv, "CONFIRM_2024", lev=float(L))[0] for L in (10, 50)}
    (run / "CHOICE.json").write_text(json.dumps(
        {"params": {"depth": p.depth, "deviation_pips": p.deviation_pips, "backstep": p.backstep},
         "chosen_by": "best 2023 Sharpe among front points accepted by the frequency rule on both 2020-2022 and 2023",
         "decision": {k: ch[f"dec_{k}"] for k in ("sharpe_daily_sqrt365", "max_drawdown_pct", "net_profit_usd", "trades_per_week", "valid")},
         "validation_2023": {k: ch[f"val_{k}"] for k in ("sharpe_daily_sqrt365", "max_drawdown_pct", "net_profit_usd", "trades_per_week", "valid")},
         "confirm_2024": conf, "confirm_2024_leverage": lev, "generations": history[-1]["generation"] if history else 0,
         "population": args.pop, "history": history, "front_points": int(len(fr)), "front_accepted": int(len(ok)),
         "label": Z.EXPERT_LABEL}, indent=1, default=str) + "\n")
    print(json.dumps({"front": len(fr), "accepted": len(ok), "chosen": [p.depth, p.deviation_pips, p.backstep],
                      "confirm_2024_sharpe": conf["sharpe_daily_sqrt365"], "cpu": time.process_time() - t0}))


# ------------------------------------------------------------------ confirmed (version b)
def confirmed(args):
    run = Path(args.run)
    ctx = Ctx(args.data)
    ch = json.loads((run / "CHOICE.json").read_text())
    fr = pd.read_csv(run / "front.csv")
    sets = [params_of(ch["params"])] + [params_of(r) for _, r in fr.head(args.sets).iterrows()]
    sets = list(dict.fromkeys(sets))
    rows, lags = [], {}
    t0 = time.process_time()
    for p in sets:
        piv = ctx.pivots(p)
        conf = Z.confirmation_bars(ctx.h, ctx.l, piv, p)
        lag = Z.lag_distribution(piv, conf)
        key = f"D{p.depth}_dev{p.deviation_pips}_B{p.backstep}"
        lags[key] = lag
        sig_a = Z.signals_from_pivots(piv)
        sig_b = Z.signals_from_pivots(piv, conf=conf)
        ks = {k: ctx.run(sig_b, piv, "DECISION_2020_2022", mode="causal", k_tp=k)[0] for k in K_TP}
        valid_k = {k: v for k, v in ks.items() if v["valid"]} or ks
        k_best = max(valid_k, key=lambda k: valid_k[k]["sharpe_daily_sqrt365"])
        for period in PERIODS:
            a, _ = ctx.run(sig_a, piv, period)
            b, _ = ctx.run(sig_b, piv, period, mode="causal", k_tp=k_best)
            rows.append({"params": key, "depth": p.depth, "deviation_pips": p.deviation_pips, "backstep": p.backstep,
                         "chosen": p == sets[0], "k_tp": k_best, "period": period, "lag_median": lag["median"], "lag_mean": lag["mean"],
                         "lag_p90": lag["p90"], "pivots": len(piv), "confirmed": lag["n"],
                         **{f"hind_{k}": a[k] for k in KEEP + ("valid", "rule_skipped_orders")},
                         **{f"conf_{k}": b[k] for k in KEEP + ("valid", "rule_skipped_orders")}})
        lags[key]["k_tp_scores_2020_2022"] = {str(k): v["sharpe_daily_sqrt365"] for k, v in ks.items()}
    pd.DataFrame(rows).to_csv(run / "confirmed.csv", index=False)
    (run / "LAGS.json").write_text(json.dumps(lags, indent=1) + "\n")
    print(json.dumps({"sets": len(sets), "cpu": time.process_time() - t0}))


# ------------------------------------------------------------------ timing-noise study
def noise(args):
    run = Path(args.run)
    ctx = Ctx(args.data)
    p = params_of(json.loads((run / "CHOICE.json").read_text())["params"])
    piv = ctx.pivots(p)
    rows = []
    t0 = time.process_time()
    for period in ("VALIDATION_2023", "CONFIRM_2024"):
        for d in DELAYS:
            for J in JITTERS:
                for seed in (SEEDS if J > 0 else SEEDS[:1]):
                    rng = np.random.default_rng([seed, d, J, 2026_10_19])
                    sig = Z.jitter_signals(Z.signals_from_pivots(piv, delay=d), rng, J)
                    err = np.mean([abs(b - i) for (b, _), (i, _, _) in zip(sorted(sig), piv)]) if sig else math.nan
                    m, _ = ctx.run(sig, piv, period)
                    rows.append({"period": period, "delay": d, "jitter": J, "seed": seed, "timing_error_bars": err, **m})
    pd.DataFrame(rows).to_csv(run / "noise.csv", index=False)
    print(json.dumps({"rows": len(rows), "cpu": time.process_time() - t0}))


# ------------------------------------------------------------------ learned causal pivot detector
def causal_features(bars):
    """Features at bar t from bars <= t only (close, high, low of completed bars)."""
    c, h, l = bars.CLOSE, bars.HIGH, bars.LOW
    f = {}
    for n in (1, 3, 6, 12, 24, 48, 120):
        f[f"ret_{n}"] = (c - c.shift(n)) / c.shift(n)
    for n in (24, 120):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"loc_{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
        f[f"dd_{n}"] = (hi - c) / c
        f[f"du_{n}"] = (c - lo) / c
    for n in (6, 24):
        f[f"pk_{n}"] = np.sqrt((np.log(h / l) ** 2).rolling(n).mean() / (4 * math.log(2)))
    f["er_24"] = (c - c.shift(24)).abs() / (c.diff().abs().rolling(24).sum())
    f["hour_s"], f["hour_c"] = np.sin(2 * np.pi * bars.index.hour / 24), np.cos(2 * np.pi * bars.index.hour / 24)
    f["dow"] = bars.index.dayofweek / 6.0
    return pd.DataFrame(f, index=bars.index)


def detector_targets(pivots, n, k):
    """y[t] in {0: none, 1: valley within (t, t+k], 2: peak within (t, t+k]} -- hindsight labels, used ONLY as targets."""
    y = np.zeros(n, int)
    for i, kind, _ in pivots:
        a, b = max(0, i - k), i - 1
        y[a:b + 1] = 1 if kind == Z.VALLEY else 2
    return y


def detector(args):
    import torch
    run = Path(args.run)
    ctx = Ctx(args.data)
    p = params_of(json.loads((run / "CHOICE.json").read_text())["params"])
    piv = ctx.pivots(p)
    F = causal_features(ctx.bars)
    n = len(F)
    Xall = F.to_numpy(float)
    ts = ctx.bars.index
    masks = {name: np.asarray((ts >= per(name)[0]) & (ts < per(name)[1])) for name in PERIODS}
    okx = np.isfinite(Xall).all(axis=1)
    dev = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    t_wall = time.time()
    results, preds = {}, {}
    for k in args.k_bars:
        y = detector_targets(piv, n, k)
        tr = masks["DECISION_2020_2022"] & okx
        mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
        Xz = (Xall - mu) / sd
        torch.manual_seed(args.seed)
        model = torch.nn.Sequential(torch.nn.Linear(Xz.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                    torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xz[tr], dtype=torch.float32, device=dev)
        yt = torch.tensor(y[tr], dtype=torch.long, device=dev)
        w = torch.tensor(1.0 / np.maximum(np.bincount(y[tr], minlength=3), 1), dtype=torch.float32, device=dev)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w / w.sum() * 3)
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 512):
                idx = perm[i:i + 512]
                opt.zero_grad()
                loss_fn(model(Xt[idx]), yt[idx]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            P = torch.softmax(model(torch.tensor(Xz, dtype=torch.float32, device=dev)), dim=1).cpu().numpy()
        P[~okx] = np.nan
        preds[k] = P
        for name in PERIODS:
            m = masks[name] & okx
            from sklearn.metrics import log_loss, roc_auc_score
            base = np.bincount(y[tr], minlength=3) / tr.sum()
            yy = y[m]
            auc_v = roc_auc_score(yy == 1, P[m, 1]) if (yy == 1).any() else math.nan
            auc_p = roc_auc_score(yy == 2, P[m, 2]) if (yy == 2).any() else math.nan
            ll = log_loss(yy, P[m], labels=[0, 1, 2])
            ll_base = log_loss(yy, np.tile(base, (m.sum(), 1)), labels=[0, 1, 2])
            results[(k, name)] = {"auc_valley": auc_v, "auc_peak": auc_p, "log_loss": ll, "log_loss_base_rate": ll_base,
                                  "base_rate_valley": float(base[1]), "base_rate_peak": float(base[2]), "n": int(m.sum())}
    wall = time.time() - t_wall
    peak_vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0
    # threshold and k chosen on 2023 by Sharpe of the reversal strategy (causal TP = k_tp sigma, k_tp from confirmed.csv)
    conf_df = pd.read_csv(run / "confirmed.csv")
    k_tp = float(conf_df[conf_df.chosen].k_tp.iloc[0])
    rows = []
    for k in args.k_bars:
        P = preds[k]
        for tau in args.thresholds:
            sig = []
            last = -10 ** 9
            for t in range(n):
                if not okx[t] or t - last < k:
                    continue
                if P[t, 1] >= tau and P[t, 1] >= P[t, 2]:
                    sig.append((t, 1)),
                    last = t
                elif P[t, 2] >= tau and P[t, 2] > P[t, 1]:
                    sig.append((t, -1))
                    last = t
            for name in PERIODS:
                m, _ = ctx.run(sig, piv, name, mode="causal", k_tp=k_tp)
                mm = masks[name] & okx
                y = detector_targets(piv, n, k)
                sel = [(t, d) for t, d in sig if mm[t]]
                prec = float(np.mean([(y[t] == (1 if d > 0 else 2)) for t, d in sel])) if sel else math.nan
                rows.append({"k_bars": k, "threshold": tau, "period": name, "signals": len(sel), "precision_at_threshold": prec,
                             **results[(k, name)], **m})
    det = pd.DataFrame(rows)
    det.to_csv(run / "detector.csv", index=False)
    v = det[(det.period == "VALIDATION_2023") & det.valid]
    v = v if not v.empty else det[det.period == "VALIDATION_2023"]
    best = v.sort_values("sharpe_daily_sqrt365", ascending=False).iloc[0]
    P = preds[int(best.k_bars)]
    pred_path = run / "detector_predictions.csv"
    pd.DataFrame({"t": ts, "p_none": P[:, 0], "p_valley": P[:, 1], "p_peak": P[:, 2]}).to_csv(pred_path, index=False, float_format="%.6f")
    seal = {"sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(), "k_bars": int(best.k_bars), "threshold": float(best.threshold),
            "trained_on": "DECISION_2020_2022", "chosen_on": "VALIDATION_2023", "confirm": "CONFIRM_2024 once",
            "features": list(F.columns), "feature_source": "SUBSTITUTE_CAUSAL_LAKE_FEATURES_NOT_MUSASHI_MANIFEST",
            "device": str(dev), "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "peak_vram_mib": peak_vram, "wall_seconds": wall, "epochs": args.epochs, "k_bars_grid": args.k_bars,
            "thresholds": args.thresholds, "k_tp": k_tp, "expert_params": json.loads((run / "CHOICE.json").read_text())["params"]}
    (run / "DETECTOR_SEAL.json").write_text(json.dumps(seal, indent=1) + "\n")
    print(json.dumps({"device": str(dev), "wall": wall, "peak_vram_mib": peak_vram, "chosen": [int(best.k_bars), float(best.threshold)],
                      "val_sharpe": float(best.sharpe_daily_sqrt365)}))


# ------------------------------------------------------------------ demonstrations
def demos(args):
    run = Path(args.run)
    ctx = Ctx(args.data)
    ch = json.loads((run / "CHOICE.json").read_text())
    p = params_of(ch["params"])
    piv = ctx.pivots(p)
    sig = Z.signals_from_pivots(piv)
    trades = []
    for period in PERIODS:
        _, res = ctx.run(sig, piv, period)
        trades += res.trades
    man = Z.write_demonstrations(run / "demonstrations", trades, p, ctx.digest,
                                 {"profile": "RealisticProfile defaults (ASSUMED_NOT_BROKER_VERIFIED)", "sizing": "rel_volume 0.10, leverage 100",
                                  "sl": f"entry -/+ {M_SL} x causal 24 h vol", "tp": "next hindsight pivot level -/+ 1 pip"},
                                 extra={"periods": PERIODS, "data_file": "market_data/forex/g10/eurusd/1h.parquet",
                                        "calendar": "config/short_horizon/eurusd_h1_calendar.dev.json"})
    print(json.dumps({"rows": man["rows"], "sha256": man["sha256"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["nsga", "confirmed", "noise", "detector", "demos"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--budget-cpu-seconds", type=float, default=85.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--k-bars", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    a = ap.parse_args()
    {"nsga": nsga, "confirmed": confirmed, "noise": noise, "detector": detector, "demos": demos}[a.stage](a)


if __name__ == "__main__":
    main()
