#!/usr/bin/env python3
"""Engine check of the declared execution variants (fix lane F3) on admitted EURUSD H1 bars, 2021-01 -> 2021-04
(TRAIN only; never reads 2025): perfect-foresight and coin-flip policies through the same BarEnv the experiments
use, legacy sizing vs risk sizing, first-of-day vs best-of-day entry, reversal waiver, Friday cutoff.

Reads only. Writes JSON + Markdown to --out."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import bar_source as BS  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import exec_variants as V  # noqa: E402
from tools.expert_guided_e1t import MUSASHI_CFG  # noqa: E402

H = 12


def policies(bars, sigma):
    n = len(bars)
    o, c = bars.OPEN.to_numpy(float), bars.CLOSE.to_numpy(float)

    def pf(tp, sl, thr=0.5):
        """perfect foresight of the next-12 h move, scored by its size in sigma (for the best-of-day gate)."""
        def f(bar, r, st, env):
            if int(st[0]) != 0 or bar + H >= n:
                return E.HOLD, tp, sl, -np.inf
            s = sigma[bar + 1]
            mv = c[bar + H] - o[bar + 1]
            if not np.isfinite(s) or abs(mv) < thr * s:
                return E.HOLD, tp, sl, -np.inf
            return (E.LONG if mv > 0 else E.SHORT), tp, sl, abs(mv) / s
        return f

    def coin(tp, sl, p, seed):
        rng = np.random.default_rng(seed)

        def f(bar, r, st, env):
            if int(st[0]) != 0:
                return E.HOLD, tp, sl, -np.inf
            if rng.random() < p:
                return (E.LONG if rng.random() < 0.5 else E.SHORT), tp, sl, rng.random()
            return E.HOLD, tp, sl, -np.inf
        return f
    return pf, coin


def replay(bars, cal, sigma, sigma_rule, span, cfg, act):
    env = E.BarEnv(bars, cal, sigma, sigma_rule, span[0], span[1], cfg)
    for b in range(env.first, env.last):
        st = env.state(b)
        a, tp, sl = act(b, b, st, env)
        env.step(b, a, tp, sl)
    m = env.result()
    tr = pd.DataFrame(env.trades)
    cost = float((tr.spread_usd + tr.slippage_usd + tr.commission_usd).sum()) if len(tr) else 0.0
    return {"trades": int(m["trades"]), "net": round(m["net_profit_usd"], 1), "gross": round(m["gross_usd"], 1),
            "trading_cost": round(cost, 1), "sharpe": round(m["sharpe_daily_sqrt365"], 2), "maxdd_pct": round(m["max_drawdown_pct"], 2),
            "win": round(m["win_rate"], 3), "opw": round(m["orders_per_week"], 2), "refused_cap": m["n_refused_DAILY_CAP"],
            "refused_variant_cutoff": m.get("n_refused_VARIANT_CUTOFF", 0), "reversal_waived": m.get("n_reversal_cap_waived", 0),
            "exits": {"TP": m["n_close_TP"], "SL": m["n_close_SL"], "TIME": m["n_close_FORCED_FLAT"], "AGENT": m["n_close_AGENT_CLOSE"], "REV": m["n_close_REVERSAL"]},
            "median_units": int(tr.units.median()) if len(tr) else 0,
            "entry_hour0_share": round(float(np.mean([t.hour == 22 for t in tr.entry_time])), 3) if len(tr) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2021-04-01")
    a = ap.parse_args()
    bars = BS.load_ohlc(f"{a.admitted}/eurusd.parquet", "2020-01-01", "2022-01-01", source="admitted")
    assert bars.index.max() < pd.Timestamp("2025-01-01", tz="UTC")
    cal = DG.calendar_from_closures(json.load(open(f"{a.admitted}/CLOSURE_INTERVALS.json")))
    sigma, sigma_rule = Z.causal_sigma(bars), Z.causal_range_sigma(bars)
    pf, coin = policies(bars, sigma)
    span = (a.start, a.end)
    variants = {"legacy": None,
                "risk1pct": V.ExecVariant(sizing_mode="risk", risk_pct=0.01),
                "risk1pct+reverse+cutoff24": V.ExecVariant(sizing_mode="risk", risk_pct=0.01, reverse_counts_as_one_entry=True,
                                                            entry_cutoff_hours_before_flat=24.0),
                "risk1pct+ask_triggers": V.ExecVariant(sizing_mode="risk", risk_pct=0.01, short_trigger_on_ask=True)}
    rows = []
    for vname, var in variants.items():
        cfg = dataclasses.replace(MUSASHI_CFG, variant=var)
        for pname, act, gate in (("pf_sign12h_tp1_sl1_first", pf(1.0, 1.0), "first"), ("pf_sign12h_tp1_sl1_best3", pf(1.0, 1.0), "best3"),
                                 ("pf_sign12h_tp0.7_sl2_first", pf(0.7, 2.0), "first")):
            g = V.first_of_day(act) if gate == "first" else V.BestOfDayGate(act, observe_bars=3)
            rows.append({"variant": vname, "identity": var.identity if var else V.LEGACY.identity, "policy": pname, **replay(bars, cal, sigma, sigma_rule, span, cfg, g), "gate": g.stats})
        for s in range(5):
            g = V.first_of_day(coin(1.0, 1.0, 0.05, s))
            rows.append({"variant": vname, "identity": var.identity if var else V.LEGACY.identity, "policy": f"coin_tp1_sl1_seed{s}", **replay(bars, cal, sigma, sigma_rule, span, cfg, g), "gate": g.stats})
        print(vname, "done", flush=True)
    # cap profile for A4-10 (per-bar effective cap in sigma multiples)
    env = E.BarEnv(bars, cal, sigma, sigma_rule, span[0], span[1], MUSASHI_CFG)
    cap = V.cap_profile(env, np.arange(env.first, env.last))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ENGINE_CHECK.json").write_text(json.dumps({"span": span, "contract": "sac (cap 1/day, 1:50)", "rows": rows, "effective_cap_sigma": cap,
                                                       "variants": {k: (v.document() if v else V.LEGACY.document()) for k, v in variants.items()}}, indent=1, default=str))
    md = ["# Engine check of the declared execution variants (admitted 2021-01 -> 2021-04, contract cap 1/day, 1:50)", "",
          "| variant | policy | trades | /wk | net USD | gross | cost | Sharpe | max DD % | win | exits TP/SL/TIME/AGENT/REV | refused cap | cutoff | rev. waived | median units | hour-0 share |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["policy"].startswith("coin") and not r["policy"].endswith("seed0"):
            continue
        ex = r["exits"]
        md.append(f"| {r['variant']} | {r['policy']} | {r['trades']} | {r['opw']} | {r['net']} | {r['gross']} | {r['trading_cost']} | {r['sharpe']} | {r['maxdd_pct']} | {r['win']} | "
                  f"{ex['TP']}/{ex['SL']}/{ex['TIME']}/{ex['AGENT']}/{ex['REV']} | {r['refused_cap']} | {r['refused_variant_cutoff']} | {r['reversal_waived']} | {r['median_units']} | {r['entry_hour0_share']} |")
    coins = {}
    for r in rows:
        if r["policy"].startswith("coin"):
            coins.setdefault(r["variant"], []).append(r["net"])
    md += ["", "Coin flip (5 seeds), mean net USD by variant: " + json.dumps({k: round(float(np.mean(v)), 1) for k, v in coins.items()}),
           "", f"Effective level cap (sigma multiples) over the span: {json.dumps(cap)}"]
    (out / "ENGINE_CHECK.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"coin_mean_net": {k: round(float(np.mean(v)), 1) for k, v in coins.items()}, "cap": cap}))


if __name__ == "__main__":
    main()
