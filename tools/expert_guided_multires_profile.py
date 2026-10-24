#!/usr/bin/env python3
"""Multi-resolution ZigZag demonstrators: how many examples each agent gets per resolution, how good those trades
are, and how the resolutions agree (owner 2026-10-24: the optimum leaves too few examples for open-buy / open-sell /
close-buy / close-sell and above all for NOP; several decreasing resolutions below the optimum give varied examples
of different performance for every model). Labels only, TRAIN only, nothing fitted, nothing launched.

Per resolution (a MetaTrader depth 'N' or a reversal threshold 'revP' pips; '3' = the registered optimum):
  demonstrations  hindsight replay under the replay contract (1 entry/day, 1:50): legs/week, orders/week, win rate,
                  validity (admission), and the causal-confirmed counterpart (imitability information)
  legs            size of every leg in pips and in causal sigma at the leg start, net of the declared cost (0.06 sigma):
                  percentiles and the share of legs above the minimum TP (0.25 sigma) and above 0.5 sigma
  examples        on the business-eligible flat origins of TRAIN whose label support is final inside TRAIN:
                  open-long / open-short / none per resolution; closers: positioned rows visited by the resolution's own
                  demonstration and their exit labels (CLOSE / HOLD)
  consensus       per origin, how many resolutions call LONG / SHORT / none; NOP candidates = origins where EVERY
                  resolution calls none; intermediate trades = labels present only below the optimum; agreement of each
                  resolution with the optimum
Writes MULTIRES_PROFILE.json and MULTIRES_PROFILE.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import oracle as OR  # noqa: E402
from tools.expert_guided_e1t import Ctx, summary  # noqa: E402
from tools.expert_guided_e3m import ExpertTrack, collect, exit_label, parse_track  # noqa: E402

COST_SIGMA = 0.06
K = ("net_profit_usd", "sharpe_daily_sqrt365", "trades", "orders_per_week", "max_drawdown_pct", "win_rate", "valid")


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {str(q): float(np.percentile(x, q)) for q in qs} if len(x) else None


def leg_quality(t, cx, train_mask_bar):
    """Every leg of the resolution inside TRAIN: pips and net sigma (at the leg's first bar + 1)."""
    piv = t.pivots
    sig = np.asarray(cx.ds.sigma, float)
    pips, net = [], []
    for (b0, k0, v0), (b1, k1, v1) in zip(piv[:-1], piv[1:]):
        if not (train_mask_bar[b0] and train_mask_bar[b1]):
            continue
        s = sig[b0 + 1] if b0 + 1 < len(sig) else np.nan
        if not np.isfinite(s) or s <= 0:
            continue
        gain = abs(v1 - v0)
        pips.append(gain / Z.PIP)
        net.append(gain / s - COST_SIGMA)
    net = np.asarray(net, float)
    return {"legs_in_train": int(len(net)), "pips": pct(pips), "net_sigma": pct(net),
            "share_net_ge_0.25": float(np.mean(net >= 0.25)) if len(net) else None,
            "share_net_ge_0.5": float(np.mean(net >= 0.5)) if len(net) else None,
            "share_net_le_0": float(np.mean(net <= 0)) if len(net) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--specs", default="3,2,rev40,rev30,rev25,rev20,rev15,rev12,rev10,rev8,rev6,rev5")
    ap.add_argument("--contract", default="consolidated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract=a.contract)
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    from tools.expert_guided_e3m import effective_contracts
    SCFG, ECFG = effective_contracts("sac", 2.0)      # the replay contract and the owner's activity rule (>= 2 orders/week)
    v = ds.bars.index.as_unit("ns").asi8
    t0, t1 = (DG._utc_ns(x) for x in ds.meta["periods"]["TRAIN"])
    train_bar = (v >= t0) & (v < t1)
    weeks = (t1 - t0) / (7 * 24 * 3600e9)
    # business-eligible flat origins (entry gate open), rows labelled by the dataset (support inside the period)
    hold = lambda *x: (E.HOLD, 1.5, 2.0)  # noqa: E731
    origins = [vv for vv in collect(cx, "TRAIN", hold, SCFG)[1] if vv[2][-1] > 0.5]
    o_rows = np.array([vv[0] for vv in origins], int)
    o_bars = np.array([vv[1] for vv in origins], int)
    tracks = []
    for spec in a.specs.split(","):
        spec = spec.strip()
        t = ExpertTrack(cx) if spec == str(DG.EXPERT.depth) else parse_track(cx, spec)
        t.spec = spec
        t.pivots = ds.pivots if t.primary else (OR.reversal_pivots(cx.h, cx.l, float(spec[3:]) * Z.PIP)[0] if spec.startswith("rev")
                                                else Z.hindsight_pivots(cx.h, cx.l, Z.Params(depth=int(spec), deviation_pips=DG.EXPERT.deviation_pips, backstep=DG.EXPERT.backstep)))
        tracks.append(t)
    rows, labels = [], {}
    for t in tracks:
        m = summary(cx.rollout("TRAIN", t.policy(cx), ECFG)[0])
        mc = summary(cx.rollout("TRAIN", t.confirmed_policy(), SCFG)[0])
        _, vis = collect(cx, "TRAIN", t.policy(cx), ECFG)
        pos = [x for x in vis if x[6] is not None]
        closes = sum(exit_label(cx, bar, info) for _, bar, _, _, _, _, info in pos)
        y = np.array([t.y[b] if t.valid(b, r) else -9 for b, r in zip(o_bars, o_rows)], int)
        labels[t.spec] = y
        rows.append({"spec": t.spec, "name": t.name, "primary": t.primary, "pivots": int(t.n_pivots),
                     "pivots_in_train": int(sum(1 for b, _, _ in t.pivots if train_bar[b])),
                     "legs_per_week_train": float(sum(1 for b, _, _ in t.pivots if train_bar[b]) / weeks),
                     "demonstration_replay": {k: m[k] for k in K if k in m}, "confirmed_causal": {k: mc[k] for k in K if k in mc},
                     "admitted": bool(m["valid"]), "legs": leg_quality(t, cx, train_bar),
                     "examples": {"origins": int(len(o_bars)), "open_long": int((y == OR.LONG_CLASS).sum()), "open_short": int((y == OR.SHORT_CLASS).sum()),
                                  "none": int((y == 0).sum()), "support_not_final": int((y == -9).sum()),
                                  "closer_rows_visited": len(pos), "closer_CLOSE": int(closes), "closer_HOLD": int(len(pos) - closes)}})
        print(json.dumps({"spec": t.spec, "legs_wk": round(rows[-1]["legs_per_week_train"], 1), "opw": round(m["orders_per_week"], 2), "valid": m["valid"],
                          "open_l": rows[-1]["examples"]["open_long"], "open_s": rows[-1]["examples"]["open_short"], "closers": len(pos),
                          "net_sigma_p50": (rows[-1]["legs"]["net_sigma"] or {}).get("50")}), flush=True)
    # consensus across resolutions on the shared origins
    L = np.stack([labels[t.spec] for t in tracks])                    # (n_res, n_origins)
    valid_all = (L != -9).all(0)
    n_long, n_short = (L == OR.LONG_CLASS).sum(0), (L == OR.SHORT_CLASS).sum(0)
    none_all = valid_all & (n_long == 0) & (n_short == 0)
    conflict = (n_long > 0) & (n_short > 0)
    opt = labels[tracks[0].spec]
    inter = valid_all & (opt == 0) & ((n_long > 0) | (n_short > 0))
    agree = {t.spec: {"same_as_optimum": float(np.mean((labels[t.spec] == opt)[valid_all])),
                      "long_where_optimum_long": float(np.mean((labels[t.spec] == OR.LONG_CLASS)[valid_all & (opt == OR.LONG_CLASS)])) if (valid_all & (opt == OR.LONG_CLASS)).any() else None}
             for t in tracks}
    cons = {"origins": int(len(o_bars)), "valid_in_every_resolution": int(valid_all.sum()),
            "nop_all_resolutions_none": int(none_all.sum()), "nop_share": float(none_all.sum() / max(valid_all.sum(), 1)),
            "any_long": int(((n_long > 0) & valid_all).sum()), "any_short": int(((n_short > 0) & valid_all).sum()),
            "conflict_long_and_short": int((conflict & valid_all).sum()),
            "intermediate_trades_below_optimum": int(inter.sum()),
            "votes_long_hist": {str(k): int(((n_long == k) & valid_all).sum()) for k in range(len(tracks) + 1)},
            "agreement_with_optimum": agree}
    doc = {"rule": "labels only; TRAIN only; replay contract (1 entry/day, 1:50); cost 0.06 sigma declared; nothing fitted",
           "contract": a.contract, "train": ds.meta["periods"]["TRAIN"], "weeks": weeks, "resolutions": rows, "consensus": cons}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MULTIRES_PROFILE.json").write_text(json.dumps(doc, indent=1))
    md = ["# Multi-resolution ZigZag demonstrators (TRAIN, replay contract)", "",
          "| res | pivots TRAIN | legs/wk | orders/wk | valid | win % | leg p50 pips | net σ p50 | ≥0.25σ | ≤0 | open L | open S | none | closer rows | CLOSE | = optimum |",
          "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        d, lg, ex = r["demonstration_replay"], r["legs"], r["examples"]
        md.append(f"| {r['spec']} | {r['pivots_in_train']} | {r['legs_per_week_train']:.1f} | {d['orders_per_week']:.2f} | {d['valid']} | {100 * d['win_rate']:.0f} | "
                  f"{(lg['pips'] or {}).get('50', float('nan')):.0f} | {(lg['net_sigma'] or {}).get('50', float('nan')):.2f} | {100 * (lg['share_net_ge_0.25'] or 0):.0f} % | "
                  f"{100 * (lg['share_net_le_0'] or 0):.0f} % | {ex['open_long']} | {ex['open_short']} | {ex['none']} | {ex['closer_rows_visited']} | {ex['closer_CLOSE']} | "
                  f"{100 * agree[r['spec']]['same_as_optimum']:.0f} % |")
    md += ["", f"Origins {cons['origins']}; valid in every resolution {cons['valid_in_every_resolution']}; NOP (all none) {cons['nop_all_resolutions_none']} "
           f"({100 * cons['nop_share']:.0f} %); any long {cons['any_long']}; any short {cons['any_short']}; long AND short {cons['conflict_long_and_short']}; "
           f"intermediate trades below the optimum {cons['intermediate_trades_below_optimum']}.", "",
           "Votes LONG histogram: " + json.dumps(cons["votes_long_hist"])]
    (out / "MULTIRES_PROFILE.md").write_text("\n".join(md) + "\n")
    print(json.dumps(cons)[:600])


if __name__ == "__main__":
    main()
