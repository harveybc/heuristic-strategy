#!/usr/bin/env python3
"""Calendar / implied-vol range experiment (owner, 2026-10-19). DEVELOPMENT.

Question: can a range forecast for business-day blocks (6 h secondary) get from HAR (~0.95x MEAN20)
down to <= ~0.85x? Inclusion and hyperparameters are decided on 2023 (fit 2020-2022, purged by one
block); 2024 is used once, for confirmation, through the sealed adapter. TEST closed.

Stages (each bounded; run under crispdm-run):
  inventory  data inventory read-only (calendar, implied vol), coverage, fetch commands (not run)
  predict    --B --model {HAR,HARX,GBM} [--cfg i] --period {2023,2024}: weekly refits -> sealed CSVs
  choose     GBM capacity from the 2023 predictions only -> CHOICE.json (digest), before any 2024 GBM run
  score      every sealed prediction file through the adapter (RF-M coin / logistic direction, 1:100 / 1:50 / 1:10)
  events     descriptive event/no-event range-error split on the calendar fixture period (2020-01..2021-04)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.short_horizon import adapter as AD
from app.short_horizon import blocks as BL
from app.short_horizon import execution as X
from app.short_horizon import noise as N
from app.short_horizon import predictors as PR
from app.short_horizon import vol_features as VF
from tools.short_horizon_oracle import load_inputs, utc

LABEL = "DEVELOPMENT_CALENDAR_RANGE_EXPERIMENT"
PERIODS = {"2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01")}
GBM_GRID = ({"max_depth": 2, "max_iter": 100}, {"max_depth": 2, "max_iter": 300},
            {"max_depth": 3, "max_iter": 100}, {"max_depth": 3, "max_iter": 300})
MIN_ROWS = 200


def gbm_tag(cfg):
    return f"GBM_d{cfg['max_depth']}_n{cfg['max_iter']}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table_and_features(bars, cal, B):
    blocks = BL.build_blocks(bars, cal, B)
    t = PR.block_table(blocks, cal, B)
    F = VF.block_features(bars, blocks, t, B)
    return blocks, t, F


def predict(args):
    bars, cal, digest = load_inputs(args.data)
    B = args.block_hours
    blocks, t, F = table_and_features(bars, cal, B)
    a, b = (utc(x) for x in PERIODS[args.period])
    out = Path(args.output) / f"B{B}" / args.period
    out.mkdir(parents=True, exist_ok=True)
    emb = pd.Timedelta(hours=B)
    t0 = time.process_time()
    if args.model == "HAR":
        u, d = PR.har_predict(t, B, "U", "weekly", MIN_ROWS), PR.har_predict(t, B, "D", "weekly", MIN_ROWS)
        tag = "HAR"
    elif args.model == "HARX":
        u = PR.linear_predict(t, F, "U", "weekly", MIN_ROWS, predict_from=a)
        d = PR.linear_predict(t, F, "D", "weekly", MIN_ROWS, predict_from=a)
        tag = "HARX"
    else:
        if args.period == "2024":
            choice = json.loads((Path(args.output) / f"B{B}" / "CHOICE.json").read_text())
            cfg = choice["chosen"]
        else:
            cfg = GBM_GRID[args.cfg]
        u = PR.gbm_predict(t, F, "U", "weekly", cfg, MIN_ROWS, predict_from=a, embargo=emb)
        d = PR.gbm_predict(t, F, "D", "weekly", cfg, MIN_ROWS, predict_from=a, embargo=emb)
        tag = gbm_tag(cfg)
    p_logit = PR.logit_predict(t, B, "weekly", MIN_ROWS)
    coin = N.ambiguous_coin(len(blocks))[t["k"].to_numpy()]
    sel = ((t["start"] >= a) & (t["start"] < b)).to_numpy() & np.isfinite(u) & np.isfinite(d)
    for direction, p in (("coin", (coin == BL.LOW_FIRST).astype(float)), ("logit", p_logit)):
        ok = sel & np.isfinite(p)
        frame = pd.DataFrame({"block_start": [s.isoformat() for s in t["start"][ok]],
                              "made_at": [s.isoformat() for s in t["start"][ok]],
                              "u_hat": u[ok], "d_hat": d[ok], "p_low_first": p[ok]})
        f = out / f"{tag}__{direction}.csv"
        frame.to_csv(f, index=False, float_format="%.10g")
        AD.seal(f, out / f"{tag}__{direction}_seal")
    print(json.dumps({"B": B, "period": args.period, "model": tag, "rows": int(sel.sum()),
                      "cpu": round(time.process_time() - t0, 1), "data_sha256": digest}))


def choose(args):
    """GBM capacity from 2023 only: lowest mean |U err| + |D err| on 2023 blocks."""
    bars, cal, _ = load_inputs(args.data)
    B = args.block_hours
    blocks, t, _ = table_and_features(bars, cal, B)
    truth = t.set_index(t["start"].map(lambda s: s.isoformat()))[["U", "D"]]
    d23 = Path(args.output) / f"B{B}" / "2023"
    scores = {}
    for f in sorted(d23.glob("GBM_*__coin.csv")):
        p = pd.read_csv(f).set_index("block_start")
        j = truth.loc[p.index]
        scores[f.stem.split("__")[0]] = float(np.mean(np.abs(p.u_hat - j.U) + np.abs(p.d_hat - j.D)))
    best_tag = min(scores, key=scores.get)
    cfg = next(c for c in GBM_GRID if gbm_tag(c) == best_tag)
    doc = {"block_hours": B, "decided_on": "2023 only (fit 2020-2022, weekly purged refits)", "scores_2023": scores,
           "chosen": cfg, "chosen_tag": best_tag, "decided_at": pd.Timestamp.now(tz="UTC").isoformat(),
           "inputs_sha256": {f.name: sha(f) for f in sorted(d23.glob("GBM_*__coin.csv"))}}
    path = Path(args.output) / f"B{B}" / "CHOICE.json"
    if path.exists():
        raise SystemExit("CHOICE.json exists: the decision is recorded once")
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))


def score(args):
    bars, cal, digest = load_inputs(args.data)
    rows = []
    for B in (24, 6):
        blocks = BL.build_blocks(bars, cal, B)
        m = AD.Market(bars, cal, blocks, B)
        for period, (a, b) in PERIODS.items():
            d = Path(args.output) / f"B{B}" / period
            if not d.exists():
                continue
            split = (utc(a), utc(b))
            for f in sorted(d.glob("*.csv")):
                seal = json.loads((d / f"{f.stem}_seal" / "SEAL.json").read_text())
                AD.verify_seal(f, seal)
                ps = AD.load_predictions(f, m, split)
                model, direction = f.stem.split("__")
                for lev in (100.0, 50.0, 10.0):
                    r = AD.score(ps, m, split, "RF-M", 1.0, sizing=X.Sizing(leverage=lev))
                    r.update({"block_hours": B, "period": period, "model": model, "direction": direction,
                              "leverage_scenario": lev, "label": LABEL})
                    rows.append(r)
    out = Path(args.output) / "scores.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({"rows": len(rows), "data_sha256": digest}))


# ------------------------------------------------------------------ inventory and event diagnostic
def load_fixture(path):
    """feature-eng test fixture 'economic_calendar_2011_2021.csv' (no header): date, time, country,
    volatility label, event, ... Times are taken as given (time zone not documented: ASSUMED)."""
    df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4], names=["date", "time", "country", "importance", "event"],
                     dtype=str)
    for c in ("country", "importance", "event"):
        df[c] = df[c].str.strip()
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M:%S", utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    if (df["ts"] >= utc("2025-01-01")).any():
        raise SystemExit("fixture reaches TEST rows")
    return df


CURRENCY = {"United States": "USD", "European Monetary Union": "EUR", "Euro Zone": "EUR", "Germany": "EUR",
            "France": "EUR", "Italy": "EUR", "Spain": "EUR", "United Kingdom": "GBP", "Japan": "JPY",
            "Switzerland": "CHF"}


def inventory(args):
    fd = Path(args.financial_data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(Path(p).relative_to(fd))  # noqa: E731
    cal = pd.read_parquet(fd / "economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet")
    ann = pd.read_parquet(fd / "economic_calendar/release_actuals/fxmacrodata/announcements.parquet")
    proxy = pd.read_parquet(fd / "economic_calendar/scheduled_events/fred_release_date_proxy/scheduled_events.parquet")
    a20 = ann[(ann.announcement_datetime_utc >= utc("2020-01-01")) & (ann.announcement_datetime_utc < utc("2025-01-01"))]
    fx = load_fixture(args.fixture)
    fx["currency"] = fx["country"].map(CURRENCY)
    fx20 = fx[(fx.ts >= utc("2020-01-01")) & (fx.ts < utc("2025-01-01"))]
    hi = fx20[fx20.importance.str.startswith("High")]
    by_month = hi.groupby([hi.ts.dt.strftime("%Y-%m"), "currency"]).size().unstack(fill_value=0)
    by_family = hi[hi.currency.isin(["USD", "EUR"])].groupby(["currency", "event"]).size().sort_values(ascending=False)
    doc = {
        "lake_objects_checked": {
            rel(fd / "economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet"): {
                "rows": int(len(cal)), "first": str(cal.announcement_datetime_utc.min()),
                "last": str(cal.announcement_datetime_utc.max()), "covers_2020_2024": False,
                "note": "upcoming-release calendar acquired 2026-05-01; forward only"},
            rel(fd / "economic_calendar/release_actuals/fxmacrodata/announcements.parquet"): {
                "rows": int(len(ann)), "first": str(ann.announcement_datetime_utc.min()),
                "last": str(ann.announcement_datetime_utc.max()), "rows_2020_2024": int(len(a20)),
                "currencies_2020_2024": sorted(a20.currency.unique().tolist()), "covers_2020_2024": False},
            rel(fd / "economic_calendar/scheduled_events/fred_release_date_proxy/scheduled_events.parquet"): {
                "rows": int(len(proxy)), "date_semantics": "FRED observation (reference-period) date, NOT the release instant",
                "usable_as_scheduled_instant": False},
            "derivatives/implied_volatility_surfaces": {"files": 0, "covers": False},
            "macro_economic/fred/stress/vixcls": {"note": "S&P 500 implied vol, not EUR/USD"},
        },
        "implied_vol_eurusd": {"status": "MISSING_OBJECT", "series": "FRED EVZCLS (CBOE EuroCurrency ETF Volatility)",
                               "fetch_command_not_run": "add 'EVZCLS' to FRED_SERIES['stress'] in financial-data "
                               "_scripts/workers/stage13 FRED follow-up worker (stage13_*_followup_worker.py), then run it with the governed FRED_API_KEY "
                               "(the worker also refreshes other sources; a dedicated single-series call is preferable)"},
        "historical_schedule": {"status": "MISSING_OBJECT",
                                "needed": "scheduled release instants 2020-2024 for USD/EUR (and GBP/JPY/CHF) with event family "
                                          "and a schedule vintage or a SCHEDULE_ASSUMED_KNOWN_IN_ADVANCE label",
                                "fetch_options_not_run": [
                                    "financial-data _scripts/workers/stage15_fxmacrodata_worker.py --currencies USD EUR "
                                    "(provider history observed to start 2024-12; may not cover 2020-2023)",
                                    "FRED release dates (fred/release/dates, realtime) for Employment Situation, CPI, GDP, "
                                    "Retail Sales: no existing adapter function; a small worker extension is needed",
                                    "central-bank meeting calendars (FOMC, ECB) published in advance: no adapter in the lake"]},
        "calendar_fixture_outside_lake": {
            "path": "feature-eng tests/data/economic_calendar_2011_2021.csv (test fixture, provenance undocumented)",
            "first": str(fx.ts.min()), "last": str(fx.ts.max()),
            "fields_used": ["scheduled date+time", "country -> currency", "event name", "importance label"],
            "fields_not_used": ["actual", "forecast/consensus", "previous"],
            "time_zone": "not documented: taken as given (TZ_ASSUMED)",
            "importance_flag": "label text 'High/Moderate/Low Volatility Expected'; whether it was assigned before release "
                               "is undocumented, so it may be retrospective (STATIC_UNVERIFIED)",
            "schedule_vintage": "SCHEDULE_ASSUMED_KNOWN_IN_ADVANCE",
            "high_impact_rows_2020_2024": int(len(hi)),
            "usable_for_2023_decision": False, "usable_for_2024_confirmation": False},
    }
    (out / "INVENTORY.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    by_month.to_csv(out / "fixture_high_impact_by_month.csv")
    by_family.head(40).to_csv(out / "fixture_high_impact_by_family_usd_eur.csv")
    print(json.dumps({"fixture_last": str(fx.ts.max()), "high_impact_2020_2024": int(len(hi))}))


def events(args):
    """Descriptive only: HAR range error on blocks with vs without high-impact USD/EUR events in (t0, t0+B]
    over the fixture period 2020-01-01 .. 2021-04-26 (TRAIN data; no decision is taken from it)."""
    bars, cal, _ = load_inputs(args.data)
    fx = load_fixture(args.fixture)
    fx["currency"] = fx["country"].map(CURRENCY)
    hi = fx[fx.importance.str.startswith("High") & fx.currency.isin(["USD", "EUR"])]
    end = hi.ts.max()
    rows = []
    for B in (24, 6):
        blocks = BL.build_blocks(bars, cal, B)
        t = PR.block_table(blocks, cal, B)
        u, d = PR.har_predict(t, B, "U", "weekly", MIN_ROWS), PR.har_predict(t, B, "D", "weekly", MIN_ROWS)
        nU, nD, _ = PR.mean_window_predict(t["U"], 20), PR.mean_window_predict(t["D"], 20), None
        ev = np.zeros(len(t), bool)
        ts = np.sort(np.array([x.value for x in hi.ts], dtype=np.int64))
        for i, k in enumerate(t["k"].to_numpy()):
            s, e = blocks.start[k].value, blocks.end[k].value      # event inside (t0, t0 + B]
            j = np.searchsorted(ts, s, side="right")
            ev[i] = j < len(ts) and ts[j] <= e
        win = ((t["start"] >= utc("2020-01-01")) & (t["start"] < end)).to_numpy() & np.isfinite(u) & np.isfinite(nU)
        R = (t.U + t.D).to_numpy()
        for name, mask in (("event", win & ev), ("no_event", win & ~ev), ("all", win)):
            if not mask.any():
                continue
            rows.append({"block_hours": B, "blocks": name, "n": int(mask.sum()),
                         "mean_range_pips": float(R[mask].mean() / 1e-4),
                         "har_mae_r_ratio_mean20": float(np.mean(np.abs((u + d)[mask] - R[mask]))
                                                         / np.mean(np.abs((nU + nD)[mask] - R[mask])))})
    out = Path(args.output) / "event_diagnostic.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inventory", "predict", "choose", "score", "events"])
    ap.add_argument("--data", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--block-hours", type=int, choices=(6, 24))
    ap.add_argument("--model", choices=["HAR", "HARX", "GBM"])
    ap.add_argument("--cfg", type=int, default=0)
    ap.add_argument("--period", choices=list(PERIODS))
    ap.add_argument("--financial-data", type=Path)
    ap.add_argument("--fixture", type=Path)
    a = ap.parse_args()
    {"inventory": inventory, "predict": predict, "choose": choose, "score": score, "events": events}[a.stage](a)


if __name__ == "__main__":
    main()
