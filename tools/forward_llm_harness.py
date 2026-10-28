#!/usr/bin/env python3
"""Forward PAPER test of an LLM direction call on EURUSD H1 (design: docs/short_horizon/FORWARD_PAPER_TEST_LLM_DESIGN_2026_10_21.md).

No orders, no broker writes. Every decision is written append-only with hashes BEFORE its outcome can exist; the
outcome is scored later from bars received afterwards. Feed contract (Musashi handoff 2026-10-21): use a snapshot only
when state == READY and it is fresh; decision time = RECEIPT time; never fill at an open already passed at the
decision; the harness enforces the lane's business rules.

  decide   (timer, every few minutes; idempotent) archive new closed bars; if the newest closed bar has no decision yet
           and the business calendar allows an entry: build the fixed prompt, ask each model, append DECISIONS.jsonl
  score    resolve decisions whose barrier outcome is final: fill = open of the first bar whose START >= receipt;
           upper/lower = fill +/- 2 sigma; first touch within 72 bars or the forced flat; SL-first in a tie bar
           -> SCORES.json and SUMMARY.md (log-loss/Brier vs the prior, accuracy among hits, paper P&L in sigma)

Business calendar (declared, America/New_York): the FX week opens Sunday 17:00 and closes Friday 17:00 NY; no entry
in the first 12 h after the open or in the last 12 h before the close; every paper order is flat at close - 12 h.
Models: local ollama models (temperature 0, fixed seed); baselines at the same instants: constant 0.5 and a declared
24-bar momentum rule (p = 0.5 + 0.05 * sign(ret_24)). Everything is DEVELOPMENT evidence, never an order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOUR = 3600
HORIZON, M_SIGMA = 72, 2.0
PROMPT_VERSION = "forward_llm_prompt.v1"
PROMPT = """You are assisting a research study. Answer with ONE number only.
Instrument: EURUSD, hourly bars. Decision time (UTC): {decision_utc}.
Market state from completed bars only:
- returns: 1h {r1:+.4%}, 4h {r4:+.4%}, 24h {r24:+.4%}, 72h {r72:+.4%}
- realized volatility 24h: {rv24:.4%}; typical 24h volatility (sigma): {sigma:.4%}
- 24h range: {range24:.4%}; close position in the 24h range (0 = low, 1 = high): {loc24:.2f}
- trend estimates (Kalman slope, units of sigma per hour): fast {kf:+.3f}, medium {km:+.3f}, slow {ks:+.3f}
- hours since / to the London 16:00 fix: {lsince:.1f} / {lto:.1f}; to the New York 10:00 cut: {nyto:.1f}
Scheduled economic releases in the next 24 hours (UTC): {events}
Question: a long order is filled at the next hourly open with take-profit +2 sigma and stop-loss -2 sigma. What is the
probability that the take-profit is reached before the stop-loss within 72 hours? Reply with a probability between 0 and 1."""


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def now() -> float:
    return time.time()


# ----------------------------------------------------------------------------- calendar
def week_bounds(ts_utc: pd.Timestamp):
    """(open, close) UTC of the FX week containing ts: Sunday 17:00 -> Friday 17:00 America/New_York."""
    ny = ts_utc.tz_convert("America/New_York")
    sunday = (ny - pd.Timedelta(days=(ny.weekday() + 1) % 7)).normalize() + pd.Timedelta(hours=17)
    if ny < sunday:
        sunday -= pd.Timedelta(days=7)
    sunday = pd.Timestamp(sunday.tz_localize(None)).tz_localize("America/New_York")
    friday = pd.Timestamp((sunday.tz_localize(None) + pd.Timedelta(days=5))).tz_localize("America/New_York")
    return sunday.tz_convert("UTC"), friday.tz_convert("UTC")


def entry_allowed(decision_utc: pd.Timestamp) -> tuple[bool, pd.Timestamp]:
    o, c = week_bounds(decision_utc)
    ok = (decision_utc >= o + pd.Timedelta(hours=12)) and (decision_utc <= c - pd.Timedelta(hours=12)) and decision_utc < c
    return bool(ok), c - pd.Timedelta(hours=12)


# ----------------------------------------------------------------------------- bars and state
def load_archive(state: Path) -> pd.DataFrame:
    p = state / "BARS.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["bar_start_epoch", "open", "high", "low", "close", "first_received_epoch"])


def archive_bars(state: Path, snap: dict) -> pd.DataFrame:
    """Append bars not yet archived (keyed by bar start); the FIRST receipt time of each bar is kept."""
    arc = load_archive(state)
    new = pd.DataFrame(snap["bars"])
    new = new[["bar_start_epoch", "open", "high", "low", "close"]].assign(first_received_epoch=snap["received_at_epoch"])
    have = set(arc["bar_start_epoch"].astype(int)) if len(arc) else set()
    add = new[~new["bar_start_epoch"].astype(int).isin(have)]
    if len(add):
        arc = pd.concat([arc, add], ignore_index=True).sort_values("bar_start_epoch").reset_index(drop=True)
        arc.to_parquet(state / "BARS.parquet", index=False)
    return arc


def kalman_slope(y: np.ndarray, q: float) -> float:
    x, P = np.array([y[0], 0.0]), np.eye(2)
    F, Q, H = np.array([[1.0, 1.0], [0.0, 1.0]]), np.diag([q, q * q]), np.array([1.0, 0.0])
    for v in y[1:]:
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H + 1.0
        K = P @ H / S
        x = x + K * (v - H @ x)
        P = P - np.outer(K, H) @ P
    return float(x[1])


def market_state(bars: pd.DataFrame, decision_utc: pd.Timestamp) -> dict:
    c = bars["close"].to_numpy(float)
    lc = np.log(c)
    r = lambda k: float(lc[-1] - lc[-1 - k]) if len(lc) > k else float("nan")  # noqa: E731
    r1 = np.diff(lc)
    rv24 = float(np.sqrt(np.sum(r1[-24:] ** 2))) if len(r1) >= 24 else float("nan")
    win = r1[-480:]
    days = len(win) // 24
    sigma = float(np.mean([np.sqrt(np.sum(win[i * 24:(i + 1) * 24] ** 2)) for i in range(days)])) if days >= 3 else float("nan")
    h24, l24 = float(bars["high"].iloc[-24:].max()), float(bars["low"].iloc[-24:].min())
    scale = np.std(r1[-480:]) or 1.0
    ks = {name: kalman_slope(lc[-240:] / scale, q) for name, q in (("kf", 1e-1), ("km", 1e-2), ("ks", 1e-3))}
    from tools.timing_event_clock_local import event_clock_at                         # vendored copy (see below)
    evc = event_clock_at(decision_utc)
    return {"r1": r(1), "r4": r(4), "r24": r(24), "r72": r(72), "rv24": rv24, "sigma": sigma,
            "sigma_days_used": days, "range24": float(np.log(h24 / l24)), "loc24": float((c[-1] - l24) / (h24 - l24)) if h24 > l24 else 0.5,
            **ks, **evc}


def upcoming_events(calendar_path: str, decision_utc: pd.Timestamp) -> str:
    try:
        cal = pd.read_parquet(calendar_path)
    except Exception:
        return "calendar unavailable"
    t = pd.to_datetime(cal["announcement_datetime_utc"], utc=True)
    m = (t > decision_utc) & (t <= decision_utc + pd.Timedelta(hours=24)) & cal["currency"].str.upper().isin(["USD", "EUR"])
    rows = cal[m].assign(t=t[m]).sort_values("t")
    if not len(rows):
        return "none"
    return "; ".join(f"{r.currency.upper()} {r.release} at {r.t.strftime('%a %H:%M')}" for r in rows.itertuples())[:600]


def ask_ollama(model: str, prompt: str, host: str) -> tuple[str, str]:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 20261021, "num_predict": 16}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out.get("response", ""), out.get("model", model)


def parse_prob(text: str) -> float | None:
    m = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])", text)
    if m:
        v = float(m.group(1))
        return v if 0 <= v <= 1 else None
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) / 100 if m and float(m.group(1)) <= 100 else None


# ----------------------------------------------------------------------------- verbs
def decide(a):
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    st = json.loads(Path(a.status).read_text())
    if st.get("state") != "READY" or now() - float(st["received_at_epoch"]) > a.max_age:
        print(json.dumps({"skip": "FEED_NOT_READY_OR_STALE", "state": st.get("state")}))
        return
    snap = json.loads(Path(st["snapshot"]).read_text())
    if snap.get("raw_sha256") != st.get("raw_sha256"):
        print(json.dumps({"skip": "SNAPSHOT_STATUS_MISMATCH"}))
        return
    bars = archive_bars(state, snap)
    last = int(bars["bar_start_epoch"].iloc[-1])
    dec_path = state / "DECISIONS.jsonl"
    done = {json.loads(l)["last_bar_start_epoch"] for l in dec_path.read_text().splitlines()} if dec_path.exists() else set()
    if last in done:
        print(json.dumps({"skip": "ALREADY_DECIDED", "last_bar_start_epoch": last}))
        return
    receipt = float(snap["received_at_epoch"])
    dec_utc = pd.Timestamp(receipt, unit="s", tz="UTC")
    ok, flat_at = entry_allowed(dec_utc)
    if len(bars) < 96:
        ok = False
    rec = {"schema": "forward_llm_decision.v1", "last_bar_start_epoch": last, "decision_receipt_epoch": receipt,
           "decision_utc": str(dec_utc), "snapshot_raw_sha256": snap["raw_sha256"], "entry_allowed": ok,
           "forced_flat_utc": str(flat_at), "prompt_version": PROMPT_VERSION, "prompt_template_sha256": sha(PROMPT)}
    if ok:
        msx = market_state(bars, dec_utc)
        events = upcoming_events(a.calendar, dec_utc)
        prompt = PROMPT.format(decision_utc=dec_utc.strftime("%Y-%m-%d %H:%M"), events=events, **msx)
        rec.update(state=msx, events=events, prompt_sha256=sha(prompt), sigma=msx["sigma"], models={})
        for model in a.models:
            try:
                txt, mid = ask_ollama(model, prompt, a.ollama)
                rec["models"][model] = {"raw": txt[:200], "p_up": parse_prob(txt), "model_reported": mid}
            except Exception as exc:
                rec["models"][model] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "p_up": None}
        rec["models"]["baseline_prior"] = {"p_up": 0.5}
        rec["models"]["baseline_momentum24"] = {"p_up": 0.5 + 0.05 * float(np.sign(msx["r24"])) if np.isfinite(msx["r24"]) else 0.5}
    rec["written_epoch"] = now()
    with open(dec_path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps({"decided": last, "entry_allowed": ok, "models": {k: v.get("p_up") for k, v in rec.get("models", {}).items()}}))


def outcome(bars: pd.DataFrame, receipt: float, sigma: float, flat_utc: pd.Timestamp):
    """(status, outcome, fill_bar_start, bars_used): fill at the open of the first bar starting at/after receipt."""
    s = bars["bar_start_epoch"].to_numpy(float)
    i = int(np.searchsorted(s, receipt, side="left"))
    if i >= len(s):
        return "PENDING", None, None, 0
    fill = float(bars["open"].iloc[i])
    up, lo = fill * math.exp(M_SIGMA * sigma), fill * math.exp(-M_SIGMA * sigma)
    end_epoch = min(s[i] + HORIZON * HOUR, flat_utc.timestamp())
    for k in range(i, len(s)):
        if s[k] + HOUR > end_epoch:
            return "FINAL", "TIMEOUT", s[i], k - i
        hu, hl = bars["high"].iloc[k] >= up, bars["low"].iloc[k] <= lo
        if hu and hl:
            return "FINAL", "BOTH_SAME_BAR", s[i], k - i + 1
        if hu:
            return "FINAL", "UPPER", s[i], k - i + 1
        if hl:
            return "FINAL", "LOWER", s[i], k - i + 1
    return "PENDING", None, s[i], len(s) - i


def score(a):
    state = Path(a.state)
    bars = load_archive(state)
    decs = [json.loads(l) for l in (state / "DECISIONS.jsonl").read_text().splitlines()] if (state / "DECISIONS.jsonl").exists() else []
    rows = []
    for d in decs:
        if not d.get("entry_allowed") or not d.get("sigma") or not np.isfinite(d["sigma"]):
            continue
        st, oc, fill, used = outcome(bars, d["decision_receipt_epoch"], d["sigma"], pd.Timestamp(d["forced_flat_utc"]))
        for m, v in d["models"].items():
            rows.append({"decision": d["last_bar_start_epoch"], "model": m, "p_up": v.get("p_up"), "status": st, "outcome": oc})
    df = pd.DataFrame(rows)
    out = {"decisions": len(decs), "entry_decisions": int(sum(1 for d in decs if d.get("entry_allowed"))), "by_model": {}}
    if len(df):
        fin = df[(df.status == "FINAL") & df.outcome.isin(["UPPER", "LOWER"]) & df.p_up.notna()]
        for m, g in fin.groupby("model"):
            y = (g.outcome == "UPPER").to_numpy(float)
            p = np.clip(g.p_up.to_numpy(float), 1e-4, 1 - 1e-4)
            out["by_model"][m] = {"resolved_hits": int(len(g)), "up_rate": float(y.mean()),
                                  "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                                  "brier": float(np.mean((p - y) ** 2)), "accuracy": float(np.mean((p > 0.5) == (y == 1))) if (p != 0.5).any() else None}
        out["status_counts"] = df.drop_duplicates("decision").status.value_counts().to_dict()
        out["outcome_counts"] = df.drop_duplicates("decision").outcome.value_counts(dropna=False).astype(int).to_dict()
    out["scored_utc"] = str(pd.Timestamp.now(tz="UTC"))
    (state / "SCORES.json").write_text(json.dumps(out, indent=1, default=str))
    L = ["# Forward paper test (LLM direction call), running summary", "", f"Scored {out['scored_utc']}; decisions {out['decisions']}, "
         f"with entry allowed {out['entry_decisions']}. Power target: ~780 resolved hits (~21 weeks).", "",
         "| model | resolved hits | up rate | log-loss | Brier | accuracy |", "|---|---|---|---|---|---|"]
    for m, v in out["by_model"].items():
        L.append(f"| {m} | {v['resolved_hits']} | {v['up_rate']:.3f} | {v['log_loss']:.4f} | {v['brier']:.4f} | {v['accuracy']} |")
    (state / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: out[k] for k in ("decisions", "entry_decisions")} | {"models": list(out["by_model"])}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="verb", required=True)
    p1 = sp.add_parser("decide")
    p1.add_argument("--status", required=True)
    p1.add_argument("--state", required=True)
    p1.add_argument("--calendar", required=True)
    p1.add_argument("--models", nargs="+", default=["llama3.2:3b"])
    p1.add_argument("--ollama", default="http://127.0.0.1:11434")
    p1.add_argument("--max-age", type=float, default=900)
    p2 = sp.add_parser("score")
    p2.add_argument("--state", required=True)
    a = ap.parse_args()
    {"decide": decide, "score": score}[a.verb](a)


if __name__ == "__main__":
    main()
