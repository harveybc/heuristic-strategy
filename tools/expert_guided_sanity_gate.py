#!/usr/bin/env python3
"""Sanity gate (owner 2026-10-25: a run with zero trades, one-sided trades or no early close measures nothing; it
must never reach a comparison as a result). Runs after EVERY unit and EVERY ensemble, on the calibration block
(INNER_A) and on the decision block (INNER_B, when the unit has one), for every seed and for the seed ensemble.

A unit is FAILED_SANITY, with its causes, when any seed or its ensemble shows on any audited block:
  zero_trades                trades == 0
  below_2_trades_per_week    orders/week < 2.0 (period average, the owner's activity rule)
  one_side_over_90pct        one side holds > 90 % of the entries
  zero_agent_closes          no voluntary close (AGENT_CLOSE + REVERSAL == 0): early close is indispensable
  single_exit_reason         every exit by one single reason
  sharpe_undefined           NaN / undefined Sharpe
  strategy_share_valid_zero  the strategy search archived no valid point (share_valid == 0 in every search)
  replay_mismatch            a seed whose metrics differ from its replay (ensemble / audit verification)
  unit_failed                the unit produced no E3M.json (the cause is the log's last error line)
A quantity that the unit's artifacts do not record (e.g. the side split of units run before the env counted it) is
listed as UNMEASURED, never assumed to pass or fail. FAILED_SANITY units are excluded from every comparison: the
report tools read ``SANITY.json`` and refuse a verdict that involves one (``stats.sanity_refusal``).

  unit   one unit dir -> <dir>/SANITY.json (exit 3 when FAILED_SANITY and --require-pass)
  all    every unit under --results -> each unit's SANITY.json + SANITY_GATE_ALL.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_OPW = 2.0
MAX_SIDE_SHARE = 0.90
BLOCKS = ("INNER_A", "INNER_B")
EXIT_KEYS = ("n_close_TP", "n_close_SL", "n_close_FORCED_FLAT", "n_close_AGENT_CLOSE", "n_close_REVERSAL")
CAUSES = ("zero_trades", "below_2_trades_per_week", "one_side_over_90pct", "zero_agent_closes", "single_exit_reason", "sharpe_undefined",
          "strategy_share_valid_zero", "replay_mismatch", "unit_failed")


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x


def side_counts(m: dict, balance: dict | None):
    """(long, short) entries: the env's own counters when recorded, else the audit's direction balance, else None."""
    if "n_entries_long" in m and "n_entries_short" in m:
        return int(m["n_entries_long"]), int(m["n_entries_short"]), "env counters"
    if balance and "long" in balance and "short" in balance:
        return int(balance["long"]["entries"]), int(balance["short"]["entries"]), "early-close audit replay"
    return None


def check_window(m: dict, balance: dict | None = None) -> dict:
    """Causes and the measured values of one replay window's metrics (an E3M trading summary or an ensemble window)."""
    causes, unmeasured = [], []
    trades = int(_num(m.get("trades", 0)) or 0)
    opw = _num(m.get("orders_per_week"))
    sh = _num(m.get("sharpe_daily_sqrt365", m.get("sharpe")))
    exits = {k: int(m.get(k, 0)) for k in EXIT_KEYS if k in m}
    agent = int(m.get("n_close_AGENT_CLOSE", 0)) + int(m.get("n_close_REVERSAL", 0))
    if trades == 0:
        causes.append("zero_trades")
    if not math.isfinite(opw) or opw < MIN_OPW:
        causes.append("below_2_trades_per_week")
    if not math.isfinite(sh):
        causes.append("sharpe_undefined")
    if exits:
        if trades > 0 and agent == 0:
            causes.append("zero_agent_closes")
        if trades > 0 and sum(1 for v in exits.values() if v > 0) == 1:
            causes.append("single_exit_reason")
    else:
        unmeasured.append("exit_reasons")
    sc = side_counts(m, balance)
    if sc is None:
        unmeasured.append("side_balance")
        side = None
    else:
        lo, sh_, src = sc
        tot = lo + sh_
        side = {"long": lo, "short": sh_, "source": src, "max_share": (max(lo, sh_) / tot) if tot else float("nan")}
        if tot and max(lo, sh_) / tot > MAX_SIDE_SHARE:
            causes.append("one_side_over_90pct")
    return {"causes": causes, "unmeasured": unmeasured, "trades": trades, "orders_per_week": opw, "sharpe": sh, "agent_closes": agent,
            "exits": exits, "side": side}


def _share_valid(strategy_doc: dict | None):
    """min share_valid over the strategy searches of a frozen strategy doc (None when the arm has no search)."""
    if not isinstance(strategy_doc, dict) or not strategy_doc.get("runs"):
        return None
    vals = [r.get("share_valid") for r in strategy_doc["runs"].values() if isinstance(r, dict) and "share_valid" in r]
    if not vals:
        pts = [p for r in strategy_doc["runs"].values() for p in (r.get("points") or [])]
        return (sum(1 for p in pts if p.get("valid")) / len(pts)) if pts else 0.0
    return max(float(v) for v in vals)


def _audit_balance(audit: dict | None, who: str, block: str):
    if not audit:
        return None
    if who == "ensemble":
        ens = audit.get("ensemble") or {}
        prim = next((k for k, v in ens.items() if isinstance(v, dict) and v.get("primary")), None)
        x = (ens.get(prim) or {}).get(block) if prim else None
    else:
        x = ((audit.get("per_seed") or {}).get(who) or {}).get(block)
    return (x or {}).get("direction_balance")


def last_error(log: Path) -> str:
    if not log.exists():
        return "no E3M.json and no log"
    lines = [x for x in log.read_text(errors="replace").splitlines() if x.strip()]
    for x in reversed(lines):                          # typed refusals / crashes, never the JSON progress lines
        if not x.lstrip().startswith("{") and re.search(r"(SystemExit|Error|INFEASIBLE|NO_SELECTABLE|STOPPED|Killed|Traceback|REFUSED|crispdm-run)", x):
            return x.strip()[:400]
    return "no E3M.json: the unit was stopped or ended before writing its results" + (f" (last log line: {lines[-1].strip()[:160]})" if lines else "")


def gate_unit(unit_dir: Path, ensemble: Path | None = None, audit: Path | None = None, log: Path | None = None) -> dict:
    unit_dir = Path(unit_dir)
    doc = {"unit": unit_dir.name, "rule": "FAILED_SANITY when any seed or the ensemble shows any cause on the calibration or decision block",
           "causes": [], "seeds": {}, "ensemble": None, "unmeasured": [], "blocks": []}
    e3m_p = unit_dir / "E3M.json"
    if not e3m_p.exists():
        lg = log if log is not None else unit_dir.with_suffix(".log")
        doc["causes"].append({"cause": "unit_failed", "where": "unit", "detail": last_error(Path(lg))})
        doc["status"] = "FAILED_SANITY"
        return doc
    e3m = json.loads(e3m_p.read_text())
    au = json.loads(Path(audit).read_text()) if audit is not None and Path(audit).exists() else None
    trials = (e3m.get("trials") or {}).get("FULL") or {}
    blocks = [b for b in BLOCKS if any(b in (t.get("trading") or {}) for t in trials.values())]
    doc["blocks"] = blocks
    for s, t in trials.items():
        rec = {}
        for b in blocks:
            m = (t.get("trading") or {}).get(b)
            if m is None:
                continue
            w = check_window(m, _audit_balance(au, str(s), b))
            rec[b] = w
            doc["causes"] += [{"cause": c, "where": f"seed {s}", "block": b} for c in w["causes"]]
            doc["unmeasured"] += [f"seed {s}/{b}/{u}" for u in w["unmeasured"]]
        sv = _share_valid((t.get("action_contract") or {}).get("strategy"))
        rec["strategy_share_valid"] = sv
        if sv is not None and sv == 0.0:
            doc["causes"].append({"cause": "strategy_share_valid_zero", "where": f"seed {s}", "block": "INNER_A"})
        doc["seeds"][str(s)] = rec
    ens_p = Path(ensemble) if ensemble is not None else unit_dir / "ENSEMBLE.json"
    if ens_p.exists():
        en = json.loads(ens_p.read_text())
        er = {}
        for b in blocks:
            w_ = (en.get("ensemble") or {}).get(b)
            if not w_:
                continue
            w = check_window(w_["metrics"], _audit_balance(au, "ensemble", b))
            er[b] = w
            doc["causes"] += [{"cause": c, "where": "ensemble", "block": b} for c in w["causes"]]
            doc["unmeasured"] += [f"ensemble/{b}/{u}" for u in w["unmeasured"]]
        sv = _share_valid(en.get("ensemble_strategy"))
        er["strategy_share_valid"] = sv
        if sv is not None and sv == 0.0:
            doc["causes"].append({"cause": "strategy_share_valid_zero", "where": "ensemble", "block": "INNER_A"})
        ver = en.get("verification") or {}
        bad = [k for k, v in (ver.get("per_seed_replay_equals_run") or {}).items() if v is False]
        if bad:
            doc["causes"].append({"cause": "replay_mismatch", "where": "ensemble verification", "detail": bad[:20]})
        doc["ensemble"] = er
    else:
        doc["unmeasured"].append("ensemble (no ENSEMBLE.json yet)")
    if au is not None:
        bad = [k for k, v in (au.get("verification") or {}).items() if isinstance(v, dict) and v.get("reproduces_run") is False]
        if bad:
            doc["causes"].append({"cause": "replay_mismatch", "where": "early-close audit verification", "detail": bad[:20]})
    doc["status"] = "FAILED_SANITY" if doc["causes"] else "PASS"
    doc["cause_counts"] = {c: sum(1 for x in doc["causes"] if x["cause"] == c) for c in CAUSES if any(x["cause"] == c for x in doc["causes"])}
    return doc


def status_of(results: Path, unit: str) -> dict:
    """{"sanity": PASS | FAILED_SANITY | NOT_GATED, "sanity_causes": ...} of a unit (read by the report tools)."""
    p = Path(results) / unit / "SANITY.json"
    if not p.exists():
        return {"sanity": "NOT_GATED", "sanity_causes": None}
    d = json.loads(p.read_text())
    return {"sanity": d["status"], "sanity_causes": d.get("cause_counts") or [c["cause"] for c in d.get("causes", [])]}


def sanity_table_md(docs: list, title: str = "Sanity gate") -> list:
    L = [f"## {title}", "", "| unit | status | causes (count) | unmeasured |", "| --- | --- | --- | --- |"]
    for d in docs:
        cc = d.get("cause_counts") or {}
        if not cc and d.get("causes"):
            cc = {d["causes"][0]["cause"]: len(d["causes"])}
        det = "; ".join(f"{k} ({v})" for k, v in cc.items()) or "—"
        if any(c["cause"] == "unit_failed" for c in d.get("causes", [])):
            det += " — " + next(c["detail"] for c in d["causes"] if c["cause"] == "unit_failed")
        un = sorted({u.split("/")[-1] for u in d.get("unmeasured", [])})
        L.append(f"| {d['unit']} | **{d['status']}** | {det} | {', '.join(un) or '—'} |")
    L.append("")
    return L


def cmd_unit(a):
    d = gate_unit(Path(a.unit_dir), Path(a.ensemble) if a.ensemble else None, Path(a.audit) if a.audit else None, Path(a.log) if a.log else None)
    if getattr(a, "superseded", ""):
        d["label"] = a.superseded
    out = Path(a.out) if a.out else Path(a.unit_dir) / "SANITY.json"
    if Path(a.unit_dir).exists():
        out.write_text(json.dumps(d, indent=1, default=float))
    print(json.dumps({"unit": d["unit"], "status": d["status"], "causes": d.get("cause_counts") or [c["cause"] for c in d["causes"]]}))
    if a.require_pass and d["status"] != "PASS":
        sys.exit(3)


def cmd_all(a):
    res = Path(a.results)
    audits = {}
    for g in a.audit_glob or []:
        for p in glob.glob(g):
            nm = Path(p).stem.replace("EARLY_CLOSE_", "")
            audits[nm] = Path(p)
    labels = dict(x.split("=", 1) for x in (a.label or []))
    docs = []
    for d in sorted(res.iterdir()):
        if not d.is_dir() or d.name.startswith("ens_pool_"):
            continue
        if not (d / "E3M.json").exists() and not d.with_suffix(".log").exists():
            continue
        if d.name.endswith("_seeds2"):
            continue                                       # seed extensions are gated with their parent's ensemble
        doc = gate_unit(d, None, audits.get(d.name), d.with_suffix(".log"))
        for pref, lab in labels.items():
            if d.name.startswith(pref):
                doc["label"] = lab
        (d / "SANITY.json").write_text(json.dumps(doc, indent=1, default=float))
        docs.append(doc)
    summary = {"rule": docs[0]["rule"] if docs else None, "units": len(docs), "failed": sum(1 for d in docs if d["status"] != "PASS"),
               "passed": sum(1 for d in docs if d["status"] == "PASS"), "docs": docs}
    Path(a.out_json).write_text(json.dumps(summary, indent=1, default=float))
    md = [f"# Sanity gate on every unit ({a.title})", "",
          "Owner 2026-10-25: a run with zero trades, one-sided trades or no early close measures nothing. A unit is FAILED_SANITY "
          "when any seed or its ensemble shows, on the calibration block (INNER_A) or the decision block (INNER_B): zero trades, "
          "< 2.0 trades/week, one side > 90 % of entries, zero agent closes, all exits by one reason, undefined Sharpe, a strategy "
          "search with no valid point, or a replay that differs from the run. FAILED_SANITY units are excluded from every comparison. "
          "UNMEASURED = the unit's artifacts do not record that quantity (never assumed).", "",
          f"Units: {summary['units']}; PASS {summary['passed']}; FAILED_SANITY {summary['failed']}.", ""]
    md += sanity_table_md(docs, "Per unit")
    lab = [d for d in docs if d.get("label")]
    if lab:
        md += ["## Labels", ""] + [f"- {d['unit']}: {d['label']}" for d in lab] + [""]
    Path(a.out_md).write_text("\n".join(md) + "\n")
    print(json.dumps({"units": summary["units"], "passed": summary["passed"], "failed": summary["failed"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unit")
    u.add_argument("--unit-dir", required=True)
    u.add_argument("--ensemble", default="")
    u.add_argument("--audit", default="")
    u.add_argument("--log", default="")
    u.add_argument("--out", default="")
    u.add_argument("--superseded", default="")
    u.add_argument("--require-pass", action="store_true")
    al = sub.add_parser("all")
    al.add_argument("--results", required=True)
    al.add_argument("--audit-glob", action="append")
    al.add_argument("--label", action="append", help="UNIT_PREFIX=LABEL (e.g. WF_V1_=SUPERSEDED_INFEASIBLE_STRATEGY_SPACE)")
    al.add_argument("--out-json", required=True)
    al.add_argument("--out-md", required=True)
    al.add_argument("--title", default="retroactive, 2026-10-25")
    a = ap.parse_args()
    {"unit": cmd_unit, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
