#!/usr/bin/env python3
"""The v13 chain's FIRST-SEED / FIRST-ENSEMBLE stop rule (2026-10-25). ``seed``: wait for the first per-seed progress
line of a unit (the ``seed_sanity`` JSON line E3-M prints after each seed) and check it on INNER_A and INNER_B; exit 0
when it passes, 3 when it fails (zero trades, < 2.0 trades/week, a side above 90 %, no agent close, a single exit
reason), 4 when the unit ended without a seed line (refused or crashed). ``ensemble``: read a unit's SANITY.json and
fail (3) when any cause sits on the ensemble. The verdict line is printed as JSON for the chain's LOG."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

BLOCKS = ("INNER_A", "INNER_B")


def check_block(ss: dict) -> list:
    c = []
    if int(ss.get("trades", 0)) <= 0:
        c.append("zero_trades")
    if float(ss.get("opw", 0.0)) < 2.0:
        c.append("below_2_trades_per_week")
    nl, ns = int(ss.get("long", -1)), int(ss.get("short", -1))
    if nl >= 0 and ns >= 0 and nl + ns > 0 and max(nl, ns) / (nl + ns) > 0.90:
        c.append("one_side_over_90pct")
    if int(ss.get("agent_closes", 0)) <= 0:
        c.append("zero_agent_closes")
    ex = ss.get("exits") or {}
    if ex and sum(1 for v in ex.values() if int(v) > 0) <= 1:
        c.append("single_exit_reason")
    return c


def first_seed_line(log: Path):
    if not log.exists():
        return None
    for ln in log.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("{") and '"seed_sanity"' in ln:
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                continue
    return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_seed(a):
    t0 = time.time()
    while True:
        d = first_seed_line(Path(a.log))
        if d is not None:
            ss = d["seed_sanity"]
            causes = {b: check_block(ss.get(b) or {}) for b in BLOCKS}
            ok = not any(causes.values())
            print(json.dumps({"check": "first_seed", "unit": a.unit, "seed": d.get("seed"), "pass": ok, "causes": causes,
                              "blocks": {b: ss.get(b) for b in BLOCKS}}))
            sys.exit(0 if ok else 3)
        if a.pid and not alive(int(a.pid)):
            d = first_seed_line(Path(a.log))
            if d is None:
                print(json.dumps({"check": "first_seed", "unit": a.unit, "pass": False, "causes": "the unit ended without a seed line (refused or crashed)"}))
                sys.exit(4)
            continue
        if time.time() - t0 > float(a.timeout):
            print(json.dumps({"check": "first_seed", "unit": a.unit, "pass": False, "causes": f"no seed line within {a.timeout} s"}))
            sys.exit(4)
        time.sleep(float(a.poll))


def cmd_ensemble(a):
    p = Path(a.sanity)
    if not p.exists():
        print(json.dumps({"check": "first_ensemble", "unit": a.unit, "pass": False, "causes": "no SANITY.json"}))
        sys.exit(4)
    d = json.loads(p.read_text())
    bad = [c for c in d.get("causes", []) if str(c.get("where", "")).startswith("ensemble")]
    print(json.dumps({"check": "first_ensemble", "unit": a.unit, "pass": not bad, "causes": bad, "ensemble": d.get("ensemble")}))
    sys.exit(0 if not bad else 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--log", required=True)
    s.add_argument("--unit", required=True)
    s.add_argument("--pid", default=None)
    s.add_argument("--timeout", default=28800)
    s.add_argument("--poll", default=30)
    e = sub.add_parser("ensemble")
    e.add_argument("--sanity", required=True)
    e.add_argument("--unit", required=True)
    a = ap.parse_args()
    {"seed": cmd_seed, "ensemble": cmd_ensemble}[a.cmd](a)


if __name__ == "__main__":
    main()
