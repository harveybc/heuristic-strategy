"""Read ledger of the decision blocks (audit A5-3: INNER_B was read ~130 times per fold and used to choose the fitness
variant while the reports called it "scored once after freeze"). Every evaluation of ANY decision block appends one
entry; a block read more than ``MAX_HELD_OUT_READS`` times is labelled DEVELOPMENT in every report and may never be
called held out again. Append-only JSONL; the count is the evidence, not the claim."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

MAX_HELD_OUT_READS = 1
STATUS_HELD_OUT, STATUS_DEVELOPMENT, STATUS_UNREAD = "HELD_OUT", "DEVELOPMENT", "UNREAD"
LEDGER_NAME = "READ_LEDGER.jsonl"


def record(path, block: str, unit: str, policy: str, variant: str, who: str, note: str = "", when: str | None = None) -> dict:
    """Append one read; returns the entry. ``block`` names the contract's decision window (e.g. 'wf2022q1/INNER_B')."""
    if not block or "/" not in block:
        raise ValueError(f"LEDGER_BLOCK_MALFORMED:{block!r} (expected '<fold>/<period>')")
    e = {"utc": when or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "block": block, "unit": unit,
         "policy": policy, "variant": variant, "who": who, "note": note}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(e, sort_keys=True) + "\n")
    return e


def entries(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def counts(path) -> dict:
    out = {}
    for e in entries(path):
        out[e["block"]] = out.get(e["block"], 0) + 1
    return out


def block_status(n_reads: int, max_reads: int = MAX_HELD_OUT_READS) -> str:
    if n_reads <= 0:
        return STATUS_UNREAD
    return STATUS_HELD_OUT if n_reads <= max_reads else STATUS_DEVELOPMENT


def status_table(path, max_reads: int = MAX_HELD_OUT_READS) -> dict:
    c = counts(path)
    return {b: {"reads": n, "status": block_status(n, max_reads)} for b, n in sorted(c.items())}


def markdown(path, max_reads: int = MAX_HELD_OUT_READS) -> list:
    t = status_table(path, max_reads)
    md = [f"## Read ledger of the decision blocks (max {max_reads} read(s) for HELD_OUT; beyond that the block is DEVELOPMENT)", "",
          "| block | reads | status |", "| --- | ---: | --- |"]
    md += [f"| {b} | {v['reads']} | {v['status']} |" for b, v in t.items()]
    return md + [""]
