#!/usr/bin/env python3
"""Derive the C3 recipe (audit A1-11): the selection's ENCODING.csv decisions applied as consumer ops to a (fold) recipe.
Usage: derive --recipe <base recipe> --store <parquet> --fold wf2022 --out <json>"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    for k in ("--recipe", "--store", "--fold", "--out"):
        d.add_argument(k, required=True)
    a = ap.parse_args()
    base = json.loads(Path(a.recipe).read_text())
    rec = C3.derive_encoding_recipe(base, hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest(), a.store, a.fold)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({"identity": rec["identity"], "columns": rec["list"]["n"], "groups": len(rec["representation"]["resolved"]["names"]),
                      "encoding_decisions_applied": rec["encoding_decisions_applied"], "out": a.out}))


if __name__ == "__main__":
    main()
