#!/usr/bin/env python3
"""Write the three v1.3 action contracts (2026-10-25, after the v12 chain: one-sided seeds and activity on the 2.0
boundary) from the committed v1.2 / C2 / C2G contracts: per-side open genes, the side-balance constraint (<= 90 %), the
activity rule held with confidence (CI90 lower bound of trades/week >= 2.0, weekly-block bootstrap; owner: "Con
confianza estadística") and the front-size diagnostic with a declared budget escalation. Prints identity and sha256."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402

PAIRS = (("ACTION_CONTRACT_v1_2.json", "ACTION_CONTRACT_v1_3.json", ACT.contract_v13, ACT.validate),
         ("ACTION_CONTRACT_C2.json", "ACTION_CONTRACT_C2_v1_3.json", C2.contract_v13, C2.validate),
         ("ACTION_CONTRACT_C2G.json", "ACTION_CONTRACT_C2G_v1_3.json", C2.contract_v13, C2.validate))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(ROOT / "docs/short_horizon/clean_round/contracts"))
    a = ap.parse_args()
    d = Path(a.dir)
    out = {}
    for src, dst, make, check in PAIRS:
        doc = make(json.loads((d / src).read_text()))
        errs = check(doc)
        if errs:
            raise SystemExit(f"CONTRACT_V13_INVALID:{dst}:{errs}")
        (d / dst).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        out[dst] = {"identity": doc["identity"], "sha256": ACT.sha256_file(d / dst), "derived_from": doc["derived_from"]["identity"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
