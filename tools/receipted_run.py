#!/usr/bin/env python3
"""Run one typed unit and seal it with an execution RECEIPT in the same format as the peer worker's
(RESULT.json: job, config identity, effective command, every sealed file hash, support, columns), so the same
verifier (tools/expert_guided_clean_round.py verify-t2) authenticates either producer. A marker is not a result.

  config   --root R --out CONFIG.json: pins every Python file of R/code (app, tools) and records the science commit;
           its identity = sha256 of the key-sorted JSON (the peer worker's convention)
  run      --root R --config CONFIG.json --unit UNIT.json --name N: executes the unit's command (``<root>`` -> R) through
           the admission wrapper, then writes R/ops/N.RESULT.json; a non-zero exit writes no receipt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def cmd_config(a):
    root = Path(a.root)
    pins = {f"<root>/code/{p.relative_to(root / 'code')}": sha(p) for p in sorted((root / "code").rglob("*.py"))
            if p.relative_to(root / "code").parts[0] in ("app", "tools")}
    cfg = {"producer": a.producer, "science_commit": a.science_commit, "pins": pins}
    Path(a.out).write_text(json.dumps(cfg, indent=1))
    print(json.dumps({"identity": identity(cfg), "pinned": len(pins)}))


def cmd_run(a):
    root = Path(a.root)
    cfg = json.loads(Path(a.config).read_text())
    for k, v in cfg["pins"].items():                       # the code that runs is the code the config pins
        if sha(k.replace("<root>", str(root))) != v:
            raise SystemExit(f"CODE_DIFFERS_FROM_PIN:{k}")
    u = json.loads(Path(a.unit).read_text())
    cmd = u["command"]
    if hashlib.sha256(cmd.encode()).hexdigest()[:16] != u.get("command_identity", u["identity"]):
        raise SystemExit("UNIT_IDENTITY")
    argv = shlex.split(cmd.replace("<root>", str(root)))
    out = Path(argv[argv.index("--out") + 1])
    wrap = [a.crispdm, "-q", "-m", a.mem, "-t", a.wall, "-n", f"rr-{a.name}", "--", a.python] + argv
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=a.gpu)
    log = root / "out" / f"{a.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        rc = subprocess.run(wrap, cwd=root / "code", env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(json.dumps({"unit": a.name, "returncode": rc, "receipt": None}))
        return rc
    e3m = json.loads((out / "E3M.json").read_text())
    tok = shlex.split(cmd)
    arg = lambda f: tok[tok.index(f) + 1] if f in tok else None  # noqa: E731
    job = {"name": a.name, "list": Path(arg("--feature-list")).name if arg("--feature-list") else None, "split": arg("--split-contract"),
           "state": arg("--agent-state"), "dropout": float(arg("--dropout") or "nan"), "wd": float(arg("--wd") or "nan"),
           "peer_command": cmd, "peer_command_identity": u.get("command_identity", u["identity"]),
           **({"unit_identity": u["identity"]} if "command_identity" in u else {})}
    if arg("--groups"):
        job["groups_file"] = Path(arg("--groups")).name
        job["groups_sha256"] = sha(arg("--groups").replace("<root>", str(root)))
    if arg("--recipe"):
        job["recipe_sha256"] = sha(arg("--recipe").replace("<root>", str(root)))
    feats = list(e3m["meta"]["input_contract"]["features"])
    req = json.loads(Path(arg("--feature-list").replace("<root>", str(root))).read_text())["features"] if arg("--feature-list") else \
        [c["store_column"] for c in json.loads(Path(arg("--recipe").replace("<root>", str(root))).read_text())["tensor"]["columns"]]
    rec = {"job": job, "config_sha256": identity(cfg), "hashes": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
           "requested_features": req, "consumed_features": [f.split("@diff")[0] if f.endswith("@diff") and f[:-5] in req else f for f in feats],
           "support_hash": e3m["meta"]["support_hash"]}
    ops = root / "ops"
    ops.mkdir(exist_ok=True)
    (ops / f"{a.name}.RESULT.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"unit": a.name, "returncode": 0, "receipt": str(ops / f"{a.name}.RESULT.json")}))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config"); c.add_argument("--root", required=True); c.add_argument("--out", required=True)
    c.add_argument("--producer", default="satoshi"); c.add_argument("--science-commit", required=True)
    r = sub.add_parser("run"); r.add_argument("--root", required=True); r.add_argument("--config", required=True)
    r.add_argument("--unit", required=True); r.add_argument("--name", required=True); r.add_argument("--gpu", required=True)
    r.add_argument("--mem", default="4G"); r.add_argument("--wall", default="43200s")
    r.add_argument("--crispdm", default=str(Path.home() / ".local/bin/crispdm-run"))
    r.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    sys.exit({"config": cmd_config, "run": cmd_run}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
