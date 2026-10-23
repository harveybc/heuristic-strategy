#!/usr/bin/env python3
"""T3: finite manifest of the extractor-TYPE contrast per feature group (Musashi's instructions 2026-10-22, §4).

One unit = one group whose extractor type is changed while every other group keeps the control type and the group's
columns / encodings stay fixed; the control unit (every group `conv`) is shared. Axes stay separate: representation
(parent / components) and encoding (probabilities / one-hot) were measured in the selection and are NOT varied here.
Types: `conv` (causal kernel-3 branch, the control), `pointwise` (kernel 1: no temporal mixing inside the extractor;
daily / COT / calendar / event / factor inputs and single-column groups), `per_feature` (one extractor per column of
the group: per-scale wavelet branches, per-component Hilbert, ...). Group ids are the groups file order (stable, pinned
by the file digest): odd ids -> Musashi's worker, even ids (and the control, id 0) -> Satoshi's. Regularization is an
explicit dependency on the T2 decision file: `queue` refuses to emit without it. Three seeds inside every E3-M run,
replay contract, >= 2 orders/week (period average), decision on the inner block; 2024 read once; a negative result
also closes a unit. Nothing here launches anything."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POINTWISE_FAMILIES = ("calendar", "ev", "cot.", "dly.", "act", "fac.", "strength/")
WORKERS = {"musashi": "odd", "satoshi": "even"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contrast_for(name: str, members: list) -> str:
    if len(members) == 1 or name.startswith(POINTWISE_FAMILIES):
        return "pointwise"
    return "per_feature"


def variant_groups(groups: dict, units: dict, gid: int, name: str, kind: str):
    """Groups file content for one unit: every group `conv` except ``name`` with ``kind``."""
    g, u, kinds = dict(groups), dict(units), {k: "conv" for k in groups}
    if kind == "pointwise":
        kinds[name] = "pointwise"
    elif kind == "per_feature":
        del g[name]
        kinds.pop(name)
        for c in groups[name]:
            g[f"{name}/{c}"] = [c]
            u[f"{name}/{c}"] = units.get(name, 2)
            kinds[f"{name}/{c}"] = "conv"
    elif kind != "conv":
        raise ValueError(f"T3_KIND_UNKNOWN:{kind}")
    return {"name": f"t3_unit_{gid:02d}_{kind}", "rule": f"T3 unit {gid}: group '{name}' -> {kind}; every other group conv",
            "groups": g, "units": u, "kinds": kinds}


def cmd_emit(a):
    spec = json.loads(Path(a.spec).read_text())
    gfile = json.loads(Path(a.groups).read_text())
    cols = json.loads(Path(a.list).read_text())["features"]
    groups, units = gfile["groups"], gfile.get("units", {})
    flat = [c for g in groups.values() for c in g]
    if sorted(flat) != sorted(cols):
        raise SystemExit("T3_GROUPS_DO_NOT_PARTITION_THE_LIST")
    out = Path(a.out)
    (out / "groups").mkdir(parents=True, exist_ok=True)
    m = spec["model"]
    fixed = {"list": str(Path(a.list).resolve()), "list_sha256": sha(a.list), "groups_file": str(Path(a.groups).resolve()),
             "groups_sha256": sha(a.groups), "model": {"units": m["units"], "depth": m["depth"], "head": m["head"], "core": m.get("core")},
             "window": a.window, "agent_state": "none", "scaling": spec["scaling"], "nonstationary": spec["nonstationary"],
             "split_contract": spec["split_contract"], "exec_contract": spec["exec_contract"],
             "min_orders_per_week": spec["min_orders_per_week"], "seeds": 3, "max_epochs": m["max_epochs"],
             "regularization": {"source": "REGULARIZATION_DECISION.json (T2)", "status": "DEPENDENCY", "values": None}}
    unit_list = [{"id": 0, "group": None, "kind": "conv", "role": "control"}]
    for gid, (name, members) in enumerate(groups.items(), start=1):
        unit_list.append({"id": gid, "group": name, "n_columns": len(members), "kind": contrast_for(name, members), "role": "contrast"})
    for u in unit_list:
        v = (variant_groups(groups, units, u["id"], u["group"], u["kind"]) if u["role"] == "contrast"
             else {"name": "t3_unit_00_control", "rule": "T3 control: every group conv", "groups": groups, "units": units,
                   "kinds": {k: "conv" for k in groups}})
        gp = out / "groups" / f"unit_{u['id']:02d}.json"
        gp.write_text(json.dumps(v, indent=1))
        u["groups_file"] = str(gp.resolve())
        u["groups_file_sha256"] = sha(gp)
        u["worker"] = "satoshi" if u["id"] % 2 == 0 else "musashi"
        u["name"] = v["name"]
        ident = {k: u[k] for k in ("id", "group", "kind", "groups_file_sha256")} | fixed
        u["identity"] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
        u["expected"] = {"file": f"out/{u['name']}/E3M.json", "fields": ["trials.FULL (3 seeds)", "decision.means.FULL",
                                                                              "meta.support_hash.TRAIN/INNER_A", "params.FULL", "arch.group_kinds"]}
    doc = {"identity": f"t3_{hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()[:12]}", "fixed": fixed,
           "assignment": {"musashi": "odd ids", "satoshi": "even ids and the control"},
           "decision": "per unit vs the control on the inner block (same TRAIN/INNER_A support hashes): INNER_A Sharpe seed-mean, "
                       "TRAIN-INNER_A gap, seed std, max DD and parameters per branch / fusion / core reported; a unit closes with a "
                       "positive OR a negative result; no type is declared optimal for SAC from this proxy",
           "axes_not_varied": ["representation parent/components (selection)", "encoding probabilities/one-hot (selection)",
                               "columns of the group", "window", "regularization (T2 decision, shared by every unit)"],
           "units": unit_list}
    (out / "T3_MANIFEST.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"units": len(unit_list), "musashi": sum(u["worker"] == "musashi" for u in unit_list),
                      "satoshi": sum(u["worker"] == "satoshi" for u in unit_list), "manifest": str(out / "T3_MANIFEST.json")}))


def cmd_queue(a):
    man = json.loads(Path(a.manifest).read_text())
    dec_path = Path(a.decision)
    if not dec_path.exists():
        raise SystemExit("T3_REGULARIZATION_DEPENDENCY_MISSING: REGULARIZATION_DECISION.json (T2) not found; the queue is not emitted")
    dec = json.loads(dec_path.read_text())
    if dec.get("how") == "DEFAULT_BY_VALIDITY":
        raise SystemExit("T3_REGULARIZATION_DEFAULT_BY_VALIDITY: not a usable decision")
    do, wd = dec["values"]["dropout"], dec["values"]["wd"]
    f = man["fixed"]
    R = a.remote_root
    L = ["#!/bin/bash", f"# T3 queue for {a.worker} ({WORKERS[a.worker]} ids), manifest {man['identity']}, regularization from "
         f"{dec_path.name} ({dec.get('how')}): dropout {do} wd {wd}; sequential, resumable (STATE markers); never lowers a refused cap",
         f"Q={R}; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run", "MEM=${MEM:-4G}; GPU=${GPU:-0}",
         "mkdir -p $Q/STATE $Q/out", 'log(){ echo "$(date -u +%FT%TZ) [T3] $*" >> $Q/LOG; }', "cd $Q/code || exit 1"]
    n = 0
    for u in man["units"]:
        if u["worker"] != a.worker:
            continue
        n += 1
        out = f"$Q/out/{u['name']}"
        cmd = (f"tools/expert_guided_e3m.py run --admitted $Q/data/admitted --csv-dir $Q/data/sel15 --store $Q/data/store/{Path(a.store).name} "
               f"--feature-list $Q/in/{Path(f['list']).name} --groups $Q/groups/{Path(u['groups_file']).name} --out {out} --variants FULL "
               f"--units {f['model']['units']} --depth {f['model']['depth']} --head {f['model']['head']} --dropout {do} --wd {wd} "
               f"--max-epochs {f['max_epochs']} --scaling {f['scaling']} --nonstationary {f['nonstationary']} --split-contract {f['split_contract']} "
               f"--agent-state none --window {f['window'][0]} --stride {f['window'][1]} --exec-contract {f['exec_contract']} "
               f"--min-orders-per-week {f['min_orders_per_week']}")
        L += [f'if [ ! -f $Q/STATE/{u["name"]} ]; then',
              f'  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-{u["id"]:02d} -- $PY {cmd} > {out}.log 2>&1 '
              f'&& {{ touch $Q/STATE/{u["name"]}; log "{u["name"]} DONE identity {u["identity"]}"; break; }}',
              f'    grep -q "REFUSED\\|QUEUED HOST_HEADROOM" {out}.log && {{ log "{u["name"]} not admitted (try $try)"; sleep 600; }} || {{ log "{u["name"]} FAILED"; break; }}',
              "  done", "fi"]
    L.append(f'log "T3_{a.worker.upper()}_QUEUE_END ({n} units; a marker is not a result: verify every E3M.json by identity)"')
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"written {a.out} ({n} units for {a.worker})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.add_argument("--spec", required=True); e.add_argument("--groups", required=True)
    e.add_argument("--list", required=True); e.add_argument("--out", required=True)
    e.add_argument("--window", type=int, nargs=2, default=[24, 1])
    q = sub.add_parser("queue"); q.add_argument("--manifest", required=True); q.add_argument("--decision", required=True)
    q.add_argument("--worker", choices=list(WORKERS), required=True); q.add_argument("--remote-root", required=True)
    q.add_argument("--store", required=True); q.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit({"emit": cmd_emit, "queue": cmd_queue}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
