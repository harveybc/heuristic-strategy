#!/usr/bin/env python3
"""T3 v2: extractor TYPE per feature group, chosen from each group's T1 profile and decided by measurement
(owner 2026-10-23: "selección del extractor basado en el perfil de cada grupo"; Musashi's instructions 2026-10-22 §4).

Profile -> candidates and a PREREGISTERED expectation; measurement -> decision. Nothing is chosen by the profile alone.
  candidates   every multi-column group tests BOTH alternatives against the control: `pointwise` (kernel 1, no temporal
               mixing inside the extractor) and `per_feature` (one small causal extractor per column); a one-column
               group tests `pointwise` (per_feature == conv there). The control is the shared causal `conv` (kernel 3).
  expectation  written before any T3 result, from the group's TRAIN profile (Musashi's T1, manifest 5729d807):
                 memory  = "persistent" if median ACF1 >= 0.9 and (median ACF24 >= 0.5 or no insignificant lag <= 48),
                           "near_white" if |median ACF1| < 0.1, else "intermediate";
                 redundancy = max |Spearman| inside the family: "high" >= 0.95, "low" < 0.5, else "mid";
               persistent -> pointwise expected >= conv (the level carries the information; mixing slots adds little);
               near_white -> conv / per_feature expected >= pointwise (short-term patterns live across slots);
               high redundancy -> the shared conv expected >= per_feature (per-column extractors duplicate redundant
               inputs); low redundancy -> per_feature expected >= shared conv. The agreement rate is REPORTED; it never
               changes a decision.
  decision     per group, frozen: an alternative replaces the control only when (a) its seed-mean INNER_A Sharpe exceeds
               the control's by more than the pooled seed std of the two, and (b) its mean TRAIN-INNER_A gap is not larger
               than the control's by more than the pooled seed std of the gaps, with the same TRAIN / INNER_A support
               hashes and all three seeds valid; two qualifying alternatives -> the higher Sharpe; ties -> fewer
               parameters; otherwise the control stays (a negative result also closes the unit). OBS_2024 never read.
  combination  one confirmation unit assembling every per-group choice, 3 seeds: adopted only if its INNER_A Sharpe is
               not below the control's by more than the pooled seed std (interaction check); otherwise the control
               representation stays and the per-group gains are reported as not additive.
Fixed for every unit: the v4 strict list, the groups file (digest), columns and encodings of each group, the T2
regularization decision, window, model (units 2 / depth 2 / head 16), replay contract, >= 2 orders/week (period
average), consolidated split, agent_state none, ZigZag labels. Duplicated information across groups (T1 annex) stays
present: T3 measures the extractor type CONDITIONAL on the other inputs, not the unique contribution of a group.
Ids are stable (groups-file order); odd ids -> Musashi's worker, even ids and the control -> Satoshi's."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKERS = {"musashi": "odd", "satoshi": "even"}
KINDS = ("pointwise", "per_feature")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def group_profile(rows: list) -> dict:
    """Median ACF1 / ACF24 / first insignificant lag, max within-family redundancy, max fold-stability ratio."""
    med = lambda k: statistics.median([v for v in (_f(r[k]) for r in rows) if v is not None]) if any(_f(r[k]) is not None for r in rows) else None  # noqa: E731
    insig = [_f(r["acf_first_insignificant_lag"]) for r in rows]
    within = [v for v in (_f(r["redundancy_within_family_max"]) for r in rows) if v is not None]
    stab = [v for v in (_f(r["fold_stability_ratio"]) for r in rows) if v is not None]
    return {"n": len(rows), "acf1_median": med("acf_lag1"), "acf24_median": med("acf_lag24"),
            "no_insignificant_lag_le_48_share": sum(v is None for v in insig) / len(rows),
            "first_insignificant_lag_median": statistics.median([v for v in insig if v is not None]) if any(v is not None for v in insig) else None,
            "within_family_redundancy_max": max(within) if within else None, "fold_stability_max": max(stab) if stab else None,
            "constant_columns": [r["column"] for r in rows if str(r.get("constant_fit")).lower() == "true"]}


def expectation(pr: dict) -> dict:
    a1, a24 = pr["acf1_median"], pr["acf24_median"]
    persistent = a1 is not None and a1 >= 0.9 and ((a24 is not None and a24 >= 0.5) or pr["no_insignificant_lag_le_48_share"] >= 0.5)
    memory = "persistent" if persistent else ("near_white" if a1 is not None and abs(a1) < 0.1 else "intermediate")
    w = pr["within_family_redundancy_max"]
    red = None if w is None else ("high" if w >= 0.95 else ("low" if w < 0.5 else "mid"))
    exp = []
    if memory == "persistent":
        exp.append("pointwise >= conv")
    if memory == "near_white":
        exp.append("conv >= pointwise")
    if red == "high":
        exp.append("conv >= per_feature")
    if red == "low":
        exp.append("per_feature >= conv")
    return {"memory": memory, "redundancy": red, "expected": exp or ["no directional expectation"]}


def variant_groups(groups: dict, units: dict, name: str, kind: str, uid: int):
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
    return {"name": f"t3_unit_{uid:02d}", "rule": f"T3 unit {uid}: group '{name}' -> {kind}; every other group conv",
            "groups": g, "units": u, "kinds": kinds}


def cmd_emit(a):
    spec = json.loads(Path(a.spec).read_text())
    gfile = json.loads(Path(a.groups).read_text())
    cols = json.loads(Path(a.list).read_text())["features"]
    groups, units = gfile["groups"], gfile.get("units", {})
    if sorted(c for g in groups.values() for c in g) != sorted(cols):
        raise SystemExit("T3_GROUPS_DO_NOT_PARTITION_THE_LIST")
    prof_rows = list(csv.DictReader(open(a.profiles)))
    base = lambda c: c.split("@")[0]  # noqa: E731
    by_col = {base(r["column"]): r for r in prof_rows}
    missing = [c for c in cols if base(c) not in by_col]
    if missing:
        raise SystemExit(f"T3_PROFILE_MISSING:{missing[:5]}")
    reg = spec["regularization"]
    if reg.get("status") != "DECIDED" or reg.get("dropout") is None:
        raise SystemExit("T3_REGULARIZATION_NOT_DECIDED: apply the T2 decision first")
    out = Path(a.out)
    (out / "groups").mkdir(parents=True, exist_ok=True)
    m = spec["model"]
    fixed = {"list": Path(a.list).name, "list_sha256": sha(a.list), "groups_file": Path(a.groups).name, "groups_sha256": sha(a.groups),
             "profiles_file": Path(a.profiles).name, "profiles_sha256": sha(a.profiles), "profiles_manifest_sha256": a.profiles_manifest_sha256,
             "model": {"units": m["units"], "depth": m["depth"], "head": m["head"], "core": m.get("core")}, "window": a.window,
             "agent_state": "none", "labels": "zigzag", "scaling": spec["scaling"], "nonstationary": spec["nonstationary"],
             "split_contract": spec["split_contract"], "exec_contract": spec["exec_contract"], "min_orders_per_week": spec["min_orders_per_week"],
             "seeds": [20261019, 20261020, 20261021], "max_epochs": m["max_epochs"],
             "regularization": {"dropout": reg["dropout"], "wd": reg["wd"], "chosen": reg.get("chosen"), "source": reg["source"],
                                "note": "selected by the T2 diagnostic at the E3-M default depth 1; carried to the clean-round model (depth 2) as declared"}}
    units_out = [{"id": 0, "group": None, "kind": "conv", "role": "control"}]
    uid = 0
    for name, members in groups.items():
        pr = group_profile([by_col[base(c)] for c in members])
        ex = expectation(pr)
        for kind in (KINDS if len(members) > 1 else ("pointwise",)):
            uid += 1
            units_out.append({"id": uid, "group": name, "n_columns": len(members), "kind": kind, "role": "contrast",
                              "profile": pr, "expectation": ex})
    for u in units_out:
        v = (variant_groups(groups, units, u["group"], u["kind"], u["id"]) if u["role"] == "contrast"
             else {"name": "t3_unit_00", "rule": "T3 control: every group conv", "groups": groups, "units": units, "kinds": {k: "conv" for k in groups}})
        gp = out / "groups" / f"unit_{u['id']:02d}.json"
        gp.write_text(json.dumps(v, indent=1))
        u.update(name=v["name"], groups_file=gp.name, groups_file_sha256=sha(gp),
                 worker="satoshi" if u["id"] % 2 == 0 else "musashi")
        ident = {k: u[k] for k in ("id", "group", "kind", "groups_file_sha256")} | fixed
        u["identity"] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    doc = {"schema": "t3_manifest.v2", "identity": f"t3v2_{hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()[:12]}",
           "fixed": fixed, "rules": __doc__, "assignment": {"musashi": "odd ids", "satoshi": "even ids and the control"},
           "units": units_out, "combination": {"status": "PENDING_PER_GROUP_DECISIONS", "id": "C"}}
    (out / "T3_MANIFEST.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"identity": doc["identity"], "units": len(units_out), "musashi": sum(u["worker"] == "musashi" for u in units_out),
                      "satoshi": sum(u["worker"] == "satoshi" for u in units_out)}))


def e3m_cmd(f, u, root, groups_name, out_name):
    return (f"tools/expert_guided_e3m.py run --admitted {root}/data/admitted --csv-dir {root}/data/sel15 "
            f"--store {root}/data/full_pool_s/full_pool_eurusd_h1.parquet --feature-list {root}/in/{f['list']} "
            f"--groups {root}/t3/groups/{groups_name} --out {root}/out/{out_name} --variants FULL "
            f"--units {f['model']['units']} --depth {f['model']['depth']} --head {f['model']['head']} "
            f"--dropout {f['regularization']['dropout']} --wd {f['regularization']['wd']} --max-epochs {f['max_epochs']} "
            f"--scaling {f['scaling']} --nonstationary {f['nonstationary']} --split-contract {f['split_contract']} --agent-state none "
            f"--window {f['window'][0]} --stride {f['window'][1]} --exec-contract {f['exec_contract']} --min-orders-per-week {f['min_orders_per_week']}")


def cmd_queue(a):
    """Typed unit listing + a sequential, resumable queue for one worker (units whose worker matches)."""
    man = json.loads(Path(a.manifest).read_text())
    f = man["fixed"]
    root = "<root>"
    mine = [u for u in man["units"] if u["worker"] == a.worker]
    typed = []
    for u in mine:
        cmd = e3m_cmd(f, u, root, u["groups_file"], u["name"])
        typed.append({"id": u["id"], "name": u["name"], "group": u["group"], "kind": u["kind"], "identity": u["identity"],
                      "command": cmd, "command_identity": hashlib.sha256(cmd.encode()).hexdigest()[:16],
                      "expected": {"file": f"out/{u['name']}/E3M.json", "arch.group_kinds": "per the unit's groups file",
                                   "split_contract.arm": "consolidated", "agent_state.mode": "none", "labels.mode": "zigzag",
                                   "exec_contract": "sac", "min_orders_per_week": 2.0, "seeds": f["seeds"]}})
    Path(a.out_units).write_text(json.dumps({"manifest": man["identity"], "worker": a.worker, "units": typed}, indent=1))
    L = ["#!/bin/bash", f"# T3 {man['identity']} queue for {a.worker} ({len(typed)} units). Sequential, resumable (STATE markers),",
         "# retries while not admitted, never lowers a refused cap. A marker is not a result: verify each E3M.json by identity.",
         'ROOT=${ROOT:?set ROOT to the package root}; PY=${PY:-$HOME/anaconda3/envs/trading-stack/bin/python}; CR=${CR:-$HOME/.local/bin/crispdm-run}',
         'MEM=${MEM:-4G}; GPU=${GPU:?set GPU to the CUDA index of the assigned card}',
         'mkdir -p $ROOT/STATE $ROOT/out; log(){ echo "$(date -u +%FT%TZ) [T3] $*" >> $ROOT/LOG; }', "cd $ROOT/code || exit 1"]
    for t in typed:
        cmd = t["command"].replace("<root>", "$ROOT")
        L += [f'if [ ! -f $ROOT/STATE/{t["name"]} ]; then',
              f'  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=$GPU $CR -q -m $MEM -t 43200s -n t3-{t["id"]:02d} -- $PY {cmd} > $ROOT/out/{t["name"]}.log 2>&1 '
              f'&& {{ touch $ROOT/STATE/{t["name"]}; log "{t["name"]} DONE identity {t["identity"]}"; break; }}',
              f'    grep -q "REFUSED\\|QUEUED HOST_HEADROOM" $ROOT/out/{t["name"]}.log && {{ log "{t["name"]} not admitted (try $try)"; sleep 600; }} || {{ log "{t["name"]} FAILED"; break; }}',
              "  done", "fi"]
    L.append(f'log "T3_{a.worker.upper()}_QUEUE_END"')
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"written {a.out} and {a.out_units} ({len(typed)} units for {a.worker})")


def _arm(e):
    tr = e["trials"]["FULL"]
    sh = [tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
    gp = [tr[s]["trading"]["TRAIN"]["sharpe_daily_sqrt365"] - tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
    return {"sharpe": statistics.mean(sh), "sharpe_std": statistics.pstdev(sh), "gap": statistics.mean(gp), "gap_std": statistics.pstdev(gp),
            "valid": bool(e["decision"]["means"]["FULL"]["all_valid"]), "params": sum(e["params"]["FULL"].values()),
            "support": {p: e["meta"]["support_hash"][p] for p in ("TRAIN", "INNER_A")}, "kinds": e.get("arch", {}).get("group_kinds")}


def cmd_decide(a):
    """Per-group decision by the frozen rule; reads only TRAIN / INNER_A. Every unit must exist and share the control's
    support; the expectation agreement is reported, never used."""
    man = json.loads(Path(a.manifest).read_text())
    res = {}
    for u in man["units"]:
        f = Path(a.results) / u["name"] / "E3M.json"
        if not f.exists():
            raise SystemExit(f"T3_UNIT_NOT_FINISHED:{u['name']}")
        res[u["id"]] = _arm(json.loads(f.read_text())) | {"sha256": sha(f)}
    ctl = res[0]
    pool = lambda x, y: math.sqrt((x ** 2 + y ** 2) / 2)  # noqa: E731
    out, agree = [], []
    by_group = {}
    for u in man["units"][1:]:
        by_group.setdefault(u["group"], []).append(u)
    for g, us in by_group.items():
        rows = []
        for u in us:
            r = res[u["id"]]
            if r["support"] != ctl["support"]:
                raise SystemExit(f"T3_SUPPORT_DIFFERS:{u['name']}")
            q = (r["valid"] and ctl["valid"] and r["sharpe"] - ctl["sharpe"] > pool(r["sharpe_std"], ctl["sharpe_std"])
                 and r["gap"] - ctl["gap"] <= pool(r["gap_std"], ctl["gap_std"]))
            rows.append({"kind": u["kind"], "unit": u["name"], "qualifies": bool(q), **{k: r[k] for k in ("sharpe", "sharpe_std", "gap", "valid", "params")}})
        qual = [x for x in rows if x["qualifies"]]
        chosen = max(qual, key=lambda x: (x["sharpe"], -x["params"]))["kind"] if qual else "conv"
        ex = us[0]["expectation"]
        for e_ in ex["expected"]:
            if ">=" in e_:
                w, l = [s.strip() for s in e_.split(">=")]
                sh = {"conv": ctl["sharpe"], **{x["kind"]: x["sharpe"] for x in rows}}
                if w in sh and l in sh:
                    agree.append({"group": g, "expectation": e_, "held": bool(sh[w] >= sh[l])})
        out.append({"group": g, "chosen": chosen, "control": {k: ctl[k] for k in ("sharpe", "sharpe_std", "gap", "valid", "params")},
                    "alternatives": rows, "profile_expectation": ex})
    doc = {"manifest": man["identity"], "rule": "see manifest rules (decision)", "groups": out,
           "expectation_agreement": {"n": len(agree), "held": sum(x["held"] for x in agree), "items": agree},
           "never_read": ["OBS_2024", "INNER_B", "TEST"]}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"changed": {x["group"]: x["chosen"] for x in out if x["chosen"] != "conv"},
                      "agreement": f"{doc['expectation_agreement']['held']}/{doc['expectation_agreement']['n']}"}, indent=1))


def combined_groups(man: dict, dec: dict, unit00_path) -> dict:
    """The combination groups document implied by a per-group decision (pure; the recipe exporter regenerates it)."""
    if dec.get("manifest") != man.get("identity"):
        raise SystemExit("T3_DECISION_OF_ANOTHER_MANIFEST")
    g0 = json.loads(Path(unit00_path).read_text())
    groups, units, kinds = dict(g0["groups"]), dict(g0["units"]), dict(g0["kinds"])
    for d in dec["groups"]:
        if d["chosen"] == "pointwise":
            kinds[d["group"]] = "pointwise"
        elif d["chosen"] == "per_feature":
            members = groups.pop(d["group"])
            kinds.pop(d["group"])
            for c in members:
                groups[f"{d['group']}/{c}"] = [c]
                units[f"{d['group']}/{c}"] = g0["units"].get(d["group"], 2)
                kinds[f"{d['group']}/{c}"] = "conv"
    return {"name": "t3_unit_C", "rule": "T3 combination: every per-group choice of the decision", "groups": groups, "units": units, "kinds": kinds}


def cmd_combine(a):
    """Groups file for the combination confirmation (every per-group choice at once)."""
    man = json.loads(Path(a.manifest).read_text())
    dec = json.loads(Path(a.decision).read_text())
    v = combined_groups(man, dec, Path(a.manifest).parent / "groups" / "unit_00.json")
    gp = Path(a.manifest).parent / "groups" / "unit_C.json"
    gp.write_text(json.dumps(v, indent=1))
    print(json.dumps({"groups_file": gp.name, "sha256": sha(gp), "changed": {d["group"]: d["chosen"] for d in dec["groups"] if d["chosen"] != "conv"}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.add_argument("--spec", required=True); e.add_argument("--groups", required=True)
    e.add_argument("--list", required=True); e.add_argument("--profiles", required=True); e.add_argument("--out", required=True)
    e.add_argument("--profiles-manifest-sha256", required=True); e.add_argument("--window", type=int, nargs=2, default=[13, 2])
    q = sub.add_parser("queue"); q.add_argument("--manifest", required=True); q.add_argument("--worker", choices=list(WORKERS), required=True)
    q.add_argument("--out", required=True); q.add_argument("--out-units", required=True)
    d = sub.add_parser("decide"); d.add_argument("--manifest", required=True); d.add_argument("--results", required=True); d.add_argument("--out", required=True)
    c = sub.add_parser("combine"); c.add_argument("--manifest", required=True); c.add_argument("--decision", required=True)
    a = ap.parse_args()
    sys.exit({"emit": cmd_emit, "queue": cmd_queue, "decide": cmd_decide, "combine": cmd_combine}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
