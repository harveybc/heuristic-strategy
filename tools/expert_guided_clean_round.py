#!/usr/bin/env python3
"""Clean expert-guided round (owner 2026-10-22): validated, gated, finite. Nothing here trains a model by itself.

  check   validate the spec against the hard rules and print the arm table with exact parameter counts
  emit    write the sequential, resumable GPU queue for one stage — REFUSED unless every gate exists
  decide  apply the frozen decision rule to a finished stage (reads INNER_A fields only; never the 2024 read)

Hard rules (a violation refuses `check` and `emit`):
  * scaling == fixed_train (TRAIN-only constants for every input); the rolling z-score is forbidden;
  * nonstationary in {diff, refuse} (a non-stationary level never reaches a fixed scaler undecided);
  * agent_state in {none, minimal}; `none` is present in every stage (no agent feedback is the owner's rule;
    `minimal` = position + can_enter is a declared control); the simulator always keeps its real state;
  * split contract == consolidated (TRAIN 2020-2023 with the inner block 2023H2 for every choice, VALIDATION 2024
    read once, TEST 2025 closed);
  * the store file and the feature lists are bound by digest; list columns exist in the store; no `raw.` control
    column and no `enc.` encoding column unless the spec adopts encodings with evidence;
  * small extractors: units per extractor group <= limits.max_extractor_units; core width and head bounded;
  * the temporal receptive field covers the window (no slot the trunk cannot see);
  * extra ZigZag demonstrators come from an admission file (training-only demonstrations, never an input);
  * regularization is filled from a declared source (the generalization-gap diagnostic), never left implicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORBIDDEN_PREFIXES = ("raw.",)
E3M_MODULES = {"open_l": (2, 2), "open_s": (2, 2), "close_l": (2, 0), "close_s": (2, 0)}   # FULL variant


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def receptive_field(depth: int) -> int:
    from app.short_horizon.expert_guided.temporal import DILATIONS, KERNEL
    dil = list(DILATIONS) + [DILATIONS[2] * 2 ** (k - 1) for k in range(2, depth + 1)]
    return 1 + sum((KERNEL - 1) * d for d in dil)


def load_groups(path, cols):
    g = json.loads(Path(path).read_text())
    idx, units, names = [], [], []
    base = [c.split("@")[0] for c in cols]
    for name, members in g["groups"].items():
        ii = [base.index(m.split("@")[0]) for m in members if m.split("@")[0] in base]
        if ii:
            idx.append(ii), units.append(int(g.get("units", {}).get(name, 2))), names.append(name)
    return idx, units, names


def arm_params(arm, spec, cols):
    from app.short_horizon.expert_guided.temporal import TemporalPolicyNet
    m = spec["model"]
    kw = dict(units=m["units"], core=m.get("core"), depth=m["depth"], head=m["head"])
    if arm["extractor"] == "per_family":
        idx, units, names = load_groups(arm["groups"], cols)
        kw.update(groups=idx, group_units=units, group_names=names)
    win = arm["window"][0]
    return {mod: int(sum(p.numel() for p in TemporalPolicyNet(len(cols), win, nc, n_reg=nr, n_state=9, **kw).parameters()))
            for mod, (nc, nr) in E3M_MODULES.items()}


def validate(spec: dict, local_root: Path | None = None) -> list:
    errs = []
    m, lim = spec["model"], spec["limits"]
    if spec["scaling"] != "fixed_train":
        errs.append(f"SCALING_FORBIDDEN:{spec['scaling']}")
    if spec["nonstationary"] not in ("diff", "refuse"):
        errs.append(f"NONSTATIONARY_POLICY_FORBIDDEN:{spec['nonstationary']}")
    if spec["split_contract"] != "consolidated":
        errs.append(f"CONTRACT_NOT_PRINCIPAL:{spec['split_contract']}")
    reg = spec.get("regularization") or {}
    if reg.get("dropout") is None or reg.get("wd") is None or not reg.get("source"):
        errs.append("REGULARIZATION_NOT_DECLARED")
    if m["units"] > lim["max_extractor_units"] or (m.get("core") or m["units"]) > lim["max_core_width"] or m["head"] > lim["max_head"]:
        errs.append("MODEL_EXCEEDS_DECLARED_LIMITS")
    store = Path(spec["store"]["path"])
    man = None
    if store.exists():
        if sha(store) != spec["store"]["sha256"]:
            errs.append("STORE_DIGEST_MISMATCH")
        man = json.loads((store.parent / "MANIFEST.json").read_text())
    adopted = bool(spec.get("encodings", {}).get("adopted"))
    for stage, arms in spec["stages"].items():
        if not any(a["agent_state"] == "none" for a in arms):
            errs.append(f"STAGE_WITHOUT_NO_FEEDBACK_ARM:{stage}")
        for a in arms:
            tag = f"{stage}:{a['name']}"
            if a["agent_state"] not in ("none", "minimal"):
                errs.append(f"AGENT_STATE_FORBIDDEN:{tag}:{a['agent_state']}")
            win, stride = a["window"]
            if receptive_field(m["depth"]) < win:
                errs.append(f"RECEPTIVE_FIELD_BELOW_WINDOW:{tag}:{receptive_field(m['depth'])}<{win}")
            if a.get("extra_experts"):
                adm = json.loads(Path(spec["demonstrators"]["file"]).read_text())
                ok = {d["depth"] for d in adm["demonstrators"] if d["admitted"]}
                miss = [d for d in a["extra_experts"] if d not in ok]
                if miss:
                    errs.append(f"DEMONSTRATOR_NOT_ADMITTED:{tag}:{miss}")
            if a.get("list") is None:
                continue                                    # stage B arms inherit the stage-A winner's list
            lp = Path(a["list"])
            if not lp.exists():
                errs.append(f"LIST_MISSING:{tag}")
                continue
            if a.get("list_sha256") and sha(lp) != a["list_sha256"]:
                errs.append(f"LIST_DIGEST_MISMATCH:{tag}")
            cols = json.loads(lp.read_text())["features"]
            bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES) or (c.startswith("enc.") and not adopted)]
            if bad:
                errs.append(f"FORBIDDEN_COLUMNS:{tag}:{bad[:5]}")
            if man is not None:
                miss = [c for c in cols if c not in man["features"]]
                if miss:
                    errs.append(f"COLUMNS_NOT_IN_STORE:{tag}:{miss[:5]}")
            if a["extractor"] == "per_family":
                if not a.get("groups") or not Path(a["groups"]).exists():
                    errs.append(f"GROUPS_FILE_MISSING:{tag}")
                else:
                    idx, units, _ = load_groups(a["groups"], cols)
                    if sorted(i for g in idx for i in g) != list(range(len(cols))):
                        errs.append(f"GROUPS_DO_NOT_PARTITION_LIST:{tag}")
                    if max(units) > lim["max_extractor_units"]:
                        errs.append(f"EXTRACTOR_TOO_LARGE:{tag}")
    return errs


def gates_open(spec) -> list:
    missing = []
    for g in spec["gates"]:
        p = Path(g["path"])
        if not p.exists() or (g.get("must_contain") and g["must_contain"] not in p.read_text()):
            missing.append(g)
    return missing


def e3m_cmd(spec, arm, out, remote):
    m, reg = spec["model"], spec["regularization"]
    win, stride = arm["window"]
    r = lambda p: str(Path(remote) / Path(p).name)  # noqa: E731
    cmd = ["tools/expert_guided_e3m.py", "run", "--admitted", f"{remote}/admitted", "--csv-dir", f"{remote}/sel15",
           "--store", f"{remote}/store/{Path(spec['store']['path']).name}", "--feature-list", r(arm["list"]),
           "--out", out, "--variants", "FULL", "--units", str(m["units"]), "--depth", str(m["depth"]), "--head", str(m["head"]),
           "--dropout", str(reg["dropout"]), "--wd", str(reg["wd"]), "--max-epochs", str(m["max_epochs"]),
           "--scaling", spec["scaling"], "--nonstationary", spec["nonstationary"], "--split-contract", spec["split_contract"],
           "--agent-state", arm["agent_state"], "--window", str(win), "--stride", str(stride)]
    if m.get("core"):
        cmd += ["--core", str(m["core"])]
    if arm["extractor"] == "per_family":
        cmd += ["--groups", r(arm["groups"])]
    if arm.get("extra_experts"):
        cmd += ["--extra-experts", ",".join(str(d) for d in arm["extra_experts"])]
    return cmd


def cmd_check(a):
    spec = json.loads(Path(a.spec).read_text())
    errs = validate(spec)
    rows = []
    for stage, arms in spec["stages"].items():
        for arm in arms:
            if arm.get("list") and Path(arm["list"]).exists() and not errs:
                cols = json.loads(Path(arm["list"]).read_text())["features"]
                p = arm_params(arm, spec, cols)
                rows.append({"stage": stage, "arm": arm["name"], "cols": len(cols), "extractor": arm["extractor"], "window": arm["window"],
                             "agent_state": arm["agent_state"], "experts": [3] + list(arm.get("extra_experts") or []),
                             "params_per_module": p, "params_total": sum(p.values()), "receptive_field": receptive_field(spec["model"]["depth"])})
    print(json.dumps({"valid": not errs, "errors": errs, "gates_missing": [g["path"] for g in gates_open(spec)], "arms": rows}, indent=1))
    return 0 if not errs else 2


def cmd_emit(a):
    spec = json.loads(Path(a.spec).read_text())
    errs = validate(spec)
    if errs:
        raise SystemExit(f"SPEC_INVALID:{errs}")
    miss = gates_open(spec)
    if miss:
        raise SystemExit(f"GATES_CLOSED:{[g['path'] for g in miss]}")
    arms = spec["stages"][a.stage]
    if a.stage != "A":
        dec = json.loads(Path(a.winner_from).read_text())
        win = next(x for x in spec["stages"]["A"] if x["name"] == dec["chosen"])
        arms = [dict(win, **{k: v for k, v in x.items() if v is not None}, name=x["name"]) for x in arms]
    R = spec["remote_root"]
    L = ["#!/bin/bash", f"# clean round {spec['identity']} stage {a.stage}: sequential, resumable (STATE markers); generated by "
         "tools/expert_guided_clean_round.py emit; the gates were open at generation time and are re-checked here",
         f"Q={R}; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run", "mkdir -p $Q/STATE $Q/out",
         'log(){ echo "$(date -u +%FT%TZ) [' + a.stage + '] $*" >> $Q/LOG; }', f'test -f $Q/RELEASE.json || {{ log "RELEASE marker missing: refusing"; exit 3; }}',
         "cd $Q/code || exit 1"]
    for arm in arms:
        out = f"$Q/out/{a.stage}_{arm['name']}"
        cmd = " ".join(e3m_cmd(spec, arm, out, "$Q/data"))
        L += [f'if [ ! -f $Q/STATE/{a.stage}_{arm["name"]} ]; then',
              f'  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=0 $CR -q -m {spec["memory"]} -t 43200s -n clean-{a.stage}-{arm["name"]} -- $PY {cmd} > {out}.log 2>&1 && {{ touch $Q/STATE/{a.stage}_{arm["name"]}; log "{arm["name"]} DONE"; break; }}',
              f'    grep -q "REFUSED\\|QUEUED HOST_HEADROOM" {out}.log && {{ log "{arm["name"]} not admitted (try $try)"; sleep 600; }} || {{ log "{arm["name"]} FAILED"; break; }}',
              "  done", "fi"]
    L.append(f'log "STAGE_{a.stage}_DONE"')
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"written {a.out} ({len(arms)} arms)")


def cmd_decide(a):
    """Frozen rule: highest seed-mean INNER_A Sharpe (consolidated inner block) among arms whose three seeds are all
    valid; ties -> fewer parameters. Reads only INNER_A fields; 2024 is never consulted for a choice."""
    spec = json.loads(Path(a.spec).read_text())
    res = []
    for arm in spec["stages"][a.stage]:
        f = Path(a.results) / f"{a.stage}_{arm['name']}" / "E3M.json"
        if not f.exists():
            raise SystemExit(f"ARM_NOT_FINISHED:{arm['name']}")
        e = json.loads(f.read_text())
        mean = e["decision"]["means"]["FULL"]
        tr = e["trials"]["FULL"]
        gap = [tr[s]["trading"]["TRAIN"]["sharpe_daily_sqrt365"] - tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
        res.append({"arm": arm["name"], "inner_a_sharpe_mean": mean["inner_a_sharpe_mean"], "all_valid": mean["all_valid"],
                    "params": sum(e["params"]["FULL"].values()), "train_minus_inner_sharpe_mean": float(sum(gap) / len(gap))})
    ok = [r for r in res if r["all_valid"]] or res
    chosen = max(ok, key=lambda r: (r["inner_a_sharpe_mean"], -r["params"]))
    doc = {"stage": a.stage, "rule": cmd_decide.__doc__.strip(), "arms": res, "chosen": chosen["arm"]}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--spec", required=True)
    e = sub.add_parser("emit"); e.add_argument("--spec", required=True); e.add_argument("--stage", default="A")
    e.add_argument("--winner-from", default=None); e.add_argument("--out", required=True)
    d = sub.add_parser("decide"); d.add_argument("--spec", required=True); d.add_argument("--stage", required=True)
    d.add_argument("--results", required=True); d.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit({"check": cmd_check, "emit": cmd_emit, "decide": cmd_decide}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
