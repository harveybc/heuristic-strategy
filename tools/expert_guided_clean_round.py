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
  * extra ZigZag demonstrators come from an admission file (training-only demonstrations, never an input) and are
    SHORTER than the registered optimum (owner 2026-10-22: intermediate trades that earn less per trade);
  * exec_contract == sac (owner 2026-10-22: the replay contract, max 1 entry/day and leverage 1:50, everywhere);
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
    """(indices, units, names, kinds) of the groups file matched to the list; ``kinds`` = extractor type per group
    ("conv" default | "pointwise", the T3 contrast)."""
    return load_groups_doc(json.loads(Path(path).read_text()), cols)


def load_groups_doc(g: dict, cols):
    idx, units, names, kinds = [], [], [], []
    base = [c.split("@")[0] for c in cols]
    for name, members in g["groups"].items():
        ii = [base.index(m.split("@")[0]) for m in members if m.split("@")[0] in base]
        if ii:
            idx.append(ii), units.append(int(g.get("units", {}).get(name, 2))), names.append(name)
            kinds.append(str(g.get("kinds", {}).get(name, "conv")))
    return idx, units, names, kinds


def arm_params(arm, spec, cols):
    from app.short_horizon.expert_guided.temporal import TemporalPolicyNet
    m = spec["model"]
    kw = dict(units=m["units"], core=m.get("core"), depth=m["depth"], head=m["head"])
    if arm["extractor"] == "per_family":
        idx, units, names, kinds = load_groups(arm["groups"], cols)
        kw.update(groups=idx, group_units=units, group_names=names, group_kinds=kinds)
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
    if float(spec.get("min_orders_per_week", 0)) != 2.0:
        errs.append(f"MIN_ORDERS_PER_WEEK_NOT_OWNER_RULE:{spec.get('min_orders_per_week')}")
    if spec.get("exec_contract") != "sac":
        errs.append(f"EXEC_CONTRACT_NOT_REPLAY:{spec.get('exec_contract')}")
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
                ok = {str(d.get("spec", d.get("depth"))) for d in adm["demonstrators"] if d["admitted"]}
                miss = [d for d in a["extra_experts"] if str(d) not in ok]
                if miss:
                    errs.append(f"DEMONSTRATOR_NOT_ADMITTED:{tag}:{miss}")
                longer = [d for d in a["extra_experts"] if not str(d).startswith("rev") and int(d) >= 3]
                if longer:
                    errs.append(f"DEMONSTRATOR_NOT_SHORTER_THAN_OPTIMUM:{tag}:{longer}")
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
                    idx, units, _, _ = load_groups(a["groups"], cols)
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
           "--agent-state", arm["agent_state"], "--window", str(win), "--stride", str(stride), "--exec-contract", spec["exec_contract"],
           "--min-orders-per-week", str(spec["min_orders_per_week"])]
    if arm.get("aux_targets"):
        cmd += ["--aux-targets"]
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
                             "agent_state": arm["agent_state"], "experts": ["3"] + [str(x) for x in (arm.get("extra_experts") or [])],
                             "aux_targets": bool(arm.get("aux_targets")),
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


REG_RULE = ("Regularization from the T2 gap diagnostic on the corrected store with the v4 lists (peer worker; consolidated "
            "contract, inner block 2023H2; replay contract; >= 2 orders/week): per list, among arms with agent_state none whose "
            "three seeds are all valid and whose TRAIN / INNER_A support hashes match across the compared settings, each "
            "setting's vector = (INNER_A Sharpe seed-mean UP, TRAIN-INNER_A Sharpe gap DOWN, INNER_A Sharpe seed-std DOWN, "
            "INNER_A max drawdown % DOWN). A setting is dominated when another is at least as good on every component of every "
            "list and strictly better on at least one. Tie-break among non-dominated settings, declared before any result is "
            "read: (1) the higher mean INNER_A Sharpe across lists when the difference exceeds the pooled seed std, (2) the "
            "smaller mean gap, (3) base (the control). A setting with no valid list is excluded; with no valid setting at all "
            "the answer is base by default (DEFAULT_BY_VALIDITY) and the round is not released on it. OBS_2024 is never read.")


def _arm_vector(e: dict):
    """(sharpe_mean, gap_mean, sharpe_seed_std, dd_mean, all_valid, supports) of one E3M.json; INNER_A / TRAIN only."""
    tr = e["trials"]["FULL"]
    sh = [tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
    gap = [tr[s]["trading"]["TRAIN"]["sharpe_daily_sqrt365"] - tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
    dd = [tr[s]["trading"]["INNER_A"]["max_drawdown_pct"] for s in tr]
    import numpy as np
    return {"inner_a_sharpe_mean": float(np.mean(sh)), "gap_mean": float(np.mean(gap)), "sharpe_seed_std": float(np.std(sh)),
            "inner_a_dd_pct_mean": float(np.mean(dd)), "all_valid": bool(e["decision"]["means"]["FULL"]["all_valid"]),
            "n_seeds": len(tr), "support": {p: e["meta"]["support_hash"][p] for p in ("TRAIN", "INNER_A")}}


LEGACY_ABSENT, DECLARED, UNKNOWN, MALFORMED, UNAUTH = ("PRODUCER_PREDATES_FIELD", "DECLARED", "UNKNOWN_MODE", "MALFORMED",
                                                      "ABSENT_WITHOUT_AUTHENTICATED_PRODUCER")


def label_mode_of(e: dict, producer: dict | None = None) -> tuple:
    """(mode, how) of an E3M.json (Musashi's C01, 2026-10-22). The `labels` block exists from heuristic-strategy e9291b7
    on. A MISSING key is read as the inherited ZigZag ONLY when ``producer`` is authenticated (receipt -> config identity
    -> pinned tools/expert_guided_e3m.py == the blob of the science commit) and that blob does not emit the field;
    otherwise the absence is UNAUTH. An explicit null or a non-object is MALFORMED; an unknown mode is UNKNOWN_MODE."""
    if "labels" not in e:
        if producer and producer.get("authenticated") and not producer.get("emits_labels"):
            return "zigzag", LEGACY_ABSENT
        return "absent", UNAUTH
    lab = e["labels"]
    if not isinstance(lab, dict):
        return type(lab).__name__ if lab is not None else "null", MALFORMED
    mode = lab.get("mode")
    if not isinstance(mode, str):
        return str(mode), MALFORMED
    if producer and producer.get("authenticated") and not producer.get("emits_labels"):
        return mode, "PRESENT_BUT_PRODUCER_DOES_NOT_EMIT"
    return (mode, DECLARED) if mode in ("zigzag", "opportunity", "both") else (mode, UNKNOWN)


def labels_ok(mode, how) -> bool:
    return mode == "zigzag" and how in (LEGACY_ABSENT, DECLARED)


def authenticate_producer(config_path, science_commit: str, pin_suffix="code/tools/expert_guided_e3m.py") -> dict:
    """Producer provenance from the peer's config (its identity = sha256 of the key-sorted JSON, as its worker computes
    it) and this repository: the pinned E3-M source must equal the blob of ``science_commit``."""
    import subprocess
    cfg = json.loads(Path(config_path).read_text())
    ident = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    pins = cfg.get("pins", {})
    hit = [k for k in pins if k.endswith(pin_suffix)]
    out = {"config_identity": ident, "science_commit": science_commit, "authenticated": False, "emits_labels": None}
    if len(hit) != 1:
        out["error"] = f"PIN_NOT_FOUND:{pin_suffix}"
        return out
    blob = subprocess.run(["git", "show", f"{science_commit}:tools/expert_guided_e3m.py"], cwd=ROOT, capture_output=True)
    if blob.returncode != 0:
        out["error"] = f"SCIENCE_COMMIT_UNKNOWN:{science_commit}"
        return out
    bh = hashlib.sha256(blob.stdout).hexdigest()
    out.update(pinned_e3m_sha256=pins[hit[0]], blob_e3m_sha256=bh, emits_labels=b'"labels"' in blob.stdout,
               archives={Path(k).name: v for k, v in pins.items() if k.endswith(".tar.gz")})
    out["authenticated"] = pins[hit[0]] == bh
    if not out["authenticated"]:
        out["error"] = "PINNED_SOURCE_DIFFERS_FROM_SCIENCE_COMMIT"
    return out


def verify_t2_arm(e: dict, unit: dict, seeds=("20261019", "20261020", "20261021"), producer=None) -> list:
    """Content errors of one T2 E3M.json against its typed unit (T2_UNITS_2026_10_23.json); [] when it conforms.
    Content only: identity is bound by ``verify_t2_receipt`` (Musashi's C02)."""
    errs = []
    if e.get("split_contract", {}).get("arm") != "consolidated":
        errs.append(f"SPLIT:{e.get('split_contract', {}).get('arm')}")
    if e.get("agent_state", {}).get("mode") != unit["agent_state"]:
        errs.append(f"AGENT_STATE:{e.get('agent_state', {}).get('mode')}!={unit['agent_state']}")
    if e.get("exec_contract") != "sac":
        errs.append(f"EXEC_CONTRACT:{e.get('exec_contract')}")
    if float(e.get("min_orders_per_week", -1)) != 2.0:
        errs.append(f"MIN_ORDERS:{e.get('min_orders_per_week')}")
    if e.get("meta", {}).get("scaling", {}).get("policy") != "fixed_train":
        errs.append("SCALING")
    mode, how = label_mode_of(e, producer)
    if not labels_ok(mode, how):
        errs.append(f"LABELS:{mode}:{how}")
    tr = e.get("trials", {}).get("FULL", {})
    if sorted(tr) != sorted(seeds):
        errs.append(f"SEEDS:{sorted(tr)}")
    for s, t in tr.items():
        for p in ("TRAIN", "INNER_A"):
            if p not in t.get("trading", {}):
                errs.append(f"PERIOD_MISSING:{s}:{p}")
    sh = e.get("meta", {}).get("support_hash", {})
    if not all(sh.get(p) for p in ("TRAIN", "INNER_A")):
        errs.append("SUPPORT_HASH_MISSING")
    if "FULL" not in e.get("decision", {}).get("means", {}):
        errs.append("DECISION_MISSING")
    return errs


def verify_t2_receipt(rcpt: dict, unit: dict, arm_dir: Path, e: dict, producer: dict, list_features=None) -> list:
    """Bind one result to its execution receipt (Musashi's C02): config identity, unit name, typed identity recomputed
    from the effective command, regularization / state / list, every sealed file hash (E3M, scaler, probe, the 12
    checkpoints), the support hashes and the consumed columns."""
    errs = []
    job = rcpt.get("job", {})
    if rcpt.get("config_sha256") != producer.get("config_identity"):
        errs.append("RECEIPT_CONFIG_IDENTITY")
    if job.get("name") != unit["name"]:
        errs.append(f"RECEIPT_NAME:{job.get('name')}")
    cmd = job.get("peer_command", "")
    if cmd != unit.get("command"):
        errs.append("RECEIPT_COMMAND_DIFFERS_FROM_UNIT")
    if hashlib.sha256(cmd.encode()).hexdigest()[:16] != unit["identity"] or job.get("peer_command_identity") != unit["identity"]:
        errs.append("RECEIPT_IDENTITY")
    if (float(job.get("dropout", -1)), float(job.get("wd", -1))) != (float(unit["dropout"]), float(unit["wd"])):
        errs.append(f"RECEIPT_REGULARIZATION:{job.get('dropout')}/{job.get('wd')}")
    if job.get("state") != unit["agent_state"]:
        errs.append(f"RECEIPT_STATE:{job.get('state')}")
    if job.get("list") != f"FEATURE_LIST_{unit['list']}.json":
        errs.append(f"RECEIPT_LIST:{job.get('list')}")
    if job.get("split") != "consolidated":
        errs.append(f"RECEIPT_SPLIT:{job.get('split')}")
    hashes = rcpt.get("hashes", {})
    need = {"E3M.json", "SCALER.json", "PROBE.json"} | {f"FULL_{m}_{s}.pt" for m in ("open_l", "open_s", "close_l", "close_s")
                                                          for s in ("20261019", "20261020", "20261021")}
    for n in sorted(need - set(hashes)):
        errs.append(f"RECEIPT_HASH_MISSING:{n}")
    for n, h in hashes.items():
        f = arm_dir / n
        if not f.exists():
            errs.append(f"ARTIFACT_MISSING:{n}")
        elif sha(f) != h:
            errs.append(f"ARTIFACT_HASH:{n}")
    rs = rcpt.get("support_hash", {})
    es = e.get("meta", {}).get("support_hash", {})
    if any(rs.get(p) != es.get(p) for p in ("TRAIN", "INNER_A")):
        errs.append("RECEIPT_SUPPORT_DIFFERS")
    if list_features is not None and (rcpt.get("consumed_features") != list_features or rcpt.get("requested_features") != list_features):
        errs.append("RECEIPT_COLUMNS_DIFFER_FROM_LIST")
    return errs


def cmd_verify_t2(a):
    """Verify every T2 result against the typed unit listing AND its authenticated execution receipt; a marker in a log
    is not a result. Writes the verification document that `regularize` requires."""
    units = json.loads(Path(a.units).read_text())["units"]
    producer = authenticate_producer(a.config, a.science_commit)
    lists = {}
    for u in units:
        lp = Path(a.lists_dir) / f"FEATURE_LIST_{u['list']}.json"
        lists[u["list"]] = json.loads(lp.read_text())["features"] if lp.exists() else None
    rows, bad = [], 0 if producer.get("authenticated") else 1
    support = {}
    for u in units:
        d = Path(a.results) / u["name"]
        f, rp = d / "E3M.json", Path(a.receipts) / f"{u['name']}.RESULT.json"
        if not f.exists() or not rp.exists():
            rows.append({"unit": u["name"], "identity": u["identity"], "status": "MISSING",
                         "errors": [x for x, ok in (("E3M_MISSING", f.exists()), ("RECEIPT_MISSING", rp.exists())) if not ok]})
            bad += 1
            continue
        e, rc = json.loads(f.read_text()), json.loads(rp.read_text())
        errs = verify_t2_arm(e, u, producer=producer) + verify_t2_receipt(rc, u, d, e, producer, lists[u["list"]])
        if lists[u["list"]] is None:
            errs.append(f"LIST_FILE_MISSING:{u['list']}")
        support.setdefault((u["list"], u["agent_state"]), set()).add(json.dumps({p: e.get("meta", {}).get("support_hash", {}).get(p)
                                                                                 for p in ("TRAIN", "INNER_A")}, sort_keys=True))
        rows.append({"unit": u["name"], "identity": u["identity"], "sha256": sha(f), "receipt_sha256": sha(rp),
                     "labels": "/".join(label_mode_of(e, producer)), "support": e.get("meta", {}).get("support_hash", {}).get("INNER_A"),
                     "status": "OK" if not errs else "REJECTED", "errors": errs})
        bad += int(bool(errs))
    mixed = [f"{k[0]}:{k[1]}" for k, v in support.items() if len(v) != 1]
    if mixed:
        bad += 1
    doc = {"units": a.units, "units_sha256": sha(a.units), "results": a.results, "producer": producer,
           "support_mismatch_within_list_and_state": mixed, "rows": rows, "all_ok": bad == 0}
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
    return 0 if bad == 0 else 2


def cmd_regularize(a):
    """Apply REG_RULE to the T2 arms (results/gap_<list>_none_<setting>/E3M.json); --apply writes the choice into the spec."""
    import numpy as np
    spec = json.loads(Path(a.spec).read_text())
    ver = json.loads(Path(a.verified).read_text())
    if not ver.get("all_ok"):
        raise SystemExit("T2_NOT_VERIFIED: the verification document is not all_ok")
    vrow = {r["unit"]: r for r in ver["rows"]}
    producer = ver["producer"]
    cands = spec["regularization"]["candidates"]
    lists = a.lists.split(",")
    table, vec = {}, {s: {} for s in cands}
    for lst in lists:
        for s in cands:
            f = Path(a.results) / f"gap_{lst}_none_{s}" / "E3M.json"
            if not f.exists():
                raise SystemExit(f"ARM_NOT_FINISHED:gap_{lst}_none_{s}")
            u = f"gap_{lst}_none_{s}"
            if u not in vrow or vrow[u]["status"] != "OK" or vrow[u]["sha256"] != sha(f):
                raise SystemExit(f"ARM_NOT_THE_VERIFIED_RESULT:{u}")
            e = json.loads(f.read_text())
            if e["agent_state"]["mode"] != "none":
                raise SystemExit(f"ARM_AGENT_STATE:{lst}:{s}:{e['agent_state']['mode']}")
            if e["split_contract"]["arm"] != spec["split_contract"]:
                raise SystemExit(f"ARM_CONTRACT:{lst}:{s}:{e['split_contract']['arm']}")
            if e.get("exec_contract") != spec["exec_contract"] or float(e.get("min_orders_per_week", -1)) != float(spec["min_orders_per_week"]):
                raise SystemExit(f"ARM_BUSINESS_RULES:{lst}:{s}")
            mode, how = label_mode_of(e, producer)
            if not labels_ok(mode, how):
                raise SystemExit(f"ARM_LABELS:{lst}:{s}:{mode}:{how}")
            table[(lst, s)] = _arm_vector(e)
            table[(lst, s)]["labels"] = f"{mode}/{how}"
        sup = {s: table[(lst, s)]["support"] for s in cands}
        if len({json.dumps(v, sort_keys=True) for v in sup.values()}) != 1:
            raise SystemExit(f"SUPPORT_MISMATCH:{lst}:{sup}")
    valid_lists = {s: [l for l in lists if table[(l, s)]["all_valid"]] for s in cands}
    common = [l for l in lists if all(table[(l, s)]["all_valid"] for s in cands)]
    doc = {"rule": REG_RULE, "lists": lists, "candidates": cands,
           "arms": {f"{l}:{s}": {k: v for k, v in table[(l, s)].items()} for (l, s) in table},
           "valid_lists": valid_lists, "compared_lists": common}
    if not common:
        chosen, how = "base", "DEFAULT_BY_VALIDITY"
    else:
        for s in cands:
            vec[s] = {l: (table[(l, s)]["inner_a_sharpe_mean"], -table[(l, s)]["gap_mean"], -table[(l, s)]["sharpe_seed_std"],
                          -table[(l, s)]["inner_a_dd_pct_mean"]) for l in common}

        def dominates(t, s):
            ge = all(vec[t][l][k] >= vec[s][l][k] for l in common for k in range(4))
            gt = any(vec[t][l][k] > vec[s][l][k] for l in common for k in range(4))
            return ge and gt
        nd = [s for s in cands if not any(dominates(t, s) for t in cands if t != s)]
        doc["non_dominated"] = nd
        if len(nd) == 1:
            chosen, how = nd[0], "NON_DOMINATED_UNIQUE"
        else:
            ms = {s: float(np.mean([table[(l, s)]["inner_a_sharpe_mean"] for l in common])) for s in nd}
            pooled = float(np.sqrt(np.mean([table[(l, s)]["sharpe_seed_std"] ** 2 for l in common for s in nd])))
            best = max(nd, key=lambda s: ms[s])
            if all(ms[best] - ms[s] > pooled for s in nd if s != best):
                chosen, how = best, f"TIE_BREAK_1_SHARPE_BEYOND_POOLED_SEED_STD({pooled:.3f})"
            else:
                mg = {s: float(np.mean([table[(l, s)]["gap_mean"] for l in common])) for s in nd}
                lo = min(mg.values())
                mins = [s for s in nd if mg[s] == lo]
                chosen, how = (mins[0], "TIE_BREAK_2_SMALLER_GAP") if len(mins) == 1 else ("base", "TIE_BREAK_3_BASE_CONTROL")
    doc.update(chosen=chosen, how=how, values=cands[chosen], never_read=["OBS_2024", "INNER_B", "TEST"],
               verification={"file": Path(a.verified).name, "sha256": sha(a.verified), "producer": producer})
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("chosen", "how", "values", "compared_lists")}, indent=1))
    if a.apply:
        if how == "DEFAULT_BY_VALIDITY":
            raise SystemExit("REGULARIZATION_DEFAULT_BY_VALIDITY_NOT_APPLIED")
        spec["regularization"].update(dropout=cands[chosen]["dropout"], wd=cands[chosen]["wd"], chosen=chosen,
                                      source=f"T2 decision {Path(a.out).name} ({how}); sha256 {sha(a.out)[:16]}", status="DECIDED")
        Path(a.spec).write_text(json.dumps(spec, indent=1) + "\n")
        print(f"spec updated: {a.spec}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--spec", required=True)
    e = sub.add_parser("emit"); e.add_argument("--spec", required=True); e.add_argument("--stage", default="A")
    e.add_argument("--winner-from", default=None); e.add_argument("--out", required=True)
    d = sub.add_parser("decide"); d.add_argument("--spec", required=True); d.add_argument("--stage", required=True)
    d.add_argument("--results", required=True); d.add_argument("--out", required=True)
    r = sub.add_parser("regularize", help="apply REG_RULE to the T2 arms"); r.add_argument("--spec", required=True)
    r.add_argument("--results", required=True); r.add_argument("--out", required=True)
    r.add_argument("--lists", default="core15,shortlist_strict"); r.add_argument("--apply", action="store_true")
    r.add_argument("--verified", required=True, help="T2_VERIFY.json from verify-t2 (all_ok); every arm must be the verified file")
    v = sub.add_parser("verify-t2", help="verify every T2 E3M.json against the typed unit listing")
    v.add_argument("--units", required=True); v.add_argument("--results", required=True); v.add_argument("--out", default=None)
    v.add_argument("--receipts", required=True, help="directory with the peer's <unit>.RESULT.json receipts")
    v.add_argument("--config", required=True, help="the peer's run config (its identity = sha256 of the key-sorted JSON)")
    v.add_argument("--science-commit", required=True, help="commit whose tools/expert_guided_e3m.py the config pins")
    v.add_argument("--lists-dir", required=True, help="directory with FEATURE_LIST_<list>.json")
    a = ap.parse_args()
    sys.exit({"check": cmd_check, "emit": cmd_emit, "decide": cmd_decide, "regularize": cmd_regularize,
              "verify-t2": cmd_verify_t2}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
