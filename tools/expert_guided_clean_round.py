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


def load_recipe(spec: dict) -> dict:
    return json.loads(Path(spec["recipe"]["path"]).read_text())


def arm_modules(arm) -> dict:
    """module -> (n_cls, n_reg) of the FULL variant; the aux heads add opportunity / hold-gain regressions."""
    return {"open_l": (2, 3), "open_s": (2, 3), "close_l": (2, 1), "close_s": (2, 1)} if arm.get("aux_targets") else dict(E3M_MODULES)


def arm_params(arm, spec, rec):
    """Parameters per module of an arm on the ACCEPTED recipe (columns, window, resolved groups and model all come
    from the recipe; the arm only chooses training-side treatments and, for aux heads, extra regression outputs)."""
    from app.short_horizon.expert_guided.temporal import TemporalPolicyNet
    r, m = rec["representation"]["resolved"], rec["representation"]["model"]
    n, win = rec["list"]["n"], int(rec["tensor"]["window"]["observations"])
    kw = dict(units=m["units"], core=m.get("core"), depth=m["depth"], head=m["head"], groups=r["indices"], group_units=r["units"],
              group_names=r["names"], group_kinds=r["kinds"])
    return {mod: int(sum(p.numel() for p in TemporalPolicyNet(n, win, nc, n_reg=nr, n_state=9, **kw).parameters()))
            for mod, (nc, nr) in arm_modules(arm).items()}


LABEL_MODES, CLOSER_RULES = ("zigzag", "opportunity", "both", "multires"), ("hold_through", "optimum")
FORBIDDEN_FLAGS = ("--feature-list", "--groups", "--window", "--stride", "--eligible-from")


def multires_arm_errors(a: dict, tag: str) -> list:
    """Multi-resolution arms (owner 2026-10-24, design §4): thresholds and sample weights exist only in that label
    mode; the closer is the multires rule (no ``closer_rule`` key); a thresholds file must exist, parse as
    ``multires.Thresholds`` and match its declared digest, so the DEAP choice is pinned, not retyped."""
    errs = []
    mr = a.get("labels", "zigzag") == "multires"
    if a.get("multires_thresholds") and not mr:
        errs.append(f"THRESHOLDS_WITHOUT_MULTIRES:{tag}")
    if a.get("sample_weights") and not mr:
        errs.append(f"SAMPLE_WEIGHTS_WITHOUT_MULTIRES:{tag}")
    if mr and "closer_rule" in a:
        errs.append(f"CLOSER_RULE_WITH_MULTIRES:{tag}")
    if a.get("multires_thresholds"):
        tp = Path(a["multires_thresholds"])
        if not tp.exists():
            errs.append(f"THRESHOLDS_FILE_MISSING:{tag}")
        else:
            try:
                from app.short_horizon.expert_guided import multires as MR
                MR.Thresholds.from_json(tp.read_text())
            except Exception as ex:                       # noqa: BLE001 -- any parse/field error is a refusal
                errs.append(f"THRESHOLDS_INVALID:{tag}:{type(ex).__name__}")
            if a.get("multires_thresholds_sha256") and sha(tp) != a["multires_thresholds_sha256"]:
                errs.append(f"THRESHOLDS_DIGEST_MISMATCH:{tag}")
    return errs


def validate(spec: dict, local_root: Path | None = None) -> list:
    """Refuse every configuration the owner forbade and every drift from the ACCEPTED shared recipe (Musashi's
    re-review 2026-10-24): the launcher consumes the recipe and varies only training-side treatments."""
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
    # --- the accepted recipe binds store, columns, scaler, window, groups, model, regularization and split
    rp = Path(spec.get("recipe", {}).get("path", ""))
    if not rp.exists():
        errs.append("RECIPE_FILE_MISSING")
        return errs
    if sha(rp) != spec["recipe"].get("sha256"):
        errs.append("RECIPE_DIGEST_MISMATCH")
    rec = json.loads(rp.read_text())
    if rec.get("identity") != spec["recipe"].get("identity"):
        errs.append("RECIPE_IDENTITY_MISMATCH")
    if rec.get("schema") != "expert_guided_consumer_recipe.v2" or rec.get("scientific_release") is not False:
        errs.append("RECIPE_SCHEMA")
    rr = rec.get("regularization", {})
    if reg.get("dropout") is not None and (float(reg["dropout"]), float(reg["wd"])) != (float(rr.get("dropout", -1)), float(rr.get("wd", -1))):
        errs.append("REGULARIZATION_DIFFERS_FROM_RECIPE")
    if spec["store"]["sha256"] != rec.get("store", {}).get("sha256") or Path(spec["store"]["path"]).name != rec["store"]["file"]:
        errs.append("STORE_DIFFERS_FROM_RECIPE")
    store = Path(spec["store"]["path"])
    if store.exists() and sha(store) != spec["store"]["sha256"]:
        errs.append("STORE_DIGEST_MISMATCH")
    rm = rec.get("representation", {}).get("model", {})
    if (m["units"], m["depth"], m["head"], m.get("core")) != (rm.get("units"), rm.get("depth"), rm.get("head"), rm.get("core")):
        errs.append("MODEL_DIFFERS_FROM_RECIPE")
    if rec.get("split", {}).get("arm") != spec["split_contract"]:
        errs.append("RECIPE_SPLIT_MISMATCH")
    win = int(rec.get("tensor", {}).get("window", {}).get("observations", 0))
    if receptive_field(m["depth"]) < win:
        errs.append(f"RECEPTIVE_FIELD_BELOW_WINDOW:{receptive_field(m['depth'])}<{win}")
    if (rec.get("business", {}).get("evaluation_and_policy", {}).get("max_entries_per_day") != 1
            or float(rec.get("business", {}).get("evaluation_and_policy", {}).get("min_orders_per_week", 0)) != 2.0):
        errs.append("RECIPE_BUSINESS_CONTRACT")
    # --- arms: training-side treatments only
    adm = {}
    dem = Path(spec["demonstrators"]["file"])
    if dem.exists():
        adm = {str(d["spec"]): bool(d.get("admitted")) for d in json.loads(dem.read_text())["demonstrators"]}
    for stage, arms in spec["stages"].items():
        if not any(a.get("agent_state", "none") == "none" for a in arms):
            errs.append(f"STAGE_WITHOUT_NO_FEEDBACK_ARM:{stage}")
        for a in arms:
            tag = f"{stage}/{a['name']}"
            if stage.startswith("A") and a.get("agent_state", "none") != "none":     # A and A_prime: no agent feedback
                errs.append(f"AGENT_STATE_FORBIDDEN:{tag}")
            if a.get("agent_state", "none") not in ("none", "minimal"):
                errs.append(f"AGENT_STATE_UNKNOWN:{tag}")
            if a.get("labels", "zigzag") not in LABEL_MODES:
                errs.append(f"LABELS_MODE_UNKNOWN:{tag}")
            if a.get("closer_rule", "hold_through") not in CLOSER_RULES:
                errs.append(f"CLOSER_RULE_UNKNOWN:{tag}")
            for k in ("list", "groups", "window", "extractor"):
                if a.get(k) is not None:
                    errs.append(f"ARM_OVERRIDES_RECIPE:{tag}:{k}")
            if a.get("aux_targets") not in (None, False, True) and a.get("aux_targets") not in AUX_MODES:
                errs.append(f"AUX_TARGETS_UNKNOWN:{tag}:{a.get('aux_targets')}")
            errs += multires_arm_errors(a, tag)
            if a.get("extra_experts"):
                miss = [d for d in a["extra_experts"] if not adm.get(str(d))]
                if miss:
                    errs.append(f"DEMONSTRATOR_NOT_ADMITTED:{tag}:{miss}")
                longer = [d for d in a["extra_experts"] if not str(d).startswith("rev") and int(d) >= 3]
                if longer:
                    errs.append(f"DEMONSTRATOR_NOT_SHORTER_THAN_OPTIMUM:{tag}:{longer}")
    for stage, arms in spec["stages"].items():
        for a in arms:
            cmd = " ".join(e3m_cmd(spec, a, "OUT", "ROOT"))
            if "--recipe" not in cmd or any(f in cmd for f in FORBIDDEN_FLAGS) or "rolling" in cmd:
                errs.append(f"COMMAND_NOT_RECIPE_BOUND:{stage}/{a['name']}")
    errs += walk_forward_errors(spec, rec)
    return errs


WF_COPIED_KEYS = ("store", "list", "tensor", "representation", "regularization", "business", "labels")
WF_STAGE = "WF"


def walk_forward_errors(spec: dict, rec: dict) -> list:
    """The diagnostic walk-forward stage (owner 2026-10-25) runs stage-A arms on DERIVED recipes: one per fold, equal to
    the accepted recipe on every identity key except the scaler constants (re-fitted on the fold's fit block) and the
    split (a declared ``wf*`` contract). Each fold recipe is pinned by digest and identity and must declare its base; any
    other divergence is refused, so the stage can never smuggle a different representation."""
    wf = spec.get("walk_forward")
    if not wf:
        return []
    errs = []
    from app.short_horizon.expert_guided import data as DG
    names = {x["name"] for x in spec["stages"].get("A", [])}
    for arm in wf.get("arms", []):
        if arm not in names:
            errs.append(f"WF_ARM_UNKNOWN:{arm}")
    if not wf.get("seeds"):
        errs.append("WF_SEEDS_MISSING")
    if not wf.get("rule"):
        errs.append("WF_RULE_NOT_PREREGISTERED")
    for fold, fd in wf.get("folds", {}).items():
        tag = f"WF/{fold}"
        if fold not in getattr(DG, "WALK_FORWARD_FOLDS", ()):
            errs.append(f"WF_FOLD_UNKNOWN:{tag}")
            continue
        fp = Path(fd.get("recipe", {}).get("path", ""))
        if not fp.exists():
            errs.append(f"WF_FOLD_RECIPE_MISSING:{tag}")
            continue
        if sha(fp) != fd["recipe"].get("sha256"):
            errs.append(f"WF_FOLD_DIGEST_MISMATCH:{tag}")
        fr = json.loads(fp.read_text())
        if fr.get("identity") != fd["recipe"].get("identity"):
            errs.append(f"WF_FOLD_IDENTITY_MISMATCH:{tag}")
        if fr.get("schema") != "expert_guided_consumer_recipe.v2" or fr.get("scientific_release") is not False:
            errs.append(f"WF_FOLD_SCHEMA:{tag}")
        d = fr.get("derived_from") or {}
        if d.get("identity") != rec.get("identity") or d.get("sha256") != spec["recipe"].get("sha256") or d.get("fold") != fold:
            errs.append(f"WF_FOLD_NOT_DERIVED_FROM_RECIPE:{tag}")
        if fr.get("split", {}).get("arm") != fold:
            errs.append(f"WF_FOLD_SPLIT_MISMATCH:{tag}")
        for k in WF_COPIED_KEYS:
            if fr.get(k) != rec.get(k):
                errs.append(f"WF_FOLD_DIVERGES:{tag}:{k}")
        sc = fr.get("scaler", {})
        if list(sc.get("fit_period", [])) != list(DG.SPLIT_CONTRACTS[fold]["scaler_fit"]) or sc.get("features") != rec.get("scaler", {}).get("features"):
            errs.append(f"WF_FOLD_SCALER_NOT_FOLD_LOCAL:{tag}")
    return errs


AUX_MODES = ("opportunity", "multires")       # --aux-targets values (True = the inherited opportunity / hold-gain heads)


def gates_open(spec) -> list:
    missing = []
    for g in spec["gates"]:
        p = Path(g["path"])
        if not p.exists() or (g.get("must_contain") and g["must_contain"] not in p.read_text()):
            missing.append(g)
    return missing


def e3m_cmd(spec, arm, out, remote, recipe_root=None, recipe_file=None, split_contract=None):
    """The pinned E3-M command of one arm: the recipe supplies columns, consumer operations, scaler constants, window,
    groups and model (E3-M refuses contradicting flags); the arm supplies only training-side treatments. ``remote`` is
    the data root (admitted / sel15 / store); ``recipe_root`` (default ``remote``) holds ``recipe/`` — the staged
    package keeps the recipe beside its digest at ``<root>/recipe/``, not under ``<root>/data/``."""
    m, reg, rec = spec["model"], spec["regularization"], spec["recipe"]
    rr = remote if recipe_root is None else recipe_root
    rname = Path(rec["path"]).name if recipe_file is None else Path(recipe_file).name     # WF: the fold's derived recipe
    cmd = ["tools/expert_guided_e3m.py", "run", "--admitted", f"{remote}/admitted", "--csv-dir", f"{remote}/sel15",
           "--store", f"{remote}/store/{Path(spec['store']['path']).name}", "--recipe", f"{rr}/recipe/{rname}",
           "--out", out, "--variants", "FULL", "--units", str(m["units"]), "--depth", str(m["depth"]), "--head", str(m["head"]),
           "--dropout", str(reg["dropout"]), "--wd", str(reg["wd"]), "--max-epochs", str(m["max_epochs"]),
           "--scaling", spec["scaling"], "--nonstationary", spec["nonstationary"], "--split-contract", split_contract or spec["split_contract"],
           "--agent-state", arm.get("agent_state", "none"), "--exec-contract", spec["exec_contract"],
           "--min-orders-per-week", str(spec["min_orders_per_week"]),
           "--labels", arm.get("labels", "zigzag")]
    if arm.get("labels", "zigzag") != "multires":                 # the multires closer is its own rule (remaining / f_close)
        cmd += ["--closer-rule", arm.get("closer_rule", "hold_through")]
    if arm.get("aux_targets"):
        cmd += ["--aux-targets"] + ([str(arm["aux_targets"])] if isinstance(arm["aux_targets"], str) else [])
    if arm.get("multires_thresholds"):
        cmd += ["--multires-thresholds", f"{rr}/recipe/{Path(arm['multires_thresholds']).name}"]
    if arm.get("sample_weights"):
        cmd += ["--sample-weights"]
    if m.get("core"):
        cmd += ["--core", str(m["core"])]
    if arm.get("extra_experts"):
        cmd += ["--extra-experts", ",".join(str(d) for d in arm["extra_experts"])]
    return cmd


def cmd_check(a):
    spec = json.loads(Path(a.spec).read_text())
    errs = validate(spec)
    rows, probe = [], None
    if not errs:
        rec = load_recipe(spec)
        for stage, arms in spec["stages"].items():
            for arm in arms:
                p = arm_params(arm, spec, rec)
                rows.append({"stage": stage, "arm": arm["name"], "labels": arm.get("labels", "zigzag"), "closer_rule": arm.get("closer_rule", "hold_through"),
                             "agent_state": arm.get("agent_state", "none"), "experts": ["3"] + [str(x) for x in (arm.get("extra_experts") or [])],
                             "aux_targets": bool(arm.get("aux_targets")), "params_per_module": p, "params_total": sum(p.values()),
                             "receptive_field": receptive_field(spec["model"]["depth"]), "command": " ".join(e3m_cmd(spec, arm, "<out>", "<root>"))})
        if a.probe_reference:
            # proof of no refit / implicit transform: the loader through the recipe reproduces the accepted reference probe
            from tools import consumer_probe as CP
            _, doc = CP.reference(spec["recipe"]["path"], a.admitted, spec["store"]["path"], verify=True)
            ref = json.loads(Path(a.probe_reference).read_text())
            probe = {"reference": Path(a.probe_reference).name, "equal": not CP.compare(ref, doc), "errors": CP.compare(ref, doc),
                     "tensor_sha256": doc["tensor_sha256"], "recipe": doc["recipe"]}
            if probe["errors"]:
                errs.append("PROBE_DIFFERS_FROM_REFERENCE")
    print(json.dumps({"valid": not errs, "errors": errs, "gates_missing": [g["path"] for g in gates_open(spec)],
                      "recipe": spec.get("recipe", {}).get("identity"), "arms": rows, "probe": probe}, indent=1))
    return 0 if not errs else 2


def cmd_emit(a):
    spec = json.loads(Path(a.spec).read_text())
    errs = validate(spec)
    if errs:
        raise SystemExit(f"SPEC_INVALID:{errs}")
    miss = gates_open(spec)
    if miss:
        raise SystemExit(f"GATES_CLOSED:{[g['path'] for g in miss]}")
    if a.stage == WF_STAGE:
        return emit_walk_forward(spec, a)
    arms = spec["stages"][a.stage]
    if a.stage == "B":                                            # A and A_prime carry their own labels; B inherits the winner
        dec = json.loads(Path(a.winner_from).read_text())
        win = next(x for x in spec["stages"]["A"] if x["name"] == dec["chosen"])
        arms = [dict(win, **{k: v for k, v in x.items() if v is not None and k != "name"}, name=x["name"]) for x in arms]
    only = [x.strip() for x in (getattr(a, "arms", None) or "").split(",") if x.strip()]
    if only:                                                      # seed extension / rerun of named arms only
        unknown = [n for n in only if n not in {x["name"] for x in arms}]
        if unknown:
            raise SystemExit(f"ARM_UNKNOWN:{unknown}")
        arms = [x for x in arms if x["name"] in only]
    seeds = getattr(a, "seeds", None) or ""
    suffix = getattr(a, "out_suffix", None) or ""
    if seeds and not suffix:
        raise SystemExit("SEEDS_REQUIRE_OUT_SUFFIX: new seeds never overwrite an arm's finished output")
    R = spec["remote_root"]
    gate = Path(spec["gates"][0]["path"]).name
    rec_name = Path(spec["recipe"]["path"]).name
    L = ["#!/bin/bash", f"# clean round {spec['identity']} stage {a.stage}: sequential, resumable (STATE markers); generated by "
         "tools/expert_guided_clean_round.py emit; the recipe digest and the launch gate are re-checked before every unit",
         f"Q={R}; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run", "mkdir -p $Q/STATE $Q/out",
         'log(){ echo "$(date -u +%FT%TZ) [' + a.stage + '] $*" >> $Q/LOG; }',
         f'test -f $Q/{gate} || {{ log "launch gate {gate} missing: refusing"; exit 3; }}',
         f'echo "{spec["recipe"]["sha256"]}  $Q/recipe/{rec_name}" | sha256sum -c --quiet || {{ log "recipe digest differs: refusing"; exit 4; }}',
         "cd $Q/code || exit 1"]
    for arm in arms:
        unit = f"{a.stage}_{arm['name']}{suffix}"
        out = f"$Q/out/{unit}"
        cmd = " ".join(e3m_cmd(spec, arm, out, "$Q/data", recipe_root="$Q") + (["--seeds", seeds] if seeds else []))
        L += [f'if [ ! -f $Q/STATE/{unit} ]; then',
              f'  echo "{spec["recipe"]["sha256"]}  $Q/recipe/{rec_name}" | sha256sum -c --quiet || {{ log "recipe digest differs before {unit}: refusing"; exit 4; }}',
              f'  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=${{GPU:-0}} $CR -q -m {spec["memory"]} -t 43200s -n clean-{unit} -- $PY {cmd} > {out}.log 2>&1 && {{ touch $Q/STATE/{unit}; log "{unit} DONE"; break; }}',
              f'    grep -q "REFUSED\\|QUEUED HOST_HEADROOM" {out}.log && {{ log "{unit} not admitted (try $try)"; sleep 600; }} || {{ log "{unit} FAILED"; break; }}',
              "  done", "fi"]
    L.append(f'log "STAGE_{a.stage}_DONE"')
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"written {a.out} ({len(arms)} arms)")


def emit_walk_forward(spec, a):
    """Stage WF: fold x arm units on the fold's derived recipe (its own digest re-checked before every unit), the fold's
    split contract and the pre-registered seeds; same gate, caps, STATE marks and retries as the other stages."""
    wf = spec["walk_forward"]
    R = spec["remote_root"]
    gate = Path(spec["gates"][0]["path"]).name
    byname = {x["name"]: x for x in spec["stages"]["A"]}
    L = ["#!/bin/bash", f"# clean round {spec['identity']} stage WF (walk-forward inside development): sequential, resumable (STATE markers); "
         "generated by tools/expert_guided_clean_round.py emit; each fold's derived recipe digest and the launch gate are re-checked before every unit",
         f"Q={R}; PY=~/anaconda3/envs/trading-stack/bin/python; CR=~/.local/bin/crispdm-run", "mkdir -p $Q/STATE $Q/out",
         'log(){ echo "$(date -u +%FT%TZ) [WF] $*" >> $Q/LOG; }',
         f'test -f $Q/{gate} || {{ log "launch gate {gate} missing: refusing"; exit 3; }}',
         "cd $Q/code || exit 1"]
    for fold, fd in wf["folds"].items():
        rname = Path(fd["recipe"]["path"]).name
        for name in wf["arms"]:
            arm = byname[name]
            unit = f"WF_{fold}_{name}"
            out = f"$Q/out/{unit}"
            cmd = " ".join(e3m_cmd(spec, arm, out, "$Q/data", recipe_root="$Q", recipe_file=rname, split_contract=fold) + ["--seeds", str(wf["seeds"])])
            L += [f'if [ ! -f $Q/STATE/{unit} ]; then',
                  f'  echo "{fd["recipe"]["sha256"]}  $Q/recipe/{rname}" | sha256sum -c --quiet || {{ log "fold recipe digest differs before {unit}: refusing"; exit 4; }}',
                  f'  for try in $(seq 1 72); do CUDA_VISIBLE_DEVICES=${{GPU:-0}} $CR -q -m {wf.get("memory", spec["memory"])} -t 43200s -n clean-{unit} -- $PY {cmd} > {out}.log 2>&1 && {{ touch $Q/STATE/{unit}; log "{unit} DONE"; break; }}',
                  f'    grep -q "REFUSED\\|QUEUED HOST_HEADROOM" {out}.log && {{ log "{unit} not admitted (try $try)"; sleep 600; }} || {{ log "{unit} FAILED"; break; }}',
                  "  done", "fi"]
    L.append('log "STAGE_WF_DONE"')
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"written {a.out} ({len(wf['folds']) * len(wf['arms'])} units)")


def cmd_decide(a):
    """Frozen rule: highest seed-mean INNER_A Sharpe (consolidated inner block) among arms whose seeds are all valid
    (the three of the run, plus any --pool-suffix seed extension of the same recipe and support); ties -> fewer
    parameters, then the declared control. Reads only INNER_A fields; 2024 is never consulted for a choice."""
    spec = json.loads(Path(a.spec).read_text())
    only = [x.strip() for x in (getattr(a, "arms", None) or "").split(",") if x.strip()]
    pool = [x.strip() for x in (getattr(a, "pool_suffix", None) or "").split(",") if x.strip()]
    control = getattr(a, "tie_control", None) or spec.get("decision_control")
    res = []
    for arm in spec["stages"][a.stage]:
        if only and arm["name"] not in only:
            continue
        f = Path(a.results) / f"{a.stage}_{arm['name']}" / "E3M.json"
        if not f.exists():
            raise SystemExit(f"ARM_NOT_FINISHED:{arm['name']}")
        e = json.loads(f.read_text())
        tr = dict(e["trials"]["FULL"])
        for suf in pool:                                          # seed extension: pooled with the arm's finished seeds
            fp = Path(a.results) / f"{a.stage}_{arm['name']}{suf}" / "E3M.json"
            if not fp.exists():
                raise SystemExit(f"ARM_NOT_FINISHED:{arm['name']}{suf}")
            ep = json.loads(fp.read_text())
            if ep.get("recipe") != e.get("recipe") or ep["meta"]["support_hash"] != e["meta"]["support_hash"]:
                raise SystemExit(f"POOL_MISMATCH:{arm['name']}{suf}")
            dup = set(ep["trials"]["FULL"]) & set(tr)
            if dup:
                raise SystemExit(f"POOL_DUPLICATE_SEEDS:{arm['name']}{suf}:{sorted(dup)}")
            tr.update(ep["trials"]["FULL"])
        sh = [tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
        gap = [tr[s]["trading"]["TRAIN"]["sharpe_daily_sqrt365"] - tr[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in tr]
        res.append({"arm": arm["name"], "inner_a_sharpe_mean": float(sum(sh) / len(sh)),
                    "all_valid": all(bool(tr[s]["trading"]["INNER_A"]["valid"]) for s in tr), "n_seeds": len(tr), "seeds": sorted(tr),
                    "params": sum(e["params"]["FULL"].values()), "train_minus_inner_sharpe_mean": float(sum(gap) / len(gap)),
                    "is_control": arm["name"] == control})
    if only and len(res) != len(only):
        raise SystemExit(f"ARM_UNKNOWN:{sorted(set(only) - {r['arm'] for r in res})}")
    ok = [r for r in res if r["all_valid"]] or res
    chosen = max(ok, key=lambda r: (r["inner_a_sharpe_mean"], -r["params"], r["is_control"]))
    doc = {"stage": a.stage, "rule": cmd_decide.__doc__.strip(), "arms": res, "chosen": chosen["arm"],
           "pooled_suffixes": pool, "restricted_to": only or None, "tie_control": control}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


def localize(spec: dict, root: str, thresholds_dir="recipe") -> dict:
    """The spec with every file path rewritten to the staged package at ``root`` (recipe/, data/store/, in/, the gate),
    so ``validate`` / ``emit`` / ``decide`` run on the host that holds the package; digests are untouched, so a package
    whose files differ from the pinned ones is still refused."""
    s = json.loads(json.dumps(spec))
    s["recipe"]["path"] = f"{root}/recipe/{Path(spec['recipe']['path']).name}"
    s["store"]["path"] = f"{root}/data/store/{Path(spec['store']['path']).name}"
    s["demonstrators"]["file"] = f"{root}/in/{Path(spec['demonstrators']['file']).name}"
    s["remote_root"] = root
    for g in s.get("gates", []):
        g["path"] = f"{root}/{Path(g['path']).name}"
    for arms in s["stages"].values():
        for arm in arms:
            if arm.get("multires_thresholds"):
                arm["multires_thresholds"] = f"{root}/{thresholds_dir}/{Path(arm['multires_thresholds']).name}"
    for fd in (s.get("walk_forward") or {}).get("folds", {}).values():
        fd["recipe"]["path"] = f"{root}/recipe/{Path(fd['recipe']['path']).name}"
    return s


def cmd_localize(a):
    spec = json.loads(Path(a.spec).read_text())
    Path(a.out).write_text(json.dumps(localize(spec, a.root), indent=1))
    print(f"written {a.out}")


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
    return (mode, DECLARED) if mode in LABEL_MODES else (mode, UNKNOWN)


def labels_ok(mode, how) -> bool:
    return mode == "zigzag" and how in (LEGACY_ABSENT, DECLARED)


def authenticate_producer(config_path, science_commit: str, pin_suffix="code/tools/expert_guided_e3m.py", blobs=None) -> dict:
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
    # the science source: git when this host has the repository; otherwise the blob exported from git by the session
    # (science_blobs/<commit>/tools/expert_guided_e3m.py, shipped with the package; its digest must equal the config's
    # pin, so a wrong blob fails exactly like a wrong pin)
    src = None
    blob = subprocess.run(["git", "show", f"{science_commit}:tools/expert_guided_e3m.py"], cwd=ROOT, capture_output=True)
    if blob.returncode == 0:
        data, src = blob.stdout, "git"
    elif blobs and (Path(blobs) / science_commit / "tools" / "expert_guided_e3m.py").exists():
        data, src = (Path(blobs) / science_commit / "tools" / "expert_guided_e3m.py").read_bytes(), "exported_blob"
    else:
        out["error"] = f"SCIENCE_COMMIT_UNKNOWN:{science_commit}"
        return out
    bh = hashlib.sha256(data).hexdigest()
    out.update(pinned_e3m_sha256=pins[hit[0]], blob_e3m_sha256=bh, emits_labels=b'"labels"' in data, science_source=src,
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
    cfgs, commits = list(a.config), list(a.science_commit)
    if len(cfgs) != len(commits):
        raise SystemExit("VERIFY_CONFIGS_AND_COMMITS_MUST_PAIR")
    producers = {}
    for cp, sc_ in zip(cfgs, commits):                    # several producers (the peer worker, this lane) may seal units
        pr = authenticate_producer(cp, sc_, blobs=a.science_blobs)
        producers[pr["config_identity"]] = pr
    producer = next(iter(producers.values()))
    lists = {}
    for u in units:
        lp = Path(a.lists_dir) / f"FEATURE_LIST_{u['list']}.json"
        lists[u["list"]] = json.loads(lp.read_text())["features"] if lp.exists() else None
    rows, bad = [], sum(0 if pr.get("authenticated") else 1 for pr in producers.values())
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
        producer = producers.get(rc.get("config_sha256"), {"authenticated": False, "config_identity": None})
        errs = verify_t2_arm(e, u, producer=producer) + verify_t2_receipt(rc, u, d, e, producer, lists[u["list"]])
        if producer.get("config_identity") is None:
            errs.append("RECEIPT_CONFIG_UNKNOWN")
        if lists[u["list"]] is None:
            errs.append(f"LIST_FILE_MISSING:{u['list']}")
        support.setdefault((u["list"], u["agent_state"]), set()).add(json.dumps({p: e.get("meta", {}).get("support_hash", {}).get(p)
                                                                                 for p in ("TRAIN", "INNER_A")}, sort_keys=True))
        rows.append({"unit": u["name"], "identity": u["identity"], "sha256": sha(f), "receipt_sha256": sha(rp),
                     "producer": {k: producer.get(k) for k in ("config_identity", "science_commit", "authenticated", "emits_labels")},
                     "labels": "/".join(label_mode_of(e, producer)), "support": e.get("meta", {}).get("support_hash", {}).get("INNER_A"),
                     "status": "OK" if not errs else "REJECTED", "errors": errs})
        bad += int(bool(errs))
    mixed = [f"{k[0]}:{k[1]}" for k, v in support.items() if len(v) != 1]
    if mixed:
        bad += 1
    doc = {"units": a.units, "units_sha256": sha(a.units), "results": a.results, "producer": producer, "producers": list(producers.values()),
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
            mode, how = label_mode_of(e, vrow[u].get("producer") or producer)
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
    c.add_argument("--probe-reference", default=None, help="accepted reference probe: the loader through the recipe must reproduce it exactly")
    c.add_argument("--admitted", default=None, help="admitted directory for the probe proof")
    e = sub.add_parser("emit"); e.add_argument("--spec", required=True); e.add_argument("--stage", default="A")
    e.add_argument("--winner-from", default=None); e.add_argument("--out", required=True)
    e.add_argument("--arms", default=None, help="comma list: emit only these arms of the stage (seed extension / rerun)")
    e.add_argument("--seeds", default=None, help="comma list of seeds passed to E3-M (--seeds); requires --out-suffix")
    e.add_argument("--out-suffix", default=None, help="suffix of the unit's out dir and STATE mark (e.g. _seeds2)")
    d = sub.add_parser("decide"); d.add_argument("--spec", required=True); d.add_argument("--stage", required=True)
    d.add_argument("--results", required=True); d.add_argument("--out", required=True)
    d.add_argument("--arms", default=None, help="comma list: decide among these arms only (pre-registered restriction)")
    d.add_argument("--pool-suffix", default=None, help="comma list of out-dir suffixes whose seeds are pooled with each arm's run")
    d.add_argument("--tie-control", default=None, help="arm preferred on a full tie (default: the spec's decision_control)")
    lz = sub.add_parser("localize", help="rewrite the spec's file paths to a staged package root"); lz.add_argument("--spec", required=True)
    lz.add_argument("--root", required=True); lz.add_argument("--out", required=True)
    r = sub.add_parser("regularize", help="apply REG_RULE to the T2 arms"); r.add_argument("--spec", required=True)
    r.add_argument("--results", required=True); r.add_argument("--out", required=True)
    r.add_argument("--lists", default="core15,shortlist_strict"); r.add_argument("--apply", action="store_true")
    r.add_argument("--verified", required=True, help="T2_VERIFY.json from verify-t2 (all_ok); every arm must be the verified file")
    v = sub.add_parser("verify-t2", help="verify every T2 E3M.json against the typed unit listing")
    v.add_argument("--units", required=True); v.add_argument("--results", required=True); v.add_argument("--out", default=None)
    v.add_argument("--receipts", required=True, help="directory with the peer's <unit>.RESULT.json receipts")
    v.add_argument("--config", action="append", required=True, help="a producer's run config (identity = sha256 of the key-sorted JSON); repeatable")
    v.add_argument("--science-commit", action="append", required=True, help="the commit each --config pins, in the same order")
    v.add_argument("--lists-dir", required=True, help="directory with FEATURE_LIST_<list>.json")
    v.add_argument("--science-blobs", default=None, help="science_blobs/<commit>/tools/expert_guided_e3m.py exported from git (hosts without the repository)")
    a = ap.parse_args()
    sys.exit({"check": cmd_check, "emit": cmd_emit, "decide": cmd_decide, "regularize": cmd_regularize,
              "verify-t2": cmd_verify_t2, "localize": cmd_localize}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
