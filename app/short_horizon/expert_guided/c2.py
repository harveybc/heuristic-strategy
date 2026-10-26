"""Stage C2 of the causal-ladder design (2026-10-25): learn the CONSEQUENCES of our own actions instead of imitating a
clairvoyant expert (``docs/short_horizon/clean_round/CAUSAL_LADDER_DESIGN_2026_10_25.md`` §3 C2).

Contract (``ACTION_CONTRACT_C2.json``, schema ``expert_guided_action_contract.c2``, identity ``action_<16 hex>`` as v1):

* entry specialists ``open_l`` / ``open_s``: one network per side with
  - a REGRESSION multi-output (order-preserving) of the engine's net result in causal sigma for every (TP, SL) cell of
    the declared grid (``outcomes.CELLS``), each head bounded in the network (lo + (hi - lo) * sigmoid) and fitted with
    Huber on the realized outcome tables (``outcomes.entry_table``: the engine replays every entry);
  - a binary softmax head "TP before SL" at the declared reference cell, Platt-calibrated (reported, and an input of the
    head metrics; the open rule reads the expected values);
* ``nop`` (kept from v1): binary NOP (multires ladder: no resolution finds a trade, or the trade is risky / poor);
* exit specialists ``close_l`` / ``close_s``: a regression of the remaining value of HOLD (net if the position is held to
  TP / SL / forced flat minus net if closed now, sigma) and a binary "SL before TP from here", Platt-calibrated;
* the upper layer (``StrategyC2``): open the side and cell with the highest predicted net when it exceeds
  ``open_margin`` and the NOP veto (when kept) does not fire; CLOSE by optimal stopping when the predicted remaining
  value < -``close_margin`` OR the calibrated P(SL first) > ``close_p``; ``allow_reverse`` permits close-and-reverse.
  The v1 machinery searches it (NSGA-II, Deb constraint domination at 2.0 trades/week, full archive, F0-F6,
  ``require_early_close``) on the calibration block ONLY;
* blocks: INNER_A = early stopping, calibration, strategy optimisation, DAgger visitation strategy; INNER_B = decision
  (scored once after the freeze); OBS_2024 = one read. Hindsight enters only as labels (the outcome tables); there is
  no teacher: the ZigZag expert only supplies VISITED STATES for the round-0 exit rows (plus exploration positions).

Encoding audit (owner 2026-10-25, the failure it guards against: one continuous output cut into intervals to choose
among commands): every categorical decision is ``softmax_ce`` over its declared classes; every continuous output is
``bounded_sigmoid``; an action produced by a continuous head is allowed ONLY through the decision-theoretic rule
"act iff E[value | x, do(action)] > margin" on a head whose role is ``expected_value_sigma`` (one threshold, one action,
never intervals mapped to several actions). Refusals: CATEGORICAL_FROM_CONTINUOUS, CONTINUOUS_THROUGH_SOFTMAX,
HEAD_DIM_MISMATCH, ENCODING_UNKNOWN, CONTINUOUS_BOUNDS, CONTINUOUS_LOSS, ACTION_UNREACHABLE, VALUE_RULE_UNKNOWN,
VALUE_TARGET_NOT_EXPECTED_VALUE, VALUE_TARGET_INTERVALS.

C2G (owner 2026-10-25, "Sí, como brazo extra"; schema ``expert_guided_action_contract.c2g``): C2 plus ONE change, the
closers alone also see the geometry of their own open order (``GEOMETRY``: signed distance to the SL and to the TP in
the order's own sigma, hours held, side), computed after bar b from the order's levels and close[b]. Every other
state slot, for every specialist, is refused (``STATE_SLOT_NOT_ALLOWED``), and so is geometry on an opener or on NOP
(``GEOMETRY_ON_NON_CLOSER``). Because a closer's output now depends on the open order, closer outputs are computed
on demand per (order, row) and cached per order (``CloserGeo``), never precomputed per row; openers and NOP are C2's.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import action_contract as ACT
from . import env as E
from . import multires as MR
from . import outcomes as OC

SCHEMA = "expert_guided_action_contract.c2"
SCHEMA_G = "expert_guided_action_contract.c2g"        # C2 + closer-only order geometry (owner 2026-10-25)
SPECIALISTS = ("open_l", "open_s", "nop", "close_l", "close_s")
OPENERS, CLOSERS = ("open_l", "open_s"), ("close_l", "close_s")
GENES = ("open_margin", "close_margin", "close_p", "allow_reverse", "nop_veto")
# scale-adaptive genes (2026-10-25, after WF_V1's zero trades): rates / quantiles of calibration-block references
GENES_Q = ("open_rate", "close_rem_rate", "close_p_rate", "allow_reverse", "nop_veto_rate")
Q_BOUNDS = {"open_rate": [0.002, 0.30], "close_rem_rate": [0.0, 0.60], "close_p_rate": [0.0, 0.60], "allow_reverse": [0.0, 1.0],
            "nop_veto_rate": [0.0, 0.50]}
# v1.3 (2026-10-25): per-side open rates (each side's predicted-net reference), side cap, activity CI (see action_contract)
GENES_Q13 = ("open_rate_l", "open_rate_s", "close_rem_rate", "close_p_rate", "allow_reverse", "nop_veto_rate")
Q_BOUNDS_13 = {"open_rate_l": [0.002, 0.30], "open_rate_s": [0.002, 0.30], "close_rem_rate": [0.0, 0.60], "close_p_rate": [0.0, 0.60],
               "allow_reverse": [0.0, 1.0], "nop_veto_rate": [0.0, 0.50]}
NET_BOUND = (-4.0, 4.0)
REM_BOUND = (-4.0, 4.0)
CLASSES = {"open_l": ["NOT_TP_FIRST", "TP_FIRST"], "open_s": ["NOT_TP_FIRST", "TP_FIRST"], "nop": ["TRADE", "NOP"],
           "close_l": ["NOT_SL_FIRST", "SL_FIRST"], "close_s": ["NOT_SL_FIRST", "SL_FIRST"]}
VALUE_RULE = "expected_value_vs_margin"
SIDE_OF = {"open_l": 0, "open_s": 1, "close_l": 0, "close_s": 1}
DEFAULT_BOUNDS = ((0.25, 3.0), (0.5, 3.0))
GEOMETRY = ("dist_sl_sigma", "dist_tp_sigma", "hours_held", "side")     # C2G closer-only inputs, in this order
N_GEO = len(GEOMETRY)
HOURS_SCALE = 24.0                                    # hours_held = elapsed hours / 24 (days); fixed, declared


def cell_name(c) -> str:
    return f"net_tp{float(c[0]):g}_sl{float(c[1]):g}"


def modules(cells=OC.CELLS) -> dict:
    """(n_classes, n_regression) per specialist."""
    return {"open_l": (2, len(cells)), "open_s": (2, len(cells)), "nop": (2, 0), "close_l": (2, 1), "close_s": (2, 1)}


def reg_bounds(m: str, cells=OC.CELLS):
    if m in OPENERS:
        return tuple(NET_BOUND for _ in cells)
    if m in CLOSERS:
        return (REM_BOUND,)
    return DEFAULT_BOUNDS


# ----------------------------------------------------------------------------- contract
def default_contract() -> dict:
    """The C2 contract as committed (sealed by ``ACT.seal``)."""
    cells = [list(c) for c in OC.CELLS]
    cont_open = {cell_name(c): {"encoding": "bounded_sigmoid", "lo": NET_BOUND[0], "hi": NET_BOUND[1], "output_dim": 1, "loss": "huber",
                                "role": "expected_value_sigma", "target": "engine net result per unit / causal sigma of the fill bar, this (TP, SL) cell",
                                "mask": "every eligible flat row whose outcome is available (all cells)"} for c in OC.CELLS}
    rem = {"remaining_sigma": {"encoding": "bounded_sigmoid", "lo": REM_BOUND[0], "hi": REM_BOUND[1], "output_dim": 1, "loss": "huber",
                               "role": "expected_value_sigma", "target": "net(hold to TP / SL / forced flat) - net(close now), per unit / position sigma",
                               "mask": "every available positioned visit of this side"}}
    spec = {}
    for m in SPECIALISTS:
        dec = {"encoding": "softmax_ce", "classes": CLASSES[m], "n_classes": 2, "output_dim": 2, "loss": "cross_entropy",
               "class_weights": "1/sqrt(train frequency), normalised (as v1)"}
        if m in OPENERS:
            dec["target"] = f"TP before SL at the reference cell {list(OC.REF_CELL)} (engine replay; TIME counts as NOT_TP_FIRST)"
            spec[m] = {"decision": dec, "continuous": cont_open}
        elif m in CLOSERS:
            dec["target"] = "SL before TP when held from here (or negative at the forced flat), engine replay"
            spec[m] = {"decision": dec, "continuous": rem}
        else:
            dec["target"] = "multires ladder NOP (nop_hard / nop_graded / negative / conflict = NOP; any other available row = TRADE)"
            spec[m] = {"decision": dec, "labels": {"ladder": list(MR.LADDER), "positive": list(ACT.NOP_POSITIVE)}}
    vprefix = {"open_l": "value:net_", "open_s": "value:net_", "close_l": "value:remaining_sigma", "close_s": "value:remaining_sigma"}
    doc = {
        "schema": SCHEMA, "scientific_release": False, "identity": None,
        "design": "docs/short_horizon/clean_round/CAUSAL_LADDER_DESIGN_2026_10_25.md (C2: action consequences, no teacher)",
        "action_space": {"actions": {"HOLD": "no order (NOP when flat)", "OPEN_LONG": "market long, TP/SL = the chosen cell",
                                     "OPEN_SHORT": "market short, TP/SL = the chosen cell", "CLOSE": "market close at the next open"},
                         "produced_by": {"HOLD": [["nop", "NOP"]], "OPEN_LONG": [["open_l", vprefix["open_l"]]], "OPEN_SHORT": [["open_s", vprefix["open_s"]]],
                                         "CLOSE": [["close_l", "SL_FIRST"], ["close_s", "SL_FIRST"], ["close_l", vprefix["close_l"]], ["close_s", vprefix["close_s"]]]},
                         "reverse": "an open of the opposite side while positioned = close-and-reverse, only when allow_reverse"},
        "outcomes": {"grid_tp": list(OC.GRID_TP), "grid_sl": list(OC.GRID_SL), "cells": cells, "ref_cell": list(OC.REF_CELL),
                     "engine": "app/short_horizon/expert_guided/env.py BarEnv.step (fills, costs, levels, protections, forced flat)",
                     "availability": "resolution bar + 1 h <= the period end; rows cut by the simulated window are censored and dropped",
                     "window_end": OC.TEST_START, "units": "causal sigma multiples (per unit)", "target_clip": [list(NET_BOUND), list(REM_BOUND)]},
        "specialists": spec,
        "calibration": {"method": "platt_binary", "fit_block": "INNER_A", "ece_bins": 15, "heads": [f"{m}.decision" for m in SPECIALISTS]},
        "strategy": {
            "genes": dict(Q_BOUNDS), "gene_space": ACT.GENE_SPACE_Q,
            "gene_map": {"open_margin": "quantile(best predicted net over side x cell, calibration flat rows, 1 - open_rate)",
                         "close_margin": "-quantile(predicted remaining value, calibration-block positioned visits, close_rem_rate)",
                         "close_p": "quantile(calibrated P(SL first), calibration-block positioned visits, 1 - close_p_rate)",
                         "allow_reverse": "gene >= 0.5", "nop_veto": "quantile(calibrated pNOP, calibration flat rows, 1 - nop_veto_rate)",
                         "positioned_visits": "the round-0 INNER_A exit visits (expert trajectory + exploration positions): strategy-free",
                         "frozen": "on the calibration block before the search; recorded in the strategy doc (gene_map)",
                         "feasibility_probe": "corners of the continuous genes + the q-max point; no point with >= 2.0 trades/week and >= 1 agent close -> STRATEGY_SPACE_INFEASIBLE"},
            "rule": ("flat: per side the cell with the highest predicted net; open the better side when its predicted net > open_margin "
                     "and (veto kept) pNOP <= nop_veto; positioned: CLOSE when predicted remaining < -close_margin OR calibrated "
                     "P(SL first) > close_p (optimal stopping), then reverse when allow_reverse and the open rule picks the other side"),
            "threshold_targets": {"close_p": "close_l.SL_FIRST|close_s.SL_FIRST", "nop_veto": "nop.NOP"},
            "value_targets": {"open_margin": "open_l.value:net_|open_s.value:net_", "close_margin": "close_l.value:remaining_sigma|close_s.value:remaining_sigma"},
            "value_rule": VALUE_RULE,
            "optimizer": {"final": {"population": 24, "generations": 12}, "visitation": {"population": 12, "generations": 4},
                          "constraint": "Deb constraint domination: >= 2 trades/week (period average) and the env rules"},
            "calibration_min_trades_per_week": ACT.MIN_TRADES_PER_WEEK,
            "choice": "the arm's fitness variant picks the frozen point among the valid non-dominated (Sharpe, DD) points; ties -> lower max DD",
            "fitness": {"default": ACT.DEFAULT_FITNESS, "variants": {k: k for k in ACT.FITNESS}},
            "require_early_close": True, "early_close_rule": ACT.EARLY_CLOSE_RULE,
            "fit_block": "INNER_A",
            "costs": "inside the targets: every predicted net is the engine's net result (spread, slippage, commission, financing)"},
        "blocks": {"calibration": "INNER_A", "decision": "INNER_B", "read_once": "OBS_2024", "requires_post_freeze_decision_block": True,
                   "calibration_block_uses": ["early_stopping", "calibration", "strategy_optimisation", "dagger_visitation_strategy"]},
        "visitation": {"round0": "expert trajectory + exploration positions (visited STATES only, never labels)",
                       "rounds": "1..3: the C2 policy at the visitation strategy frozen on INNER_A (closer rows appended, closers refitted)",
                       "openers_and_nop": "fitted once in round 0 (their rows are every eligible flat row; visitation does not change them)"},
        "constraints": {"agent_state": "none", "variants": ["FULL"], "teacher": "none"},
    }
    return ACT.seal(doc)


def validate(doc: dict) -> list:
    errs = []
    if doc.get("schema") not in (SCHEMA, SCHEMA_G):
        errs.append("C2_CONTRACT_SCHEMA")
    if doc.get("identity") != ACT.identity_of(doc):
        errs.append("C2_CONTRACT_IDENTITY")
    if doc.get("scientific_release") is not False:
        errs.append("C2_CONTRACT_RELEASE_FLAG")
    sp = doc.get("specialists") or {}
    if tuple(sp) != SPECIALISTS:
        errs.append(f"C2_CONTRACT_SPECIALISTS:{list(sp)}")
    errs += encoding_audit(doc)["errors"]
    errs += state_slot_errors(doc)
    oc = doc.get("outcomes") or {}
    cells = [tuple(c) for c in oc.get("cells") or []]
    if cells != [tuple(c) for c in OC.CELLS] or tuple(oc.get("ref_cell") or ()) != OC.REF_CELL or oc.get("window_end") != OC.TEST_START:
        errs.append("C2_CONTRACT_OUTCOMES")
    for m in OPENERS:
        ct = (sp.get(m) or {}).get("continuous") or {}
        if list(ct) != [cell_name(c) for c in OC.CELLS] or any((x.get("lo"), x.get("hi")) != NET_BOUND for x in ct.values()):
            errs.append(f"C2_CONTRACT_NET_HEADS:{m}")
    for m in CLOSERS:
        ct = (sp.get(m) or {}).get("continuous") or {}
        if list(ct) != ["remaining_sigma"] or (ct["remaining_sigma"].get("lo"), ct["remaining_sigma"].get("hi")) != REM_BOUND:
            errs.append(f"C2_CONTRACT_REMAINING_HEADS:{m}")
    nl = (sp.get("nop") or {}).get("labels") or {}
    if tuple(nl.get("ladder", ())) != tuple(MR.LADDER) or tuple(nl.get("positive", ())) != ACT.NOP_POSITIVE:
        errs.append("C2_CONTRACT_NOP_LABELS")
    cal = doc.get("calibration") or {}
    if cal.get("method") != "platt_binary" or cal.get("fit_block") != "INNER_A" or int(cal.get("ece_bins", 0)) != 15:
        errs.append("C2_CONTRACT_CALIBRATION")
    s = doc.get("strategy") or {}
    genes = s.get("genes") or {}
    if s.get("gene_space") == ACT.GENE_SPACE_Q:
        if not ((tuple(genes) == GENES_Q and genes == Q_BOUNDS) or (tuple(genes) == GENES_Q13 and genes == Q_BOUNDS_13)):
            errs.append(f"C2_CONTRACT_GENES:{list(genes)}")
        errs += [f"C2_CONTRACT_{e}" for e in ACT.v13_rule_errors(s, tuple(genes) == GENES_Q13)]
    elif s.get("gene_space") is not None or tuple(genes) != GENES:
        errs.append(f"C2_CONTRACT_GENES:{list(genes)}")
    for g, b in genes.items():
        if not (isinstance(b, list) and len(b) == 2 and float(b[0]) < float(b[1])):
            errs.append(f"C2_CONTRACT_GENE_BOUNDS:{g}")
    opt = s.get("optimizer") or {}
    for k in ("final", "visitation"):
        o = opt.get(k) or {}
        if int(o.get("population", 0)) < 4 or int(o.get("population", 0)) % 4 or int(o.get("generations", 0)) < 1:
            errs.append(f"C2_CONTRACT_OPTIMIZER:{k}")
    if (opt.get("constraint") or "").split(":")[0] != "Deb constraint domination":
        errs.append("C2_CONTRACT_CONSTRAINT_HANDLING")
    if float(s.get("calibration_min_trades_per_week", 0)) != ACT.MIN_TRADES_PER_WEEK:
        errs.append("C2_CONTRACT_ACTIVITY_RULE")
    fz = s.get("fitness") or {}
    if fz.get("default") != ACT.DEFAULT_FITNESS or sorted(fz.get("variants") or {}) != sorted(ACT.FITNESS):
        errs.append("C2_CONTRACT_FITNESS")
    if s.get("require_early_close") is not True:
        errs.append("C2_CONTRACT_EARLY_CLOSE_RULE")
    if s.get("fit_block") != "INNER_A":
        errs.append("C2_CONTRACT_STRATEGY_BLOCK")
    bl = doc.get("blocks") or {}
    uses = bl.get("calibration_block_uses") or []
    if bl.get("calibration") != "INNER_A" or bl.get("decision") != "INNER_B" or bl.get("read_once") != "OBS_2024" \
            or sorted(uses) != sorted(["early_stopping", "calibration", "strategy_optimisation", "dagger_visitation_strategy"]) \
            or bl.get("requires_post_freeze_decision_block") is not True:
        errs.append("C2_CONTRACT_BLOCKS")
    c = doc.get("constraints") or {}
    if c.get("agent_state") != "none" or c.get("variants") != ["FULL"] or c.get("teacher") != "none":
        errs.append("C2_CONTRACT_CONSTRAINTS")
    return errs


def default_contract_geometry() -> dict:
    """The C2G contract: C2 (``default_contract``) plus the closer-only order geometry block, sealed."""
    doc = json.loads(json.dumps(default_contract()))
    doc["schema"], doc["identity"] = SCHEMA_G, None
    doc["design"] += ("; C2G (owner 2026-10-25, 'Sí, como brazo extra'): the closers alone also see the geometry of their "
                      "own open order; openers and NOP unchanged")
    for m in SPECIALISTS:
        doc["specialists"][m]["state_slots"] = list(GEOMETRY) if m in CLOSERS else []
    doc["closer_order_geometry"] = {
        "order": list(GEOMETRY), "applies_to": list(CLOSERS), "hours_scale": HOURS_SCALE,
        "inputs": {"dist_sl_sigma": "side * (close[b] - SL) / sigma_entry", "dist_tp_sigma": "side * (TP - close[b]) / sigma_entry",
                   "hours_held": f"(open time of bar b + 1 h - open time of the fill bar) / 1 h / {HOURS_SCALE:g}",
                   "side": "+1 long, -1 short"},
        "sigma_entry": "the decision sigma the engine used to set this order's levels (Position.sigma)",
        "causality": "computed after bar b from the open order's own levels and close[b]; nothing after b enters",
        "network_input": "closers: recipe window (+) the nine agent-state slots as constant zeros (as C2) (+) these four values; openers / NOP: as C2",
        "forbidden": "any other state slot for any specialist (balance, equity, PnL history, entries count, agent outputs, can_enter, ...); geometry on openers or NOP",
        "owner_words": "Sí, como brazo extra (2026-10-25)"}
    doc["constraints"]["closer_inputs"] = "recipe_window+order_geometry"
    return ACT.seal(doc)


def contract_v13(base: dict) -> dict:
    """C2 / C2G v1.3 from the committed C2 / C2G contract: per-side open rates, the side cap, the activity CI rule and
    the front-size diagnostic (``ACT.v13_strategy_block``); every other field kept; sealed with a new identity."""
    d = json.loads(json.dumps(base))
    d["identity"] = None
    d["strategy"] = ACT.v13_strategy_block(d["strategy"], Q_BOUNDS_13, {
        "open_margin_l": "quantile(best predicted LONG net over the cells, calibration flat rows, 1 - open_rate_l)",
        "open_margin_s": "quantile(best predicted SHORT net over the cells, calibration flat rows, 1 - open_rate_s)"})
    d["strategy"]["gene_map"].pop("open_margin", None)
    d["strategy"]["rule"] = ("flat: per side the cell with the highest predicted net; open the better side among sides whose predicted net > "
                             "open_margin_side (per-side margins) and (veto kept) pNOP <= nop_veto; positioned: CLOSE when predicted remaining "
                             "< -close_margin OR calibrated P(SL first) > close_p (optimal stopping), then reverse when allow_reverse and the "
                             "open rule picks the other side")
    d["strategy"]["value_targets"] = {"open_margin_l": "open_l.value:net_", "open_margin_s": "open_s.value:net_",
                                      "close_margin": "close_l.value:remaining_sigma|close_s.value:remaining_sigma"}
    d["derived_from"] = {"identity": base.get("identity"), "why": ("v12 chain: C2/C2G ensembles at exactly 2.0 trades/week on calibration, 1.48-1.94 out of "
                                                                   "sample; one-sided seeds")}
    return ACT.seal(d)


def state_slot_errors(doc: dict) -> list:
    """C2G's input rule: only the closers carry state slots, and only exactly ``GEOMETRY``; a C2 contract carries none."""
    errs = []
    g = doc.get("schema") == SCHEMA_G
    for m, d in (doc.get("specialists") or {}).items():
        slots = list(d.get("state_slots") or [])
        for x in slots:
            if x not in GEOMETRY:
                errs.append(f"STATE_SLOT_NOT_ALLOWED:{m}:{x}")
            elif m not in CLOSERS:
                errs.append(f"GEOMETRY_ON_NON_CLOSER:{m}:{x}")
            elif not g:
                errs.append(f"STATE_SLOT_NOT_ALLOWED:{m}:{x}:a C2 contract carries no state slot")
        if g and m in CLOSERS and slots != list(GEOMETRY):
            errs.append(f"C2G_CLOSER_GEOMETRY:{m}:{slots}")
    geo = doc.get("closer_order_geometry")
    if g:
        geo = geo or {}
        if list(geo.get("order") or []) != list(GEOMETRY) or sorted(geo.get("inputs") or {}) != sorted(GEOMETRY) \
                or list(geo.get("applies_to") or []) != list(CLOSERS) or float(geo.get("hours_scale", -1)) != HOURS_SCALE:
            errs.append("C2G_GEOMETRY_BLOCK")
        if (doc.get("constraints") or {}).get("closer_inputs") != "recipe_window+order_geometry":
            errs.append("C2G_CONSTRAINTS")
    elif geo is not None:
        errs.append("STATE_SLOT_NOT_ALLOWED:closer_order_geometry on a C2 contract")
    return errs


def encoding_audit(doc: dict) -> dict:
    """v1's audit for every categorical head and every continuous head, plus the value-rule checks of C2 (see the module
    docstring). Actions produced by value heads are checked here; the class producers by ``ACT.encoding_audit``."""
    errs = []
    asp = doc.get("action_space") or {}
    prod = asp.get("produced_by") or {}
    cls_prod = {a: [p for p in v if isinstance(p, list) and len(p) == 2 and not str(p[1]).startswith("value:")] for a, v in prod.items()}
    val_prod = {a: [p for p in v if isinstance(p, list) and len(p) == 2 and str(p[1]).startswith("value:")] for a, v in prod.items()}
    shadow = dict(doc)
    shadow["action_space"] = dict(asp, actions={a: d for a, d in (asp.get("actions") or {}).items() if cls_prod.get(a)}, produced_by=cls_prod)
    shadow["strategy"] = {"threshold_targets": (doc.get("strategy") or {}).get("threshold_targets") or {}}
    base = ACT.encoding_audit(shadow)
    errs += base["errors"]
    sp = doc.get("specialists") or {}
    values = {m: {q for q, c in (d.get("continuous") or {}).items() if c.get("role") == "expected_value_sigma"} for m, d in sp.items()}
    for m, d in sp.items():
        for q, c in (d.get("continuous") or {}).items():
            if any(k in c for k in ACT.FROM_CONTINUOUS_MARKERS):
                errs.append(f"VALUE_TARGET_INTERVALS:{m}.{q}")

    def value_ok(head, ref):
        pref = str(ref)[len("value:"):]
        return head in values and any(q == pref or (pref.endswith("_") and q.startswith(pref)) for q in values[head])

    for a in asp.get("actions") or {}:
        if not cls_prod.get(a) and not any(value_ok(h, r) for h, r in val_prod.get(a, [])):
            errs.append(f"ACTION_UNREACHABLE:{a}")
        for h, r in val_prod.get(a, []):
            if not value_ok(h, r):
                errs.append(f"VALUE_TARGET_NOT_EXPECTED_VALUE:{a}:{h}.{r}")
    s = doc.get("strategy") or {}
    if (s.get("value_targets") or {}) and s.get("value_rule") != VALUE_RULE:
        errs.append(f"VALUE_RULE_UNKNOWN:{s.get('value_rule')}")
    for g, tgt in (s.get("value_targets") or {}).items():
        if isinstance(tgt, (list, dict)):
            errs.append(f"VALUE_TARGET_INTERVALS:strategy.{g}: one threshold per gene, never a list of cut points")
            continue
        for t in str(tgt).split("|"):
            h, _, r = t.partition(".")
            if not value_ok(h, r):
                errs.append(f"VALUE_TARGET_NOT_EXPECTED_VALUE:strategy.{g}:{t}")
    heads = base["heads"]
    return {"errors": errs, "heads": heads, "verdict": "PASS" if not errs else "REFUSED",
            "rule": ("categorical = softmax_ce over its declared classes; continuous = bounded_sigmoid; a continuous head produces an "
                     "action only through 'act iff E[value | x, do(action)] > margin' on an expected-value head (one threshold, one action); "
                     "no class is ever read off an interval of a continuous output")}


def load(path) -> dict:
    doc = json.loads(Path(path).read_text())
    errs = validate(doc)
    if errs:
        raise SystemExit(f"C2_CONTRACT_INVALID:{errs}")
    return doc


def is_c2(doc: dict) -> bool:
    """C2 or C2G (both run through this module)."""
    return isinstance(doc, dict) and doc.get("schema") in (SCHEMA, SCHEMA_G)


def is_c2g(doc: dict) -> bool:
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA_G


# ----------------------------------------------------------------------------- C2G: closer-only order geometry
def order_geometry(pos, bars, close, v_ns) -> np.ndarray:
    """(n, 4) ``GEOMETRY`` of the open ``pos`` at decision bars ``bars``: signed distances to its SL / TP in its own
    sigma, hours held since the fill bar opened (scaled), side. Causal: close of bar b and the order's own levels."""
    b = np.asarray(bars, np.int64)
    side, sg = float(pos.direction), float(pos.sigma)
    c = np.asarray(close, float)[b]
    hours = (np.asarray(v_ns, np.int64)[b] + E.H_NS - int(np.asarray(v_ns, np.int64)[int(pos.entry_bar)])) / E.H_NS
    return np.c_[side * (c - float(pos.sl)) / sg, side * (float(pos.tp) - c) / sg, hours / HOURS_SCALE,
                 np.full(len(b), side)].astype(np.float32)


def geometry_fn(cx):
    """pos, bar -> (4,) geometry on ``cx``'s bars (for the exit tables)."""
    close = cx.ds.bars["CLOSE"].to_numpy(float)
    v = cx.ds.bars.index.as_unit("ns").asi8

    def f(pos, bar):
        return order_geometry(pos, [bar], close, v)[0]
    return f


def X_geo(Xw, rows, n_state, G) -> np.ndarray:
    """Closer input under C2G: recipe window (+) ``n_state`` constant zeros (agent_state none, as C2) (+) the geometry."""
    rows = np.asarray(rows, np.int64)
    G = np.asarray(G, np.float32).reshape(len(rows), N_GEO)
    return np.c_[Xw[rows], np.zeros((len(rows), n_state), np.float32), G].astype(np.float32)


def exit_sets_geo(cx, xt: dict, period: str, side_index: int):
    """``OC.exit_sets`` plus the geometry of every kept visit: (rows, y, R, G)."""
    m = OC.exit_mask(cx, xt, period, side_index)
    if not len(m) or not m.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 1), np.float32), np.zeros((0, N_GEO), np.float32)
    return (xt["row"][m], xt["sl_first"][m].astype(np.int64), xt["remaining"][m][:, None].astype(np.float32),
            np.asarray(xt["geo"], np.float32)[m])


class CloserGeo:
    """The closers' outputs under C2G. They depend on the open order, so they are computed on demand: the first time an
    order is consulted at row r0, one batched forward pass covers rows r0 .. r0 + CHUNK - 1 (an order cannot outlive the
    week's forced flat), cached per order; later rows extend with a new segment. ``members`` = [(pols, calibration)]:
    each member's calibrated P(SL first) and remaining value; several members (seed ensemble) are averaged, then
    ``ens_cal`` (one Platt map per closer refitted on the ensemble) is applied to the mean probability's logit, exactly
    as the per-row table path does (mean of calibrated tables -> logits -> Platt). ``allowed`` = visible rows (the
    strategy optimiser's restriction to the calibration block); a row outside raises, as a blank table row does."""
    CHUNK = 168

    def __init__(self, cx, members, n_state, allowed=None, ens_cal=None, cache=None):
        self.cx, self.members, self.ns, self.allowed, self.ens_cal = cx, list(members), int(n_state), allowed, ens_cal
        self.cache = {} if cache is None else cache
        self.close = cx.ds.bars["CLOSE"].to_numpy(float)
        self.v = cx.ds.bars.index.as_unit("ns").asi8

    def restricted(self, keep):
        allowed = keep if self.allowed is None else (np.asarray(keep, bool) & self.allowed)
        return CloserGeo(self.cx, self.members, self.ns, allowed, self.ens_cal, self.cache)

    def with_ens_cal(self, cal: dict):
        return CloserGeo(self.cx, self.members, self.ns, self.allowed, {m: {"a": float(cal[m]["a"]), "b": float(cal[m]["b"])} for m in CLOSERS})

    @staticmethod
    def merge(cgs: list):
        c0 = cgs[0]
        return CloserGeo(c0.cx, [mb for c in cgs for mb in c.members], c0.ns)

    def _mean(self, m, X):
        ps, rm = [], []
        for pols, cal in self.members:
            z, r = ACT.module_outputs(pols[m], X)
            ps.append(ACT.platt(z, cal[m]["a"], cal[m]["b"]))
            rm.append(np.asarray(r, float)[:, 0])
        return np.mean(ps, axis=0), np.mean(rm, axis=0)

    def rows_values(self, m, rows, G):
        """(remaining value, P(SL first)) of closer ``m`` on (rows, G), with the ensemble map when set (references)."""
        X = X_geo(self.cx.Xw, rows, self.ns, G)
        p, rem = self._mean(m, X)
        if self.ens_cal is not None:
            p = ACT.platt(ACT.logit(np.nan_to_num(p, nan=0.5)), self.ens_cal[m]["a"], self.ens_cal[m]["b"])
        return rem, p

    def raw_logit_rows(self, m, rows, G) -> np.ndarray:
        """Logit of the members' mean calibrated P(SL first) on (rows, G) (the ensemble's Platt refit input)."""
        if not len(rows):
            return np.zeros(0)
        p, _ = self._mean(m, X_geo(self.cx.Xw, rows, self.ns, G))
        return ACT.logit(np.nan_to_num(p, nan=0.5))

    def _segment(self, pos, r0):
        n = len(self.cx.Xw)
        rows = np.arange(r0, min(r0 + self.CHUNK, n), dtype=np.int64)
        bars = np.asarray(self.cx.ds.row_bar, np.int64)[rows]
        ok = bars >= int(pos.entry_bar)
        G = order_geometry(pos, np.where(ok, bars, int(pos.entry_bar)), self.close, self.v)
        m = "close_l" if pos.direction > 0 else "close_s"
        p, rem = self._mean(m, X_geo(self.cx.Xw, rows, self.ns, G))
        if self.ens_cal is not None:
            p = ACT.platt(ACT.logit(np.nan_to_num(p, nan=0.5)), self.ens_cal[m]["a"], self.ens_cal[m]["b"])
        return r0, np.where(ok, rem, np.nan), np.where(ok, p, np.nan)

    def value(self, r, bar, pos):
        """(predicted remaining value, P(SL first)) of the open ``pos`` at row r (decision after bar ``bar``)."""
        r = int(r)
        if self.allowed is not None and not self.allowed[r]:
            raise ValueError(f"C2_ROW_OUTSIDE_TABLE:closer:{r}")
        if int(self.cx.ds.row_bar[r]) != int(bar):
            raise ValueError(f"C2G_ROW_BAR_MISMATCH:{r}:{bar}")
        # the order is identified by its fill bar, side, levels and sigma; a cache is never shared across calibrations
        # (``with_ens_cal`` / ``merge`` start a fresh one), only across the restricted views of one evaluator
        segs = self.cache.setdefault((int(pos.entry_bar), int(pos.direction), float(pos.sl), float(pos.tp), float(pos.sigma)), [])
        for r0, rem, p in segs:
            if r0 <= r < r0 + len(rem):
                break
        else:
            r0, rem, p = self._segment(pos, r)
            segs.append((r0, rem, p))
        x, q = float(rem[r - r0]), float(p[r - r0])
        if not (math.isfinite(x) and math.isfinite(q)):
            raise ValueError(f"C2_ROW_OUTSIDE_TABLE:closer:{r}")
        return x, q


# ----------------------------------------------------------------------------- per-row table
FIELDS = ("net", "ptp", "pn", "rem", "psl")
RAW = {"ptp": "z_ptp", "pn": "z_pn", "psl": "z_psl"}


def raw_table(pols: dict, Xw, rows, n_state: int, n_cells: int = len(OC.CELLS), closers: bool = True) -> dict:
    """Uncalibrated outputs on ``rows`` (agent state zeros: agent_state none); NaN elsewhere. ``z_*`` = logit margins.
    ``closers=False`` (C2G): the closers' per-row columns stay NaN (their outputs depend on the order: ``CloserGeo``)."""
    n = len(Xw)
    t = {"net": np.full((n, 2, n_cells), np.nan), "z_ptp": np.full((n, 2), np.nan), "z_pn": np.full(n, np.nan),
         "rem": np.full((n, 2), np.nan), "z_psl": np.full((n, 2), np.nan)}
    rows = np.asarray(rows, np.int64)
    X = np.c_[Xw[rows], np.zeros((len(rows), n_state), np.float32)].astype(np.float32)
    for m in OPENERS:
        z, r = ACT.module_outputs(pols[m], X)
        t["z_ptp"][rows, SIDE_OF[m]], t["net"][rows, SIDE_OF[m], :] = z, r
    for m in (CLOSERS if closers else ()):
        z, r = ACT.module_outputs(pols[m], X)
        t["z_psl"][rows, SIDE_OF[m]], t["rem"][rows, SIDE_OF[m]] = z, r[:, 0]
    t["z_pn"][rows] = ACT.module_outputs(pols["nop"], X)[0]
    return t


def calibrate_table(raw: dict, cal: dict) -> dict:
    """Probabilities through each specialist's Platt map; regressions unchanged."""
    t = {"net": raw["net"].copy(), "rem": raw["rem"].copy()}
    pt = np.full(raw["z_ptp"].shape, np.nan)
    ps = np.full(raw["z_psl"].shape, np.nan)
    for m in OPENERS:
        z = raw["z_ptp"][:, SIDE_OF[m]]
        pt[:, SIDE_OF[m]] = np.where(np.isfinite(z), ACT.platt(np.nan_to_num(z), cal[m]["a"], cal[m]["b"]), np.nan)
    for m in CLOSERS:
        z = raw["z_psl"][:, SIDE_OF[m]]
        ps[:, SIDE_OF[m]] = np.where(np.isfinite(z), ACT.platt(np.nan_to_num(z), cal[m]["a"], cal[m]["b"]), np.nan)
    z = raw["z_pn"]
    t["ptp"], t["psl"] = pt, ps
    t["pn"] = np.where(np.isfinite(z), ACT.platt(np.nan_to_num(z), cal["nop"]["a"], cal["nop"]["b"]), np.nan)
    if "closer" in raw:                                   # C2G ensemble: the refitted closer Platt maps on the mean logit
        t["closer"] = raw["closer"].with_ens_cal(cal)
    return t


def mean_tables(tables: list) -> dict:
    out = {f: np.mean([t[f] for t in tables], axis=0) for f in FIELDS}
    if "closer" in tables[0]:                             # C2G: the closers' members are pooled (averaged on demand)
        out["closer"] = CloserGeo.merge([t["closer"] for t in tables])
    return out


def to_raw(table: dict) -> dict:
    """Ensemble: probabilities back to logits (for one Platt refit on the ensemble), regressions unchanged."""
    r = {"net": table["net"].copy(), "rem": table["rem"].copy()}
    for f, z in RAW.items():
        p = table[f]
        r[z] = np.where(np.isfinite(p), ACT.logit(np.nan_to_num(p, nan=0.5)), np.nan)
    if "closer" in table:
        r["closer"] = table["closer"]
    return r


def restrict(table: dict, rows) -> dict:
    """Every row outside ``rows`` blanked (NaN): the only view the strategy optimiser gets."""
    keep = np.zeros(len(table["pn"]), bool)
    keep[np.asarray(rows, np.int64)] = True
    return {f: (v.restricted(keep) if f == "closer" else np.where(keep.reshape((-1,) + (1,) * (v.ndim - 1)), v, np.nan))
            for f, v in table.items()}


# ----------------------------------------------------------------------------- the strategy (upper layer)
@dataclass(frozen=True)
class StrategyC2:
    open_margin: float
    close_margin: float
    close_p: float
    allow_reverse: bool
    nop_veto: float | None

    @classmethod
    def from_genes(cls, g: dict, veto: bool):
        return cls(float(g["open_margin"]), float(g["close_margin"]), float(g["close_p"]), bool(float(g["allow_reverse"]) >= 0.5),
                   float(g["nop_veto"]) if veto else None)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class StrategyC2PS:
    """C2 / C2G v1.3: the open margin is PER SIDE (each from its own side's predicted-net calibration distribution)."""
    open_margin_l: float
    open_margin_s: float
    close_margin: float
    close_p: float
    allow_reverse: bool
    nop_veto: float | None

    @classmethod
    def from_genes(cls, g: dict, veto: bool):
        return cls(float(g["open_margin_l"]), float(g["open_margin_s"]), float(g["close_margin"]), float(g["close_p"]),
                   bool(float(g["allow_reverse"]) >= 0.5), float(g["nop_veto"]) if veto else None)

    def to_dict(self):
        return dataclasses.asdict(self)


def strategy_from_dict(d: dict):
    """Rebuild a frozen C2 strategy from its recorded dict (``StrategyC2`` or, v1.3, ``StrategyC2PS``)."""
    if "close_margin_l" in d:                             # C3 (2026-10-26): per-side close thresholds, early-close switch
        from . import c3 as C3
        return C3.StrategyC3(**d)
    return StrategyC2PS(**d) if "open_margin_l" in d else StrategyC2(**d)


NO_TRADE = StrategyC2(1e9, 0.0, 1.0, False, None)


class PolicyC2:
    """Frozen upper layer over a C2 table; ``policy(cx)`` -> f(bar, r, st, env) -> (action, tp, sl). A blank row raises."""

    def __init__(self, table: dict, strategy: StrategyC2, cells=OC.CELLS):
        self.t, self.s, self.cells = table, strategy, [tuple(c) for c in cells]

    def _v(self, f, r, *idx):
        x = self.t[f][(r,) + idx]
        if np.ndim(x) == 0 and not np.isfinite(x):
            raise ValueError(f"C2_ROW_OUTSIDE_TABLE:{f}:{r}")
        if np.ndim(x) and not np.all(np.isfinite(x)):
            raise ValueError(f"C2_ROW_OUTSIDE_TABLE:{f}:{r}")
        return x

    def open_decision(self, r):
        s = self.s
        if s.nop_veto is not None and float(self._v("pn", r)) > s.nop_veto:
            return None
        best = None
        for si, d in ((0, E.LONG), (1, E.SHORT)):
            net = np.asarray(self._v("net", r, si), float)
            ci = int(np.argmax(net))
            mar = (s.open_margin_l if si == 0 else s.open_margin_s) if isinstance(s, StrategyC2PS) else s.open_margin
            if net[ci] > mar and (best is None or net[ci] > best[0]):
                best = (float(net[ci]), d, self.cells[ci][0], self.cells[ci][1])
        return None if best is None else best[1:]

    def close_now(self, r, si, bar=None, pos=None) -> bool:
        if "closer" in self.t:                            # C2G: the closer reads its own order's geometry
            rem, psl = self.t["closer"].value(r, bar, pos)
            return rem < -self.s.close_margin or psl > self.s.close_p
        return float(self._v("rem", r, si)) < -self.s.close_margin or float(self._v("psl", r, si)) > self.s.close_p

    def __call__(self, cx):
        def f(bar, r, st, env):
            pos = int(st[0])
            if pos != 0:
                si = 0 if pos > 0 else 1
                if not self.close_now(r, si, bar, env.pos if "closer" in self.t else None):
                    return E.HOLD, 1.5, 2.0
                o = self.open_decision(r)
                if o is not None and self.s.allow_reverse and o[0] != (E.LONG if pos > 0 else E.SHORT):
                    return o[0], float(o[1]), float(o[2])
                return E.CLOSE, 1.5, 2.0
            o = self.open_decision(r)
            return (E.HOLD, 1.5, 2.0) if o is None else (o[0], float(o[1]), float(o[2]))
        return f


def strategy_of(doc: dict):
    s = doc.get("strategy")
    return None if s is None else strategy_from_dict(s)


def closer_reference(table: dict, src: dict) -> dict:
    """The closers' calibration-block references on positioned visits: ``src`` = {closer: {"rows", "G" (C2G)}} (the
    round-0 INNER_A exit visits); C2 reads the per-row table, C2G evaluates each visit with its own order's geometry.
    Both sides pooled: {"rem": ..., "psl": ...}."""
    rem, psl = [], []
    for m in CLOSERS:
        d = src.get(m) or {}
        r = np.asarray(d.get("rows", []), np.int64)
        if not len(r):
            continue
        if "closer" in table:
            x, q = table["closer"].rows_values(m, r, d["G"])
        else:
            x, q = table["rem"][r, SIDE_OF[m]], table["psl"][r, SIDE_OF[m]]
        rem.append(np.asarray(x, float))
        psl.append(np.asarray(q, float))
    return {"rem": np.concatenate(rem) if rem else np.zeros(0), "psl": np.concatenate(psl) if psl else np.zeros(0)}


def gene_map(view: dict, rows, closer_ref: dict) -> ACT.GeneMap:
    r = np.asarray(rows, np.int64)
    best = np.nanmax(np.where(np.isfinite(view["net"][r]), view["net"][r], -np.inf).reshape(len(r), -1), axis=1) if len(r) else np.zeros(0)
    refs = {"open_score": np.where(np.isfinite(best), best, np.nan), "rem_ref": closer_ref["rem"], "psl_ref": closer_ref["psl"], "nop_score": view["pn"][r]}
    rules = {"open_margin": ("open_rate", "open_score", "upper_rate"), "close_margin": ("close_rem_rate", "rem_ref", "neg_lower_rate"),
             "close_p": ("close_p_rate", "psl_ref", "upper_rate"), "allow_reverse": ("allow_reverse", None, "copy"),
             "nop_veto": ("nop_veto_rate", "nop_score", "upper_rate")}
    return ACT.GeneMap(refs, rules, "C2: open on calibration flat rows; closers on the round-0 INNER_A positioned visits")


def gene_map_v13(view: dict, rows, closer_ref: dict) -> ACT.GeneMap:
    """C2 v1.3: per-side open references (each side's best predicted net over the cells), the closer references as C2."""
    r = np.asarray(rows, np.int64)
    refs = {"rem_ref": closer_ref["rem"], "psl_ref": closer_ref["psl"], "nop_score": view["pn"][r]}
    for si, k in ((0, "open_score_l"), (1, "open_score_s")):
        x = view["net"][r, si] if len(r) else np.zeros((0, 1))
        b = np.nanmax(np.where(np.isfinite(x), x, -np.inf).reshape(len(r), -1), axis=1) if len(r) else np.zeros(0)
        refs[k] = np.where(np.isfinite(b), b, np.nan)
    rules = {"open_margin_l": ("open_rate_l", "open_score_l", "upper_rate"), "open_margin_s": ("open_rate_s", "open_score_s", "upper_rate"),
             "close_margin": ("close_rem_rate", "rem_ref", "neg_lower_rate"), "close_p": ("close_p_rate", "psl_ref", "upper_rate"),
             "allow_reverse": ("allow_reverse", None, "copy"), "nop_veto": ("nop_veto_rate", "nop_score", "upper_rate")}
    return ACT.GeneMap(refs, rules, "C2 v1.3: per-side open references on calibration flat rows; closers on the round-0 INNER_A positioned visits")


def freeze(table, cal_rows, score_table, contract, seed, budget, primary, closer_src=None):
    gm = None
    st = contract.get("strategy") or {}
    v13 = ACT.is_v13(st)
    if st.get("gene_space") == ACT.GENE_SPACE_Q:
        ref = closer_reference(table, closer_src or {})
        gm = (lambda view: gene_map_v13(view, cal_rows, ref)) if v13 else (lambda view: gene_map(view, cal_rows, ref))  # noqa: E731
    return ACT.freeze_strategy(table, cal_rows, score_table, contract, seed, budget, primary,
                               gene_names=GENES, make_strategy=(StrategyC2PS if v13 else StrategyC2).from_genes, restrict_fn=restrict, gene_map_fn=gm)


# ----------------------------------------------------------------------------- label sets and metrics
def X_of(Xw, rows, n_state):
    rows = np.asarray(rows, np.int64)
    return np.c_[Xw[rows], np.zeros((len(rows), n_state), np.float32)].astype(np.float32)


def clip_targets(R, bound):
    R = np.asarray(R, np.float32)
    n = int(np.sum((R < bound[0]) | (R > bound[1])))
    return np.clip(R, bound[0], bound[1]).astype(np.float32), n


def nop_rows(cx, et, period, nopl):
    """(rows, y) of the NOP specialist on the eligible flat rows of ``period`` (censored multires rows dropped)."""
    rr, yy = [], []
    for r in np.flatnonzero(et["eligible"] & (cx.ds.period == period)):
        lab = nopl.label(int(cx.ds.row_bar[r]), int(r), False)
        if lab is not None:
            rr.append(int(r))
            yy.append(int(lab[0]))
    return np.asarray(rr, np.int64), np.asarray(yy, np.int64)


def sets(cx, et, xt, period, nopl, n_state, clipped=None, geo=False, geo_out=None):
    """{module: (X, y, R, Rm)} (nop: (X, y)) and {module: rows} for ``period``: openers from the entry table, closers
    from the exit table ``xt``, NOP from the ladder. Regression targets clipped to the declared bounds (counted)."""
    D, rows = {}, {}
    for m in OPENERS:
        r, y, R = OC.entry_sets(cx, et, period, SIDE_OF[m])
        R, n = clip_targets(R, NET_BOUND)
        if clipped is not None:
            clipped[f"{period}/{m}"] = n
        D[m] = (X_of(cx.Xw, r, n_state), y, R, np.ones_like(R))
        rows[m] = r
    for m in CLOSERS:
        if geo:                                           # C2G: closer rows carry their order's geometry
            r, y, R, G = exit_sets_geo(cx, xt, period, SIDE_OF[m])
            X = X_geo(cx.Xw, r, n_state, G)
            if geo_out is not None:
                geo_out[m] = G
        else:
            r, y, R = OC.exit_sets(cx, xt, period, SIDE_OF[m])
            X = X_of(cx.Xw, r, n_state)
        R, n = clip_targets(R, REM_BOUND)
        if clipped is not None:
            clipped[f"{period}/{m}"] = n
        D[m] = (X, y, R, np.ones_like(R))
        rows[m] = r
    r, y = nop_rows(cx, et, period, nopl)
    D["nop"] = (X_of(cx.Xw, r, n_state), y)
    rows["nop"] = r
    return D, rows


def cat_sets(a, b):
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def reg_metrics(pred, y) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    if len(y) == 0:
        return {"rows": 0, "mae": float("nan"), "r2": float("nan"), "spearman": float("nan")}
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"rows": int(len(y)), "mae": float(np.mean(np.abs(pred - y))), "r2": (1.0 - float(np.sum((pred - y) ** 2)) / ss) if ss > 0 else float("nan"),
            "spearman": spearman(pred, y), "target_mean": float(y.mean()), "pred_mean": float(pred.mean())}


def head_report(pols, cal, D, bins=15) -> dict:
    """Per specialist: binary AUC / Brier / ECE (raw and calibrated) and regression MAE / R2 / Spearman (per output)."""
    out = {}
    for m, pol in pols.items():
        if m not in D or len(D[m][1]) == 0:
            out[m] = {"rows": 0}
            continue
        z, reg = ACT.module_outputs(pol, D[m][0])
        y = np.asarray(D[m][1], int)
        rec = {"rows": int(len(y)), "positives": int(y.sum()), "raw": ACT.head_metrics(ACT.sigmoid(z), y, bins)}
        if m in cal:
            rec["calibrated"] = ACT.head_metrics(ACT.platt(z, cal[m]["a"], cal[m]["b"]), y, bins)
        if reg is not None and len(D[m]) > 2:
            R = np.asarray(D[m][2], float)
            per = [reg_metrics(reg[:, j], R[:, j]) for j in range(R.shape[1])]
            rec["regression"] = {"per_output": per, "mean_mae": float(np.mean([p["mae"] for p in per])),
                                 "mean_spearman": float(np.nanmean([p["spearman"] for p in per])) if any(np.isfinite(p["spearman"]) for p in per) else float("nan")}
        out[m] = rec
    return out


def calibrate(pols, D):
    cal = {}
    for m, pol in pols.items():
        z, _ = ACT.module_outputs(pol, D[m][0])
        y = np.asarray(D[m][1], int)
        a_, b_ = ACT.fit_platt(z, y)
        cal[m] = {"a": a_, "b": b_, "rows": int(len(y)), "positives": int(y.sum()), "single_class": bool(y.min() == y.max()) if len(y) else True,
                  "before": ACT.head_metrics(ACT.sigmoid(z), y), "after": ACT.head_metrics(ACT.platt(z, a_, b_), y)}
    return cal


def save_sets(path, named: dict):
    """``named``: {key: {module: rows / y / R}} -> one npz (the ensemble's Platt refit and the reported-only metrics)."""
    arrs = {}
    for key, per in named.items():
        for m, t in per.items():
            rows, y, R = t[:3]
            arrs[f"{key}|{m}|rows"], arrs[f"{key}|{m}|y"] = np.asarray(rows, np.int64), np.asarray(y, np.int64)
            if R is not None:
                arrs[f"{key}|{m}|R"] = np.asarray(R, np.float32)
            if len(t) > 3 and t[3] is not None:              # C2G closers: the geometry of every saved visit
                arrs[f"{key}|{m}|G"] = np.asarray(t[3], np.float32)
    np.savez_compressed(path, **arrs)


def load_sets(path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        key, m, part = k.split("|")
        out.setdefault(key, {}).setdefault(m, {})[part] = z[k]
    return out


def sets_from_saved(cx, per: dict, n_state) -> dict:
    D = {}
    for m, d in per.items():
        X = X_geo(cx.Xw, d["rows"], n_state, d["G"]) if "G" in d else X_of(cx.Xw, d["rows"], n_state)
        D[m] = (X, d["y"]) if "R" not in d else (X, d["y"], d["R"], np.ones_like(d["R"]))
    return D


# ----------------------------------------------------------------------------- one seed
def run_seed(cx, seed, d):
    """One seed under C2. ``d``: contract, fit_module, SCFG, NS, rows_all, cal_rows, et (entry table), x0 (round-0 exit
    tables by period), nopl, rounds, led, ledger, fitness, sig. Every fitted or chosen quantity reads TRAIN / INNER_A only."""
    AC, SCFG, NS = d["contract"], d["SCFG"], d["NS"]
    G = is_c2g(AC)
    clipped = {}
    xt = {p: d["x0"][p] for p in ("TRAIN", "INNER_A")}
    D_tr, _ = sets(cx, d["et"], xt["TRAIN"], "TRAIN", d["nopl"], NS, clipped, geo=G)
    go_in = {}
    D_in, rows_in = sets(cx, d["et"], xt["INNER_A"], "INNER_A", d["nopl"], NS, clipped, geo=G, geo_out=go_in)
    closer_src = {m: {"rows": rows_in[m], "G": go_in.get(m)} for m in CLOSERS}
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    primary = d["fitness"]

    def score_table(view, strat):
        m, trades = ACT.rollout_with_trades(cx, "INNER_A", PolicyC2(view, strat)(cx), SCFG)
        return ACT.window_components(m, trades, d["sig"], SCFG.capital, min_opw, **ACT.score_kwargs(AC))

    pols, infos, rounds, comp = {}, {}, [], None
    for rnd in range(d["rounds"] + 1):
        fit_now = list(SPECIALISTS) if rnd == 0 else list(CLOSERS)
        if rnd > 0 and comp is not None:
            for p, store in (("TRAIN", D_tr), ("INNER_A", D_in)):
                xr = OC.exit_table(cx, SCFG, OC.visit_positions(cx, p, comp(cx), SCFG), geometry_fn=geometry_fn(cx) if G else None)
                for m in CLOSERS:
                    if G:
                        r, y, R, Gm = exit_sets_geo(cx, xr, p, SIDE_OF[m])
                        X = X_geo(cx.Xw, r, NS, Gm)
                    else:
                        r, y, R = OC.exit_sets(cx, xr, p, SIDE_OF[m])
                        X = X_of(cx.Xw, r, NS)
                    R, _ = clip_targets(R, REM_BOUND)
                    store[m] = cat_sets(store[m], (X, y, R, np.ones_like(R)))
        for m in fit_now:
            if len(D_tr[m][1]) == 0 or len(D_in[m][1]) == 0 or len(np.unique(D_tr[m][1])) < 2:
                if m in pols:
                    d["ledger"](d["led"], rung="E3M_C2", event="MODULE_KEPT", module=m, seed=seed, round=rnd)
                    continue
                raise SystemExit(f"C2_EMPTY_PARTITION:{m}:round{rnd}")
            d["ledger"](d["led"], rung="E3M_C2", event="FIT", module=m, seed=seed, round=rnd, rows_train=int(len(D_tr[m][1])),
                        rows_inner=int(len(D_in[m][1])), label_counts=np.bincount(D_tr[m][1], minlength=2).tolist())
            pols[m], infos[m] = d["fit_module"]("FULL", m, D_tr[m], D_in[m], seed)
        cal = calibrate(pols, D_in)
        table = calibrate_table(raw_table(pols, cx.Xw, d["rows_all"], NS, closers=not G), cal)
        if G:
            table["closer"] = CloserGeo(cx, [(pols, cal)], NS)
        if rnd == d["rounds"]:
            break
        fr = freeze(table, d["cal_rows"], score_table, AC, seed, "visitation", primary, closer_src)
        st_ = strategy_of(fr)
        rounds.append({"round": rnd, "visitation_strategy": fr["strategy"], "valid": fr["valid"], "chosen": fr["chosen"],
                       "fallback": None if st_ else "no valid visitation strategy on INNER_A: the round adds no visited closer rows",
                       "closer_rows": {m: int(len(D_tr[m][1])) for m in CLOSERS}})
        d["ledger"](d["led"], rung="E3M_C2", event="VISITATION_STRATEGY", seed=seed, round=rnd, strategy=fr["strategy"])
        comp = PolicyC2(table, st_) if st_ is not None else None
    fr = freeze(table, d["cal_rows"], score_table, AC, seed, "final", primary, closer_src)
    st_ = strategy_of(fr) or NO_TRADE
    d["ledger"](d["led"], rung="E3M_C2", event="FREEZE", seed=seed, strategy=fr["strategy"], chosen=fr["chosen"], chosen_on="INNER_A")
    extra = {"calibration": {m: {k: c[k] for k in ("a", "b", "rows", "positives", "single_class", "before", "after")} for m, c in cal.items()},
             "strategy": fr, "visitation_rounds": rounds, "targets_clipped": clipped,
             "heads_calibration_block": head_report(pols, cal, D_in),
             "best_epochs": {m: int(infos[m]["best_epoch"]) for m in infos}}
    return pols, infos, extra, PolicyC2(table, st_), table


# ----------------------------------------------------------------------------- the E3-M run under C2
def run_e3m(ctx: dict) -> dict:
    """Called by ``tools/expert_guided_e3m.py run`` when the action contract is a C2 contract (after the shared data,
    recipe, architecture and comparator setup). Writes the checkpoints, PROBE.json, C2_SETS.npz and returns ``res``."""
    import torch
    cx, ds, a, AC, res, out = ctx["cx"], ctx["ds"], ctx["a"], ctx["AC"], ctx["res"], Path(ctx["out"])
    SCFG, ECFG, MCFG, NS, led, ledger, summary = ctx["SCFG"], ctx["ECFG"], ctx["MUSASHI_CFG"], ctx["NS"], ctx["led"], ctx["ledger"], ctx["summary"]
    DEV, SEEDS = ctx["DEV"], ctx["SEEDS"]
    # outcome tables (engine replay of every entry): built once per fold and stamped; reused when a stamped file exists
    op = Path(a.outcomes) if getattr(a, "outcomes", None) else out / "OUTCOMES_ENTRY.npz"
    if op.exists():
        et = OC.load_table(op, cx, SCFG)
    else:
        et = OC.entry_table(cx, SCFG, progress=lambda i, n: print(json.dumps({"outcomes": f"{i}/{n}"}), flush=True))
        op.parent.mkdir(parents=True, exist_ok=True)
        OC.save_table(op, et)
    res["c2"] = {"outcomes_file": str(op), "outcomes_stamp": et["stamp"], "outcomes_refused": et["refused"],
                 "eligible_rows": {p: int((et["eligible"] & (ds.period == p)).sum()) for p in ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")},
                 "teacher": "none (the expert only supplies round-0 visited states)", "contract_rule": AC["strategy"]["rule"]}
    # round-0 exit visits (states only): the expert's trajectory + exploration positions, per period
    G = is_c2g(AC)
    if G:
        res["c2"]["closer_order_geometry"] = {"inputs": list(GEOMETRY), "applies_to": list(CLOSERS), "hours_scale": HOURS_SCALE,
                                              "closer_n_state": NS + N_GEO, "openers_and_nop_n_state": NS,
                                              "owner_words": "Sí, como brazo extra (2026-10-25)"}
    x0 = {}
    for p in ("TRAIN", "INNER_A", "INNER_B"):
        vis = OC.visit_positions(cx, p, cx.expert, ECFG) + OC.visit_positions(cx, p, cx.prior_random(ctx["prior"], 0), SCFG)
        x0[p] = OC.exit_table(cx, SCFG, vis, geometry_fn=geometry_fn(cx) if G else None)
    res["c2"]["round0_exit_visits"] = {p: int(len(x0[p]["row"])) for p in x0}
    mr = MR.MultiResLabels(cx, MR.Thresholds(), MR.LADDER, ctx["EXPERT"])
    nopl = ACT.NopLabels(mr)
    rows_all = np.flatnonzero(cx.obs_ok)
    d = {"contract": AC, "fit_module": ctx["fit_module"], "SCFG": SCFG, "NS": NS, "rows_all": rows_all,
         "cal_rows": np.flatnonzero(cx.obs_ok & (ds.period == AC["blocks"]["calibration"])), "et": et, "x0": x0, "nopl": nopl,
         "rounds": ctx["ROUNDS"], "led": led, "ledger": ledger, "fitness": a.strategy_fitness or AC["strategy"]["fitness"]["default"],
         "sig": np.asarray(ds.sigma, float)}
    res["action_contract"]["primary_fitness"] = d["fitness"]
    named = {}
    for p in ("INNER_A", "INNER_B"):
        go = {}
        Dp, rp = sets(cx, et, x0[p], p, ACT.NopLabels(mr), NS, geo=G, geo_out=go)
        named[p] = {m: (rp[m], Dp[m][1], Dp[m][2] if len(Dp[m]) > 2 else None, go.get(m)) for m in SPECIALISTS}
    save_sets(out / "C2_SETS.npz", named)
    D_ib = sets_from_saved(cx, {m: {"rows": r, "y": y, **({"R": R} if R is not None else {}), **({"G": g} if g is not None else {})}
                                for m, (r, y, R, g) in named["INNER_B"].items()}, NS)
    frozen = {}
    for seed in SEEDS:
        pols, infos, extra, c, table = run_seed(cx, seed, d)
        frozen[seed] = c
        extra["reported_only_INNER_B"] = {"heads": head_report(pols, extra["calibration"], D_ib),
                                          "note": "frozen calibration and heads; nothing fitted or chosen on INNER_B"}
        chosen = extra["strategy"]["chosen"]
        best = {"arbiter": "action_contract_c2", "rate": None, "tau": None, "strategy": extra["strategy"]["strategy"],
                "nop_veto_kept": extra["strategy"]["nop_veto_kept"], "sharpe": None if chosen is None else chosen["sharpe"],
                "lcb": None if chosen is None else chosen["lcb"], "valid": chosen is not None}
        rec = {"fits": {m: {k: infos[m][k] for k in ("epochs", "best_epoch", "params", "stop_reason")} for m in infos},
               "arbiter_trials": extra["strategy"]["runs"]["with_veto"]["front"] + extra["strategy"]["runs"]["without_veto"]["front"],
               "arbiter": best, "trading": {}, "musashi_contract": {}, "action_contract": extra}
        for p in DEV:
            rec["trading"][p] = summary(cx.rollout(p, c(cx), SCFG)[0])
            rec["musashi_contract"][p] = summary(cx.rollout(p, c(cx), MCFG)[0])
        for f, bf in extra["strategy"]["by_fitness"].items():
            if bf["strategy"] is not None:
                cf = PolicyC2(table, strategy_from_dict(bf["strategy"]))
                bf["trading"] = {p: summary(cx.rollout(p, cf(cx), SCFG)[0]) for p in DEV}
            bu = bf["archive_view"]["best_unconstrained_by_fitness"]
            if bu is not None:
                cu = PolicyC2(table, strategy_from_dict(bu["strategy"]))
                bu["trading_not_selectable"] = {p: summary(cx.rollout(p, cu(cx), SCFG)[0]) for p in DEV}
        rec["action_contract"]["out_of_sample_activity"] = {
            "INNER_B_trades_per_week": rec["trading"].get("INNER_B", {}).get("orders_per_week"),
            "owner_rule": "below 2 trades/week on INNER_B = INVALID", "valid_INNER_B": rec["trading"].get("INNER_B", {}).get("valid")}
        ci_p = {p: ACT.period_activity(cx, p, c(cx), SCFG) for p in DEV} if ACT.is_v13(AC["strategy"]) else {}
        if ci_p:
            rec["action_contract"]["activity_ci90"] = ci_p
        for m, pol in pols.items():
            torch.save(pol.m.state_dict(), out / f"FULL_{m}_{seed}.pt")
            res["probe"][f"FULL_{m}_{seed}"] = ctx["probe"](pol)
        (out / "PROBE.json").write_text(json.dumps({"split_contract": a.split_contract, "agent_state": a.agent_state,
                                                    "scaling": a.scaling, "probe": res["probe"]}, indent=1))
        res["trials"].setdefault("FULL", {})[str(seed)] = rec
        print(json.dumps({"variant": "FULL", "seed": seed, "arbiter": "action_contract_c2", "strategy": best["strategy"],
                          "sharpe": {p: round(rec["trading"][p]["sharpe_daily_sqrt365"], 2) for p in rec["trading"]},
                          "opw": {p: round(rec["trading"][p]["orders_per_week"], 2) for p in rec["trading"]},
                          "seed_sanity": {p: ACT.seed_sanity(rec["trading"][p], ci_p.get(p)) for p in rec["trading"]}}), flush=True)
    DEC = AC["blocks"]["decision"]
    sh = [res["trials"]["FULL"][str(s)]["trading"][DEC]["sharpe_daily_sqrt365"] for s in SEEDS]
    res["decision"] = {"means": {"FULL": {"decision_block": DEC, "seed_mean_sharpe": float(np.mean(sh)),
                                          "all_valid": all(res["trials"]["FULL"][str(s)]["trading"][DEC]["valid"] for s in SEEDS)}},
                       "chosen": "FULL", "rule": "C2: single FULL variant; decision between arms by the seed ENSEMBLE on INNER_B (report)"}
    ledger(led, rung="E3M_C2", event="READ", periods=["OBS_2024"], purpose="single read per seed")
    for seed in SEEDS:
        c = frozen[seed]
        rec = res["trials"]["FULL"][str(seed)]
        rec["trading"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), SCFG)[0])
        rec["musashi_contract"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), MCFG)[0])
    return res


# ----------------------------------------------------------------------------- seed tables (ensemble / audit)
def seed_table(pols, cx, NS, calibration, geo: bool = False) -> dict:
    t = calibrate_table(raw_table(pols, cx.Xw, np.flatnonzero(cx.obs_ok), NS, closers=not geo), calibration)
    if geo:
        t["closer"] = CloserGeo(cx, [(pols, calibration)], NS)
    return t


def ensemble_calibration(ens_table: dict, cx, saved_inner_a: dict) -> tuple:
    """ONE Platt map per specialist refitted on the ensemble (mean calibrated probabilities -> logits) over the saved
    round-0 INNER_A label rows; returns (calibrated ensemble table, calibration doc)."""
    raw = to_raw(ens_table)
    cal = {}
    for m in SPECIALISTS:
        r, y = saved_inner_a[m]["rows"], np.asarray(saved_inner_a[m]["y"], int)
        if m in OPENERS:
            z = raw["z_ptp"][r, SIDE_OF[m]]
        elif m in CLOSERS:
            z = raw["closer"].raw_logit_rows(m, r, saved_inner_a[m]["G"]) if "closer" in raw else raw["z_psl"][r, SIDE_OF[m]]
        else:
            z = raw["z_pn"][r]
        a_, b_ = ACT.fit_platt(z, y)
        cal[m] = {"a": a_, "b": b_, "rows": int(len(y)), "before": ACT.head_metrics(ACT.sigmoid(z), y), "after": ACT.head_metrics(ACT.platt(z, a_, b_), y)}
    return calibrate_table(raw, cal), cal


def exit_prob(cx, table):
    """P(SL first) of the open position's side at the decision row (the early-close audit's closer score)."""
    if "closer" in table:                                # C2G: the closer's score needs the open order itself
        def g(bar, st, pos):
            return table["closer"].value(int(cx.rob[bar]), bar, pos)[1]
        g.wants_position = True
        return g

    def f(bar, st):
        return table["psl"][cx.rob[bar], 0 if int(st[0]) > 0 else 1]
    return f
