"""Output (action) contract v1 of the E3-M agent (owner 2026-10-25: the outputs deserve the same rigour as the inputs;
design ``docs/short_horizon/clean_round/OUTPUT_CONTRACT_AUDIT_AND_DESIGN_2026_10_25.md``).

The contract (``ACTION_CONTRACT_v1.json``, identity ``action_<16 hex>`` = sha256 of its canonical JSON without the
identity field, file pinned by sha256 like the recipe) declares:

* the action space {HOLD (= NOP when flat), OPEN_LONG, OPEN_SHORT, CLOSE}; an open of the opposite side while
  positioned is close-and-reverse, allowed only when the strategy's ``allow_reverse`` is true;
* five specialists, each a binary softmax classifier trained with class-weighted cross-entropy: ``open_l``,
  ``open_s`` (each with TP / SL heads in causal sigma, bounded INSIDE the network by lo + (hi - lo) * sigmoid(z) at
  the env bounds, Huber loss masked to the rows whose label is an OPEN of that side), ``nop``, ``close_l``,
  ``close_s``;
* calibration: Platt (affine) scaling of each specialist's logit margin, p = sigmoid(a * z + b), fitted on the
  calibration block. A temperature (a = 1 / T) alone cannot remove the prior shift that the 1/sqrt(freq) class
  weights put into the logits; the bias b does;
* one parametric strategy (the upper layer) over the calibrated probabilities: open threshold, expected-gain margin
  (gain = p * TP - (1 - p) * SL - cost, every term in sigma; cost = the declared COST_SIGMA allowance PLUS the replay
  contract's real round-trip cost converted to sigma at the row), close threshold, close-and-reverse permission and,
  when the pre-registered test keeps it, the NOP veto. NSGA-II search (Sharpe up, max drawdown down) on the
  calibration block only; a declared FITNESS variant (``FITNESS``: F0 sharpe_lcb by default, F1 sharpe_plus_return,
  F2 sharpe_plus_return_plus_activity, F3 trade_tstat (reused from app/walk_forward_optimizer.py), F4 calmar,
  F5 return_minus_dd (agent-multi sac_agent.fitness), F6 balanced; composites scaled by ``term_scales``) picks the
  frozen point among the valid front points;
  feasibility = >= 2 trades/week (period average, no upper cap) on the calibration block, at least one trade and a
  defined Sharpe; the search uses Deb's constraint domination and every evaluated point is archived (never discarded,
  never -inf: undefined values are NaN with their reason);
* every evaluated strategy point of every search is archived with its components (nothing discarded, never -inf);
* the blocks: INNER_A (inner semester) = early stopping, calibration and strategy optimisation; INNER_B (after the
  freeze) = decision, scored once; OBS_2024 = one read. A split without a post-freeze decision block is refused.

Everything here is a function of the rows it is given: ``restrict`` blanks every row outside the calibration block
and ``ContractPolicy`` refuses a blank row, so a strategy optimised on a restricted table cannot read another block.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import env as E
from . import multires as MR
from . import stats as ST

SCHEMA = "expert_guided_action_contract.v1"
SPECIALISTS = ("open_l", "open_s", "nop", "close_l", "close_s")
OPENERS, CLOSERS = ("open_l", "open_s"), ("close_l", "close_s")
FIELDS = ("pL", "tpL", "slL", "pS", "tpS", "slS", "pN", "pCL", "pCS", "cost")
PROB_OF = {"open_l": "pL", "open_s": "pS", "nop": "pN", "close_l": "pCL", "close_s": "pCS"}
STRATEGY_GENES = ("open_threshold", "gain_margin", "close_threshold", "allow_reverse", "nop_veto")
NOP_POSITIVE = ("nop_hard", "nop_graded", "negative", "conflict")


# ----------------------------------------------------------------------------- identity / validation
def canonical_bytes(doc: dict) -> bytes:
    return json.dumps({k: v for k, v in doc.items() if k != "identity"}, sort_keys=True, separators=(",", ":")).encode()


def identity_of(doc: dict) -> str:
    return "action_" + hashlib.sha256(canonical_bytes(doc)).hexdigest()[:16]


def seal(doc: dict) -> dict:
    out = dict(doc)
    out["identity"] = identity_of(out)
    return out


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(doc: dict) -> list:
    """Typed refusals; [] when the contract is well formed and consistent with the env and the multires labels."""
    errs = []
    if doc.get("schema") != SCHEMA:
        errs.append("ACTION_CONTRACT_SCHEMA")
    if doc.get("identity") != identity_of(doc):
        errs.append("ACTION_CONTRACT_IDENTITY")
    if doc.get("scientific_release") is not False:
        errs.append("ACTION_CONTRACT_RELEASE_FLAG")
    sp = doc.get("specialists") or {}
    if tuple(sp) != SPECIALISTS:
        errs.append(f"ACTION_CONTRACT_SPECIALISTS:{list(sp)}")
    errs += encoding_audit(doc)["errors"]
    cfg = E.EnvConfig()
    for m in OPENERS:
        ct = (sp.get(m) or {}).get("continuous") or {}
        tp, sl = ct.get("tp_sigma") or {}, ct.get("sl_sigma") or {}
        if (tp.get("lo"), tp.get("hi")) != tuple(cfg.tp_bounds) or (sl.get("lo"), sl.get("hi")) != tuple(cfg.sl_bounds):
            errs.append(f"ACTION_CONTRACT_TP_SL_BOUNDS:{m}")
        if any(x.get("mask") != "rows whose label is an OPEN of this side" for x in (tp, sl)):
            errs.append(f"ACTION_CONTRACT_TP_SL_MASK:{m}")
    nl = (sp.get("nop") or {}).get("labels") or {}
    if tuple(nl.get("ladder", ())) != tuple(MR.LADDER) or tuple(nl.get("positive", ())) != NOP_POSITIVE:
        errs.append("ACTION_CONTRACT_NOP_LABELS")
    cal = doc.get("calibration") or {}
    if cal.get("method") != "platt_binary" or cal.get("fit_block") != "INNER_A" or int(cal.get("ece_bins", 0)) != 15:
        errs.append("ACTION_CONTRACT_CALIBRATION")
    s = doc.get("strategy") or {}
    genes = s.get("genes") or {}
    if tuple(genes) != STRATEGY_GENES:
        errs.append(f"ACTION_CONTRACT_GENES:{list(genes)}")
    for g, b in genes.items():
        if not (isinstance(b, list) and len(b) == 2 and float(b[0]) < float(b[1])):
            errs.append(f"ACTION_CONTRACT_GENE_BOUNDS:{g}")
    opt = s.get("optimizer") or {}
    for k in ("final", "visitation"):
        o = opt.get(k) or {}
        if int(o.get("population", 0)) < 4 or int(o.get("population", 0)) % 4 or int(o.get("generations", 0)) < 1:
            errs.append(f"ACTION_CONTRACT_OPTIMIZER:{k}")
    if float(s.get("calibration_min_trades_per_week", 0)) != MIN_TRADES_PER_WEEK:
        errs.append("ACTION_CONTRACT_ACTIVITY_RULE")
    if (opt.get("constraint") or "").split(":")[0] != "Deb constraint domination":
        errs.append("ACTION_CONTRACT_CONSTRAINT_HANDLING")
    if s.get("choice") != "the arm's fitness variant picks the frozen point among the valid non-dominated (Sharpe, DD) points; ties -> lower max DD":
        errs.append("ACTION_CONTRACT_CHOICE")
    fz = s.get("fitness") or {}
    if fz.get("default") != DEFAULT_FITNESS or sorted(fz.get("variants") or {}) != sorted(FITNESS):
        errs.append("ACTION_CONTRACT_FITNESS")
    if s.get("fit_block") != "INNER_A":
        errs.append("ACTION_CONTRACT_STRATEGY_BLOCK")
    bl = doc.get("blocks") or {}
    uses = bl.get("calibration_block_uses") or []
    if bl.get("calibration") != "INNER_A" or bl.get("decision") != "INNER_B" or bl.get("read_once") != "OBS_2024" \
            or sorted(uses) != sorted(["early_stopping", "calibration", "strategy_optimisation", "dagger_visitation_strategy"]) \
            or bl.get("requires_post_freeze_decision_block") is not True:
        errs.append("ACTION_CONTRACT_BLOCKS")
    c = doc.get("constraints") or {}
    if c.get("agent_state") != "none" or c.get("variants") != ["FULL"]:
        errs.append("ACTION_CONTRACT_CONSTRAINTS")
    return errs


CATEGORICAL_ENCODING, CONTINUOUS_ENCODING = "softmax_ce", "bounded_sigmoid"
FROM_CONTINUOUS_MARKERS = ("bins", "thresholds", "intervals", "derived_from_continuous", "source_continuous")


def encoding_audit(doc: dict) -> dict:
    """The output-encoding audit (owner 2026-10-25; the failure it guards against: one continuous output cut into
    intervals to choose among commands). Every categorical decision must be ``softmax_ce`` with n_classes = len(classes)
    = output_dim; every continuous parameter ``bounded_sigmoid`` with lo < hi, output_dim 1 and a declared loss; every
    action must be produced by some (head, class); every strategy threshold must read a categorical probability.
    Refusals: CATEGORICAL_FROM_CONTINUOUS, CONTINUOUS_THROUGH_SOFTMAX, HEAD_DIM_MISMATCH, ENCODING_UNKNOWN,
    CONTINUOUS_BOUNDS, CONTINUOUS_LOSS, ACTION_UNREACHABLE. Returns {"errors": [...], "heads": [...], "verdict"}."""
    errs, heads = [], []
    sp = doc.get("specialists") or {}
    classes = {}
    for m, d in sp.items():
        dec = d.get("decision")
        if not isinstance(dec, dict):
            errs.append(f"ENCODING_UNKNOWN:{m}:no categorical decision declared")
            continue
        enc = str(dec.get("encoding", ""))
        if any(k in dec for k in FROM_CONTINUOUS_MARKERS) or any(k in enc for k in ("interval", "threshold", "bin", "continuous", "tanh", "regression")):
            errs.append(f"CATEGORICAL_FROM_CONTINUOUS:{m}")
        elif enc != CATEGORICAL_ENCODING:
            errs.append(f"ENCODING_UNKNOWN:{m}:{enc}")
        cl = list(dec.get("classes") or [])
        if not (int(dec.get("n_classes", -1)) == len(cl) == int(dec.get("output_dim", -2)) and len(cl) >= 2):
            errs.append(f"HEAD_DIM_MISMATCH:{m}:n_classes={dec.get('n_classes')} classes={len(cl)} output_dim={dec.get('output_dim')}")
        if dec.get("loss") != "cross_entropy":
            errs.append(f"CATEGORICAL_LOSS:{m}")
        classes[m] = set(cl)
        heads.append({"head": f"{m}.decision", "encoding": enc, "n_classes": dec.get("n_classes"), "classes": cl, "output_dim": dec.get("output_dim")})
        for q, c in (d.get("continuous") or {}).items():
            ce = str(c.get("encoding", ""))
            if ce.startswith("softmax"):
                errs.append(f"CONTINUOUS_THROUGH_SOFTMAX:{m}.{q}")
            elif ce != CONTINUOUS_ENCODING:
                errs.append(f"ENCODING_UNKNOWN:{m}.{q}:{ce}")
            if not (isinstance(c.get("lo"), (int, float)) and isinstance(c.get("hi"), (int, float)) and c["lo"] < c["hi"]):
                errs.append(f"CONTINUOUS_BOUNDS:{m}.{q}")
            if int(c.get("output_dim", -1)) != 1:
                errs.append(f"HEAD_DIM_MISMATCH:{m}.{q}:output_dim={c.get('output_dim')}")
            if c.get("loss") not in ("huber",):
                errs.append(f"CONTINUOUS_LOSS:{m}.{q}")
            heads.append({"head": f"{m}.{q}", "encoding": ce, "lo": c.get("lo"), "hi": c.get("hi"), "output_dim": c.get("output_dim"), "loss": c.get("loss")})
    asp = doc.get("action_space") or {}
    for act_ in asp.get("actions") or {}:
        prod = (asp.get("produced_by") or {}).get(act_) or []
        if not any(isinstance(x, list) and len(x) == 2 and x[0] in classes and x[1] in classes[x[0]] for x in prod):
            errs.append(f"ACTION_UNREACHABLE:{act_}")
    for g, tgt in ((doc.get("strategy") or {}).get("threshold_targets") or {}).items():
        for t in str(tgt).split("|"):
            h, _, c = t.partition(".")
            if h not in classes or c not in classes[h]:
                errs.append(f"CATEGORICAL_FROM_CONTINUOUS:strategy.{g}:{t} is not a class of a categorical head")
    return {"errors": errs, "heads": heads, "verdict": "PASS" if not errs else "REFUSED",
            "rule": "categorical = softmax_ce over its declared classes; continuous = bounded_sigmoid; no class is ever read off an interval of a continuous output"}


def check_head_shapes(doc: dict, nets: dict) -> dict:
    """Runtime check (E3-M at startup, the ensemble after loading): every network's class head has n_classes outputs, its
    regression head one output per declared continuous parameter, and its bounds equal the declared lo / hi."""
    out = {}
    for m, d in doc["specialists"].items():
        net = nets.get(m)
        if net is None:
            raise SystemExit(f"HEAD_SHAPE_MISMATCH:{m}:no network")
        nc = int(net.cls.out_features)
        cont = list((d.get("continuous") or {}).values())
        nr = int(getattr(net, "n_reg", 0) or 0)
        if nc != int(d["decision"]["n_classes"]) or nr != len(cont):
            raise SystemExit(f"HEAD_SHAPE_MISMATCH:{m}:classes {nc}!={d['decision']['n_classes']} or continuous {nr}!={len(cont)}")
        if nr:
            lo, hi = [float(x) for x in net.lo.tolist()], [float(x) for x in net.hi.tolist()]
            if lo != [float(c["lo"]) for c in cont] or hi != [float(c["hi"]) for c in cont]:
                raise SystemExit(f"HEAD_SHAPE_MISMATCH:{m}:bounds {lo}/{hi}")
        out[m] = {"classes": nc, "continuous": nr, "status": "MATCH"}
    return out


def load(path) -> dict:
    doc = json.loads(Path(path).read_text())
    errs = validate(doc)
    if errs:
        raise SystemExit(f"ACTION_CONTRACT_INVALID:{errs}")
    return doc


def require_decision_block(doc: dict, split: dict, name: str) -> None:
    """The contract runs only on a split whose decision block exists and starts at or after the calibration block's end
    (``consolidated`` has no such block: refused, never silently scored on the calibration block)."""
    per = split["periods"]
    cb, db = doc["blocks"]["calibration"], doc["blocks"]["decision"]
    if cb not in per or db not in per or per[db][0] < per[cb][1]:
        raise SystemExit(f"ACTION_CONTRACT_NO_POST_FREEZE_DECISION_BLOCK:{name}")


# ----------------------------------------------------------------------------- calibration and head metrics
def sigmoid(x):
    x = np.asarray(x, float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def margin(logits) -> np.ndarray:
    """Binary softmax: P(class 1) = sigmoid(l1 - l0), so the margin carries the whole prediction."""
    lg = np.asarray(logits, float)
    return lg[:, 1] - lg[:, 0]


def fit_platt(z, y, l2=1e-6, iters=100, tol=1e-12) -> tuple:
    """(a, b) minimising the mean log loss of sigmoid(a z + b) on (z, y) by Newton's method (convex; a tiny ridge keeps a
    separable sample finite). Deterministic. (1, 0) when y has one class (nothing to calibrate; recorded by the caller)."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    if len(z) == 0 or y.min() == y.max():
        return 1.0, 0.0
    w = np.array([1.0, 0.0])
    X = np.c_[z, np.ones_like(z)]
    n = len(z)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / n + l2 * w
        H = (X.T * (p * (1 - p))) @ X / n + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if float(np.abs(step).max()) < tol:
            break
    return float(w[0]), float(w[1])


def platt(z, a, b) -> np.ndarray:
    return sigmoid(a * np.asarray(z, float) + b)


def auc(p, y) -> float:
    """Mann-Whitney AUC with average ranks for ties; nan when one class is absent."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(p, y, bins=15) -> dict:
    """Equal-mass bins (each ~ len/bins rows, sorted by p): per-bin mean prediction and observed rate; ECE = the
    row-weighted mean |mean p - rate| over bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return {"ece": float("nan"), "bins": []}
    order = np.argsort(p, kind="mergesort")
    parts = np.array_split(order, min(bins, len(p)))
    rows, ece = [], 0.0
    for k, ix in enumerate(parts):
        if not len(ix):
            continue
        mp, mr = float(p[ix].mean()), float(y[ix].mean())
        ece += len(ix) / len(p) * abs(mp - mr)
        rows.append({"bin": k, "n": int(len(ix)), "p_lo": float(p[ix].min()), "p_hi": float(p[ix].max()), "p_mean": mp, "rate": mr})
    return {"ece": float(ece), "bins": rows}


def head_metrics(p, y, bins=15) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, int)
    rel = reliability(p, y, bins)
    return {"n": int(len(y)), "positives": int(y.sum()), "auc": auc(p, y), "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
            "ece": rel["ece"], "reliability": rel["bins"]}


# ----------------------------------------------------------------------------- per-row output table
def module_outputs(pol, X, batch=4096):
    """(margin z, regression outputs or None) of a TorchPolicy-like callable over X, in batches."""
    zs, rs = [], []
    for i in range(0, len(X), batch):
        lg, rg = pol(X[i:i + batch])
        zs.append(margin(lg))
        rs.append(None if rg is None else np.asarray(rg, float))
    z = np.concatenate(zs) if zs else np.zeros(0)
    r = None if not rs or rs[0] is None else np.concatenate(rs)
    return z, r


def raw_table(pols: dict, Xw, rows, n_state: int, cost) -> dict:
    """Uncalibrated per-row outputs (the agent state is all zeros under the contract: agent_state none) for ``rows``;
    NaN elsewhere. The probability fields (pL, pS, pN, pCL, pCS) hold the raw logit MARGINS here; ``calibrate_table``
    maps them to calibrated probabilities. ``cost`` (per row, sigma) is carried as is."""
    n = len(Xw)
    t = {f: np.full(n, np.nan) for f in FIELDS}
    rows = np.asarray(rows, np.int64)
    X = np.c_[Xw[rows], np.zeros((len(rows), n_state), np.float32)].astype(np.float32)
    for m, (zf, tpf, slf) in (("open_l", ("pL", "tpL", "slL")), ("open_s", ("pS", "tpS", "slS"))):
        z, r = module_outputs(pols[m], X)
        t[zf][rows], t[tpf][rows], t[slf][rows] = z, r[:, 0], r[:, 1]
    for m in ("nop", "close_l", "close_s"):
        t[PROB_OF[m]][rows] = module_outputs(pols[m], X)[0]
    t["cost"][rows] = np.asarray(cost, float)[rows]
    return t


def calibrate_table(raw: dict, cal: dict) -> dict:
    """Probability table: every specialist margin through its Platt map; TP/SL and cost unchanged."""
    t = {k: v.copy() for k, v in raw.items()}
    for m, f in PROB_OF.items():
        a, b = cal[m]["a"], cal[m]["b"]
        t[f] = np.where(np.isfinite(raw[f]), platt(np.nan_to_num(raw[f]), a, b), np.nan)
    return t


def mean_tables(tables: list) -> dict:
    """Seed ensemble: mean calibrated probabilities and mean TP/SL; the cost must be identical across members."""
    out = {f: np.mean([t[f] for t in tables], axis=0) for f in FIELDS if f != "cost"}
    c0 = tables[0]["cost"]
    if any(not np.array_equal(np.isnan(t["cost"]), np.isnan(c0)) or not np.allclose(np.nan_to_num(t["cost"]), np.nan_to_num(c0)) for t in tables):
        raise ValueError("ACTION_CONTRACT_ENSEMBLE_COST_MISMATCH")
    out["cost"] = c0.copy()
    return out


def restrict(table: dict, rows) -> dict:
    """A copy with every row outside ``rows`` blanked (NaN): what the strategy optimiser is allowed to see."""
    keep = np.zeros(len(table["cost"]), bool)
    keep[np.asarray(rows, np.int64)] = True
    return {f: np.where(keep, v, np.nan) for f, v in table.items()}


def row_cost_sigma(round_trip_price: float, sigma_price, row_bar, cost_sigma: float) -> np.ndarray:
    """Per row, in sigma: the declared COST_SIGMA allowance plus the replay contract's round-trip cost (price units)
    divided by the causal sigma (price units) of the fill bar (bar + 1). Both terms are dimensionless sigma multiples."""
    sig = np.asarray(sigma_price, float)
    rb = np.asarray(row_bar, np.int64)
    s = np.where((rb >= 0) & (rb + 1 < len(sig)), sig[np.clip(rb + 1, 0, len(sig) - 1)], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        real = np.where(np.isfinite(s) & (s > 0), float(round_trip_price) / s, np.nan)
    return float(cost_sigma) + real


# ----------------------------------------------------------------------------- the strategy (upper layer)
@dataclass(frozen=True)
class Strategy:
    open_threshold: float
    gain_margin: float
    close_threshold: float
    allow_reverse: bool
    nop_veto: float | None          # None = veto disabled (pre-registered test failed or not offered)

    @classmethod
    def from_genes(cls, g: dict, veto: bool):
        return cls(float(g["open_threshold"]), float(g["gain_margin"]), float(g["close_threshold"]), bool(float(g["allow_reverse"]) >= 0.5),
                   float(g["nop_veto"]) if veto else None)

    def to_dict(self):
        return dataclasses.asdict(self)


def expected_gain(p, tp, sl, cost):
    """p * TP - (1 - p) * SL - cost: every term in causal sigma multiples."""
    return p * tp - (1.0 - p) * sl - cost


class ContractPolicy:
    """The frozen upper layer over a calibrated table (rows -> probabilities, TP/SL, cost). Same call contract as
    ``Composed``: ``policy(cx)`` -> f(bar, r, st, env) -> (action, tp, sl). A blank (NaN) row raises: the policy can
    only act on the rows its table was built for."""

    def __init__(self, table: dict, strategy: Strategy):
        self.t, self.s = table, strategy

    def _get(self, f, r):
        x = self.t[f][r]
        if not np.isfinite(x):
            raise ValueError(f"ACTION_CONTRACT_ROW_OUTSIDE_TABLE:{f}:{r}")
        return float(x)

    def open_decision(self, r):
        s, g = self.s, self._get
        if s.nop_veto is not None and g("pN", r) > s.nop_veto:
            return None
        best = None
        for d, pf, tpf, slf in ((E.LONG, "pL", "tpL", "slL"), (E.SHORT, "pS", "tpS", "slS")):
            p, tp, sl = g(pf, r), g(tpf, r), g(slf, r)
            gain = expected_gain(p, tp, sl, g("cost", r))
            if p > s.open_threshold and gain > s.gain_margin and (best is None or gain > best[0]):
                best = (gain, d, tp, sl)
        return None if best is None else best[1:]

    def __call__(self, cx):
        def f(bar, r, st, env):
            pos = int(st[0])
            if pos != 0:
                pc = self._get("pCL" if pos > 0 else "pCS", r)
                if pc <= self.s.close_threshold:
                    return E.HOLD, 1.5, 2.0
                o = self.open_decision(r)
                if o is not None and self.s.allow_reverse and o[0] != (E.LONG if pos > 0 else E.SHORT):
                    return o[0], float(o[1]), float(o[2])          # close-and-reverse (env CLOSE_AND_OPEN)
                return E.CLOSE, 1.5, 2.0
            o = self.open_decision(r)
            return (E.HOLD, 1.5, 2.0) if o is None else (o[0], float(o[1]), float(o[2]))
        return f


def rollout_with_trades(cx, period, act, cfg):
    """``Ctx.rollout`` (same loop, same env, same HOLD on unobservable rows) that also returns the env's trade records,
    so the per-trade fitness can be computed without touching the shared engine."""
    a, b = cx.ds.meta["periods"][period] if isinstance(period, str) else period
    env = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, cfg)
    for bar in range(env.first, env.last):
        r = cx.rob[bar]
        st = env.state(bar)
        if r >= 0 and cx.obs_ok[r]:
            action, tp, sl = act(bar, r, st, env)
        else:
            action, tp, sl = E.HOLD, 1.5, 2.0
        env.step(bar, action, tp, sl)
    m = env.result()
    return m, list(env.trades)


TRADING_DAYS_PER_WEEK = 5.0      # orders/day = orders/week / 5: in [0, 1] under the 1 entry/day replay cap


def trade_returns_sigma(trades, sigma_price) -> np.ndarray:
    """Per trade: net result per unit (price units, every cost and financing included) / causal sigma (price units) of
    the fill bar = a dimensionless sigma multiple."""
    sig = np.asarray(sigma_price, float)
    out = []
    for t in trades:
        s = sig[int(t["entry_bar_abs"])]
        out.append(float(t["net_usd"]) / max(int(t["units"]), 1) / s if np.isfinite(s) and s > 0 else np.nan)
    return np.asarray(out, float)


MIN_TRADES_PER_WEEK = 2.0        # the owner's activity rule (period average), a SELECTION criterion (owner 2026-10-25)
BIG = 1e6                        # constraint-domination encoding offset (search only; never a recorded fitness)


def window_components(m: dict, trades, sigma_price, capital: float, min_opw: float = MIN_TRADES_PER_WEEK) -> dict:
    """Every component of one replay, in declared units, its feasibility and its constraint violation. Nothing is
    discarded or set to -inf: a zero-trade point records trades = 0 and Sharpe = NaN with its reason.
      sharpe   annualized daily Sharpe (the engine's sqrt(365) figure; NaN with zero trades)   se_lo  its Lo (2002) SE
      lcb      sharpe - se_lo                                  r_ann  net profit / initial balance x 365 / calendar days of the block
      opd      orders/day = orders/week / 5, in [0, 1] under the 1 entry/day cap      opw  orders/week (period average)
      trade_t  per-trade mean / std (ddof 1) x sqrt(N) in sigma, reported; F3 uses trade_tstat_reused (the reused source)
      calmar   annualized return (%) / max drawdown (%)        return_minus_dd  F5 (agent-multi sac_agent.fitness)
      weeks_below_2  decision weeks with fewer than 2 entries (diagnostic only)
    feasible = >= ``min_opw`` trades/week AND the env's own rules (1 entry/day, not ruined) AND at least one trade AND a
    defined Sharpe; violation = max(0, min_opw - opw) + 1 for each other broken condition (Deb's constraint domination)."""
    n = len(trades)
    ret = ST.daily_returns(m["daily_equity"], capital)
    se = ST.sharpe_se_lo(ret) if n else float("nan")
    sh = float(m["sharpe_daily_sqrt365"]) if n else float("nan")
    reasons = []
    if n == 0:
        reasons.append("zero_trades")
    elif not (math.isfinite(sh) and math.isfinite(se)):
        reasons.append("sharpe_undefined")
    days = max(len(m["daily_equity"]), 1)
    r_ann = float(m["net_profit_usd"]) / float(capital) * 365.0 / days
    tr = trade_returns_sigma(trades, sigma_price)
    tr = tr[np.isfinite(tr)]
    trade_t = float(tr.mean() / tr.std(ddof=1) * math.sqrt(len(tr))) if len(tr) >= 2 and tr.std(ddof=1) > 0 else float("nan")
    dd = float(m["max_drawdown_pct"])
    calmar = (100.0 * r_ann / dd) if dd > 0 else float("nan")
    tstat_reused = trade_tstat(tr)
    opw = float(m["orders_per_week"])
    env_ok = int(m.get("max_entries_per_day", 0)) <= 1 and not bool(m.get("ruined", False))
    if opw < float(min_opw):
        reasons.append(f"below_{min_opw:g}_trades_per_week")
    if not env_ok:
        reasons.append("env_rule_broken")
    violation = max(0.0, float(min_opw) - opw) + (0.0 if env_ok else 1.0) + (1.0 if (n and reasons and reasons[0] == "sharpe_undefined") else 0.0)
    feasible = not reasons
    return {"sharpe": sh, "se_lo": se, "lcb": (sh - se) if math.isfinite(sh) and math.isfinite(se) else float("nan"), "dd": dd,
            "opw": opw, "opd": opw / TRADING_DAYS_PER_WEEK, "r_ann": r_ann, "trade_t": trade_t, "trade_mean_sigma": float(tr.mean()) if len(tr) else float("nan"),
            "trade_tstat_reused": tstat_reused, "calmar": calmar, "return_minus_dd": return_minus_dd(r_ann, dd), "trades": int(n), "days": int(days),
            "weeks_below_2": int(m.get("weeks_below_min_orders", 0)), "weeks_below_2_share": float(m.get("weeks_below_min_share", 0.0)),
            "valid": bool(feasible), "violation": float(violation), "reasons": reasons}


def search_objectives(c: dict) -> tuple:
    """Deb's constraint domination encoded for NSGA-II (weights (+1, -1)): a feasible point keeps (Sharpe, DD); an
    infeasible one gets (-BIG - violation, BIG + violation), so every feasible point dominates every infeasible one and
    between infeasible points the smaller violation dominates. Search only: the recorded components are untouched."""
    if c["valid"] and math.isfinite(c["sharpe"]) and math.isfinite(c["dd"]):
        return (float(c["sharpe"]), float(c["dd"]))
    v = float(c["violation"]) if math.isfinite(c.get("violation", float("nan"))) else BIG
    return (-BIG - v, BIG + v)


def cdominates(a: dict, b: dict) -> bool:
    """Deb's rule, stated directly (the reference for ``search_objectives``)."""
    if a["valid"] and not b["valid"]:
        return True
    if not a["valid"] and not b["valid"]:
        return a["violation"] < b["violation"]
    if not a["valid"] and b["valid"]:
        return False
    return a["sharpe"] >= b["sharpe"] and a["dd"] <= b["dd"] and (a["sharpe"] > b["sharpe"] or a["dd"] < b["dd"])


TRADE_TSTAT_MIN_TRADES = 15      # heuristic-strategy app/walk_forward_optimizer.py::_sharpe_fitness (min_trades=15)
DD_LAMBDA = 1.0                  # agent-multi agent_plugins/sac_agent.py::fitness, default ga_fitness_dd_lambda = 1.0


def trade_tstat(trade_returns_sig) -> float:
    """F3, REUSED: heuristic-strategy ``app/walk_forward_optimizer.py::_sharpe_fitness`` (per-trade mean / std (ddof 0)
    x sqrt(N), rejected below 15 trades) called on the per-trade returns in sigma net of every cost. Its -1e6 rejection
    is recorded here as NaN (undefined: the point stays in the archive and is never selectable by F3)."""
    from app.walk_forward_optimizer import _sharpe_fitness
    tr = [{"pnl": float(x)} for x in np.asarray(trade_returns_sig, float) if np.isfinite(x)]
    v = float(_sharpe_fitness(None, tr, min_trades=TRADE_TSTAT_MIN_TRADES))
    return float("nan") if v <= -1e6 else v


def return_minus_dd(r_ann: float, dd_pct: float, lam: float = DD_LAMBDA) -> float:
    """F5, REIMPLEMENTED with citation (no cross-repo import): agent-multi ``agent_plugins/sac_agent.py::fitness`` =
    total_return - lambda * max_drawdown_pct / 100 with lambda = ga_fitness_dd_lambda = 1.0; the return here is the
    block's net return annualized to the block length (r_ann)."""
    return float(r_ann) - float(lam) * float(dd_pct) / 100.0


# composite terms and their declared fallback scales (owner 2026-10-25; DD% has no owner-declared fallback: 5.0 is
# Satoshi's, declared here and in the contract)
SCALE_TERMS = {"sharpe": lambda c: c["sharpe"], "r_ann": lambda c: c["r_ann"], "opd": lambda c: c["opd"], "dd": lambda c: c["dd"]}
FALLBACK_SCALES = {"sharpe": 1.0, "r_ann": 0.2, "opd": 0.6, "dd": 5.0, "calmar": 5.0, "return_minus_dd": 0.1}
SCALE_FLOOR = 1e-6
COMPOSITES = {   # F = sum(sign_k * term_k / s_k)
    "sharpe_plus_return": (("sharpe", 1), ("r_ann", 1)),
    "sharpe_plus_return_plus_activity": (("sharpe", 1), ("r_ann", 1), ("opd", 1)),
    "balanced": (("sharpe", 1), ("r_ann", 1), ("opd", 1), ("dd", -1)),
}
FITNESS = {   # pre-registered variants (owner 2026-10-25); every term defined in window_components
    "sharpe_lcb": lambda c, sc: c["lcb"],                                                                     # F0
    "sharpe_plus_return": lambda c, sc: _composite("sharpe_plus_return", c, sc),                             # F1 (scaled)
    "sharpe_plus_return_plus_activity": lambda c, sc: _composite("sharpe_plus_return_plus_activity", c, sc), # F2 (scaled)
    "trade_tstat": lambda c, sc: c["trade_tstat_reused"],                                                     # F3 (reused)
    "calmar": lambda c, sc: c["calmar"],                                                                      # F4
    "return_minus_dd": lambda c, sc: return_minus_dd(c["r_ann"], c["dd"]),                                    # F5 (reused)
    "balanced": lambda c, sc: _composite("balanced", c, sc),                                                  # F6 (scaled)
}
DEFAULT_FITNESS = "sharpe_lcb"


def _composite(name, c, scales):
    if scales is None:
        raise ValueError(f"FITNESS_SCALES_REQUIRED:{name}")
    return float(sum(sign * SCALE_TERMS[t](c) / scales[t]["s"] for t, sign in COMPOSITES[name]))


def term_scales(reference: list) -> dict:
    """Pre-registered coefficient rule: s_k = median |term_k| over the VALID points of the reference front (the F0
    search's front on the calibration block of this fold and arm); below 1e-6 the declared fallback scale is used and
    recorded; c_k = 1 / s_k. Never computed on the decision block (the reference front only ever sees INNER_A)."""
    pts = [p for p in reference if p.get("valid")]
    out = {}
    for t, f in SCALE_TERMS.items():
        vals = np.abs(np.array([f(p) for p in pts], float))
        vals = vals[np.isfinite(vals)]
        med = float(np.median(vals)) if len(vals) else float("nan")
        fb = not (math.isfinite(med) and med >= SCALE_FLOOR)
        s = FALLBACK_SCALES[t] if fb else med
        out[t] = {"s": float(s), "c": 1.0 / float(s), "median_abs": med, "fallback_used": bool(fb), "points": int(len(vals))}
    return out


def fitness(name: str, c: dict, scales: dict | None = None) -> float:
    """The variant's value on a point, recorded for EVERY point (valid or not); NaN when a component is undefined (zero
    trades, undefined Sharpe, F3 below 15 trades, F4 with DD = 0), never 0 and never -inf. Selection uses only valid
    points with a finite value (``choose``). Composite variants (F1, F2, F6) need the scales of ``term_scales``."""
    if name not in FITNESS:
        raise ValueError(f"STRATEGY_FITNESS_UNKNOWN:{name}")
    try:
        v = FITNESS[name](c, scales)
    except (KeyError, TypeError):
        return float("nan")
    return float(v) if v is not None and math.isfinite(v) else float("nan")


def _nondominated(pts):
    """Valid points not dominated on the search objectives (Sharpe up, DD down)."""
    out = []
    for p in pts:
        if not any(q is not p and q["sharpe"] >= p["sharpe"] and q["dd"] <= p["dd"] and (q["sharpe"] > p["sharpe"] or q["dd"] < p["dd"]) for q in pts):
            out.append(p)
    return out


def score_front(front: list, scales: dict) -> None:
    """Attach every variant's fitness to every front point (components already recorded)."""
    for r in front:
        r["fitness"] = {f: fitness(f, r, scales) for f in FITNESS}


def choose(front: list, name: str):
    """The frozen point of a fitness variant: the front point with the highest fitness (ties -> lower DD, then the
    gene vector); None when no front point has a finite fitness."""
    cands = [r for r in front if math.isfinite(r["fitness"][name])]
    if not cands:
        return None
    return max(cands, key=lambda r: (r["fitness"][name], -r["dd"], tuple(-x for x in r["genes"].values())))


def optimize(score, genes: dict, seed: int, population: int, generations: int, veto: bool) -> dict:
    """NSGA-II (DEAP) over the bounded genes (``nop_veto`` only when ``veto``); search objectives (Sharpe up, DD down)
    under Deb's constraint domination (``search_objectives``: feasible beats infeasible, smaller violation beats larger),
    so the search stays informed near the activity boundary. EVERY evaluated point of every generation is archived
    with its genes, generation, components, feasibility and violation (nothing discarded); the valid non-dominated
    archive points form the front on which each variant picks (``freeze_strategy``). Deterministic for a seed; the
    global ``random`` state is restored."""
    from deap import base, creator, tools
    names = [g for g in STRATEGY_GENES if veto or g != "nop_veto"]
    lo = [float(genes[g][0]) for g in names]
    hi = [float(genes[g][1]) for g in names]
    if not hasattr(creator, "ACFitnessV1"):
        creator.create("ACFitnessV1", base.Fitness, weights=(1.0, -1.0))
        creator.create("ACIndividualV1", list, fitness=creator.ACFitnessV1)
    state = random.getstate()
    random.seed(int(seed))
    cache, archive, gen = {}, [], [0]

    def evaluate(ind):
        key = tuple(round(float(x), 6) for x in ind)
        if key not in cache:
            g = dict(zip(names, key))
            if not veto:
                g["nop_veto"] = 1.0
            sc = score(Strategy.from_genes(g, veto))
            cache[key] = {"genes": dict(zip(names, key)), "first_generation": gen[0], "veto": veto, **sc}
        archive.append({"generation": gen[0], "key": list(key)})                 # every evaluation, all generations
        return search_objectives(cache[key])

    try:
        pop = [creator.ACIndividualV1([random.uniform(a, b) for a, b in zip(lo, hi)]) for _ in range(population)]
        for ind in pop:
            ind.fitness.values = evaluate(ind)
        pop = tools.selNSGA2(pop, len(pop))
        for g_ in range(1, generations + 1):
            gen[0] = g_
            off = [creator.ACIndividualV1(list(x)) for x in tools.selTournamentDCD(pop, len(pop))]
            for i in range(0, len(off) - 1, 2):
                if random.random() < 0.9:
                    tools.cxSimulatedBinaryBounded(off[i], off[i + 1], eta=15.0, low=lo, up=hi)
            for ind in off:
                tools.mutPolynomialBounded(ind, eta=20.0, low=lo, up=hi, indpb=0.2)
                ind.fitness.values = evaluate(ind)
            pop = tools.selNSGA2(pop + off, population)
    finally:
        random.setstate(state)
    points = list(cache.values())
    valid = [r for r in points if r["valid"]]
    front = sorted(_nondominated(valid), key=lambda r: -r["sharpe"])
    return {"genes": names, "bounds": {g: [a, b] for g, a, b in zip(names, lo, hi)}, "evaluations": len(archive), "distinct_points": len(points),
            "valid": len(valid), "share_valid": (len(valid) / len(points)) if points else 0.0, "points": points, "evaluation_log": archive,
            "front": front, "front_size": len(front), "veto": veto, "population": population, "generations": generations,
            "objectives": ["sharpe max", "max_drawdown_pct min"], "constraint": f"Deb constraint domination: >= {MIN_TRADES_PER_WEEK:g} trades/week (period average) and the env rules"}


def _strategy_dict(point: dict) -> dict:
    g = dict(point["genes"])
    g.setdefault("nop_veto", 1.0)
    return Strategy.from_genes(g, bool(point.get("veto"))).to_dict()


def archive_view(points: list, name: str) -> dict:
    """What a fitness variant would do without the activity rule, on the full archive of both searches: the best
    unconstrained point by Sharpe, the best unconstrained point by the variant (NOT SELECTABLE), the best valid point by
    the variant (anywhere in the archive, front or not) and the share of valid points."""
    fin = [p for p in points if math.isfinite(p["fitness"][name])]
    shp = [p for p in points if math.isfinite(p["sharpe"])]
    best_sh = max(shp, key=lambda p: p["sharpe"]) if shp else None
    best_un = max(fin, key=lambda p: p["fitness"][name]) if fin else None
    vfin = [p for p in fin if p["valid"]]
    best_v = max(vfin, key=lambda p: p["fitness"][name]) if vfin else None
    pick = lambda p: None if p is None else {**{k: v for k, v in p.items() if k != "fitness"}, "fitness": p["fitness"][name], "strategy": _strategy_dict(p)}  # noqa: E731
    return {"points": len(points), "share_valid": (sum(p["valid"] for p in points) / len(points)) if points else 0.0,
            "best_sharpe_unconstrained": pick(best_sh), "best_unconstrained_by_fitness": pick(best_un), "best_valid_by_fitness": pick(best_v),
            "note": "best_unconstrained_* are NOT SELECTABLE (the activity rule selects); shown so the owner sees what each fitness would do without it"}


def freeze_strategy(table: dict, cal_rows, score_table, contract: dict, seed: int, budget: str = "final", primary: str = DEFAULT_FITNESS) -> dict:
    """The upper layer, frozen on the calibration block only: the table is restricted to ``cal_rows`` BEFORE the
    optimiser sees it. Two searches (pre-registered): without the NOP veto and with it. The composite scales come from
    the valid front points of both searches (``term_scales``); every archived point gets every variant's value; each
    variant chooses among the VALID front points, and its veto stays only when it strictly improves that variant's
    chosen calibration-block lower confidence bound. ``primary`` names the variant the arm runs with."""
    if primary not in FITNESS:
        raise ValueError(f"STRATEGY_FITNESS_UNKNOWN:{primary}")
    view = restrict(table, cal_rows)
    s = contract["strategy"]
    o = s["optimizer"][budget]
    runs = {}
    for veto in (False, True):
        runs["with_veto" if veto else "without_veto"] = optimize(lambda st: score_table(view, st), s["genes"], seed, int(o["population"]), int(o["generations"]), veto)
    reference = runs["without_veto"]["front"] + runs["with_veto"]["front"]
    scales = term_scales(reference)
    allp = []
    for r in runs.values():
        for p in r["points"]:
            p["fitness"] = {f: fitness(f, p, scales) for f in FITNESS}
        r["chosen"] = {f: choose(r["front"], f) for f in FITNESS}
        allp += r["points"]
    by = {}
    for f in FITNESS:
        a, b = runs["without_veto"]["chosen"][f], runs["with_veto"]["chosen"][f]
        keep = bool(b is not None and (a is None or (math.isfinite(b["lcb"]) and (not math.isfinite(a["lcb"]) or b["lcb"] > a["lcb"]))))
        pick = b if keep else a
        view_f = archive_view(allp, f)
        if pick is None:
            by[f] = {"chosen": None, "strategy": None, "nop_veto_kept": False, "valid": False, "archive_view": view_f,
                     "note": f"NO_VALID_FRONT_POINT_ON_THE_CALIBRATION_BLOCK (>= {MIN_TRADES_PER_WEEK:g} trades/week); INVALID"}
            continue
        by[f] = {"chosen": {k: v for k, v in pick.items()}, "strategy": _strategy_dict(dict(pick, veto=keep)), "nop_veto_kept": keep, "valid": True,
                 "archive_view": view_f}
    p = by[primary]
    doc = {"budget": budget, "runs": runs, "primary_fitness": primary, "by_fitness": by, "term_scales": scales,
           "scale_rule": "s_k = median |term_k| over the valid points of the calibration-block front (both searches); fallback below 1e-6 (declared); c_k = 1/s_k",
           "nop_rule": "per fitness variant: the NOP veto stays only if it strictly improves the chosen point's calibration-block lower confidence bound (pre-registered)",
           "choice_rule": "search NSGA-II (Sharpe up, DD down) under Deb's constraint domination; each fitness variant picks its frozen point among the VALID front points",
           "chosen": p["chosen"], "strategy": p["strategy"], "nop_veto_kept": p["nop_veto_kept"], "valid": p["valid"]}
    if not p["valid"]:
        doc["note"] = p["note"]
    return doc


def strategy_of(doc: dict) -> Strategy | None:
    s = doc.get("strategy")
    return None if s is None else Strategy(**s)


# ----------------------------------------------------------------------------- NOP labels (consistency with the openers)
class NopLabels:
    """NOP specialist labels from the multires ladder, made consistent with the arm's own opener labels:
      arm opens on the row (any side)           -> NOP = 0, weight 1 (consistency: NOP = 1 implies open_l = open_s = 0)
      arm does not open, multires kind 'open'   -> NOP = 0, weight 1 (a trade exists at some resolution)
      arm does not open, kind in NOP_POSITIVE   -> NOP = 1, weight = 1 for nop_hard, the multires weight otherwise
      multires censored / undefined             -> row dropped (never filled)
    ``table`` counts every (arm_open, kind) cell; ``rows`` keeps (row, y, weight, kind, arm_open) for diagnostics."""

    def __init__(self, mr_labels):
        self.mr = mr_labels
        self.table = {k: {kk: 0 for kk in MR.KINDS + ("censored",)} for k in ("arm_open", "arm_no_open")}
        self.rows = []

    def label(self, bar, r, arm_open: bool):
        lab = self.mr.label(bar, r)
        key = "arm_open" if arm_open else "arm_no_open"
        if lab is None:
            self.table[key]["censored"] += 1
            return None
        kind = lab[3]
        self.table[key][kind] += 1
        if arm_open or kind == "open":
            y, w = 0, 1.0
        else:
            y, w = 1, (1.0 if kind == "nop_hard" else float(lab[1]))
        self.rows.append((int(r), y, w, kind, bool(arm_open)))
        return y, w


def nop_information(table: dict, rows, y) -> dict:
    """Does pNOP add information beyond 1 - max(pL, pS) on the calibration block? AUC of each, the in-sample log loss of a
    logistic fit on [1 - max] alone and on [1 - max, pNOP] (delta < 0 = pNOP helps), and the joint confusion of the
    two signals at 0.5 against the label."""
    rows, y = np.asarray(rows, np.int64), np.asarray(y, int)
    pN = table["pN"][rows]
    q = 1.0 - np.maximum(table["pL"][rows], table["pS"][rows])
    ok = np.isfinite(pN) & np.isfinite(q)
    pN, q, y = pN[ok], q[ok], y[ok]

    def logloss(X):
        if len(y) == 0 or y.min() == y.max():
            return float("nan")
        Xb = np.c_[X, np.ones(len(y))]
        w = np.zeros(Xb.shape[1])
        for _ in range(100):
            p = sigmoid(Xb @ w)
            g = Xb.T @ (p - y) / len(y) + 1e-6 * w
            H = (Xb.T * (p * (1 - p))) @ Xb / len(y) + 1e-6 * np.eye(Xb.shape[1])
            st = np.linalg.solve(H, g)
            w -= st
            if np.abs(st).max() < 1e-12:
                break
        p = np.clip(sigmoid(Xb @ w), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    ll_q, ll_qn = logloss(q[:, None]), logloss(np.c_[q, pN])
    conf = {}
    for a_ in (False, True):
        for b_ in (False, True):
            m = ((pN > 0.5) == a_) & ((q > 0.5) == b_)
            conf[f"pNOP>0.5={a_}|1-max(pL,pS)>0.5={b_}"] = {"rows": int(m.sum()), "nop_label_1": int(y[m].sum()), "nop_label_0": int((1 - y[m]).sum())}
    return {"rows": int(len(y)), "auc_pnop": auc(pN, y), "auc_one_minus_max_open": auc(q, y), "logloss_one_minus_max": ll_q,
            "logloss_with_pnop": ll_qn, "delta_logloss": (ll_qn - ll_q) if math.isfinite(ll_q) and math.isfinite(ll_qn) else float("nan"),
            "joint_confusion": conf}
