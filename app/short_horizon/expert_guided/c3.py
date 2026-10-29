"""Stage C3 (2026-10-26): the label / target fixes of the adversarial audit register
(``docs/short_horizon/clean_round/ADVERSARIAL_AUDIT_REGISTER_2026_10_26.md``) and of the end-to-end controls
(``PIPELINE_CONTROLS_2026_10_26.md``), on top of C2G (``c2.py``: engine-exact outcome tables, closer-only order geometry,
the v1.3 search machinery). Contract ``ACTION_CONTRACT_C3.json`` (schema ``expert_guided_action_contract.c3``).

What changes, each tied to the register:
* A2-2 / A2-1 / A4-3  openers are trained on the ENGINE OUTCOME TABLE, never on pivot bars: per side a 3-class
  softmax {TP_FIRST, SL_FIRST, TIME} at the reference cell (A2-3: TIME is its own class, never folded into "not TP"),
  temperature-calibrated on INNER_A, plus the expected net per (TP, SL) cell (regression, order-preserving). There is
  NO stop-loss head: the SL is the chosen cell's declared constant; a regression target with std < ``MIN_TARGET_STD``
  on TRAIN refuses the run (``C3_CONSTANT_TARGET``).
* A2-9 / A4-10  the grid is bounded by the engine's effective cap (p1 2.09 sigma on TRAIN: 2.0 is never clipped) and
  extended below 0.75 sigma (0.5); TP >= SL cells exist in every row (RR >= 1).
* A4-5 / A2-8  the open rule is decision-theoretic with the RIGHT probabilities:
  EV = P(TP first) * TP - P(SL first) * SL + P(TIME) * E[net | TIME] - cost > open_margin (ONE pooled margin), where the cell is the one
  with the highest predicted net among the ALLOWED cells (TP >= SL, or predicted net > cost + ``RR_MARGIN_SIGMA``);
  "no trade" is a valid answer; activity (>= 2 trades/week) stays a VALIDITY criterion, never forced into negative EV.
* A2-4 / A2-6 / A4-6  closers see the causal geometry of their own order (as C2G) and are trained WITHOUT the
  clairvoyant expert's positions: round-0 positioned states come from exploration only, with the TP/SL drawn from
  the grid the student will use; contradictory duplicate keys are deduplicated; per-side close thresholds, each
  mapped on the positioned rows of its own side; after the freeze the closer must beat "never close" on the
  calibration block (paired daily P&L, stationary bootstrap, lower 90 % bound > 0) or early close is DISABLED for
  the unit and flagged (``early_close_disabled``).
* A2-5  DAgger rounds relabel the closers with the SAME outcome-table rule as round 0 (the student's own visits);
  openers are labelled on every clean-account eligible flat row (market-only): the daily cap lives in the env only.
* A2-10 / A2-11  gate-closed rows never enter the opener sets (``outcomes.entry_rows`` opens the gate on a clean
  account); no ZigZag target remains (NOP keeps its ladder labels).
* A2-12  loss balance: the Huber term is normalised by the masked row count and the CE / Huber weights are declared.
Blocks as C2: INNER_A = early stopping, calibration, strategy, visitation; INNER_B scored once after the freeze.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import action_contract as ACT
from . import c2 as C2
from . import env as E
from . import exec_variants as XV
from . import multires as MR
from . import outcomes as OC
from . import stats as ST

SCHEMA = "expert_guided_action_contract.c3"
GRID_TP = (0.5, 0.75, 1.0, 1.5, 2.0)
GRID_SL = (0.5, 0.75, 1.0, 1.5, 2.0)
CELLS = tuple((tp, sl) for tp in GRID_TP for sl in GRID_SL)
REF = (1.0, 1.0)                              # the 3-class head's cell: symmetric, RR = 1 (declared, not chosen)
CAP_P1_SIGMA = 2.09                           # engine effective cap 1.5 * sigma_rule / sigma, p1 on TRAIN (audit A2-9 / A4-10)
CLASSES_OPEN = ["TP_FIRST", "SL_FIRST", "TIME"]
HIT_TP, HIT_SL, HIT_TIME = OC.HIT_TP, OC.HIT_SL, OC.HIT_TIME
RR_MARGIN_SIGMA = 0.10                        # a TP < SL cell is allowed only when its predicted net > cost + this
OPEN_MARGIN_EPS = 1e-6                        # an open needs EV > margin + eps (sigma): the margin is a quantile SAMPLE of the EV reference, so the
                                              # defining row sits exactly on it and a 1e-7 float difference decided it (behaviour matrix mirror witness)
MIN_TARGET_STD = 0.05                         # a regression target below this std on TRAIN is a constant: refused
LOSS_BALANCE = {"ce_weight": 1.0, "huber_weight": 1.0, "huber_norm": "masked_rows", "ce_norm": "class_weighted_mean"}
EARLY_CLOSE_BENEFIT = {"block": "INNER_A", "statistic": "mean daily P&L difference (with early close - never close), USD",
                       "bootstrap": {"method": "stationary", "block_days": 5.0, "draws": 2000, "seed": 20261026, "level": 0.90},
                       "min_agent_close_share": 0.05,
                       "rule": "early close is kept only when exec_variants.early_close_ok passes: agent closes >= 5 % of the exits on the block AND "
                               "the lower 90 % bound of the mean daily difference >= 0; otherwise the unit runs with early close DISABLED and is "
                               "flagged early_close_disabled (reported, never hidden)"}
FULL_F3 = XV.ExecVariant(reverse_counts_as_one_entry=True, sizing_mode="risk", risk_pct=0.01, entry_cutoff_hours_before_flat=24.0,
                         short_trigger_on_ask=True, note="full_f3 (lane F3: A4-1 / A4-4 / A4-9 / A4-12)")
FULL_F3_12H = dataclasses.replace(FULL_F3, entry_cutoff_hours_before_flat=12.0, note="full_f3 with the legacy 12 h entry cutoff (declared alternative)")
# FULL_F4 = the shared business contract (owner decision 2026-10-29, business_contract.py, agent-multi 82862801): leverage 50,
# rel_volume 0.10, NO strategy-imposed daily entry cap, entries from 12 h after the session open, forced flat 12 h before the
# close and no entry at or after that cutoff (the calendar's own rule, entry_cutoff_hours_before_flat None: the 24 h rule of
# FULL_F3 is the declared variant, default OFF); absolute levels per horizon (S2) and the risk-budget sizing (S3: rho baseline
# 0.005, c = 1, funding reserve 0; the genes join STRATEGY_PARAMS v2 in S6)
FULL_F4 = XV.ExecVariant(reverse_counts_as_one_entry=True, sizing_mode="risk_budget", rho=0.005, entry_cutoff_hours_before_flat=None,
                         short_trigger_on_ask=True, leverage=50.0, no_daily_cap=True, level_mode="absolute_pips",
                         note="full_f4 (business contract 73f32e43: leverage 50, no daily cap, 12 h cutoff, absolute levels, risk-budget sizing rho 0.005 c 1)")
EXEC_VARIANTS = {"legacy": XV.LEGACY, "full_f3": FULL_F3, "full_f3_12h": FULL_F3_12H, "full_f4": FULL_F4}


def exec_variant_of(params_doc: dict | None) -> "XV.ExecVariant":
    """The execution variant a STRATEGY_PARAMS document selects: its exec_profile (absent = full_f3, the audited default,
    byte-identical), with the document's rho carried by the variant when the profile sizes by risk budget."""
    name = str((params_doc or {}).get("exec_profile", EXEC_DEFAULT))
    xv = EXEC_VARIANTS[name]
    if xv.sizing_mode == "risk_budget" and params_doc is not None and "rho" in params_doc:
        xv = dataclasses.replace(xv, rho=float(params_doc["rho"]))
    return xv
CUTOFF_ELIGIBILITY = {"measured_on": "wf2022 INNER_A (2021-07..2022-01), cheat DIAGNOSTIC store, 2026-10-26 (slots.py)",
                      "eligible_days_per_week": {"legacy_12h_cutoff": 5.02, "full_f3_24h_cutoff": 3.98},
                      "eligible_bars_per_day_p10_p50_p90": {"legacy_12h_cutoff": [11, 20, 24], "full_f3_24h_cutoff": [10.3, 13, 24]},
                      "consequence": "under the 1 entry/day cap the 24 h cutoff leaves ~4 slots/week, so the activity rule (CI90 lower bound >= 2.0) "
                                     "needs entries on most eligible days; both cutoffs stay declared variants and Phase 0's simulation decides "
                                     "which one the frozen parameters use (--exec-variant full_f3 | full_f3_12h | legacy)"}
EXEC_DEFAULT = "full_f3"                       # the C3 default (lane F3); legacy stays a declared control (--exec-variant legacy)
BEST_OF_DAY = {"observe_bars": 1, "rule": "A4-2: causal secretary rule per business day (exec_variants.BestOfDayGate): the first eligible open "
                                            "signal of the day is observed only; then the first open whose EV >= the day's running best is placed "
                                            "(observe 3 was infeasible under full_f3's 24 h cutoff: the 1 entry/day cap leaves ~4 slots/week and the "
                                            "secretary rule loses the day whenever the best signal fell inside the window; smoke 2026-10-26: best 1.26/wk)"}
EARLY_CLOSE_MIN_SHARE = 0.05                   # A4-7 (exec_variants.early_close_ok): agent closes >= 5 % of exits AND value LCB >= 0
# ONE open rate (side symmetry, owner alert 2026-10-26): the margin is a quantile of the POOLED EV reference (both sides'
# EVs over the calibration rows), so both sides clear the same absolute margin and the higher EV wins; per-side open rates
# (v1.3) let the search privilege a side through its own quantile scale (cheat smoke: short rate at the 0.50 bound ->
# short margin -0.72 vs long -0.06 -> 73 % shorts). Per-side CLOSE thresholds stay (closers see their own order).
GENES_Q = ("open_rate", "close_rem_rate_l", "close_rem_rate_s", "close_p_l", "close_p_s", "allow_reverse", "nop_veto")
# close_p_* and nop_veto are ABSOLUTE calibrated-probability thresholds (behaviour matrix 2026-10-27: a quantile over a
# saturated reference -- perfect closers give P(SL first) in {1e-6, 1 - 1e-6} -- mapped close_p to 1e-6 and closed every
# short position, nop_veto to 1.0 and never vetoed)
SIDE_SYMMETRY_RULE = ("open: EV_L and EV_S are computed by the same function with mirrored inputs and compared against ONE absolute margin; the higher EV "
                      "wins; ties -> no privileged side (HOLD); the NOP veto, the gate and the activity rule see no side; witness = the swap test "
                      "(mirrored head outputs must mirror the actions) and the per-tick decision ledger (EV_L, EV_S, margin, winner, reason)")
# open rates up to 0.50 (2026-10-26): under the 1 entry/day cap + the best-of-day gate the open rate governs how many
# DAYS carry a qualifying signal, not the trade count; with full_f3's 24 h cutoff (~4 eligible days/week) a 0.30 ceiling
# left no probe point with an activity lower bound >= 2.0 (smoke: best 2.67/week mean, bound below 2.0)
Q_BOUNDS = {"open_rate": [0.002, 0.50], "close_rem_rate_l": [0.0, 0.60], "close_rem_rate_s": [0.0, 0.60],
            "close_p_l": [0.30, 0.95], "close_p_s": [0.30, 0.95], "allow_reverse": [0.0, 1.0], "nop_veto": [0.30, 0.95]}
GENES = ("open_margin", "close_margin_l", "close_margin_s", "close_p_l", "close_p_s", "allow_reverse", "nop_veto")
SPECIALISTS, OPENERS, CLOSERS, SIDE_OF = C2.SPECIALISTS, C2.OPENERS, C2.CLOSERS, C2.SIDE_OF
NET_BOUND, REM_BOUND, GEOMETRY, N_GEO = C2.NET_BOUND, C2.REM_BOUND, C2.GEOMETRY, C2.N_GEO
CALIBRATION_METHOD = "temperature_multiclass_openers+platt_binary"
ENCODING_DECISIONS = {"regime_gmm": "USE_onehot", "event_clock": "USE_window2h", "source": "selection_v5/ENCODING.csv (audit A1-11: decided, never applied)",
                      "regime_gmm_rule": "rg.gmm.p0/p1/p2 -> one-hot of the row's argmax regime (3 binary columns, entropy kept)",
                      "event_clock_rule": "ev.<x>.hours_since -> 1 within 2 h after the event or 2 h before the next (period 24 h); ev.<x>.hours_to dropped (= 24 - hours_since, A1-4 duplicate)"}
ARCH_U = {"trunk": "u", "fusion_units": 8, "core_blocks": 2, "readout": "flatten"}            # the sealed u-design (default)
ARCH_TCN = {"trunk": "residual_tcn", "fusion_units": 8, "core_blocks": 2, "readout": "flatten"}  # Musashi's minimal grouped TCN
ARCH_KEYS = ("trunk", "fusion_units", "core_blocks", "readout")


def is_c3(doc) -> bool:
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA


def modules(cells=CELLS) -> dict:
    """(n_classes, n_regression) per specialist: openers 3 classes + one net per cell; closers 2 + remaining; NOP 2."""
    return {"open_l": (3, len(cells)), "open_s": (3, len(cells)), "nop": (2, 0), "close_l": (2, 1), "close_s": (2, 1)}


def reg_bounds(m: str, cells=CELLS):
    return C2.reg_bounds(m, cells)


def cell_index(cell, cells=CELLS) -> int:
    return [tuple(float(x) for x in c) for c in cells].index(tuple(float(x) for x in cell))


def allowed_cells(net, cells, cost: float, rr_margin: float = RR_MARGIN_SIGMA, cap: float = float("nan"), lo: float = float("nan")) -> np.ndarray:
    """Boolean mask over the cells of one (row, side): TP >= SL always allowed (RR >= 1); a TP < SL cell only when its
    predicted net clears the cost by the declared margin (A2-2: no tiny-TP / wide-SL trade on faith); every level must
    sit inside the engine's per-bar effective band [lo, cap] (A2-9 / A4-10: a cell the engine would clip is not chosen)."""
    net = np.asarray(net, float)
    c = np.asarray(cells, float)
    ok = (c[:, 0] >= c[:, 1]) | (net > float(cost) + float(rr_margin))
    if np.isfinite(cap):
        ok &= (c[:, 0] <= cap) & (c[:, 1] <= cap)
    if np.isfinite(lo):
        ok &= (c[:, 0] >= lo) & (c[:, 1] >= lo)
    return ok


CELL_TIE_EPS = 1e-9   # sigma: cells whose values differ by less are a tie, resolved by the declared cell order


def argmax_cell(values, ok) -> int:
    """The allowed cell with the highest value; a TIE (within CELL_TIE_EPS) goes to the first cell in the declared grid
    order. Without the tolerance, equal engine nets that differ by float rounding (1.0 vs 0.9999999999999922 between a
    world and its exact mirror) chose different SLs in the two worlds (behaviour matrix v10, E6 / W14 mirror failures)."""
    x = np.where(np.asarray(ok, bool), np.asarray(values, float), -np.inf)
    m = float(np.max(x))
    return int(np.flatnonzero(x >= m - CELL_TIE_EPS)[0]) if np.isfinite(m) else int(np.argmax(x))


def best_cell_ev(net, p3, cells, e_time: float, cost: float, rr_margin: float = RR_MARGIN_SIGMA, cap: float = float("nan"), lo: float = float("nan")):
    """(cell index or -1, EV) of one (row, side): the allowed cell with the highest predicted net, valued by the
    decision-theoretic rule EV = P(TP)*TP - P(SL)*SL + P(TIME)*E[net|TIME] - cost (A4-5 / A2-8)."""
    net = np.asarray(net, float)
    if not np.all(np.isfinite(net)) or not np.all(np.isfinite(p3)):
        return -1, float("nan")
    ok = allowed_cells(net, cells, cost, rr_margin, cap, lo)
    if not ok.any():
        return -1, float("nan")
    ci = argmax_cell(net, ok)
    tp, sl = float(cells[ci][0]), float(cells[ci][1])
    ev = float(p3[0]) * tp - float(p3[1]) * sl + float(p3[2]) * float(e_time) - float(cost)
    return ci, ev


def ev_rows(table: dict, rows, si: int, cells, e_time: float, cost: float) -> np.ndarray:
    """Vectorised EV reference of one side over ``rows`` (NaN where no cell is allowed or the row is blank)."""
    r = np.asarray(rows, np.int64)
    out = np.full(len(r), np.nan)
    cap, lo = table.get("cap"), table.get("lo")
    from . import menu as M
    if M.is_menu(cells):                                         # full_f4 decision menu: the same EV rule on the slot's own distances in sigma
        for i, rr in enumerate(r):
            net, p3 = np.asarray(table["net"][rr, si], float), np.asarray(table["p3"][rr, si], float)
            if not (np.all(np.isfinite(net)) and np.all(np.isfinite(p3))):
                continue
            ok = M.allowed(net, np.asarray(table["zz_menu"][rr, si], float) == 1.0, table["menu_tp_s"][rr, si], table["menu_sl_s"][rr, si], cost)
            if ok.any():
                ci = M.argmax_slot(net, ok)
                out[i] = float(p3[0]) * float(table["menu_tp_s"][rr, si, ci]) - float(p3[1]) * float(table["menu_sl_s"][rr, si, ci]) + float(p3[2]) * float(e_time) - float(cost)
        return out
    for i, rr in enumerate(r):
        out[i] = best_cell_ev(table["net"][rr, si], table["p3"][rr, si], cells, e_time, cost,
                              cap=float(cap[rr]) if cap is not None else float("nan"), lo=float(lo[rr]) if lo is not None else float("nan"))[1]
    return out


def cap_arrays(cx, SCFG) -> tuple:
    """Per dataset row: the engine's effective cap and minimum multiple at that decision (NaN where undefined)."""
    env = OC.make_env(cx, SCFG)
    n = len(cx.ds.row_bar)
    cap, lo = np.full(n, np.nan), np.full(n, np.nan)
    for r in range(n):
        b = int(cx.ds.row_bar[r])
        if 0 <= b < len(env.sigma) - 1:
            cap[r], lo[r] = XV.effective_cap(env, b), XV.min_multiple(env, b)
    return cap, lo


# ----------------------------------------------------------------------------- the strategy (upper layer)
@dataclass(frozen=True)
class StrategyC3:
    """Per-side open margins on the EV reference, per-side close thresholds (remaining value and P(SL first)) each
    mapped on the positioned rows of its own side (A4-6), reverse permission, optional NOP veto, and the early-close
    switch (False only when the calibration-block benefit test fails: flagged)."""
    open_margin: float
    close_margin_l: float
    close_margin_s: float
    close_p_l: float
    close_p_s: float
    allow_reverse: bool
    nop_veto: float | None
    early_close: bool = True

    @classmethod
    def from_genes(cls, g: dict, veto: bool):
        return cls(float(g["open_margin"]), float(g["close_margin_l"]), float(g["close_margin_s"]),
                   float(g["close_p_l"]), float(g["close_p_s"]), bool(float(g["allow_reverse"]) >= 0.5), float(g["nop_veto"]) if veto else None, True)

    def mirrored(self):
        """The same strategy with the sides swapped (the swap witness): one open margin, per-side close thresholds swapped."""
        return dataclasses.replace(self, close_margin_l=self.close_margin_s, close_margin_s=self.close_margin_l, close_p_l=self.close_p_s, close_p_s=self.close_p_l)

    def to_dict(self):
        return dataclasses.asdict(self)

    def no_early_close(self):
        return dataclasses.replace(self, early_close=False)


NO_TRADE = StrategyC3(1e9, 0.0, 0.0, 1.0, 1.0, False, None, True)
TIE_EPS = 1e-12


class PolicyC3:
    """Frozen upper layer over a C3 table (``net`` (n, 2, cells), ``p3`` (n, 2, 3), ``pn``, ``closer``):
    policy(cx) -> f(bar, r, st, env) -> (action, tp, sl). A blank row raises, as C2."""

    def __init__(self, table: dict, strategy: StrategyC3, cells=CELLS, e_time=(0.0, 0.0), cost: float = 0.0, observe_bars: int = BEST_OF_DAY["observe_bars"],
                 record: bool = False):
        self.t, self.s, self.cells = table, strategy, [tuple(c) for c in cells]
        self.e_time, self.cost, self.observe = (float(e_time[0]), float(e_time[1])), float(cost), int(observe_bars)
        self.last_gate = None
        self.record, self.ledger = bool(record), []          # per-tick decision ledger (flat decisions): EV_L, EV_S, margin, winner, reason

    def _cap(self, r):
        cap, lo = self.t.get("cap"), self.t.get("lo")
        return (float(cap[r]) if cap is not None else float("nan")), (float(lo[r]) if lo is not None else float("nan"))

    def _v(self, f, r, *idx):
        x = self.t[f][(r,) + idx]
        if not np.all(np.isfinite(x)):
            raise ValueError(f"C3_ROW_OUTSIDE_TABLE:{f}:{r}")
        return x

    def side_evs(self, r):
        """(EV_L, cell_L, EV_S, cell_S) at row r: the same function on each side's own outputs (no side is privileged)."""
        cap, lo = self._cap(r)
        out = []
        for si in (0, 1):
            ci, ev = best_cell_ev(self._v("net", r, si), self._v("p3", r, si), self.cells, self.e_time[si], self.cost, cap=cap, lo=lo)
            out += [ev if (ci >= 0 and np.isfinite(ev)) else float("nan"), ci]
        return tuple(out)

    def open_decision(self, r, with_score: bool = False):
        s = self.s
        rec = {"row": int(r), "ev_l": float("nan"), "ev_s": float("nan"), "margin": s.open_margin, "winner": "none", "reason": ""} if self.record else None
        if s.nop_veto is not None and float(self._v("pn", r)) > s.nop_veto:
            if rec is not None:
                rec["reason"] = "nop_veto"
                self.ledger.append(rec)
            return None
        ev_l, ci_l, ev_s, ci_s = self.side_evs(r)
        cl = bool(np.isfinite(ev_l) and ev_l > s.open_margin + OPEN_MARGIN_EPS)
        cs = bool(np.isfinite(ev_s) and ev_s > s.open_margin + OPEN_MARGIN_EPS)
        if rec is not None:
            rec.update(ev_l=float(ev_l), ev_s=float(ev_s), cleared_l=cl, cleared_s=cs, both_cleared=cl and cs)
        if cl and cs:
            if abs(ev_l - ev_s) <= TIE_EPS:                 # exact tie: no privileged side
                best, why = None, "tie"
            elif ev_l > ev_s:
                best, why = (ev_l, E.LONG, ci_l), "both_cleared_long_higher"
            else:
                best, why = (ev_s, E.SHORT, ci_s), "both_cleared_short_higher"
        elif cl:
            best, why = (ev_l, E.LONG, ci_l), "only_long_cleared"
        elif cs:
            best, why = (ev_s, E.SHORT, ci_s), "only_short_cleared"
        else:
            best, why = None, "none_cleared"
        if rec is not None:
            rec["reason"] = why
            rec["winner"] = "none" if best is None else ("long" if best[1] == E.LONG else "short")
            self.ledger.append(rec)
        if best is None:
            return None
        ev, d, ci = best
        out = (d, self.cells[ci][0], self.cells[ci][1])
        return out + (ev,) if with_score else out

    def ledger_summary(self) -> dict:
        L = self.ledger
        n = len(L)
        both = [x for x in L if x.get("both_cleared")]
        c = lambda k: sum(1 for x in L if x["reason"] == k)  # noqa: E731
        return {"flat_decisions": n, "both_cleared": len(both), "both_cleared_long_won": sum(1 for x in both if x["winner"] == "long"),
                "both_cleared_short_won": sum(1 for x in both if x["winner"] == "short"), "only_long_cleared": c("only_long_cleared"),
                "only_short_cleared": c("only_short_cleared"), "none_cleared": c("none_cleared"), "nop_veto": c("nop_veto"), "ties": c("tie"),
                "mean_ev_l": float(np.nanmean([x["ev_l"] for x in L])) if n else float("nan"), "mean_ev_s": float(np.nanmean([x["ev_s"] for x in L])) if n else float("nan"),
                "margin": float(self.s.open_margin), "rule": SIDE_SYMMETRY_RULE}

    def reverse_ok(self, r) -> bool:
        """Whether a close may reverse into the other side at row r (C3: always, when allow_reverse)."""
        return True

    def close_now(self, r, si, bar, pos) -> bool:
        if not self.s.early_close:
            return False
        rem, psl = self.t["closer"].value(r, bar, pos)
        mar = self.s.close_margin_l if si == 0 else self.s.close_margin_s
        p = self.s.close_p_l if si == 0 else self.s.close_p_s
        return rem < -mar or psl > p

    def scored(self, cx):
        """The SCORED policy: (action, tp, sl, score = EV of the open; -inf otherwise), for the best-of-day gate."""
        NEG = float("-inf")

        def f(bar, r, st, env):
            self._ctx = (int(bar), env)                              # S4: the decision bar and the engine (the absolute-level envelope reads the quote and the variant)
            pos = int(st[0])
            if pos != 0:
                si = 0 if pos > 0 else 1
                if not self.close_now(r, si, bar, env.pos):
                    return E.HOLD, 1.5, 2.0, NEG
                o = self.open_decision(r, with_score=True)
                if o is not None and self.s.allow_reverse and self.reverse_ok(r) and o[0] != (E.LONG if pos > 0 else E.SHORT):
                    return o[0], float(o[1]), float(o[2]), float(o[3])
                return E.CLOSE, 1.5, 2.0, NEG
            o = self.open_decision(r, with_score=True)
            return (E.HOLD, 1.5, 2.0, NEG) if o is None else (o[0], float(o[1]), float(o[2]), float(o[3]))
        return f

    def __call__(self, cx):
        """A plain policy for ``Ctx.rollout``: the scored policy behind the best-of-day gate (A4-2; ``observe_bars`` = 0
        degenerates to the audited first-qualifying rule). A fresh gate per rollout; its first-vs-best diagnostics are
        kept on ``last_gate``."""
        self.last_gate = XV.BestOfDayGate(self.scored(cx), observe_bars=self.observe)
        return self.last_gate


def strategy_from_dict(d: dict) -> StrategyC3:
    return StrategyC3(**d)


# ----------------------------------------------------------------------------- calibration: temperature for the 3-class openers
def softmax(lg, T: float = 1.0) -> np.ndarray:
    z = np.asarray(lg, float) / float(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(lg, y, T: float) -> float:
    p = softmax(lg, T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0))))


def fit_temperature(lg, y, grid=None) -> float:
    """Temperature scaling (Guo et al. 2017) of a multiclass head on the calibration block: the T on a declared grid
    with the lowest NLL (a one-parameter map that cannot change the argmax)."""
    lg, y = np.asarray(lg, float), np.asarray(y, int)
    if len(y) == 0:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 61)) if grid is None else np.asarray(grid, float)
    return float(grid[int(np.argmin([nll(lg, y, t) for t in grid]))])


def multiclass_metrics(p3, y, bins=15) -> dict:
    """AUC / Brier / ECE of P(TP_FIRST) against TP_FIRST, the multiclass Brier, accuracy and the class shares."""
    p3, y = np.asarray(p3, float), np.asarray(y, int)
    if len(y) == 0:
        return {"n": 0}
    onehot = np.eye(3)[y]
    tp = ACT.head_metrics(p3[:, 0], (y == HIT_TP).astype(int), bins)
    return {"n": int(len(y)), "class_shares": np.bincount(y, minlength=3).tolist(), "tp_first": tp,
            "brier_multiclass": float(np.mean(np.sum((p3 - onehot) ** 2, axis=1))), "accuracy": float(np.mean(np.argmax(p3, axis=1) == y)),
            "nll": float(-np.mean(np.log(np.clip(p3[np.arange(len(y)), y], 1e-12, 1.0))))}


def full_logits(pol, X, batch=4096) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        lg, _ = pol(X[i:i + batch])
        out.append(np.asarray(lg, float))
    return np.concatenate(out) if out else np.zeros((0, 3))


def calibrate(pols, D) -> dict:
    """Openers: temperature on INNER_A (3 classes); closers / NOP: Platt as C2."""
    cal = C2.calibrate({m: p for m, p in pols.items() if m not in OPENERS}, D)
    for m in OPENERS:
        if m not in pols:
            continue
        lg, y = full_logits(pols[m], D[m][0]), np.asarray(D[m][1], int)
        T = fit_temperature(lg, y)
        cal[m] = {"method": "temperature", "T": T, "rows": int(len(y)), "class_counts": np.bincount(y, minlength=3).tolist(),
                  "before": multiclass_metrics(softmax(lg, 1.0), y), "after": multiclass_metrics(softmax(lg, T), y)}
    return cal


# ----------------------------------------------------------------------------- per-row table
def raw_table(pols: dict, Xw, rows, n_state: int, n_cells: int) -> dict:
    """Uncalibrated outputs on ``rows``: opener logits (n, 2, 3) and nets (n, 2, cells), the NOP margin; the closers
    are evaluated on demand with their order's geometry (``C2.CloserGeo``)."""
    n = len(Xw)
    t = {"net": np.full((n, 2, n_cells), np.nan), "lg_open": np.full((n, 2, 3), np.nan), "z_pn": np.full(n, np.nan)}
    rows = np.asarray(rows, np.int64)
    X = C2.X_of(Xw, rows, n_state)
    for m in OPENERS:
        lg, r = [], []
        for i in range(0, len(X), 4096):
            a_, b_ = pols[m](X[i:i + 4096])
            lg.append(np.asarray(a_, float))
            r.append(np.asarray(b_, float))
        t["lg_open"][rows, SIDE_OF[m], :] = np.concatenate(lg)
        t["net"][rows, SIDE_OF[m], :] = np.concatenate(r)
    t["z_pn"][rows] = ACT.module_outputs(pols["nop"], X)[0]
    return t


def calibrate_table(raw: dict, cal: dict) -> dict:
    t = {"net": raw["net"].copy(), "p3": np.full(raw["lg_open"].shape, np.nan)}
    for m in OPENERS:
        si = SIDE_OF[m]
        lg = raw["lg_open"][:, si, :]
        ok = np.all(np.isfinite(lg), axis=1)
        t["p3"][ok, si, :] = softmax(lg[ok], float(cal[m]["T"]))
    z = raw["z_pn"]
    t["pn"] = np.where(np.isfinite(z), ACT.platt(np.nan_to_num(z), cal["nop"]["a"], cal["nop"]["b"]), np.nan)
    if "closer" in raw:
        t["closer"] = raw["closer"].with_ens_cal(cal)
    for k in ("cap", "lo"):
        if k in raw:
            t[k] = raw[k]
    return t


def seed_table(pols, cx, NS, calibration, cells=CELLS, caps=None) -> dict:
    t = calibrate_table(raw_table(pols, cx.Xw, np.flatnonzero(cx.obs_ok), NS, len(cells)), calibration)
    t["closer"] = C2.CloserGeo(cx, [(pols, calibration)], NS)
    if caps is not None:
        t["cap"], t["lo"] = caps
    return t


# ----------------------------------------------------------------------------- ensemble helpers (seed ensemble / audits)
FIELDS = ("net", "p3", "pn")


def mean_tables(tables: list) -> dict:
    out = {f: np.mean([t[f] for t in tables], axis=0) for f in FIELDS}
    out["closer"] = C2.CloserGeo.merge([t["closer"] for t in tables])
    for k in ("cap", "lo"):
        if k in tables[0]:
            out[k] = tables[0][k]
    return out


def to_raw(table: dict) -> dict:
    """Ensemble: mean probabilities back to logits (openers: log p, a valid 3-class logit; NOP: logit), nets unchanged."""
    r = {"net": table["net"].copy(), "closer": table["closer"]}
    p3 = table["p3"]
    r["lg_open"] = np.where(np.isfinite(p3), np.log(np.clip(np.nan_to_num(p3, nan=1.0), 1e-12, 1.0)), np.nan)
    pn = table["pn"]
    r["z_pn"] = np.where(np.isfinite(pn), ACT.logit(np.nan_to_num(pn, nan=0.5)), np.nan)
    for k in ("cap", "lo"):
        if k in table:
            r[k] = table[k]
    return r


def ensemble_calibration(ens_table: dict, cx, saved_cal: dict) -> tuple:
    """ONE calibration per specialist refitted on the ensemble over the saved calibration-block label rows: temperature
    for the 3-class openers, Platt for NOP and the closers (on the members' mean closer logits)."""
    raw = to_raw(ens_table)
    cal = {}
    for m in SPECIALISTS:
        r, y = np.asarray(saved_cal[m]["rows"], np.int64), np.asarray(saved_cal[m]["y"], int)
        if m in OPENERS:
            lg = raw["lg_open"][r, SIDE_OF[m], :]
            T = fit_temperature(lg, y)
            cal[m] = {"method": "temperature", "T": T, "rows": int(len(y)), "before": multiclass_metrics(softmax(lg, 1.0), y), "after": multiclass_metrics(softmax(lg, T), y)}
            continue
        z = raw["closer"].raw_logit_rows(m, r, saved_cal[m]["G"]) if m in CLOSERS else raw["z_pn"][r]
        a_, b_ = ACT.fit_platt(z, y)
        cal[m] = {"a": a_, "b": b_, "rows": int(len(y)), "before": ACT.head_metrics(ACT.sigmoid(z), y), "after": ACT.head_metrics(ACT.platt(z, a_, b_), y)}
    return calibrate_table(raw, cal), cal


def run_params(e3m: dict) -> dict:
    """The C3 run's declared policy parameters (cells, E[net|TIME], cost, exec variant) from its E3M.json."""
    c = e3m["c3"]
    return {"cells": [tuple(x) for x in c["cells"]], "e_time": tuple(c["ev_rule"]["e_time_by_side"]), "cost": float(c["ev_rule"]["cost_sigma"]),
            "exec_variant": (c.get("exec_variant") or {}).get("name", EXEC_DEFAULT)}


def cfg_for(e3m: dict, base_cfg):
    """The replay contract the run used: the base contract with the run's declared execution variant."""
    xv = EXEC_VARIANTS[run_params(e3m)["exec_variant"]]
    return dataclasses.replace(base_cfg, variant=None if xv.is_legacy() else xv)


def policy(table: dict, strategy, e3m: dict) -> "PolicyC3":
    """A C3 policy over ``table`` with the run's parameters; ``strategy`` is a StrategyC3 or its dict."""
    rp = run_params(e3m)
    st_ = strategy if isinstance(strategy, StrategyC3) else strategy_from_dict(strategy)
    return PolicyC3(table, st_, rp["cells"], rp["e_time"], rp["cost"])


def exit_prob(cx, table):
    return C2.exit_prob(cx, table)


# ----------------------------------------------------------------------------- label sets
def entry_sets3(cx, t: dict, period: str, side_index: int, ref=REF, cells=CELLS):
    """(rows, y in {TP_FIRST, SL_FIRST, TIME} at the reference cell, R = net grid) of one side in ``period``, every
    cell available (resolution + 1 h <= period end, not censored)."""
    rows = np.flatnonzero(t["eligible"] & (cx.ds.period == period))
    ok = np.all(OC.available(cx, t["exit_bar"][rows, side_index, :], t["censored"][rows, side_index, :], period), axis=1)
    ok &= np.all(np.isfinite(t["net_sigma"][rows, side_index, :]), axis=1)
    rows = rows[ok]
    ci = cell_index(ref, cells)
    y = t["hit"][rows, side_index, ci].astype(np.int64)
    return rows, y, t["net_sigma"][rows, side_index, :].astype(np.float32)


def dedup_closer(rows, y, R, G, decimals: int = 3):
    """Contradictory duplicate keys (same row, same side, same geometry): one row per key, majority label, mean
    remaining; keys with a tie are DROPPED (A2-4). Returns (rows, y, R, G, n_dropped, n_merged)."""
    rows, y, R, G = np.asarray(rows, np.int64), np.asarray(y, np.int64), np.asarray(R, np.float32), np.asarray(G, np.float32)
    if not len(rows):
        return rows, y, R, G, 0, 0
    key = np.c_[rows.astype(np.float64), np.round(G, decimals).astype(np.float64)]
    _, first, inv, counts = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    keep_r, keep_y, keep_R, keep_G, dropped, merged = [], [], [], [], 0, 0
    for k in range(len(first)):
        m = inv == k
        yy = y[m]
        n1 = int(yy.sum())
        if counts[k] > 1:
            merged += 1
        if 2 * n1 == len(yy):
            dropped += 1
            continue
        keep_r.append(rows[first[k]])
        keep_y.append(1 if 2 * n1 > len(yy) else 0)
        keep_R.append(R[m].mean(axis=0))
        keep_G.append(G[first[k]])
    return (np.asarray(keep_r, np.int64), np.asarray(keep_y, np.int64), np.asarray(keep_R, np.float32).reshape(-1, R.shape[1]),
            np.asarray(keep_G, np.float32).reshape(-1, G.shape[1]), dropped, merged)


def check_target_std(D_tr: dict, cells=CELLS, min_std: float = MIN_TARGET_STD):
    """A2-1 / A4-3: every regression target must vary on TRAIN; a constant target (the old 2-sigma SL) is refused."""
    bad = {}
    for m in OPENERS + CLOSERS:
        if m not in D_tr or len(D_tr[m]) < 3 or not len(D_tr[m][2]):
            continue
        R = np.asarray(D_tr[m][2], float)
        sd = R.std(axis=0)
        for j, s in enumerate(sd):
            if s < min_std:
                bad[f"{m}[{C2.cell_name(cells[j]) if m in OPENERS else 'remaining_sigma'}]"] = float(s)
    if bad:
        raise SystemExit(f"C3_CONSTANT_TARGET:{json.dumps(bad)}")
    return {m: np.asarray(D_tr[m][2], float).std(axis=0).tolist() for m in OPENERS + CLOSERS if m in D_tr and len(D_tr[m]) > 2 and len(D_tr[m][2])}


def sets(cx, et, xt, period, nopl, n_state, cells=CELLS, clipped=None, geo_out=None, dedup_doc=None, ref=REF):
    """{module: (X, y, R, Rm)} (nop: (X, y)) and {module: rows}: openers from the entry table (3-class + nets),
    closers from the exit table with their geometry (deduplicated), NOP from the ladder."""
    D, rows = {}, {}
    for m in OPENERS:
        r, y, R = entry_sets3(cx, et, period, SIDE_OF[m], ref, cells)
        R, n = C2.clip_targets(R, NET_BOUND)
        if clipped is not None:
            clipped[f"{period}/{m}"] = n
        D[m] = (C2.X_of(cx.Xw, r, n_state), y, R, np.ones_like(R))
        rows[m] = r
    for m in CLOSERS:
        r, y, R, G = C2.exit_sets_geo(cx, xt, period, SIDE_OF[m])
        r, y, R, G, dropped, merged = dedup_closer(r, y, R, G)
        if dedup_doc is not None:
            dedup_doc[f"{period}/{m}"] = {"rows": int(len(r)), "contradictory_keys_dropped": dropped, "duplicate_keys_merged": merged}
        R, n = C2.clip_targets(R, REM_BOUND)
        if clipped is not None:
            clipped[f"{period}/{m}"] = n
        if geo_out is not None:
            geo_out[m] = G
        D[m] = (C2.X_geo(cx.Xw, r, n_state, G), y, R, np.ones_like(R))
        rows[m] = r
    r, y = C2.nop_rows(cx, et, period, nopl)
    D["nop"] = (C2.X_of(cx.Xw, r, n_state), y)
    rows["nop"] = r
    return D, rows


def prior_random_cells(cx, prior, seed, cells=CELLS):
    """Exploration policy for the round-0 positioned states: the action prior as ``Ctx.prior_random`` but the TP / SL
    drawn uniformly from the grid the student will use (A2-4: no fixed 1.5 / 2.0 orders); never the expert."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(prior)
    cells = [tuple(c) for c in cells]

    def f(bar, r, st, env):
        a = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        tp, sl = cells[int(rng.integers(0, len(cells)))]
        return a, float(tp), float(sl)
    return f


def round0_exit_tables(cx, periods, prior, SCFG, cells=CELLS, seed: int = 0) -> dict:
    """Round-0 positioned states per period from EXPLORATION ONLY (A2-6: the expert's trajectory never enters the closer
    training; its hindsight TP geometry would leak)."""
    out = {}
    for p in periods:
        vis = OC.visit_positions(cx, p, prior_random_cells(cx, prior, seed, cells), SCFG)
        out[p] = OC.exit_table(cx, SCFG, vis, geometry_fn=C2.geometry_fn(cx))
    return out


def e_time_of(cx, et: dict, period: str = "TRAIN", ref=REF, cells=CELLS) -> list:
    """E[net | TIME] per side at the reference cell on ``period`` (the forced-flat outcome's mean net; 0 if none)."""
    ci = cell_index(ref, cells)
    out = []
    for si in (0, 1):
        rows = np.flatnonzero(et["eligible"] & (cx.ds.period == period))
        h, n = et["hit"][rows, si, ci], et["net_sigma"][rows, si, ci]
        m = (h == HIT_TIME) & np.isfinite(n)
        out.append(float(n[m].mean()) if m.any() else 0.0)
    return out


def cost_sigma_of(cx, SCFG, cost_allowance: float, rows) -> float:
    """One declared cost in sigma for the EV rule: the median per-row cost (allowance + round trip / sigma) over the
    calibration rows (the nets in the tables already carry the engine's exact costs; this term is the EV's)."""
    c = ACT.row_cost_sigma(SCFG.profile.round_trip_price(), cx.ds.sigma, cx.ds.row_bar, cost_allowance)
    c = c[np.asarray(rows, np.int64)]
    c = c[np.isfinite(c)]
    return float(np.median(c)) if len(c) else float(cost_allowance)


# ----------------------------------------------------------------------------- gene map / freeze
def closer_reference_sides(table: dict, src: dict) -> dict:
    """Per-side calibration references of the closers on the positioned visits of THAT side (A4-6)."""
    out = {}
    for m in CLOSERS:
        d = src.get(m) or {}
        r = np.asarray(d.get("rows", []), np.int64)
        k = "l" if SIDE_OF[m] == 0 else "s"
        if len(r):
            x, q = table["closer"].rows_values(m, r, d["G"])
            out[f"rem_ref_{k}"], out[f"psl_ref_{k}"] = np.asarray(x, float), np.asarray(q, float)
        else:
            out[f"rem_ref_{k}"], out[f"psl_ref_{k}"] = np.zeros(0), np.zeros(0)
    return out


def gene_map(view: dict, rows, closer_ref: dict, cells, e_time, cost) -> ACT.GeneMap:
    r = np.asarray(rows, np.int64)
    refs = dict(closer_ref)
    ev_l, ev_s = ev_rows(view, r, 0, cells, e_time[0], cost), ev_rows(view, r, 1, cells, e_time[1], cost)
    refs["open_score"] = np.r_[ev_l, ev_s]                      # POOLED: one margin for both sides (side symmetry)
    refs["open_score_l"], refs["open_score_s"] = ev_l, ev_s      # reported only (the per-side distributions)
    refs["nop_score"] = view["pn"][r]
    rules = {"open_margin": ("open_rate", "open_score", "upper_rate"),
             "close_margin_l": ("close_rem_rate_l", "rem_ref_l", "neg_lower_rate"), "close_margin_s": ("close_rem_rate_s", "rem_ref_s", "neg_lower_rate"),
             "close_p_l": ("close_p_l", None, "copy"), "close_p_s": ("close_p_s", None, "copy"),
             "allow_reverse": ("allow_reverse", None, "copy"), "nop_veto": ("nop_veto", None, "copy")}
    return ACT.GeneMap(refs, rules, "C3: ONE open margin on the pooled EV reference (both sides, calibration flat rows); per-side closer references on the exploration visits of that side")


# ----------------------------------------------------------------------------- the early-close benefit test
def daily_pnl(trades) -> dict:
    out = {}
    for t in trades:
        d = str(t["exit_time"])[:10]
        out[d] = out.get(d, 0.0) + float(t["net_usd"])
    return out


def mean_diff_lower_bound(diff, block: float, draws: int, seed: int, level: float) -> float:
    diff = np.asarray(diff, float)
    if len(diff) < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([diff[ST.stationary_bootstrap_indices(len(diff), block, rng)].mean() for _ in range(draws)])
    return float(np.quantile(means, 1.0 - level))


def early_close_benefit(cx, table, strat: StrategyC3, SCFG, cells, e_time, cost, spec=EARLY_CLOSE_BENEFIT) -> dict:
    """With early close vs never close on the calibration block: paired daily P&L (union of exit days), the stationary
    bootstrap lower bound of the mean difference. ``beneficial`` False -> the unit runs without early close (flagged)."""
    p = spec["block"]
    m1, t1 = ACT.rollout_with_trades(cx, p, PolicyC3(table, strat, cells, e_time, cost)(cx), SCFG)
    m0, t0 = ACT.rollout_with_trades(cx, p, PolicyC3(table, strat.no_early_close(), cells, e_time, cost)(cx), SCFG)
    d1, d0 = daily_pnl(t1), daily_pnl(t0)
    days = sorted(set(d1) | set(d0))
    diff = np.array([d1.get(d, 0.0) - d0.get(d, 0.0) for d in days])
    b = spec["bootstrap"]
    lb = mean_diff_lower_bound(diff, b["block_days"], b["draws"], b["seed"], b["level"])
    return {"block": p, "days": int(len(days)), "mean_daily_diff_usd": float(diff.mean()) if len(diff) else float("nan"), "lower_bound": lb,
            "with_early_close": {k: m1[k] for k in ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct", "trades", "orders_per_week", "n_close_AGENT_CLOSE")},
            "never_close": {k: m0[k] for k in ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct", "trades", "orders_per_week")},
            "beneficial": bool(np.isfinite(lb) and lb > 0), "rule": spec["rule"], "bootstrap": b}


# ----------------------------------------------------------------------------- contract
def check_recipe_encoding(doc: dict, rec: dict):
    """A1-11: the C3 tensor must come from a recipe that APPLIED the selection's encoding decisions (derived by
    ``derive_encoding_recipe``); a recipe without them is refused."""
    got = rec.get("encoding_decisions_applied") or {}
    want = doc.get("encoding_decisions") or {}
    if {k: got.get(k) for k in ("regime_gmm", "event_clock")} != {k: want.get(k) for k in ("regime_gmm", "event_clock")}:
        raise SystemExit(f"C3_RECIPE_ENCODING_NOT_APPLIED:{got}")


def derive_encoding_recipe(base: dict, base_sha: str, store_parquet, fold: str) -> dict:
    """The C3 recipe derived from a (fold) recipe: the ENCODING.csv decisions applied as consumer ops (regime one-hot;
    event-clock 2 h windows; the duplicate hours_to clocks dropped), groups re-resolved, scaler constants re-fitted on
    the fold's fit block with the loader's rules; a new identity declaring ``encoding_decisions_applied``."""
    import copy
    from tools import expert_guided_wf_recipe as WR
    rec = copy.deepcopy(base)
    cols, out = rec["tensor"]["columns"], []
    gmm = [c["store_column"] for c in cols if c["store_column"] in ("rg.gmm.p0", "rg.gmm.p1", "rg.gmm.p2")]
    for c in cols:
        n = c["store_column"]
        if n.startswith("ev.") and n.endswith(".hours_to"):
            continue                                                   # A1-4: hours_to = 24 - hours_since
        c = dict(c)
        if n in gmm:
            c["consumer_ops"] = [{"op": "onehot_argmax", "sources": list(gmm), "index": gmm.index(n)}]
            c["tensor_name"] = f"rg.gmm.onehot{gmm.index(n)}"
            c["then"] = "one-hot of the argmax regime (ENCODING.csv USE_onehot), then fixed z with this recipe's constants"
        elif n.startswith("ev.") and n.endswith(".hours_since"):
            c["consumer_ops"] = [{"op": "window_hours", "hours": 2.0, "period_hours": 24.0}]
            c["tensor_name"] = n[:-len(".hours_since")] + ".window2h"
            c["then"] = "2 h event window (ENCODING.csv USE_window2h), then fixed z with this recipe's constants"
        out.append(c)
    rec["tensor"]["columns"] = out
    names = [c["store_column"] for c in out]
    tnames = [c["tensor_name"] for c in out]
    rename = {c["store_column"]: c["tensor_name"] for c in out}            # the group map names TENSOR columns (as e3m resolves them)
    rec["list"] = dict(rec["list"], n=len(names), columns=names, derived="C3 encoding decisions applied; hours_to clocks dropped")
    g = rec["representation"]["groups"]
    g["groups"] = {k: [rename[x] for x in v if x in rename] for k, v in g["groups"].items()}
    g["groups"] = {k: v for k, v in g["groups"].items() if v}
    for key in ("units", "kinds"):
        if isinstance(g.get(key), dict):
            g[key] = {k: v for k, v in g[key].items() if k in g["groups"]}
    res = rec["representation"]["resolved"]
    keep = [i for i, nm in enumerate(res["names"]) if nm in g["groups"]]
    res["names"] = [res["names"][i] for i in keep]
    res["units"], res["kinds"] = [res["units"][i] for i in keep], [res["kinds"][i] for i in keep]
    res["indices"] = [[tnames.index(x) for x in g["groups"][nm]] for nm in res["names"]]
    covered = sorted(i for ii in res["indices"] for i in ii)
    if covered != list(range(len(tnames))):
        raise ValueError(f"C3_DERIVED_GROUPS_DO_NOT_PARTITION:{len(covered)}/{len(tnames)}")
    scl = dict(rec["scaler"])
    scl.update(WR.fold_constants(Path(store_parquet), rec, fold))
    scl["features"], scl["n_features"] = [c["tensor_name"] for c in out], len(out)
    scl.pop("artifact_sha256", None)
    scl["derived"] = f"constants re-fitted on the {fold} fit block after the encoding ops (loader rules)"
    rec["scaler"] = scl
    rec["encoding_decisions_applied"] = dict(ENCODING_DECISIONS)
    rec["identity"] = WR.identity(rec)
    rec["derived_from"] = {"identity": base.get("identity"), "sha256": base_sha, "deviation": ["encoding_decisions", "list", "tensor.columns", "representation", "scaler"], "fold": fold}
    rec["provenance"] = {"base_provenance": copy.deepcopy(base.get("provenance")), "derived_by": "app/short_horizon/expert_guided/c3.py derive_encoding_recipe"}
    return rec


def architecture_of_arch(ARCH: dict) -> dict:
    """The declared trunk architecture of an E3-M run (``ARCH`` of e3m), with the u-design defaults filled in."""
    return {"trunk": str(ARCH.get("trunk", "u")), "fusion_units": int(ARCH.get("fusion_units", 8)), "core_blocks": int(ARCH.get("core_blocks", 2)),
            "readout": str(ARCH.get("readout", "flatten"))}


def check_architecture(doc: dict, ARCH: dict):
    """The sealed contract carries the trunk architecture: a run whose flags differ is refused (the identity would lie)."""
    want, have = doc.get("architecture") or {}, architecture_of_arch(ARCH)
    if {k: want.get(k) for k in ARCH_KEYS} != have:
        raise SystemExit(f"C3_ARCHITECTURE_MISMATCH:contract {want} vs run {have}")


def head_outputs(cells) -> dict:
    """Owner order 2026-10-26 (lane F9 reconciles these with the strategy proof on simulated outputs): per head, the
    encoding, the exact output range, the calibration and HOW the strategy consumes it. Nothing here is a free text
    only: the validator checks the block against the module constants."""
    n = len(cells)
    return {
        "open_l.decision": {"encoding": "softmax_ce", "classes": list(CLASSES_OPEN), "range": "3 probabilities in [0, 1], sum 1",
                            "calibration": "temperature scaling on INNER_A (argmax-preserving)",
                            "consumed_by": "EV = P(TP)*TP_cell - P(SL)*SL_cell + P(TIME)*E[net|TIME] - cost (probabilities at the reference cell); OPEN_LONG iff EV_L > open_margin (ONE pooled quantile gene) and EV_L > EV_S when both clear, and the gate places it"},
        "open_s.decision": {"encoding": "softmax_ce", "classes": list(CLASSES_OPEN), "range": "3 probabilities in [0, 1], sum 1",
                            "calibration": "temperature scaling on INNER_A (argmax-preserving)", "consumed_by": "as open_l for OPEN_SHORT (same margin; the higher EV wins; exact tie -> HOLD)"},
        "open_l.continuous": {"encoding": "bounded_sigmoid", "outputs": n, "range": f"each in [{NET_BOUND[0]:g}, {NET_BOUND[1]:g}] sigma (predicted engine net per unit / sigma)",
                              "calibration": "none (Huber-fitted regression; order-preserving)",
                              "consumed_by": "cell choice only: argmax over the ALLOWED cells (TP >= SL, or net > cost + rr_margin; inside the per-bar band); never thresholded into an action by itself"},
        "open_s.continuous": {"encoding": "bounded_sigmoid", "outputs": n, "range": f"each in [{NET_BOUND[0]:g}, {NET_BOUND[1]:g}] sigma", "calibration": "none", "consumed_by": "as open_l"},
        "nop.decision": {"encoding": "softmax_ce", "classes": ["TRADE", "NOP"], "range": "P(NOP) in [0, 1]", "calibration": "Platt on INNER_A",
                         "consumed_by": "veto: no open when P(NOP) > nop_veto (quantile gene); kept only if it improves the selection-block lower bound"},
        "close_l.decision": {"encoding": "softmax_ce", "classes": ["NOT_SL_FIRST", "SL_FIRST"], "range": "P(SL first) in [0, 1]", "calibration": "Platt on INNER_A",
                             "consumed_by": "CLOSE when P(SL first) > close_p_l (per-side quantile gene on long positioned visits)"},
        "close_s.decision": {"encoding": "softmax_ce", "classes": ["NOT_SL_FIRST", "SL_FIRST"], "range": "P(SL first) in [0, 1]", "calibration": "Platt on INNER_A",
                             "consumed_by": "CLOSE when P(SL first) > close_p_s"},
        "close_l.continuous": {"encoding": "bounded_sigmoid", "outputs": 1, "range": f"[{REM_BOUND[0]:g}, {REM_BOUND[1]:g}] sigma (predicted remaining value of HOLD)", "calibration": "none",
                               "consumed_by": "CLOSE when remaining < -close_margin_l (per-side quantile gene)"},
        "close_s.continuous": {"encoding": "bounded_sigmoid", "outputs": 1, "range": f"[{REM_BOUND[0]:g}, {REM_BOUND[1]:g}] sigma", "calibration": "none", "consumed_by": "CLOSE when remaining < -close_margin_s"},
        "closer_inputs": "recipe window + the order's own geometry (dist to SL / TP in sigma_entry, hours held / 24, side); openers / NOP: recipe window only",
        "gate": "the EV of the chosen open is the SCORE of exec_variants.BestOfDayGate (observe 1 signal per business day, then the first EV >= the day's running best)",
        "early_close_switch": "exec_variants.early_close_ok(agent close share >= 0.05, value lower bound >= 0) on the selection block; on failure the closers are not consulted (flagged)"}


STRATEGY_PARAMS_FILE = "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"
STRATEGY_PARAMS_SHA256 = "fdddbe481cce15e8676dcb8d2f2a2d87a733f20c3bab74ba60c42b47edb843be"   # the v10 canonical point; replaced by the DEAP point once it is proven on the full matrix
CLOSER_HEAD_CONTRACT = {
    "name": "first-passage curve head (market-only best continuation), tools/strategy_output_simulation.BCHead",
    "inputs": "market data only: the bars from the decision bar on, the per-bar spread profile, the session calendar (compulsory flats, rollovers) and the swap rates by side; NO position, entry, SL, TP, units or balance",
    "indexing": "per decision bar b and side s; lead h = 1 .. H over the future bars b + h",
    "horizon_H": "variable: the bars up to the first compulsory flat after b + 1 (<= one trading week of hourly bars)",
    "units": "sigma of now = the causal decision sigma at b; values relative to closing now at the open of b + 1",
    "channels": {"adv": "running adverse excursion reached by bar b + h (the stop side, shorts at ask)", "fav": "running favourable excursion reached by bar b + h (the take-profit side, through limit_through)",
                 "pm": "best net value of a discretionary close decided at or before b + h (fill at the next open, exit cost and financing included)",
                 "o": "open of b + h relative to the open of b + 1", "gap": "the gap test price of b + h (shorts at ask)", "cs": "stop exit cost relative to a market close now",
                 "ct": "take-profit exit cost relative to a market close now", "fo": "financing accrued to an open fill at b + h", "fc": "financing accrued to an intrabar / close fill at b + h",
                 "ff": "net value of the compulsory flat at the close of the last bar (scalar)"},
    "decoder": "the strategy (L2) supplies only its position's remaining distances d_sl = s (liq_now - SL) / sigma and d_tp = s (TP - liq_now) / sigma; first stop bar = first h with adv >= d_sl, first take-profit bar = first h with fav >= d_tp (a shared bar goes to the stop unless the take-profit gaps at the open); BC = max(pm before the terminating bar, the termination's net value); A(t) = -BC in sigma of the trade",
    "quality": "below quality 1 the head is a CALIBRATED CONDITIONAL MEAN: y = BC + e, e = sd_e(h) x AR(1) stream across ticks (phi = 0.587, measured on the committed hourly predictor errors), sd_e(h) = sigma_for_rho(rho, sd_BC) x sqrt(h / 24) (rho at h_ref = 24 bars, h the decisive lead); BC_hat = mu + rho_h^2 (y - mu), rho_h^2 = sd_BC^2 / (sd_BC^2 + sd_e(h)^2), mu / sd_BC the calibration-block moments; a predictive-sd channel s_hat = sd_BC sqrt(1 - rho_h^2)",
    "decision_rule": "the strategy closes when A - z_close x s >= -closer_adv_margin on k_persist consecutive executable ticks (STRATEGY_PARAMS genes); with closer_filter = 1, (A, s) = the Kalman posterior mean / sd of A given this position's own past head outputs (strategy_zz.CloserFilterModel: A_hat = alpha(h) + beta(h) A + AR(1) eps, A a random walk; phi, variances, alpha / beta per lead bin and the prior ESTIMATED on the calibration block only, head output vs realized hindsight A), otherwise (A_hat, s_hat); at quality 1, s_hat = 0 and the head is read as is",
    "horizon_map": "the head is market-only and horizon-free: it is decoded at the position's remaining distances whatever the opener's horizon (full_f4: the absolute SL / TP of the decision's MT horizon); the horizon enters the closer only through those distances",
    "proof_market_only": "BCHead.row(b, s) takes no position argument; SimCloser.advantage reads the position only to compute (d_sl, d_tp) and never the trade's own path (tests/test_strategy_behaviour_matrix.py)",
}
MENU_CHANGELOG = {"when": "2026-10-29", "change": "v2.1 (full_f4 only): the DECISION MENU (menu.py; coordinator ruling ~05:50Z) -- the policy values exactly the orders it places: "
                  "the candidate orders {(h, side): h in 3 / 6 / 12 / 24} built at the decision bar by horizon_levels.levels (+ NOP), a horizon without a confirming MT level "
                  "out of the menu at that bar; the engine truth = the exact replay of THAT menu on the profile engine (net sigma and USD, first barrier, MAE / MFE, exit reason); "
                  "heads, quality ladder, truth oracle, O*, hold oracle, forced-entry episodes and exploration visits on the same menu; EV ranks (h, side); horizon_rule = the "
                  "tie-break among confirming horizons; W7 = the executed (h, side) is the best of the truth menu among the allowed slots, W2 / W10 judge the executed order; "
                  "the protective TP (k_tp_protect x MFE, tp_mode) is INERT under full_f4 (a contracted TP is never inflated) and out of that profile's DEAP genome",
                  "why": "tier 1 on fbe52e7 ranked the sides on sigma-cell predictions and executed absolute horizon orders (MENU_MISMATCH): 643 trades, 58 % SL exits, 31 % of re-opens "
                         "within one bar of the previous close on B_cal_cutoff12 s0; supersedes fbe52e7's 'truth on sigma, orders absolute' for full_f4; full_f3 / LEGACY keep the sigma grid, byte-identical"}
CONTRACT_CHANGELOG = (
    {"when": "2026-10-29", "change": "v2: the shared business contract (owner decision: Musashi's packet adopted wholesale) as the execution profile full_f4 -- leverage 50, no daily cap, the calendar's 12 h cutoff, absolute per-horizon levels from the TRAIN references, risk-budget sizing, the four MT horizon members, the fixed-delay closer comparator; STRATEGY_PARAMS v2 (a_sl, b_sl, buffer_pips, a_tp, b_tp, rho, ref_p75, horizon_rule)",
     "why": "S1-S6 of the owner's order 2026-10-29; the audited full_f3 profile stays the declared control (exec_profile full_f3 / absent): byte-identical"},
    {"when": "2026-10-28", "change": "the closer filters its own past head outputs per position (closer_filter gene; parameters estimated on the calibration block)",
     "why": "QUALIFICATION_SPEC D7: the per-tick rule failed regret <= the information-limited Bayes closer + 0.05 (the Bayes closer averages the head history); coordinator ruling: a causal, market-only filter of the strategy's own past signals, never the simulator's true noise model"},
    {"when": "2026-10-28", "change": "the closer's output contract declared: the market-only first-passage curve head (closer_head)",
     "why": "the architecture contract requires a market-only model observation; the earlier advantage head read the trade's own path (position-conditioned); the head's horizon, channels, units, decoder, horizon-dependent quality noise and the market-only proof are now part of the sealed identity"},
    {"when": "2026-10-27", "change": "contract regenerated from the code and re-sealed; the strategy parameter file and its sha are embedded",
     "why": "since 6a38ccf close_p_l / close_p_s / nop_veto are ABSOLUTE calibrated-probability thresholds (genes in [0.30, 0.95]); the committed contracts "
            "were edited by hand and drifted from default_contract() (TCN file: action_bef6627ca7881000 vs code action_db1380893d680db6; the U control "
            "file still carried the quantile genes close_p_rate / nop_veto_rate). The requirement change was a fix proven by the behaviour matrix: a "
            "quantile over a saturated reference mapped close_p to 1e-6 under perfect closers and closed 49 value-positive positions in one cell, and "
            "nop_veto mapped to 1.0 and never vetoed"},
)


def default_contract(architecture: dict | None = None, exec_profile: str = EXEC_DEFAULT) -> dict:
    """C3 contract; the DEFAULT trunk is ``residual_tcn`` (owner 2026-10-26: every model of this line uses the TCN trunk);
    the u-design conv1d trunk is a declared CONTROL arm (``ACTION_CONTRACT_C3_U.json``), never the default."""
    base = C2.default_contract_geometry()
    architecture = dict(ARCH_TCN if architecture is None else architecture)
    doc = json.loads(json.dumps(base))
    doc["schema"], doc["identity"] = SCHEMA, None
    doc["design"] = ("docs/short_horizon/clean_round/ADVERSARIAL_AUDIT_REGISTER_2026_10_26.md + PIPELINE_CONTROLS_2026_10_26.md (C3: label / target "
                     "fixes A2-1, A2-2, A2-3, A2-4, A2-5, A2-6, A2-8, A2-9, A2-10, A2-11, A2-12, A4-3, A4-5, A4-6) on C2G")
    cells = [list(c) for c in CELLS]
    cont_open = {C2.cell_name(c): {"encoding": "bounded_sigmoid", "lo": NET_BOUND[0], "hi": NET_BOUND[1], "output_dim": 1, "loss": "huber",
                                   "role": "expected_value_sigma", "target": "engine net result per unit / causal sigma of the fill bar, this (TP, SL) cell",
                                   "mask": "every clean-account eligible flat row whose outcome is available (all cells)"} for c in CELLS}
    for m in OPENERS:
        doc["specialists"][m]["decision"] = {"encoding": "softmax_ce", "classes": list(CLASSES_OPEN), "n_classes": 3, "output_dim": 3, "loss": "cross_entropy",
                                             "class_weights": "1/sqrt(train frequency), normalised (as v1)",
                                             "target": f"which barrier the engine hits first at the reference cell {list(REF)}: TP_FIRST / SL_FIRST / TIME (A2-3: TIME is its own class)"}
        doc["specialists"][m]["continuous"] = cont_open
        doc["specialists"][m]["stop_loss_head"] = "none: the SL is the chosen cell's declared constant (A2-1 / A4-3); a regression target with std < MIN_TARGET_STD on TRAIN refuses the run"
    doc["outcomes"] = {"grid_tp": list(GRID_TP), "grid_sl": list(GRID_SL), "cells": cells, "ref_cell": list(REF),
                       "cap_rule": f"grid max 2.0 sigma <= the engine's effective cap p1 on TRAIN ({CAP_P1_SIGMA} sigma): no cell is clipped (A2-9 / A4-10); extended below 0.75 (0.5)",
                       "engine": base["outcomes"]["engine"], "availability": base["outcomes"]["availability"], "window_end": OC.TEST_START,
                       "units": base["outcomes"]["units"], "target_clip": [list(NET_BOUND), list(REM_BOUND)], "min_target_std": MIN_TARGET_STD}
    doc["calibration"] = {"method": CALIBRATION_METHOD, "fit_block": "INNER_A", "ece_bins": 15, "heads": [f"{m}.decision" for m in SPECIALISTS],
                          "openers": "temperature scaling on the 3-class logits (argmax-preserving), NLL on a declared grid", "closers_and_nop": "platt_binary"}
    st = ACT.v13_strategy_block(doc["strategy"], Q_BOUNDS, {
        "open_margin": "quantile(EV over the allowed cells POOLED over both sides, calibration flat rows, 1 - open_rate): one absolute margin for both sides",
        "close_margin_l": "-quantile(predicted remaining value, calibration LONG positioned visits, close_rem_rate_l)",
        "close_margin_s": "-quantile(predicted remaining value, calibration SHORT positioned visits, close_rem_rate_s)",
        "close_p_l": "absolute threshold on calibrated P(SL first) (gene close_p_l in [0.30, 0.95])",
        "close_p_s": "absolute threshold on calibrated P(SL first) (gene close_p_s in [0.30, 0.95])",
        "nop_veto": "absolute threshold on calibrated P(NOP) (gene nop_veto in [0.30, 0.95])"})
    for k in ("open_margin_l", "open_margin_s", "close_margin", "close_p"):
        st["gene_map"].pop(k, None)
    st["side_symmetry"] = SIDE_SYMMETRY_RULE
    st["gene_map"]["positioned_visits"] = "the round-0 INNER_A exploration visits (TP / SL drawn from the grid; NO expert position): strategy-free"
    st["name"] = "c3_ev_per_side"
    st["rule"] = ("flat: per side, the ALLOWED cell (TP >= SL, or predicted net > cost + rr_margin) with the highest predicted net; "
                  "EV = P(TP first) * TP - P(SL first) * SL + P(TIME) * E[net | TIME] - cost (probabilities of the 3-class head at the reference "
                  "cell, temperature-calibrated); open the side with the higher EV among those with EV > open_margin (one pooled margin; exact tie -> no trade) and (veto kept) pNOP <= nop_veto; 'no trade' "
                  "is valid; positioned: CLOSE when the closer's predicted remaining < -close_margin_side OR calibrated P(SL first) > "
                  "close_p_side (per-side thresholds), then reverse when allow_reverse and the open rule names the other side")
    st["ev_rule"] = {"formula": "P(TP)*TP - P(SL)*SL + P(TIME)*E[net|TIME] - cost", "probabilities": f"3-class head at {list(REF)}",
                     "e_time": "mean engine net of TIME outcomes per side at the reference cell on TRAIN (declared constant)",
                     "cost": "median over the calibration rows of the per-row cost (declared allowance + round trip / sigma)",
                     "rr_floor": {"rule": "TP >= SL always allowed; TP < SL only when predicted net > cost + rr_margin", "rr_margin_sigma": RR_MARGIN_SIGMA},
                     "activity": "a VALIDITY criterion (>= 2.0 trades/week lower CI bound on calibration); never a reason to open at negative EV"}
    st["threshold_targets"] = {"close_p_l": "close_l.SL_FIRST", "close_p_s": "close_s.SL_FIRST", "nop_veto": "nop.NOP"}
    st["value_targets"] = {"open_margin": "open_l.value:net_|open_s.value:net_",
                           "close_margin_l": "close_l.value:remaining_sigma", "close_margin_s": "close_s.value:remaining_sigma"}
    st["early_close_benefit"] = dict(EARLY_CLOSE_BENEFIT)
    doc["strategy"] = st
    doc["action_space"]["produced_by"]["OPEN_LONG"] = [["open_l", "value:net_"], ["open_l", "TP_FIRST"]]
    doc["action_space"]["produced_by"]["OPEN_SHORT"] = [["open_s", "value:net_"], ["open_s", "TP_FIRST"]]
    doc["label_rules"] = {
        "openers": "every clean-account eligible flat row of the period (gate open: A2-10); the daily cap is enforced by the env only, never in a label (A2-5)",
        "closers": "exploration visits only in round 0 (A2-6), the student's own visits in DAgger rounds relabelled with the SAME engine rule (A2-5); duplicates by (row, side, geometry) deduplicated, ties dropped (A2-4)",
        "zigzag": "no ZigZag target (A2-11 not applicable); NOP keeps the multires ladder labels",
        "teacher": "none"}
    doc["loss_balance"] = dict(LOSS_BALANCE)
    doc["head_outputs"] = head_outputs(cells)
    doc["encoding_decisions"] = dict(ENCODING_DECISIONS)
    xv = EXEC_VARIANTS[exec_profile]
    doc["exec_variant"] = {"name": exec_profile, "identity": xv.identity, "options": xv.document(),
                           "legacy_control": {"name": "legacy", "identity": XV.LEGACY.identity, "how": "--exec-variant legacy (declared control)"},
                           "fixes": "A4-1 reversal = one entry decision; A4-4 risk 1 % sizing; A4-9 entry cutoff 24 h before the forced flat; A4-12 ask-side short triggers",
                           "alternatives": {"full_f3_12h": {"identity": FULL_F3_12H.identity, "options": FULL_F3_12H.document()}, "legacy": XV.LEGACY.identity,
                                            "full_f3": FULL_F3.identity, "full_f4": FULL_F4.identity},
                           "cutoff_eligibility": dict(CUTOFF_ELIGIBILITY)}
    if exec_profile == "full_f4":                                     # the shared business contract (owner decision 2026-10-29)
        from . import business_contract as BC
        from . import horizon_levels as HL
        from . import strategy_zz as ZZ
        doc["business_contract"] = {"file": str(BC.BUSINESS_CONTRACT_PATH.relative_to(BC.ROOT)), "sha256": BC.BUSINESS_CONTRACT_SHA256, "source": dict(BC.BUSINESS_CONTRACT_SOURCE),
                                    "profile": BC.profile(),
                                    "levels": {"mode": "absolute_pips", "references": {"file": str(HL.PROFILES_PATH.relative_to(HL.ROOT)), "sha256": HL.PROFILES_SHA256, "quantile": "median (ref_p75 = 1: p75)",
                                                                                        "status": "hindsight TRAIN oracle populations used as priors; our own level labels are recomputed after S4 and reported next to them"},
                                               "rule": "dSL = a_sl x MAE_ref + b_sl x sigma_h + buffer_pips x pip; dTP = a_tp x TP_ref - b_tp x sigma_h; SL rounded away from the entry, TP toward it, tick 1e-5; "
                                                       "NOP when a level is not beyond the reference close and the liquidation-side quote by the broker minimum; no sigma bounds, no effectiveness band, nothing clipped; a contracted TP is never inflated",
                                               "sigma_h": HL.SIGMA_H_RULE, "tick": ZZ.LEVEL_TICK},
                                    "horizons": {"members": {k: {"depth": v[0], "backstep": v[1], "hours": v[2]} for k, v in ZZ.MT_HORIZONS.items()}, "deviation_pips": ZZ.MT_DEVIATION_PIPS,
                                                 "deviation_status": "declared constant, INERT in the replica (verified 2026-10-29)", "rule": ZZ.HORIZON_RULE, "options": list(ZZ.HORIZON_RULES),
                                                 "no_horizon": "NOP", "d2": "the MT confirmations pass the same quality ladder as every hierarchy head; the causal-confirmed control reads every candidate at i + backstep"},
                                    "sizing": {"rule": "q = c x min(1, rho x balance / L1), L1 = units_max(q = 1) x (SL distance + cost per unit + funding reserve); units = floor(q x units_max); c = 1 (fixed until head confidence is calibrated); funding reserve 0",
                                               "cost_per_unit": "the engine's convention: slippage market + slippage stop + 2 x commission per lot / lot units; the spread is in the quotes; swaps by side per rollover in the P&L",
                                               "loss_bound_status": "a modeled stop loss, never a bound on gap losses (W9 stays)"},
                                    "decision_menu": {"module": "app/short_horizon/expert_guided/menu.py", "horizons": [3, 6, 12, 24], "slots": "(h, side) absolute orders + NOP; the world's cells are ('H', h)",
                                                      "membership": "a slot is in the menu at bar b iff its MT level confirms the side at b (HORIZON_RULE per level); none = NOP",
                                                      "truth": "exact replay of every menu order on the profile engine (clean account, profile sizing and costs): net sigma / USD, first barrier, MAE / MFE, exit reason; the ideal confirmations as the truth menu",
                                                      "ranking": "EV over the allowed slots of each side (RR rule on the order's own distances, RR floor on the MAE), the higher side wins; horizon_rule (finest | coarsest) breaks ties",
                                                      "witnesses": "W7 = executed (h, side) best of the truth menu among the allowed slots (both sides); W2 / W10 judge the executed order's slot; W13 menu form = executed TP / SL equal the slot's levels (1 tick); W14 judged by its oracle control with the side-swapped-reference mirror as the corrected form",
                                                      "inert_under_menu": {"k_tp_protect": "a contracted TP is never inflated", "tp_mode": "the protective envelope is off"},
                                                      "ref_slot": "h = 6 (the 3-class opener head and E[net | TIME]; unused by ev_from_net = 1)",
                                                      "exploration": "a slot drawn uniformly among the four horizons (confirmation-free order geometry)",
                                                      "level_genes_guard": "the truth is stamped with a_sl / b_sl / buffer_pips / a_tp / b_tp / quantile; a policy with other level genes is refused (MENU_GENES_MISMATCH)"},
                                    "closers": {"fixed_delay_comparator": "TRAIN-selected fixed delay (CAL block, mean cash-risk-normalized regret, passive hold included) as a named D7 comparator",
                                                "separating_fixture": "native-engine unit test: quote 100, path 98 -> 104, A SL 99 / TP 106 -> CLOSE (A = +0.333, the next-open close is the optimum), B SL 97 / TP 104 -> HOLD; mirror; A' flips to HOLD iff market-close cost - stop cost > d_SL + EPS_CLOSE"}}
    doc["best_of_day"] = dict(BEST_OF_DAY)
    doc["outcomes"]["per_bar_band"] = "cells outside the engine's per-bar effective band [min_multiple, effective_cap] are not selectable at that bar (exec_variants.effective_cap / min_multiple)"
    doc["architecture"] = {k: architecture[k] for k in ARCH_KEYS}
    doc["architecture"]["note"] = ("inside the sealed identity (2026-10-26): residual_tcn (DEFAULT, owner order: every model of this line) = per-group "
                                   "single causal conv, SiLU, 1x1 fusion 8, two residual causal blocks (dilations 1, 2), flatten readout "
                                   "(temporal.py, parity-tested against Musashi's GroupedTCNExtractor); u = the sealed u-design conv1d trunk, "
                                   "a declared CONTROL arm only")
    doc["protocol_v11"] = {"blocks": "early stopping on sc['early_stop'], calibration + gene maps on sc['calibration'], the strategy search / fitness "
                                     "choice on sc['selection'], the decision block read once after the freeze and ledgered (READ_LEDGER.jsonl)",
                           "search_overfit": "per final search: deflated Sharpe of the chosen point (n_trials = distinct archived points, T = "
                                             "selection-block days) and PBO over the valid front points' daily returns on the selection block"}
    doc["visitation"] = {"round0": "exploration positions only (visited STATES, TP / SL from the grid), never the expert",
                         "rounds": "1..3: the C3 policy at the visitation strategy frozen on INNER_A (closer rows appended, closers refitted)",
                         "openers_and_nop": "fitted once in round 0 (their rows are every clean-account eligible flat row)"}
    doc["constraints"]["teacher"] = "none"
    doc["derived_from"] = {"identity": base.get("identity"), "why": "adversarial audit 2026-10-26: labels taught losing trades (TP 0.43 sigma vs constant SL 2 sigma), closers blind / leaky, DAgger contradictions, wrong probability in the gain rule"}
    doc["strategy"]["params"] = {"file": STRATEGY_PARAMS_FILE, "sha256": STRATEGY_PARAMS_SHA256,
                                 "rule": "every strategy coefficient lives in this file (validated against strategy_params.BOUNDS); the frozen contract embeds the sha of the point it was proven with"}
    doc["closer_head"] = dict(CLOSER_HEAD_CONTRACT)
    doc["changelog"] = ([dict(MENU_CHANGELOG)] if exec_profile == "full_f4" else []) + list(CONTRACT_CHANGELOG)   # the menu entry belongs to the full_f4 contract only
    return ACT.seal(doc)


def validate(doc: dict) -> list:
    errs = []
    if doc.get("schema") != SCHEMA:
        errs.append("C3_CONTRACT_SCHEMA")
    if doc.get("identity") != ACT.identity_of(doc):
        errs.append("C3_CONTRACT_IDENTITY")
    if doc.get("scientific_release") is not False:
        errs.append("C3_CONTRACT_RELEASE_FLAG")
    sp = doc.get("specialists") or {}
    if tuple(sp) != SPECIALISTS:
        errs.append(f"C3_CONTRACT_SPECIALISTS:{list(sp)}")
    errs += C2.encoding_audit(doc)["errors"]
    for m, d in sp.items():
        slots = list(d.get("state_slots") or [])
        for x in slots:
            if x not in GEOMETRY:
                errs.append(f"STATE_SLOT_NOT_ALLOWED:{m}:{x}")
            elif m not in CLOSERS:
                errs.append(f"GEOMETRY_ON_NON_CLOSER:{m}:{x}")
        if m in CLOSERS and slots != list(GEOMETRY):
            errs.append(f"C3_CLOSER_GEOMETRY:{m}:{slots}")
        for q, c in (d.get("continuous") or {}).items():
            if "sl" in q.lower().split("_")[0] or (c.get("role") or "") in ("stop_loss", "sl_sigma"):
                errs.append(f"C3_SL_HEAD_FORBIDDEN:{m}.{q}")
    oc = doc.get("outcomes") or {}
    cells = [tuple(c) for c in oc.get("cells") or []]
    if cells != [tuple(c) for c in CELLS] or tuple(oc.get("ref_cell") or ()) != REF or oc.get("window_end") != OC.TEST_START:
        errs.append("C3_CONTRACT_OUTCOMES")
    if max(max(c) for c in cells or [(9, 9)]) > CAP_P1_SIGMA:
        errs.append("C3_GRID_ABOVE_EFFECTIVE_CAP")
    if float(oc.get("min_target_std", 0)) != MIN_TARGET_STD:
        errs.append("C3_CONTRACT_MIN_TARGET_STD")
    for m in OPENERS:
        dec = (sp.get(m) or {}).get("decision") or {}
        if list(dec.get("classes") or []) != CLASSES_OPEN or int(dec.get("n_classes", 0)) != 3:
            errs.append(f"C3_CONTRACT_OPENER_CLASSES:{m}")
        ct = (sp.get(m) or {}).get("continuous") or {}
        if list(ct) != [C2.cell_name(c) for c in CELLS] or any((x.get("lo"), x.get("hi")) != NET_BOUND for x in ct.values()):
            errs.append(f"C3_CONTRACT_NET_HEADS:{m}")
    for m in CLOSERS:
        ct = (sp.get(m) or {}).get("continuous") or {}
        if list(ct) != ["remaining_sigma"] or (ct["remaining_sigma"].get("lo"), ct["remaining_sigma"].get("hi")) != REM_BOUND:
            errs.append(f"C3_CONTRACT_REMAINING_HEADS:{m}")
    nl = (sp.get("nop") or {}).get("labels") or {}
    if tuple(nl.get("ladder", ())) != tuple(MR.LADDER) or tuple(nl.get("positive", ())) != ACT.NOP_POSITIVE:
        errs.append("C3_CONTRACT_NOP_LABELS")
    cal = doc.get("calibration") or {}
    if cal.get("method") != CALIBRATION_METHOD or cal.get("fit_block") != "INNER_A" or int(cal.get("ece_bins", 0)) != 15:
        errs.append("C3_CONTRACT_CALIBRATION")
    s = doc.get("strategy") or {}
    genes = s.get("genes") or {}
    if s.get("gene_space") != ACT.GENE_SPACE_Q or tuple(genes) != GENES_Q or genes != Q_BOUNDS:
        errs.append(f"C3_CONTRACT_GENES:{list(genes)}")
    errs += [f"C3_CONTRACT_{e}" for e in ACT.v13_rule_errors(s, True)]
    if s.get("side_symmetry") != SIDE_SYMMETRY_RULE:
        errs.append("C3_CONTRACT_SIDE_SYMMETRY")
    ev = s.get("ev_rule") or {}
    if float((ev.get("rr_floor") or {}).get("rr_margin_sigma", -1)) != RR_MARGIN_SIGMA:
        errs.append("C3_CONTRACT_EV_RULE")
    if s.get("require_early_close") is not True or (s.get("early_close_benefit") or {}) != EARLY_CLOSE_BENEFIT:
        errs.append("C3_CONTRACT_EARLY_CLOSE_RULE")
    if float(s.get("calibration_min_trades_per_week", 0)) != ACT.MIN_TRADES_PER_WEEK or s.get("fit_block") != "INNER_A":
        errs.append("C3_CONTRACT_ACTIVITY_RULE")
    if (doc.get("loss_balance") or {}) != LOSS_BALANCE:
        errs.append("C3_CONTRACT_LOSS_BALANCE")
    lr = doc.get("label_rules") or {}
    if lr.get("teacher") != "none" or not str(lr.get("closers")).startswith("exploration visits only") or "env only" not in str(lr.get("openers")):
        errs.append("C3_CONTRACT_LABEL_RULES")
    if (doc.get("encoding_decisions") or {}) != ENCODING_DECISIONS:
        errs.append("C3_CONTRACT_ENCODING_DECISIONS")
    if (doc.get("head_outputs") or {}) != head_outputs(cells or CELLS):
        errs.append("C3_CONTRACT_HEAD_OUTPUTS")
    ev_ = doc.get("exec_variant") or {}
    try:
        ok_xv = ev_.get("name") in EXEC_VARIANTS and XV.ExecVariant(**(ev_.get("options") or {})).identity == ev_.get("identity") == EXEC_VARIANTS[ev_["name"]].identity
    except (TypeError, ValueError):
        ok_xv = False
    if not ok_xv:
        errs.append("C3_CONTRACT_EXEC_VARIANT")
    if (doc.get("best_of_day") or {}) != BEST_OF_DAY:
        errs.append("C3_CONTRACT_BEST_OF_DAY")
    ar = doc.get("architecture") or {}
    if ar.get("trunk") not in ("u", "residual_tcn") or ar.get("readout") not in ("flatten", "last") or int(ar.get("fusion_units", 0)) < 1 or int(ar.get("core_blocks", 0)) < 1:
        errs.append("C3_CONTRACT_ARCHITECTURE")
    if not (doc.get("protocol_v11") or {}).get("blocks"):
        errs.append("C3_CONTRACT_PROTOCOL_V11")
    bl = doc.get("blocks") or {}
    if bl.get("calibration") != "INNER_A" or bl.get("decision") != "INNER_B" or bl.get("read_once") != "OBS_2024" or bl.get("requires_post_freeze_decision_block") is not True:
        errs.append("C3_CONTRACT_BLOCKS")
    c = doc.get("constraints") or {}
    if c.get("agent_state") != "none" or c.get("variants") != ["FULL"] or c.get("teacher") != "none" or c.get("closer_inputs") != "recipe_window+order_geometry":
        errs.append("C3_CONTRACT_CONSTRAINTS")
    return errs


def load(path) -> dict:
    doc = json.loads(Path(path).read_text())
    errs = validate(doc)
    if errs:
        raise SystemExit(f"C3_CONTRACT_INVALID:{errs}")
    return doc


# ----------------------------------------------------------------------------- protocol v11: blocks, overfit statistics, ledger
def blocks_of(sc: dict) -> dict:
    """The contract's blocks (nested folds name four; wf2022 / wf2023 / consolidated fold them into INNER_A / INNER_B)."""
    return {"early_stop": sc.get("early_stop", "INNER_A"), "calibration": sc.get("calibration", "INNER_A"),
            "selection": sc.get("selection", "INNER_A"), "decision": sc.get("decision", "INNER_B")}


def search_overfit(fr: dict, cx, table, SCFG, cells, e_time, cost, sel_period: str, gm=None) -> dict:
    """Per final search (v11 hook 3): the deflated Sharpe of the chosen point (n_trials = distinct archived points, T =
    selection-block days) and the PBO over the valid front points' daily returns on the selection block."""
    pts = fr["runs"]["without_veto"]["points"] + fr["runs"]["with_veto"]["points"]
    n_trials = len({tuple(sorted(p["genes"].items())) + (bool(p.get("veto")),) for p in pts})
    front = [p for p in (fr["runs"]["without_veto"].get("front", []) + fr["runs"]["with_veto"].get("front", [])) if p.get("valid")]
    rets, days = [], None
    for p in front[:48]:
        g = gm.to_absolute(p["genes"]) if gm is not None else dict(p["genes"])      # archived genes are the quantile genes
        st_ = StrategyC3.from_genes(dict(g, nop_veto=g.get("nop_veto", 1.0)), bool(p.get("veto")))
        m, _ = ACT.rollout_with_trades(cx, sel_period, PolicyC3(table, st_, cells, e_time, cost)(cx), SCFG)
        r = ST.daily_returns(m["daily_equity"], SCFG.capital)
        rets.append(r)
        days = len(r)
    out = {"n_trials_archived": int(n_trials), "front_points_replayed": len(rets), "selection_block": sel_period}
    ch = fr.get("chosen")
    if ch is not None and days:
        sr_daily = float(ch["sharpe"]) / ST.ANNUALIZATION if ch.get("sharpe") is not None and np.isfinite(ch["sharpe"]) else float("nan")
        out["deflated_sharpe"] = ST.deflated_sharpe(sr_daily, max(n_trials, 1), days) if np.isfinite(sr_daily) else {"note": "chosen Sharpe undefined"}
    if len(rets) >= 2:
        T = min(len(r) for r in rets)
        out["pbo"] = ST.pbo(np.stack([r[:T] for r in rets], axis=1))
    else:
        out["pbo"] = {"note": "fewer than 2 valid front points"}
    return out


# ----------------------------------------------------------------------------- one seed / the run
def run_seed(cx, seed, d):
    """One seed under C3 (the C2 loop with the C3 sets, calibration, policy, per-side closer references and the
    early-close benefit test). Blocks (v11): early stopping on ``blocks['early_stop']``, calibration and gene maps on
    ``blocks['calibration']``, the strategy search / fitness choice / benefit test on ``blocks['selection']``; the
    decision block is never read here."""
    AC, SCFG, NS, cells, e_time, cost, B = d["contract"], d["SCFG"], d["NS"], d["cells"], d["e_time"], d["cost"], d["blocks"]
    clipped, dedup = {}, {}
    xt = d["x0"]
    D_tr, _ = sets(cx, d["et"], xt["TRAIN"], "TRAIN", d["nopl"], NS, cells, clipped, dedup_doc=dedup)
    target_std = check_target_std(D_tr, cells)
    D_es, _ = sets(cx, d["et"], xt[B["early_stop"]], B["early_stop"], d["nopl"], NS, cells, clipped, dedup_doc=dedup)
    go_in = {}
    D_in, rows_in = sets(cx, d["et"], xt[B["calibration"]], B["calibration"], d["nopl"], NS, cells, clipped, geo_out=go_in, dedup_doc=dedup)
    closer_src = {m: {"rows": rows_in[m], "G": go_in.get(m)} for m in CLOSERS}
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    primary = d["fitness"]
    SEL = B["selection"]

    def score_table(view, strat):
        m, trades = ACT.rollout_with_trades(cx, SEL, PolicyC3(view, strat, cells, e_time, cost)(cx), SCFG)
        return ACT.window_components(m, trades, d["sig"], SCFG.capital, min_opw, **ACT.score_kwargs(AC))

    pols, infos, rounds, comp = {}, {}, [], None
    visit_periods = tuple(dict.fromkeys(("TRAIN", B["early_stop"], B["calibration"])))
    stores = {"TRAIN": D_tr, B["early_stop"]: D_es, B["calibration"]: D_in}
    for rnd in range(d["rounds"] + 1):
        fit_now = list(SPECIALISTS) if rnd == 0 else list(CLOSERS)
        if rnd > 0 and comp is not None:
            for p in visit_periods:
                store = stores[p]
                xr = OC.exit_table(cx, SCFG, OC.visit_positions(cx, p, comp(cx), SCFG), geometry_fn=C2.geometry_fn(cx))
                for m in CLOSERS:
                    r, y, R, Gm = C2.exit_sets_geo(cx, xr, p, SIDE_OF[m])
                    r, y, R, Gm, _, _ = dedup_closer(r, y, R, Gm)
                    R, _ = C2.clip_targets(R, REM_BOUND)
                    store[m] = C2.cat_sets(store[m], (C2.X_geo(cx.Xw, r, NS, Gm), y, R, np.ones_like(R)))
        for m in fit_now:
            nc = 3 if m in OPENERS else 2
            if len(D_tr[m][1]) == 0 or len(D_es[m][1]) == 0 or len(np.unique(D_tr[m][1])) < 2:
                if m in pols:
                    d["ledger"](d["led"], rung="E3M_C3", event="MODULE_KEPT", module=m, seed=seed, round=rnd)
                    continue
                raise SystemExit(f"C3_EMPTY_PARTITION:{m}:round{rnd}")
            d["ledger"](d["led"], rung="E3M_C3", event="FIT", module=m, seed=seed, round=rnd, rows_train=int(len(D_tr[m][1])),
                        rows_early_stop=int(len(D_es[m][1])), early_stop_block=B["early_stop"], label_counts=np.bincount(D_tr[m][1], minlength=nc).tolist())
            pols[m], infos[m] = d["fit_module"]("FULL", m, D_tr[m], D_es[m], seed)
        cal = calibrate(pols, D_in)
        table = calibrate_table(raw_table(pols, cx.Xw, d["rows_all"], NS, len(cells)), cal)
        table["closer"] = C2.CloserGeo(cx, [(pols, cal)], NS)
        table["cap"], table["lo"] = d["caps"]
        if rnd == d["rounds"]:
            break
        fr = freeze(table, d["view_rows"], score_table, AC, seed, "visitation", primary, closer_src, cells, e_time, cost, d["cal_rows"])
        st_ = C2.strategy_of(fr)
        rounds.append({"round": rnd, "visitation_strategy": fr["strategy"], "valid": fr["valid"], "chosen": fr["chosen"],
                       "fallback": None if st_ else "no valid visitation strategy on the selection block: the round adds no visited closer rows",
                       "closer_rows": {m: int(len(D_tr[m][1])) for m in CLOSERS}})
        d["ledger"](d["led"], rung="E3M_C3", event="VISITATION_STRATEGY", seed=seed, round=rnd, strategy=fr["strategy"])
        comp = PolicyC3(table, st_, cells, e_time, cost) if st_ is not None else None
    fr = freeze(table, d["view_rows"], score_table, AC, seed, "final", primary, closer_src, cells, e_time, cost, d["cal_rows"])
    st_ = C2.strategy_of(fr) or NO_TRADE
    gm = gene_map(C2.restrict(table, d["view_rows"]), d["cal_rows"], closer_reference_sides(table, closer_src), cells, e_time, cost)
    fr["search_overfit"] = search_overfit(fr, cx, table, SCFG, cells, e_time, cost, SEL, gm)
    benefit = early_close_benefit(cx, table, st_, SCFG, cells, e_time, cost, dict(EARLY_CLOSE_BENEFIT, block=SEL))
    w = benefit["with_early_close"]
    share = float(w["n_close_AGENT_CLOSE"]) / max(int(w["trades"]), 1)
    ok, reason = XV.early_close_ok(share, benefit["lower_bound"], EARLY_CLOSE_MIN_SHARE)
    benefit.update({"agent_close_share": share, "early_close_ok": ok, "reason": reason, "beneficial": ok})
    flag = None
    if not benefit["beneficial"]:
        st_ = st_.no_early_close()
        flag = "early_close_disabled"
        fr["strategy"] = st_.to_dict()
    d["ledger"](d["led"], rung="E3M_C3", event="FREEZE", seed=seed, strategy=fr["strategy"], chosen=fr["chosen"], chosen_on=SEL, early_close_flag=flag)
    extra = {"calibration": {m: {k: v for k, v in c.items()} for m, c in cal.items()},
             "strategy": fr, "visitation_rounds": rounds, "targets_clipped": clipped, "closer_dedup": dedup, "target_std_train": target_std,
             "early_close_benefit": benefit, "early_close_disabled": flag is not None, "flags": [flag] if flag else [],
             "ev_rule": {"e_time": list(e_time), "cost_sigma": cost, "rr_margin_sigma": RR_MARGIN_SIGMA}, "blocks_used": dict(B),
             "heads_calibration_block": head_report(pols, cal, D_in), "best_epochs": {m: int(infos[m]["best_epoch"]) for m in infos}}
    return pols, infos, extra, PolicyC3(table, st_, cells, e_time, cost), table


def freeze(table, view_rows, score_table, contract, seed, budget, primary, closer_src, cells, e_time, cost, cal_rows=None):
    """The view the optimiser sees = calibration + selection rows; the gene-map references read the calibration rows."""
    ref = closer_reference_sides(table, closer_src or {})
    cal_rows = view_rows if cal_rows is None else cal_rows
    return ACT.freeze_strategy(table, view_rows, score_table, contract, seed, budget, primary, gene_names=GENES,
                               make_strategy=StrategyC3.from_genes, restrict_fn=C2.restrict,
                               gene_map_fn=lambda view: gene_map(view, cal_rows, ref, cells, e_time, cost))


def head_report(pols, cal, D, bins=15) -> dict:
    out = C2.head_report({m: p for m, p in pols.items() if m not in OPENERS}, cal, D, bins)
    for m in OPENERS:
        if m not in pols or m not in D or len(D[m][1]) == 0:
            out[m] = {"rows": 0}
            continue
        lg, y = full_logits(pols[m], D[m][0]), np.asarray(D[m][1], int)
        rec = {"rows": int(len(y)), "raw": multiclass_metrics(softmax(lg), y, bins)}
        if m in cal:
            rec["calibrated"] = multiclass_metrics(softmax(lg, float(cal[m]["T"])), y, bins)
        _, reg = ACT.module_outputs(pols[m], D[m][0])
        if reg is not None and len(D[m]) > 2:
            R = np.asarray(D[m][2], float)
            per = [C2.reg_metrics(reg[:, j], R[:, j]) for j in range(R.shape[1])]
            rec["regression"] = {"per_output": per, "mean_mae": float(np.mean([p["mae"] for p in per])),
                                 "mean_spearman": float(np.nanmean([p["spearman"] for p in per]))}
        out[m] = rec
    return out


def run_e3m(ctx: dict) -> dict:
    """Called by ``tools/expert_guided_e3m.py run`` when the action contract is C3."""
    import torch
    from . import read_ledger as RL
    cx, ds, a, AC, res, out = ctx["cx"], ctx["ds"], ctx["a"], ctx["AC"], ctx["res"], Path(ctx["out"])
    SCFG, MCFG, NS, led, ledger, summary = ctx["SCFG"], ctx["MUSASHI_CFG"], ctx["NS"], ctx["led"], ctx["ledger"], ctx["summary"]
    DEV, SEEDS = ctx["DEV"], ctx["SEEDS"]
    check_architecture(AC, ctx.get("ARCH") or {})
    check_recipe_encoding(AC, ctx.get("RECIPE") or {})
    xv_name = EXEC_DEFAULT if getattr(a, "exec_variant", "contract") in ("", "contract") else a.exec_variant
    if xv_name not in EXEC_VARIANTS:
        raise SystemExit(f"C3_EXEC_VARIANT_UNKNOWN:{xv_name}")
    xv = EXEC_VARIANTS[xv_name]
    declared = {(AC.get("exec_variant") or {}).get("identity")} | {v.get("identity") for v in ((AC.get("exec_variant") or {}).get("alternatives") or {}).values() if isinstance(v, dict)} | {XV.LEGACY.identity}
    if xv.identity not in declared:
        raise SystemExit("C3_EXEC_VARIANT_MISMATCH: the contract's declared execution variant differs from the module's")
    SCFG = dataclasses.replace(SCFG, variant=None if xv.is_legacy() else xv)
    MCFG = dataclasses.replace(MCFG, variant=None if xv.is_legacy() else xv)
    B = blocks_of(ctx.get("sc") or {})
    DEC = B["decision"]
    fold = ctx.get("split_name", "fold")
    ledger_path = out.parent / RL.LEDGER_NAME
    cells = [tuple(c) for c in AC["outcomes"]["cells"]]
    ctx["fit_extra"].update({"reg_norm": "masked", "reg_weight": float(LOSS_BALANCE["huber_weight"])})
    op = Path(a.outcomes) if getattr(a, "outcomes", None) else out / f"OUTCOMES_ENTRY_C3_{xv_name}.npz"
    if op.exists():
        et = OC.load_table(op, cx, SCFG)
        if [tuple(c) for c in et["cells"].tolist()] != cells:
            raise SystemExit("C3_OUTCOMES_GRID_MISMATCH: the outcome table was built on another grid")
    else:
        et = OC.entry_table(cx, SCFG, cells=cells, progress=lambda i, n: print(json.dumps({"outcomes": f"{i}/{n}"}), flush=True))
        op.parent.mkdir(parents=True, exist_ok=True)
        OC.save_table(op, et)
    cal_rows = np.flatnonzero(cx.obs_ok & (ds.period == B["calibration"]))
    sel_rows = np.flatnonzero(cx.obs_ok & (ds.period == B["selection"]))
    view_rows = np.union1d(cal_rows, sel_rows)
    e_time = e_time_of(cx, et, "TRAIN", tuple(AC["outcomes"]["ref_cell"]), cells)
    cost = cost_sigma_of(cx, SCFG, float(res.get("cost_sigma", 0.0)), cal_rows)
    caps = cap_arrays(cx, SCFG)
    env_cap = OC.make_env(cx, SCFG)
    tr_bars = np.asarray(cx.ds.row_bar, np.int64)[np.flatnonzero(cx.obs_ok & (ds.period == "TRAIN"))]
    res["blocks_used"] = dict(B, note="early stopping / calibration / selection as named; the decision block is read once after the freeze and ledgered")
    res["c3"] = {"outcomes_file": str(op), "outcomes_stamp": et["stamp"], "outcomes_refused": et["refused"], "cells": [list(c) for c in cells],
                 "eligible_rows": {p: int((et["eligible"] & (ds.period == p)).sum()) for p in ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")},
                 "teacher": "none (round-0 positioned states from exploration only)", "contract_rule": AC["strategy"]["rule"],
                 "ev_rule": {"e_time_by_side": e_time, "cost_sigma": cost, "rr_margin_sigma": RR_MARGIN_SIGMA},
                 "architecture": architecture_of_arch(ctx.get("ARCH") or {}), "blocks": dict(B),
                 "exec_variant": {"name": xv_name, "identity": xv.identity, "options": xv.document(), "legacy_control": XV.LEGACY.identity},
                 "best_of_day": dict(BEST_OF_DAY), "cap_profile_train": XV.cap_profile(env_cap, tr_bars[tr_bars < len(env_cap.sigma) - 1]),
                 "loss_balance": dict(LOSS_BALANCE), "closer_order_geometry": {"inputs": list(GEOMETRY), "applies_to": list(CLOSERS), "closer_n_state": NS + N_GEO}}
    periods0 = tuple(dict.fromkeys(("TRAIN", B["early_stop"], B["calibration"], B["selection"], DEC)))
    x0 = round0_exit_tables(cx, periods0, ctx["prior"], SCFG, cells, seed=0)
    res["c3"]["round0_exit_visits"] = {p: int(len(x0[p]["row"])) for p in x0}
    mr = MR.MultiResLabels(cx, MR.Thresholds(), MR.LADDER, ctx["EXPERT"])
    nopl = ACT.NopLabels(mr)
    rows_all = np.flatnonzero(cx.obs_ok)
    d = {"contract": AC, "fit_module": ctx["fit_module"], "SCFG": SCFG, "NS": NS, "rows_all": rows_all, "cal_rows": cal_rows, "view_rows": view_rows,
         "et": et, "x0": x0, "nopl": nopl, "rounds": ctx["ROUNDS"], "led": led, "ledger": ledger, "blocks": B,
         "fitness": a.strategy_fitness or AC["strategy"]["fitness"]["default"], "sig": np.asarray(ds.sigma, float), "cells": cells, "e_time": e_time, "cost": cost,
         "caps": caps}
    res["action_contract"]["primary_fitness"] = d["fitness"]
    named = {}
    for p in (B["calibration"], DEC):
        go = {}
        Dp, rp = sets(cx, et, x0[p], p, ACT.NopLabels(mr), NS, cells, geo_out=go)
        named[p] = {m: (rp[m], Dp[m][1], Dp[m][2] if len(Dp[m]) > 2 else None, go.get(m)) for m in SPECIALISTS}
    C2.save_sets(out / "C3_SETS.npz", named)
    D_ib = C2.sets_from_saved(cx, {m: {"rows": r, "y": y, **({"R": R} if R is not None else {}), **({"G": g} if g is not None else {})}
                                   for m, (r, y, R, g) in named[DEC].items()}, NS)
    frozen = {}
    for seed in SEEDS:
        pols, infos, extra, c, table = run_seed(cx, seed, d)
        frozen[seed] = c
        RL.record(ledger_path, f"{fold}/{DEC}", out.name, f"seed {seed} heads", "FULL", "e3m_c3", note="reported-only head metrics on the decision block")
        extra["reported_only_decision_block"] = {"block": DEC, "heads": head_report(pols, extra["calibration"], D_ib), "note": "frozen calibration and heads; nothing fitted or chosen here"}
        chosen = extra["strategy"]["chosen"]
        best = {"arbiter": "action_contract_c3", "rate": None, "tau": None, "strategy": extra["strategy"]["strategy"],
                "nop_veto_kept": extra["strategy"]["nop_veto_kept"], "sharpe": None if chosen is None else chosen["sharpe"],
                "lcb": None if chosen is None else chosen["lcb"], "valid": chosen is not None}
        rec = {"fits": {m: {k: infos[m][k] for k in ("epochs", "best_epoch", "params", "stop_reason")} for m in infos},
               "arbiter_trials": extra["strategy"]["runs"]["with_veto"]["front"] + extra["strategy"]["runs"]["without_veto"]["front"],
               "arbiter": best, "trading": {}, "musashi_contract": {}, "action_contract": extra}
        gate_stats = {}
        for p in DEV:
            if p == DEC:
                RL.record(ledger_path, f"{fold}/{DEC}", out.name, f"seed {seed} frozen policy", "FULL", "e3m_c3", note="single post-freeze read (sac + musashi contracts)")
            rec["trading"][p] = summary(cx.rollout(p, c(cx), SCFG)[0])
            gate_stats[p] = dict(c.last_gate.stats) if c.last_gate is not None else None
            rec["musashi_contract"][p] = summary(cx.rollout(p, c(cx), MCFG)[0])
        rec["action_contract"]["best_of_day_gate"] = gate_stats
        led_ = {}
        for p in DEV:
            if p == DEC:
                RL.record(ledger_path, f"{fold}/{DEC}", out.name, f"seed {seed} side ledger", "FULL", "e3m_c3", note="per-tick decision ledger replay")
            pl = PolicyC3(table, c.s, cells, e_time, cost, record=True)
            cx.rollout(p, pl(cx), SCFG)
            led_[p] = pl.ledger_summary()
        rec["action_contract"]["side_ledger"] = led_
        for f, bf in extra["strategy"]["by_fitness"].items():
            if bf["strategy"] is not None:
                cf = PolicyC3(table, strategy_from_dict(bf["strategy"]), cells, e_time, cost)
                bf["trading"] = {p: summary(cx.rollout(p, cf(cx), SCFG)[0]) for p in DEV if p != DEC}
            bu = bf["archive_view"]["best_unconstrained_by_fitness"]
            if bu is not None:
                cu = PolicyC3(table, strategy_from_dict(bu["strategy"]), cells, e_time, cost)
                bu["trading_not_selectable"] = {p: summary(cx.rollout(p, cu(cx), SCFG)[0]) for p in DEV if p != DEC}
        rec["action_contract"]["out_of_sample_activity"] = {f"{DEC}_trades_per_week": rec["trading"].get(DEC, {}).get("orders_per_week"),
                                                            "owner_rule": f"below 2 trades/week on {DEC} = INVALID", f"valid_{DEC}": rec["trading"].get(DEC, {}).get("valid")}
        ci_p = {p: ACT.period_activity(cx, p, c(cx), SCFG) for p in DEV}
        rec["action_contract"]["activity_ci90"] = ci_p
        for m, pol in pols.items():
            torch.save(pol.m.state_dict(), out / f"FULL_{m}_{seed}.pt")
            res["probe"][f"FULL_{m}_{seed}"] = ctx["probe"](pol)
        (out / "PROBE.json").write_text(json.dumps({"split_contract": a.split_contract, "agent_state": a.agent_state, "scaling": a.scaling, "probe": res["probe"]}, indent=1))
        res["trials"].setdefault("FULL", {})[str(seed)] = rec
        print(json.dumps({"variant": "FULL", "seed": seed, "arbiter": "action_contract_c3", "strategy": best["strategy"], "early_close_disabled": extra["early_close_disabled"],
                          "sharpe": {p: round(rec["trading"][p]["sharpe_daily_sqrt365"], 2) for p in rec["trading"]},
                          "opw": {p: round(rec["trading"][p]["orders_per_week"], 2) for p in rec["trading"]},
                          "seed_sanity": {p: ACT.seed_sanity(rec["trading"][p], ci_p.get(p)) for p in rec["trading"]}}), flush=True)
    sh = [res["trials"]["FULL"][str(s)]["trading"][DEC]["sharpe_daily_sqrt365"] for s in SEEDS]
    res["decision"] = {"means": {"FULL": {"decision_block": DEC, "seed_mean_sharpe": float(np.mean(sh)),
                                          "all_valid": all(res["trials"]["FULL"][str(s)]["trading"][DEC]["valid"] for s in SEEDS)}},
                       "chosen": "FULL", "rule": "C3: single FULL variant; decision between arms by the seed ENSEMBLE on the decision block (report)"}
    ledger(led, rung="E3M_C3", event="READ", periods=["OBS_2024"], purpose="single read per seed")
    for seed in SEEDS:
        c = frozen[seed]
        rec = res["trials"]["FULL"][str(seed)]
        RL.record(ledger_path, f"{fold}/OBS_2024", out.name, f"seed {seed} frozen policy", "FULL", "e3m_c3", note="single read")
        rec["trading"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), SCFG)[0])
        rec["musashi_contract"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), MCFG)[0])
    return res
