# Declared execution-layer variants (fix lane F3, 2026-10-26; audit register A4-1 / A4-2 / A4-4 / A4-7 / A4-9 / A4-10 / A4-12)

Module `app/short_horizon/expert_guided/exec_variants.py`; engine hooks in `env.py` (`EnvConfig.variant`, `None` =
audited behaviour, byte-identical; tests prove it). Every variant has an identity (sha256 of its declared options).

| variant | identity | options |
| --- | --- | --- |
| legacy (audited behaviour) | `exec_5d80473b21dc61a0` | `{'reverse_counts_as_one_entry': False, 'sizing_mode': 'notional_share', 'risk_pct': 0.01, 'entry_cutoff_hours_before_flat': None, 'short_trigger_on_ask': False}` |
| risk1pct | `exec_39cd10868a384c45` | sizing_mode risk, risk_pct 0.01 |
| risk1pct+reverse+cutoff24 | `exec_eb21e5f51af8102f` | + reverse_counts_as_one_entry, entry cutoff 24 h before the forced flat |
| risk1pct+ask_triggers | `exec_c0d66c1e7de43c1b` | + short stops/limits trigger on the ask (bid + spread) |
| full_f3 (all four) | `exec_709f9b1fa3512486` | all of the above |

Hooks for the other lanes (nothing in e3m / action_contract / c2 / outcomes was edited here):

| fix | where | how a contract uses it |
| --- | --- | --- |
| A4-2 best of day | `exec_variants.BestOfDayGate(act, observe_bars, threshold)` / `first_of_day(act)` | wrap a SCORED policy `act -> (action, tp, sl, score)`; causal secretary rule per business day; `gate.stats` reports first-vs-best diagnostics |
| A4-1 reversal = one entry decision | `ExecVariant(reverse_counts_as_one_entry=True)` → `env.step` | the slot of the same-day position being reversed is re-used; new entries still obey the cap; count `n_reversal_cap_waived` |
| A4-4 risk sizing | `ExecVariant(sizing_mode="risk", risk_pct)` → `exec_variants.risk_units` | units = risk% · balance / stop distance, capped by rel_volume · balance · leverage / price and the margin, integer, else refused (UNITS); SL and TP stay on every order |
| A4-9 Friday cutoff / calendar clock | `ExecVariant(entry_cutoff_hours_before_flat=24)`; `exec_variants.calendar_clock_mask(base_mode)` | refusal `VARIANT_CUTOFF`; the two session-clock slots of `env.STATE_NAMES` as a declared contract option (calendar, not feedback); engine modes and CLI unchanged |
| A4-7 early-close requirement | `exec_variants.early_close_ok(agent_close_share, value_lcb, min_share=0.05)` | the strategy selector calls it on held-out calibration rows: min share of agent closes AND early-close value lower CI ≥ 0 |
| A4-10 effective cap | `exec_variants.effective_cap(env, b)`, `clip_multiples`, `cap_profile` | bound heads and grids at the per-bar engine cap (2021-01→04: p50 2.41σ, p1 2.21, p99 2.54) |
| A4-12 ask/bid triggers | `ExecVariant(short_trigger_on_ask=True)` → `env.step` | short SL triggers at bid high ≥ SL − spread, TP at bid low ≤ TP − spread; fills at the levels (declared approximation) |

Engine check (`tools/exec_variants_check.py`, admitted 2021-01→2021-04, contract 1 entry/day, 1:50; `ENGINE_CHECK.{md,json}`):
perfect foresight profits under every variant and the coin flip loses ≈ its costs (5-seed mean net: legacy −151, risk −83,
risk+reverse+cutoff −120, risk+ask −69 USD). Best-of-day (observe 3) vs first-of-day on the same foresight: Sharpe 9.4 → 14.8,
max DD 4.6 % → 1.4 %, hour-0 (rollover) entries 21 % → 0 %, cost −39 %, trades 47 → 37 (still 2.6/wk). Risk sizing 1 %: same
Sharpe (9.4 → 9.5), max DD halves (4.57 → 2.24 %), absolute net lower (7 536 → 3 600 USD) because 1 % risk sizes below the
legacy fixed 5× notional exposure; the risk fraction, not the rule, sets the scale. The 24 h cutoff removes the Friday
TIME exits (7 → 3 / 6 → 0). TEST never read.
