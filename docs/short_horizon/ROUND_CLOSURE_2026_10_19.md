# Round closure — short-horizon strategy lane, 2026-10-18 → 2026-10-19

**State of the whole branch: DEVELOPMENT. Nothing on it is trading-eligible.** TEST rows (2025+) were
never read; 2024 was looked at, so it is DEVELOPMENT, not a holdout.

## 1. Scope

Branch `satoshi/short-horizon-strategy-20261018`, 32 commits from `036c521` (S0 contract) to `178c6c1`
(causal event study), tests first for every module (121 short-horizon tests, all passing), every run under
`crispdm-run -m 2G -t 120s` on CPU from an isolated worktree.

| Deliverable | Where |
|---|---|
| S0 contract, S1 oracle/noise harness and frontier, S2 learnings | `STRATEGY_CONTRACT.md`, `REQUIRED_ACCURACY_FRONTIER.md`, `LEARNINGS_CARRIED_FORWARD.md` |
| Validation 2024 noise degradation, price-path sweep, realism pass, frequency rule | `VALIDATION_NOISE_DEGRADATION.md`, `PRICE_PATH_NOISE_SWEEP.md` |
| Sealed-prediction adapter and cheap causal baselines | `PREDICTORS_BASELINES.md` |
| Direction-free V1/V2, integrated plan, SL/TP and sizing business rules | `VOLATILITY_STRATEGIES.md`, `INTEGRATED_STRATEGY_PLAN.md`, contract §14–§16 |
| Calendar / implied-vol range experiment (HAR, HAR-X, GBM) | `CALENDAR_RANGE_EXPERIMENT.md` |
| Causal event-window strategy on the research-only fixtures | `CAUSAL_EVENT_STRATEGY.md` |
| **Closure table (generated)** | `results/CLOSURE_TABLE_2026_10_19.md` (+ `.csv`), generator `tools/short_horizon_closure_table.py`, test `tests/unit_tests/test_short_horizon_closure_table.py` |

Sibling deliverables of the same round (pointers, not reproduced here):
- financial-data `satoshi/calendar-sources-20261019` @ `6d99bc44b` — economic-calendar datasets, automatable sources and the acquisition request (2020–2024 schedule, EVZ).
- causal-inference `satoshi/event-impact-20261019` @ `a4cb9ea` — causal event-impact estimator, EI01–EI14 acceptance tests, descriptive fixture run.
- predictor: matched MG1/MG2 campaign CLOSED, MG2_SELECTED; R2 contrast `i6dmo-9d3c198c77b1` running (release `c34b1e48`).

## 2. Closure table

Generated from the retained artifacts by `tools/short_horizon_closure_table.py` (never retyped): per
measured predictor or strategy, the error with its scale, the naive on the same rows, skill
= (err − naive)/naive, a literature value with its source where one exists, and a comparability
verdict; rows without a new measurement say NO_NEW_MEASUREMENT.

→ https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/CLOSURE_TABLE_2026_10_19.md

Headline rows (from the table):
- Range forecasts: HAR 0,855× / 0,833× MEAN20 at 24 h (skill −0,15 / −0,17), GBM 0,765× / 0,774× at 6 h (−0,24 / −0,23). COMPARABLE in kind with the 10–30 % HAR-RV gains of Corsi 2009 / ABDL 2003, NOT_COMPARABLE in scale (block excursions, MAE).
- Per-side excursions: 0,86–0,98× (skill −0,02 … −0,14).
- Direction (logistic first-extreme and surprise-sign): 0,43–0,56 — inside the literature's no-skill band.
- Event vol ratio: 3,9× (release bar vs other hours on NFP days), 1,36× vs the same hour a day earlier; COMPARABLE in kind with Ederington & Lee 1993 / ABDV 2003.
- V1 / V2 / RF ideal Sharpe: NO_NEW_MEASUREMENT this round (earlier round, pointer kept).

## 3. Verdicts

1. **Price-path and direction strategies are infeasible at achievable accuracy.** Persistence is the
   only path predictor scored (no skill; Meese–Rogoff); the real direction models sit at 0,43–0,56 against
   break-evens of 0,66–0,72 for the event rule and the frontier's first-touch requirement. Evidence:
   `PRICE_PATH_NOISE_SWEEP.md`, `PREDICTORS_BASELINES.md`, `CAUSAL_EVENT_STRATEGY.md` §3–4,
   `results/causal_event/BREAK_EVENS.json`.
2. **Volatility-only strategies.** V1 (OCO breakout) refuted: ruined/invalid at 24 h, Sharpe −2,1 / −3,7 at
   6 h even with ideal information. V2 (range reversion) marginal: +1,7 train / +0,04 validation at 6 h with
   ideal information. Range-first needs per-side errors ≤ ~0,67× (24 h) and ~0,53–0,6× (6 h) MEAN20 while the
   achievable per-side errors are 0,86–0,98×; the real HAR/GBM markers sit 1,5–1,8 Sharpe below the A1r
   curve at the same range MAE. Evidence: `results/volatility_strategies/break_even_v1_v2.json`,
   `VOLATILITY_STRATEGIES.md`, `CALENDAR_RANGE_EXPERIMENT.md` §4 and figures `calendario_rango_B24/B6.png`.
3. **Calendar.** The volatility impulse is real (release bar 3,3–3,9×; per-event half-life 0,3–0,8 h in the
   family means of 2,8–5,4 bars… read: the first-hour excess decays within a few bars); the trend exists only
   through the surprise and is too small and too short at hourly resolution (fitted drifts 1–8 pips per unit
   surprise, below the round-trip cost); the selected-event set under the causal gate is **empty** (0 of 26
   families at p < 0,05 on 2011–2017). **Calendar = risk filter**, not a signal. Evidence:
   `CAUSAL_EVENT_STRATEGY.md`, `results/causal_event/families_decision.csv`,
   `results/calendar_range/event_diagnostic.csv`, causal-inference @ `a4cb9ea`.
4. **Methodological learnings** (each enforced in code or doc): additive Gaussian noise around the truth is
   optimistic (75 % direction at naive MAE; blends A1r/A3r are the quotable family); free abstention and
   exact-extreme fill were artifacts and are now named and priced; the frequency rule (≥ 3/week, ≤ 4/day)
   is enforced inside the strategy and invalid points are hollow; the total-range error cancels per-side
   errors, so the requirement is per side; STL and centred filters are forbidden as inputs (non-causal);
   the NFP POC clock error (first-Friday/13:30 UTC assumption vs the detected label hour and the
   exporter's non-DST offsets); the Investing.com fixture is research-only (ToS), aggregates and hashes only.
   Evidence: `LEARNINGS_CARRIED_FORWARD.md` items 3, 11–16.

## 4. What would change each verdict

| Verdict | Would change with |
|---|---|
| 1 (direction) | minute data at the release (LATENCY_0 bound is +2 Sharpe), a point-in-time consensus, a direction model that reaches ≥ 0,66 on 2023 before 2024 is touched |
| 2 (volatility) | per-side excursion models at ≤ 0,67× / 0,6× on 2023; broker-verified costs (the margin is one spread) |
| 3 (calendar) | vintage-safe 2020–2024 calendar and EVZ in the lake (financial-data @ 6d99bc44b request), minute bars around releases |
| all | repaired vendor bar clock; broker-verified spread, swap and minimum size; a second market path (another pair or another year) |

## 5. Limits
- One instrument, one market path per year; 2024 and the fixture periods are DEVELOPMENT.
- Costs ASSUMED_NOT_BROKER_VERIFIED; clock semantics under repair; fixture calendar retrospective.
- Literature magnitudes in the table are approximate recalled values with sources, not reproduced.
