# Forward paper test of an LLM direction call (option 2, owner decision 2026-10-21) — design, not running

Why forward: any evaluation of an LLM / foundation model on 2020–2025 is contaminated by pretraining memorisation
(Sarkar & Vafa 2024; Glasserman & Lin 2023; predictor `docs/timing_selection/LITERATURE_DATA_PHASE_2026_10_21.md`
§6). Only decisions made BEFORE the outcome exists, after the model's training cutoff, are clean.

## Protocol (pre-registered)

1. **Decision instants.** Every completed EURUSD H1 bar inside the business calendar (no entry in the first 12 h of a
   session or the last 12 h before the forced flat), plus every scheduled USD/EUR high-impact release + 1 bar.
2. **Inputs at the instant (nothing later):** last 72 H1 bars summarised as numbers (returns at 1/4/24/72 h, causal
   σ24, range, Kalman slopes fast/mid/slow, activity probability P(hit) from the confirmed activity model),
   the session and fix clock, the scheduled calendar for the next 24 h, and the latest releases with actual,
   previous and — when available — the pre-release consensus (FXMacroData point-in-time rows only).
3. **Question (fixed text, versioned by sha256):** "Will a long order filled at the next open with TP = +2σ and
   SL = −2σ reach TP before SL within 72 bars? Answer with a probability in [0, 1]." Same barriers as the lane's
   label, so scoring is mechanical.
4. **Models:** local `llama3.2:3b` (ollama, available now on the coordinator) as the control model; Laya through
   M5PHET when its owner allocates the GPU; the constant prior (0.5 for direction-given-hit, the TRAIN rate for
   hit) and the frozen FULL specialists as baselines on the same instants.
5. **Record before outcome:** each call writes {instant, input sha256, prompt sha256, model id + digest, raw answer,
   parsed probability, receipt clock} append-only; the outcome is joined only after the barrier label is final
   (support ledger as in `timing_profit_targets.barrier_outcomes`).
6. **Scoring:** log-loss and Brier vs the prior, calibration, and the paper strategy (lane contract: SL/TP, integer
   units, ≥ 3 orders/week, ≤ 4/day) with Sharpe, drawdown and orders/week; Diebold–Mariano on log-loss vs prior.
7. **Duration / power:** detecting an up-rate of 0.55 vs 0.50 among barrier hits (α 0.05, power 0.8) needs ≈ 780
   resolved hits; at ≈ 37 hits per week that is ≈ 21 weeks. Interim looks are reported but do not stop the test.

## Blockers (both need the governed data path; I do not handle credentials)

- **Live EURUSD H1 bars:** no live price feed is readable on the coordinator (the admitted source ends 2025-12-31).
  Needs a read-only quote/bar feed (e.g. the broker practice stream already configured for the agents) exposed
  as a file or API without exposing the credential.
- **Live calendar with consensus:** FXMacroData pulls run with its key; the new bounded worker
  (financial-data `satoshi/fxmacrodata-history-20261021`, `stage16_fxmacrodata_history_worker.py`) does history
  (2020–2024, consensus vintages) and can be scheduled forward by the same governed unit.

Until both exist, the harness can be built and replay-tested on 2024 ONLY as a plumbing check (contaminated, never
reported as a result).
