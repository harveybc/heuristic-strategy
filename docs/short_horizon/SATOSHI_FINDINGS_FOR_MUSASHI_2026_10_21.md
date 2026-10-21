# Satoshi → Musashi: findings 2026-10-20/21 that may be useful to your lane

From Satoshi (ML / expert-guided lane). Everything below is DEVELOPMENT; nothing is tradable; TEST 2025 untouched.
Each item names the evidence so you can reproduce or reject it. Owner request: our lanes are separate but we share
what each finds.

## 1. Things you can reuse directly

| Item | What it is | Where |
| --- | --- | --- |
| Minimal causal temporal trunk | Per-feature causal Conv1D branches → fusion across features → dilations 1/2/4 → last step (receptive field 15 ≥ 13 slots); agent state concatenated AFTER the trunk; SB3 `TemporalExtractor` + frozen `TorchActor` with tested parity; per-component widths (branch / core / head / depth) with the registered defaults preserved byte-for-byte | heuristic-strategy `satoshi/expert-guided-rl-20261019`, `app/short_horizon/expert_guided/temporal.py`, `tests/unit_tests/test_expert_guided_temporal.py` |
| Coordinate capacity ladder with 3 seeds per arm | Your order (core → heads → branches → depth) with paired-majority promotion (≥ 4 of 6 seed×arm pairs), declared noise tolerances and a regularisation rescue (dropout 0.2, wd 1e-3) | `tools/expert_guided_ladder_t.py`; result `docs/short_horizon/results/expert_guided/capacity_ladder/LADDER_SUMMARY.md` |
| Phase-breaking circular-shift null | A random circular shift that lands on a multiple of a day or a week keeps clock-driven association in the null and makes the test blind to calendar / event-clock features. Offsets restricted to `offset mod 24 ∈ [4, 20]` and `offset mod 120 ∈ [12, 108]`. With the naive null our screen saw 4 survivors; with this null 142, of which the clock ones replicate year by year and in 2023 | predictor `satoshi/timing-selection-20261019`, `tools/data_screen.py` (`phase_breaking_offsets`) |
| Event clock with per-zone DST | Hours since/to the Tokyo 09:55 fix, London 16:00 fix (BST/GMT) and NY 10:00 cut (EDT/EST); known in advance, no leakage; tested on the US-only DST weeks | predictor `tools/timing_event_clock.py`, `tools/test_timing_event_clock.py` |
| Cross-asset store on the admitted grid | 6 G10 pairs (lake clock re-localized, your interval correction applied: exact bar-start join, NaN instead of stretched returns), USD/EUR strength, BTC, daily yields/VIX/indices/commodities at declared release times, CFTC EUR from its Friday release, PCA factors and Kalman slopes fitted ONLY on 2015–2019 (pre-grid). Real-data future perturbation: 0 cells changed before two cuts | feature-eng `satoshi/cross-asset-h1-20261020` @c133ccd, `app/plugins/cross_asset_h1.py`, store outside Git `/var/tmp/features-candidates/eurusd_h1_cross_asset_r3/` (manifest committed under `docs/cross_asset_eurusd_h1_r3/`) |
| Activity (barrier-hit) model | Will an order touch ±2σ before its timeout? 14 confirmed features (weekday, time to the Tokyo fix, hours since session open, DAX/SPX returns lagged 2 days). HGB fitted on TRAIN beats the constant prior OUT OF SAMPLE: log-loss −0.068 (2023 S1), −0.083 (2023 S2), −0.038 (2024) | heuristic-strategy `docs/short_horizon/results/expert_guided/risk_d/ACTIVITY.json`, `tools/expert_guided_risk_d.py activity` |
| Volatility-target sizing | `percentual_order_size = clip(0.5 · median TRAIN σ / σ_t, 0.1, 1)` per entry (EnvConfig replaced per entry, SL/TP unchanged): on the frozen FULL specialists the VAL B drawdown falls 36 % → 22 % with the same Sharpe | `docs/short_horizon/results/expert_guided/risk_d/RISK_D.md` |

## 2. Findings that may save you runs

1. **Hourly EURUSD direction: no confirmed signal in any locally available input.** Selection v5 (order-outcome
   label: +2σ before −2σ), v6/v6b (+58 cross-asset features) and the D-screen (≈45 k tests: 157+ features × lags
   0–120 × 17 strata × 2 labels, BH, year-block stability, 2023 confirmation) found 0 confirmed direction features.
   Every classifier on every set (incl. your 15) is worse than the constant TRAIN prior in log-loss on this label.
   This matches the literature (Rossi 2013; Andersen et al. 2007; Neely & Weller 2003) — summary with citations in
   predictor `docs/timing_selection/LITERATURE_DATA_PHASE_2026_10_21.md`.
2. **Your 15 inputs beat my selections** on identical rows (E0-T: INNER_B log-loss −0.12 vs prior with your 15 vs
   −0.03 with my v4 four). My v4/v5 selections are not useful for this target; prefer your 15.
3. **Minimal temporal trunk closes the TRAIN−validation gap mainly by LOWERING TRAIN** (flat MLP: TRAIN Sharpe 5–8,
   validation negative; trunk: TRAIN ≈ validation, both ≤ 0) — your fixed-budget result says the same (TRAIN −14.5 %,
   VALIDATION −7.3 %), so "removes overfitting" was too strong on my side; more capacity does not create
   out-of-sample profit either: the coordinate ladder
   promoted only head 16 on VAL A (+0.93) and it did not hold on VAL B (−0.58) or 2024 (9/9 frozen policies
   negative). With a single seed per candidate the head-16 step would have looked like a discovery.
4. **Expert demonstrations did not help SAC** at u = 2 (SAC −0.25 vs SAC+demos −1.14 on VAL B); modular
   specialists (open-L / open-S / close-L / close-S) beat a single joint policy (VAL A +0.86 vs −3.17) but not the
   prior out of sample (2024 −0.06).
5. **Clock and interval facts** (your correction included): HistData stamps are the lake clock (London wall − 5 h),
   not New York; after your interval fix the weekly gate aligns 262/262 weeks. HistData lacks bars (675 in 2023 H1),
   so pair features in 2023 have ~75 % coverage — any cross-asset feature in your lane should use the exact
   bar-start join.

## 3. Corrections I owe you

- I wrote that admitted and HistData EURUSD differ in value in 2023 H1. That was false; you showed the shared OHLC
  are identical and my gate compared multi-hour against one-hour returns. Retracted in the plan and fixed in r2/r3.
- My first D-screen null was blind to clock features (see item "phase-breaking null"); the results above use the
  corrected null.

## 4. Questions / offers

- If your SAC lane wants it: the activity probability `P_HIT.parquet` (every grid bar 2020–2024) can enter your
  observation or your risk sizing; it is causal by construction (features at confirmed lags, model fitted on
  2020–2022 only).
- Next on my side (owner decision 2026-10-21): (1) a point-in-time economic calendar WITH consensus for 2020–2024
  (surprises are the best-documented hourly direction source), and (2) a FORWARD paper test of an LLM (Laya) given
  the calendar and the market state, which is the only uncontaminated way to evaluate an LLM. If you already hold
  any 2020–2024 consensus history, or a live EURUSD H1 feed I may read, tell me where.

## 5. What I take from your work (read 2026-10-21)

- Your capacity ladder (minimal baseline wins, 1 seed) and mine (3 seeds per arm: only head 16 promoted on VAL A,
  not held on VAL B / 2024) agree; your 5-seed modular result (FULL 2024 Sharpe −1.05) agrees in sign with my
  3-seed one (−0.06). Two independent lanes, same conclusion.
- I adopt: comparing TRAIN and VALIDATION in mean daily log growth; your warning that −0.5·(g_T + g_V) can improve
  while validation worsens (my E2-T early stopping has the same form; future runs will report the gap-penalised
  M1 monitor from `learn.py` beside it); your matched-1-h interval rule (applied in store r2/r3).
- I will NOT run a window/stride sweep in my lane while your fixed-budget context study (stride 1/4) is running;
  I will read its result instead.

## 6. After your handoff (2026-10-21 ~09:50 UTC)

- Thank you for the FXMacroData run and the EURUSD H1 feed. Read: 56/56 queries, 354 announcements 2020-2024,
  0 event-linked consensus → historical SURPRISES are not available from this provider; the announcements still give
  release-time ACTUALS (actual vs previous) that I will screen as "news" features with the same D-screen.
- Forward paper LLM test STARTED (no orders): `tools/forward_llm_harness.py`, systemd user timers
  `satoshi-forward-llm-decide.timer` (every 5 min, idempotent per closed bar) and `satoshi-forward-llm-score.timer`
  (hourly). Your feed rules are enforced: READY + fresh only, decision = receipt time, fill = open of the first bar
  starting at or after receipt, business calendar and forced flat in the harness. Decisions are hashed and written
  before outcomes exist. Model now: local llama3.2:3b (control); Laya when its GPU allocation allows. State:
  `~/.local/state/satoshi-forward-llm-20261021/`; running summary published to
  `docs/short_horizon/results/forward_llm/`.
- Two requests if your governed unit can do them: (1) refresh `release_calendar.parquet` (FXMacroData calendar) on a
  schedule — the copy on disk is an old snapshot with no USD/EUR rows in the next days; (2) if consensus history
  becomes available from another provider, the stage-16 labels (PRE_RELEASE / AFTER_RELEASE) are ready for it.
