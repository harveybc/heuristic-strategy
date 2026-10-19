# Learnings carried forward (S2)

What the previous programme taught, and where this lane applies it. Each item
names the old failure, then the rule and how it is enforced here.

1. **Leakage audit before scores.** Earlier sweeps used observation offsets as
   if they were hours (H144 was really 191–316 h) and one feature family carried
   a point-in-time leak. Here labels are computed from the block's own bars only.
   Tests show that perturbing any bar after a block leaves its label unchanged
   and that perturbing a bar inside it changes only that block; the decision
   price is the last close at or before the block start. The naive uses completed
   blocks only (tested). TEST rows (2025+) are refused by the loader, not just
   unused.

2. **Per-horizon naive, on identical rows.** The old strategy had no per-horizon
   naive at all, and a single MAE line at H144 was read as a profitability
   threshold. Here every cell reports its MAE next to the naive on the same rows
   (NAIVE_MEAN20 and NAIVE_PERSIST for high/low; persistence per horizon for the
   comparator), and every baseline runs on the same block population (tested).

3. **Equal MAE is not equal information.** The 2026-09-27 audit showed that
   truth-plus-noise at naive MAE still had about 75 % directional accuracy.
   This lane reproduces that trap: Gaussian noise stays profitable at 1.8× the
   naive MAE. The frontier is therefore stated on a model-like blend family, with
   the Gaussian family kept for comparison and labelled optimistic.

4. **Execution must be the strategy you claim.** The old defects: entry from the
   long family only, TP/SL as close-only signals, a default exit variant that
   contradicted itself three ways, friction clipped to the bar range, and swap
   subtracted from reported PnL but never debited. Here TP/SL are native intrabar
   orders, the open is honoured first, stop-first applies on ambiguous bars, a
   limit-fill bar never credits the TP, both directions come from one prediction
   set, there is no prediction-based early exit, and each cost is a cash debit
   charged once. The ledger must reconcile (asserted in every run) and a round
   trip is checked against a hand calculation.

5. **Business calendar as configuration.** Weekday-name rules (`weekday() == 4`)
   are gone. The calendar is a versioned JSON with explicit entry-delay,
   forced-flat and cutoff offsets. A bar outside a session is an error, nothing is
   ever filled, and the strategy modules contain no weekday vocabulary (tested).

6. **Labelled assumptions.** Costs are ASSUMED_NOT_BROKER_VERIFIED, with a ×0–×3
   sensitivity; the clock caveat and the calendar-derivation proxy are stated in
   the contract; every output row carries
   `DEVELOPMENT_ORACLE_NOISE_STUDY_NOT_A_MODEL_RESULT`.

7. **Sealed inputs and retained evidence.** The input digest is pinned and
   checked at run time. The full cell table is retained compressed with its
   digest, chunk receipts record CPU per chunk, and the generated tables come
   from the retained table, never hand-typed. Before a model's predictions are
   scored here they must be sealed (digest, model id, training window) before the
   backtest reads them; the adapter for that is the next step, not done here.

8. **Closure-table rule.** Any later model closure must carry the table: model
   error and its scale, naive error on the same rows, skill, a literature value
   with its source or NOT_COMPARABLE with a reason, and NO_NEW_MEASUREMENT when
   nothing was measured. This study supplies the naive columns (13.4/10.1 pips at
   6 h, 24.2/17.9 at 24 h) and the required-accuracy bands; it contains no model row.

9. **No tuning on results, no forced conclusions.** Parameters were frozen in the
   contract before the first score. The frontier is bracketed by the grid, not
   fitted; non-monotonic responses (the comparator under Gaussian noise) are
   reported, not smoothed. 2024 was looked at, so it is DEVELOPMENT, not a holdout.

10. **Bounded, owner-safe compute.** Every command ran under
    `crispdm-run -m 2G -t 120s`, CPU only, in chunks (16 chunks, 333 CPU-s), from
    an isolated worktree with outputs under scratch; the dirty main checkout was
    never touched.

## Added at the 2026-10-19 closure (see `ROUND_CLOSURE_2026_10_19.md`)

11. **Per-side requirement, not total range.** Up and down errors partly cancel in the range, so
    a range MAE at 0,85× MEAN20 coexists with 0,91–0,98× per side and a losing bracket. Quote and
    check Û and D̂ separately (`CALENDAR_RANGE_EXPERIMENT.md` §4).

12. **Free abstention and exact-extreme fills are artifacts.** Both are named, priced and tested
    (participation metrics; limit trade-through ≥ 1 pip; one-bar latency).

13. **Frequency rule inside the strategy.** ≥ 3 trades/week and ≤ 4 orders/day are enforced at
    decision time; a configuration failing them is not a result and is hollow in every figure.

14. **No centred or two-sided filters as inputs.** STL, centred moving averages and repainting
    indicators (ZigZag) are non-causal; they may define labels or hindsight experts, never features.

15. **Clocks are measured, never assumed.** The NFP POC assumed first-Friday 13:30 UTC on an
    unverified bar clock; here the bar clock is detected from the release jump
    (CLOCK_DETECTED_NOT_VERIFIED) and the calendar export's offsets are a measured per-period table,
    not a DST rule.

16. **Research-only data stays outside the repo.** The Investing.com fixture (ToS) is read in place;
    only hashes and aggregates are published; it never feeds anything live.

17. **The calendar is a risk filter.** Vol impulse real, trend only via surprise and too small/short
    at hourly resolution, causal gate empty → abstain or widen stops around releases.

**Branch state 2026-10-19: DEVELOPMENT; nothing trading-eligible.**

18. **A hindsight expert's edge is half timing, half extent.** Delaying ZigZag pivots by their median
    confirmation lag keeps Sharpe ≈ 8–9 while the causally confirmed version with a vol-based TP is
    negative: the next-pivot level (the extent of the leg) carries as much information as the pivot
    time. Demonstrations must be judged on both (`ZIGZAG_EXPERT_STUDY.md` §4). The learned detector
    run in that study is a SMOKE_PROBE_WEAK_DETECTOR (no lookback window, substitute features, one seed):
    a weak probe did not close the gap; a proper windowed model on the selected features is pending, and
    the probe is not evidence about the achievable ceiling.

## Added 2026-10-19 with the clock identity (plan items 14c/16; K13/K14 of `SATOSHI_WORK_PLAN.md`)

19. **Lake clock = New York-like with DST, but on the EU calendar.** The lake's naive labels match the
    admitted UTC root at +5 h in winter and +4 h in summer. That is "NY with DST" for most of the year. In
    the ~4 weeks a year between the US and EU transitions, though, the NY rule is off by one hour on every
    bar (1,953 rows, 0 exact), while `EU_DST_UTC_MINUS_5` (Europe/London wall − 5 h) is exact on all of them.
    The pinned identity is `cb565a65…` (mapping `b3f617ef…`, tzdata 2024b). It has 0 clock residuals, 8 source
    residuals, and OHLC-exact rates of 1.0 / 1.0 / 0.99984 / 0.99926 / 0.99952 for 2020–2024
    (`CLOCK_IDENTITY.md`). Rule: no study joins the lake to anything on the vendor label or with a hand
    shift; it goes through `bar_source` (`admitted` or `lake_mapped`). The legacy `lake_label` TEST boundary
    is on the label clock (4–5 h behind UTC); on this file it read no 2025 bar, by luck.

20. **The hindsight optimum is set by the frequency cap, not by the expert's parameters (K13).** NSGA-II
    v2 (pop 64 × 80 gen × 3 seeds, convergence by hypervolume) found these optima. Cap 1/day: D6/19.6/B3,
    Sharpe 13.7–14.5. Cap 2: D5/11.7/B3, Sharpe 15.8–17.2. Cap 4: D3/40.7/B0, Sharpe 17.7/18.2/18.2. Raising
    the cap moves the optimum more than any parameter does. A hindsight Sharpe is therefore quoted with its
    cap, never alone.

21. **Deviation is nearly irrelevant under MT semantics (K13).** Tied objective vectors collapse across wide
    Deviation ranges, and Backstep 0 is a structural floor. Depth carries the timing. Search effort on
    Deviation is wasted, and a "tuned Deviation" claim needs a tie-group report.

22. **Executable fill is b+1 (K13; K12 finding).** The earliest executable fill of a signal decided on bar
    b is the open of b+1. The selection adapter used to fill at b+2 (one bar of extra latency against k=2).
    The confirmation v1 overestimated the lag by one bar. b+1 vs b+2 changes Sharpe by ≤ 0.1 here, but the
    executable path is b+1 (additive) and is the one cited.

23. **Label ≠ fill.** A pivot label names the bar where the hindsight extreme occurred. The fill is at
    the earliest executable instant after the decision (bar start + 1 h availability, then b+1 open). Scoring
    a strategy at the label bar's price is an exact-extreme artifact (item 12). Labels and fills are stored
    as separate columns and are never collapsed.

24. **2024 is already observed.** 2024 was looked at repeatedly (validation sweeps, ZigZag v2 reruns).
    It is relabelled as previously observed DEVELOPMENT (@abbfaa6) and is no holdout for any later claim. Only
    2025+ (TEST, refused by every loader) is unseen.

25. **Selection v3: useful for classification, not for trading (K14).** TIMING_PROXY_SELECTION v3
    (`b4da1012…`, 6 variables, on the v2 expert D3/40.7/B0 with b+1 execution) was compared with the 15 on
    identical rows. PR-AUC was 0.38–0.40 vs 0.35–0.37 (base 0.20), and log-loss was significantly better in
    both neural arms. Trading nearly always loses: Sharpe −4.0…+0.8, m2 2024 significantly worse, DSR
    ≤ 0.057. A classification gain is not a strategy result. v3 also ran on the lake label clock, and
    2023H1 there is missing 666 bars that the admitted root has (+27.2 %). The rerun on the admitted source
    waits for Musashi's decision on v3.
