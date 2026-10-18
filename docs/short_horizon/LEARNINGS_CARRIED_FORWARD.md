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
