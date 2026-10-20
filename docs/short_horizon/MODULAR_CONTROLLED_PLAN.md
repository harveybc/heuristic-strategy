# Controlled modular specialists, 2026-10-20

Owner-authorized sequential GPU experiment. Expert imitation (BC + DAgger),
not a new SAC learner. The prior E3M result remains immutable. This initiative
tests specialization, abstention and aggregate capacity before another context
or feature search. Public files contain roles only; deployment paths are private.

## Frozen experiment

- Inputs: admitted UTC bar-start EURUSD, 15 selected features with pinned CSVs,
  13 observed-bar slots at stride 2, current completed bar included, past-only
  rolling normalization. Missing closed-market hours never become observations.
- Preserve E3M partitions for a controlled follow-up: gradient TRAIN 2020-2022;
  2023 H1 checkpoint/threshold selection; 2023 H2 diagnostic; full 2024 development
  evaluation after configuration freeze. This is NOT the four-year SAC fit.
  The four-year SAC recipe remains separately recorded. No 2025 TEST access.
- Business: one entry per business day, one position, leverage 50, capital
  USD 10000, rel_volume .1, order fraction 1, mandatory TP/SL, minimum units 1,
  entry delay 12 hours, forced flat 12 hours before session close. Positioned
  policies may hold or close only; no same-tick reversal. No minimum trading
  frequency forces selection or trades; zero activity is explicitly reported.
- Reuse the existing costed bar simulator: spread, slippage, commission and
  financing all enabled. Costs are assumed, not broker verified, and differ
  from the SAC accounting recipe. No cross-protocol ranking against SAC.
- Three fits per seed: JOINT16 (610 parameters), JOINT_MATCHED (nearest hidden
  width to FULL total, computed before outcomes), FULL (4 modules, 2236 total).
  All share causal temporal branch/core widths 2, core depth 1, head16 except
  JOINT_MATCHED's head. Dropout .1, AdamW weight decay 1e-4, LR .001, batch256.
- BC + three DAgger rounds; final round fixed. Each module refits from scratch
  as in E3M. Checkpoint monitor .5*(TRAIN loss + 2023H1 loss), inference mode,
  patience10, absolute min_delta1e-8, maximum1000 epochs, best restored.
- Same seeds 20261019/20/21 then 20261022/23. All 15 cells run, including poor
  results. Seeds are paired comparisons, not independent market histories.
- Common probability arbiter for all architectures. Threshold grid .02/.03/
  .04/.06/.08/.12/.15/.18 (entry score quantiles) selected on 2023H1 only.
  Calibration uses actual flat eligible states with their business state intact.
  Ties: lower turnover, then higher threshold. No gain/risk arbiter search.
- Causal inference eligibility never depends on retrospective target labels.
  Retrospective labels used only for training/monitor; label support is purged
  at the period boundary, including the positioned exit target.

## Controls without additional fits

Each JOINT gets an argmax control using the exact same fitted weights. FULL
gets: fixed protection exits, long-only, short-only, and the combined policy.
These use the frozen combined-policy threshold. A separate fixed-exit threshold
selected on 2023H1 is also reported, to distinguish threshold mismatch from
closer benefit. Position trajectories can differ; these are system ablations,
not claims of identical realized trades. No hindsight timing in inference.

## Results and decision

Keep per module/per round weights, training curves, seeds, exact configuration,
data/code hashes, calibrated threshold, per-period trade and equity ledgers.
Report profit USD and %, annualized daily Sharpe sqrt365, drawdown, entries,
closed trades, win fraction, profit factor, costs, exposure, buy/sell breakdown,
and annualized daily-log growth for comparable train/development gaps.
No-trade Sharpe is undefined (null), equity and profit are still measured.
Aggregate paired differences and all five seed results; no best-seed selection.
The three original seeds and two new seeds must remain distinguishable.

No promotion/deployment of a trading policy is implied. Future studies, in order:
context/window and stride contrasts (fixed elapsed span versus fixed slot count
separated), feature diagnostics and then candidate-specific regularization or
budget experiments. Those designs are recorded here but not automatically run
by this finite campaign. The current authorized sequence ends after 15 fits,
control evaluations, publication and an independently verified artifact copy.

## Automation and verification

One heavy GPU process group; assigned GPU must be the 5090. systemd user timer
and flock own sequential progress, including after reboot. Each cell has two
maximum fresh-root attempts; no partial fit is falsely resumed as full state.
Failure logs remain and other cells advance; integrity mismatches stop with a
typed state. Resource shortage waits without consuming fit attempts. Scientific
completion, publication completion and independent-copy completion are distinct.
Status/ETA are file-based; no agent polling. Warehouse failure never retrains.

Acceptance trace: MC01 routing/masks; MC02 causal temporal prefix/current t;
MC03 identical business and next-open fills; MC04 fit/monitor/2024 separation;
MC05 capacity matching; MC06 threshold uses only selection period; MC07 weights
reload parity; MC08 bounded restart/identity/lock; MC09 metrics/trades/direction
and zero-activity handling; MC10 publication and independent artifact readback.
Tests are designed before implementation, using synthetic routing, future
perturbations, loss monitoring, trade accounting and fake-worker recovery.
