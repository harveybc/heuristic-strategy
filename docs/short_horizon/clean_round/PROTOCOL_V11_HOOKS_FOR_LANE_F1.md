# Protocol v11: hooks that lane F1 must wire in the files it owns (e3m / action_contract / c2)

Lane F2 did not edit `tools/expert_guided_e3m.py`, `action_contract.py`, `c2.py`, `outcomes.py`, `multires.py`. The
nested contracts (`data.WALK_FORWARD_NESTED`, keys `early_stop`, `calibration`, `selection`, `decision`) need:

1. E3-M early stopping must read `sc["early_stop"]` (INNER_ES) instead of the hard-coded INNER_A; Platt calibration and
   gene maps read `sc["calibration"]` (INNER_A); the NSGA-II search, fitness choice and arbiter trials read
   `sc["selection"]` (INNER_SEL); nothing reads `sc["decision"]` (INNER_B) before the freeze. Each use must be recorded
   in E3M.json under `blocks_used`.
2. Every evaluation of INNER_B (E3-M, seed ensemble, audits, reports) must call
   `read_ledger.record(<package>/READ_LEDGER.jsonl, f"{fold}/INNER_B", unit, policy, variant, who)`.
3. `freeze_strategy` / `C2.freeze` must return, per search, `stats.deflated_sharpe(...)` (n_trials = distinct archived
   points, T = calibration days) and `stats.pbo(...)` over the archived points' daily-return matrix on the selection
   block, under `strategy.search_overfit`.
4. The launcher refuses a nested-fold unit whose recipe binds the consolidated store (`fold_local_store_required`):
   `NESTED_FOLD_STORE_NOT_FOLD_LOCAL`. Rebuild job: `NESTED_FOLDS_STORE_REBUILD_SPEC.json` (predictor
   `tools/fold_local_producers.py`, one scope per fold, fit window = the fold's TRAIN).
