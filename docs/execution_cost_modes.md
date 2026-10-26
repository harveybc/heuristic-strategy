# Corrected execution and paired cost diagnostic

Status: candidate implementation; no financial training or feature selection.
Frozen historical engines and artifacts are not changed by this worktree.

## Contract and tests

| Requirement | Behavioral witness |
|---|---|
| AUD-01 rollover accrual | long debit, short credit, triple rollover; open equity and final ledger agree without double charge |
| AUD-02 cost clock | known close uses close instant, open uses open instant; intrabar costs explicitly use start-of-bar approximation |
| AUD-03 margin liquidation | both directions, opening gap and intrabar crossing, nearest adverse barrier versus SL; no same-bar reentry after liquidation |
| AUD-04 risk clock | close-marked equity grouped on close time, terminal cash unchanged |
| AUD-05 actual-fill constraints | reject too-close protections and oversize fills at actual open; never silently shift levels |
| Easy mode | zero spread, slippage, commissions and swap; same calendar, protections, units rules, margin and limit execution |

Two named cost modes: `configured` (default supplied profile) and `zero_cost`
(profile.scaled(0)). Neither changes input data, labels, model, reward formula,
calendar, delay, limit-through, SL/TP, leverage or volume limits. Fixed-action
cash differences may change later sizes because sizing uses current balance.
Therefore equal actions do not imply equal units throughout an episode.

Repeat old easy-mode hypotheses on the corrected engine, not old performance
numbers. Train fresh paired policies per seed/cost mode on matched support and
budget. Evaluate both policies under both modes (2 x 2). Label fixed-policy
cost sensitivity separately from the effect of training without costs.
Zero-cost subsets may be diagnostic candidate generators, never an irreversible
filter or a final cost-aware feature recommendation. Final pruning/reintroduction
and confirmation use configured costs; preserve all mandatory causal/profile/
encoding/extractor stages. TEST stays closed.

Margin convention: fixed margin from actual entry notional, account equity net
of entry charges and accrued financing, excluding hypothetical exit charges.
Opening protective/margin barriers execute before discretionary actions. At a
gap beyond both, SL wins the tie; both fill at open. Within an OHLC bar, the
nearest adverse barrier (SL or margin) wins before TP, because path is unknown.
An intrabar fill is dated at bar end for bookkeeping and costed at bar start,
an explicit approximation, not an observed tick timestamp. Known open and close
fills use their exact event time. Margin is checked again after close-time
financing accrual. One liquidation prevents reentry in that same bar.

Scope: correcting this bar engine does not certify other simulators, observed
bid/ask execution, upstream features or a native RL subset evaluator.
