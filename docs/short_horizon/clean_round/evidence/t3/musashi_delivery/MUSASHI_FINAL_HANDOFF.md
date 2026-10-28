# T3 completed: final handoff to Satoshi

Observed 2026-10-23, 21:32 UTC. This closes the assigned T3 experiments and
combination/replay, not the complete upstream audit or a native SAC experiment.

## Completed evidence

- All 43 original T3 units are complete and authenticated: 21 own, 22 peer.
- The native frozen decision changes three groups: `cot.eur` to `per_feature`,
  `gmm/logret_1+rv_24+log_range` and `fracdiff/log_close` to `pointwise`.
  The other 22 groups retain their shared causal Conv1D extractor.
- Unit C combines exactly those changes. It completed on its first attempt;
  the frozen confirmation decision is `COMBINATION_SELECTED`.
- The 43 original units reproduce their saved probes exactly: 516 checkpoints.
  C adds 12 exactly replayed checkpoints. These checks use the original CPU and
  launcher thread environment; cross-CPU bitwise portability is not claimed.
- Control and C each reproduce all nine complete trading metric dictionaries
  exactly (three seeds times TRAIN, INNER_A, OBS_2024), without fitting or
  reselecting the saved arbiter/threshold. TEST was not scored.
- The C mirror and independent replica are hash verified. Both final reports
  were delivered automatically and rechecked against the local report hashes.

Evidence in the existing shared delivery folder: `NATIVE_REPLAY_ALL43.json`,
`CONTROL_FULL_REPLAY.json`, `COMBINATION_DECISION.json`,
`COMBINATION_REPLAY.json`, and the earlier readiness/coverage reports.

| Artifact | SHA256 |
| --- | --- |
| COMBINATION_DECISION.json | 86d771f12918158230804154cd692c66fd27e7e429ecc884b34a6a0fc887ce39 |
| COMBINATION_REPLAY.json | 16a5ad8eec879629fd686560e756079a07c7bc185ea13287eebf5a5900a0da50 |

The original science commit is `1aaf367923fae467988b1ca87d5aa3c74321141f`.
C producer config: `7ffebafb8253936ed6100b6ca4dd9aff53243b7a763189b39320777f7b4257d1`.
Groups: `a9d878cf8c63a5b0ecf13f3e3a664e020ae70cc9cf2fa2d252f1643b227f76ef`.

## Results, not only the selection statistic

Arithmetic means of three seeds; not a pooled or compounded account. Sharpe is
the native `sharpe_daily_sqrt365`; drawdown is the mean of per-seed maxima.
Profit is net USD under the recorded cost model, not broker-verified profit.

| Arm | Period | Sharpe | Net USD | Trades | Wins % | Max DD % | Valid seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | TRAIN | 1.9747 | 30682.15 | 665.00 | 50.79 | 23.38 | 3/3 |
| Combined | TRAIN | 2.6863 | 40217.10 | 575.67 | 48.52 | 12.98 | 3/3 |
| Control | INNER_A | 0.7800 | 701.62 | 89.00 | 47.69 | 9.14 | 3/3 |
| Combined | INNER_A | 1.5583 | 921.80 | 80.00 | 48.08 | 7.62 | 3/3 |
| Control | OBS_2024 | -0.7601 | -743.58 | 193.33 | 41.82 | 19.04 | 3/3 |
| Combined | OBS_2024 | -0.1247 | -341.52 | 150.33 | 41.06 | 17.21 | 2/3 |

Combined 2024 by seed:

| Seed | Sharpe | Net USD | Trades | Orders/week | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| 20261019 | 1.7808 | 1587.64 | 80 | 1.5094 | No: below minimum period-average activity |
| 20261020 | -1.2992 | -1678.42 | 226 | 4.2642 | Yes |
| 20261021 | -0.8558 | -933.78 | 145 | 2.7358 | Yes |

Do not omit the invalid seed or present its gain as an admissible trading
winner. The frozen selection reads TRAIN and INNER_A, not OBS_2024: its
`COMBINATION_SELECTED` result remains intact, separately from 2024 suitability.
The TRAIN-minus-INNER Sharpe gap is 1.1948 control versus 1.1279 combined.
No improved generalization on all periods is claimed. OBS_2024 is the main
VALIDATION/development year, not an untouched holdout; TEST remains closed.

## Instructions to Satoshi

1. Reuse the authenticated 43 units and C. Do not duplicate these fits, replace
   seeds, adjust the activity rule, or tune thresholds on 2024 to repair it.
2. Assemble the final audit request now: inventory and coverage, availability
   and causal transforms, TRAIN-only fitted decisions, selected columns and
   encodings, T1 numeric profiles, redundancy across groups, T2 regularization,
   T3 paired support and actual tensor consumption, C interactions, both replay
   scopes, and every period/seed's trading metrics and validity. Link existing
   evidence instead of manufacturing a new acceptance declaration.
3. Export one identified consumer recipe shared by both lanes: ordered columns,
   store/list digests, transform/scaler scope and constants, groups and extractor
   choices, calendar/business rules, split definitions and runtime identities.
   Preserve strict116, window13/stride2, fixed TRAIN scaling and no agent feedback.
   Existing labels are ZigZag, not the newer opportunity labels. A label change
   is a separate experiment; hindsight demonstrations never enter observations.
4. Keep responsibilities separate: Satoshi handles expert-guided learning;
   Musashi handles SAC without expert input. E3-M representation selection is
   a proxy, not evidence that SAC already benefits. The next SAC contrast is
   selected inputs versus their temporal representation, with identical data,
   business enforcement, budget and seeds; no rolling-z or silent clipping.
5. State actual publication custody: these reports and verified physical
   copies are complete. They do not certify OLAP ingestion/read-back. Include
   the existing governed outbox/warehouse receipt if available; otherwise mark
   that delivery unverified and use the existing publisher, not a new database.

No additional fit or owner approval is needed to gather these audit documents.
Final audit, shared consumer integration and the independent SAC comparison
remain subsequent work. There is no global RELEASE in this handoff.
