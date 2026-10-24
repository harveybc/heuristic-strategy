# T3 v5 pending-tail redistribution

Owner requested reuse of the idle 4070. No scientific parameters, data,
seeds, selected columns, transforms or manifest changed. The scientific
package remains dc8d4aac63ba973f67124c1e0968a123fb39bfa1, manifest
`t3v2_7db1953f99ad`. GPU class changes for the reassigned units and is
recorded; cross-device bitwise equivalence is not asserted.

## Execution

| Shard | Units | Disposition |
| --- | --- | --- |
| Initial 4070 | 01, 03, 05 | Complete, mirrored, original receipts retained |
| Original 5090 | 07 through 23, odd | Eight complete; native unit 23 preserved |
| Remaining 5090 | 25 through 35, odd | Six admitted; automatic start after unit 23 exits and authenticates |
| Additional 4070 | 37, 39, 41, 43 | Native CUDA fit 37 verified running |

The original dispatch manager alone is paused. Its scientific subprocess
is not signalled. Its timer is disabled, and a durable service condition
prevents a reboot from resurrecting the old assignment. The boundary
controller seals unit 23 under its original identity before stopping the
old manager and activating the six-unit successor. Original results are
not re-signed under a successor configuration. The old shard is marked
relocated, not falsely marked as having completed all 19 units.

The successor configurations are:

- 5090: `72127e463a36409b954f9dd5635ed1853de355d6f97388f0cfb3d3b5263c9c6b`.
- 4070: `af83f434aee2bc7deb41288aebe04f51f6f6689e9f509d4677458051921aa016`.

Both retain the 5 GiB per-fit admission. Collectors copy and authenticate
each result on the archive volume and the peer. The periodic coverage
index validates disjoint coverage of all 22 assigned odd units, manifest
identity, receipts and physical read-back counts. It publishes
`COVERAGE_REBALANCED.json` to the private peer delivery directory. The
private `INDEX_REBALANCED.private.json` lists every replica root and its
allowed units. Use these four roots, not the original assumption that all
19 results will appear under the original fast root. Do not duplicate any
of the 22 assigned units. The peer's 23-unit queue is untouched.

## Tests and limits

PRE: new redistribution tests failed collection before implementation.
POST: 78 focused CPU tests passed, including disjoint/full coverage,
rejecting an entered tail or changed cursor, preserving an active child,
refusing incomplete native output, idempotent harvest, and empty/duplicate
mirror rejection. Native 4070 process and preserved 5090 process verified.

The boundary handoff has not occurred at publication; its durable timer
is active and reports WAIT_NATIVE_FIT. That is not a completed handoff.
No final extractor selection, combined C, replay or scientific release is
claimed. An incomplete native result is retained and rejected, not hidden.

Estimated completion with this allocation: approximately 11:00-12:00
UTC-5, subject to remaining unit runtimes, followed by combination/replay.
