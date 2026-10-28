# T3 v5: disjoint parallel deployment

Start checked 2026-10-24 around 00:09 America/Bogota (05:09 UTC).
This is execution status, not final scientific acceptance or a new SAC result.

## Health and allocation

Before launch the 5090 worker had approximately 11.2 GiB available host RAM,
zero swap use, zero memory PSI and no GPU training process. No reboot was
needed. The 4070 worker had approximately 22.9 GiB available and no compute
process on its GPU. Both disks had sufficient margin; the archive volume had
673 GiB free. A second GPU on the 5090 host is intentionally not loaded because
it shares that host's 14.3 GiB RAM.

| Worker | Assigned ids | Units | Initial native CUDA job | Host-memory cap |
| --- | --- | ---: | --- | ---: |
| 5090 | 7,9,...,43 | 19 | t3_unit_07 | 5 GiB |
| 4070 | 1,3,5 | 3 | t3_unit_01 | 5 GiB |

The ratio uses the owner's approximate relative speed, not a new benchmark.
The assignments partition all 22 odd ids exactly and never overlap the peer's
23 even/control units. No currently running peer process was stopped.
After start the 5090 worker still had about 9 GiB available, no swap and PSI=0.

## Scientific identity and verification

- Manifest `t3v2_7db1953f99ad`, SHA256
  `35636427e5f9fbfaf11010f8dd8bdfb8aee6e2866e49ad680b82f2cbc3a7716f`.
- Peer package inventory SHA256
  `c20ddcd7e04da97a955861d9fec76ff5e66d44a4fb40d6d75ba2c25f426efaaa`.
- Scientific code `heuristic-strategy dc8d4aac63ba973f67124c1e0968a123fb39bfa1`.
  Python sources in the delivered package match an archive exported from this
  commit; no model source was edited.
- Consumer store `cd0eb8d2...`, strict list 121 columns `80219f7c...`;
  fixed TRAIN scaling, no agent feedback, window 13/2, model 2/2/16,
  strong regularization dropout 0.3 / wd 0.01, original seeds and business rules.
- Effective runtime versions, manifest and assignment agree across workers.
  Commands are taken byte-for-byte from the peer's typed units, apart from
  resolving the local root. Hardware differs; no cross-hardware bitwise result
  equivalence is asserted.
- Fast configuration identity:
  `a28ae667a6008aafdf6885a1c1786cf2e077c7300891a355c2dfe3fbc9d484d6`.
- Slow configuration identity:
  `a69198efdc03a75af56e68770dd907f9699755d9d81da4f4a371c4dd4bbb3a49`.

The existing worker protocol retains its internal `v4_t3` dispatcher name;
the new manifest and provenance identify selection v5. The dispatcher now
recognizes these two explicitly pinned scientific contracts and rejects old
regularization under the v5 identity. The old v4 queue is not relaunched.
Operational code: predictor `337b7fdb`. PRE: two new v5 tests failed. POST:
56 focused CPU tests passed, including old contracts and the new partition.
The authentic package passed full pin and archive/source checks on deployment.

## Automation and delivery

Both queues use the existing sequential worker, locks, receipt validation,
bounded recovery and persistent boot timers. Each training process acquires
one memory reservation; no outer duplicate reservation was introduced.

Two collectors copy outputs to the archive volume, validate sealed receipts
and replicate to the peer worker with read-back. Initial collector executions
passed for both empty-result snapshots. They do not certify future results in
advance. Timers stop at terminal closure. OLAP publication is not claimed here.

For Satoshi: consume the fast and slow replicas described in the private
delivery index, retaining each receipt's original configuration. Do not run
our odd ids. Combine only after all 22 plus your 23 are authenticated and
paired; then apply the frozen per-group rule, confirm C and request replay.
The new combination and common consumer recipe remain subsequent steps.

No completed v5 T3 measurement or measured batch ETA exists at this start check.
The owner's faster-5090 preference is reflected in the 19/3 allocation.
