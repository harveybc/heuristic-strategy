# V5 dependent queue recovery

Observation: 2026-10-24 04:03 UTC (2026-10-23 local date).

## Verified status

- V5 ledger: 3,882 distinct units, zero recorded errors.
- Families: 40 RETAINED, 2 RESCUED_INTERACTION, 15 not retained with this evidence.
- Retained columns before the consumer window filter: 168. Exported shortlist:
  165; strict list: 121; core control: 15. The excluded columns are the three
  `fx.xauusd.ret_{1,4,24}` columns, not rejected for lack of predictive signal.
- Every exported list matches SHORTLIST.json in both content and order, with
  no duplicate columns. All lists required by the next spec exist.
- The 31,116 timestamps and all core15 values match the preceding consumer
  store exactly, including NaNs; the core list also matches. This supports
  reusing that unchanged control, subject to the existing receipt verifier.
- Satoshi's missing-list correction `fb1c25b` was already deployed when checked.
  Consumer store, lists, justification, T1 profiles and spec are now complete.

## Additional operational defect and correction

The T2 orchestration parent reserved 6 GiB, then the receipted child requested
another 4 GiB. The first reservation held approximately 6 GiB unused while
only about 1.8 GiB remained admissible for the child. No fit had started.

Patch `5edfcbf` gives admission ownership to the existing receipted runner for
T2 and T3 only. It still launches the scientific child through crispdm-run
with its unchanged 4 GiB cap. CPU preparation retains its original admission.
An explicit guard rejects this mode for any command other than the receipted
runner's `run` command. Retry logs append instead of erasing previous output.

Tests: 5 failures / 2 passes before, 7 passes after. They cover single ownership,
CPU admission, refusal of other commands, both GPU stages, completed-step
reuse, log retention, and exact list order/replay.

The waiting acquisition was cancelled through the admission API (not by
removing a live lease). It exited rc75, the parent exited and released its
reservation. No training child was interrupted. The patched standalone
orchestrator was then started once, preserving all completed step markers.

Start check: one native training PID on the GPU, 888 MiB VRAM, one 4 GiB lease,
approximately 2.34 GiB observed host memory. The base strict T2 arm is running.
This is a start confirmation, not a completed result or a measured ETA.

Producer CONFIG.json SHA256 before and after:
`5ba0c7087eb2853dca56dd865fb8ceaaa932ed12e29db5fbecde3c9f2da667fe`.
Scientific package code, parameters, seeds, splits, data and budgets were not
changed. The orchestrator is deployed separately; it does not alter the pinned
training package.

## Handoff to Satoshi

Do not start a duplicate queue. Integrate `5edfcbf` for subsequent deployments;
the current standalone orchestrator already contains it. The existing sequence
continues through strict T2 base/strong, authenticated reuse/regularization,
new T3 manifest, worker queues and your T3 half. The other worker's T3 package
is still pending that manifest. Send it when produced, with new identities.

Then: collect both halves, apply the frozen rule, confirm combination C, replay
and finish the A1-A4 review/recipe. The later two lanes remain expert-guided
learning and SAC without expert labels/weights. The source-to-tensor probe
must match before their downstream comparison.

This operational recovery is not final scientific acceptance of V5. In
particular, report the disclosed reuse of the internal consumer period in
feature-selection decisions accurately; do not call it an independent holdout.
No TEST scoring or new SAC run was performed here.
