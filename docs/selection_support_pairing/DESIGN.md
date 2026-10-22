# Family Contrast Support Repair

Base: 1e840de. Candidate only; do not edit or redeploy an active scientific tree.

Requirement: all arms of one (task, family, context, fold) must use the same
origins and fold boundaries, including parents and every component/subgroup.
Feature columns still differ by arm; columns defining eligible rows do not.

Test design: execute the real Engine.arms method with a synthetic engine and
verify the identical union of context + parent + components for every arm;
verify that baseline input columns do not acquire the withheld family.
Verify that full and core contexts retain distinct, internally paired support.
No model fit, GPU, price data, service or active ledger is used by these tests.

Implementation: use Engine.unit's existing row_cols argument, already used by
encoding. Add support_protocol=family_union_v1 to the recorded contrast rows.

Deployment requirement: new output identity; legacy contrast keys must not be
silently reused under their new support. The base is not changed. Reuse of any
unchanged fit requires a verified identical support/val hash, seed and input
column hash. Rescue comparisons still require their own paired controls; this
candidate does not claim to repair them or finish the scientific audit.
