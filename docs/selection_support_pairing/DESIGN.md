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

Second demonstrated defect: directory globbing can read the 2023-fit subset
while MANIFEST.json declares the full decomposition pool. Silent column
intersection then hides the missing families. The candidate now resolves the
declared filename and verifies its digest; absent a filename it requires one
unique digest match. Missing declared columns are rejected before any fit.
Six additional PRE failures / POST passes cover this loading contract.
Changing this input changes the experiment, so the old base cannot serve as the
base for the declared complete pool. Recompute the affected selection under a
fresh identity, not just the contrast repair described for an unchanged store.
