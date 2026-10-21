# Monitor generalization contrast and HistData discrepancy, 2026-10-20

Owner approved the next diagnostic/monitor experiment and supplied the HistData
clock warning. Keep the two research lanes complementary; do not modify the
other lane's running selection or integrate its new inputs into this contrast.

## Requirements and test design (before implementation)

MG01: independently reproduce the reported HistData/accepted-grid mismatch.
Separate clock offset, absent source rows, OHLC disagreement and the interval
over which each return is calculated. A multihour return cannot be compared to
a one-hour return. No filling or interpolation. Test with identical synthetic
prices with a deleted observation, deliberate clock shift, DST transition and
real price alteration. Record all observed OHLC mismatches, not just the median.
The accepted source and selected-input identities must remain pinned.

MG02: inspect all final module learning curves from the closed study. Report
epoch of minimum monitor, minimum internal-validation loss, and the difference;
this is a diagnosis, not a retrospective replacement of scientific outcomes.

MG03: small controlled FULL experiment, five paired seeds, two predeclared
selectors. Reference M0=.5*(TRAIN_loss+INNER_A_loss). Treatment
M1=M0+.25*max(0,INNER_A_loss-TRAIN_loss). Both losses are the existing end-of-
epoch inference losses; patience10, min_delta1e-8, max1000 epochs, best restored.
The .25 coefficient is an engineering hypothesis fixed before outcomes, not a
claimed optimum or a new sweep. Verify zero penalty preserves native fitting.
Fit both arms fresh under the same new code so donor/code differences cannot
explain an effect. CPU formula/fit/reload tests precede GPU dispatch.

MG04: only the monitor selector changes. Same four specialists, trunk/head,
dropout .1, AdamW .001 / wd1e-4, BC+3 DAgger rounds, 15 selected variables,
13 slots/stride2, business rules and costs as the completed modular study.
Same TRAIN2020-2022, INTERNAL2023H1, diagnostic2023H2 and complete2024 development.
TEST2025 excluded. Changing the selector can change later DAgger trajectories;
that is part of its total intervention, not an identical-label claim.

MG05: all ten cells run sequentially on the preferred GPU with native finite
queue/retries/publication/independent-copy automation. Refuse scientific
dispatch if the source audit finds an unresolved error affecting the admitted
inputs/target. A difference in the unconsumed HistData candidate alone does not
invalidate accepted-source results. Keep every weight, loss curve, seed,
threshold, trade ledger and full TRAIN/development metrics, and all outcomes.
Criteria: paired validation profit/daily growth, gap, drawdown, activity and
cross-seed stability; shrinking the gap by damaging TRAIN alone is insufficient.

## Subsequent designs, not this batch

Short term: diagnostic module/cost/quarter panels; this monitor contrast.
Medium term: limited window/stride contrasts with matched historical span and
receptive-field coverage; native SAC specialization with economic reward and
critic/replay/exploration diagnostics. Long term: integrate useful new input
families from the other lane under point-in-time clocks; causal adaptation
cadence/regime experiments, frozen evaluation and paper trading.

Every stage must be a finite resumable script with status and measured ETA.
The current queue ends after ten cells, publication and independent copies.
