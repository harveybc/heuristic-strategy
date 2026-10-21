# HistData 2023H1: price-difference claim refuted on shared observations

Read-only reproduction, 2026-10-20. No edits to the running cross-asset selection,
the admitted root, old experiments or source prices. Reproducer:

```bash
python -m tools.source_interval_audit --admitted "$ADMITTED_FILE" \
  --histdata "$HISTDATA_FILE" --output "$PRIVATE_AUDIT_ROOT"
```

Exact inputs: admitted c80847bb149c7af1ad50e42301b5cd70608e6c37702ce8cc6a7b9eb70f36d2ca;
lake HistData c6628375b2872f16dcdbd6b1b90ed8e68bc5bdd852efdef01f2515b8712d4e2e.
Use the observed cohort clock `(wall+5h).tz_localize(Europe/London).tz_convert(UTC)`.
UTC bar start, decisions after completion. No interpolation or price adjustments.

2023H1 admitted rows3111, shared2436, missing from candidate675. ALL FOUR OHLC
fields match exactly on the shared rows (zero mismatched cells).
The claim of widespread conflicting price values in this half-year is not
supported by direct OHLC comparison on the actual common timestamps.

The clock gate currently calculates `log(admitted).diff()` and
`log(histdata).diff()` separately, then aligns the returns. When HistData misses
an observation, its next diff spans multiple hours while the admitted return
spans one hour. A matched end timestamp is not a matched return interval.
On Feb-Jul2023, the naive interval comparison gives correlation0.82757; comparing
only actual consecutive hourly intervals in BOTH feeds gives correlation1.0
(1579 matched hourly pairs;703 other pairs had unequal/nonhourly intervals).

Correction for the other lane: mask each return unless its own previous
timestamp is exactly one hour earlier; then compare at common timestamps.
Keep genuine absent source bars explicit. Recompute the clock-gate diagnostic
and its wording; this evidence does not call for changing shared price values
or rerunning feature generation solely because of this diagnostic correction.
Inspect interval support for any cross-asset metric advertised as fixed hours.
Observed-bar lag features may intentionally span longer wall time, but must not
be interpreted as fixed-hour returns across missing open-session observations.

Across2020-2024 there are30251 shared rows,865 candidate absences, and16 isolated
OHLC mismatch cells outside2023H1 (five close cells among them); the reproducible
audit retains all timestamps, fields and sources for scoped review. Whole-sample
matched one-hour return correlation0.99999838. Agreement between sources is not
an independent certification of their common upstream prices.

This finding affects the interpretation of the reported gate, not the target
used by Musashi's closed RL/modular experiments: those used the pinned admitted
root and selected CSVs, never this incomplete candidate as their target source.
