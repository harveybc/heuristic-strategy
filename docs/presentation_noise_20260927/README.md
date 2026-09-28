# Sources for the doctoral interview presentation

Archived 2026-09-27 at the author's request. No simulation was executed for this
publication. These are TWO DIFFERENT experiments, not two versions of one curve.

## Figure actually displayed in the presentation

`historical/figure_as_presented.png` is the exact embedded PNG from slide 5.
It was not redrawn, replaced, recolored or edited for this source archive.
Its 20 points come from `historical/sweep_noise_results.csv`, whose SHA256 is
`b164d34a0f58dd512357de8bc64e157b780873d0432539c72695346869d46bfa`.
The source bars are `historical/input.csv`, identical to the repository's
`tests/data/phase_2_3_base_d3.csv` (18,475 hourly rows, 2017-03-22 to 2020-03-19).
The retained producer is `historical/sweep_noise.py`, using the directional
oracle service and `historical/plugin_direction_atr.py`, NOT the two-price-
forecast native strategy used in the later experiment.

The plotted noiseless profit is USD 306,398.8839101093, with 753 trades. The
CSV's sign change is between noise standard deviations 0.35 and 0.40. The
producer's printed label `NAIVE POINT` means a profit sign change: it is NOT a
persistence-forecast MAE and must not be cited as one. The historical `sharpe`
column is not a return-based annualized Sharpe. Configured swap was deducted in
the plugin's trade display but not in the broker-equity profit plotted here.
These limitations are preserved; the original curve is not promoted to a
validated live-trading or noisy-price experiment. The old service state and its
exact historical random draws are not retained in this archive, so a bitwise
backtest replay is not claimed. Data/code context is reconstructed from retained
files, not a recovered sealed execution manifest.

## Latest experiment, not substituted into the presentation

`latest_native/` contains the latest completed conditional H6/H144 price-noise
experiment: raw input, timestamp origins, targets, three noise realizations,
source snapshots, frozen design, 241-cell result table, matched naive errors,
paired summaries, execution diagnostics and the four latest PNGs. Start with
`latest_native/RESULTS.md`. H6 and H144 persistence MAE are respectively
0.0017059896983075775 and 0.007522118469462839 USD/EUR on 13,590 common origins.
The data SHA256 agrees with the historical bars; the task and simulator differ.

This compact archive does not include the approximately 1 GiB of per-cell
decision, fill and equity logs from the full local run. The retained independent
verification is the receipt of that full run, not evidence that those omitted
logs can be verified from this compact archive alone. Do not claim a complete
ledger replay from this folder. Forecast values can be reconstructed from the
stored targets, noise arrays and the manifest's amplitudes without guessing a
random generator version.

## Integrity

`FILES.json` lists the size and SHA256 of every retained source artifact. Archive
documentation is not an additional scientific observation. The thesis scatter
plot has a separate source: the master's dissertation retained in predictor,
`docs/tesis_maestria_ds/tesis_maestria_ciencia_datos_2025.docx`, Figure 43.
