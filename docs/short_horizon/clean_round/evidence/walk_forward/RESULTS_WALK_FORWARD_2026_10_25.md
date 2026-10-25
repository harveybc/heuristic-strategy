# Walk-forward dentro del desarrollo: resultados con errores estándar (spec `clean-expert-guided-a1-20261024-v6`, 2026-10-25)

Dos pliegues: `wf2022` ajusta 2020-01→2022-01 (bloque interno 2021 S2) y puntúa **2022 entero** tras congelar; `wf2023` ajusta 2020-01→2023-01 (interno 2022 S2) y puntúa **2023 entero**. Constantes del escalador re-ajustadas por pliegue (receta derivada, identidad propia); misma representación, modelo, regularización y contrato. Replays de estudiantes; **no es una estrategia operable**; 2024 una lectura por unidad; TEST cerrado.

Regla pre-registrada (antes de leer): PRE-REGISTERED 2026-10-25 before any walk-forward result is read: an arm HOLDS only if its seed-ensemble validation (INNER_B) Sharpe minus one Lo standard error is > 0 in BOTH folds (wf2022 and wf2023); two arms are SEPARABLE on a window only when |delta Sharpe| > the pooled Lo SE sqrt(SE_a^2 + SE_b^2), otherwise NOT_SEPARABLE; nothing is chosen on INNER_B or OBS_2024; the walk-forward is diagnostic and releases nothing; stage B inherits by the consolidated 5-seed rule regardless

### Pliegue wf2022 (INNER_B = año de validación)

| unit | status | n seeds | TRAIN seeds | INNER_A seeds | INNER_B seeds | OBS_2024 seeds | TRAIN mean ± seed σ | INNER_A mean ± seed σ | INNER_B mean ± seed σ | OBS_2024 mean ± seed σ | TRAIN ensemble ± SE [95 %] | INNER_A ensemble ± SE [95 %] | INNER_B ensemble ± SE [95 %] | OBS_2024 ensemble ± SE [95 %] | valid | params |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| WF_wf2022_zz_hold | DONE | 5 | 2.93/5.70/3.93/2.70/4.37 | -0.34/0.43/-0.40/1.66/-2.37 | -0.12/-0.92/1.33/-0.83/0.61 | -1.49/-2.24/0.76/0.71/-0.44 | 3.93 ± 1.08 | -0.20 ± 1.31 | 0.01 ± 0.86 | -0.54 ± 1.19 | 6.00 ± 0.84 [4.66, 7.42] | -0.36 ± 1.41 [-3.15, 2.66] | 0.46 ± 1.00 [-1.51, 2.43] | -1.16 ± 1.00 [-3.18, 0.85] | True | 5460 |
| WF_wf2022_both_hold | DONE | 5 | 6.00/5.51/4.93/4.38/5.02 | 0.02/1.06/0.84/2.21/2.26 | 0.92/0.04/-0.31/-0.65/-0.55 | -2.15/-0.37/-0.87/0.00/-1.74 | 5.17 ± 0.55 | 1.28 ± 0.86 | -0.11 ± 0.57 | -1.03 ± 0.81 | 7.09 ± 0.85 [6.00, 8.16] | 1.46 ± 1.41 [-0.96, 3.68] | 0.60 ± 1.00 [-1.48, 2.78] | -1.43 ± 1.00 [-3.50, 0.54] | False | 5460 |
| WF_wf2022_both_hold_shorter | DONE | 5 | 4.55/5.55/7.60/6.43/4.96 | 0.74/1.93/0.32/1.01/1.57 | 0.06/0.53/0.80/-1.47/-0.19 | 0.73/-0.11/-0.42/-1.62/0.38 | 5.82 ± 1.09 | 1.11 ± 0.58 | -0.06 ± 0.79 | -0.21 ± 0.81 | 8.57 ± 0.86 [7.30, 9.91] | 0.87 ± 1.41 [-1.57, 3.40] | 0.55 ± 1.00 [-1.40, 2.35] | -0.95 ± 1.00 [-2.99, 0.90] | True | 5460 |

### Pliegue wf2023 (INNER_B = año de validación)

| unit | status | n seeds | TRAIN seeds | INNER_A seeds | INNER_B seeds | OBS_2024 seeds | TRAIN mean ± seed σ | INNER_A mean ± seed σ | INNER_B mean ± seed σ | OBS_2024 mean ± seed σ | TRAIN ensemble ± SE [95 %] | INNER_A ensemble ± SE [95 %] | INNER_B ensemble ± SE [95 %] | OBS_2024 ensemble ± SE [95 %] | valid | params |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| WF_wf2023_zz_hold | DONE | 5 | 3.30/3.49/1.23/2.58/2.35 | 1.71/0.54/2.08/1.07/1.98 | -1.73/-2.73/-1.11/-0.29/-1.59 | -0.51/0.82/-0.10/-0.63/-0.99 | 2.59 ± 0.80 | 1.48 ± 0.59 | -1.49 ± 0.80 | -0.28 ± 0.62 | 4.07 ± 0.64 [2.99, 5.14] | 1.45 ± 1.41 [-1.37, 3.91] | -1.51 ± 1.01 [-3.25, 0.26] | 1.47 ± 1.00 [-0.63, 3.66] | False | 5460 |
| WF_wf2023_both_hold | DONE | 5 | 3.88/3.43/1.72/2.92/2.82 | 0.60/1.43/1.09/-0.36/0.66 | -0.00/-0.48/-1.29/-0.94/-1.64 | -1.67/0.48/-0.08/-0.43/0.84 | 2.95 ± 0.72 | 0.68 ± 0.60 | -0.87 ± 0.58 | -0.17 ± 0.87 | 5.23 ± 0.64 [4.24, 6.22] | -0.09 ± 1.41 [-3.04, 2.73] | -0.71 ± 1.00 [-3.03, 1.38] | -0.83 ± 1.00 [-2.67, 0.97] | False | 5460 |
| WF_wf2023_both_hold_shorter | DONE | 5 | 4.58/3.48/2.94/3.13/2.87 | -0.63/1.67/-0.17/1.89/-0.27 | 0.74/-0.06/-0.86/-0.79/-2.11 | 0.31/-0.13/-1.94/0.01/1.18 | 3.40 ± 0.63 | 0.50 ± 1.06 | -0.62 ± 0.95 | -0.11 ± 1.02 | 5.72 ± 0.65 [4.67, 6.77] | -0.36 ± 1.41 [-3.42, 2.45] | -0.35 ± 1.00 [-2.46, 1.98] | 0.56 ± 1.00 [-1.40, 2.45] | False | 5460 |

## Veredictos

- **zz_hold**: DOES_NOT_HOLD — wf2022: 0.46 − SE 1.00, wf2023: -1.51 − SE 1.01
- **both_hold**: DOES_NOT_HOLD — wf2022: 0.60 − SE 1.00, wf2023: -0.71 − SE 1.00
- **both_hold_shorter**: DOES_NOT_HOLD — wf2022: 0.55 − SE 1.00, wf2023: -0.35 − SE 1.00

## Separabilidad frente al control (ensamble, año de validación)

- wf2022/both_hold_vs_zz_hold: Δ +0.14 vs SE agrupado 1.42 → NOT_SEPARABLE
- wf2022/both_hold_shorter_vs_zz_hold: Δ +0.09 vs SE agrupado 1.42 → NOT_SEPARABLE
- wf2023/both_hold_vs_zz_hold: Δ +0.80 vs SE agrupado 1.42 → NOT_SEPARABLE
- wf2023/both_hold_shorter_vs_zz_hold: Δ +1.17 vs SE agrupado 1.42 → NOT_SEPARABLE

## Tabla de cierre (regla del propietario)

| unit | period | metric | model | naive, same rows | skill | literature | comparability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WF_wf2022_both_hold | INNER_B | Sharpe daily sqrt365 of the student replay (dimensionless) / max DD % | ensemble Sharpe 0.60 ± 1.00 (SE) [95 % -1.48, 2.78] / DD 29.1 % / 2.3 orders/wk / valid True | control WF_wf2022_zz_hold ensemble: Sharpe 0.46 ± 1.00 / DD 31.2 % | delta +0.14 vs pooled SE 1.42: NOT_SEPARABLE | NOT_COMPARABLE: behaviour-cloning student under a private replay contract on EURUSD H1; no published value with this contract | same recipe/support (58fa0048cb3ed1c8), same replay contract and costs; wf2022: validation year scored after freeze (no choice made on it) |
| WF_wf2022_both_hold_shorter | INNER_B | Sharpe daily sqrt365 of the student replay (dimensionless) / max DD % | ensemble Sharpe 0.55 ± 1.00 (SE) [95 % -1.40, 2.35] / DD 13.5 % / 2.8 orders/wk / valid True | control WF_wf2022_zz_hold ensemble: Sharpe 0.46 ± 1.00 / DD 31.2 % | delta +0.09 vs pooled SE 1.42: NOT_SEPARABLE | NOT_COMPARABLE: behaviour-cloning student under a private replay contract on EURUSD H1; no published value with this contract | same recipe/support (58fa0048cb3ed1c8), same replay contract and costs; wf2022: validation year scored after freeze (no choice made on it) |

| unit | period | metric | model | naive, same rows | skill | literature | comparability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WF_wf2023_both_hold | INNER_B | Sharpe daily sqrt365 of the student replay (dimensionless) / max DD % | ensemble Sharpe -0.71 ± 1.00 (SE) [95 % -3.03, 1.38] / DD 23.2 % / 2.1 orders/wk / valid True | control WF_wf2023_zz_hold ensemble: Sharpe -1.51 ± 1.01 / DD 31.1 % | delta +0.80 vs pooled SE 1.42: NOT_SEPARABLE | NOT_COMPARABLE: behaviour-cloning student under a private replay contract on EURUSD H1; no published value with this contract | same recipe/support (3cae2db7ad2e7280), same replay contract and costs; wf2023: validation year scored after freeze (no choice made on it) |
| WF_wf2023_both_hold_shorter | INNER_B | Sharpe daily sqrt365 of the student replay (dimensionless) / max DD % | ensemble Sharpe -0.35 ± 1.00 (SE) [95 % -2.46, 1.98] / DD 25.4 % / 2.5 orders/wk / valid True | control WF_wf2023_zz_hold ensemble: Sharpe -1.51 ± 1.01 / DD 31.1 % | delta +1.17 vs pooled SE 1.42: NOT_SEPARABLE | NOT_COMPARABLE: behaviour-cloning student under a private replay contract on EURUSD H1; no published value with this contract | same recipe/support (3cae2db7ad2e7280), same replay contract and costs; wf2023: validation year scored after freeze (no choice made on it) |
