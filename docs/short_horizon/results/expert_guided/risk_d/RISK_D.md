# Phase D: risk engine over the frozen FULL specialists (DEVELOPMENT)

Chosen on VAL A: **gate=None|size=voltarget0.5|levels=(1.0, 1.0)**; reference = unmodified policy. Mean of 3 seeds.

| Rule | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Sharpe 2024 | DD % VAL B | Orders/wk VAL B | Valid VAL A |
|---|---|---|---|---|---|---|---|
| gate=None|size=voltarget0.5|levels=(1.0, 1.0) **(chosen)** | -0.11 | 0.93 | -1.48 | -0.18 | 21.59 | 4.00 | True |
| gate=None|size=voltarget1.0|levels=(1.0, 1.0) | -0.10 | 0.90 | -1.47 | — | 35.86 | 4.00 | True |
| gate=None|size=fixed|levels=(1.0, 1.0) *(reference)* | -0.05 | 0.86 | -1.47 | -0.06 | 35.91 | 4.00 | True |
| gate=0.35|size=voltarget0.5|levels=(1.0, 1.0) | -0.12 | 0.67 | -0.89 | — | 13.02 | 2.05 | False |
| gate=0.35|size=voltarget1.0|levels=(1.0, 1.0) | -0.12 | 0.60 | -0.87 | — | 22.89 | 2.05 | False |
| gate=0.25|size=voltarget0.5|levels=(1.0, 1.0) | -0.13 | 0.48 | -1.19 | — | 16.70 | 2.47 | False |
| gate=0.25|size=voltarget1.0|levels=(1.0, 1.0) | -0.12 | 0.48 | -1.18 | — | 28.77 | 2.47 | False |
| gate=0.35|size=fixed|levels=(1.0, 1.0) | -0.07 | 0.45 | -0.87 | — | 22.88 | 2.05 | False |
| gate=0.25|size=fixed|levels=(1.0, 1.0) | -0.06 | 0.39 | -1.18 | — | 28.86 | 2.47 | False |
| gate=0.3|size=voltarget1.0|levels=(1.0, 1.0) | -0.11 | 0.26 | -0.78 | — | 24.95 | 2.25 | False |
| gate=0.3|size=voltarget0.5|levels=(1.0, 1.0) | -0.13 | 0.26 | -0.79 | — | 14.15 | 2.25 | False |
| gate=0.25|size=voltarget1.0|levels=(1.25, 0.75) | -0.49 | 0.21 | -1.13 | — | 28.42 | 3.16 | False |
| gate=0.25|size=fixed|levels=(1.25, 0.75) | -0.43 | 0.19 | -1.13 | — | 28.45 | 3.16 | False |
| gate=0.25|size=voltarget0.5|levels=(1.25, 0.75) | -0.50 | 0.16 | -1.13 | — | 16.21 | 3.16 | False |
| gate=0.3|size=fixed|levels=(1.0, 1.0) | -0.09 | 0.14 | -0.78 | — | 24.99 | 2.25 | False |
| gate=0.3|size=voltarget1.0|levels=(1.25, 0.75) | -0.55 | -0.48 | -1.10 | — | 27.25 | 2.91 | False |
| gate=0.25|size=voltarget0.5|levels=(0.75, 1.25) | 0.02 | -0.50 | 0.05 | — | 14.34 | 2.48 | False |
| gate=0.3|size=voltarget0.5|levels=(1.25, 0.75) | -0.56 | -0.53 | -1.09 | — | 15.43 | 2.91 | False |
| gate=0.35|size=voltarget0.5|levels=(1.25, 0.75) | -0.51 | -0.53 | -1.20 | — | 14.30 | 2.72 | False |
| gate=0.3|size=fixed|levels=(1.25, 0.75) | -0.51 | -0.54 | -1.10 | — | 27.29 | 2.91 | False |
| gate=0.35|size=voltarget1.0|levels=(1.25, 0.75) | -0.51 | -0.54 | -1.19 | — | 25.58 | 2.72 | False |
| gate=0.25|size=fixed|levels=(0.75, 1.25) | -0.02 | -0.58 | 0.11 | — | 24.98 | 2.48 | False |
| gate=0.25|size=voltarget1.0|levels=(0.75, 1.25) | -0.02 | -0.58 | 0.10 | — | 25.00 | 2.48 | False |
| gate=0.35|size=fixed|levels=(1.25, 0.75) | -0.47 | -0.62 | -1.20 | — | 25.63 | 2.72 | False |
| gate=None|size=voltarget0.5|levels=(0.75, 1.25) | -0.13 | -0.68 | -0.30 | — | 16.53 | 3.86 | True |
| gate=0.3|size=voltarget0.5|levels=(0.75, 1.25) | 0.17 | -0.70 | 0.10 | — | 14.71 | 2.22 | False |
| gate=None|size=fixed|levels=(1.25, 0.75) | -0.52 | -0.72 | -1.96 | — | 40.98 | 5.54 | True |
| gate=None|size=voltarget1.0|levels=(1.25, 0.75) | -0.51 | -0.72 | -1.96 | — | 40.96 | 5.54 | True |
| gate=None|size=voltarget0.5|levels=(1.25, 0.75) | -0.56 | -0.74 | -1.93 | — | 24.35 | 5.54 | True |
| gate=None|size=fixed|levels=(0.75, 1.25) | -0.12 | -0.75 | -0.26 | — | 29.00 | 3.86 | True |
| gate=None|size=voltarget1.0|levels=(0.75, 1.25) | -0.14 | -0.76 | -0.27 | — | 29.03 | 3.86 | True |
| gate=0.3|size=voltarget1.0|levels=(0.75, 1.25) | 0.15 | -0.76 | 0.13 | — | 25.51 | 2.22 | False |
| gate=0.3|size=fixed|levels=(0.75, 1.25) | 0.11 | -0.76 | 0.13 | — | 25.53 | 2.22 | False |
| gate=0.35|size=voltarget0.5|levels=(0.75, 1.25) | 0.11 | -1.61 | 0.11 | — | 13.15 | 2.01 | False |
| gate=0.35|size=voltarget1.0|levels=(0.75, 1.25) | 0.10 | -1.65 | 0.15 | — | 23.13 | 2.01 | False |
| gate=0.35|size=fixed|levels=(0.75, 1.25) | 0.10 | -1.68 | 0.15 | — | 23.13 | 2.01 | False |
