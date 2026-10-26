# Engine check of the declared execution variants (admitted 2021-01 -> 2021-04, contract cap 1/day, 1:50)

| variant | policy | trades | /wk | net USD | gross | cost | Sharpe | max DD % | win | exits TP/SL/TIME/AGENT/REV | refused cap | cutoff | rev. waived | median units | hour-0 share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| legacy | pf_sign12h_tp1_sl1_first | 47 | 3.36 | 7536.0 | 8077.5 | 470.3 | 9.37 | 4.57 | 0.872 | 36/4/7/0/0 | 0 | 0 | 0 | 56239 | 0.213 |
| legacy | pf_sign12h_tp1_sl1_best3 | 37 | 2.64 | 8178.0 | 8476.0 | 287.7 | 14.82 | 1.41 | 1.0 | 31/0/6/0/0 | 0 | 0 | 0 | 54824 | 0.0 |
| legacy | pf_sign12h_tp0.7_sl2_first | 57 | 4.07 | 9298.2 | 9902.8 | 605.0 | 17.76 | 2.3 | 0.982 | 53/0/4/0/0 | 0 | 0 | 0 | 57247 | 0.298 |
| legacy | coin_tp1_sl1_seed0 | 25 | 1.79 | 54.1 | 278.7 | 177.9 | 0.21 | 5.77 | 0.52 | 10/9/6/0/0 | 0 | 0 | 0 | 41640 | 0.08 |
| risk1pct | pf_sign12h_tp1_sl1_first | 47 | 3.36 | 3599.8 | 3848.3 | 215.6 | 9.49 | 2.24 | 0.872 | 36/4/7/0/0 | 0 | 0 | 0 | 28368 | 0.213 |
| risk1pct | pf_sign12h_tp1_sl1_best3 | 37 | 2.64 | 3779.7 | 3918.8 | 134.2 | 14.95 | 0.87 | 1.0 | 31/0/6/0/0 | 0 | 0 | 0 | 27449 | 0.0 |
| risk1pct | pf_sign12h_tp0.7_sl2_first | 57 | 4.07 | 1930.6 | 2056.6 | 125.9 | 17.7 | 0.64 | 0.982 | 53/0/4/0/0 | 0 | 0 | 0 | 12451 | 0.298 |
| risk1pct | coin_tp1_sl1_seed0 | 25 | 1.79 | 51.0 | 171.6 | 94.8 | 0.27 | 3.06 | 0.52 | 10/9/6/0/0 | 0 | 0 | 0 | 22324 | 0.08 |
| risk1pct+reverse+cutoff24 | pf_sign12h_tp1_sl1_first | 38 | 2.71 | 2934.6 | 3126.0 | 160.2 | 8.09 | 2.02 | 0.868 | 31/4/3/0/0 | 0 | 0 | 0 | 27253 | 0.132 |
| risk1pct+reverse+cutoff24 | pf_sign12h_tp1_sl1_best3 | 29 | 2.07 | 3225.0 | 3328.5 | 100.0 | 13.39 | 0.87 | 1.0 | 29/0/0/0/0 | 0 | 0 | 0 | 26678 | 0.0 |
| risk1pct+reverse+cutoff24 | pf_sign12h_tp0.7_sl2_first | 47 | 3.36 | 1693.9 | 1791.1 | 97.2 | 15.58 | 0.64 | 1.0 | 47/0/0/0/0 | 0 | 0 | 0 | 12186 | 0.234 |
| risk1pct+reverse+cutoff24 | coin_tp1_sl1_seed0 | 21 | 1.5 | -364.2 | -265.8 | 73.7 | -1.82 | 5.02 | 0.429 | 8/11/2/0/0 | 0 | 0 | 0 | 21915 | 0.048 |
| risk1pct+ask_triggers | pf_sign12h_tp1_sl1_first | 47 | 3.36 | 3599.8 | 3848.3 | 215.6 | 9.49 | 2.24 | 0.872 | 36/4/7/0/0 | 0 | 0 | 0 | 28368 | 0.213 |
| risk1pct+ask_triggers | pf_sign12h_tp1_sl1_best3 | 37 | 2.64 | 3779.7 | 3918.8 | 134.2 | 14.95 | 0.87 | 1.0 | 31/0/6/0/0 | 0 | 0 | 0 | 27449 | 0.0 |
| risk1pct+ask_triggers | pf_sign12h_tp0.7_sl2_first | 57 | 4.07 | 1930.8 | 2056.6 | 126.6 | 17.7 | 0.64 | 0.982 | 53/0/4/0/0 | 0 | 0 | 0 | 12451 | 0.298 |
| risk1pct+ask_triggers | coin_tp1_sl1_seed0 | 25 | 1.79 | 57.5 | 175.3 | 92.5 | 0.29 | 3.06 | 0.52 | 10/9/6/0/0 | 0 | 0 | 0 | 22338 | 0.04 |

Coin flip (5 seeds), mean net USD by variant: {"legacy": -151.1, "risk1pct": -82.8, "risk1pct+reverse+cutoff24": -120.2, "risk1pct+ask_triggers": -69.1}

Effective level cap (sigma multiples) over the span: {"n": 1514, "p1": 2.212181904381138, "p5": 2.2304973743494445, "p25": 2.3486726939651517, "p50": 2.4111912880035913, "p75": 2.460188523861836, "p95": 2.51284074714982, "p99": 2.538686497239844}
