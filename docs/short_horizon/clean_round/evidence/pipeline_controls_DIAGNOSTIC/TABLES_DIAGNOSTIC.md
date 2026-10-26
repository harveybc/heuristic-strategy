### Heads: calibrated AUC INNER_A / INNER_B (Brier INNER_A / INNER_B)

| unit | open_l | open_s | close_l | close_s | nop |
|---|---|---|---|---|---|
| cheat_v13 | 0.81 / 0.80 (0.046 / 0.105) | 0.80 / 0.88 (0.045 / 0.102) | 0.86 / 0.70 (0.110 / 0.235) | 0.84 / 0.74 (0.095 / 0.204) | 0.61 / 0.62 (0.197 / 0.211) |
| cheat_c2 | 0.82 / 0.84 (0.116 / 0.135) | 0.90 / 0.84 (0.094 / 0.124) | 0.68 / 0.65 (0.152 / 0.223) | 0.75 / 0.71 (0.112 / 0.215) | 0.62 / 0.61 (0.176 / 0.179) |
| cheat_noise0.5_v13 | 0.80 / 0.83 (0.043 / 0.104) | 0.77 / 0.86 (0.046 / 0.104) | 0.82 / 0.70 (0.125 / 0.235) | 0.84 / 0.74 (0.108 / 0.198) | 0.62 / 0.61 (0.196 / 0.212) |
| cheat_noise0.5_c2 | 0.80 / 0.75 (0.125 / 0.177) | 0.90 / 0.84 (0.095 / 0.130) | 0.69 / 0.63 (0.149 / 0.227) | 0.81 / 0.67 (0.107 / 0.228) | 0.62 / 0.61 (0.176 / 0.179) |
| cheat_noise1_v13 | 0.81 / 0.85 (0.044 / 0.103) | 0.80 / 0.88 (0.045 / 0.101) | 0.85 / 0.68 (0.113 / 0.243) | 0.82 / 0.68 (0.121 / 0.220) | 0.61 / 0.61 (0.195 / 0.213) |
| cheat_noise1_c2 | 0.78 / 0.75 (0.133 / 0.178) | 0.88 / 0.81 (0.108 / 0.144) | 0.69 / 0.55 (0.146 / 0.242) | 0.77 / 0.61 (0.140 / 0.239) | 0.62 / 0.60 (0.176 / 0.179) |
| cheat_noise2_v13 | 0.76 / 0.81 (0.046 / 0.109) | 0.75 / 0.87 (0.046 / 0.106) | 0.68 / 0.59 (0.177 / 0.255) | 0.62 / 0.53 (0.179 / 0.242) | 0.63 / 0.63 (0.194 / 0.211) |
| cheat_noise2_c2 | 0.73 / 0.68 (0.141 / 0.188) | 0.81 / 0.70 (0.136 / 0.194) | 0.54 / 0.50 (0.206 / 0.240) | 0.74 / 0.58 (0.167 / 0.239) | 0.62 / 0.60 (0.176 / 0.179) |
| syn_v13 | 0.68 / 0.75 (0.046 / 0.115) | 0.63 / 0.72 (0.051 / 0.115) | 0.56 / 0.55 (0.243 / 0.251) | 0.55 / 0.58 (0.216 / 0.204) | 0.59 / 0.56 (0.198 / 0.213) |
| syn_c2 | 0.53 / 0.55 (0.171 / 0.197) | 0.59 / 0.59 (0.187 / 0.193) | 0.50 / 0.43 (0.250 / 0.250) | 0.54 / 0.41 (0.240 / 0.220) | 0.58 / 0.56 (0.179 / 0.177) |
| syn_clean_c2 | 0.61 / 0.65 (0.168 / 0.189) | 0.66 / 0.73 (0.178 / 0.174) | 0.52 / 0.50 (0.250 / 0.250) | 0.50 / 0.61 (0.240 / 0.217) | 0.59 / 0.57 (0.178 / 0.177) |
| shuf_v13 | 0.53 / 0.50 (0.045 / 0.119) | 0.51 / 0.50 (0.045 / 0.119) | 0.80 / 0.66 (0.149 / 0.224) | 0.82 / 0.67 (0.150 / 0.226) | 0.50 / 0.48 (0.199 / 0.220) |
| shuf_c2 | 0.71 / 0.76 (0.144 / 0.175) | 0.84 / 0.79 (0.116 / 0.142) | 0.69 / 0.60 (0.161 / 0.228) | 0.67 / 0.69 (0.131 / 0.226) | 0.52 / 0.49 (0.182 / 0.184) |
| gshuf_v13 | 0.54 / 0.49 (0.048 / 0.119) | 0.52 / 0.49 (0.058 / 0.117) | 0.52 / 0.48 (0.198 / 0.240) | 0.50 / 0.50 (0.229 / 0.225) | 0.50 / 0.49 (0.206 / 0.218) |
| gshuf_c2 | 0.52 / 0.48 (0.161 / 0.208) | 0.51 / 0.50 (0.177 / 0.184) | 0.53 / 0.51 (0.237 / 0.231) | 0.51 / 0.50 (0.247 / 0.236) | 0.52 / 0.48 (0.182 / 0.184) |
| noiseinp_v13 | 0.50 / 0.51 (0.048 / 0.119) | 0.50 / 0.51 (0.057 / 0.117) | 0.51 / 0.53 (0.211 / 0.233) | 0.52 / 0.55 (0.240 / 0.227) | 0.52 / 0.49 (0.205 / 0.219) |
| noiseinp_c2 | 0.51 / 0.50 (0.161 / 0.207) | 0.54 / 0.54 (0.176 / 0.183) | 0.52 / 0.51 (0.220 / 0.231) | 0.51 / 0.53 (0.237 / 0.226) | 0.50 / 0.50 (0.182 / 0.183) |

### Trading (engine, sac replay contract)

| unit | period | trades/wk | % buy / sell | win % buy / sell | PF | expectancy (sigma) | Sharpe | max DD % | net USD | exits TP/SL/TIME/AGENT/REV |
|---|---|---|---|---|---|---|---|---|---|---|
| cheat_v13 | INNER_A | 4.30 | 34 / 66 | 75 / 75 | 7.68 | 0.379 | 12.22 | 2.3 | 9619 | 60/0/8/41/7 |
| cheat_v13 | INNER_B | 3.87 | 42 / 58 | 76 / 69 | 4.32 | 0.374 | 8.33 | 6.4 | 85054 | 107/0/17/68/13 |
| cheat_c2 | INNER_A | 2.96 | 42 / 57 | 88 / 96 | 44.44 | 0.927 | 12.53 | 2.5 | 21365 | 22/0/8/50/0 |
| cheat_c2 | INNER_B | 2.77 | 46 / 54 | 74 / 85 | 11.56 | 0.760 | 9.67 | 12.3 | 247458 | 34/0/27/86/0 |
| cheat_noise0.5_v13 | INNER_A | 3.70 | 58 / 42 | 76 / 83 | 16.14 | 0.451 | 12.64 | 1.6 | 10125 | 52/0/4/44/0 |
| cheat_noise0.5_v13 | INNER_B | 3.83 | 55 / 45 | 68 / 70 | 7.21 | 0.367 | 9.53 | 6.8 | 83070 | 86/0/12/105/0 |
| cheat_noise0.5_c2 | INNER_A | 2.93 | 52 / 48 | 71 / 92 | 29.41 | 0.772 | 11.15 | 2.5 | 15649 | 21/0/11/37/10 |
| cheat_noise0.5_c2 | INNER_B | 3.08 | 45 / 55 | 76 / 87 | 8.56 | 0.696 | 9.02 | 16.5 | 244648 | 31/2/25/78/27 |
| cheat_noise1_v13 | INNER_A | 3.85 | 46 / 54 | 73 / 75 | 9.33 | 0.401 | 11.86 | 2.8 | 9155 | 57/0/11/31/5 |
| cheat_noise1_v13 | INNER_B | 3.60 | 43 / 57 | 75 / 67 | 4.50 | 0.368 | 7.50 | 12.6 | 69050 | 100/2/18/61/10 |
| cheat_noise1_c2 | INNER_A | 3.07 | 51 / 49 | 67 / 88 | 28.57 | 0.694 | 10.14 | 3.5 | 14468 | 18/0/7/47/11 |
| cheat_noise1_c2 | INNER_B | 2.81 | 50 / 50 | 68 / 85 | 5.43 | 0.561 | 7.40 | 13.8 | 101811 | 27/2/18/90/12 |
| cheat_noise2_v13 | INNER_A | 3.59 | 60 / 40 | 66 / 85 | 5.05 | 0.368 | 8.48 | 2.9 | 7377 | 49/0/8/35/5 |
| cheat_noise2_v13 | INNER_B | 3.17 | 45 / 55 | 63 / 72 | 2.41 | 0.230 | 4.07 | 12.1 | 22359 | 74/6/25/60/3 |
| cheat_noise2_c2 | INNER_A | 2.67 | 38 / 62 | 70 / 69 | 11.12 | 0.577 | 7.38 | 3.9 | 9039 | 14/1/17/40/0 |
| cheat_noise2_c2 | INNER_B | 2.75 | 45 / 55 | 54 / 60 | 1.78 | 0.203 | 2.47 | 18.8 | 12578 | 16/7/23/100/0 |
| syn_v13 | INNER_A | 2.89 | 64 / 36 | 60 / 68 | 2.14 | 0.166 | 3.31 | 7.4 | 4000 | 19/1/9/49/0 |
| syn_v13 | INNER_B | 2.70 | 53 / 47 | 55 / 64 | 1.08 | 0.036 | 0.48 | 19.7 | 974 | 35/13/21/70/4 |
| syn_c2 | INNER_A | 2.78 | 79 / 21 | 58 / 62 | 2.23 | 0.114 | 2.83 | 6.0 | 2319 | 2/0/0/73/0 |
| syn_c2 | INNER_B | 2.89 | 58 / 42 | 36 / 49 | 0.81 | -0.044 | -0.85 | 30.9 | -1731 | 5/5/2/138/3 |
| syn_clean_c2 | INNER_A | 3.15 | 46 / 54 | 49 / 57 | 2.07 | 0.143 | 3.39 | 8.6 | 3527 | 9/0/0/76/0 |
| syn_clean_c2 | INNER_B | 3.09 | 60 / 40 | 37 / 48 | 1.10 | 0.018 | 0.49 | 16.1 | 771 | 10/0/0/154/0 |
| shuf_v13 | INNER_A | 3.78 | 55 / 45 | 64 / 78 | 9.00 | 0.383 | 10.04 | 2.5 | 8480 | 51/0/13/38/0 |
| shuf_v13 | INNER_B | 3.34 | 37 / 63 | 68 / 66 | 2.85 | 0.284 | 5.33 | 10.5 | 32719 | 82/5/17/73/0 |
| shuf_c2 | INNER_A | 2.41 | 52 / 48 | 82 / 97 | 22.46 | 0.993 | 11.26 | 2.2 | 16757 | 22/0/15/28/0 |
| shuf_c2 | INNER_B | 2.38 | 53 / 47 | 67 / 86 | 4.60 | 0.734 | 7.30 | 18.2 | 128425 | 34/3/34/55/0 |
| gshuf_v13 | INNER_A | 3.81 | 13 / 87 | 85 / 52 | 1.73 | 0.123 | 3.39 | 4.9 | 1963 | 21/1/13/64/4 |
| gshuf_v13 | INNER_B | 3.85 | 17 / 83 | 50 / 42 | 0.91 | -0.011 | -0.29 | 30.4 | -1010 | 31/2/20/137/14 |
| gshuf_c2 | INNER_A | 2.33 | 21 / 79 | 46 / 46 | 1.29 | 0.019 | 0.85 | 4.1 | 233 | 0/0/0/62/1 |
| gshuf_c2 | INNER_B | 2.79 | 23 / 77 | 59 / 37 | 0.97 | -0.013 | -0.01 | 25.5 | -194 | 3/1/4/139/1 |
| noiseinp_v13 | INNER_A | 2.74 | 19 / 81 | 93 / 75 | 2.87 | 0.255 | 4.14 | 4.6 | 3666 | 49/1/14/10/0 |
| noiseinp_v13 | INNER_B | 3.53 | 29 / 71 | 65 / 65 | 1.09 | 0.035 | 0.62 | 31.9 | 1600 | 107/15/32/33/0 |
| noiseinp_c2 | INNER_A | 3.48 | 18 / 82 | 47 / 39 | 1.13 | 0.003 | 0.54 | 5.8 | 180 | 1/0/3/88/2 |
| noiseinp_c2 | INNER_B | 3.47 | 22 / 78 | 41 / 38 | 0.86 | -0.010 | -0.49 | 17.2 | -864 | 2/2/3/170/7 |

### Frozen strategy genes (chosen on INNER_A)

| unit | genes | strategy |
|---|---|---|
| cheat_v13 | `{"open_rate_l": 0.134672, "open_rate_s": 0.168685, "gain_quantile_l": 0.897345, "gain_quantile_s": 0.285337, "close_rate": 0.322329, "allow_reverse": 0.935148, "nop_veto_rate": 0.009373}` | `{"open_threshold_l": 0.123, "open_threshold_s": 0.114, "gain_margin_l": -1.76, "gain_margin_s": -2.114, "close_threshold": 0.501, "allow_reverse": true, "nop_veto": 0.872}` |
| cheat_c2 | `{"open_rate_l": 0.170785, "open_rate_s": 0.265542, "close_rem_rate": 0.424697, "close_p_rate": 0.100523, "allow_reverse": 0.243551, "nop_veto_rate": 0.260387}` | `{"open_margin_l": 0.827, "open_margin_s": 0.796, "close_margin": 0.151, "close_p": 0.472, "allow_reverse": false, "nop_veto": 0.82}` |
| cheat_noise0.5_v13 | `{"open_rate_l": 0.088547, "open_rate_s": 0.052065, "gain_quantile_l": 0.380955, "gain_quantile_s": 0.904376, "close_rate": 0.549695, "allow_reverse": 0.303514, "nop_veto_rate": 0.3195}` | `{"open_threshold_l": 0.134, "open_threshold_s": 0.153, "gain_margin_l": -2.106, "gain_margin_s": -1.784, "close_threshold": 0.307, "allow_reverse": false, "nop_veto": 0.765}` |
| cheat_noise0.5_c2 | `{"open_rate_l": 0.259582, "open_rate_s": 0.272998, "close_rem_rate": 0.354981, "close_p_rate": 0.131709, "allow_reverse": 0.548521, "nop_veto_rate": 0.252029}` | `{"open_margin_l": 0.668, "open_margin_s": 0.678, "close_margin": 0.188, "close_p": 0.489, "allow_reverse": true, "nop_veto": 0.822}` |
| cheat_noise1_v13 | `{"open_rate_l": 0.071537, "open_rate_s": 0.105684, "gain_quantile_l": 0.159896, "gain_quantile_s": 0.666003, "close_rate": 0.259928, "allow_reverse": 0.544642, "nop_veto_rate": 0.11552}` | `{"open_threshold_l": 0.156, "open_threshold_s": 0.146, "gain_margin_l": -2.141, "gain_margin_s": -1.968, "close_threshold": 0.509, "allow_reverse": true, "nop_veto": 0.813}` |
| cheat_noise1_c2 | `{"open_rate_l": 0.182074, "open_rate_s": 0.267492, "close_rem_rate": 0.374451, "close_p_rate": 0.059591, "allow_reverse": 0.53158, "nop_veto_rate": 0.288481}` | `{"open_margin_l": 0.86, "open_margin_s": 0.645, "close_margin": 0.207, "close_p": 0.528, "allow_reverse": true, "nop_veto": 0.816}` |
| cheat_noise2_v13 | `{"open_rate_l": 0.081513, "open_rate_s": 0.079392, "gain_quantile_l": 0.226548, "gain_quantile_s": 0.646767, "close_rate": 0.135271, "allow_reverse": 0.92176}` | `{"open_threshold_l": 0.123, "open_threshold_s": 0.131, "gain_margin_l": -2.119, "gain_margin_s": -1.986, "close_threshold": 0.383, "allow_reverse": true, "nop_veto": null}` |
| cheat_noise2_c2 | `{"open_rate_l": 0.11908, "open_rate_s": 0.271989, "close_rem_rate": 0.144223, "close_p_rate": 0.110564, "allow_reverse": 0.264109, "nop_veto_rate": 0.260275}` | `{"open_margin_l": 0.893, "open_margin_s": 0.498, "close_margin": 0.165, "close_p": 0.511, "allow_reverse": false, "nop_veto": 0.82}` |
| syn_v13 | `{"open_rate_l": 0.063786, "open_rate_s": 0.032087, "gain_quantile_l": 0.268397, "gain_quantile_s": 0.757997, "close_rate": 0.321741, "allow_reverse": 0.626644, "nop_veto_rate": 0.139789}` | `{"open_threshold_l": 0.097, "open_threshold_s": 0.103, "gain_margin_l": -2.059, "gain_margin_s": -1.934, "close_threshold": 0.423, "allow_reverse": true, "nop_veto": 0.783}` |
| syn_c2 | `{"open_rate_l": 0.244154, "open_rate_s": 0.056435, "close_rem_rate": 0.283636, "close_p_rate": 0.045084, "allow_reverse": 0.616952, "nop_veto_rate": 0.460901}` | `{"open_margin_l": 0.28, "open_margin_s": 1.14, "close_margin": -0.233, "close_p": 0.484, "allow_reverse": true, "nop_veto": 0.776}` |
| syn_clean_c2 | `{"open_rate_l": 0.24309, "open_rate_s": 0.286823, "close_rem_rate": 0.578373, "close_p_rate": 0.069801, "allow_reverse": 0.011304}` | `{"open_margin_l": 0.386, "open_margin_s": 0.323, "close_margin": -0.125, "close_p": 0.501, "allow_reverse": false, "nop_veto": null}` |
| shuf_v13 | `{"open_rate_l": 0.131619, "open_rate_s": 0.277804, "gain_quantile_l": 0.249306, "gain_quantile_s": 0.854479, "close_rate": 0.387239, "allow_reverse": 0.34776, "nop_veto_rate": 0.220824}` | `{"open_threshold_l": 0.052, "open_threshold_s": 0.049, "gain_margin_l": -2.033, "gain_margin_s": -2.008, "close_threshold": 0.47, "allow_reverse": false, "nop_veto": 0.726}` |
| shuf_c2 | `{"open_rate_l": 0.235943, "open_rate_s": 0.230497, "close_rem_rate": 0.212275, "close_p_rate": 0.091506, "allow_reverse": 0.009062, "nop_veto_rate": 0.408947}` | `{"open_margin_l": 0.538, "open_margin_s": 0.779, "close_margin": 0.349, "close_p": 0.4, "allow_reverse": false, "nop_veto": 0.767}` |
| gshuf_v13 | `{"open_rate_l": 0.042577, "open_rate_s": 0.258775, "gain_quantile_l": 0.277139, "gain_quantile_s": 0.636272, "close_rate": 0.121914, "allow_reverse": 0.943898, "nop_veto_rate": 0.0278}` | `{"open_threshold_l": 0.061, "open_threshold_s": 0.065, "gain_margin_l": -2.017, "gain_margin_s": -1.972, "close_threshold": 0.357, "allow_reverse": true, "nop_veto": 0.711}` |
| gshuf_c2 | `{"open_rate_l": 0.010251, "open_rate_s": 0.043059, "close_rem_rate": 0.435486, "close_p_rate": 0.175298, "allow_reverse": 0.642795, "nop_veto_rate": 0.114792}` | `{"open_margin_l": 0.62, "open_margin_s": 0.655, "close_margin": -0.164, "close_p": 0.447, "allow_reverse": true, "nop_veto": 0.779}` |
| noiseinp_v13 | `{"open_rate_l": 0.171408, "open_rate_s": 0.263749, "gain_quantile_l": 0.690805, "gain_quantile_s": 0.582138, "close_rate": 0.009291, "allow_reverse": 0.279809}` | `{"open_threshold_l": 0.052, "open_threshold_s": 0.062, "gain_margin_l": -2.008, "gain_margin_s": -1.987, "close_threshold": 0.42, "allow_reverse": false, "nop_veto": null}` |
| noiseinp_c2 | `{"open_rate_l": 0.031467, "open_rate_s": 0.161194, "close_rem_rate": 0.286991, "close_p_rate": 0.111988, "allow_reverse": 0.702552, "nop_veto_rate": 0.271465}` | `{"open_margin_l": 0.835, "open_margin_s": 0.672, "close_margin": -0.149, "close_p": 0.386, "allow_reverse": true, "nop_veto": 0.762}` |
