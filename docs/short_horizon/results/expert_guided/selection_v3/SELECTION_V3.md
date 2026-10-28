# Pool selection v2

Columns 317 in 57 families; units done 3894; failed 0.
Informative tasks (HGB base CV loss below prior): ['opener3', 'hit']

Scope: every negative statement here means 'no improvement over the prior in these tasks / models / supports', never absence of signal in general; unpaired arms are listed per family and never subtracted.

## Base losses (full pool)

| task | model | CV loss | prior | train loss | params |
|---|---|---|---|---|---|
| opener3 | hgb | 0.4434 | 0.6963 | 0.3327 | 9741 |
| opener3 | temporal | 0.5305 | 0.6907 | 0.4640 | 5165 |
| hit | hgb | 0.1667 | 0.1718 | 0.1399 | 735 |
| hit | temporal | 0.1785 | 0.1766 | 0.1519 | 5162 |
| up_given_hit | hgb | 0.7017 | 0.6946 | 0.6747 | 21 |
| up_given_hit | temporal | 0.6968 | 0.6941 | 0.6789 | 5162 |
| sl_first_long | hgb | 0.6985 | 0.6925 | 0.6703 | 21 |
| sl_first_long | temporal | 0.7052 | 0.6921 | 0.6674 | 5162 |
| sl_first_short | hgb | 0.7009 | 0.6951 | 0.6796 | 21 |
| sl_first_short | temporal | 0.6996 | 0.6944 | 0.6795 | 5162 |
| net_long | hgb | 1.7079 | 1.6926 | 1.6222 | 27 |
| net_long | temporal | 1.7597 | 1.6826 | 1.6408 | 5159 |
| net_short | hgb | 1.7079 | 1.6926 | 1.6222 | 27 |
| net_short | temporal | 1.6960 | 1.6826 | 1.6996 | 5159 |

## Family decisions

| family | cols | status | retained arm | evidence | unpaired groups |
|---|---|---|---|---|---|
| btc | 5 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| dly.dax | 2 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| dly.us10y2y | 3 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| dly.us2y | 3 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| fractal | 12 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| fx.audusd | 4 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| fx.eur_strength | 3 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| fx.eurjpy | 4 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| fx.usdjpy | 4 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| hilbert/log_close | 4 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| multitaper/logret_1 | 7 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| px.atr | 1 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| px.unexpected_gap_hours | 1 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| ssa/BidClose | 3 | NOT_RETAINED_WITH_THIS_EVIDENCE |  |  | 0 |
| dly.oil | 2 | RESCUED_INTERACTION |  | interaction:opener3: +0.0007 (lo90 0.0004) | 0 |
| fx.usd_strength | 3 | RESCUED_INTERACTION |  | interaction:opener3: +0.0007 (lo90 0.0004) | 0 |
| act | 1 | RETAINED |  | block_permutation:opener3:hgb +0.0001 (lo90 0.0000); block_permutation:opener3:temporal +0.0004 (lo90 0.0000); block_permutation:hit:hgb +0.0098 (lo90 0.0076); block_permutation:hit:temporal +0.0005 (lo90 0.0001) | 0 |
| calendar | 13 | RETAINED |  | block_permutation:opener3:temporal +0.0011 (lo90 0.0009); block_permutation:hit:temporal +0.0010 (lo90 0.0002); retrained_contrast:opener3:with +0.0073 (lo90 0.0056) | 0 |
| cdl | 6 | RETAINED |  | block_permutation:opener3:hgb +0.0104 (lo90 0.0093); block_permutation:opener3:temporal +0.0817 (lo90 0.0759); retrained_contrast:opener3:with +0.0187 (lo90 0.0151) | 0 |
| cot.eur | 4 | RETAINED |  | block_permutation:hit:hgb +0.0017 (lo90 0.0009); retrained_contrast:hit:with +0.0018 (lo90 0.0016) | 0 |
| dly.gold | 2 | RETAINED |  | block_permutation:opener3:hgb +0.0001 (lo90 0.0000); retrained_contrast:hit:with +0.0009 (lo90 0.0004) | 0 |
| dly.spx | 2 | RETAINED |  | block_permutation:opener3:temporal +0.0007 (lo90 0.0000) | 0 |
| dly.us10y | 3 | RETAINED |  | block_permutation:opener3:hgb +0.0008 (lo90 0.0003); block_permutation:hit:temporal +0.0002 (lo90 0.0001) | 0 |
| dly.vix | 2 | RETAINED |  | retrained_contrast:hit:with +0.0006 (lo90 0.0001) | 0 |
| ev | 6 | RETAINED |  | block_permutation:opener3:hgb +0.0003 (lo90 0.0001); block_permutation:opener3:temporal +0.0033 (lo90 0.0029); block_permutation:hit:temporal +0.0001 (lo90 0.0001); retrained_contrast:opener3:with +0.0008 (lo90 0.0007) | 0 |
| fac | 2 | RETAINED |  | block_permutation:opener3:hgb +0.0003 (lo90 0.0001) | 0 |
| fac.fx_f1 | 4 | RETAINED |  | retrained_contrast:opener3:with +0.0028 (lo90 0.0013); retrained_contrast:hit:with +0.0018 (lo90 0.0003) | 0 |
| fac.fx_f2 | 4 | RETAINED |  | block_permutation:opener3:hgb +0.0004 (lo90 0.0002) | 0 |
| fac.fx_f3 | 4 | RETAINED |  | block_permutation:opener3:hgb +0.0001 (lo90 0.0000) | 0 |
| fracdiff/log_close | 1 | RETAINED | P_raw+Cgroup | retrained_contrast:opener3:P_decided+Cgroup +0.0036 (lo90 0.0018); retrained_contrast:opener3:P_raw+Cgroup +0.0039 (lo90 0.0011); retrained_contrast:opener3:Cgroup +0.0037 (lo90 0.0027); retrained_contrast:opener3:C:fracdiff_d0.3_w256 +0.0037 (lo90 0.0027) | 0 |
| fx.eurgbp | 4 | RETAINED |  | block_permutation:hit:temporal +0.0008 (lo90 0.0000) | 0 |
| fx.gbpusd | 4 | RETAINED |  | retrained_contrast:hit:with +0.0001 (lo90 0.0000) | 0 |
| fx.usdchf | 4 | RETAINED |  | block_permutation:opener3:hgb +0.0002 (lo90 0.0000); block_permutation:opener3:temporal +0.0016 (lo90 0.0007) | 0 |
| fx.xauusd | 3 | RETAINED |  | retrained_contrast:opener3:with +0.0007 (lo90 0.0000) | 0 |
| gmm/logret_1+rv_24+log_range | 4 | RETAINED | P_decided+Cgroup | block_permutation:opener3:temporal +0.0011 (lo90 0.0001); retrained_contrast:opener3:C:posterior_entropy +0.0003 (lo90 0.0000); retrained_contrast:opener3:P_decided+Cgroup +0.1399 (lo90 0.1353); retrained_contrast:opener3:Cgroup +0.0067 (lo90 0.0059) | 0 |
| hilbert/logret_1 | 4 | RETAINED | S:phase | block_permutation:opener3:temporal +0.0102 (lo90 0.0082); retrained_contrast:opener3:C:phase_sin +0.0006 (lo90 0.0001); retrained_contrast:opener3:P_decided+Cgroup +0.0029 (lo90 0.0016); retrained_contrast:opener3:C:inst_freq +0.0019 (lo90 0.0007) | 0 |
| hmm/logret_1+rv_24+log_range | 5 | RETAINED | P_decided+Cgroup | block_permutation:opener3:temporal +0.0003 (lo90 0.0001); retrained_contrast:opener3:P_decided+Cgroup +0.0088 (lo90 0.0070); retrained_contrast:hit:Cgroup +0.0148 (lo90 0.0055); retrained_contrast:hit:P_decided +0.0139 (lo90 0.0088) | 0 |
| inv.momentum | 8 | RETAINED |  | block_permutation:opener3:hgb +0.0011 (lo90 0.0004); block_permutation:opener3:temporal +0.0028 (lo90 0.0024); retrained_contrast:opener3:with +0.0149 (lo90 0.0139); retrained_contrast:hit:with +0.0027 (lo90 0.0005) | 0 |
| inv.raw | 8 | RETAINED |  | block_permutation:opener3:hgb +0.7021 (lo90 0.6934); block_permutation:opener3:temporal +0.1292 (lo90 0.1132); retrained_contrast:opener3:with +0.0708 (lo90 0.0657); retrained_contrast:opener3:with +0.1688 (lo90 0.1662) | 0 |
| inv.trend | 6 | RETAINED |  | block_permutation:opener3:temporal +0.0011 (lo90 0.0001) | 0 |
| inv.volatility | 4 | RETAINED |  | retrained_contrast:opener3:with +0.0033 (lo90 0.0028) | 0 |
| kal.eurusd | 3 | RETAINED |  | block_permutation:opener3:temporal +0.0003 (lo90 0.0001) | 0 |
| kalman/log_close | 4 | RETAINED | S:approximation | block_permutation:opener3:temporal +0.0165 (lo90 0.0099); retrained_contrast:opener3:Cgroup +0.0074 (lo90 0.0055); retrained_contrast:opener3:S:approximation +0.1856 (lo90 0.1826); retrained_contrast:opener3:P_decided +0.0764 (lo90 0.0502) | 0 |
| px.atr_ratio | 1 | RETAINED |  | block_permutation:opener3:temporal +0.0003 (lo90 0.0001); retrained_contrast:hit:with +0.0001 (lo90 0.0000) | 0 |
| px.bar_body | 1 | RETAINED |  | retrained_contrast:opener3:with +0.0004 (lo90 0.0001) | 0 |
| px.close_loc | 7 | RETAINED |  | block_permutation:opener3:hgb +0.0689 (lo90 0.0624); block_permutation:opener3:temporal +0.0277 (lo90 0.0227); retrained_contrast:opener3:with +0.0064 (lo90 0.0036); retrained_contrast:opener3:with +0.0439 (lo90 0.0393) | 0 |
| px.elapsed_hours | 1 | RETAINED |  | block_permutation:opener3:temporal +0.0001 (lo90 0.0000) | 0 |
| px.er | 7 | RETAINED |  | block_permutation:opener3:hgb +0.0016 (lo90 0.0010); retrained_contrast:opener3:with +0.0033 (lo90 0.0023) | 0 |
| px.hours_since_session_open | 1 | RETAINED |  | block_permutation:hit:hgb +0.0005 (lo90 0.0002); block_permutation:hit:temporal +0.0027 (lo90 0.0015); retrained_contrast:opener3:with +0.0002 (lo90 0.0000) | 0 |
| px.logret | 7 | RETAINED |  | block_permutation:opener3:hgb +0.0184 (lo90 0.0159); block_permutation:opener3:temporal +0.0197 (lo90 0.0130); retrained_contrast:opener3:with +0.0015 (lo90 0.0014); retrained_contrast:opener3:with +0.0151 (lo90 0.0136) | 0 |
| px.range | 7 | RETAINED |  | block_permutation:opener3:temporal +0.0010 (lo90 0.0001); retrained_contrast:opener3:with +0.0072 (lo90 0.0062) | 0 |
| px.range_ratio | 1 | RETAINED |  | block_permutation:opener3:hgb +0.0005 (lo90 0.0003); block_permutation:opener3:temporal +0.0004 (lo90 0.0002); block_permutation:hit:temporal +0.0001 (lo90 0.0000) | 0 |
| px.rv | 7 | RETAINED |  | block_permutation:hit:temporal +0.0009 (lo90 0.0001) | 0 |
| strength/fx.eur_strength.ret_1 | 13 | RETAINED | C:A4 | retrained_contrast:opener3:C:A4 +0.0008 (lo90 0.0003); retrained_contrast:opener3:S:approximation +0.0008 (lo90 0.0003); retrained_contrast:hit:C:energy_D1 +0.0001 (lo90 0.0000); retrained_contrast:hit:C:energy_D4 +0.0002 (lo90 0.0000) | 0 |
| strength/fx.usd_strength.ret_1 | 13 | RETAINED | C:energy_D4 | retrained_contrast:opener3:C:energy_D4 +0.0006 (lo90 0.0001); retrained_contrast:hit:P_decided +0.0001 (lo90 0.0000) | 0 |
| wavelet/log_close | 9 | RETAINED | P_decided+Cgroup | block_permutation:opener3:hgb +0.0114 (lo90 0.0100); block_permutation:opener3:temporal +0.0022 (lo90 0.0011); retrained_contrast:opener3:P_decided+Cgroup +0.0036 (lo90 0.0027); retrained_contrast:opener3:P_raw+Cgroup +0.0040 (lo90 0.0039) | 0 |
| wavelet/logret_1 | 9 | RETAINED | P_decided+Cgroup | retrained_contrast:opener3:C:energy_D2 +0.0007 (lo90 0.0001); retrained_contrast:opener3:P_decided +0.0187 (lo90 0.0123); retrained_contrast:opener3:P_decided+Cgroup +0.0953 (lo90 0.0319); retrained_contrast:opener3:S:approximation +0.0187 (lo90 0.0123) | 0 |

## Encoding contrasts (same information, different encoding; gain vs the base encoding)

| family | variant | model | task | gain vs base | lo90 | decision |
|---|---|---|---|---|---|---|
| event_clock | expdecay4 | hgb | hit | +0.0000 | +0.0000 | KEEP_BASE_ENCODING |
| event_clock | expdecay4 | hgb | opener3 | +0.0000 | +0.0000 | KEEP_BASE_ENCODING |
| event_clock | expdecay4 | temporal | hit | +0.0006 | -0.0002 | KEEP_BASE_ENCODING |
| event_clock | expdecay4 | temporal | opener3 | -0.0008 | -0.0020 | KEEP_BASE_ENCODING |
| event_clock | window2h | hgb | hit | +0.0000 | -0.0000 | KEEP_BASE_ENCODING |
| event_clock | window2h | hgb | opener3 | -0.0003 | -0.0011 | KEEP_BASE_ENCODING |
| event_clock | window2h | temporal | hit | +0.0029 | -0.0055 | KEEP_BASE_ENCODING |
| event_clock | window2h | temporal | opener3 | -0.1047 | -0.1874 | KEEP_BASE_ENCODING |
| hod | onehot24 | hgb | hit | -0.0001 | -0.0001 | USE_sessions |
| hod | onehot24 | hgb | opener3 | +0.0002 | -0.0013 | USE_sessions |
| hod | onehot24 | temporal | hit | +0.0118 | -0.0034 | USE_sessions |
| hod | onehot24 | temporal | opener3 | +0.0177 | -0.0614 | USE_sessions |
| hod | sessions | hgb | hit | -0.0001 | -0.0001 | USE_sessions |
| hod | sessions | hgb | opener3 | +0.0001 | -0.0007 | USE_sessions |
| hod | sessions | temporal | hit | +0.0150 | +0.0002 | USE_sessions |
| hod | sessions | temporal | opener3 | +0.0205 | -0.1025 | USE_sessions |
| price_level | diff_plus_rel480 | hgb | hit | -0.0023 | -0.0056 | USE_diff_plus_rel480 |
| price_level | diff_plus_rel480 | hgb | opener3 | -0.0003 | -0.0005 | USE_diff_plus_rel480 |
| price_level | diff_plus_rel480 | temporal | hit | +0.0061 | +0.0027 | USE_diff_plus_rel480 |
| price_level | diff_plus_rel480 | temporal | opener3 | -0.0156 | -0.0875 | USE_diff_plus_rel480 |
| range_relative | rel24 | hgb | hit | +0.0005 | +0.0001 | USE_rel24 |
| range_relative | rel24 | hgb | opener3 | +0.0001 | -0.0017 | USE_rel24 |
| range_relative | rel24 | temporal | hit | -0.0007 | -0.0019 | USE_rel24 |
| range_relative | rel24 | temporal | opener3 | +0.0036 | -0.0051 | USE_rel24 |
| range_scale | log1p | temporal | hit | +0.0000 | -0.0000 | KEEP_BASE_ENCODING |
| range_scale | log1p | temporal | opener3 | +0.0000 | -0.0001 | KEEP_BASE_ENCODING |
| regime_gmm | onehot | hgb | hit | -0.0005 | -0.0013 | USE_onehot |
| regime_gmm | onehot | hgb | opener3 | +0.0003 | -0.0011 | USE_onehot |
| regime_gmm | onehot | temporal | hit | +0.0007 | +0.0002 | USE_onehot |
| regime_gmm | onehot | temporal | opener3 | -0.0001 | -0.0004 | USE_onehot |
| regime_hmm | onehot | hgb | hit | -0.0009 | -0.0025 | KEEP_BASE_ENCODING |
| regime_hmm | onehot | hgb | opener3 | +0.0001 | -0.0009 | KEEP_BASE_ENCODING |
| regime_hmm | onehot | temporal | hit | -0.0001 | -0.0003 | KEEP_BASE_ENCODING |
| regime_hmm | onehot | temporal | opener3 | -0.0000 | -0.0001 | KEEP_BASE_ENCODING |
