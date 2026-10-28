#!/usr/bin/env python3
"""Keras plot_model diagrams of the minimal temporal models (E0-T, E1-T, E2-T actor and critic).

The torch models in ``temporal.py`` are the ones trained; this builds layer-for-layer Keras mirrors only to
draw them, and refuses to draw unless every mirrored layer has the same parameter count as the torch layer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V4 = ["RAW_BAR_LOG_RETURN", "cal_utc_session_us", "hurst_spectrum_width_w200", "px_close_loc_2"]
WIN, N_STATE, N_ACT = 13, 9, 3


def trunk(keras, L, x, u, tag):
    """Per-feature causal Conv1D branches -> concat -> dropout -> fusion (d=2) -> core (d=4) -> last step."""
    br = []
    for i, f in enumerate(V4):
        s = L.Lambda(lambda t, i=i: t[:, :, i:i + 1], output_shape=(WIN, 1), name=f"{tag}slice_{f}")(x)
        br.append(L.Conv1D(u, 3, dilation_rate=1, padding="causal", activation="gelu",
                           name=f"{tag}branch_{f}__k3_d1__{4 * u}p")(s))
    z = L.Concatenate(axis=-1, name=f"{tag}concat_branches")(br)
    z = L.Dropout(0.1, name=f"{tag}dropout_0.1")(z)
    z = L.Conv1D(u, 3, dilation_rate=2, padding="causal", activation="gelu",
                 name=f"{tag}fusion_across_features__k3_d2__{3 * 4 * u * u + u}p")(z)
    z = L.Conv1D(u, 3, dilation_rate=4, padding="causal", activation="gelu",
                 name=f"{tag}core__k3_d4__{3 * u * u + u}p")(z)
    return L.Lambda(lambda t: t[:, -1, :], output_shape=(u,), name=f"{tag}last_step_t__receptive_field_15")(z)


def build(u=2):
    import keras
    from keras import layers as L
    out = {}
    w = keras.Input((WIN, len(V4)), name="window_13x4__v4_features")
    e = trunk(keras, L, w, u, "")
    out["E0T"] = keras.Model(w, L.Dense(3, activation="softmax", name=f"head_none_valley_peak__{u * 3 + 3}p")(e),
                             name="E0T_student")
    w = keras.Input((WIN, len(V4)), name="window_13x4__v4_features")
    s = keras.Input((N_STATE,), name="agent_state_9")
    e = L.Concatenate(name="trunk_embedding_plus_state")([trunk(keras, L, w, u, ""), s])
    d = u + N_STATE
    a = L.Dense(4, activation="softmax", name=f"head_action_HOLD_LONG_SHORT_CLOSE__{d * 4 + 4}p")(e)
    r = L.Dense(2, activation="sigmoid", name=f"head_TP_SL_sigma_multiples__{d * 2 + 2}p")(e)
    r = L.Lambda(lambda t: t * 1.0, output_shape=(2,), name="bounded_TP_0.25_3__SL_0.5_3")(r)
    out["E1T"] = keras.Model([w, s], [a, r], name="E1T_policy")
    w = keras.Input((WIN, len(V4)), name="window_13x4__v4_features")
    s = keras.Input((N_STATE,), name="agent_state_9")
    f = L.Concatenate(name="features_extractor_out_11")([trunk(keras, L, w, u, ""), s])
    h = L.Dense(u, activation="relu", name=f"actor_hidden__{d * u + u}p")(f)
    mu = L.Dense(N_ACT, name=f"actor_mean_3__{u * 3 + 3}p")(h)
    ls = L.Dense(N_ACT, name=f"actor_log_std_3__{u * 3 + 3}p")(h)
    out["E2T_actor"] = keras.Model([w, s], [mu, ls], name="E2T_SAC_actor")
    w = keras.Input((WIN, len(V4)), name="window_13x4__v4_features")
    s = keras.Input((N_STATE,), name="agent_state_9")
    act = keras.Input((N_ACT,), name="action_3")
    f = L.Concatenate(name="features_extractor_out_11")([trunk(keras, L, w, u, ""), s])
    q = []
    for k in (1, 2):
        x = L.Concatenate(name=f"q{k}_features_plus_action_14")([f, act])
        x = L.Dense(u, activation="relu", name=f"q{k}_hidden__{(d + N_ACT) * u + u}p")(x)
        q.append(L.Dense(1, name=f"q{k}_value__{u + 1}p")(x))
    out["E2T_critic"] = keras.Model([w, s, act], q, name="E2T_SAC_critic_twin_Q")
    return out


def check(models, u):
    import torch  # noqa: F401
    from app.short_horizon.expert_guided import temporal as T
    ref = {"E0T": sum(p.numel() for p in T.TemporalPolicyNet(4, WIN, 3, units=u).parameters()),
           "E1T": sum(p.numel() for p in T.TemporalPolicyNet(4, WIN, 4, n_reg=2, n_state=N_STATE, units=u).parameters())}
    for k, v in ref.items():
        got = models[k].count_params()
        if got != v:
            raise SystemExit(f"PARAM_MISMATCH {k}: keras {got} vs torch {v}")
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", type=int, default=2)
    a = ap.parse_args()
    import keras
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = build(a.units)
    ref = check(models, a.units)
    for name, m in models.items():
        p = out / f"arquitectura_{name}_u{a.units}.png"
        keras.utils.plot_model(m, to_file=str(p), show_shapes=True, show_layer_names=True, show_layer_activations=True,
                               show_dtype=False, expand_nested=True, dpi=130, rankdir="TB")
        print(name, m.count_params(), "params ->", p)
    print("torch parity:", ref)


if __name__ == "__main__":
    main()
