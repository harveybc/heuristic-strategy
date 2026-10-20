"""Minimal temporal trunk: exact parameter counts, causality, receptive field, flat-vector reshape."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.short_horizon.expert_guided import learn as L  # noqa: E402
from app.short_horizon.expert_guided import temporal as T  # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_parameter_counts_match_the_declared_design():
    e0 = T.TemporalPolicyNet(4, 13, 3, units=2)
    assert n_params(e0.trunk.branch) == 32 and n_params(e0.trunk.fuse) == 50 and n_params(e0.trunk.core) == 14
    assert n_params(e0.trunk) == 96 and n_params(e0) == 105
    e1 = T.TemporalPolicyNet(4, 13, 4, n_reg=2, n_state=9, units=2)
    assert n_params(e1) == 96 + 48 + 24 == 168
    assert n_params(T.TemporalPolicyNet(15, 13, 3, units=2)) == 120 + 182 + 14 + 9
    assert T.param_table(e0)[-1] == {"layer": "TOTAL", "type": "", "params": 105}
    assert n_params(L.PolicyNet(195, 3)) == 33539          # the flattened MLP comparator, unchanged


def _sequence(trunk, w):
    x = w.transpose(1, 2)
    x = torch.nn.functional.gelu(trunk.branch(x))
    x = torch.nn.functional.gelu(trunk.fuse(x))
    return torch.nn.functional.gelu(trunk.core(x))      # (B, u, T)


def test_trunk_is_causal_and_sees_the_whole_window():
    torch.manual_seed(0)
    tr = T.TemporalTrunk(4, units=2, dropout=0.0).eval()
    w = torch.randn(1, 13, 4)
    base = _sequence(tr, w)
    for j in range(13):
        w2 = w.clone()
        w2[0, j, :] += 5.0
        diff = (_sequence(tr, w2) - base).abs().sum(1)[0]
        assert torch.all(diff[:j] == 0), f"slot {j} leaked into earlier positions"
    assert T.receptive_field() == 15 >= 13
    w2 = w.clone()
    w2[0, 0, :] += 5.0                                    # the oldest slot reaches the current step
    assert not torch.allclose(tr(w2), tr(w))


def test_flat_vector_is_reshaped_back_to_time_by_features():
    torch.manual_seed(1)
    m = T.TemporalPolicyNet(4, 13, 4, n_reg=2, n_state=9, units=2, dropout=0.0).eval()
    w = torch.randn(5, 13, 4)
    s = torch.randn(5, 9)
    flat = torch.cat([w.reshape(5, -1), s], 1)
    z = torch.cat([m.trunk(w), s], 1)
    assert torch.allclose(m(flat)[0], m.cls(z))


def test_fit_accepts_the_temporal_factory():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 13, 4)).astype(np.float32)
    y = rng.integers(0, 3, 300)
    model, hist, info = L.fit(X[:200], y[:200], X[200:], y[200:], seed=0, max_epochs=3,
                              net=lambda: T.TemporalPolicyNet(4, 13, 3, units=2))
    assert info["architecture"] == "TemporalPolicyNet" and info["params"] == 105 and info["hidden"] is None
    P, _ = L.predict(model, X[200:])
    assert P.shape == (100, 3) and np.allclose(P.sum(1), 1, atol=1e-5)


def test_sb3_extractor_and_frozen_actor_match_the_live_actor():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (13 * 4 + 9,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(61, np.float32), {}

        def step(self, a):
            return np.random.randn(61).astype(np.float32), 0.0, True, False, {}

    m = SAC("MlpPolicy", Toy(), seed=0, device="cpu", learning_starts=10,
            policy_kwargs=dict(features_extractor_class=T.TemporalExtractor,
                               features_extractor_kwargs=dict(n_feat=4, win=13, n_state=9, units=2), net_arch=[2]))
    assert n_params(m.actor) == 96 + 42 == 138
    assert n_params(m.critic) == 96 + 2 * 33 == 162
    m.learn(64)
    fa = T.TorchActor(m.actor.state_dict(), 4, 13, 9, 2, [2])
    x = np.random.default_rng(3).standard_normal(61).astype(np.float32)
    m.actor.eval()
    with torch.no_grad():
        live = m.actor(torch.as_tensor(x)[None], deterministic=True).numpy()[0]
    assert np.allclose(fa(x), live, atol=1e-6)
