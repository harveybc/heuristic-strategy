"""E2: SB3 2.9 SAC on the bar env, with and without hindsight-expert demonstrations (no sb3_contrib).

* ``SegmentEnv`` (gymnasium): one episode = one calendar segment (trading week) drawn at random inside the
  period; observation = flattened 13 x 15 admitted-core window + agent state (204 floats); action = the
  continuous contract of ``actions.py``; bars without an observation are stepped with HOLD inside the env;
  reward = 100 x change of mark-to-market equity / initial capital, NET of every cost (spread, slippage,
  commission, swap), UNSHAPED (no potential, no hindsight term). Capital restarts at 10,000 USD per episode;
  the episode ends at the segment end (forced flat makes the terminal state flat).
* ``DemoSAC``: SAC whose ``train`` mixes a fraction rho_t of every batch from a fixed demonstration buffer
  (DQfD-style demo replay, rho_t = rho0 x max(0, 1 - t / decay)), and adds to the actor loss a behaviour-cloning
  term on the demonstration rows gated by a Q-filter (Nair et al. 2018): lambda_t x mean over demo rows where
  min_i Q_i(s, a_demo) > min_i Q_i(s, mu(s)) of ||mu(s) - a_demo||^2, mu = the deterministic actor mean,
  lambda_t = lambda0 x max(0, 1 - t / decay). With rho0 = lambda0 = 0 it is exactly SB3's SAC update
  (the no-demonstration arm uses this same class with zeros, so both arms share one code path).
* ``DropoutExtractor``: Linear(204, 128) + ReLU + Dropout(0.1) (owner: dropout 0.1, no BatchNorm); the
  optimiser is AdamW with weight decay 1e-4 for actor and critic.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import nn
from torch.nn import functional as F

from . import actions as A
from . import data as DG
from .env import HOLD, BarEnv, EnvConfig

REWARD_SCALE = 100.0


class DropoutExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128, dropout: float = 0.1):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(nn.Linear(int(np.prod(observation_space.shape)), features_dim), nn.ReLU(),
                                 nn.Dropout(dropout))

    def forward(self, obs):
        return self.net(obs)


class SegmentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ctx, period: str, seed: int, cfg: EnvConfig = EnvConfig(level_policy="clip"),
                 sequential: bool = False):
        super().__init__()
        self.cx, self.cfg, self.sequential = ctx, cfg, sequential
        ds = ctx.ds
        a, b = (DG._utc_ns(x) for x in DG.PERIODS[period])
        segs = []
        for s in ds.cal.segments:
            if s.market_open.value >= a and s.market_close.value <= b:
                lo = int(np.searchsorted(ctx.v, s.market_open.value))
                hi = int(np.searchsorted(ctx.v, s.market_close.value))
                if hi - lo >= 24 and (ctx.obs_ok[ctx.rob[lo:hi][ctx.rob[lo:hi] >= 0]]).sum() >= 12:
                    segs.append(s)
        self.segs = segs
        self.rng = np.random.default_rng(seed)
        self.k = -1
        n = ctx.Xw.shape[1] + 9
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    def _obs(self):
        r = self.cx.rob[self.b]
        return np.r_[self.cx.Xw[r], self.env.state(self.b)].astype(np.float32)

    def _advance(self):
        """Step HOLD through bars without an observation; returns the accumulated reward, done flag."""
        rew = 0.0
        while self.b < self.env.last:
            r = self.cx.rob[self.b]
            if r >= 0 and self.cx.obs_ok[r]:
                return rew, False
            before = A.mark_to_market(self.env, self.b)
            self.env.step(self.b, HOLD)
            self.b += 1
            rew += A.mark_to_market(self.env, self.b) - before
        return rew, True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.sequential:
            self.k = (self.k + 1) % len(self.segs)
            s = self.segs[self.k]
        else:
            s = self.segs[int(self.rng.integers(len(self.segs)))]
        self.env = BarEnv(self.cx.ds.bars, self.cx.ds.cal, self.cx.ds.sigma, self.cx.ds.sigma_rule, s.market_open,
                          s.market_close, self.cfg)
        self.b = self.env.first
        self._advance()
        return self._obs(), {}

    def position(self):
        return 0 if self.env.pos is None else self.env.pos.direction

    def step(self, action):
        a, tp, sl = A.decode(np.asarray(action, float), self.position())
        before = A.mark_to_market(self.env, self.b)
        info = self.env.step(self.b, a, tp, sl)
        self.b += 1
        rew = A.mark_to_market(self.env, self.b) - before
        r2, done = self._advance()
        rew += r2
        done = done or self.b >= self.env.last
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(REWARD_SCALE * rew / self.cfg.capital), bool(done), False, info


class DemoSAC(SAC):
    def __init__(self, *args, demo=None, demo_frac0: float = 0.0, bc_lambda0: float = 0.0, demo_decay: int = 50_000,
                 **kw):
        super().__init__(*args, **kw)
        self.demo_frac0, self.bc_lambda0, self.demo_decay = demo_frac0, bc_lambda0, demo_decay
        self.demo = None
        if demo is not None:
            self.demo = {k: th.as_tensor(v, device=self.device, dtype=th.float32) for k, v in demo.items()}
        self.demo_log = []

    def _schedule(self):
        f = max(0.0, 1.0 - self._n_updates / self.demo_decay)
        return self.demo_frac0 * f, self.bc_lambda0 * f

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        for gradient_step in range(gradient_steps):
            frac, lam = self._schedule()
            n_demo = int(round(batch_size * frac)) if self.demo is not None else 0
            rb = self.replay_buffer.sample(batch_size - n_demo, env=self._vec_normalize_env)
            if n_demo:
                idx = th.randint(0, self.demo["obs"].shape[0], (n_demo,), device=self.device)
                d = self.demo
                rb = ReplayBufferSamples(th.cat([rb.observations, d["obs"][idx]]), th.cat([rb.actions, d["act"][idx]]),
                                         th.cat([rb.next_observations, d["next_obs"][idx]]),
                                         th.cat([rb.dones, d["done"][idx]]), th.cat([rb.rewards, d["rew"][idx]]),
                                         None if rb.discounts is None else th.cat([rb.discounts, th.full((n_demo, 1), self.gamma,
                                                                                                      device=self.device)]))
            discounts = rb.discounts if rb.discounts is not None else self.gamma
            actions_pi, log_prob = self.actor.action_log_prob(rb.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            else:
                ent_coef = self.ent_coef_tensor
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(rb.next_observations)
                next_q = th.min(th.cat(self.critic_target(rb.next_observations, next_actions), dim=1), dim=1, keepdim=True)[0]
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = rb.rewards + (1 - rb.dones) * discounts * next_q
            current_q = self.critic(rb.observations, rb.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            q_pi = th.min(th.cat(self.critic(rb.observations, actions_pi), dim=1), dim=1, keepdim=True)[0]
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            bc_loss, kept = th.zeros((), device=self.device), 0
            if n_demo and lam > 0:
                od, ad = rb.observations[-n_demo:], rb.actions[-n_demo:]
                mu = self.actor(od, deterministic=True)
                with th.no_grad():
                    q_demo = th.min(th.cat(self.critic(od, ad), dim=1), dim=1, keepdim=True)[0]
                    q_mu = th.min(th.cat(self.critic(od, mu.detach()), dim=1), dim=1, keepdim=True)[0]
                    mask = (q_demo > q_mu).float()
                kept = int(mask.sum().item())
                bc_loss = (((mu - ad) ** 2).sum(1, keepdim=True) * mask).sum() / max(n_demo, 1)
                actor_loss = actor_loss + lam * bc_loss
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                from stable_baselines3.common.utils import polyak_update
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if self._n_updates % 1000 == 0:
                self.demo_log.append({"update": int(self._n_updates), "demo_frac": frac, "bc_lambda": lam, "n_demo": n_demo,
                                      "qfilter_kept": kept, "critic_loss": float(critic_loss.item()),
                                      "actor_loss": float(actor_loss.item()), "bc_loss": float(bc_loss.item()),
                                      "ent_coef": float(ent_coef.item())})
            self._n_updates += 1
