# The main TD3 agent

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.td3.config import TD3Config
from agents.td3.networks import Actor, Critic
from agents.td3.buffer import ReplayBuffer
from agents.td3.utils import set_seeds, soft_update


class TD3Agent:
    # TD3 (Twin Delayed DDPG)
    # Two critics (take min target)

    def __init__(self, env, config=None, device=None, hidden=(256, 256)):
        self.cfg = config if config is not None else TD3Config()

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Environment spaces
        obs_space = env.observation_space
        act_space = env.action_space
        self.obs_dim = int(obs_space.shape[0])
        self.act_dim = int(act_space.shape[0])
        self.action_limit = float(np.asarray(act_space.high, dtype=np.float32)[0])

        # Seeds
        set_seeds(self.cfg.seed)

        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim, self.action_limit, hidden).to(self.device)
        self.actor_t = Actor(self.obs_dim, self.act_dim, self.action_limit, hidden).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(self.obs_dim, self.act_dim, hidden).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.act_dim, hidden).to(self.device)
        self.critic1_t = Critic(self.obs_dim, self.act_dim, hidden).to(self.device)
        self.critic2_t = Critic(self.obs_dim, self.act_dim, hidden).to(self.device)
        self.critic1_t.load_state_dict(self.critic1.state_dict())
        self.critic2_t.load_state_dict(self.critic2.state_dict())

        # Optimisers
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.opt_q1 = optim.Adam(self.critic1.parameters(), lr=self.cfg.critic_lr)
        self.opt_q2 = optim.Adam(self.critic2.parameters(), lr=self.cfg.critic_lr)

        # Loss and buffer
        self.mse = nn.MSELoss()
        self.replay = ReplayBuffer(self.cfg.buffer_size)

        # Counter
        self.update_steps = 0

    # Acting Stuff

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        # Get an action for a single observation

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)  # add batch dimension

        action = self.actor(obs_t).cpu().numpy()[0]
        if not deterministic:
            noise = np.random.normal(0.0, self.cfg.exploration_noise, size=self.act_dim)
            action = action + noise

        action = np.clip(action, -self.action_limit, self.action_limit)
        return action.astype(np.float32)

    def remember(self, s, a, r, ns, done):
        self.replay.add(s, a, r, ns, done)


    # Training Stuff

    def train_step(self):
        # Check if we have enough data

        if not self.replay.can_sample(self.cfg.batch_size):
            return None

        # Sample a batch
        s_np, a_np, r_np, ns_np, d_np = self.replay.sample(self.cfg.batch_size)

        # Tensors on device
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a_np, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r_np, dtype=torch.float32, device=self.device)   # (B,1)
        ns = torch.as_tensor(ns_np, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d_np, dtype=torch.float32, device=self.device)   # (B,1)

        # Target Q with policy smoothing
        with torch.no_grad():
            noise = torch.randn_like(a) * self.cfg.policy_noise
            noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)
            next_a = self.actor_t(ns) + noise
            next_a = torch.clamp(next_a, -self.action_limit, self.action_limit)

            tq1 = self.critic1_t(ns, next_a)
            tq2 = self.critic2_t(ns, next_a)
            tq_min = torch.min(tq1, tq2)

            y = r + self.cfg.gamma * (1.0 - d) * tq_min  # Bellman target

        # Update the critics
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss_q1 = self.mse(q1, y)
        loss_q2 = self.mse(q2, y)

        self.opt_q1.zero_grad()
        loss_q1.backward()
        self.opt_q1.step()

        self.opt_q2.zero_grad()
        loss_q2.backward()
        self.opt_q2.step()

        info = {"critic1_loss": float(loss_q1.detach().cpu().item()),
                "critic2_loss": float(loss_q2.detach().cpu().item())}

        # Delayed actor & target updates
        self.update_steps += 1
        if self.update_steps % self.cfg.policy_delay == 0:
            pred_actions = self.actor(s)
            actor_loss = -self.critic1(s, pred_actions).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # Soft update targets
            soft_update(self.actor, self.actor_t, self.cfg.tau)
            soft_update(self.critic1, self.critic1_t, self.cfg.tau)
            soft_update(self.critic2, self.critic2_t, self.cfg.tau)

            info["actor_loss"] = float(actor_loss.detach().cpu().item())

        return info

    # Save / Load

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(),   os.path.join(path, "actor.pt"))
        torch.save(self.critic1.state_dict(), os.path.join(path, "critic1.pt"))
        torch.save(self.critic2.state_dict(), os.path.join(path, "critic2.pt"))
        torch.save(self.actor_t.state_dict(), os.path.join(path, "actor_t.pt"))
        torch.save(self.critic1_t.state_dict(), os.path.join(path, "critic1_t.pt"))
        torch.save(self.critic2_t.state_dict(), os.path.join(path, "critic2_t.pt"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
        self.critic1.load_state_dict(torch.load(os.path.join(path, "critic1.pt"), map_location=self.device))
        self.critic2.load_state_dict(torch.load(os.path.join(path, "critic2.pt"), map_location=self.device))

        # Targets are optional
        p = os.path.join
        if os.path.exists(p(path, "actor_t.pt")):
            self.actor_t.load_state_dict(torch.load(p(path, "actor_t.pt"), map_location=self.device))
        if os.path.exists(p(path, "critic1_t.pt")):
            self.critic1_t.load_state_dict(torch.load(p(path, "critic1_t.pt"), map_location=self.device))
        if os.path.exists(p(path, "critic2_t.pt")):
            self.critic2_t.load_state_dict(torch.load(p(path, "critic2_t.pt"), map_location=self.device))
