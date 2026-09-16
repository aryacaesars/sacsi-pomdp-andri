"""Residual Recurrent Warm-Start SAC with sequence-aware replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sac_basic.agent import Actor, Critic, SACConfig


@dataclass(frozen=True)
class RecurrentSACConfig(SACConfig):
    sequence_length: int = 24
    lstm_hidden_dim: int = 64


class RecurrentReplayBuffer:
    def __init__(self, capacity: int, config: RecurrentSACConfig, seed: int = 11) -> None:
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        shape = (capacity, config.observation_dim)
        sequence_shape = (capacity, config.sequence_length, config.observation_dim)
        self.current = np.zeros(shape, np.float32)
        self.history = np.zeros(sequence_shape, np.float32)
        self.actions = np.zeros((capacity, config.action_dim), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.next_current = np.zeros(shape, np.float32)
        self.next_history = np.zeros(sequence_shape, np.float32)
        self.dones = np.zeros((capacity, 1), np.float32)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.position
        self.current[i], self.history[i] = state
        self.actions[i] = np.asarray(action).reshape(-1)
        self.rewards[i], self.next_current[i], self.next_history[i], self.dones[i] = (
            reward, next_state[0], next_state[1], done
        )
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.integers(0, self.size, batch_size)
        arrays = (
            self.current, self.history, self.actions, self.rewards,
            self.next_current, self.next_history, self.dones,
        )
        return tuple(torch.as_tensor(array[indices], device=device) for array in arrays)

    def __len__(self) -> int:
        return self.size


class ResidualActor(nn.Module):
    def __init__(self, config: RecurrentSACConfig) -> None:
        super().__init__()
        self.base = Actor(SACConfig(**{
            key: value for key, value in asdict(config).items() if key in SACConfig.__dataclass_fields__
        }))
        self.lstm = nn.LSTM(config.observation_dim, config.lstm_hidden_dim, batch_first=True)
        self.residual_mean = nn.Linear(config.lstm_hidden_dim, config.action_dim)
        nn.init.zeros_(self.residual_mean.weight)
        nn.init.zeros_(self.residual_mean.bias)
        self.action_scale = config.action_max / 2

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, current, history, deterministic=False, with_log_prob=True):
        base_hidden = self.base.body(current)
        _, (hidden, _) = self.lstm(history)
        mean = self.base.mean(base_hidden) + self.residual_mean(hidden[-1])
        log_std = self.base.log_std(base_hidden).clamp(-20, 2)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(raw)
        action = (squashed + 1) * self.action_scale
        log_prob = None
        if with_log_prob:
            log_prob = distribution.log_prob(raw) - torch.log(
                self.action_scale * (1 - squashed.pow(2)) + 1e-6
            )
            log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class ResidualCritic(nn.Module):
    def __init__(self, config: RecurrentSACConfig) -> None:
        super().__init__()
        self.base = Critic(SACConfig(**{
            key: value for key, value in asdict(config).items() if key in SACConfig.__dataclass_fields__
        }))
        self.lstm = nn.LSTM(config.observation_dim, config.lstm_hidden_dim, batch_first=True)
        self.residual = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim + config.action_dim, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, current, history, action):
        _, (hidden, _) = self.lstm(history)
        residual = self.residual(torch.cat((hidden[-1], action), -1))
        return self.base(current, action) + residual


class RecurrentSACAgent:
    def __init__(self, config: RecurrentSACConfig | None = None, device: str | None = None) -> None:
        self.config = config or RecurrentSACConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = ResidualActor(self.config).to(self.device)
        self.critic1 = ResidualCritic(self.config).to(self.device)
        self.critic2 = ResidualCritic(self.config).to(self.device)
        self.target1 = ResidualCritic(self.config).to(self.device)
        self.target2 = ResidualCritic(self.config).to(self.device)
        self.log_alpha = torch.tensor(
            np.log(self.config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.target_entropy = self.config.target_entropy

    def warm_start(self, checkpoint: Path) -> None:
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.actor.base.load_state_dict(state["actor"])
        self.critic1.base.load_state_dict(state["critic1"])
        self.critic2.base.load_state_dict(state["critic2"])
        self.actor.freeze_base()
        self.critic1.freeze_base()
        self.critic2.freeze_base()
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.freeze_base()
        self.target2.freeze_base()
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=self.config.actor_lr
        )
        critics = [p for network in (self.critic1, self.critic2) for p in network.parameters() if p.requires_grad]
        self.critic_optimizer = torch.optim.Adam(critics, lr=self.config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        current, history = state
        current = torch.as_tensor(current, dtype=torch.float32, device=self.device).unsqueeze(0)
        history = torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action, _ = self.actor(current, history, deterministic, False)
        return action.cpu().numpy()[0]

    def update(self, replay: RecurrentReplayBuffer):
        cfg = self.config
        current, history, actions, rewards, next_current, next_history, dones = replay.sample(
            cfg.batch_size, self.device
        )
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_current, next_history)
            target_q = torch.minimum(
                self.target1(next_current, next_history, next_actions),
                self.target2(next_current, next_history, next_actions),
            ) - self.alpha.detach() * next_log_prob
            target = rewards + cfg.gamma * (1 - dones) * target_q
        q1 = self.critic1(current, history, actions)
        q2 = self.critic2(current, history, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for network in (self.critic1, self.critic2) for p in network.parameters() if p.requires_grad], 10
        )
        self.critic_optimizer.step()

        new_actions, log_prob = self.actor(current, history)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(
            self.critic1(current, history, new_actions), self.critic2(current, history, new_actions)
        )).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_([p for p in self.actor.parameters() if p.requires_grad], 10)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_network, network in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_parameter, parameter in zip(target_network.parameters(), network.parameters()):
                    target_parameter.mul_(1 - cfg.tau).add_(parameter, alpha=cfg.tau)
        return {
            "actor_loss": actor_loss.detach(), "critic_loss": critic_loss.detach(),
            "alpha_loss": alpha_loss.detach(), "alpha": self.alpha.detach(),
        }

    def residual_norm(self) -> float:
        parameters = list(self.actor.residual_mean.parameters())
        return float(torch.sqrt(sum(parameter.detach().pow(2).sum() for parameter in parameters)).cpu())

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(), "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(), "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "config": asdict(self.config), "metadata": metadata,
        }, path)
