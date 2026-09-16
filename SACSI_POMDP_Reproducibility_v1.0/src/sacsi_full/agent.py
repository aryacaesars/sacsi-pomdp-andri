"""Residual multi-context SACSI actor and twin critics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sac_basic.agent import Actor, Critic, SACConfig


@dataclass(frozen=True)
class SACSIConfig(SACConfig):
    sequence_length: int = 24
    lstm_hidden_dim: int = 64
    forecast_dim: int = 3
    forecast_hidden_dim: int = 32


def _basic_config(config: SACSIConfig) -> SACConfig:
    values = asdict(config)
    return SACConfig(**{key: values[key] for key in SACConfig.__dataclass_fields__})


class SACSIReplayBuffer:
    def __init__(self, capacity: int, config: SACSIConfig, seed: int = 11) -> None:
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.current = np.zeros((capacity, config.observation_dim), np.float32)
        self.history = np.zeros((capacity, config.sequence_length, config.observation_dim), np.float32)
        self.forecast = np.zeros((capacity, config.forecast_dim), np.float32)
        self.actions = np.zeros((capacity, config.action_dim), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.next_current = np.zeros_like(self.current)
        self.next_history = np.zeros_like(self.history)
        self.next_forecast = np.zeros_like(self.forecast)
        self.dones = np.zeros((capacity, 1), np.float32)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.position
        self.current[i], self.history[i], self.forecast[i] = state
        self.actions[i], self.rewards[i] = np.asarray(action).reshape(-1), reward
        self.next_current[i], self.next_history[i], self.next_forecast[i] = next_state
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        indices = self.rng.integers(0, self.size, batch_size)
        arrays = (
            self.current, self.history, self.forecast, self.actions, self.rewards,
            self.next_current, self.next_history, self.next_forecast, self.dones,
        )
        return tuple(torch.as_tensor(array[indices], device=device) for array in arrays)

    def __len__(self):
        return self.size


class SACSIActor(nn.Module):
    def __init__(self, config: SACSIConfig) -> None:
        super().__init__()
        self.base = Actor(_basic_config(config))
        self.history_lstm = nn.LSTM(config.observation_dim, config.lstm_hidden_dim, batch_first=True)
        self.history_mean = nn.Linear(config.lstm_hidden_dim, config.action_dim)
        self.forecast_encoder = nn.Sequential(
            nn.Linear(config.forecast_dim, config.forecast_hidden_dim), nn.ReLU(),
        )
        self.forecast_mean = nn.Linear(config.forecast_hidden_dim, config.action_dim)
        for layer in (self.history_mean, self.forecast_mean):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.action_scale = config.action_max / 2

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, current, history, forecast, deterministic=False, with_log_prob=True):
        base_hidden = self.base.body(current)
        _, (history_hidden, _) = self.history_lstm(history)
        mean = (
            self.base.mean(base_hidden)
            + self.history_mean(history_hidden[-1])
            + self.forecast_mean(self.forecast_encoder(forecast))
        )
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


class SACSICritic(nn.Module):
    def __init__(self, config: SACSIConfig) -> None:
        super().__init__()
        self.base = Critic(_basic_config(config))
        self.history_lstm = nn.LSTM(config.observation_dim, config.lstm_hidden_dim, batch_first=True)
        self.forecast_encoder = nn.Sequential(
            nn.Linear(config.forecast_dim, config.forecast_hidden_dim), nn.ReLU(),
        )
        context_dim = config.lstm_hidden_dim + config.forecast_hidden_dim + config.action_dim
        self.residual = nn.Sequential(
            nn.Linear(context_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, current, history, forecast, action):
        _, (history_hidden, _) = self.history_lstm(history)
        forecast_hidden = self.forecast_encoder(forecast)
        context = torch.cat((history_hidden[-1], forecast_hidden, action), -1)
        return self.base(current, action) + self.residual(context)


class SACSIRecurrentAgent:
    def __init__(self, config: SACSIConfig | None = None, device: str | None = None) -> None:
        self.config = config or SACSIConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = SACSIActor(self.config).to(self.device)
        self.critic1 = SACSICritic(self.config).to(self.device)
        self.critic2 = SACSICritic(self.config).to(self.device)
        self.target1 = SACSICritic(self.config).to(self.device)
        self.target2 = SACSICritic(self.config).to(self.device)
        self.log_alpha = torch.tensor(
            np.log(self.config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.target_entropy = self.config.target_entropy

    def warm_start(self, checkpoint: Path) -> None:
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.actor.base.load_state_dict(state["actor"])
        self.critic1.base.load_state_dict(state["critic1"])
        self.critic2.base.load_state_dict(state["critic2"])
        for network in (self.actor, self.critic1, self.critic2):
            network.freeze_base()
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.freeze_base()
        self.target2.freeze_base()
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=self.config.actor_lr
        )
        critic_parameters = [
            p for network in (self.critic1, self.critic2) for p in network.parameters() if p.requires_grad
        ]
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
            for value in state
        ]
        with torch.inference_mode():
            action, _ = self.actor(*tensors, deterministic, False)
        return action.cpu().numpy()[0]

    def update(self, replay: SACSIReplayBuffer):
        cfg = self.config
        current, history, forecast, actions, rewards, next_current, next_history, next_forecast, dones = (
            replay.sample(cfg.batch_size, self.device)
        )
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_current, next_history, next_forecast)
            target_q = torch.minimum(
                self.target1(next_current, next_history, next_forecast, next_actions),
                self.target2(next_current, next_history, next_forecast, next_actions),
            ) - self.alpha.detach() * next_log_prob
            target = rewards + cfg.gamma * (1 - dones) * target_q
        q1 = self.critic1(current, history, forecast, actions)
        q2 = self.critic2(current, history, forecast, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for network in (self.critic1, self.critic2) for p in network.parameters() if p.requires_grad], 10
        )
        self.critic_optimizer.step()
        new_actions, log_prob = self.actor(current, history, forecast)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(
            self.critic1(current, history, forecast, new_actions),
            self.critic2(current, history, forecast, new_actions),
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

    def context_norms(self):
        def norm(module):
            return float(torch.sqrt(sum(p.detach().pow(2).sum() for p in module.parameters())).cpu())
        return {
            "history_residual_norm": norm(self.actor.history_mean),
            "forecast_residual_norm": norm(self.actor.forecast_mean),
        }

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(), "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(), "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "config": asdict(self.config), "metadata": metadata,
        }, path)
