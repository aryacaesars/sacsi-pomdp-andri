"""Minimal PyTorch Soft Actor-Critic implementation for one continuous action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class SACConfig:
    observation_dim: int = 8
    action_dim: int = 1
    hidden_dim: int = 64
    batch_size: int = 64
    warmup: int = 500
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    alpha_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.05
    target_entropy: float = -3.0
    actor_mean_bias: float = -1.5
    action_max: float = 5.0


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, seed: int = 11) -> None:
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, observation, action, reward, next_observation, done) -> None:
        i = self.position
        self.observations[i] = observation
        self.actions[i] = np.asarray(action).reshape(1)
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.integers(0, self.size, size=batch_size)
        arrays = (self.observations, self.actions, self.rewards, self.next_observations, self.dones)
        return tuple(torch.as_tensor(array[indices], device=device) for array in arrays)

    def __len__(self) -> int:
        return self.size


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.body = mlp(config.observation_dim, config.hidden_dim, config.hidden_dim)[:-1]
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)
        nn.init.constant_(self.mean.bias, config.actor_mean_bias)
        self.action_scale = config.action_max / 2

    def forward(self, observation, deterministic: bool = False, with_log_prob: bool = True):
        hidden = self.body(observation)
        mean, log_std = self.mean(hidden), self.log_std(hidden).clamp(-20, 2)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(raw)
        action = (squashed + 1) * self.action_scale
        log_prob = None
        if with_log_prob:
            log_prob = distribution.log_prob(raw) - torch.log(
                self.action_scale * (1 - squashed.pow(2)) + 1e-6
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.network = mlp(config.observation_dim + config.action_dim, 1, config.hidden_dim)

    def forward(self, observation, action):
        return self.network(torch.cat((observation, action), dim=-1))


class SACAgent:
    def __init__(self, config: SACConfig | None = None, device: str | None = None) -> None:
        self.config = config or SACConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor(self.config).to(self.device)
        self.critic1 = Critic(self.config).to(self.device)
        self.critic2 = Critic(self.config).to(self.device)
        self.target1 = Critic(self.config).to(self.device)
        self.target2 = Critic(self.config).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.config.critic_lr)
        self.log_alpha = torch.tensor(
            np.log(self.config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.target_entropy = self.config.target_entropy

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, observation, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(tensor, deterministic=deterministic, with_log_prob=False)
        return action.cpu().numpy()[0]

    def update(self, replay: ReplayBuffer) -> dict[str, float]:
        cfg = self.config
        observations, actions, rewards, next_observations, dones = replay.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_observations)
            target_q = torch.minimum(
                self.target1(next_observations, next_actions),
                self.target2(next_observations, next_actions),
            ) - self.alpha.detach() * next_log_prob
            target = rewards + cfg.gamma * (1 - dones) * target_q

        q1, q2 = self.critic1(observations, actions), self.critic2(observations, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 10.0)
        self.critic_optimizer.step()

        new_actions, log_prob = self.actor(observations)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(
            self.critic1(observations, new_actions), self.critic2(observations, new_actions)
        )).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
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
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
        }

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": asdict(self.config),
            "metadata": metadata,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> tuple["SACAgent", dict]:
        state = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(SACConfig(**state["config"]), device)
        for name in ("actor", "critic1", "critic2", "target1", "target2"):
            getattr(agent, name).load_state_dict(state[name])
        agent.log_alpha.data.copy_(state["log_alpha"].to(agent.device))
        return agent, state.get("metadata", {})
