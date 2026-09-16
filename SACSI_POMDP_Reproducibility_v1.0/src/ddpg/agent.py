"""Minimal DDPG for the locked 8-D irrigation environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sac_basic import ReplayBuffer


@dataclass(frozen=True)
class DDPGConfig:
    observation_dim: int = 8
    action_dim: int = 1
    hidden_dim: int = 64
    batch_size: int = 64
    warmup: int = 500
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.005
    exploration_noise_std: float = 0.5
    action_max: float = 5.0


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, config: DDPGConfig) -> None:
        super().__init__()
        self.network = mlp(config.observation_dim, config.action_dim, config.hidden_dim)
        nn.init.constant_(self.network[-1].bias, -1.5)
        self.action_scale = config.action_max / 2

    def forward(self, observation):
        return (torch.tanh(self.network(observation)) + 1) * self.action_scale


class Critic(nn.Module):
    def __init__(self, config: DDPGConfig) -> None:
        super().__init__()
        self.network = mlp(config.observation_dim + config.action_dim, 1, config.hidden_dim)

    def forward(self, observation, action):
        return self.network(torch.cat((observation, action), dim=-1))


class DDPGAgent:
    def __init__(self, config: DDPGConfig | None = None, device: str | None = None) -> None:
        self.config = config or DDPGConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor(self.config).to(self.device)
        self.critic = Critic(self.config).to(self.device)
        self.target_actor = Actor(self.config).to(self.device)
        self.target_critic = Critic(self.config).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def select_action(self, observation) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(tensor).cpu().numpy()[0]

    def update(self, replay: ReplayBuffer) -> dict[str, torch.Tensor]:
        cfg = self.config
        observations, actions, rewards, next_observations, dones = replay.sample(
            cfg.batch_size, self.device
        )
        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            target = rewards + cfg.gamma * (1 - dones) * self.target_critic(
                next_observations, next_actions
            )

        critic_loss = nn.functional.mse_loss(self.critic(observations, actions), target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(observations, self.actor(observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_network, network in (
                (self.target_actor, self.actor), (self.target_critic, self.critic)
            ):
                for target_parameter, parameter in zip(
                    target_network.parameters(), network.parameters()
                ):
                    target_parameter.mul_(1 - cfg.tau).add_(parameter, alpha=cfg.tau)
        return {"actor_loss": actor_loss.detach(), "critic_loss": critic_loss.detach()}

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.config),
            "metadata": metadata,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> tuple["DDPGAgent", dict]:
        state = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(DDPGConfig(**state["config"]), device)
        for name in ("actor", "critic", "target_actor", "target_critic"):
            getattr(agent, name).load_state_dict(state[name])
        agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
        return agent, state["metadata"]
