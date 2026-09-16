"""Minimal TD3 for the locked 8-D irrigation environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ddpg.agent import Actor, Critic
from sac_basic import ReplayBuffer


@dataclass(frozen=True)
class TD3Config:
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
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    action_max: float = 5.0


class TD3Agent:
    def __init__(self, config: TD3Config | None = None, device: str | None = None) -> None:
        self.config = config or TD3Config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor(self.config).to(self.device)
        self.critic1 = Critic(self.config).to(self.device)
        self.critic2 = Critic(self.config).to(self.device)
        self.target_actor = Actor(self.config).to(self.device)
        self.target_critic1 = Critic(self.config).to(self.device)
        self.target_critic2 = Critic(self.config).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.config.critic_lr)
        self.total_updates = 0
        self.last_actor_loss = torch.zeros((), device=self.device)

    def select_action(self, observation) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(tensor).cpu().numpy()[0]

    def update(self, replay: ReplayBuffer) -> dict[str, torch.Tensor | bool]:
        cfg = self.config
        observations, actions, rewards, next_observations, dones = replay.sample(
            cfg.batch_size, self.device
        )
        with torch.no_grad():
            noise = (torch.randn_like(actions) * cfg.target_noise_std).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            next_actions = (self.target_actor(next_observations) + noise).clamp(0, cfg.action_max)
            target_q = torch.minimum(
                self.target_critic1(next_observations, next_actions),
                self.target_critic2(next_observations, next_actions),
            )
            target = rewards + cfg.gamma * (1 - dones) * target_q

        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 10.0
        )
        self.critic_optimizer.step()

        self.total_updates += 1
        actor_updated = self.total_updates % cfg.policy_delay == 0
        if actor_updated:
            actor_loss = -self.critic1(observations, self.actor(observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            self.last_actor_loss = actor_loss.detach()
            self._soft_update_targets()

        return {
            "actor_loss": self.last_actor_loss,
            "critic_loss": critic_loss.detach(),
            "actor_updated": actor_updated,
            "target_noise_abs_max": noise.abs().max().detach(),
        }

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for target_network, network in (
                (self.target_actor, self.actor),
                (self.target_critic1, self.critic1),
                (self.target_critic2, self.critic2),
            ):
                for target_parameter, parameter in zip(
                    target_network.parameters(), network.parameters()
                ):
                    target_parameter.mul_(1 - self.config.tau).add_(
                        parameter, alpha=self.config.tau
                    )

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "config": asdict(self.config),
            "metadata": metadata,
            "total_updates": self.total_updates,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> tuple["TD3Agent", dict]:
        state = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(TD3Config(**state["config"]), device)
        for name in (
            "actor", "critic1", "critic2", "target_actor", "target_critic1", "target_critic2"
        ):
            getattr(agent, name).load_state_dict(state[name])
        agent.total_updates = state["total_updates"]
        return agent, state["metadata"]
