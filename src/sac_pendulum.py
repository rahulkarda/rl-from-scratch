"""
SAC agent for Pendulum-v1: skeleton implementation.

Rationale:
- Implements Soft Actor-Critic with automatic entropy tuning.
- Designed for continuous control. Uses minimal utilities from replay, logger, explorer, utils.
- Does not use stable-baselines or wrappers; everything explicit.
- In-progress: see ROADMAP.
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay import ReplayBuffer, Transition
from logger import Logger
from utils import set_seed

class ActorCritic(nn.Module):
    """
    Combined actor-critic network for SAC.
    - Actor: outputs mean and log_std for Gaussian policy
    - Critic: two Q networks (Q1, Q2) for double Q learning
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Linear(hidden_dim, act_dim)
        # Critic networks
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_actor(self, obs):
        """
        Forward pass for actor: returns mean and log_std.
        Args:
            obs: torch.Tensor (batch_size, obs_dim)
        Returns:
            mean: torch.Tensor (batch_size, act_dim)
            log_std: torch.Tensor (batch_size, act_dim)
        """
        x = self.actor_net(obs)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # for numerical stability
        return mean, log_std

    def forward_q1(self, obs, act):
        """
        Forward pass for Q1 network.
        Args:
            obs: torch.Tensor (batch_size, obs_dim)
            act: torch.Tensor (batch_size, act_dim)
        Returns:
            q1: torch.Tensor (batch_size, 1)
        """
        x = torch.cat([obs, act], dim=-1)
        return self.q1_net(x)

    def forward_q2(self, obs, act):
        """
        Forward pass for Q2 network.
        Args:
            obs: torch.Tensor (batch_size, obs_dim)
            act: torch.Tensor (batch_size, act_dim)
        Returns:
            q2: torch.Tensor (batch_size, 1)
        """
        x = torch.cat([obs, act], dim=-1)
        return self.q2_net(x)

# TODO: implement SAC agent class, actor/critic update, entropy tuning, training loop

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    set_seed(42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    net = ActorCritic(obs_dim, act_dim)
    print(f"ActorCritic initialized: obs_dim={obs_dim}, act_dim={act_dim}")
