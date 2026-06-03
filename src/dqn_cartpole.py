import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay import ReplayBuffer, Transition
from explorer import EpsilonGreedyExplorer
from logger import Logger
from utils import set_seed

class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def main():
    env = gym.make("CartPole-v1")
    set_seed(42)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    dqn = DQN(obs_dim, n_actions)
    target_dqn = DQN(obs_dim, n_actions)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=10000)
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=10000)
    logger = Logger(log_dir="logs/dqn_cartpole")
    # TODO: training loop, loss, sampling, update, logging

if __name__ == "__main__":
    main()
