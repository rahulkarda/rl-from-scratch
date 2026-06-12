"""
DQN training loop for CartPole-v1: end-to-end example using minimal RL utilities.

Rationale:
- Implements classic DQN with a simple MLP, uniform replay, epsilon-greedy exploration, and basic logging.
- Designed to match published reward curves and serve as a readable reference for value-based RL.
- No wrappers, no stable-baselines: every piece is explicit for clarity and reproducibility.
- Uses the provided ReplayBuffer, EpsilonGreedyExplorer, Logger, and set_seed utilities.

Usage:
    python dqn_cartpole.py

This file is intentionally minimal for didactic purposes. See README for project goals.
"""
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
    # Tweak: slower epsilon decay for CartPole (was 10_000)
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=50000)
    logger = Logger(log_dir="logs/dqn_cartpole")

    num_episodes = 20  # tiny for initial loop test
    episode_rewards = []
    total_steps = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            q_values = dqn(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
            action = explorer.select_action(q_values)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done
            ))
            episode_reward += reward
            obs = next_obs
            explorer.step()
            total_steps += 1
        episode_rewards.append(episode_reward)
        logger.log_episode_return(episode_reward, episode=ep)
        logger.log_scalars({"epsilon": explorer.epsilon()}, step=total_steps)
        print(f"Episode {ep}: return={episode_reward:.2f}, epsilon={explorer.epsilon():.4f}, steps={total_steps}")

if __name__ == "__main__":
    main()
