import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import set_seed

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        logits = self.forward(torch.from_numpy(state).float().unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs[0, action]


def run_episode(env, policy):
    states = []
    actions = []
    rewards = []
    probs = []
    state, _ = env.reset()
    done = False
    while not done:
        action, prob = policy.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        probs.append(prob)
        state = next_state
    return states, actions, rewards, probs


def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_reinforce(env_name="CartPole-v1", seed=42, episodes=200, gamma=0.99, lr=1e-2):
    set_seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    episode_returns = []
    for episode in range(episodes):
        states, actions, rewards, probs = run_episode(env, policy)
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack([torch.log(p) for p in probs])
        loss = -torch.sum(log_probs * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Return={episode_return:.2f}")
    env.close()
    return episode_returns

if __name__ == "__main__":
    train_reinforce()
