import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from src.utils import set_seed

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        x = self.trunk(state)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=1e-3, entropy_coef=0.005):
        self.net = ActorCriticNet(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action, dist.log_prob(torch.tensor(action)), dist.entropy()
    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        return returns
    def update(self, states, actions, log_probs, rewards, dones, values, entropies, last_value):
        returns = self.compute_returns(rewards, dones, last_value)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = policy_loss + value_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), entropy_loss.item()

def train_a2c_cartpole(seed=42, episodes=500, rollout_steps=5):
    set_seed(seed)
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2CAgent(state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=1e-3, entropy_coef=0.005)
    episode_returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        states, actions, log_probs, rewards, dones, values, entropies = [], [], [], [], [], [], []
        while not done:
            for _ in range(rollout_steps):
                logits, value = agent.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                entropy = dist.entropy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                finished = terminated or truncated
                states.append(torch.tensor(state, dtype=torch.float32))
                actions.append(torch.tensor(action))
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(finished))
                values.append(value.squeeze())
                entropies.append(entropy)
                total_reward += reward
                state = next_state
                if finished:
                    done = True
                    break
            if done:
                last_value = 0.0
            else:
                _, last_value = agent.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                last_value = last_value.item()
            agent.update(states, actions, log_probs, rewards, dones, values, entropies, last_value)
            states, actions, log_probs, rewards, dones, values, entropies = [], [], [], [], [], [], []
        episode_returns.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg_ret = np.mean(episode_returns[-10:])
            print(f"Episode {episode+1}: Return={total_reward:.1f} Avg(10)={avg_ret:.1f}")
    env.close()
    return episode_returns

if __name__ == "__main__":
    returns = train_a2c_cartpole(seed=42, episodes=100)
