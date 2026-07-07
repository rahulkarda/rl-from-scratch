import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import namedtuple
from utils import set_seed

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared(x)
        policy_probs = self.policy(x)
        value = self.value(x)
        return policy_probs, value.squeeze(-1)

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.net = ActorCriticNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy_probs, _ = self.net(state)
        dist = torch.distributions.Categorical(policy_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        return returns

    def update(self, transitions, last_value):
        states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.int64)
        rewards = [t.reward for t in transitions]
        dones = [t.done for t in transitions]
        log_probs = []
        values = []
        for state in states:
            policy_probs, value = self.net(state.unsqueeze(0))
            dist = torch.distributions.Categorical(policy_probs)
            action = actions[len(log_probs)]
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(0))
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        returns = self.compute_returns(rewards, dones, last_value)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    set_seed(42)
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2CAgent(state_dim, action_dim)
    num_episodes = 10  # Minimal test run
    for episode in range(num_episodes):
        state, _ = env.reset()
        transitions = []
        done = False
        ep_reward = 0.0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transitions.append(
                Transition(state, action, reward, next_state, done)
            )
            state = next_state
            ep_reward += reward
        # Bootstrap last value
        last_value = 0.0
        if not transitions[-1].done:
            last_state = torch.tensor(transitions[-1].next_state, dtype=torch.float32).unsqueeze(0)
            _, last_value = agent.net(last_state)
            last_value = last_value.item()
        agent.update(transitions, last_value)
        print(f"Episode {episode}: reward = {ep_reward}")

if __name__ == "__main__":
    main()
