import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from utils import set_seed, soft_update

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action
    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(ns),
            np.array(d, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
    def select_action(self, state, noise_std=0.2):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy()[0]
        noise = np.random.normal(0, noise_std, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        return action
    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions).squeeze(-1)
            q_target = rewards + self.gamma * (1 - dones) * q_next
        q_val = self.critic(states, actions).squeeze(-1)
        critic_loss = nn.MSELoss()(q_val, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Polyak averaging
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    set_seed(42)
    env.reset(seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPGAgent(state_dim, action_dim, max_action)
    buffer = ReplayBuffer()
    num_episodes = 5  # Minimal run for skeleton
    batch_size = 64
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            agent.train(buffer, batch_size)
            episode_reward += reward
            state = next_state
            if done:
                break
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Return = {episode_reward:.2f}")
    env.close()
    print("DDPG skeleton run complete.")
