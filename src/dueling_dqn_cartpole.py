import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
import random

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):  # Increased from 128 to 256
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)           # shape: (batch, 1)
        advantage = self.advantage_stream(x)   # shape: (batch, action_dim)
        # Dueling Q: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def push(self, t: Transition):
        self.buffer.append(t)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([b.state for b in batch], dtype=np.float32)
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.array([b.next_state for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim,
                 hidden_dim=256,  # Increased from 128 to 256
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 target_update=500):
        self.q_net = DuelingDQN(state_dim, action_dim, hidden_dim)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.action_dim = action_dim
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state)
        return int(torch.argmax(q).item())

    def update(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a)
        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_value = next_q_values.max(1)[0]
            target = rewards + self.gamma * next_q_value * (1 - dones)
        loss = nn.functional.mse_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


def train_dueling_dqn_cartpole(
        episodes=400,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=30000,
        hidden_dim=256  # Increased from 128 to 256
    ):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DuelingDQNAgent(state_dim, action_dim, hidden_dim=hidden_dim)
    buffer = ReplayBuffer(50000)
    epsilon = epsilon_start
    epsilon_step = (epsilon_start - epsilon_final) / epsilon_decay
    returns = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(Transition(state, action, reward, next_state, done))
            loss = agent.update(buffer)
            state = next_state
            total_reward += reward
            # Decay epsilon
            if epsilon > epsilon_final:
                epsilon -= epsilon_step
        returns.append(total_reward)
        if (ep+1) % 10 == 0:
            avg = np.mean(returns[-10:])
            print(f"Episode {ep+1}, avg return {avg:.2f}, epsilon {epsilon:.3f}")
    env.close()
    return returns

if __name__ == "__main__":
    train_dueling_dqn_cartpole()