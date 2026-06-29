import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prioritized_replay import PrioritizedReplayBuffer, Transition
from explorer import EpsilonGreedyExplorer
from logger import Logger
from utils import set_seed

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)


def train_prioritized_dqn(
    env_name="CartPole-v1",
    seed=42,
    episodes=20,
    capacity=50000,
    batch_size=32,
    gamma=0.99,
    lr=1e-3,
    alpha=0.6,
    beta_start=0.4,
    beta_frames=10000,
    min_buffer=1000,
    update_target_every=1000,
    log_dir=None
):
    set_seed(seed)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha)
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=20000)
    logger = Logger(log_dir=log_dir or "logs/prio_dqn")

    steps = 0
    beta = beta_start
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = np.array(obs, dtype=np.float32)
        done = False
        total_reward = 0
        while not done:
            q_values = q_net(torch.from_numpy(state).float().unsqueeze(0)).detach().numpy()[0]
            action = explorer.select_action(q_values)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated
            t = Transition(state, action, reward, next_state, done)
            max_priority = buffer.max_priority() if len(buffer) > 0 else 1.0
            buffer.push(t, priority=max_priority)
            state = next_state
            total_reward += reward
            explorer.step()
            steps += 1

            # Training
            if len(buffer) >= min_buffer:
                beta = min(1.0, beta_start + steps * (1.0 - beta_start) / beta_frames)
                batch, indices, weights = buffer.sample(batch_size, beta=beta)
                states = torch.tensor([b.state for b in batch], dtype=torch.float32)
                actions = torch.tensor([b.action for b in batch], dtype=torch.int64)
                rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
                next_states = torch.tensor([b.next_state for b in batch], dtype=torch.float32)
                dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
                weights = torch.tensor(weights, dtype=torch.float32)

                q_vals = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_vals = target_net(next_states).max(1)[0]
                target = rewards + gamma * next_q_vals * (1.0 - dones)
                td_error = target - q_vals
                loss = (weights * td_error.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update priorities
                prio = td_error.abs().detach().cpu().numpy() + 1e-6
                buffer.update_priorities(indices, prio)

                logger.log_scalar("loss", float(loss.item()), steps)

            if steps % update_target_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        logger.log_episode_return(total_reward, ep)
        returns.append(total_reward)
        print(f"Episode {ep}: return={total_reward:.2f}")

    env.close()
    return returns

if __name__ == "__main__":
    train_prioritized_dqn(episodes=5)
