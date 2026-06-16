"""
Dueling DQN training loop for CartPole-v1: builds on classic DQN but uses dueling MLP architecture.

Rationale:
- Implements Dueling DQN (Wang et al. 2016) with separate value and advantage streams.
- Uses uniform replay, epsilon-greedy explorer, and logger utilities as in base DQN.
- No prioritized replay yet (see roadmap).

Usage:
    python dueling_dqn_cartpole.py

"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay import ReplayBuffer, Transition
from explorer import EpsilonGreedyExplorer
from logger import Logger
from utils import set_seed, soft_update

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)  # shape: (batch, 1)
        advantage = self.advantage_stream(x)  # shape: (batch, n_actions)
        # Dueling: Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

def main():
    env = gym.make("CartPole-v1")
    set_seed(42)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    dqn = DuelingDQN(obs_dim, n_actions)
    target_dqn = DuelingDQN(obs_dim, n_actions)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=10000)
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=50000)
    logger = Logger(log_dir="logs/dueling_dqn_cartpole")

    num_episodes = 20  # small run for initial test
    batch_size = 32
    gamma = 0.99
    target_update_tau = 0.01  # Polyak averaging for target net
    min_buffer_size = 500
    episode_rewards = []
    total_steps = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = dqn(obs_tensor).detach().numpy()[0]
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
            # Training
            if len(buffer) >= min_buffer_size:
                batch = buffer.sample(batch_size)
                states = np.stack([t.state for t in batch])
                actions = np.array([t.action for t in batch])
                rewards = np.array([t.reward for t in batch], dtype=np.float32)
                next_states = np.stack([t.next_state for t in batch])
                dones = np.array([t.done for t in batch], dtype=np.float32)

                states_tensor = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.int64)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32)

                q_pred = dqn(states_tensor)
                q_pred = q_pred.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_dqn(next_states_tensor)
                    q_next_max = q_next.max(dim=1)[0]
                    q_target = rewards_tensor + gamma * q_next_max * (1.0 - dones_tensor)
                loss = nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Polyak averaging
                soft_update(target_dqn, dqn, tau=target_update_tau)

        episode_rewards.append(episode_reward)
        logger.log_episode_return(episode_reward, episode=ep)
        logger.log_scalars({"epsilon": explorer.epsilon()}, step=total_steps)
        print(f"Episode {ep}: return={episode_reward:.2f}, epsilon={explorer.epsilon():.4f}, steps={total_steps}")

if __name__ == "__main__":
    main()
