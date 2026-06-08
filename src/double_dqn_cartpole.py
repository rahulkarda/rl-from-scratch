"""
Double DQN training loop for CartPole-v1: minimal end-to-end example.

Rationale:
- Implements Double DQN: online net selects action, target net evaluates value.
- Uses uniform replay, epsilon-greedy, and logging utilities from project.
- Designed for comparison with classic DQN and easy reward curve reproduction.

Usage:
    python double_dqn_cartpole.py
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
    online_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=10000)
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=10000)
    logger = Logger(log_dir="logs/double_dqn_cartpole")

    gamma = 0.99
    batch_size = 32
    min_buffer = 1000
    target_update_freq = 100
    num_episodes = 20  # tiny initial loop for test
    total_steps = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            q_values = online_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
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

            # Training step
            if len(buffer) >= min_buffer:
                batch = buffer.sample(batch_size)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.long)
                rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

                # Double DQN target:
                with torch.no_grad():
                    # Online net selects action
                    next_q_online = online_net(next_states)
                    next_actions = torch.argmax(next_q_online, dim=1)
                    # Target net evaluates value
                    next_q_target = target_net(next_states)
                    next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets = rewards + gamma * (1 - dones) * next_q_values

                q_pred = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(q_pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % target_update_freq == 0:
                soft_update(target_net, online_net, tau=1.0)  # hard update

        logger.log_episode_return(episode_reward, episode=ep)
        logger.log_scalars({
            "epsilon": explorer.epsilon(),
            "steps": total_steps,
            "buffer": len(buffer)
        }, step=total_steps)
        print(f"Episode {ep}: return={episode_reward:.2f}, epsilon={explorer.epsilon():.4f}, steps={total_steps}, buffer={len(buffer)}")

if __name__ == "__main__":
    main()
