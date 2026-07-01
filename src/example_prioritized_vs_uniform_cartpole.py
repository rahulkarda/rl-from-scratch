import gymnasium as gym
import numpy as np
import torch
from prioritized_replay import PrioritizedReplayBuffer, Transition as PrioTransition
from replay import ReplayBuffer, Transition as UniformTransition
from explorer import EpsilonGreedyExplorer
from logger import Logger
from utils import seed_everything, moving_average

# --- Simple MLP Q-network ---
class QNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Training loop ---
def train_agent(env_fn, replay_type="uniform", episodes=200, batch_size=64):
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    seed_everything(42, env)
    qnet = QNet(obs_dim, act_dim)
    tgt_qnet = QNet(obs_dim, act_dim)
    tgt_qnet.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=1e-3)
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=50000)
    logger = Logger(log_dir=f"logs/prio_vs_uniform_{replay_type}")
    gamma = 0.99
    update_freq = 1
    target_update_freq = 100
    if replay_type == "uniform":
        buffer = ReplayBuffer(capacity=10000)
        Transition = UniformTransition
    elif replay_type == "prio":
        buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        Transition = PrioTransition
    else:
        raise ValueError("replay_type must be 'uniform' or 'prio'")
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            q_vals = qnet(obs_tensor).detach().numpy()
            action = explorer.select_action(q_vals)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            t = Transition(obs, action, reward, next_obs, done)
            if replay_type == "prio":
                buffer.push(t, priority=1.0)  # Initial priority
            else:
                buffer.push(t)
            obs = next_obs
            explorer.step()
            if len(buffer) >= batch_size:
                if replay_type == "prio":
                    batch, idxs, weights = buffer.sample(batch_size, beta=0.4)
                else:
                    batch = buffer.sample(batch_size)
                states = torch.as_tensor(np.array([b.state for b in batch]), dtype=torch.float32)
                actions = torch.as_tensor(np.array([b.action for b in batch]), dtype=torch.long)
                rewards = torch.as_tensor(np.array([b.reward for b in batch]), dtype=torch.float32)
                next_states = torch.as_tensor(np.array([b.next_state for b in batch]), dtype=torch.float32)
                dones = torch.as_tensor(np.array([b.done for b in batch]), dtype=torch.float32)
                q = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    tgt_q = tgt_qnet(next_states).max(1)[0]
                    target = rewards + gamma * tgt_q * (1 - dones)
                if replay_type == "prio":
                    loss = ((q - target).pow(2) * torch.as_tensor(weights, dtype=torch.float32)).mean()
                    td_errors = torch.abs(q.detach() - target.detach()).cpu().numpy()
                    buffer.update_priorities(idxs, td_errors + 1e-6)
                else:
                    loss = torch.nn.functional.mse_loss(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.log_scalar("loss", float(loss.item()), step=ep * 1000 + steps)
        logger.log_episode_return(ep_reward, episode=ep)
        returns.append(ep_reward)
        if ep % 10 == 0:
            print(f"Ep {ep}: return {ep_reward:.2f}")
        if ep % target_update_freq == 0:
            tgt_qnet.load_state_dict(qnet.state_dict())
    env.close()
    return returns

if __name__ == "__main__":
    print("Training with uniform replay...")
    uniform_returns = train_agent(lambda: gym.make("CartPole-v1"), replay_type="uniform", episodes=100)
    print("Training with prioritized replay...")
    prio_returns = train_agent(lambda: gym.make("CartPole-v1"), replay_type="prio", episodes=100)
    import matplotlib.pyplot as plt
    plt.plot(moving_average(uniform_returns, 10), label="Uniform")
    plt.plot(moving_average(prio_returns, 10), label="Prioritized")
    plt.xlabel("Episode")
    plt.ylabel("Moving avg return (window=10)")
    plt.legend()
    plt.title("Prioritized vs Uniform Replay: CartPole")
    plt.show()
