import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from replay import ReplayBuffer, Transition
from utils import set_seed

# --- Networks ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        h = self.hidden(x)
        mu = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mu

# --- SAC Agent ---
class SACAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.005, alpha=0.2,
                 lr=3e-4, hidden_dim=256, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        # Networks
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dim).to(device)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dim).to(device)
        self.q1_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(device)
        self.q2_target = MLP(obs_dim + act_dim, 1, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # Optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)
        # Entropy tuning (optional, not yet implemented)
        # self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        # self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            with torch.no_grad():
                mu, _ = self.policy.forward(obs)
                action = torch.tanh(mu).cpu().numpy()[0]
            return action
        else:
            with torch.no_grad():
                action, _, _ = self.policy.sample(obs)
                return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        # Sample batch
        batch = replay_buffer.sample(batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([t.action for t in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.stack([t.reward for t in batch]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.stack([t.done for t in batch]), dtype=torch.float32, device=self.device).unsqueeze(-1)

        # --- Q targets ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_action], dim=1))
            q2_next = self.q2_target(torch.cat([next_states, next_action], dim=1))
            q_next = torch.min(q1_next, q2_next)
            target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_prob)

        # --- Q losses ---
        q1_pred = self.q1(torch.cat([states, actions], dim=1))
        q2_pred = self.q2(torch.cat([states, actions], dim=1))
        q1_loss = nn.MSELoss()(q1_pred, target)
        q2_loss = nn.MSELoss()(q2_pred, target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # --- Policy loss ---
        new_action, log_prob, mu = self.policy.sample(states)
        q1_new = self.q1(torch.cat([states, new_action], dim=1))
        q2_new = self.q2(torch.cat([states, new_action], dim=1))
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- Target update ---
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'q1_loss': float(q1_loss.item()),
            'q2_loss': float(q2_loss.item()),
            'policy_loss': float(policy_loss.item()),
            'alpha': float(self.alpha)
        }

# --- Minimal training loop skeleton ---
def main():
    env = gym.make('Pendulum-v1')
    set_seed(42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, device='cpu')
    replay_buffer = ReplayBuffer(capacity=100000)
    obs, _ = env.reset()
    total_steps = 0
    episode_return = 0
    for episode in range(1):  # Placeholder: 1 episode for skeleton
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            # Rescale action to env's range
            low = env.action_space.low[0]
            high = env.action_space.high[0]
            scaled_action = low + (action + 1) * 0.5 * (high - low)
            next_obs, reward, terminated, truncated, _ = env.step([scaled_action])
            done = terminated or truncated
            replay_buffer.push(Transition(obs, action, reward, next_obs, done))
            obs = next_obs
            episode_return += reward
            total_steps += 1
            if len(replay_buffer.buffer) > 256:
                agent.update(replay_buffer, batch_size=256)
        print(f"Episode return: {episode_return:.2f}")

if __name__ == "__main__":
    main()
