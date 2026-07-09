import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob'])

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

def compute_returns(rewards, dones, gamma):
    returns = []
    R = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_eps):
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    return torch.mean(torch.min(unclipped, clipped))

def train_ppo(env_name="CartPole-v1", epochs=10, steps_per_epoch=2048, batch_size=64, gamma=0.99, clip_eps=0.2, lr=3e-4):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        state, _ = env.reset()
        traj = []
        ep_rewards = []
        for step in range(steps_per_epoch):
            action, log_prob, value = model.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            traj.append(Transition(state, action, reward, next_state, done, log_prob))
            state = next_state
            ep_rewards.append(reward)
            if done:
                state, _ = env.reset()
        # Prepare batch
        states = torch.tensor([t.state for t in traj], dtype=torch.float32)
        actions = torch.tensor([t.action for t in traj], dtype=torch.long)
        rewards = [t.reward for t in traj]
        dones = [t.done for t in traj]
        old_log_probs = torch.tensor([t.log_prob for t in traj], dtype=torch.float32)
        returns = torch.tensor(compute_returns(rewards, dones, gamma), dtype=torch.float32)
        with torch.no_grad():
            _, values = model.forward(states)
        advantages = returns - values.squeeze(-1)
        # PPO update
        for _ in range(4):  # 4 minibatch epochs
            idx = np.random.permutation(len(traj))
            for start in range(0, len(traj), batch_size):
                batch_idx = idx[start:start+batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                logits, batch_values = model.forward(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                batch_new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                loss = -ppo_clip_loss(batch_old_log_probs, batch_new_log_probs, batch_advantages, clip_eps)
                value_loss = torch.nn.functional.mse_loss(batch_values.squeeze(-1), returns[batch_idx])
                total_loss = loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        print(f"Epoch {epoch+1}: avg reward {np.mean(ep_rewards):.2f}")

if __name__ == "__main__":
    train_ppo()
