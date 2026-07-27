import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay import ReplayBuffer, Transition

class SACAgent:
    """
    Soft Actor-Critic agent skeleton with basic update logic.

    Features:
      - Maintains actor, critic, and target networks
      - Stores transitions in ReplayBuffer
      - Performs Q updates and policy updates
      - Tracks alpha (entropy coefficient) but does not tune it yet
    """
    def __init__(self, obs_dim, act_dim, actor_critic_class, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_capacity=100000):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.ac = actor_critic_class(obs_dim, act_dim).to(self.device)
        self.ac_target = actor_critic_class(obs_dim, act_dim).to(self.device)
        self.ac_target.load_state_dict(self.ac.state_dict())
        self.actor_optimizer = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.ac.critic1.parameters()) + list(self.ac.critic2.parameters()), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def push_transition(self, state, action, reward, next_state, done):
        t = Transition(state, action, reward, next_state, done)
        self.buffer.push(t)

    def sample_batch(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t.action for t in batch]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array([t.reward for t in batch]), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([t.done for t in batch]), dtype=torch.float32).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        # Critic update
        with torch.no_grad():
            next_action, next_logp = self.ac.actor.sample(next_states)
            q1_target = self.ac_target.critic1(next_states, next_action)
            q2_target = self.ac_target.critic2(next_states, next_action)
            min_q_target = torch.minimum(q1_target, q2_target)
            target = rewards + self.gamma * (1 - dones) * (min_q_target - self.alpha * next_logp)
        q1 = self.ac.critic1(states, actions)
        q2 = self.ac.critic2(states, actions)
        critic_loss = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor update
        new_action, logp = self.ac.actor.sample(states)
        q1_pi = self.ac.critic1(states, new_action)
        q2_pi = self.ac.critic2(states, new_action)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update target
        for param, target_param in zip(self.ac.parameters(), self.ac_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
