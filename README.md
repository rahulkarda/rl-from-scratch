# rl-from-scratch

Implementing classic reinforcement learning algorithms from the ground up. No stable-baselines, no high-level wrappers — just PyTorch and Gym.

## Goals

- Understand each algorithm by writing it from scratch (not copying or using wrappers)
- Match published reward curves on standard envs (CartPole, Pendulum, etc.)
- Keep each algorithm in a single readable file (no spaghetti, easy to follow)
- Document hyperparameter choices and what actually mattered (not just copy defaults)
- Provide clear utilities for replay, exploration, logging, and seeding

## Status

Just starting. See [ROADMAP.md](ROADMAP.md).

## Setup

```bash
pip install -r requirements.txt
```

## Usage examples

- **ReplayBuffer**:
  ```python
  from replay import ReplayBuffer, Transition
  buf = ReplayBuffer(capacity=10000)
  buf.push(Transition(state, action, reward, next_state, done))
  batch = buf.sample(batch_size=32)
  buf.save('buffer.pkl')
  buf.load('buffer.pkl')
  ```

- **PrioritizedReplayBuffer**:
  ```python
  from prioritized_replay import PrioritizedReplayBuffer, Transition
  buf = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
  buf.push(Transition(state, action, reward, next_state, done), priority=1.0)
  batch, indices, weights = buf.sample(batch_size=32, beta=0.4)
  buf.update_priorities(indices, new_priorities)
  buf.save('buffer_prio.pkl')
  buf.load('buffer_prio.pkl')
  ```

- **EpsilonGreedyExplorer**:
  ```python
  from explorer import EpsilonGreedyExplorer
  explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=50000)
  action = explorer.select_action(q_values)
  explorer.step()
  explorer.save('explorer.json')
  explorer.load('explorer.json')
  ```

- **Logger**:
  ```python
  from logger import Logger
  logger = Logger(log_dir="logs/test")
  logger.log_scalar("loss", 0.123, step=10)
  logger.log_episode_return(42.0, episode=3)
  logger.log_scalars({"epsilon": 0.13, "reward": 17.5}, step=20)
  scalars = logger.read_scalars()
  returns = logger.read_episode_returns()
  ```

- **Seeding**:
  ```python
  from utils import set_seed, seed_everything
  set_seed(42)
  # Or with Gym env:
  seed_everything(42, env=env)
  ```
