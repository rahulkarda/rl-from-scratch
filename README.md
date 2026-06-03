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
