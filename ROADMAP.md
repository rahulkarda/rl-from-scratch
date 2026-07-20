# Roadmap

## Phase 1: foundations
- [x] Replay buffer (uniform)
- [x] Epsilon-greedy explorer with decay schedules
- [x] Logger that writes scalar metrics + episode returns to disk
- [x] Seed control utility

## Phase 2: value-based
- [x] DQN on CartPole
- [x] Double DQN
- [x] Dueling DQN
- [x] Prioritized replay  [in progress]
    - [x] Buffer skeleton with push, sample, save/load
    - [x] Priority update method (update_priorities)
    - [x] Integration in agent
    - [x] Comparison with uniform replay

## Phase 3: policy-gradient
- [x] REINFORCE on CartPole
- [x] A2C with shared trunk
- [x] PPO clipped objective
- [x] GAE for advantage estimation

## Phase 4: continuous control
- [x] DDPG on Pendulum
- [ ] SAC with automatic entropy tuning  [in progress]

## Phase 5: comparison
- [ ] Reward-curve plots across algorithms on the same env
- [ ] Wall-clock vs sample-efficiency comparison
- [ ] Notebook write-up
