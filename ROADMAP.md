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
- [ ] Prioritized replay

## Phase 3: policy-gradient
- [ ] REINFORCE on CartPole
- [ ] A2C with shared trunk
- [ ] PPO clipped objective
- [ ] GAE for advantage estimation

## Phase 4: continuous control
- [ ] DDPG on Pendulum
- [ ] SAC with automatic entropy tuning

## Phase 5: comparison
- [ ] Reward-curve plots across algorithms on the same env
- [ ] Wall-clock vs sample-efficiency comparison
- [ ] Notebook write-up
