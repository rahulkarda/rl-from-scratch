from utils import compute_reward_stats, compute_quantiles

# Example reward returns for an RL agent
returns = [10, 15, 22, 9, 17, 30, 11, 12, 21]

stats = compute_reward_stats(returns)
print("Reward stats:")
for k, v in stats.items():
    print(f"  {k}: {v}")

quantiles = compute_quantiles(returns, qs=[0.0, 0.5, 0.9, 1.0])
print("Quantiles:")
for q, val in quantiles.items():
    print(f"  q={q:.2f}: {val}")
