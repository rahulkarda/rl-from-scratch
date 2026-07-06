import numpy as np
from utils import compute_reward_stats, compute_quantiles

# --- Minimal example: reward stats and quantiles ---
if __name__ == "__main__":
    rewards = [10, 12, 9, 16, 8, 14, 15]
    # Compute stats
    stats = compute_reward_stats(rewards)
    print("Reward stats:", stats)
    # Compute quantiles
    quantiles = compute_quantiles(rewards, qs=[0.0, 0.25, 0.5, 0.75, 1.0])
    print("Reward quantiles:", quantiles)
    # Edge case: empty input
    empty_stats = compute_reward_stats([])
    empty_quantiles = compute_quantiles([], qs=[0.5])
    print("Empty stats:", empty_stats)
    print("Empty quantile:", empty_quantiles)
