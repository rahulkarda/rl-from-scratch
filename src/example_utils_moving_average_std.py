import numpy as np
from utils import moving_average, moving_std

if __name__ == "__main__":
    # Example: reward curve smoothing
    rewards = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    window = 3

    avg = moving_average(rewards, window)
    std = moving_std(rewards, window)

    print(f"Rewards:      {rewards}")
    print(f"Moving avg ({window}): {avg}")
    print(f"Moving std ({window}): {std}")

    # Quick plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        steps = np.arange(len(avg))
        plt.plot(steps, avg, label="Moving Avg")
        plt.fill_between(steps, avg - std, avg + std, alpha=0.2, label="Std band")
        plt.title(f"Moving Average and Std (window={window})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()
    except ImportError:
        pass
