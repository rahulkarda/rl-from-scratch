from explorer import EpsilonGreedyExplorer
import numpy as np
import matplotlib.pyplot as plt

# Minimal example: visualize epsilon decay schedule

def main():
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=1000)
    epsilons = []
    for _ in range(1200):
        epsilons.append(explorer.epsilon())
        explorer.step()
    # Optionally plot
    plt.plot(epsilons, label="epsilon")
    plt.axhline(0.05, color="red", ls="--", label="final epsilon")
    plt.xlabel("Steps")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay (Explorer)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
