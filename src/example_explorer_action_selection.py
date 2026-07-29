from explorer import EpsilonGreedyExplorer
import numpy as np

# Minimal example: epsilon-greedy and greedy action selection
if __name__ == "__main__":
    # Dummy Q-values for 3 actions
    q_values = np.array([1.2, 0.5, 3.1])
    explorer = EpsilonGreedyExplorer(epsilon_start=0.5, epsilon_final=0.1, epsilon_decay=100)

    # Epsilon-greedy selection (may pick random or argmax)
    action = explorer.select_action(q_values)
    print(f"Epsilon-greedy selected action: {action} (epsilon={explorer.epsilon():.2f})")

    # Greedy selection (always argmax)
    greedy_action = explorer.argmax_action(q_values)
    print(f"Greedy (argmax) action: {greedy_action}")

    # Random action
    random_action = explorer.random_action(num_actions=3)
    print(f"Random action: {random_action}")

    # Batch epsilon-greedy selection
    batch_q = np.array([[0.1, 1.0, 0.2], [1.3, 0.4, 0.2], [0.5, 2.1, 0.7]])
    batch_actions = explorer.sample_action(batch_q)
    print(f"Batch epsilon-greedy actions: {batch_actions}")