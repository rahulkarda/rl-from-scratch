from explorer import EpsilonGreedyExplorer
import numpy as np

def test_epsilon_greedy():
    explorer = EpsilonGreedyExplorer(epsilon_start=1.0, epsilon_final=0.0, epsilon_decay=1)
    q_values = np.array([1.0, 2.0, 3.0])
    # With epsilon=1.0, should be random
    random_actions = set(explorer.select_action(q_values) for _ in range(20))
    assert random_actions == {0, 1, 2}
    explorer.steps_done = explorer.epsilon_decay  # set epsilon to final (0.0)
    # Should always pick argmax
    greedy_actions = [explorer.select_action(q_values) for _ in range(10)]
    assert all(a == 2 for a in greedy_actions)

def test_argmax_action():
    explorer = EpsilonGreedyExplorer()
    q_values = [0.5, 0.1, 7.3, 3.2]
    action = explorer.argmax_action(q_values)
    assert action == 2  # 7.3 is highest
    # Ties: lowest index
    q_values = [7.3, 7.3, 0.1]
    action = explorer.argmax_action(q_values)
    assert action == 0

def run():
    test_epsilon_greedy()
    test_argmax_action()
    print("EpsilonGreedyExplorer basic test passed.")

if __name__ == "__main__":
    run()
