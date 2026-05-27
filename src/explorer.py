import random
import numpy as np

class EpsilonGreedyExplorer:
    """
    Epsilon-greedy action selection with support for decay schedules.

    Usage:
        explorer = EpsilonGreedyExplorer(
            epsilon_start=1.0,
            epsilon_final=0.05,
            epsilon_decay=10000
        )
        action = explorer.select_action(q_values)
        explorer.step()
    """
    def __init__(self, epsilon_start: float = 1.0, epsilon_final: float = 0.05, epsilon_decay: int = 10000):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def epsilon(self) -> float:
        """
        Returns current epsilon value, linearly decayed.
        """
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            max(0.0, 1.0 - self.steps_done / self.epsilon_decay)
        return float(eps)

    def select_action(self, q_values: np.ndarray | list) -> int:
        """
        Selects action according to epsilon-greedy policy.

        Args:
            q_values: Array-like (list or np.ndarray) of action values.

        Returns:
            action index (int)
        """
        eps = self.epsilon()
        if random.random() < eps:
            return random.randrange(len(q_values))
        else:
            return int(np.argmax(q_values))

    def step(self) -> None:
        """
        Increments the internal step counter (for decay).
        """
        self.steps_done += 1

    def reset(self) -> None:
        """
        Resets the step counter to zero.
        """
        self.steps_done = 0
