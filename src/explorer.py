import random
import numpy as np

class EpsilonGreedyExplorer:
    """
    Epsilon-greedy action selection with decay schedules.

    Rationale:
        Used in value-based RL for balancing exploration/exploitation. Linear decay. Greedy/random modes. Batch support.
    """
    def __init__(self, epsilon_start: float = 1.0, epsilon_final: float = 0.1, epsilon_decay: int = 10000):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def epsilon(self) -> float:
        """Current epsilon decayed linearly."""
        decay_ratio = min(self.steps_done / self.epsilon_decay, 1.0)
        return float(self.epsilon_final + (self.epsilon_start - self.epsilon_final) * (1.0 - decay_ratio))

    def select_action(self, q_values: np.ndarray | list) -> int:
        """
        Epsilon-greedy action selection.
        Returns random action with probability epsilon, else argmax.
        """
        eps = self.epsilon()
        num_actions = len(q_values)
        if num_actions == 0:
            raise ValueError("No actions provided to select_action.")
        if random.random() < eps:
            return random.randrange(num_actions)
        return int(np.argmax(q_values))

    def argmax_action(self, q_values: np.ndarray | list) -> int:
        """
        Greedy action selection (always argmax).
        """
        num_actions = len(q_values)
        if num_actions == 0:
            raise ValueError("No actions provided to argmax_action.")
        return int(np.argmax(q_values))

    def random_action(self, num_actions: int) -> int:
        """
        Random action index from [0, num_actions-1].
        """
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        return random.randrange(num_actions)

    def sample_action(self, batch_q_values: np.ndarray | list) -> list:
        """
        Batch epsilon-greedy action selection for array shape (batch_size, num_actions).
        """
        eps = self.epsilon()
        batch_q_values = np.array(batch_q_values)
        batch_size, num_actions = batch_q_values.shape
        actions = []
        for i in range(batch_size):
            if random.random() < eps:
                actions.append(random.randrange(num_actions))
            else:
                actions.append(int(np.argmax(batch_q_values[i])))
        return actions

    def step(self) -> None:
        """
        Increment step counter for epsilon decay.
        """
        self.steps_done += 1

    def reset(self) -> None:
        """
        Reset step counter to zero.
        """
        self.steps_done = 0

    def get_epsilon_schedule(self, max_steps: int = None) -> np.ndarray:
        """
        Array of epsilon values over steps (for plotting).
        """
        if max_steps is None:
            max_steps = self.epsilon_decay
        steps = np.arange(max_steps)
        decay_ratio = np.minimum(steps / self.epsilon_decay, 1.0)
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * (1.0 - decay_ratio)
