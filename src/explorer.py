import random
import numpy as np

class EpsilonGreedyExplorer:
    """
    Epsilon-greedy action selection with support for decay schedules.

    Usage:
        explorer = EpsilonGreedyExplorer(
            epsilon_start=1.0,
            epsilon_final=0.1,
            epsilon_decay=10000
        )
        action = explorer.select_action(q_values)
        explorer.step()
    """
    def __init__(self, epsilon_start: float = 1.0, epsilon_final: float = 0.1, epsilon_decay: int = 10000):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def epsilon(self) -> float:
        """
        Returns current epsilon value, linearly decayed between start and final.
        """
        decay_ratio = min(self.steps_done / self.epsilon_decay, 1.0)
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * (1.0 - decay_ratio)
        return float(eps)

    def select_action(self, q_values: np.ndarray | list) -> int:
        """
        Selects action using epsilon-greedy policy.
        With probability epsilon, returns a random action.
        Otherwise, returns the action with the highest Q-value (argmax).

        Args:
            q_values: Array-like (list or np.ndarray) of action values.

        Returns:
            action index (int)
        """
        eps = self.epsilon()
        num_actions = len(q_values)
        if num_actions == 0:
            raise ValueError("No actions provided to select_action.")
        rand = random.random()
        if rand < eps:
            return random.randrange(num_actions)
        # np.argmax always returns the lowest index in case of ties
        return int(np.argmax(q_values))

    def sample_action(self, batch_q_values: np.ndarray | list) -> list:
        """
        Select actions for a batch of Q-value vectors.
        Each action is chosen with epsilon-greedy per sample.

        Args:
            batch_q_values: array-like shape (batch_size, num_actions)

        Returns:
            List of action indices (int)
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

    def random_action(self, num_actions: int) -> int:
        """
        Selects a random action index from [0, num_actions-1].
        Useful for pure exploration or debugging.

        Args:
            num_actions: Total number of actions (int)

        Returns:
            action index (int)
        """
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        return random.randrange(num_actions)

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

    def get_epsilon_schedule(self, max_steps: int = None) -> np.ndarray:
        """
        Returns an array of epsilon values over steps for plotting/analysis.
        Args:
            max_steps: Number of steps to include (defaults to epsilon_decay)
        Returns:
            np.ndarray of epsilon values shape (max_steps,)
        """
        if max_steps is None:
            max_steps = self.epsilon_decay
        steps = np.arange(max_steps)
        decay_ratio = np.minimum(steps / self.epsilon_decay, 1.0)
        epsilons = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * (1.0 - decay_ratio)
        return epsilons
