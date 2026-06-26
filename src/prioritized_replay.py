import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, List, Tuple
import pickle
import numpy as np

@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for RL: stores transitions with priorities, enables prioritized sampling, and provides simple serialization.
    Skeleton implementation.

    Usage:
        buf = PrioritizedReplayBuffer(capacity=500_000, alpha=0.6)
        buf.push(Transition(...), priority=1.0)
        batch, indices, weights = buf.sample(batch_size=32, beta=0.4)
        buf.update_priorities(indices, new_priorities)
        buf.save('buffer_prio.pkl')
        buf.load('buffer_prio.pkl')
        buf.clear()
        recent = buf.sample_recent(10)
        all_transitions = buf.export_to_list()
    """
    def __init__(self, capacity: int = 500_000, alpha: float = 0.6):
        """
        Initialize the prioritized replay buffer with a fixed capacity and prioritization parameter.

        Args:
            capacity (int): Maximum number of transitions to store.
            alpha (float): Priority exponent (0 = uniform, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)

    def push(self, t: Transition, priority: float = 1.0) -> None:
        """
        Add a transition with its priority to the buffer.
        Oldest is discarded if capacity is reached.

        Args:
            t (Transition): Transition to store.
            priority (float): Transition priority.
        """
        self.buffer.append(t)
        self.priorities.append(priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], List[int], np.ndarray]:
        """
        Prioritized sampling: sample a batch of transitions according to their priorities.
        Returns the sampled transitions, their indices, and importance-sampling weights.

        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): Importance-sampling exponent (0 = no correction, 1 = full correction).

        Returns:
            batch (List[Transition]): Sampled transitions.
            indices (List[int]): Indices in buffer for updating priorities.
            weights (np.ndarray): IS weights for sampled transitions.

        Raises:
            ValueError: If batch_size > number of transitions in buffer.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        priorities = np.array(self.priorities, dtype=np.float32)
        if priorities.size == 0:
            priorities = np.ones(len(self.buffer), dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        # Compute IS weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        return batch, indices.tolist(), weights

    def update_priorities(self, indices: List[int], new_priorities: List[float]) -> None:
        """
        Update priorities for transitions at the given indices.

        Args:
            indices (List[int]): Indices of transitions to update.
            new_priorities (List[float]): New priority values.
        """
        if len(indices) != len(new_priorities):
            raise ValueError("Length of indices and new_priorities must match.")
        priorities_list = list(self.priorities)
        for idx, prio in zip(indices, new_priorities):
            if idx < 0 or idx >= len(priorities_list):
                continue  # skip invalid index
            priorities_list[idx] = prio
        self.priorities = deque(priorities_list, maxlen=self.capacity)

    def sample_recent(self, batch_size: int) -> List[Transition]:
        """
        Sample the most recent N transitions from the buffer (not random).
        Useful for debugging, visualization, or on-policy algorithms.

        Args:
            batch_size (int): Number of transitions to return.
        Returns:
            List[Transition]: List of the latest transitions (ordered: oldest to newest).

        Raises:
            ValueError: If batch_size > number of transitions in buffer.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample_recent batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        return list(self.buffer)[-batch_size:]

    def export_to_list(self) -> List[Transition]:
        """
        Export all transitions in the buffer as a list.
        Useful for analysis, conversion, or integration with external tools.

        Returns:
            List[Transition]: List containing all transitions in order (oldest to newest).
        """
        return list(self.buffer)

    def __len__(self) -> int:
        """
        Returns the current number of transitions stored.
        """
        return len(self.buffer)

    def save(self, path: str) -> None:
        """
        Save buffer contents to a file using pickle.
        Stores as a list of (transition_dict, priority) tuples.

        Args:
            path (str): File path to save buffer.
        """
        with open(path, 'wb') as f:
            items = [(asdict(t), p) for t, p in zip(self.buffer, self.priorities)]
            pickle.dump(items, f)

    def load(self, path: str) -> None:
        """
        Load buffer contents from a file previously saved by .save().

        Args:
            path (str): File path to load buffer from.
        """
        with open(path, 'rb') as f:
            items = pickle.load(f)
        self.buffer.clear()
        self.priorities.clear()
        for item_dict, priority in items:
            t = Transition(**item_dict)
            self.buffer.append(t)
            self.priorities.append(priority)

    def clear(self) -> None:
        """
        Remove all transitions from the buffer, emptying it.
        """
        self.buffer.clear()
        self.priorities.clear()
