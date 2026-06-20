"""Prioritized Experience Replay buffer (PER): supports sampling transitions by priority, for value-based RL.

Rationale:
- Classic DQN replay is uniform; PER (Schaul et al. 2015) samples based on TD error, improving sample efficiency.
- Implements basic sum-tree for efficient sampling and priority updates.
- API: push, sample, update_priorities, save, load, clear.
- This is a skeleton; sampling and priority update logic will be implemented in future steps.

Usage:
    buf = PrioritizedReplayBuffer(capacity=100_000, alpha=0.6)
    buf.push(Transition(...), priority=1.0)
    batch, indices, weights = buf.sample(batch_size=32, beta=0.4)
    buf.update_priorities(indices, new_priorities)
"""
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, List
import pickle

@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (PER).

    Stores transitions with associated priorities, enabling non-uniform sampling.
    Skeleton version: supports push, sample, update_priorities, save, load, clear.
    SumTree and sampling logic will be implemented in future steps.
    """
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        """
        Initialize buffer with fixed capacity and prioritization exponent.

        Args:
            capacity (int): Max number of transitions.
            alpha (float): Controls priority effect (0=uniform, 1=full priority).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, t: Transition, priority: float = 1.0) -> None:
        """
        Add a transition and its priority.

        Args:
            t (Transition): Transition to store.
            priority (float): Initial priority (typically abs(TD error)).
        """
        self.buffer.append(t)
        self.priorities.append(priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample a batch of transitions based on priority.
        Returns transitions, their indices, and importance-sampling weights.
        (Skeleton: uniform sampling, weights=1.0)

        Args:
            batch_size (int): Number to sample.
            beta (float): Importance-sampling exponent for bias correction.

        Returns:
            transitions (list): List[Transition]
            indices (list): List[int]
            weights (list): List[float]
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        indices = random.sample(range(len(self.buffer)), batch_size)
        transitions = [self.buffer[i] for i in indices]
        weights = [1.0] * batch_size  # Uniform weights for now
        return transitions, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for given indices.
        Args:
            indices (list): Buffer indices.
            priorities (list): New priorities.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str) -> None:
        """
        Save buffer contents and priorities to a file using pickle.
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'transitions': [asdict(t) for t in self.buffer],
                'priorities': list(self.priorities)
            }, f)

    def load(self, path: str) -> None:
        """
        Load buffer and priorities from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buffer.clear()
        self.priorities.clear()
        for tdict in data['transitions']:
            t = Transition(**tdict)
            self.buffer.append(t)
        for prio in data['priorities']:
            self.priorities.append(prio)

    def clear(self) -> None:
        """
        Remove all transitions and priorities.
        """
        self.buffer.clear()
        self.priorities.clear()
