"""Uniform replay buffer for RL: stores transitions as FIFO, enables random sampling, and provides simple serialization.

Rationale:
- FIFO queue (deque) ensures oldest transitions are discarded first, matching classic DQN-style replay.
- Uniform sampling breaks temporal correlations for value-based algorithms.
- Serialization (.save/.load) stores as dicts for compatibility and easy inspection.
- Not thread-safe; intended for single-process use.

Usage:
    buf = ReplayBuffer(capacity=100_000)
    buf.push(Transition(...))
    batch = buf.sample(batch_size=32)
"""
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, List
import pickle

@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class ReplayBuffer:
    """
    Uniform replay buffer for storing and sampling transitions.

    Stores transitions as a fixed-size FIFO queue (deque) with uniform sampling.
    Used in value-based RL algorithms (e.g., DQN) to break correlation between sequential samples.

    Usage:
        buf = ReplayBuffer(capacity=100_000)
        buf.push(Transition(...))
        batch = buf.sample(batch_size=32)

    Limitations:
        - Only supports uniform sampling (no prioritization).
        - Assumes transitions are dataclass objects (Transition).
        - Not thread-safe; only use from one thread/process.
        - Serialization via .save/.load uses dict conversion for compatibility.
    """

    def __init__(self, capacity: int = 100_000):
        """
        Initialize the replay buffer with a fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        """
        Add a transition to the buffer. Oldest is discarded if capacity is reached.

        Args:
            t (Transition): Transition to store.
        """
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Uniformly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            List[Transition]: List of sampled transitions.

        Raises:
            ValueError: If batch_size > number of transitions in buffer.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        return random.sample(self.buffer, batch_size)

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
        # Slice from right (most recent), but return in order (oldest to newest)
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

        Returns:
            int: Number of transitions in the buffer.
        """
        return len(self.buffer)

    def save(self, path: str) -> None:
        """
        Save buffer contents to a file using pickle.
        Stores as a list of transition dicts for compatibility.

        Args:
            path (str): File path to save buffer.
        """
        with open(path, 'wb') as f:
            pickle.dump([asdict(t) for t in self.buffer], f)

    def load(self, path: str) -> None:
        """
        Load buffer contents from a file previously saved by .save().

        Args:
            path (str): File path to load buffer from.
        """
        with open(path, 'rb') as f:
            items = pickle.load(f)
        self.buffer.clear()
        for item in items:
            t = Transition(**item)
            self.buffer.append(t)

    def clear(self) -> None:
        """
        Remove all transitions from the buffer, emptying it.
        """
        self.buffer.clear()
