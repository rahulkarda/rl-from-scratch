"""Uniform replay buffer for RL: stores transitions as FIFO, enables random sampling, and provides simple serialization.

Rationale:
- FIFO queue (deque) ensures oldest transitions are discarded first, matching classic DQN-style replay.
- Uniform sampling breaks temporal correlations for value-based algorithms.
- Serialization (.save/.load) stores as dicts for compatibility and easy inspection (see below).
- Not thread-safe; intended for single-process use.

Usage:
    buf = ReplayBuffer(capacity=50_000)
    buf.push(Transition(...))
    batch = buf.sample(batch_size=32)
    buf.save('buffer.pkl')  # Stores buffer as a list of dicts
    buf.load('buffer.pkl')  # Loads buffer from dicts, reconstructs Transition dataclass (order preserved)
    buf.clear()             # Removes all transitions
    recent = buf.sample_recent(10)  # Returns most recent 10 transitions
    all_transitions = buf.export_to_list()  # List of all transitions
    # NEW:
    for batch in buf.random_batch_iter(batch_size=32):
        ... # iterates over random minibatches (non-overlapping)

Serialization:
- .save(path): Pickles a list of transition dicts (not raw objects), so the file is portable and readable.
- .load(path): Reads dicts, reconstructs Transition dataclass (order preserved).
- Useful for analysis, debugging, and integration with external tools.
- You can inspect the buffer file with Python or tools like pandas, since it's a list of dicts (not opaque objects).

"""
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, List, Iterator
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
        buf = ReplayBuffer(capacity=50_000)
        buf.push(Transition(...))
        batch = buf.sample(batch_size=32)
        buf.save('buffer.pkl')
        buf.load('buffer.pkl')
        buf.clear()
        recent = buf.sample_recent(10)
        all_transitions = buf.export_to_list()
        for batch in buf.random_batch_iter(batch_size=32):
            ... # iterates over random minibatches

    Limitations:
        - Only supports uniform sampling (no prioritization).
        - Assumes transitions are dataclass objects (Transition).
        - Not thread-safe; only use from one thread/process.
        - Serialization via .save/.load uses dict conversion for compatibility.
    """

    def __init__(self, capacity: int = 50_000):
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
        # Return most recent N transitions, ordered: oldest to newest
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
        return len(self.buffer)

    def clear(self) -> None:
        """
        Remove all transitions from the buffer.
        """
        self.buffer.clear()

    def save(self, path: str) -> None:
        """
        Save buffer to a file as a list of dicts (not raw objects).

        Args:
            path (str): File path to save buffer.
        """
        # Convert Transition objects to dicts for portability
        dicts = [asdict(t) for t in self.buffer]
        with open(path, "wb") as f:
            pickle.dump(dicts, f)

    def load(self, path: str) -> None:
        """
        Load buffer from a file containing a list of dicts.

        Args:
            path (str): File path to load buffer from.
        """
        with open(path, "rb") as f:
            dicts = pickle.load(f)
        self.clear()
        for t_dict in dicts:
            t = Transition(**t_dict)
            self.push(t)

    def random_batch_iter(self, batch_size: int) -> Iterator[List[Transition]]:
        """
        Iterate over random, non-overlapping minibatches from the buffer.
        Each batch is sampled without replacement from the current buffer contents.

        Args:
            batch_size (int): Number of transitions per batch.
        Yields:
            List[Transition]: Random batch of transitions.

        Notes:
            - Each transition is used at most once per call to this iterator.
            - If buffer size is not divisible by batch_size, the last batch may be smaller.
            - Shuffles indices prior to batching for randomness.
        """
        num_transitions = len(self.buffer)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if num_transitions == 0:
            return
        indices = list(range(num_transitions))
        random.shuffle(indices)
        buffer_list = list(self.buffer)  # Avoid repeated conversion inside loop
        for start in range(0, num_transitions, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch = [buffer_list[j] for j in batch_indices]
            yield batch
