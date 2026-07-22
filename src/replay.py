"""Uniform replay buffer for RL: stores transitions as FIFO, enables random sampling, and provides simple serialization.

Rationale:
- FIFO queue ensures oldest transitions are discarded first, matching classic DQN-style replay.
- Uniform sampling breaks temporal correlations for value-based algorithms.
- Serialization stores transitions as dicts for compatibility and easy inspection.
- Not thread-safe; intended for single-process use.

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

Serialization:
- .save(path): Pickles a list of transition dicts (not raw objects), so the file is portable and readable.
- .load(path): Reads dicts, reconstructs Transition dataclass (order preserved).
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

    Stores transitions in a fixed-size FIFO queue (deque) with uniform sampling.
    Used in value-based RL algorithms (e.g., DQN) to break correlation between sequential samples.

    Features:
        - push: Add a transition, discards oldest if full
        - sample: Uniformly sample a batch of transitions
        - sample_recent: Get the most recent N transitions (ordered)
        - export_to_list: Export all transitions as a list
        - clear: Remove all transitions
        - save/load: Serialize buffer to/from pickle file (as dicts)
        - random_batch_iter: Yield random minibatches (for evaluation/analysis)
        - length: Current buffer size (NEW)
        - __len__: Allows len(buffer) usage (NEW)
        - size(): Returns current buffer size (NEW)

    Limitations:
        - Only supports uniform sampling (no prioritization)
        - Assumes transitions are dataclass objects (Transition)
        - Not thread-safe; only use from one thread/process
        - Serialization via .save/.load uses dict conversion for compatibility
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
        Add a transition to the buffer. Discards oldest if full.

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
            List[Transition]: Randomly chosen transitions.
        Raises:
            ValueError: If batch_size > number of transitions.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        return random.sample(self.buffer, batch_size)

    def sample_recent(self, batch_size: int) -> List[Transition]:
        """
        Get the most recent N transitions (not random; ordered oldest to newest).
        Useful for debugging, visualization, or on-policy algorithms.

        Args:
            batch_size (int): Number of transitions to return.
        Returns:
            List[Transition]: Latest transitions (ordered).
        Raises:
            ValueError: If batch_size > number of transitions.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample_recent batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        return list(self.buffer)[-batch_size:]

    def export_to_list(self) -> List[Transition]:
        """
        Export all transitions in the buffer as a list (ordered oldest to newest).
        Useful for analysis or integration with external tools.

        Returns:
            List[Transition]: All transitions (ordered).
        """
        return list(self.buffer)

    def clear(self) -> None:
        """
        Remove all transitions from the buffer.
        """
        self.buffer.clear()

    def save(self, path: str) -> None:
        """
        Serialize buffer to a pickle file as a list of dicts.

        Args:
            path (str): File path to save buffer.
        """
        with open(path, "wb") as f:
            data = [asdict(t) for t in self.buffer]
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """
        Load buffer from a pickle file containing a list of transition dicts.

        Args:
            path (str): File path to load buffer from.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.clear()
        for t_dict in data:
            self.buffer.append(Transition(**t_dict))

    def random_batch_iter(self, batch_size: int) -> Iterator[List[Transition]]:
        """
        Yield random minibatches (non-overlapping) of transitions.
        Useful for evaluation/analysis.

        Args:
            batch_size (int): Size of each minibatch.
        Yields:
            List[Transition]: Random batch.
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

    @property
    def length(self) -> int:
        """
        Current number of transitions in the buffer.
        Returns:
            int: Buffer size.
        """
        return len(self.buffer)

    def size(self) -> int:
        """
        Returns current buffer size (number of transitions).
        Returns:
            int: Buffer size.
        """
        return len(self.buffer)

    def __len__(self) -> int:
        """
        Allow usage of len(buffer) to query current size.
        Returns:
            int: Buffer size.
        """
        return len(self.buffer)
