"""Uniform replay buffer for RL: stores transitions as FIFO, enables random sampling, and provides simple serialization.

Overview:
- Stores transitions as a FIFO queue (deque), discarding the oldest as new ones arrive.
- Enables uniform random sampling for minibatch updates, breaking temporal correlations for stability in value-based RL (e.g. DQN).
- Provides easy serialization: saves all transitions as dicts, not opaque objects, so buffer files are portable and inspectable.
- Supports sampling the most recent N transitions (ordered), useful for debugging or on-policy algorithms.
- Offers export of all transitions as a list for analysis or integration with external tools.
- Not thread-safe; intended for single-process training.

Usage:
    from replay import ReplayBuffer, Transition
    buf = ReplayBuffer(capacity=50_000)
    buf.push(Transition(state, action, reward, next_state, done))
    batch = buf.sample(batch_size=32)
    recent = buf.sample_recent(10)
    all_transitions = buf.export_to_list()
    for batch in buf.random_batch_iter(batch_size=32):
        ... # iterates over random minibatches
    buf.save('buffer.pkl')
    buf.load('buffer.pkl')
    buf.clear()
    length = len(buf)
    size = buf.size()

Serialization:
- .save(path): Pickles a list of transition dicts (not raw objects), so the file is portable and readable. Useful for analysis or sharing.
- .load(path): Reads dicts, reconstructs Transition dataclass (order preserved).
- Buffer files can be inspected with Python or pandas (it's a list of dicts).

Design notes:
- Uniform sampling ensures every transition has equal chance to be picked, matching classic DQN.
- FIFO queue (deque) makes insertions/removals efficient and predictable.
- Transitions are dataclasses for clarity and reliable serialization.
- Not thread-safe (no locks); single-thread use only.
- Sampling raises ValueError if batch_size exceeds buffer contents.
- Methods return ordered lists (oldest to newest) where appropriate.

"""
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, List, Iterator
import pickle

@dataclass
class Transition:
    """
    Single transition tuple for RL replay buffer.

    Fields:
        state (Any): Observation/state (can be np.ndarray, list, etc)
        action (int): Action taken
        reward (float): Reward received
        next_state (Any): Next observation after action
        done (bool): Whether episode terminated
    """
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

    def random_batch_iter(self, batch_size: int) -> Iterator[List[Transition]]:
        """
        Yield random batches (minibatches) from the buffer.
        Useful for evaluation or batch analysis.

        Args:
            batch_size (int): Size of each batch.
        Yields:
            List[Transition]: Random batch.
        """
        buf_list = list(self.buffer)
        total = len(buf_list)
        if batch_size > total:
            raise ValueError(f"Cannot random_batch_iter batch_size={batch_size} from buffer with {total} transitions.")
        indices = list(range(total))
        random.shuffle(indices)
        for i in range(0, total, batch_size):
            batch_idx = indices[i:i+batch_size]
            yield [buf_list[j] for j in batch_idx]

    def save(self, path: str) -> None:
        """
        Serialize buffer to a pickle file as a list of transition dicts.
        Args:
            path (str): File path to save buffer.
        """
        with open(path, "wb") as f:
            dicts = [asdict(t) for t in self.buffer]
            pickle.dump(dicts, f)

    def load(self, path: str) -> None:
        """
        Load buffer from a pickle file containing transition dicts.
        Args:
            path (str): File path to load buffer.
        """
        with open(path, "rb") as f:
            dicts = pickle.load(f)
            self.buffer.clear()
            for d in dicts:
                self.buffer.append(Transition(**d))

    def clear(self) -> None:
        """
        Remove all transitions from the buffer.
        """
        self.buffer.clear()

    def length(self) -> int:
        """
        Return current buffer size (number of transitions in the buffer).
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
