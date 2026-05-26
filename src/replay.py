"""Uniform replay buffer. The first thing every value-based algorithm needs."""
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
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} transitions.")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str) -> None:
        """
        Save buffer contents to a file using pickle.
        """
        # We store list of transition dicts to avoid issues with dataclass serialization
        with open(path, 'wb') as f:
            pickle.dump([asdict(t) for t in self.buffer], f)

    def load(self, path: str) -> None:
        """
        Load buffer contents from a file saved by .save().
        """
        with open(path, 'rb') as f:
            items = pickle.load(f)
        self.buffer.clear()
        for item in items:
            t = Transition(**item)
            self.buffer.append(t)
