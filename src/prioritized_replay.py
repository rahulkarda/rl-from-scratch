import random
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
    priority: float = 1.0  # Default priority

class PrioritizedReplayBuffer:
    """
    Minimal prioritized replay buffer skeleton.
    Stores transitions with priorities, enables sampling weighted by priority.
    Implements push, sample, save, load.
    """
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        """
        Args:
            capacity (int): Max buffer size
            alpha (float): Priority exponent (0=uniform, 1=max bias)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities: List[float] = []
        self.position = 0

    def push(self, t: Transition) -> None:
        """
        Add a transition with priority to buffer.
        Oldest is overwritten when full.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
            self.priorities.append(abs(t.priority))
        else:
            self.buffer[self.position] = t
            self.priorities[self.position] = abs(t.priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample transitions weighted by priority^alpha.
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer.")
        priorities = [p ** self.alpha for p in self.priorities]
        total = sum(priorities)
        if total == 0.0:
            probs = [1.0 / len(priorities)] * len(priorities)
        else:
            probs = [p / total for p in priorities]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str) -> None:
        """
        Save buffer as list of transition dicts.
        """
        with open(path, 'wb') as f:
            pickle.dump([asdict(t) for t in self.buffer], f)

    def load(self, path: str) -> None:
        """
        Load buffer from saved dict list.
        """
        with open(path, 'rb') as f:
            items = pickle.load(f)
        self.buffer.clear()
        self.priorities.clear()
        for item in items:
            t = Transition(**item)
            self.buffer.append(t)
            self.priorities.append(abs(getattr(t, 'priority', 1.0)))
        self.position = len(self.buffer) % self.capacity
