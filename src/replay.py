"""Uniform replay buffer. The first thing every value-based algorithm needs."""
import random
from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class Transition:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
