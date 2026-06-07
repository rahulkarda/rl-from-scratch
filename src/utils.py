"""
Basic utilities for RL experiments: seeding, running/moving averages.

Rationale:
- set_seed: Reproducibility is critical in RL due to noisy training and variance. Sets seeds for Python, NumPy, and PyTorch.
- running_average: Useful for smoothing reward curves or losses over time. Computes cumulative average (up to each point).
- moving_average: Computes average over a fixed window. Used for plotting recent episode returns and smoothing metrics.
- soft_update: Polyak averaging for target networks, needed in Double DQN/DDPG/SAC. Simple utility for updating target model parameters.

These functions are intentionally minimal and avoid dependencies beyond numpy and torch.
"""
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def running_average(values):
    """
    Compute running (cumulative) average for a sequence.
    Each value is the average of all previous values up to that index.

    Args:
        values: Sequence of numbers (list, np.ndarray).

    Returns:
        np.ndarray of running averages (same length as values).

    Example:
        running_average([1, 2, 3, 4])
        # returns array([1., 1.5, 2., 2.5])
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    cumsum = np.cumsum(values)
    return cumsum / (np.arange(1, values.size + 1))


def moving_average(values, window_size: int):
    """
    Compute simple moving average over a list or array.

    Returns a sequence of averages where each average is computed over a sliding window of length `window_size`.

    Args:
        values: Sequence of numbers (list, np.ndarray).
        window_size: Size of window (int).

    Returns:
        np.ndarray of moving averages (len = len(values) - window_size + 1)

    Example:
        moving_average([1, 2, 3, 4, 5], window_size=3)
        # returns array([2., 3., 4.])
    """
    values = np.array(values, dtype=float)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if values.size < window_size:
        return np.array([])
    cumsum = np.cumsum(values)
    cumsum[window_size:] -= cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    Polyak averaging for target network updates.
    Copies parameters from `source` to `target` using:
        target_param = tau * source_param + (1 - tau) * target_param
    Args:
        target (torch.nn.Module): Target network to update.
        source (torch.nn.Module): Source network (e.g., online network).
        tau (float): Mixing factor (0 < tau <= 1). tau=1 means hard update.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
