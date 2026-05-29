import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Set seed for Python, NumPy, and PyTorch (CPU/CUDA) for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def moving_average(values, window_size: int):
    """
    Compute simple moving average over a list or array.

    Args:
        values: Sequence of numbers (list, np.ndarray).
        window_size: Size of window (int).

    Returns:
        np.ndarray of moving averages (len = len(values) - window_size + 1)
    """
    values = np.array(values)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(values) < window_size:
        return np.array([])
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def running_average(values):
    """
    Compute running (cumulative) average for a sequence.
    Returns np.ndarray with same length as values.
    """
    values = np.array(values)
    if len(values) == 0:
        return np.array([])
    cumsum = np.cumsum(values)
    counts = np.arange(1, len(values)+1)
    return cumsum / counts
