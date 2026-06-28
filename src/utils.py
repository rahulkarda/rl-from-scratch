"""
Basic utilities for RL experiments: seeding, running/moving averages, moving std, Polyak averaging, dict flattening, and reward stats.

Rationale:
- set_seed: Reproducibility for Python, NumPy, PyTorch
- seed_everything: Global seeding incl. Gymnasium envs
- running_average: Cumulative average, e.g. smoothing reward curves
- moving_average: Rolling average over window, for reward/loss smoothing
- moving_std: Rolling std over window, for error bands/uncertainty plots
- soft_update: Polyak averaging for target networks
- flatten_dict: Makes nested dicts flat for CSV logging
- compute_reward_stats: Summarizes reward distributions
- compute_quantiles: Arbitrary quantile extraction for distributions

Minimal dependencies: only numpy and torch.
"""
import random
import numpy as np
import torch

# --- Seeding utilities ---
def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything(seed: int, env=None) -> None:
    """
    Set seeds for Python, NumPy, PyTorch, AND Gymnasium env (if provided).
    Also sets torch deterministic flags for reproducibility.
    """
    set_seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if env is not None:
        try:
            env.seed(seed)
        except AttributeError:
            try:
                env.reset(seed=seed)
            except Exception:
                pass

# --- Averaging utilities ---
def running_average(values):
    """
    Compute cumulative average up to each point.
    Args:
        values (array-like): Sequence of numbers
    Returns:
        np.ndarray: running average array (same length)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def moving_average(values, window_size: int):
    """
    Compute rolling average over fixed window.
    Args:
        values (array-like)
        window_size (int): Window length
    Returns:
        np.ndarray: averaged array (len = len(values) - window_size + 1)
    """
    values = np.array(values, dtype=float)
    if values.size < window_size or window_size < 1:
        return np.array([])
    return np.convolve(values, np.ones(window_size), 'valid') / window_size


def moving_std(values, window_size: int):
    """
    Rolling std deviation over fixed window.
    Args:
        values (array-like)
        window_size (int)
    Returns:
        np.ndarray: std array (len = len(values) - window_size + 1)
    """
    values = np.array(values, dtype=float)
    if values.size < window_size or window_size < 1:
        return np.array([])
    out = np.empty(values.size - window_size + 1)
    for i in range(out.size):
        out[i] = np.std(values[i:i+window_size])
    return out

# --- Polyak averaging ---
def soft_update(target_net, source_net, tau: float):
    """
    Polyak averaging for target network update.
    Args:
        target_net: torch.nn.Module to update
        source_net: torch.nn.Module source
        tau (float): Blend factor (tau=1 is hard copy)
    """
    for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
        t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

# --- Dict flattening ---
def flatten_dict(d, parent_key='', sep='.'):  # flatten_dict({'a': {'b': 1}}) -> {'a.b': 1}
    """
    Flatten arbitrarily nested dicts using dot notation.
    Args:
        d (dict): Nested dict
        parent_key (str): prefix
        sep (str): separator
    Returns:
        dict: flat dict
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# --- Reward stats ---
def compute_reward_stats(returns):
    """
    Summarize reward distribution: mean, std, min, max.
    Args:
        returns (array-like): Sequence of rewards/returns
    Returns:
        dict: {'mean', 'std', 'min', 'max'}
    """
    arr = np.array(returns, dtype=float)
    if arr.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

# --- Quantile extraction ---
def compute_quantiles(values, qs):
    """
    Compute arbitrary quantiles from sequence.
    Args:
        values (array-like): Input values (rewards, metrics, etc)
        qs (list/array): Quantile values in [0,1]
    Returns:
        dict: {q: quantile_value, ...}
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {q: 0.0 for q in qs}
    quantile_vals = np.quantile(arr, qs)
    return {float(q): float(v) for q, v in zip(qs, quantile_vals)}
