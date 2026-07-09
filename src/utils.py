"""
Basic utilities for RL experiments: seeding, averaging, Polyak update, dict flattening, and reward stats.

Purpose:
- set_seed: Reproducibility for Python, NumPy, PyTorch
- seed_everything: Global seeding incl. Gymnasium envs
- running_average: Cumulative average smoothing
- moving_average: Rolling mean for reward/loss curves
- moving_std: Rolling std for error bands
- soft_update: Polyak averaging for target networks
- flatten_dict: Flatten nested dicts for logging
- compute_reward_stats: Summarize reward distributions
- compute_quantiles: Extract arbitrary quantiles
- compute_median: Median utility for reward stats
- min_max_normalize: Scale array to [0, 1] (NEW)

Notes:
- Functions accept lists, arrays, or sequences (e.g., rewards, losses) and return np.ndarray or dict.
- All averaging and stats utilities handle empty input gracefully (returns empty array or zeros).
- Serialization (seed_everything) is robust to envs missing .seed or .reset(seed=...).
- Polyak averaging (soft_update) assumes matching parameter shapes between models.
- flatten_dict flattens nested dicts for logging and CSV compatibility.

Minimal dependencies: numpy, torch.
"""
import random
import numpy as np
import torch
import collections.abc

# === Seeding ===
def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch.
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
    Set seeds for Python, NumPy, PyTorch, and Gymnasium env (if provided).
    Also sets torch deterministic flags.
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

# === Averaging ===
def running_average(values):
    """
    Compute cumulative average for each step.
    Args:
        values: sequence of numbers
    Returns:
        np.ndarray: running average (same length)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def moving_average(values, window_size: int):
    """
    Compute rolling mean over window.
    Args:
        values: sequence
        window_size: window length
    Returns:
        np.ndarray: averaged array (len = len(values) - window_size + 1)
    """
    values = np.array(values, dtype=float)
    if values.size < window_size or window_size < 1:
        return np.array([])
    return np.convolve(values, np.ones(window_size), 'valid') / window_size


def moving_std(values, window_size: int):
    """
    Rolling std deviation over window.
    Args:
        values: sequence
        window_size: window length
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

# === Polyak Averaging ===
def soft_update(target_net, source_net, tau: float):
    """
    Polyak averaging for target network update.
    Args:
        target_net: torch.nn.Module to update
        source_net: torch.nn.Module source
        tau: blend factor (tau=1 is hard copy)
    """
    for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
        t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

# === Dict Flattening ===
def flatten_dict(d, parent_key='', sep='.'):  # flatten_dict({'a': {'b': 1}}) -> {'a.b': 1}
    """
    Flatten nested dicts using dot notation.
    Args:
        d: nested dict or mapping
        parent_key: prefix
        sep: separator
    Returns:
        dict: flat dict
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Only recursively flatten if v is a dict (not generic Mapping)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# === Reward Stats ===
def compute_reward_stats(returns):
    """
    Compute mean, std, min, max for rewards/returns.
    Args:
        returns: sequence
    Returns:
        dict: {'mean', 'std', 'min', 'max', 'median'}
    """
    arr = np.array(returns, dtype=float)
    if arr.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }

def compute_quantiles(values, qs):
    """
    Compute arbitrary quantiles from values.
    Args:
        values: sequence
        qs: list/tuple of quantile floats (e.g. [0.25, 0.5, 0.75])
    Returns:
        dict: {quantile: value}
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {q: 0.0 for q in qs}
    quantile_vals = np.quantile(arr, qs)
    return {float(q): float(v) for q, v in zip(qs, quantile_vals)}

# === Median Utility ===
def compute_median(values):
    """
    Compute median of values, returns 0.0 if empty.
    Args:
        values: sequence
    Returns:
        float: median value
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))

# === Min-Max Normalize ===
def min_max_normalize(values):
    """
    Scale values to [0, 1] by min-max normalization.
    Args:
        values: sequence
    Returns:
        np.ndarray: normalized array (empty if input empty)
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.array([])
    min_val = arr.min()
    max_val = arr.max()
    if min_val == max_val:
        # Avoid division by zero; return zeros
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)
