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
- compute_gae_advantages: Generalized Advantage Estimation (NEW)

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
        d (dict or Mapping): nested dict or mapping to flatten
        parent_key (str): prefix for keys (internal use)
        sep (str): separator (default '.')
    Returns:
        dict: flat dict with keys representing path in original dict
    Example:
        flatten_dict({'a': {'b': 1}, 'c': 2}) -> {'a.b': 1, 'c': 2}
    Notes:
        Only recursively flattens items of type dict (not generic Mapping).
        This avoids flattening e.g. defaultdict or custom mapping types that may not be true dicts.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten only if v is a dict (not generic Mapping)
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
    returns = np.array(returns, dtype=float)
    if returns.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    return {
        'mean': float(np.mean(returns)),
        'std': float(np.std(returns)),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'median': float(np.median(returns))
    }


def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a list/array.
    Args:
        values: sequence
        quantiles: list of quantile values (e.g., [0.25, 0.5, 0.75])
    Returns:
        dict: {q: value}
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {q: 0.0 for q in quantiles}
    qs = np.quantile(values, quantiles)
    return {float(q): float(v) for q, v in zip(quantiles, qs)}


def compute_median(values):
    """
    Compute median for a sequence.
    Args:
        values: sequence
    Returns:
        float: median value
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.median(values))

# === Min-Max Normalize ===
def min_max_normalize(values):
    """
    Scale values to [0, 1] via min-max normalization.
    Args:
        values: sequence
    Returns:
        np.ndarray: scaled array
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    vmin = np.min(values)
    vmax = np.max(values)
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)

# === GAE Advantage Estimation ===
def compute_gae_advantages(rewards, values, dones, gamma: float, lam: float):
    """
    Compute Generalized Advantage Estimation (GAE).
    Args:
        rewards: sequence (length T)
        values: sequence (length T or T+1)
        dones: sequence (length T), boolean
        gamma: discount factor
        lam: GAE lambda
    Returns:
        np.ndarray: advantages (length T)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    dones = np.array(dones, dtype=bool)
    T = len(rewards)
    if values.shape[0] == T:
        # pad with zero for terminal
        values = np.append(values, 0.0)
    advantages = np.zeros(T, dtype=float)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * (not dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (not dones[t])
        advantages[t] = gae
    return advantages
