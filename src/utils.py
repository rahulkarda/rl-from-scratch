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
- chunked: Split sequence into fixed-size batches (NEW)
- exponential_moving_average: Exponential smoothing for reward/loss curves (NEW)
- compute_discounted_sum: Compute discounted sum over a sequence (NEW)

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
    if values.size == 0 or window_size < 1:
        return np.array([])
    if window_size > values.size:
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
    if values.size == 0 or window_size < 1:
        return np.array([])
    if window_size > values.size:
        return np.array([])
    out = np.empty(values.size - window_size + 1)
    for i in range(out.size):
        out[i] = np.std(values[i:i+window_size])
    return out

# === Exponential Moving Average ===
def exponential_moving_average(values, alpha: float):
    """
    Compute exponential moving average (EMA) for a sequence.
    Args:
        values: sequence of numbers
        alpha: smoothing factor (0 < alpha <= 1), higher is less smoothing
    Returns:
        np.ndarray: exponential moving average (same length)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    ema = np.empty_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema

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

# === Dict flattening ===
def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dicts (for logging, CSV export).
    Args:
        d: dict (possibly nested)
        parent_key: prefix for keys
        sep: separator
    Returns:
        dict: flattened
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# === Reward stats ===
def compute_reward_stats(rewards):
    """
    Compute mean, std, min, max for a sequence of rewards.
    Args:
        rewards: sequence
    Returns:
        dict: {mean, std, min, max}
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards))
    }

# === Quantiles ===
def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a sequence.
    Args:
        values: sequence
        quantiles: list of floats (0-1)
    Returns:
        np.ndarray: quantile values
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([0.0 for _ in quantiles])
    return np.quantile(values, quantiles)


def compute_median(values):
    """
    Compute median for a sequence.
    Args:
        values: sequence
    Returns:
        float: median
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.median(values))

# === Min-max normalize ===
def min_max_normalize(values):
    """
    Scale array to [0, 1] range.
    Args:
        values: sequence of numbers
    Returns:
        np.ndarray: normalized to [0, 1]
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    vmin = np.min(values)
    vmax = np.max(values)
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)

# === GAE Advantage estimation ===
def compute_gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    Args:
        rewards: [T]
        values: [T+1]
        dones: [T] (bool)
        gamma: discount factor
        lam: GAE lambda
    Returns:
        np.ndarray: advantages [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        advantages[t] = last_adv = delta + gamma * lam * nonterminal * last_adv
    return advantages

# === Chunking ===
def chunked(iterable, batch_size: int):
    """
    Split sequence or iterable into batches of batch_size.
    Args:
        iterable: sequence or iterable
        batch_size: int
    Yields:
        list: batches
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    # Convert to list for slicing if not already
    if isinstance(iterable, collections.abc.Sequence):
        n = len(iterable)
        for i in range(0, n, batch_size):
            yield iterable[i:i+batch_size]
    else:
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# === Discounted sum ===
def compute_discounted_sum(values, gamma=0.99):
    """
    Compute discounted sum over a sequence.
    Useful for estimating return from rewards.
    Args:
        values: sequence (e.g., rewards)
        gamma: discount factor
    Returns:
        np.ndarray: discounted sums (same shape as input)
    """
    values = np.array(values, dtype=float)
    n = len(values)
    out = np.zeros_like(values)
    running_sum = 0.0
    for t in reversed(range(n)):
        running_sum = values[t] + gamma * running_sum
        out[t] = running_sum
    return out
