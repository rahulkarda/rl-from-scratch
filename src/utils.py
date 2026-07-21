"""
Basic utilities for RL experiments: seeding, averaging, Polyak update, dict flattening, reward and transition stats.

Purpose:
- set_seed: Reproducibility for Python, NumPy, PyTorch
- seed_everything: Global seeding incl. Gymnasium envs
- running_average: Cumulative average smoothing
- moving_average: Rolling mean for reward/loss curves
- moving_std: Rolling std for error bands
- exponential_moving_average: Exponential smoothing
- soft_update: Polyak averaging for target networks
- flatten_dict: Flatten nested dicts for logging
- compute_reward_stats: Summarize reward distributions
- compute_quantiles: Extract arbitrary quantiles
- compute_median: Median utility for reward stats
- min_max_normalize: Scale array to [0, 1]
- compute_gae_advantages: Generalized Advantage Estimation
- chunked: Split sequence into fixed-size batches
- compute_discounted_sum: Compute discounted sum over a sequence
- is_monotonic: Check if sequence is monotonic
- compute_mean_and_std: Compute mean and std for sequence
- transitions_to_dicts: Convert list of Transition objects to list of dicts
- pad_sequence_to_length: Pad sequence to fixed length with a value (NEW)

Notes:
- All averaging and stats utilities handle empty input gracefully (returns empty array or zeros).
- Polyak averaging (soft_update) assumes matching parameter shapes between models.
- flatten_dict flattens nested dicts for logging and CSV compatibility.
- Serialization (seed_everything) is robust to envs missing .seed or .reset(seed=...).

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
        target_net: torch.nn.Module (target)
        source_net: torch.nn.Module (source)
        tau: float (mixing coefficient, 0 < tau <= 1)
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.mul_(1 - tau).add_(tau * source_param.data)

# === Dict flattening ===
def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten nested dicts for logging and CSV compatibility.
    Args:
        d: dict (possibly nested)
        parent_key: prefix for keys
        sep: separator
    Returns:
        dict: flat key-value pairs
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
    Compute mean, std, min, max for sequence of rewards.
    Args:
        rewards: sequence of floats
    Returns:
        dict: {mean, std, min, max, count}
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "count": int(rewards.size)
    }


def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for sequence.
    Args:
        values: sequence of floats
        quantiles: sequence of floats (e.g. [0.25, 0.5, 0.75])
    Returns:
        dict: {q: v} mapping quantile to value
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {q: 0.0 for q in quantiles}
    qs = np.quantile(values, quantiles)
    return {float(q): float(v) for q, v in zip(quantiles, qs)}


def compute_median(values):
    """
    Compute median for sequence.
    Args:
        values: sequence of floats
    Returns:
        float: median value
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.median(values))

# === Normalization ===
def min_max_normalize(arr):
    """
    Scale array to [0, 1] range (min-max normalization).
    Args:
        arr: sequence of floats
    Returns:
        np.ndarray: normalized array
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.array([])
    minv, maxv = np.min(arr), np.max(arr)
    if minv == maxv:
        return np.zeros_like(arr)
    return (arr - minv) / (maxv - minv)

# === GAE ===
def compute_gae_advantages(rewards, values, next_values, dones, gamma: float, lam: float):
    """
    Generalized Advantage Estimation (GAE).
    Args:
        rewards: np.ndarray (shape [T])
        values: np.ndarray (shape [T])
        next_values: np.ndarray (shape [T])
        dones: np.ndarray (shape [T])
        gamma: float (discount)
        lam: float (lambda)
    Returns:
        np.ndarray: advantages (shape [T])
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=float)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        adv[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    return adv

# === Chunking ===
def chunked(seq, chunk_size: int):
    """
    Split sequence into fixed-size chunks.
    Args:
        seq: sequence
        chunk_size: int
    Yields:
        list: chunk
    """
    seq = list(seq)
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]

# === Discounted sum ===
def compute_discounted_sum(rewards, gamma: float):
    """
    Compute discounted sum over rewards (for value estimation).
    Args:
        rewards: sequence of floats
        gamma: float (discount factor)
    Returns:
        np.ndarray: discounted sum for each timestep
    """
    rewards = np.array(rewards, dtype=float)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out

# === Monotonicity check ===
def is_monotonic(seq):
    """
    Check if sequence is monotonic non-decreasing or non-increasing.
    Args:
        seq: sequence of floats
    Returns:
        bool: True if monotonic
    """
    seq = list(seq)
    if len(seq) < 2:
        return True
    increasing = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
    decreasing = all(seq[i] >= seq[i+1] for i in range(len(seq)-1))
    return increasing or decreasing

# === Mean and std ===
def compute_mean_and_std(values):
    """
    Compute mean and std for sequence (handles empty).
    Args:
        values: sequence of floats
    Returns:
        tuple: (mean, std)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))

# === Transition dict conversion ===
def transitions_to_dicts(transitions):
    """
    Convert list of dataclass Transition objects to list of dicts.
    Args:
        transitions: list of dataclass Transition objects (must have __dict__ or asdict)
    Returns:
        list[dict]: Each transition as a dict (for serialization, logging, inspection)
    """
    # Try dataclass asdict if available, else fallback to __dict__
    from dataclasses import asdict
    out = []
    for t in transitions:
        try:
            out.append(asdict(t))
        except Exception:
            out.append(dict(t.__dict__))
    return out

# === Pad sequence utility ===
def pad_sequence_to_length(seq, length, pad_value=0):
    """
    Pad a sequence to a fixed length with a pad value.
    Args:
        seq: sequence (list or np.ndarray)
        length: int, desired length
        pad_value: value to use for padding (default 0)
    Returns:
        np.ndarray: padded sequence of shape (length,)
    """
    seq_arr = np.array(seq)
    current_len = seq_arr.size
    if current_len >= length:
        return seq_arr[:length]
    pad = np.full(length - current_len, pad_value, dtype=seq_arr.dtype)
    return np.concatenate([seq_arr, pad])
