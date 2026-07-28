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
- elementwise_min_max: Compute elementwise min or max between two arrays (NEW)
- clip_array_values: Clip array values to [min, max] (NEW)

Usage:
    from utils import set_seed, moving_average, soft_update, compute_gae_advantages
    set_seed(42)
    avg = moving_average([1,2,3,4], window_size=2)
    soft_update(target_net, source_net, tau=0.005)
    adv = compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    clipped = clip_array_values([1, 2, 10], min_value=0, max_value=5)

Design notes:
- All averaging and stats utilities handle empty input gracefully (returns empty array or zeros).
- Polyak averaging (soft_update) assumes matching parameter shapes between models.
- flatten_dict flattens nested dicts for logging and CSV compatibility.
- Serialization (seed_everything) is robust to envs missing .seed or .reset(seed=...).
- Utilities are type-agnostic (list, np.ndarray, etc) whenever possible.
- No dependencies beyond numpy and torch.

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
        np.ndarray: smoothed array
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    out = np.empty_like(values)
    if values.size == 0:
        return np.array([])
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

# === Polyak averaging ===
def soft_update(target_net, source_net, tau: float) -> None:
    """
    Polyak averaging: update target_net parameters as tau * source + (1-tau) * target.
    Args:
        target_net: torch.nn.Module
        source_net: torch.nn.Module
        tau: float, mixing factor (0 < tau <= 1)
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# === Dict flattening ===
def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dicts for logging (CSV, JSON).
    Args:
        d: dict
        parent_key: str
        sep: str
    Returns:
        flat dict
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
    Compute mean, std, min, max for reward sequence.
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
        "max": float(np.max(rewards)),
    }


def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a sequence.
    Args:
        values: sequence
        quantiles: list of floats in [0,1]
    Returns:
        np.ndarray: quantile values
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([0.0]*len(quantiles))
    return np.quantile(values, quantiles)


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

# === Min/max normalization ===
def min_max_normalize(arr):
    """
    Normalize array to [0, 1] scale.
    Args:
        arr: sequence or np.ndarray
    Returns:
        np.ndarray: normalized array
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.array([])
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

# === GAE advantage estimation ===
def compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    Args:
        rewards: sequence of rewards
        values: sequence of values
        next_values: sequence of next values
        dones: sequence of done flags
        gamma: discount factor
        lam: GAE lambda
    Returns:
        np.ndarray: GAE advantages
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    next_values = np.array(next_values, dtype=float)
    dones = np.array(dones, dtype=float)
    T = len(rewards)
    adv = np.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        adv[t] = last_gae
    return adv

# === Chunking ===
def chunked(seq, chunk_size):
    """
    Split sequence into fixed-size chunks.
    Args:
        seq: sequence
        chunk_size: int
    Returns:
        generator of chunks
    """
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i+chunk_size]

# === Discounted sum ===
def compute_discounted_sum(rewards, gamma=0.99):
    """
    Compute discounted sum over a sequence.
    Args:
        rewards: sequence
        gamma: discount factor
    Returns:
        np.ndarray: discounted sums
    """
    rewards = np.array(rewards, dtype=float)
    out = np.zeros_like(rewards)
    running_sum = 0.0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        out[t] = running_sum
    return out

# === Monotonicity ===
def is_monotonic(seq):
    """
    Check if sequence is monotonic (increasing or decreasing).
    Args:
        seq: sequence
    Returns:
        bool
    """
    seq = np.array(seq)
    if seq.size < 2:
        return True
    return np.all(np.diff(seq) >= 0) or np.all(np.diff(seq) <= 0)

# === Mean and std ===
def compute_mean_and_std(seq):
    """
    Compute mean and std for a sequence.
    Args:
        seq: sequence
    Returns:
        tuple: (mean, std)
    """
    seq = np.array(seq, dtype=float)
    if seq.size == 0:
        return 0.0, 0.0
    return float(np.mean(seq)), float(np.std(seq))

# === Transition dict conversion ===
def transitions_to_dicts(transitions):
    """
    Convert list of Transition objects to list of dicts.
    Args:
        transitions: list of dataclass objects
    Returns:
        list of dicts
    """
    return [t.__dict__ if hasattr(t, '__dict__') else t for t in transitions]

# === Pad utility ===
def pad_sequence_to_length(seq, length, pad_value=0):
    """
    Pad sequence to fixed length with pad_value. If longer, truncate.
    Args:
        seq: sequence
        length: target length
        pad_value: value to pad
    Returns:
        np.ndarray: padded or truncated sequence
    """
    seq = np.array(seq)
    if seq.size == 0:
        return np.array([])
    if seq.size >= length:
        return seq[:length]
    out = np.full(length, pad_value)
    out[:seq.size] = seq
    return out

# === Elementwise min/max ===
def elementwise_min_max(a, b, mode="min"):
    """
    Compute elementwise min or max between two arrays.
    Args:
        a: array-like
        b: array-like
        mode: 'min' or 'max'
    Returns:
        np.ndarray
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    if mode == "min":
        return np.minimum(a_arr, b_arr)
    elif mode == "max":
        return np.maximum(a_arr, b_arr)
    else:
        raise ValueError("mode must be 'min' or 'max'")

# === Clip Utility ===
def clip_array_values(arr, min_value, max_value):
    """
    Clip all values in an array or sequence to [min_value, max_value].
    Args:
        arr: sequence or np.ndarray
        min_value: float (lower bound)
        max_value: float (upper bound)
    Returns:
        np.ndarray: clipped array
    """
    arr = np.array(arr)
    return np.clip(arr, min_value, max_value)
