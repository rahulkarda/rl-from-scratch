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
- compute_l2_norm: Compute L2 norm of a vector or array (NEW)

Usage:
    from utils import set_seed, moving_average, soft_update, compute_gae_advantages
    set_seed(42)
    avg = moving_average([1,2,3,4], window_size=2)
    soft_update(target_net, source_net, tau=0.005)
    adv = compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    clipped = clip_array_values([1, 2, 10], min_value=0, max_value=5)
    norm = compute_l2_norm([1, 2, 2])  # returns 3.0

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
        np.ndarray: EMA (same length as values)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema

# === Polyak update ===
def soft_update(target_net, source_net, tau: float):
    """
    Polyak averaging for target network parameters.
    Args:
        target_net: torch.nn.Module
        source_net: torch.nn.Module
        tau: float, interpolation factor (0 < tau <= 1)
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# === Dict flatten ===
def flatten_dict(d, parent_key='', sep='/'):
    """
    Flatten nested dict for logging.
    Args:
        d: dict
        parent_key: str
        sep: str
    Returns:
        dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# === Reward stats ===
def compute_reward_stats(rewards):
    """
    Compute mean, median, std, min, max for reward sequence.
    Args:
        rewards: sequence
    Returns:
        dict
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(rewards)),
        'median': float(np.median(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards))
    }

# === Quantiles ===
def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles from a sequence.
    Args:
        values: sequence
        quantiles: sequence of floats (0-1)
    Returns:
        dict {q: value}
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {q: 0.0 for q in quantiles}
    return {q: float(np.quantile(values, q)) for q in quantiles}


def compute_median(values):
    """
    Median utility (for reward stats).
    Args:
        values: sequence
    Returns:
        float
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.median(values))

# === Normalization ===
def min_max_normalize(arr):
    """
    Scale array to [0, 1].
    Args:
        arr: sequence or np.ndarray
    Returns:
        np.ndarray
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.array([])
    minv = np.min(arr)
    maxv = np.max(arr)
    if maxv - minv == 0:
        return np.zeros_like(arr)
    return (arr - minv) / (maxv - minv)

# === GAE ===
def compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    Args:
        rewards: [T]
        values: [T]
        next_values: [T]
        dones: [T]
        gamma: float
        lam: float
    Returns:
        np.ndarray: advantages [T]
    """
    T = len(rewards)
    adv = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv

# === Chunked ===
def chunked(seq, chunk_size):
    """
    Split sequence into fixed-size batches (chunks).
    Args:
        seq: sequence
        chunk_size: int
    Returns:
        list of chunks
    """
    seq = list(seq)
    if chunk_size < 1:
        return []
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

# === Discounted sum ===
def compute_discounted_sum(rewards, gamma=0.99):
    """
    Compute discounted sum over a sequence (for returns).
    Args:
        rewards: sequence
        gamma: float
    Returns:
        float
    """
    out = 0.0
    for r in reversed(rewards):
        out = r + gamma * out
    return out

# === Monotonic check ===
def is_monotonic(seq, mode='increasing'):
    """
    Check if sequence is monotonic increasing or decreasing.
    Args:
        seq: sequence
        mode: 'increasing'|'decreasing'
    Returns:
        bool
    """
    seq = list(seq)
    if len(seq) < 2:
        return True
    if mode == 'increasing':
        return all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
    elif mode == 'decreasing':
        return all(seq[i] >= seq[i+1] for i in range(len(seq)-1))
    else:
        raise ValueError("mode must be 'increasing' or 'decreasing'")

# === Mean and std ===
def compute_mean_and_std(seq):
    """
    Compute mean and std for a sequence.
    Args:
        seq: sequence
    Returns:
        tuple (mean, std)
    """
    arr = np.array(seq, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(arr)), float(np.std(arr)))

# === Transitions to dicts ===
def transitions_to_dicts(transitions):
    """
    Convert list of Transition objects to list of dicts.
    Args:
        transitions: list of Transition objects
    Returns:
        list of dicts
    """
    return [t.__dict__ if hasattr(t, '__dict__') else dict(t) for t in transitions]

# === Pad sequence ===
def pad_sequence_to_length(seq, length, pad_value=0):
    """
    Pad sequence to fixed length with pad_value.
    Args:
        seq: sequence
        length: int
        pad_value: value to pad
    Returns:
        list (length)
    """
    seq = list(seq)
    if length <= 0:
        return []
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_value] * (length - len(seq))

# === Elementwise min/max ===
def elementwise_min_max(a, b, mode='min'):
    """
    Compute elementwise min or max between two arrays/lists.
    Args:
        a: sequence or np.ndarray
        b: sequence or np.ndarray
        mode: 'min' or 'max'
    Returns:
        np.ndarray
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    if mode == 'min':
        return np.minimum(a_arr, b_arr)
    elif mode == 'max':
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

# === L2 Norm Utility ===
def compute_l2_norm(vec):
    """
    Compute L2 norm (Euclidean norm) of a vector or array.
    Args:
        vec: sequence or np.ndarray
    Returns:
        float: L2 norm
    """
    arr = np.array(vec, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr, ord=2))
