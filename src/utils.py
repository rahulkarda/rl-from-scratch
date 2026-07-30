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
- log_mean_exp: Compute log mean exp for numerical stability (NEW)

Usage:
    from utils import set_seed, moving_average, soft_update, compute_gae_advantages
    set_seed(42)
    avg = moving_average([1,2,3,4], window_size=2)
    soft_update(target_net, source_net, tau=0.005)
    adv = compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    clipped = clip_array_values([1, 2, 10], min_value=0, max_value=5)
    norm = compute_l2_norm([1, 2, 2])  # returns 3.0
    lme = log_mean_exp([1.0, 2.0, 3.0])  # returns log(mean(exp([1,2,3])))

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
        values: sequence
        alpha: smoothing factor (0-1)
    Returns:
        np.ndarray: EMA array (same length)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    out = np.empty(values.size)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    return out

# === Polyak Averaging ===
def soft_update(target_net, source_net, tau: float):
    """
    Polyak averaging update for target_net parameters.
    Args:
        target_net: torch.nn.Module (target)
        source_net: torch.nn.Module (source)
        tau: float (0-1)
    """
    for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

# === Dict Flattening ===
def flatten_dict(d, parent_key='', sep='.'):  # for logging
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# === Reward Stats ===
def compute_reward_stats(rewards):
    """
    Compute mean, std, min, max, and median for a reward sequence.
    Args:
        rewards: sequence
    Returns:
        dict: stats
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards))
    }


def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a sequence.
    Args:
        values: sequence
        quantiles: list of floats (0-1)
    Returns:
        dict: {q: value}
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {q: 0.0 for q in quantiles}
    out = {}
    for q in quantiles:
        out[q] = float(np.quantile(values, q))
    return out


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

# === Normalization ===
def min_max_normalize(arr):
    """
    Normalize array to [0, 1].
    Args:
        arr: sequence or np.ndarray
    Returns:
        np.ndarray: normalized array
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.array([])
    minv = np.min(arr)
    maxv = np.max(arr)
    if maxv == minv:
        return np.zeros_like(arr)
    return (arr - minv) / (maxv - minv)

# === GAE Advantage ===
def compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Compute GAE (Generalized Advantage Estimation).
    Args:
        rewards: sequence
        values: sequence
        next_values: sequence
        dones: sequence (bool)
        gamma: float
        lam: float
    Returns:
        np.ndarray: advantages
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    next_values = np.array(next_values, dtype=float)
    dones = np.array(dones, dtype=float)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    adv = np.zeros_like(deltas)
    gae = 0.0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv

# === Sequence Utilities ===
def chunked(seq, batch_size):
    """
    Split sequence into fixed-size batches.
    Args:
        seq: sequence
        batch_size: int
    Returns:
        generator of batches
    """
    seq = list(seq)
    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size]


def compute_discounted_sum(rewards, gamma):
    """
    Compute discounted sum over a sequence.
    Args:
        rewards: sequence
        gamma: float
    Returns:
        float: discounted sum
    """
    rewards = np.array(rewards, dtype=float)
    total = 0.0
    for r in reversed(rewards):
        total = r + gamma * total
    return total


def is_monotonic(seq):
    """
    Check if sequence is monotonic (increasing or decreasing).
    Args:
        seq: sequence
    Returns:
        bool
    """
    seq = list(seq)
    if len(seq) < 2:
        return True
    inc = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
    dec = all(seq[i] >= seq[i+1] for i in range(len(seq)-1))
    return inc or dec


def compute_mean_and_std(seq):
    """
    Compute mean and std for sequence.
    Args:
        seq: sequence
    Returns:
        (mean, std)
    """
    seq = np.array(seq, dtype=float)
    if seq.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(seq)), float(np.std(seq)))

# === Transition Utilities ===
def transitions_to_dicts(transitions):
    """
    Convert list of Transition objects to list of dicts.
    Args:
        transitions: list of Transition
    Returns:
        list of dicts
    """
    return [t.__dict__ for t in transitions]

# === Padding Utility ===
def pad_sequence_to_length(seq, length, pad_value=0):
    """
    Pad sequence to fixed length with pad_value.
    Args:
        seq: sequence
        length: desired length
        pad_value: value to pad
    Returns:
        list
    """
    seq = list(seq)
    if len(seq) == 0:
        return []
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_value] * (length - len(seq))

# === Elementwise Min/Max Utility ===
def elementwise_min_max(arr1, arr2, mode="min"):
    """
    Compute elementwise min or max between two arrays.
    Args:
        arr1: array-like
        arr2: array-like
        mode: "min" or "max"
    Returns:
        np.ndarray
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    if mode == "min":
        return np.minimum(arr1, arr2)
    else:
        return np.maximum(arr1, arr2)

# === Clipping Utility ===
def clip_array_values(arr, min_value, max_value):
    """
    Clip array values to [min, max].
    Args:
        arr: array-like
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

# === Log Mean Exp Utility ===
def log_mean_exp(arr):
    """
    Compute log(mean(exp(arr))) in a numerically stable way.
    Useful for soft value estimation, SAC, etc.
    Args:
        arr: array-like
    Returns:
        float
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return float('-inf')
    m = np.max(arr)
    return float(m + np.log(np.mean(np.exp(arr - m))))
