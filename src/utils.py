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
        tau (float): interpolation factor (0 < tau < 1)
    """
    with torch.no_grad():
        for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

# === Dict flattening ===
def flatten_dict(d, parent_key='', sep='.'):  # For logging nested dicts
    """
    Flatten a nested dict for logging.
    Args:
        d: dict (possibly nested)
        parent_key: str (prefix)
        sep: separator
    Returns:
        dict: flattened dict with composite keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# === Reward stats ===
def compute_reward_stats(rewards):
    """
    Compute mean, std, min, max, median of rewards.
    Args:
        rewards: sequence
    Returns:
        dict: {mean, std, min, max, median}
    """
    r = np.array(rewards, dtype=float)
    if r.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "median": float(np.median(r)),
    }

def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a sequence.
    Args:
        values: sequence
        quantiles: list of float (0-1)
    Returns:
        dict: {q: value}
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {q: 0.0 for q in quantiles}
    return {q: float(np.quantile(arr, q)) for q in quantiles}


def compute_median(values):
    """
    Compute median for a sequence.
    Args:
        values: sequence
    Returns:
        float: median value
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def min_max_normalize(values):
    """
    Normalize array to [0, 1]. Returns empty for empty input.
    Args:
        values: sequence
    Returns:
        np.ndarray: normalized array
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.array([])
    min_v = arr.min()
    max_v = arr.max()
    if min_v == max_v:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)

# === GAE ===
def compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    Args:
        rewards: np.ndarray, shape (steps,)
        values: np.ndarray, shape (steps,)
        next_values: np.ndarray, shape (steps,)
        dones: np.ndarray, shape (steps,)
        gamma: float
        lam: float
    Returns:
        np.ndarray: advantages, shape (steps,)
    """
    adv = np.zeros_like(rewards, dtype=float)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        adv[t] = last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
    return adv

# === Chunking ===
def chunked(seq, chunk_size):
    """
    Split sequence into fixed-size chunks.
    Args:
        seq: sequence
        chunk_size: int
    Returns:
        iterator of chunks (list)
    """
    seq = list(seq)
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i+chunk_size]

# === Discounted sum ===
def compute_discounted_sum(rewards, gamma=0.99):
    """
    Compute discounted sum for a sequence.
    Args:
        rewards: sequence
        gamma: float
    Returns:
        np.ndarray: discounted sum (same length as rewards)
    """
    r = np.array(rewards, dtype=float)
    out = np.zeros_like(r)
    running = 0.0
    for t in reversed(range(len(r))):
        running = r[t] + gamma * running
        out[t] = running
    return out

# === Monotonicity ===
def is_monotonic(seq, increasing=True):
    """
    Check if sequence is monotonic (increasing or decreasing).
    Args:
        seq: sequence
        increasing: bool
    Returns:
        bool
    """
    arr = np.array(seq)
    if arr.size < 2:
        return True
    if increasing:
        return np.all(arr[1:] >= arr[:-1])
    else:
        return np.all(arr[1:] <= arr[:-1])

# === Mean/std ===
def compute_mean_and_std(values):
    """
    Compute mean and std for a sequence. Returns (mean, std).
    Args:
        values: sequence
    Returns:
        tuple: (mean, std)
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))

# === Transition dict conversion ===
def transitions_to_dicts(transitions):
    """
    Convert list of Transition objects (dataclass) to list of dicts.
    Args:
        transitions: list of dataclass Transition
    Returns:
        list of dicts (for serialization or analysis)
    """
    return [t.__dict__ for t in transitions]

# === Pad sequence utility ===
def pad_sequence_to_length(seq, length, pad_value=0):
    """
    Pad sequence to a fixed length with a pad value.
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

# === Elementwise min/max utility ===
def elementwise_min_max(a, b, mode="min"):
    """
    Compute elementwise min or max between two arrays.
    Args:
        a: sequence or np.ndarray
        b: sequence or np.ndarray
        mode: "min" or "max" (default "min")
    Returns:
        np.ndarray: elementwise min or max
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Shape mismatch: {} vs {}".format(a_arr.shape, b_arr.shape))
    if mode == "min":
        return np.minimum(a_arr, b_arr)
    elif mode == "max":
        return np.maximum(a_arr, b_arr)
    else:
        raise ValueError("mode must be 'min' or 'max'")
