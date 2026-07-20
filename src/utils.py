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
        for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
            t_param.data.mul_(1 - tau).add_(tau * s_param.data)

# === Dict flattening ===
def flatten_dict(d, parent_key='', sep='.'):  # for logging
    """
    Flatten a nested dict. Keys are joined with '.'
    Args:
        d: dict (possibly nested)
        parent_key: prefix (internal)
        sep: separator between keys
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
    Compute reward statistics: mean, std, min, max, median, count.
    Args:
        rewards: sequence of rewards
    Returns:
        dict[str, float]
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
        "count": rewards.size
    }

# === Quantiles and median ===
def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a sequence.
    Args:
        values: sequence
        quantiles: list of floats in [0, 1]
    Returns:
        dict[str, float]: quantile value for each
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {str(q): 0.0 for q in quantiles}
    out = {}
    for q in quantiles:
        out[str(q)] = float(np.quantile(values, q))
    return out


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

# === Min-max normalization ===
def min_max_normalize(values):
    """
    Normalize array to [0, 1] range.
    Args:
        values: sequence or array
    Returns:
        np.ndarray: normalized array
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    minv = np.min(values)
    maxv = np.max(values)
    if minv == maxv:
        return np.zeros_like(values)
    return (values - minv) / (maxv - minv)

# === GAE advantage estimation ===
def compute_gae_advantages(rewards, values, next_values, dones, gamma: float, lam: float):
    """
    Generalized Advantage Estimation (GAE).
    Args:
        rewards: array-like, shape (T,)
        values: array-like, shape (T,)
        next_values: array-like, shape (T,)
        dones: array-like, shape (T,) (bool)
        gamma: float
        lam: float (lambda)
    Returns:
        np.ndarray: advantage estimates, shape (T,)
    """
    T = len(rewards)
    adv = np.zeros(T)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    return adv

# === Sequence chunking ===
def chunked(seq, chunk_size: int):
    """
    Split sequence (list, array) into fixed-size chunks.
    Args:
        seq: sequence
        chunk_size: int
    Yields:
        list: chunk of elements
    """
    seq = list(seq)
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i+chunk_size]

# === Discounted sum ===
def compute_discounted_sum(rewards, gamma: float):
    """
    Compute discounted sum of rewards.
    Args:
        rewards: sequence
        gamma: float
    Returns:
        np.ndarray: discounted sum array (same length)
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return np.array([])
    out = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out

# === Monotonicity check ===
def is_monotonic(values, increasing=True):
    """
    Check if sequence is monotonic (increasing or decreasing).
    Args:
        values: sequence
        increasing: bool (default True)
    Returns:
        bool
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return True
    diffs = np.diff(values)
    if increasing:
        return np.all(diffs >= 0)
    else:
        return np.all(diffs <= 0)

# === Mean and std ===
def compute_mean_and_std(values):
    """
    Compute mean and std for a sequence.
    Args:
        values: sequence
    Returns:
        tuple: (mean, std)
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(values)), float(np.std(values)))

# === Transition conversion ===
def transitions_to_dicts(transitions):
    """
    Convert list of Transition dataclass objects to list of dicts.
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
