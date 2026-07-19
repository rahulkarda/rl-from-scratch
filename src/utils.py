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
- is_monotonic: Check if sequence is monotonic (NEW)

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
        tau: blend factor (0 < tau <= 1)
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

# === Dict flattening ===
def flatten_dict(d, parent_key='', sep='/'):
    """
    Flatten nested dicts for logging and CSV compatibility.
    Args:
        d: dict (possibly nested)
        parent_key: prefix for keys
        sep: separator
    Returns:
        dict: flattened keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# === Reward stats ===
def compute_reward_stats(rewards):
    """
    Summarize reward distribution: mean, std, min, max, median.
    Args:
        rewards: sequence of reward values
    Returns:
        dict: stats summary
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
    }

# === Quantile computation ===
def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles of a sequence.
    Args:
        values: sequence
        quantiles: list of quantile floats in [0,1]
    Returns:
        dict: quantile -> value
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {q: 0.0 for q in quantiles}
    qvals = np.quantile(values, quantiles)
    return {q: float(v) for q, v in zip(quantiles, qvals)}

# === Median computation ===
def compute_median(values):
    """
    Compute median of a sequence.
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
    Scale array to [0, 1] using min-max normalization.
    Args:
        values: sequence
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
def compute_gae_advantages(rewards, values, dones, gamma: float, lam: float):
    """
    Generalized Advantage Estimation (GAE) for policy gradient methods.
    Args:
        rewards: [T] float
        values: [T+1] float (bootstrap value at final step)
        dones: [T] bool
        gamma: discount factor
        lam: GAE lambda (smoothing)
    Returns:
        np.ndarray: advantage estimates [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t+1] * next_nonterminal - values[t]
        advantages[t] = last_adv = delta + gamma * lam * next_nonterminal * last_adv
    return advantages

# === Sequence chunking ===
def chunked(seq, chunk_size: int):
    """
    Split sequence into fixed-size chunks (for batching).
    Args:
        seq: sequence
        chunk_size: chunk length
    Yields:
        list: chunk of chunk_size
    """
    seq = list(seq)
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i+chunk_size]

# === Discounted sum utility ===
def compute_discounted_sum(seq, gamma: float):
    """
    Compute discounted sum over a sequence: sum_t gamma^t * seq[t].
    Args:
        seq: sequence
        gamma: discount factor
    Returns:
        float: discounted sum
    """
    seq = np.array(seq, dtype=float)
    if seq.size == 0:
        return 0.0
    return float(np.dot(seq, gamma ** np.arange(seq.size)))

# === Monotonicity check ===
def is_monotonic(values, mode="increasing"):
    """
    Check if sequence is monotonic.
    Args:
        values: sequence
        mode: 'increasing', 'decreasing', 'strict_increasing', 'strict_decreasing'
    Returns:
        bool: monotonicity result
    """
    values = np.array(values, dtype=float)
    if values.size <= 1:
        return True
    if mode == "increasing":
        return np.all(values[1:] >= values[:-1])
    elif mode == "decreasing":
        return np.all(values[1:] <= values[:-1])
    elif mode == "strict_increasing":
        return np.all(values[1:] > values[:-1])
    elif mode == "strict_decreasing":
        return np.all(values[1:] < values[:-1])
    else:
        raise ValueError("mode must be one of 'increasing', 'decreasing', 'strict_increasing', 'strict_decreasing'")
