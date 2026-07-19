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
- compute_mean_and_std: Compute mean and std for sequence (NEW)

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
    with torch.no_grad():
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

# === Dict Flattening ===
def flatten_dict(d, parent_key='', sep='/'):
    """
    Flatten a nested dictionary for logging.
    Args:
        d: dict
        parent_key: root key
        sep: separator
    Returns:
        dict: flattened
    """
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
    Compute mean, std, min, max for reward sequence.
    Args:
        rewards: sequence
    Returns:
        dict: {'mean', 'std', 'min', 'max'}
    """
    rewards = np.array(rewards, dtype=float)
    if rewards.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
    }

# === Quantiles ===
def compute_quantiles(values, quantiles):
    """
    Compute arbitrary quantiles for a sequence.
    Args:
        values: sequence
        quantiles: list of quantiles (0..1)
    Returns:
        dict: quantile -> value
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return {q: 0.0 for q in quantiles}
    return {q: float(np.quantile(values, q)) for q in quantiles}

# === Median ===
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
    Scale values to [0, 1] with min-max normalization.
    Args:
        values: sequence
    Returns:
        np.ndarray: normalized array
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
def compute_gae_advantages(rewards, values, next_values, dones, gamma, lam):
    """
    Generalized Advantage Estimation (GAE) for policy gradient methods.
    Args:
        rewards: np.ndarray or list (length T)
        values: np.ndarray or list (length T)
        next_values: np.ndarray or list (length T), V(s') for each step
        dones: np.ndarray or list (length T) (bool)
        gamma: float (discount)
        lam: float (lambda)
    Returns:
        np.ndarray: advantage estimates (length T)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    next_values = np.array(next_values, dtype=float)
    dones = np.array(dones, dtype=bool)
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (not dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (not dones[t])
        advantages[t] = gae
    return advantages

# === Chunked batching ===
def chunked(seq, chunk_size: int):
    """
    Yield successive chunk_size batches from seq.
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
def compute_discounted_sum(rewards, gamma):
    """
    Compute discounted sum of rewards.
    Args:
        rewards: sequence
        gamma: discount factor
    Returns:
        np.ndarray: discounted sums
    """
    rewards = np.array(rewards, dtype=float)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out

# === Monotonicity check ===
def is_monotonic(values, mode="increasing"):
    """
    Check if sequence is monotonic.
    mode: 'increasing', 'decreasing', 'strict_increasing', 'strict_decreasing'
    Returns True if monotonic, else False.
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

# === Mean and Std utility ===
def compute_mean_and_std(values):
    """
    Compute mean and standard deviation for a sequence.
    Args:
        values: sequence of numbers
    Returns:
        tuple (mean, std): both floats
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))
