"""
Basic utilities for RL experiments: seeding, running/moving averages, moving std, and Polyak averaging.

Rationale:
- set_seed: Reproducibility is critical in RL due to noisy training and variance. Sets seeds for Python, NumPy, and PyTorch.
- seed_everything: Sets seeds for Python, NumPy, PyTorch, AND Gymnasium environments (if provided). Ensures global reproducibility for full RL setup.
- running_average: Useful for smoothing reward curves or losses over time. Computes cumulative average (up to each point).
- moving_average: Computes average over a fixed window. Used for plotting recent episode returns and smoothing metrics.
- moving_std: Computes standard deviation over a fixed window. Useful for plotting reward curve uncertainty bands (shaded region).
- soft_update: Polyak averaging for target networks, needed in Double DQN/DDPG/SAC. Simple utility for updating target model parameters.
- flatten_dict: Flattens nested dictionaries for logging (e.g., metrics) or serialization. Converts {a: {b: 1}} to {'a.b': 1}.

Examples:
    set_seed(42)
    ra = running_average([1, 2, 3, 4])  # array([1., 1.5, 2., 2.5])
    ma = moving_average([1, 2, 3, 4, 5], window_size=3)  # array([2., 3., 4.])
    ms = moving_std([1, 2, 3, 4, 5], window_size=3)  # array([0.8165, 0.8165, 0.8165])
    # Polyak averaging for target networks:
    soft_update(target_net, source_net, tau=0.005)
    # tau=1.0 gives a hard update (copy params exactly)
    # Global seeding (including Gym):
    seed_everything(42, env=env)
    # Flatten metrics dict:
    flatten_dict({'loss': 0.1, 'stats': {'mean': 1, 'std': 2}})  # {'loss': 0.1, 'stats.mean': 1, 'stats.std': 2}
    # Real-world: flatten_dict before logger.log_scalars for CSV:
    metrics = {'loss': 0.1, 'stats': {'mean': 1, 'std': 2}}
    logger.log_scalars(flatten_dict(metrics), step=100)

These functions are intentionally minimal and avoid dependencies beyond numpy and torch.
"""
import random
import numpy as np
import torch

# --- Seeding utilities ---
def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
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
    Set random seeds for Python, NumPy, PyTorch, AND Gymnasium env (if provided).
    Ensures global reproducibility for RL experiments.
    """
    set_seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if env is not None:
        # Gymnasium: seed via reset(seed=seed)
        try:
            env.reset(seed=seed)
        except Exception:
            # Older gym envs may use env.seed(seed)
            try:
                env.seed(seed)
            except Exception:
                pass

# --- Moving/running statistics ---
def running_average(values):
    """
    Compute running (cumulative) average for a sequence.
    Each value is the average of all previous values up to that index.
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])  # Fix: return empty array for empty input
    cumsum = np.cumsum(values)
    return cumsum / (np.arange(1, values.size + 1))


def moving_average(values, window_size: int):
    """
    Compute simple moving average over a list or array.
    Returns a sequence of averages where each average is computed over a sliding window of length `window_size`.
    """
    values = np.array(values, dtype=float)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if values.size < window_size:
        return np.array([])
    cumsum = np.cumsum(values)
    cumsum[window_size:] -= cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def moving_std(values, window_size: int):
    """
    Compute moving (sliding window) standard deviation over a list or array.
    Returns a sequence of stds where each is computed over a sliding window of length `window_size`.
    """
    values = np.array(values, dtype=float)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if values.size < window_size:
        return np.array([])
    out = np.empty(values.size - window_size + 1)
    for i in range(out.size):
        out[i] = values[i:i+window_size].std()
    return out

# --- Polyak averaging ---
def soft_update(target_net, source_net, tau: float):
    """
    Polyak averaging (soft update) for target network parameters.
    Each parameter in target_net is updated:
        target = tau * source + (1 - tau) * target
    tau=1.0 gives a hard update (copy).
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

# --- Dict flattening ---
def flatten_dict(d, parent_key='', sep='.'): 
    """
    Flatten nested dictionaries. E.g. {'a': {'b': 1}} -> {'a.b': 1}
    Args:
        d (dict): Input dict.
        parent_key (str): Prefix for keys (used recursively).
        sep (str): Separator to use.
    Returns:
        dict: Flattened dict.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# --- Utility: compute total steps from scalars log ---
def total_steps_from_scalars(scalars):
    """
    Compute total steps from a list of scalars log entries (as read from logger.read_scalars()).
    Returns the maximum step value, or 0 if empty.
    Useful for restoring training progress or plotting.

    Args:
        scalars (list of dict): Each dict must have 'step' (int).
    Returns:
        int: Maximum step value in scalars, or 0 if empty.
    """
    if not scalars:
        return 0
    return max(entry.get('step', 0) for entry in scalars)
