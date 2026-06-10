"""
Basic utilities for RL experiments: seeding, running/moving averages, and Polyak averaging.

Rationale:
- set_seed: Reproducibility is critical in RL due to noisy training and variance. Sets seeds for Python, NumPy, and PyTorch.
- seed_everything: Sets seeds for Python, NumPy, PyTorch, AND Gymnasium environments (if provided). Ensures global reproducibility for full RL setup.
- running_average: Useful for smoothing reward curves or losses over time. Computes cumulative average (up to each point).
- moving_average: Computes average over a fixed window. Used for plotting recent episode returns and smoothing metrics.
- soft_update: Polyak averaging for target networks, needed in Double DQN/DDPG/SAC. Simple utility for updating target model parameters.
- flatten_dict: Flattens nested dictionaries for logging (e.g., metrics) or serialization. Converts {a: {b: 1}} to {'a.b': 1}.

Examples:
    set_seed(42)
    ra = running_average([1, 2, 3, 4])  # array([1., 1.5, 2., 2.5])
    ma = moving_average([1, 2, 3, 4, 5], window_size=3)  # array([2., 3., 4.])
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

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value to set.
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

    Args:
        seed (int): Seed value to set.
        env (optional): Gymnasium environment. If provided, env is seeded via env.reset(seed=seed).

    Example:
        seed_everything(42, env=my_env)
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


def running_average(values):
    """
    Compute running (cumulative) average for a sequence.
    Each value is the average of all previous values up to that index.

    Args:
        values: Sequence of numbers (list, np.ndarray).

    Returns:
        np.ndarray of running averages (same length as values).

    Example:
        running_average([1, 2, 3, 4])
        # returns array([1., 1.5, 2., 2.5])
    """
    values = np.array(values, dtype=float)
    if values.size == 0:
        return np.array([])
    cumsum = np.cumsum(values)
    return cumsum / (np.arange(1, values.size + 1))


def moving_average(values, window_size: int):
    """
    Compute simple moving average over a list or array.

    Returns a sequence of averages where each average is computed over a sliding window of length `window_size`.

    Args:
        values: Sequence of numbers (list, np.ndarray).
        window_size: Size of window (int).

    Returns:
        np.ndarray of moving averages (len = len(values) - window_size + 1)

    Example:
        moving_average([1, 2, 3, 4, 5], window_size=3)
        # returns array([2., 3., 4.])
    """
    values = np.array(values, dtype=float)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if values.size < window_size:
        return np.array([])
    cumsum = np.cumsum(values)
    cumsum[window_size:] -= cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    Polyak averaging for target network updates.
    Copies parameters from `source` to `target` using:
        target_param = tau * source_param + (1 - tau) * target_param

    Args:
        target (torch.nn.Module): Target network to update.
        source (torch.nn.Module): Source network (e.g., online network).
        tau (float): Mixing factor (0 < tau <= 1). tau=1 means hard update.

    Example:
        # For DQN target updates:
        soft_update(target_net, source_net, tau=0.005)
        # Hard update:
        soft_update(target_net, source_net, tau=1.0)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a nested dictionary.
    For logging and serialization, converts {'a': {'b': 1}, 'c': 2} to {'a.b': 1, 'c': 2}.
    Typical use: flatten RL metrics dict before logging to CSV or tensorboard.

    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): Prefix for keys (used internally).
        sep (str): Separator for key levels.

    Returns:
        dict: Flattened dictionary.

    Example:
        flatten_dict({'loss': 0.1, 'stats': {'mean': 1, 'std': 2}})
        # {'loss': 0.1, 'stats.mean': 1, 'stats.std': 2}
        # logger.log_scalars(flatten_dict(metrics), step=100)
    """
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
