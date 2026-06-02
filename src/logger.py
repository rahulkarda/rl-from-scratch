import os
import csv
from typing import Dict, Any

class Logger:
    """
    Simple logger that writes scalar metrics and episode returns to disk (CSV).

    Usage:
        logger = Logger(log_dir="logs/test_run")
        logger.log_scalar("loss", 0.32, step=10)
        logger.log_episode_return(42.0, episode=3)
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.scalar_path = os.path.join(log_dir, "scalars.csv")
        self.returns_path = os.path.join(log_dir, "episode_returns.csv")
        self._init_scalar_file()
        self._init_returns_file()

    def _init_scalar_file(self):
        if not os.path.isfile(self.scalar_path):
            with open(self.scalar_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "name", "value"])

    def _init_returns_file(self):
        if not os.path.isfile(self.returns_path):
            with open(self.returns_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])

    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a scalar metric (e.g., loss, epsilon) at a given step.
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, value])

    def log_episode_return(self, episode_return: float, episode: int):
        """
        Log episode return (reward sum) for a given episode.
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, episode_return])

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        """
        Log multiple scalar metrics at given step.
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, value])
