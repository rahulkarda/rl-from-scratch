import os
import csv
from typing import Dict, Any, List

class Logger:
    """
    Logger for writing scalar metrics and episode returns to disk (CSV).

    Usage:
        logger = Logger(log_dir="logs/test_run")
        logger.log_scalar("loss", 0.32, step=10)
        logger.log_episode_return(42.0, episode=3)
        logger.log_scalars({"loss": 0.32, "epsilon": 0.13}, step=10)
        # Reading logged data:
        scalars = logger.read_scalars()
        returns = logger.read_episode_returns()
    """
    def __init__(self, log_dir: str):
        """
        Initialize logger and create CSV files if they do not exist.

        Args:
            log_dir (str): Directory to store logs.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.scalar_path = os.path.join(log_dir, "scalars.csv")
        self.returns_path = os.path.join(log_dir, "episode_returns.csv")
        self._init_scalar_file()
        self._init_returns_file()

    def _init_scalar_file(self):
        """
        Create scalars.csv file with header if it does not exist.
        """
        if not os.path.isfile(self.scalar_path):
            with open(self.scalar_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "name", "value"])

    def _init_returns_file(self):
        """
        Create episode_returns.csv file with header if it does not exist.
        """
        if not os.path.isfile(self.returns_path):
            with open(self.returns_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])

    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a single scalar metric (e.g., loss, epsilon) for a given step.
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, self._format_value(value)])

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        """
        Log multiple scalar metrics for a given step.
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, self._format_value(value)])

    def log_episode_return(self, episode_return: float, episode: int):
        """
        Log episode return (sum of rewards) for a given episode.
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, self._format_value(episode_return)])

    def _format_value(self, value: Any) -> Any:
        """
        Format float values for CSV output: always round to 6 decimals for consistent logs.
        """
        # Ensure every float is formatted as a string with 6 decimals (not scientific notation)
        if isinstance(value, float):
            return f"{value:.6f}"
        elif isinstance(value, int):
            return value
        elif isinstance(value, str):
            return value
        # Try to convert numpy float types
        try:
            import numpy as np
            if isinstance(value, np.floating):
                return f"{float(value):.6f}"
        except Exception:
            pass
        return value

    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read logged scalar metrics from scalars.csv.
        Returns a list of dicts: [{"step": int, "name": str, "value": float}, ...]
        """
        scalars = []
        if not os.path.isfile(self.scalar_path):
            return scalars
        with open(self.scalar_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row["step"])
                    name = row["name"]
                    value = float(row["value"])
                except Exception:
                    continue
                scalars.append({"step": step, "name": name, "value": value})
        return scalars

    def read_episode_returns(self) -> List[Dict[str, Any]]:
        """
        Read logged episode returns from episode_returns.csv.
        Returns a list of dicts: [{"episode": int, "return": float}, ...]
        """
        returns = []
        if not os.path.isfile(self.returns_path):
            return returns
        with open(self.returns_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episode = int(row["episode"])
                    ret = float(row["return"])
                except Exception:
                    continue
                returns.append({"episode": episode, "return": ret})
        return returns
