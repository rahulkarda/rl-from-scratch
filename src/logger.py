import os
import csv
from typing import Dict, Any, List
import numpy as np

class Logger:
    """
    Logger for writing scalar metrics, episode returns, and histograms to disk (CSV).

    Provides:
      - Scalar metric logging: log_scalar, log_scalars
      - Episode return logging: log_episode_return
      - Histogram logging: log_histogram
      - Reading logged data: read_scalars, read_episode_returns, read_histograms
      - Episodic average logging: log_episode_average
      - Scalar step lookup: read_scalar_steps
      - Episode averages reading: read_episode_averages

    CSV file layout:
      - scalars.csv: step, name, value         # Each scalar metric (loss, epsilon, etc) per step
      - episode_returns.csv: episode, return   # Total reward per episode
      - histograms.csv: step, name, values     # Array metrics (weights, Q-values) as comma-separated strings
      - episode_averages.csv: episode, name, average, count  # Per-episode averages (e.g. reward, loss)

    Reading methods:
      - read_scalars(): returns list of dicts {step, name, value} for all scalar metrics
      - read_episode_returns(): returns list of floats (episode returns)
      - read_histograms(): returns list of dicts {step, name, values (np.ndarray)}
      - read_episode_averages(): returns list of dicts {episode, name, average, count}
      - read_scalar_steps(name): returns sorted list of (step, value) pairs for a given scalar

    Example usage:
        logger = Logger(log_dir="logs/test_run")
        logger.log_scalar("loss", 0.32, step=10)
        logger.log_episode_return(42.0, episode=3)
        logger.log_scalars({"loss": 0.32, "epsilon": 0.13}, step=10)
        logger.log_histogram("weights", np.array([1,2,3]), step=10)
        logger.log_episode_average("reward", [1.0, 2.0, 3.0], episode=3)
        # Reading logged data:
        scalars = logger.read_scalars()
        returns = logger.read_episode_returns()
        histos = logger.read_histograms()
        avgs = logger.read_episode_averages()
        loss_steps = logger.read_scalar_steps("loss")
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
        self.histogram_path = os.path.join(log_dir, "histograms.csv")
        self.episode_avg_path = os.path.join(log_dir, "episode_averages.csv")
        self._init_scalar_file()
        self._init_returns_file()
        self._init_histogram_file()
        self._init_episode_avg_file()

    # --- CSV file initializers ---
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

    def _init_histogram_file(self):
        """
        Create histograms.csv file with header if it does not exist.
        Format: step, name, values (comma-separated string)
        """
        if not os.path.isfile(self.histogram_path):
            with open(self.histogram_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "name", "values"])

    def _init_episode_avg_file(self):
        """
        Create episode_averages.csv file with header if it does not exist.
        Format: episode, name, average, count
        """
        if not os.path.isfile(self.episode_avg_path):
            with open(self.episode_avg_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "name", "average", "count"])

    # --- Scalar logging ---
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a single scalar metric (e.g., loss, epsilon) for a given step.

        Args:
            name (str): Metric name
            value (float): Metric value
            step (int): Step index
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, self._format_value(value)])

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        """
        Log multiple scalar metrics for a given step.

        Args:
            scalars (dict): Mapping from metric name to value
            step (int): Step index
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, self._format_value(value)])

    def _format_value(self, value: Any) -> str:
        """
        Format value for csv: ensures float is not scientific notation and is string.
        """
        if isinstance(value, float):
            return str(value)
        return str(value)

    # --- Episode return logging ---
    def log_episode_return(self, episode_return: float, episode: int):
        """
        Log episode return (total reward) for a given episode.

        Args:
            episode_return (float): Total reward
            episode (int): Episode index
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, self._format_value(episode_return)])

    # --- Histogram logging ---
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log a histogram (array of values) as a comma-separated string for a given step.

        Args:
            name (str): Metric name
            values (np.ndarray): Array of values
            step (int): Step index
        """
        vals_str = ','.join([self._format_value(v) for v in values.tolist()])
        with open(self.histogram_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, vals_str])

    def log_episode_average(self, name: str, values: List[float], episode: int):
        """
        Log average (mean) of values for a given episode, with count.

        Args:
            name (str): Metric name
            values (list): Values to average
            episode (int): Episode index
        """
        avg = float(np.mean(values)) if values else 0.0
        count = len(values)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, self._format_value(avg), count])

    # --- Reading methods ---
    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all scalar metrics from scalars.csv.

        Returns:
            list of dicts: {step, name, value}
        """
        scalars = []
        with open(self.scalar_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scalars.append({
                        "step": int(row["step"]),
                        "name": row["name"],
                        "value": float(row["value"])
                    })
                except Exception:
                    continue
        return scalars

    def read_episode_returns(self) -> List[float]:
        """
        Read episode returns (total rewards) from episode_returns.csv.

        Returns:
            list of floats: episode returns
        """
        returns = []
        with open(self.returns_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    returns.append(float(row["return"]))
                except Exception:
                    continue
        return returns

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read histograms from histograms.csv.

        Returns:
            list of dicts: {step, name, values (np.ndarray)}
        """
        histos = []
        with open(self.histogram_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    values = np.array([float(v) for v in row["values"].split(',') if v])
                    histos.append({
                        "step": int(row["step"]),
                        "name": row["name"],
                        "values": values
                    })
                except Exception:
                    continue
        return histos

    def read_episode_averages(self) -> List[Dict[str, Any]]:
        """
        Read episode averages (mean metric per episode) from episode_averages.csv.

        Returns:
            list of dicts: {episode, name, average, count}
        """
        avgs = []
        with open(self.episode_avg_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    avgs.append({
                        "episode": int(row["episode"]),
                        "name": row["name"],
                        "average": float(row["average"]),
                        "count": int(row["count"])
                    })
                except Exception:
                    continue
        return avgs

    def read_scalar_steps(self, name: str) -> List[tuple]:
        """
        Read all (step, value) pairs for a given scalar metric name, sorted by step.

        Returns:
            list of (step, value) tuples
        """
        pairs = []
        with open(self.scalar_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if row["name"] != name:
                        continue
                    step = int(row["step"])
                    value = float(row["value"])
                except Exception:
                    continue
                pairs.append((step, value))
        pairs.sort()
        return pairs
