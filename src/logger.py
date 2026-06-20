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

    # --- Episode return logging ---
    def log_episode_return(self, episode_return: float, episode: int):
        """
        Log episode return (sum of rewards) for a given episode.

        Args:
            episode_return (float): Return value
            episode (int): Episode index
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, self._format_value(episode_return)])

    # --- Episode average logging ---
    def log_episode_average(self, name: str, values: Any, episode: int):
        """
        Log the average of a metric (e.g. reward, loss) over an episode.
        Useful for tracking per-episode averages (not just returns).

        Args:
            name (str): Metric name
            values (list/array): Sequence of values for the episode
            episode (int): Episode index
        """
        arr = np.array(values, dtype=float)
        avg = float(np.mean(arr)) if arr.size > 0 else 0.0
        count = int(arr.size)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, f"{avg:.6f}", count])

    # --- Histogram logging ---
    def log_histogram(self, name: str, values: Any, step: int):
        """
        Log a histogram/distribution for a given step.
        Values are recorded as a comma-separated string.

        Args:
            name (str): Metric name
            values (array-like): Distribution values (e.g. weights, Q-values)
            step (int): Step index
        """
        arr = np.array(values).flatten()
        str_values = ','.join(f"{float(v):.6f}" for v in arr)
        with open(self.histogram_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, str_values])

    # --- Value formatting ---
    def _format_value(self, value: Any) -> Any:
        """
        Format float values for CSV output: always round to 6 decimals for consistent logs.
        """
        if isinstance(value, float):
            return f"{value:.6f}"
        elif isinstance(value, int):
            return value
        elif isinstance(value, np.floating):
            return f"{float(value):.6f}"
        elif isinstance(value, np.integer):
            return int(value)
        else:
            return value

    # --- Reading logs ---
    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all logged scalar metrics from scalars.csv.

        Returns:
            List[dict]: Each dict contains step, name, value.
        """
        scalars = []
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
        Read all logged episode returns from episode_returns.csv.

        Returns:
            List[dict]: Each dict contains episode, return.
        """
        returns = []
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

    def read_episode_averages(self) -> List[Dict[str, Any]]:
        """
        Read all logged episode averages from episode_averages.csv.

        Returns:
            List[dict]: Each dict contains episode, name, average, count.
        """
        episode_avgs = []
        with open(self.episode_avg_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episode = int(row["episode"])
                    name = row["name"]
                    avg = float(row["average"])
                    count = int(row["count"])
                except Exception:
                    continue
                episode_avgs.append({"episode": episode, "name": name, "average": avg, "count": count})
        return episode_avgs

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read all logged histograms from histograms.csv.

        Returns:
            List[dict]: Each dict contains step, name, values (np.array).
        """
        histos = []
        with open(self.histogram_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row["step"])
                    name = row["name"]
                    values_str = row["values"]
                    values = np.array([float(v) for v in values_str.split(",") if v.strip() != ""])
                except Exception:
                    continue
                histos.append({"step": step, "name": name, "values": values})
        return histos
