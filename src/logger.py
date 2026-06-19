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

    Example usage:
        logger = Logger(log_dir="logs/test_run")
        logger.log_scalar("loss", 0.32, step=10)
        logger.log_episode_return(42.0, episode=3)
        logger.log_scalars({"loss": 0.32, "epsilon": 0.13}, step=10)
        logger.log_histogram("weights", np.array([1,2,3]), step=10)
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
        self._init_scalar_file()
        self._init_returns_file()
        self._init_histogram_file()

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
        elif isinstance(value, str):
            return value
        try:
            import numpy as np
            if isinstance(value, np.floating):
                return f"{float(value):.6f}"
        except Exception:
            pass
        return value

    # --- Reading logged data ---
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
        Read episode returns from episode_returns.csv.
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

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read logged histograms from histograms.csv.
        Returns a list of dicts: [{"step": int, "name": str, "values": np.ndarray}, ...]
        """
        histos = []
        if not os.path.isfile(self.histogram_path):
            return histos
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
