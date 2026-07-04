import os
import csv
from typing import Dict, Any, List
import numpy as np
import json

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

      - JSON logging: log_json, read_json  # NEW

    CSV file layout:
      - scalars.csv: step, name, value         # Each scalar metric (loss, epsilon, etc) per step
      - episode_returns.csv: episode, return   # Total reward per episode
      - histograms.csv: step, name, values     # Array metrics (weights, Q-values) as comma-separated strings
      - episode_averages.csv: episode, name, average, count  # Per-episode averages (e.g. reward, loss)
    JSON file layout:
      - logs.json: list of dicts (arbitrary experiment metadata, config, etc)

    Reading methods:
      - read_scalars(): list[dict] {step, name, value} for all scalar metrics
      - read_episode_returns(): list[float] (episode returns)
      - read_histograms(): list[dict] {step, name, values (np.ndarray)}
      - read_episode_averages(): list[dict] {episode, name, average, count}
      - read_scalar_steps(name): sorted list[(step, value)] for a given scalar
      - read_json(): list[dict] for all JSON logs

    Example usage:
        logger = Logger(log_dir="logs/test_run")
        logger.log_scalar("loss", 0.32, step=10)
        logger.log_episode_return(42.0, episode=3)
        logger.log_scalars({"loss": 0.32, "epsilon": 0.13}, step=10)
        logger.log_histogram("weights", np.array([1,2,3]), step=10)
        logger.log_episode_average("reward", [1.0, 2.0, 3.0], episode=3)
        logger.log_json({"config": {"lr": 1e-3, "batch_size": 32}})
        # Reading logged data:
        scalars = logger.read_scalars()
        returns = logger.read_episode_returns()
        histos = logger.read_histograms()
        avgs = logger.read_episode_averages()
        loss_steps = logger.read_scalar_steps("loss")
        json_logs = logger.read_json()
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
        self.json_path = os.path.join(log_dir, "logs.json")  # NEW
        self._init_all_files()

    def _init_all_files(self):
        """
        Initialize all CSV files with correct headers if not present.
        """
        self._init_csv_file(self.scalar_path, ["step", "name", "value"])
        self._init_csv_file(self.returns_path, ["episode", "return"])
        self._init_csv_file(self.histogram_path, ["step", "name", "values"])
        self._init_csv_file(self.episode_avg_path, ["episode", "name", "average", "count"])
        self._init_json_file(self.json_path)

    def _init_csv_file(self, path: str, header: List[str]):
        """
        Create CSV file with header if not exists.
        Args:
            path (str): CSV file path
            header (list[str]): CSV column names
        """
        if not os.path.isfile(path):
            with open(path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def _init_json_file(self, path: str):
        """
        Create JSON log file if not exists; initializes as empty list.
        """
        if not os.path.isfile(path):
            with open(path, "w") as f:
                json.dump([], f)

    # --- Scalar logging ---
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a single scalar metric (e.g., loss, epsilon) for a step.
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
            scalars (dict): {metric name: value}
            step (int): Step index
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, self._format_value(value)])

    # --- Episode return logging ---
    def log_episode_return(self, value: float, episode: int):
        """
        Log total episode return for a given episode.
        Args:
            value (float): Episode return
            episode (int): Episode index
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, self._format_value(value)])

    # --- Histogram logging ---
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log a histogram (array metric) for a given step.
        Args:
            name (str): Metric name
            values (np.ndarray): Array of metric values
            step (int): Step index
        """
        values_str = ','.join(str(v) for v in values.tolist())
        with open(self.histogram_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, values_str])

    # --- Episode average logging ---
    def log_episode_average(self, name: str, values: List[float], episode: int):
        """
        Log average of a metric for a given episode.
        Args:
            name (str): Metric name
            values (List[float]): Values to average
            episode (int): Episode index
        """
        avg = float(np.mean(values)) if values else 0.0
        count = len(values)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, self._format_value(avg), count])

    # --- JSON logging ---
    def log_json(self, obj: dict):
        """
        Log a dict of arbitrary metadata/config/events to logs.json.
        Args:
            obj (dict): Metadata/config/experiment info. Must be JSON serializable.
        """
        # Append to list in logs.json
        # Read file, append, write back
        try:
            with open(self.json_path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        logs.append(obj)
        with open(self.json_path, "w") as f:
            json.dump(logs, f, indent=2)

    def read_json(self) -> List[dict]:
        """
        Read all JSON log entries (list of dicts).
        Returns:
            List[dict]: List of logged metadata/config/events.
        """
        try:
            with open(self.json_path, "r") as f:
                logs = json.load(f)
            return logs
        except Exception:
            return []

    # --- Reading methods ---
    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all scalar metrics from scalars.csv.
        Returns:
            List[dict]: [{step, name, value}, ...]
        """
        out = []
        with open(self.scalar_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({
                    "step": int(row["step"]),
                    "name": row["name"],
                    "value": float(row["value"])
                })
        return out

    def read_episode_returns(self) -> List[float]:
        """
        Read episode returns from episode_returns.csv.
        Returns:
            List[float]: List of episode returns.
        """
        out = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(float(row["return"]))
        return out

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read all histogram metrics from histograms.csv.
        Returns:
            List[dict]: [{step, name, values (np.ndarray)}, ...]
        """
        out = []
        with open(self.histogram_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = np.fromstring(row["values"], sep=',')
                out.append({
                    "step": int(row["step"]),
                    "name": row["name"],
                    "values": values
                })
        return out

    def read_episode_averages(self) -> List[Dict[str, Any]]:
        """
        Read episode averages from episode_averages.csv.
        Returns:
            List[dict]: [{episode, name, average, count}, ...]
        """
        out = []
        with open(self.episode_avg_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({
                    "episode": int(row["episode"]),
                    "name": row["name"],
                    "average": float(row["average"]),
                    "count": int(row["count"])
                })
        return out

    def read_scalar_steps(self, name: str) -> List[tuple]:
        """
        Read all steps and values for a given scalar metric name.
        Returns:
            List[(step, value)]: sorted by step
        """
        pairs = []
        with open(self.scalar_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == name:
                    step = int(row["step"])
                    value = float(row["value"])
                    pairs.append((step, value))
        pairs.sort()
        return pairs

    @staticmethod
    def _format_value(value: Any) -> str:
        """
        Format value as string for CSV to avoid scientific notation.
        Args:
            value: scalar value
        Returns:
            str: formatted value
        """
        if isinstance(value, float):
            # Use plain string to avoid CSV scientific notation
            return str(value)
        return str(value)
