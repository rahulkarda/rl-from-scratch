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
                # Avoid trailing newline by writing compact JSON
                f.write(json.dumps([]))

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
    def log_episode_return(self, episode_return: float, episode: int):
        """
        Log episode return (total reward) for a given episode.
        Args:
            episode_return (float): Episode total reward
            episode (int): Episode index
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, self._format_value(episode_return)])

    # --- Histogram logging ---
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log an array metric (weights, Q-values, etc) as comma-separated string.
        Args:
            name (str): Metric name
            values (np.ndarray): Array values
            step (int): Step index
        """
        values_str = ",".join([str(float(x)) for x in np.asarray(values).flatten()])
        with open(self.histogram_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, values_str])

    # --- Episode average logging ---
    def log_episode_average(self, name: str, values: List[float], episode: int):
        """
        Log per-episode average of a metric (e.g., reward, loss).
        Args:
            name (str): Metric name
            values (list[float]): Values to average
            episode (int): Episode index
        """
        avg = float(np.mean(values)) if values else 0.0
        count = len(values)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, self._format_value(avg), count])

    # --- JSON logging (experiment metadata, config, etc) ---
    def log_json(self, data: Dict[str, Any]):
        """
        Append arbitrary dict to logs.json (for experiment metadata, config, etc).
        Args:
            data (dict): Dict to append
        """
        logs = self.read_json()
        logs.append(data)
        with open(self.json_path, "w") as f:
            f.write(json.dumps(logs))

    def read_json(self) -> List[Dict[str, Any]]:
        """
        Read all dicts from logs.json.
        Returns:
            list[dict]: All logged dicts
        """
        if not os.path.isfile(self.json_path):
            return []
        with open(self.json_path, "r") as f:
            return json.load(f)

    # --- Reading methods ---
    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all scalar metrics from scalars.csv.
        Returns:
            list[dict]: [{step, name, value}]
        """
        scalars = []
        with open(self.scalar_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Attempt float conversion for value
                try:
                    value = float(row["value"])
                except Exception:
                    value = row["value"]
                scalars.append({
                    "step": int(row["step"]),
                    "name": row["name"],
                    "value": value
                })
        return scalars

    def read_episode_returns(self) -> List[float]:
        """
        Read episode returns (total rewards) from episode_returns.csv.
        Returns:
            list[float]: Each episode's return
        """
        returns = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert to float robustly
                try:
                    returns.append(float(row["return"]))
                except Exception:
                    returns.append(row["return"])
        return returns

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read histogram metrics from histograms.csv.
        Returns:
            list[dict]: [{step, name, values: np.ndarray}]
        """
        histos = []
        with open(self.histogram_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values_str = row["values"]
                values_arr = np.array([float(x) for x in values_str.split(",") if x])
                histos.append({
                    "step": int(row["step"]),
                    "name": row["name"],
                    "values": values_arr
                })
        return histos

    def read_episode_averages(self) -> List[Dict[str, Any]]:
        """
        Read per-episode averages from episode_averages.csv.
        Returns:
            list[dict]: [{episode, name, average, count}]
        """
        avgs = []
        with open(self.episode_avg_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    avg = float(row["average"])
                except Exception:
                    avg = row["average"]
                avgs.append({
                    "episode": int(row["episode"]),
                    "name": row["name"],
                    "average": avg,
                    "count": int(row["count"])
                })
        return avgs

    def read_scalar_steps(self, name: str) -> List[tuple]:
        """
        Read all (step, value) pairs for a scalar metric by name, sorted by step.
        Args:
            name (str): Metric name to filter
        Returns:
            list[(step, value)]: Sorted by step
        """
        pairs = []
        with open(self.scalar_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == name:
                    try:
                        step = int(row["step"])
                        value = float(row["value"])
                    except Exception:
                        step = row["step"]
                        value = row["value"]
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
