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
      - CSV export: export_scalars_csv, export_episode_returns_csv  # NEW

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
      - export_scalars_csv(path): export scalars.csv to readable CSV (NEW)
      - export_episode_returns_csv(path): export episode_returns.csv to readable CSV (NEW)

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
        logger.export_scalars_csv("exported_scalars.csv")  # NEW
        logger.export_episode_returns_csv("exported_returns.csv")  # NEW
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

    def log_episode_return(self, value: float, episode: int):
        """
        Log episode return (total reward) for an episode.
        Args:
            value (float): Episode return
            episode (int): Episode index
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, self._format_value(value)])

    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """
        Log histogram (array metric) for a step. Values are stored as comma-separated string.
        Args:
            name (str): Metric name
            values (np.ndarray): Array values
            step (int): Step index
        """
        values_str = ','.join(map(str, values.tolist()))
        with open(self.histogram_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, values_str])

    def log_episode_average(self, name: str, values: List[float], episode: int):
        """
        Log per-episode average for a metric (e.g. reward, loss).
        Args:
            name (str): Metric name
            values (list[float]): Values to average
            episode (int): Episode index
        """
        avg = np.mean(values) if values else 0.0
        count = len(values)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, self._format_value(avg), count])

    # --- JSON logging ---
    def log_json(self, record: Dict[str, Any]):
        """
        Log arbitrary dict to logs.json (append as new entry).
        Args:
            record (dict): Experiment metadata, config, etc.
        """
        logs = self.read_json()
        logs.append(record)
        with open(self.json_path, "w") as f:
            json.dump(logs, f)

    def read_json(self) -> List[Dict[str, Any]]:
        """
        Read list of dicts from logs.json.
        Returns:
            list[dict]: All JSON log entries
        """
        with open(self.json_path, "r") as f:
            logs = json.load(f)
        return logs

    # --- Reading methods ---
    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all scalar metrics from scalars.csv as list of dicts.
        Returns:
            list[dict]: {step, name, value}
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
            list[float]: Episode returns
        """
        returns = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                returns.append(float(row["return"]))
        return returns

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read histograms from histograms.csv.
        Returns:
            list[dict]: {step, name, values}
        """
        out = []
        with open(self.histogram_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = np.array([float(x) for x in row["values"].split(",") if x.strip() != ""])
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
            list[dict]: {episode, name, average, count}
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
        Read all (step, value) pairs for a scalar metric.
        Args:
            name (str): Metric name
        Returns:
            list[tuple]: Sorted (step, value)
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

    # --- CSV export methods ---
    def export_scalars_csv(self, export_path: str) -> None:
        """
        Export scalars.csv to a new CSV file (all scalar metrics).
        Args:
            export_path (str): Path to export CSV.
        """
        with open(self.scalar_path, "r") as src:
            with open(export_path, "w", newline='') as dst:
                for line in src:
                    dst.write(line)

    def export_episode_returns_csv(self, export_path: str) -> None:
        """
        Export episode_returns.csv to a new CSV file (all episode returns).
        Args:
            export_path (str): Path to export CSV.
        """
        with open(self.returns_path, "r") as src:
            with open(export_path, "w", newline='') as dst:
                for line in src:
                    dst.write(line)
