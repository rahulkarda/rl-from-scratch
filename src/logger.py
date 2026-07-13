"""
Logger for writing scalar metrics, episode returns, and histograms to disk (CSV).

Overview:
- Log scalar metrics (loss, epsilon, etc) per step to scalars.csv
- Log episode returns (total reward per episode) to episode_returns.csv
- Log histograms (arrays, e.g. weights, Q-values) to histograms.csv
- Log per-episode averages (e.g. reward, loss) to episode_averages.csv
- Log arbitrary experiment metadata/config as JSON to logs.json
- Read back all logged data via read_* methods for analysis, plotting, or debugging
- Export CSV files for further inspection or sharing

File formats:
- scalars.csv: step, name, value         # Each scalar metric per step (loss, epsilon, etc)
- episode_returns.csv: episode, return   # Total reward per episode
- histograms.csv: step, name, values     # Array metrics as comma-separated string
- episode_averages.csv: episode, name, average, count  # Per-episode averages (for stats)
- logs.json: list of dicts (experiment metadata, config, hyperparameters, etc)

Usage Example:
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
    logger.export_scalars_csv("exported_scalars.csv")  # Export readable CSV
    logger.export_episode_returns_csv("exported_returns.csv")  # Export readable CSV

Notes:
- All logs are appended, never overwritten. CSV files are readable by pandas, Excel, etc.
- Scalar logging supports single and batch metrics; episode returns are always float.
- JSON logging is for arbitrary experiment metadata/config (not structured metrics).
- Reading methods return lists of dicts or arrays for easy plotting/analysis.
- Export methods copy CSV files for sharing or archiving results.
"""
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
            writer.writerow([step, name, value])

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """
        Log multiple scalar metrics for a step.
        Args:
            scalars (dict): {name: value}
            step (int): Step index
        """
        with open(self.scalar_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, value])

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
            writer.writerow([episode, episode_return])

    # --- Histogram logging ---
    def log_histogram(self, name: str, values: np.ndarray | list, step: int):
        """
        Log a histogram (array of values) for a given step and metric name.
        Args:
            name (str): Metric name
            values (array-like): Values to log
            step (int): Step index
        """
        values_arr = np.array(values)
        values_str = ','.join(map(str, values_arr.flatten()))
        with open(self.histogram_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, values_str])

    # --- Episode averages ---
    def log_episode_average(self, name: str, values: List[float], episode: int):
        """
        Log per-episode average metric (e.g., reward, loss).
        Args:
            name (str): Metric name
            values (list[float]): Values to average
            episode (int): Episode index
        """
        avg = float(np.mean(values)) if values else 0.0
        count = len(values)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, avg, count])

    # --- JSON logging ---
    def log_json(self, obj: Dict[str, Any]):
        """
        Log arbitrary experiment metadata/config as JSON.
        Args:
            obj (dict): Any dict to log
        """
        logs = self.read_json()
        logs.append(obj)
        with open(self.json_path, "w") as f:
            json.dump(logs, f)

    # --- Scalar reading ---
    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all scalar metrics from scalars.csv as list of dicts.
        Returns:
            list[dict]: [{step, name, value}]
        """
        out = []
        with open(self.scalar_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({"step": int(row["step"]), "name": row["name"], "value": float(row["value"])})
        return out

    def read_scalar_steps(self, name: str) -> List[tuple]:
        """
        Read (step, value) pairs for a given scalar metric (sorted by step).
        Args:
            name (str): Metric name
        Returns:
            list[(step, value)]
        """
        steps = []
        with open(self.scalar_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == name:
                    steps.append((int(row["step"]), float(row["value"])))
        steps.sort()
        return steps

    # --- Episode returns reading ---
    def read_episode_returns(self) -> List[float]:
        """
        Read episode returns (total reward) from episode_returns.csv.
        Returns:
            list[float]: episode returns
        """
        returns = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                returns.append(float(row["return"]))
        return returns

    # --- Histogram reading ---
    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read all histograms from histograms.csv as list of dicts.
        Returns:
            list[dict]: [{step, name, values (np.ndarray)}]
        """
        out = []
        with open(self.histogram_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = [float(v) for v in row["values"].split(',') if v]
                out.append({"step": int(row["step"]), "name": row["name"], "values": np.array(values)})
        return out

    # --- Episode averages reading ---
    def read_episode_averages(self) -> List[Dict[str, Any]]:
        """
        Read all per-episode averages from episode_averages.csv as list of dicts.
        Returns:
            list[dict]: [{episode, name, average, count}]
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

    # --- JSON reading ---
    def read_json(self) -> List[Dict[str, Any]]:
        """
        Read all experiment metadata/config logs from logs.json.
        Returns:
            list[dict]: All logged dicts
        """
        if not os.path.isfile(self.json_path):
            return []
        with open(self.json_path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []

    # --- CSV export ---
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

