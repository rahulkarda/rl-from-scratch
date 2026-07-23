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
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.scalars_path = os.path.join(log_dir, "scalars.csv")
        self.returns_path = os.path.join(log_dir, "episode_returns.csv")
        self.histograms_path = os.path.join(log_dir, "histograms.csv")
        self.episode_averages_path = os.path.join(log_dir, "episode_averages.csv")
        self.json_path = os.path.join(log_dir, "logs.json")
        # Ensure headers exist
        self._init_files()

    def _init_files(self):
        if not os.path.exists(self.scalars_path):
            with open(self.scalars_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "name", "value"])
        if not os.path.exists(self.returns_path):
            with open(self.returns_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])
        if not os.path.exists(self.histograms_path):
            with open(self.histograms_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "name", "values"])
        if not os.path.exists(self.episode_averages_path):
            with open(self.episode_averages_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "name", "average", "count"])
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump([], f)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        with open(self.scalars_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, value])

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        with open(self.scalars_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, value])

    def log_episode_return(self, episode_return: float, episode: int) -> None:
        """
        Log episode return (total reward for an episode) to episode_returns.csv.
        Args:
            episode_return (float): Total reward for episode.
            episode (int): Episode index.
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            # Always write return as float for consistency
            writer.writerow([episode, f"{float(episode_return):.6f}"])

    def log_histogram(self, name: str, values: np.ndarray, step: int) -> None:
        with open(self.histograms_path, "a", newline='') as f:
            writer = csv.writer(f)
            # Convert values to comma-separated string
            values_str = ','.join(str(v) for v in np.asarray(values).flatten())
            writer.writerow([step, name, values_str])

    def log_episode_average(self, name: str, values: List[float], episode: int) -> None:
        avg = float(np.mean(values)) if values else 0.0
        count = len(values)
        with open(self.episode_averages_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, avg, count])

    def log_json(self, obj: Dict[str, Any]) -> None:
        # Append JSON dict to logs.json
        try:
            with open(self.json_path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        logs.append(obj)
        with open(self.json_path, "w") as f:
            json.dump(logs, f, indent=2)

    def read_scalars(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.scalars_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({
                    "step": int(row["step"]),
                    "name": row["name"],
                    "value": float(row["value"])
                })
        return out

    def read_episode_returns(self) -> List[float]:
        out = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Always parse return as float
                out.append(float(row["return"]))
        return out

    def read_histograms(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.histograms_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = [float(v) for v in row["values"].split(',') if v.strip() != '']
                out.append({
                    "step": int(row["step"]),
                    "name": row["name"],
                    "values": np.array(values)
                })
        return out

    def read_episode_averages(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.episode_averages_path, "r") as f:
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
        Return sorted list of (step, value) for a given scalar metric.
        """
        steps = []
        with open(self.scalars_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == name:
                    steps.append((int(row["step"]), float(row["value"])))
        steps.sort()
        return steps

    def read_json(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        return logs

    def export_scalars_csv(self, export_path: str) -> None:
        """
        Export scalars.csv to a new CSV file (all scalar metrics).
        Args:
            export_path (str): Path to export CSV.
        """
        with open(self.scalars_path, "r") as src:
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

