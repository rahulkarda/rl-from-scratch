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

Design notes:
- Logs are append-only; files are never overwritten, so all experiment data is preserved.
- CSV files are readable by pandas, Excel, or any spreadsheet tool.
- Scalar logging supports both single metrics and batches (via log_scalars).
- Histograms are stored as comma-separated values (convertible to np.ndarray).
- Episode averages accumulate statistics for each episode, useful for reward/loss curves.
- JSON logs are for metadata/config, not structured metrics; intended for experiment tracking.
- Reading methods return lists of dicts or arrays for easy plotting/analysis.
- Export methods copy CSV files for archiving or sharing results.
- Logger is thread-unsafe; intended for single-process training.
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
        self.episode_avg_path = os.path.join(log_dir, "episode_averages.csv")
        self.json_path = os.path.join(log_dir, "logs.json")
        # Initialize files if not exist
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
        if not os.path.exists(self.episode_avg_path):
            with open(self.episode_avg_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "name", "average", "count"])
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump([], f)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        with open(self.scalars_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, value])

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        with open(self.scalars_path, "a", newline='') as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                writer.writerow([step, k, v])

    def log_episode_return(self, value: float, episode: int) -> None:
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, value])

    def log_histogram(self, name: str, values: np.ndarray, step: int) -> None:
        values_str = ','.join(map(str, values.tolist()))
        with open(self.histograms_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, values_str])

    def log_episode_average(self, name: str, values: Any, episode: int) -> None:
        avg = float(np.mean(values)) if len(values) > 0 else 0.0
        count = len(values)
        with open(self.episode_avg_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, avg, count])

    def log_json(self, obj: Dict[str, Any]) -> None:
        # Append dict to logs.json
        logs = []
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
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
                out.append({"step": int(row["step"]), "name": row["name"], "value": float(row["value"])})
        return out

    def read_episode_returns(self) -> List[float]:
        out = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(float(row["return"]))
        return out

    def read_histograms(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.histograms_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = np.array(list(map(float, row["values"].split(','))))
                out.append({"step": int(row["step"]), "name": row["name"], "values": values})
        return out

    def read_episode_averages(self) -> List[Dict[str, Any]]:
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
        out = []
        with open(self.scalars_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == name:
                    out.append((int(row["step"]), float(row["value"])))
        out.sort()
        return out

    def read_json(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.json_path):
            return []
        with open(self.json_path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []

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

