"""
Logger for writing scalar metrics, episode returns, histograms, and now plain text logs to disk (CSV and TXT).

Overview:
- Log scalar metrics (loss, epsilon, etc) per step to scalars.csv
- Log episode returns (total reward per episode) to episode_returns.csv
- Log histograms (arrays, e.g. weights, Q-values) to histograms.csv
- Log per-episode averages (e.g. reward, loss) to episode_averages.csv
- Log arbitrary experiment metadata/config as JSON to logs.json
- Log plain text messages to text.log (NEW)
- Read back all logged data via read_* methods for analysis, plotting, or debugging
- Export CSV files for further inspection or sharing

File formats:
- scalars.csv: step, name, value         # Each scalar metric per step (loss, epsilon, etc)
- episode_returns.csv: episode, return   # Total reward per episode
- histograms.csv: step, name, values     # Array metrics as comma-separated string
- episode_averages.csv: episode, name, average, count  # Per-episode averages (for stats)
- logs.json: list of dicts (experiment metadata, config, hyperparameters, etc)
- text.log: plain text log lines (NEW)

Usage Example:
    logger = Logger(log_dir="logs/test_run")
    logger.log_scalar("loss", 0.32, step=10)
    logger.log_episode_return(42.0, episode=3)
    logger.log_scalars({"loss": 0.32, "epsilon": 0.13}, step=10)
    logger.log_histogram("weights", np.array([1,2,3]), step=10)
    logger.log_episode_average("reward", [1.0, 2.0, 3.0], episode=3)
    logger.log_json({"config": {"lr": 1e-3, "batch_size": 32}})
    logger.log_text("Training started.")  # NEW
    # Reading logged data:
    scalars = logger.read_scalars()
    returns = logger.read_episode_returns()
    histos = logger.read_histograms()
    avgs = logger.read_episode_averages()
    loss_steps = logger.read_scalar_steps("loss")
    json_logs = logger.read_json()
    text_lines = logger.read_text()  # NEW
    logger.export_scalars_csv("exported_scalars.csv")  # Export readable CSV
    logger.export_episode_returns_csv("exported_returns.csv")  # Export readable CSV

Design notes:
- Logs are append-only; files are never overwritten, so all experiment data is preserved.
- CSV files are readable by pandas, Excel, or any spreadsheet tool.
- Scalar logging supports both single metrics and batches (via log_scalars).
- Histograms are stored as comma-separated values (convertible to np.ndarray).
- Episode averages accumulate statistics for each episode, useful for reward/loss curves.
- JSON logs are for metadata/config, not structured metrics; intended for experiment tracking.
- Plain text logs (text.log) allow arbitrary info/debug messages alongside structured metrics (NEW).
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
    Logger for writing scalar metrics, episode returns, histograms, and plain text logs to disk.

    Provides:
      - Scalar metric logging: log_scalar, log_scalars
      - Episode return logging: log_episode_return
      - Histogram logging: log_histogram
      - Reading logged data: read_scalars, read_episode_returns, read_histograms
      - Episodic average logging: log_episode_average
      - Scalar step lookup: read_scalar_steps
      - Episode averages reading: read_episode_averages

      - JSON logging: log_json, read_json
      - CSV export: export_scalars_csv, export_episode_returns_csv
      - Plain text logging: log_text, read_text (NEW)

    CSV file layout:
      - scalars.csv: step, name, value         # Each scalar metric (loss, epsilon, etc) per step
      - episode_returns.csv: episode, return   # Total reward per episode
      - histograms.csv: step, name, values     # Array metrics (weights, Q-values) as comma-separated strings
      - episode_averages.csv: episode, name, average, count  # Per-episode averages (e.g. reward, loss)
    JSON file layout:
      - logs.json: list of dicts (arbitrary experiment metadata, config, etc)
    Plain text layout:
      - text.log: raw text lines (append-only, NEW)

    Reading methods:
      - read_scalars(): list[dict] {step, name, value} for all scalar metrics
      - read_episode_returns(): list[float] (episode returns)
      - read_histograms(): list[dict] {step, name, values (np.ndarray)}
      - read_episode_averages(): list[dict] {episode, name, average, count}
      - read_scalar_steps(name): sorted list[(step, value)] for a given scalar
      - read_json(): list[dict] for all JSON logs
      - read_text(): list[str] for all text log lines (NEW)
      - export_scalars_csv(path): export scalars.csv to readable CSV
      - export_episode_returns_csv(path): export episode_returns.csv to readable CSV

    Example usage:
        logger = Logger(log_dir="logs/test_run")
        logger.log_scalar("loss", 0.32, step=10)
        logger.log_episode_return(42.0, episode=3)
        logger.log_scalars({"loss": 0.32, "epsilon": 0.13}, step=10)
        logger.log_histogram("weights", np.array([1,2,3]), step=10)
        logger.log_episode_average("reward", [1.0, 2.0, 3.0], episode=3)
        logger.log_json({"config": {"lr": 1e-3, "batch_size": 32}})
        logger.log_text("Training started.")  # NEW
        # Reading logged data:
        scalars = logger.read_scalars()
        returns = logger.read_episode_returns()
        histos = logger.read_histograms()
        avgs = logger.read_episode_averages()
        loss_steps = logger.read_scalar_steps("loss")
        json_logs = logger.read_json()
        text_lines = logger.read_text()  # NEW
        logger.export_scalars_csv("exported_scalars.csv")
        logger.export_episode_returns_csv("exported_returns.csv")
    """
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.scalars_path = os.path.join(log_dir, "scalars.csv")
        self.returns_path = os.path.join(log_dir, "episode_returns.csv")
        self.histograms_path = os.path.join(log_dir, "histograms.csv")
        self.episode_averages_path = os.path.join(log_dir, "episode_averages.csv")
        self.json_path = os.path.join(log_dir, "logs.json")
        self.text_path = os.path.join(log_dir, "text.log")  # NEW
        # Write headers if files don't exist
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
        if not os.path.exists(self.text_path):  # NEW
            with open(self.text_path, "w") as f:
                pass

    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a single scalar metric at a given step.
        Args:
            name (str): Metric name (e.g. "loss", "epsilon")
            value (float): Metric value
            step (int): Step number
        """
        with open(self.scalars_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, float(value)])

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """
        Log multiple scalar metrics at a given step.
        Args:
            scalars (dict): {name: value}
            step (int): Step number
        """
        with open(self.scalars_path, "a", newline='') as f:
            writer = csv.writer(f)
            for name, value in scalars.items():
                writer.writerow([step, name, float(value)])

    def log_episode_return(self, ret: float, episode: int):
        """
        Log total episode return (reward sum).
        Args:
            ret (float): Episode return
            episode (int): Episode number
        """
        with open(self.returns_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, float(ret)])

    def log_histogram(self, name: str, values, step: int):
        """
        Log an array metric (histogram) at a given step.
        Args:
            name (str): Metric name (e.g. "weights")
            values: Array-like (np.ndarray, list, etc)
            step (int): Step number
        """
        arr = np.array(values)
        value_str = ",".join([str(float(v)) for v in arr.flatten()])
        with open(self.histograms_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, value_str])

    def log_episode_average(self, name: str, values, episode: int):
        """
        Log per-episode average metric (e.g. reward, loss).
        Args:
            name (str): Metric name
            values: list of values for the episode
            episode (int): Episode number
        """
        if not values:
            avg = 0.0
            cnt = 0
        else:
            avg = float(np.mean(values))
            cnt = len(values)
        with open(self.episode_averages_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, name, avg, cnt])

    def log_json(self, obj: Dict[str, Any]):
        """
        Log arbitrary experiment metadata/config as JSON.
        Args:
            obj (dict): Arbitrary dict
        """
        # Append to logs.json (list of dicts)
        try:
            with open(self.json_path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        logs.append(obj)
        with open(self.json_path, "w") as f:
            json.dump(logs, f)

    def log_text(self, text: str):
        """
        Log a plain text message to text.log.
        Args:
            text (str): Text message
        """
        with open(self.text_path, "a") as f:
            f.write(text.rstrip("\n") + "\n")

    def read_scalars(self) -> List[Dict[str, Any]]:
        """
        Read all scalar metrics.
        Returns:
            list of dicts: {step, name, value}
        """
        out = []
        with open(self.scalars_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({"step": int(row["step"]), "name": row["name"], "value": float(row["value"])} )
        return out

    def read_episode_returns(self) -> List[float]:
        """
        Read all episode returns.
        Returns:
            list of floats (returns)
        """
        out = []
        with open(self.returns_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(float(row["return"]))
        return out

    def read_histograms(self) -> List[Dict[str, Any]]:
        """
        Read all histogram metrics.
        Returns:
            list of dicts: {step, name, values (np.ndarray)}
        """
        out = []
        with open(self.histograms_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals = row["values"].split(",") if row["values"] else []
                arr = np.array([float(v) for v in vals])
                out.append({"step": int(row["step"]), "name": row["name"], "values": arr})
        return out

    def read_episode_averages(self) -> List[Dict[str, Any]]:
        """
        Read all per-episode averages.
        Returns:
            list of dicts: {episode, name, average, count}
        """
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
        Read all steps and values for a given scalar metric.
        Args:
            name (str): Metric name
        Returns:
            sorted list of (step, value)
        """
        steps = []
        with open(self.scalars_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == name:
                    steps.append( (int(row["step"]), float(row["value"])) )
        steps.sort()
        return steps

    def read_json(self) -> List[Dict[str, Any]]:
        """
        Read all JSON logs (experiment metadata/config).
        Returns:
            list of dicts
        """
        try:
            with open(self.json_path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        return logs

    def read_text(self) -> List[str]:
        """
        Read all plain text log lines from text.log.
        Returns:
            list of str
        """
        lines = []
        with open(self.text_path, "r") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
        return lines

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

