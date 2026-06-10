import csv
import os

# Minimal example: read logger CSV files for analysis/plotting

def read_scalars(log_dir):
    scalar_path = os.path.join(log_dir, "scalars.csv")
    scalars = []
    with open(scalar_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse step as int, value as float
            scalars.append({
                "step": int(row["step"]),
                "name": row["name"],
                "value": float(row["value"])
            })
    return scalars

def read_episode_returns(log_dir):
    returns_path = os.path.join(log_dir, "episode_returns.csv")
    returns = []
    with open(returns_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            returns.append({
                "episode": int(row["episode"]),
                "return": float(row["return"])
            })
    return returns

if __name__ == "__main__":
    # Example usage: assumes logs/dqn_cartpole exists
    log_dir = "logs/dqn_cartpole"
    if os.path.isdir(log_dir):
        scalars = read_scalars(log_dir)
        returns = read_episode_returns(log_dir)
        print("First 2 scalars:", scalars[:2])
        print("First 2 episode returns:", returns[:2])
    else:
        print(f"Log directory '{log_dir}' does not exist. Run dqn_cartpole.py first.")
