from logger import Logger
import os
import csv

def test_logger():
    log_dir = "test_logs"
    logger = Logger(log_dir=log_dir)
    logger.log_scalar("loss", 0.123, step=1)
    logger.log_episode_return(12.5, episode=1)
    logger.log_scalars({"epsilon": 0.9, "reward": 7.5}, step=2)
    # Check scalars.csv exists and has correct rows
    scalar_path = os.path.join(log_dir, "scalars.csv")
    with open(scalar_path, "r") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "name", "value"]
    assert ["1", "loss", "0.123"] in rows
    assert ["2", "epsilon", "0.9"] in rows
    assert ["2", "reward", "7.5"] in rows
    # Check episode_returns.csv
    returns_path = os.path.join(log_dir, "episode_returns.csv")
    with open(returns_path, "r") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "return"]
    assert ["1", "12.5"] in rows
    # Cleanup
    os.remove(scalar_path)
    os.remove(returns_path)
    os.rmdir(log_dir)

def run():
    test_logger()
    print("Logger basic test passed.")

if __name__ == "__main__":
    run()
