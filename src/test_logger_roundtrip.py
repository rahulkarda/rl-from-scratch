from logger import Logger
import os
import csv

def test_logger_roundtrip():
    log_dir = "test_logs_roundtrip"
    logger = Logger(log_dir=log_dir)
    # Log some scalars and returns
    logger.log_scalar("loss", 0.321, step=10)
    logger.log_scalars({"epsilon": 0.12, "reward": 7.8}, step=11)
    logger.log_episode_return(42.0, episode=3)
    logger.log_episode_return(99.9, episode=4)
    # Read back scalars
    scalars = logger.read_scalars()
    names = set([s["name"] for s in scalars])
    values = {s["name"]: s["value"] for s in scalars}
    steps = [s["step"] for s in scalars]
    assert "loss" in names
    assert "epsilon" in names
    assert "reward" in names
    assert values["loss"] == 0.321
    assert steps.count(10) == 1
    assert steps.count(11) == 2
    # Read back returns
    returns = logger.read_episode_returns()
    assert len(returns) == 2
    assert returns[0]["episode"] == 3
    assert returns[0]["return"] == 42.0
    assert returns[1]["episode"] == 4
    assert returns[1]["return"] == 99.9
    # Cleanup
    scalar_path = os.path.join(log_dir, "scalars.csv")
    returns_path = os.path.join(log_dir, "episode_returns.csv")
    os.remove(scalar_path)
    os.remove(returns_path)
    os.rmdir(log_dir)

def run():
    test_logger_roundtrip()
    print("Logger CSV roundtrip test passed.")

if __name__ == "__main__":
    run()
