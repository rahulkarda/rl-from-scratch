from logger import Logger
import os

# Minimal example: log scalars and episode returns, then read back

def main():
    log_dir = "example_logs_roundtrip"
    logger = Logger(log_dir=log_dir)

    # Log scalars for steps 1-3
    logger.log_scalar("loss", 0.123, step=1)
    logger.log_scalar("epsilon", 0.95, step=1)
    logger.log_scalars({"loss": 0.115, "epsilon": 0.93}, step=2)
    logger.log_scalar("reward", 12.7, step=3)

    # Log episode returns
    logger.log_episode_return(15.0, episode=1)
    logger.log_episode_return(17.2, episode=2)

    # Read back scalars
    scalars = logger.read_scalars()
    print("Scalars read:")
    for s in scalars:
        print(s)

    # Read back episode returns
    returns = logger.read_episode_returns()
    print("Episode returns read:", returns)

    # Cleanup log files
    scalar_path = os.path.join(log_dir, "scalars.csv")
    returns_path = os.path.join(log_dir, "episode_returns.csv")
    os.remove(scalar_path)
    os.remove(returns_path)
    try:
        os.rmdir(log_dir)
    except OSError:
        pass

if __name__ == "__main__":
    main()
