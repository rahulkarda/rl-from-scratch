from logger import Logger
import os

# Minimal example: log scalars and episode returns
log_dir = "logs/example_logger_basic"
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_dir=log_dir)

# Log a single scalar metric (loss)
logger.log_scalar("loss", 0.123, step=1)
logger.log_scalar("epsilon", 0.9, step=1)

# Log a batch of scalars
logger.log_scalars({"loss": 0.101, "epsilon": 0.85}, step=2)

# Log episode returns
logger.log_episode_return(21.0, episode=1)
logger.log_episode_return(42.0, episode=2)

# Read back scalars and returns
scalars = logger.read_scalars()   # List of dicts: {step, name, value}
returns = logger.read_episode_returns()  # List of floats

print("Scalars:", scalars)
print("Episode returns:", returns)
