from logger import Logger
import os
import tempfile

def test_logger_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir)
        # Log some scalars and returns
        logger.log_scalar("loss", 0.123456789, step=5)
        logger.log_scalars({"epsilon": 0.987654321, "score": 42}, step=6)
        logger.log_episode_return(99.99, episode=1)
        logger.log_episode_return(101.01, episode=2)
        # Read back
        scalars = logger.read_scalars()
        returns = logger.read_episode_returns()
        # Check shape and rounded float values
        assert len(scalars) == 3
        assert scalars[0]["name"] == "loss"
        assert abs(scalars[0]["value"] - 0.123457) < 1e-6
        assert scalars[1]["name"] == "epsilon"
        assert abs(scalars[1]["value"] - 0.987654) < 1e-6
        assert scalars[2]["name"] == "score"
        assert scalars[2]["value"] == 42.0
        assert len(returns) == 2
        assert returns[0]["episode"] == 1
        assert abs(returns[0]["return"] - 99.99) < 1e-2
        assert returns[1]["episode"] == 2
        assert abs(returns[1]["return"] - 101.01) < 1e-2
    print("Logger roundtrip test passed.")

if __name__ == "__main__":
    test_logger_roundtrip()
