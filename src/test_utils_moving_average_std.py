from utils import moving_average, moving_std
import numpy as np

def test_moving_average():
    x = [1, 2, 3, 4, 5]
    ma = moving_average(x, window_size=3)
    expected = np.array([2., 3., 4.])
    assert np.allclose(ma, expected), f"moving_average failed: {ma} != {expected}"
    # Window larger than input: should return empty
    empty = moving_average([1, 2], window_size=3)
    assert empty.size == 0

def test_moving_std():
    x = [1, 2, 3, 4, 5]
    ms = moving_std(x, window_size=3)
    expected = np.array([np.std([1,2,3]), np.std([2,3,4]), np.std([3,4,5])])
    assert np.allclose(ms, expected), f"moving_std failed: {ms} != {expected}"
    # Window larger than input: should return empty
    empty = moving_std([1, 2], window_size=3)
    assert empty.size == 0

def run():
    test_moving_average()
    test_moving_std()
    print("moving_average and moving_std tests passed.")

if __name__ == "__main__":
    run()
