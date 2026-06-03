from utils import moving_average, running_average
import numpy as np

def test_running_average():
    vals = [1, 2, 3, 4]
    ra = running_average(vals)
    expected = np.array([1.0, 1.5, 2.0, 2.5])
    assert np.allclose(ra, expected)

def test_moving_average():
    vals = [1, 2, 3, 4, 5]
    ma = moving_average(vals, window_size=3)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(ma, expected)
    # window_size > len(vals)
    empty = moving_average(vals, window_size=10)
    assert empty.size == 0
    # window_size == 1 returns original
    ma1 = moving_average(vals, window_size=1)
    assert np.allclose(ma1, np.array(vals))

def run():
    test_running_average()
    test_moving_average()
    print("Utils basic test passed.")

if __name__ == "__main__":
    run()
