from utils import elementwise_min_max
import numpy as np

if __name__ == "__main__":
    a = [1, 5, 3, 7]
    b = [2, 4, 6, 1]
    min_result = elementwise_min_max(a, b, mode="min")
    max_result = elementwise_min_max(a, b, mode="max")
    print("Elementwise min:", min_result)
    print("Elementwise max:", max_result)
    # Check correctness
    assert np.allclose(min_result, np.array([1, 4, 3, 1]))
    assert np.allclose(max_result, np.array([2, 5, 6, 7]))
    print("elementwise_min_max example passed.")
