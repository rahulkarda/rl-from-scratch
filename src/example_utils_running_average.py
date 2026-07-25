from utils import running_average
import numpy as np

if __name__ == "__main__":
    values = [1, 2, 3, 4, 5]
    avg = running_average(values)
    print("Input:", values)
    print("Running average:", avg)
    # Should print: [1., 1.5, 2., 2.5, 3.]
    assert np.allclose(avg, [1., 1.5, 2., 2.5, 3.])
    print("running_average utility works.")
