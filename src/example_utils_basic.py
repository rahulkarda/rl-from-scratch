import numpy as np
import torch
import random
from utils import set_seed, seed_everything, running_average, moving_average, moving_std, soft_update, flatten_dict

def example_seeding():
    # Set seed for reproducibility
    set_seed(42)
    # Check reproducibility
    a = np.random.randn(3)
    b = torch.rand(3)
    c = random.randint(0, 100)
    set_seed(42)
    a2 = np.random.randn(3)
    b2 = torch.rand(3)
    c2 = random.randint(0, 100)
    assert np.allclose(a, a2), "NumPy seed failed"
    assert torch.allclose(b, b2), "Torch seed failed"
    assert c == c2, "Random seed failed"
    print("Seeding example passed.")

def example_averaging():
    values = [1, 2, 3, 4, 5]
    run_avg = running_average(values)
    mov_avg = moving_average(values, window_size=3)
    mov_std = moving_std(values, window_size=3)
    print("Running average:", run_avg)
    print("Moving average (window=3):", mov_avg)
    print("Moving std (window=3):", mov_std)

def example_flatten_dict():
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_dict(nested)
    print("Flattened dict:", flat)

def main():
    example_seeding()
    example_averaging()
    example_flatten_dict()

if __name__ == "__main__":
    main()
