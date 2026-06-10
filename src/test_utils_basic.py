from utils import moving_average, running_average, soft_update, flatten_dict
import numpy as np
import torch


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


def test_soft_update():
    # 2-layer linear nets
    src = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2)
    )
    tgt = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2)
    )
    # Set src params to ones, tgt params to zeros
    for p in src.parameters():
        p.data.fill_(1.0)
    for p in tgt.parameters():
        p.data.fill_(0.0)
    # tau=0.5: tgt = 0.5*src + 0.5*tgt = 0.5*1 + 0.5*0 = 0.5
    soft_update(tgt, src, tau=0.5)
    for p in tgt.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 0.5))
    # tau=1.0: tgt = src
    soft_update(tgt, src, tau=1.0)
    for p in tgt.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 1.0))


def test_flatten_dict():
    nested = {
        'loss': 0.1,
        'stats': {
            'mean': 1,
            'std': 2,
            'hist': {
                'min': 0,
                'max': 5
            }
        },
        'lr': 0.001
    }
    flat = flatten_dict(nested)
    expected = {
        'loss': 0.1,
        'stats.mean': 1,
        'stats.std': 2,
        'stats.hist.min': 0,
        'stats.hist.max': 5,
        'lr': 0.001
    }
    assert flat == expected
    # Test empty dict
    assert flatten_dict({}) == {}
    # Test single-layer dict
    assert flatten_dict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}


def run():
    test_running_average()
    test_moving_average()
    test_soft_update()
    test_flatten_dict()
    print("Utils basic test passed.")

if __name__ == "__main__":
    run()
