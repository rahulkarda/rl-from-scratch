from replay import ReplayBuffer, Transition

def test_basic_buffer():
    buffer = ReplayBuffer(capacity=5)
    # Push 5 transitions
    for i in range(5):
        t = Transition(state=i, action=i%2, reward=float(i), next_state=i+1, done=i==4)
        buffer.push(t)
    assert len(buffer) == 5
    # Sampling batch of 3
    batch = buffer.sample(3)
    assert len(batch) == 3
    # Check types
    assert all(isinstance(t, Transition) for t in batch)
    # Push more, test capacity
    buffer.push(Transition(99, 1, 9.9, 100, False))
    assert len(buffer) == 5  # maxlen enforced
    # Sample all
    all_batch = buffer.sample(5)
    assert len(all_batch) == 5

def run():
    test_basic_buffer()
    print("ReplayBuffer basic test passed.")

if __name__ == "__main__":
    run()
