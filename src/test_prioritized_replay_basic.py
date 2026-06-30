from prioritized_replay import PrioritizedReplayBuffer, Transition


def test_prioritized_push_and_sample():
    buf = PrioritizedReplayBuffer(capacity=5, alpha=0.6)
    # Push transitions with increasing priority
    for i in range(5):
        t = Transition(state=i, action=i % 2, reward=float(i), next_state=i + 1, done=i == 4)
        buf.push(t, priority=1.0 + i)
    assert len(buf) == 5
    # Sample a batch
    batch, indices, weights = buf.sample(batch_size=3, beta=0.4)
    assert len(batch) == 3
    assert len(indices) == 3
    assert len(weights) == 3
    # Check types
    assert all(isinstance(t, Transition) for t in batch)
    assert all(isinstance(idx, int) for idx in indices)
    assert all(isinstance(w, float) for w in weights)
    # Update priorities
    new_priorities = [2.0, 3.0, 4.0]
    buf.update_priorities(indices, new_priorities)
    # Save and load
    buf.save('test_prio_buf.pkl')
    buf.load('test_prio_buf.pkl')
    assert len(buf) == 5


def run():
    test_prioritized_push_and_sample()
    print("PrioritizedReplayBuffer basic test passed.")


if __name__ == "__main__":
    run()
