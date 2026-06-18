from replay import ReplayBuffer, Transition

# Tiny example: push, sample, recent sample, clear, save/load
if __name__ == "__main__":
    buf = ReplayBuffer(capacity=5)
    # Push five transitions
    for i in range(5):
        t = Transition(state=i, action=i % 2, reward=float(i), next_state=i + 1, done=(i == 4))
        buf.push(t)
    print("Buffer size:", len(buf))
    # Sample batch
    batch = buf.sample(batch_size=3)
    print("Sampled transitions:", batch)
    # Sample most recent
    recent = buf.sample_recent(batch_size=2)
    print("Most recent transitions:", recent)
    # Save buffer
    buf.save("example_buffer.pkl")
    buf.clear()
    print("Buffer cleared. Size:", len(buf))
    # Load buffer
    buf.load("example_buffer.pkl")
    print("Loaded buffer size:", len(buf))
    print("Loaded transitions:", list(buf.buffer))
