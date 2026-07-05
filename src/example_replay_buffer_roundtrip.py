from replay import ReplayBuffer, Transition
import numpy as np
import os

# Minimal example: push transitions, sample, save/load roundtrip

def main():
    buf = ReplayBuffer(capacity=10)
    # Push some transitions
    for i in range(5):
        t = Transition(state=np.array([i]), action=i % 2, reward=1.0 * i, next_state=np.array([i+1]), done=(i == 4))
        buf.push(t)
    print(f"Buffer size after push: {len(buf)}")
    batch = buf.sample(batch_size=3)
    print("Sampled transitions:")
    for t in batch:
        print(vars(t))

    # Save buffer
    path = "example_buffer.pkl"
    buf.save(path)

    # Load into new buffer
    buf2 = ReplayBuffer(capacity=10)
    buf2.load(path)
    print(f"Loaded buffer size: {len(buf2)}")
    # Sample again
    batch2 = buf2.sample(batch_size=2)
    print("Sampled transitions after load:")
    for t in batch2:
        print(vars(t))

    # Clean up
    os.remove(path)

if __name__ == "__main__":
    main()
