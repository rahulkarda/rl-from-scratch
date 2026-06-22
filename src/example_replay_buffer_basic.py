from replay import ReplayBuffer, Transition
import numpy as np

def run_example():
    buf = ReplayBuffer(capacity=5)
    # Add 5 transitions
    for i in range(5):
        t = Transition(state=np.array([i]), action=i % 2, reward=float(i), next_state=np.array([i+1]), done=(i==4))
        buf.push(t)
    print(f"Buffer size: {len(buf)}")
    # Sample 3 random transitions
    batch = buf.sample(batch_size=3)
    for idx, t in enumerate(batch):
        print(f"Sample {idx}: state={t.state}, action={t.action}, reward={t.reward}, done={t.done}")

if __name__ == "__main__":
    run_example()
