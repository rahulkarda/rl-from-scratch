from replay import ReplayBuffer, Transition
import numpy as np
import os

def run_example():
    buf = ReplayBuffer(capacity=5)
    for i in range(5):
        t = Transition(state=np.array([i]), action=i % 2, reward=float(i), next_state=np.array([i+1]), done=(i==4))
        buf.push(t)
    save_path = "test_buffer.pkl"
    buf.save(save_path)
    print(f"Saved buffer with {len(buf)} transitions to {save_path}")
    # Load into new buffer
    buf2 = ReplayBuffer(capacity=5)
    buf2.load(save_path)
    print(f"Loaded buffer with {len(buf2)} transitions")
    for idx, t in enumerate(buf2.export_to_list()):
        print(f"Loaded {idx}: state={t.state}, action={t.action}, reward={t.reward}, done={t.done}")
    # Cleanup
    os.remove(save_path)

if __name__ == "__main__":
    run_example()
