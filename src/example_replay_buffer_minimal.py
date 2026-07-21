from replay import ReplayBuffer, Transition
import numpy as np

# Create buffer with small capacity for demo
buf = ReplayBuffer(capacity=5)

# Push a few transitions
t1 = Transition(state=np.array([0.0]), action=0, reward=1.0, next_state=np.array([1.0]), done=False)
t2 = Transition(state=np.array([1.0]), action=1, reward=2.0, next_state=np.array([2.0]), done=False)
t3 = Transition(state=np.array([2.0]), action=0, reward=3.0, next_state=np.array([3.0]), done=True)
buf.push(t1)
buf.push(t2)
buf.push(t3)

# Sample a batch
batch = buf.sample(batch_size=2)
print("Sampled transitions:")
for t in batch:
    print(asdict(t))

# Clear buffer
buf.clear()
print("Buffer length after clear:", len(buf.buffer))
