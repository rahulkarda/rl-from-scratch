from replay import ReplayBuffer, Transition
import numpy as np

# Create a buffer and add transitions
buf = ReplayBuffer(capacity=5)

for i in range(6):
    state = np.array([i, i+1])
    action = i % 2
    reward = float(i)
    next_state = np.array([i+1, i+2])
    done = i % 3 == 0
    buf.push(Transition(state, action, reward, next_state, done))

# Sample a batch
batch = buf.sample(batch_size=3)
print("Sampled transitions:")
for t in batch:
    print(t)

# Save buffer
buf.save("buffer_serialized.pkl")

# Clear and reload buffer
buf.clear()
buf.load("buffer_serialized.pkl")
print("\nTransitions after reload:")
for t in buf.export_to_list():
    print(t)
