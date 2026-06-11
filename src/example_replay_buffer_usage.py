from replay import ReplayBuffer, Transition

# Minimal example: store and sample transitions

def main():
    buf = ReplayBuffer(capacity=5)
    # Add several transitions
    for i in range(7):
        t = Transition(
            state=f"s{i}",
            action=i % 2,
            reward=float(i),
            next_state=f"s{i+1}",
            done=(i % 3 == 0)
        )
        buf.push(t)
        print(f"Pushed: {t}")
    print(f"Buffer size after pushes: {len(buf)} (should be 5)")
    # Sample a batch
    batch = buf.sample(batch_size=3)
    print("Sampled batch:")
    for t in batch:
        print(t)
    # Save and reload to test serialization
    buf.save("replay_test.pkl")
    buf2 = ReplayBuffer(capacity=5)
    buf2.load("replay_test.pkl")
    print(f"Reloaded buffer size: {len(buf2)}")
    for t in buf2.sample(batch_size=2):
        print("Reloaded sample:", t)

if __name__ == "__main__":
    main()
