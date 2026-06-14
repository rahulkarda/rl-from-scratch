from replay import ReplayBuffer, Transition
import os

def test_save_load():
    buffer = ReplayBuffer(capacity=3)
    # Fill buffer
    for i in range(3):
        buffer.push(Transition(state=i, action=i, reward=float(i), next_state=i+1, done=False))
    path = "test_replay_save.pkl"
    buffer.save(path)
    # Create new buffer, load
    new_buffer = ReplayBuffer(capacity=3)
    new_buffer.load(path)
    assert len(new_buffer) == 3
    # Check transitions match
    for orig, loaded in zip(buffer.buffer, new_buffer.buffer):
        assert orig.state == loaded.state
        assert orig.action == loaded.action
        assert orig.reward == loaded.reward
        assert orig.next_state == loaded.next_state
        assert orig.done == loaded.done
    os.remove(path)

def run():
    test_save_load()
    print("ReplayBuffer save/load test passed.")

if __name__ == "__main__":
    run()
