from replay import ReplayBuffer, Transition
import os
import tempfile

def test_replay_save_load_roundtrip():
    buf = ReplayBuffer(capacity=10)
    # Push a few transitions
    for i in range(3):
        t = Transition(state=i, action=i%2, reward=float(i), next_state=i+1, done=(i==2))
        buf.push(t)
    assert len(buf) == 3
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    path = tmpfile.name
    tmpfile.close()
    buf.save(path)
    # Load into new buffer
    buf2 = ReplayBuffer(capacity=10)
    buf2.load(path)
    assert len(buf2) == 3
    # Compare contents
    orig = buf.export_to_list()
    loaded = buf2.export_to_list()
    for t1, t2 in zip(orig, loaded):
        assert t1 == t2
    os.remove(path)

def run():
    test_replay_save_load_roundtrip()
    print("ReplayBuffer save/load roundtrip test passed.")

if __name__ == "__main__":
    run()
