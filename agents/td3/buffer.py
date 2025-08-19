# Replay buffer, keeps the most recent transitions and drops the oldest when full
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    # Stores transitions as (state, action, reward, next_state, done)

    def __init__(self, capacity):
        # So we dont have to manually pop old entries
        self.data = deque(maxlen=int(capacity))

    def add(self, s, a, r, ns, done):
        # Cast to consistent dtypes - rewards/done as floats, others as float32 arrays
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = float(r)

        ns = np.asarray(ns, dtype=np.float32)
        done = float(done)
        self.data.append((s, a, r, ns, done))

    def __len__(self):
        # Allows len(buffer) to return how many transitions we have
        return len(self.data)

    def can_sample(self, batch_size):
        # So callers dont sample before we have enough data
        return len(self.data) >= batch_size

    def sample(self, batch_size):
        # Uniform random sample (without replacement) for a training batch
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, d = zip(*batch)

        # Stack into batch-first arrays - keep rewards/dones as column vectors
        s = np.stack(s, axis=0)
        a = np.stack(a, axis=0)
        r = np.asarray(r, dtype=np.float32).reshape(-1, 1)
        ns = np.stack(ns, axis=0)
        d = np.asarray(d, dtype=np.float32).reshape(-1, 1)
        
        return s, a, r, ns, d
