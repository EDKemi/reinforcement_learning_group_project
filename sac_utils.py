import random
import numpy as np
import tensorflow as tf
import gymnasium as gym
import os
import logging


def get_logger(name: str = "sac", log_path: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Simple reusable logger.
    Always logs to console; optionally logs to a file if log_path is given.
    Includes file and line number for clarity.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
    )

    # Console handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler if requested
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1_000_000, dtype=np.float32):
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.o = np.zeros((size, obs_dim), dtype=dtype)
        self.a = np.zeros((size, act_dim), dtype=dtype)
        self.r = np.zeros((size,), dtype=dtype)
        self.o2 = np.zeros((size, obs_dim), dtype=dtype)
        self.d = np.zeros((size,), dtype=dtype)

    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            s=self.o[idx],
            a=self.a[idx],
            r=self.r[idx],
            s2=self.o2[idx],
            d=self.d[idx],
        )
        return batch


def soft_update_vars(target_vars, source_vars, tau=0.005):
    """Polyak averaging: target = (1 - tau) * target + tau * source.
    Works inside @tf.function."""
    for tv, sv in zip(target_vars, source_vars):
        tv.assign((1.0 - tau) * tv + tau * sv)


def set_seed(seed=0):
    random.seed(seed);
    np.random.seed(seed);
    tf.random.set_seed(seed)


def make_env(env_id="BipedalWalker-v3", seed=0, **kwargs):
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    return env


def to_tensor(batch, dtype=tf.float32):
    return {k: tf.convert_to_tensor(v, dtype=dtype) for k, v in batch.items()}

