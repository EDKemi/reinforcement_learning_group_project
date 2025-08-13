import random
from collections import deque
import numpy as np

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    @property
    def size(self):
        return len(self.buffer)

    def add_experience(self, state, action, reward, state_prime):
        experience = (state, action, reward, state_prime)

        if self.size == self.capacity:
            self.buffer.popleft()
        self.buffer.append(experience)

    def sample(self, batch_size):
        if self.size < batch_size:
            return random.sample(self.buffer, self.size)

        return random.sample(self.buffer, batch_size)

def update_target_network(current_net, target_net):
    beta = 0.05

    current_weights = current_net.get_weights()
    target_weights = target_net.get_weights()

    for i in range(len(current_weights)):
        target_weights[i] = beta * current_weights[i] + (1 - beta) * target_weights[i]
    target_net.set_weights(current_weights)

    return target_net


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, size, mu=0.0, sigma=0.2, theta=0.15, dt=0.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.state = x + dx
        return self.state


def ddpg_add_exploration_noise(exploration_noise, action, noise_scale):
    noise = noise_scale * exploration_noise()

    action = np.clip(action + noise, -1.0, 1.0)

    return action