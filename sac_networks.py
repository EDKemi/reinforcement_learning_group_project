import math
import tensorflow as tf
from tensorflow.keras import layers, Model

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(sizes, activation="relu", final_activation=None):
    seq = tf.keras.Sequential()
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else final_activation
        seq.add(layers.Dense(sizes[i+1], activation=act))
    return seq


class GaussianPolicy(Model):
    """
    Stochastic actor: outputs mean & log std, samples with reparameterisation,
    tanh squashes to [-1, 1], returns action and corrected log-prob.
    """
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.backbone = mlp([obs_dim, *hidden, 2*act_dim])

    def call(self, obs):
        x = self.backbone(obs)
        mu, log_std = tf.split(x, num_or_size_splits=2, axis=-1)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        return mu, std

    @tf.function(reduce_retracing=True)
    def sample(self, obs):
        mu, std = self(obs)
        eps = tf.random.normal(tf.shape(mu))
        z = mu + std * eps
        a = tf.tanh(z)
        # log N(z|mu,std)
        log_prob = -0.5 * (((z - mu) / (std + 1e-8))**2 + 2.0*tf.math.log(std + 1e-8) + tf.math.log(2.0*math.pi))
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        # tanh correction
        log_prob -= tf.reduce_sum(tf.math.log(1.0 - tf.square(a) + 1e-6), axis=-1)
        return a, log_prob


class QCritic(Model):
    """Q(s,a) scalar head."""
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1])

    @tf.function(reduce_retracing=True)
    def call(self, obs, act):
        x = tf.concat([obs, act], axis=-1)
        q = self.net(x)
        return tf.squeeze(q, axis=-1)