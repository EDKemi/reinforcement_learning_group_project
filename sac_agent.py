import tensorflow as tf
from sac_networks import GaussianPolicy, QCritic
from sac_utils import soft_update_vars
from sac_utils import get_logger


class SACAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=None):
        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # networks
        self.actor = GaussianPolicy(obs_dim, act_dim)
        self.critic1 = QCritic(obs_dim, act_dim)
        self.critic2 = QCritic(obs_dim, act_dim)
        self.target1 = QCritic(obs_dim, act_dim)
        self.target2 = QCritic(obs_dim, act_dim)
        # build
        dummy_s = tf.zeros((1, obs_dim), dtype=tf.float32)
        dummy_a = tf.zeros((1, act_dim), dtype=tf.float32)
        self.actor.sample(dummy_s)
        self.critic1(dummy_s, dummy_a); self.critic2(dummy_s, dummy_a)
        self.target1(dummy_s, dummy_a); self.target2(dummy_s, dummy_a)
        # copy weights
        self.target1.set_weights(self.critic1.get_weights())
        self.target2.set_weights(self.critic2.get_weights())

        # optimisers
        self.pi_opt = tf.keras.optimizers.Adam(lr)
        self.q1_opt = tf.keras.optimizers.Adam(lr)
        self.q2_opt = tf.keras.optimizers.Adam(lr)

        # temperature (alpha)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha_opt = tf.keras.optimizers.Adam(lr)
        self.target_entropy = -float(act_dim) if target_entropy is None else target_entropy
        self.logger = get_logger(name="sac.agent")
        self.logger.info("Initialized SACAgent")

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @tf.function(reduce_retracing=True)
    def act(self, obs, eval_mode=False):
        obs = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        if eval_mode:
            mu, std = self.actor(obs)
            a = tf.tanh(mu)
            return tf.squeeze(a, axis=0)
        a, _ = self.actor.sample(obs)
        return tf.squeeze(a, axis=0)

    @tf.function()
    def update(self, batch):
        s = tf.convert_to_tensor(batch["s"], dtype=tf.float32)
        a = tf.convert_to_tensor(batch["a"], dtype=tf.float32)
        r = tf.expand_dims(tf.convert_to_tensor(batch["r"], dtype=tf.float32), -1)
        s2 = tf.convert_to_tensor(batch["s2"], dtype=tf.float32)
        d = tf.expand_dims(tf.convert_to_tensor(batch["d"], dtype=tf.float32), -1)

        with tf.GradientTape(persistent=True) as tape:
            # target value
            a2, logp2 = self.actor.sample(s2)
            q1_t = tf.expand_dims(self.target1(s2, a2), -1)
            q2_t = tf.expand_dims(self.target2(s2, a2), -1)
            q_t_min = tf.minimum(q1_t, q2_t)
            backup = r + (1.0 - d) * self.gamma * (q_t_min - self.alpha * tf.expand_dims(logp2, -1))

            # Q losses
            q1 = tf.expand_dims(self.critic1(s, a), -1)
            q2 = tf.expand_dims(self.critic2(s, a), -1)
            q1_loss = tf.reduce_mean(tf.square(q1 - tf.stop_gradient(backup)))
            q2_loss = tf.reduce_mean(tf.square(q2 - tf.stop_gradient(backup)))

            # Policy loss
            a_pi, logp = self.actor.sample(s)
            q1_pi = self.critic1(s, a_pi)
            q2_pi = self.critic2(s, a_pi)
            q_pi = tf.minimum(q1_pi, q2_pi)
            pi_loss = tf.reduce_mean(self.alpha * logp - q_pi)

            # Temperature loss (auto-tuning alpha)
            alpha_loss = -tf.reduce_mean(self.log_alpha * (self.target_entropy + tf.stop_gradient(logp)))

        # apply grads
        self.q1_opt.apply_gradients(zip(tape.gradient(q1_loss, self.critic1.trainable_variables),
                                        self.critic1.trainable_variables))
        self.q2_opt.apply_gradients(zip(tape.gradient(q2_loss, self.critic2.trainable_variables),
                                        self.critic2.trainable_variables))
        self.pi_opt.apply_gradients( zip(tape.gradient(pi_loss, self.actor.trainable_variables),
                                         self.actor.trainable_variables))
        self.alpha_opt.apply_gradients( zip(tape.gradient(alpha_loss, [self.log_alpha]),
                                            [self.log_alpha]) )

        # soft update targets
        soft_update_vars(self.target1.trainable_variables, self.critic1.trainable_variables, self.tau)
        soft_update_vars(self.target2.trainable_variables, self.critic2.trainable_variables, self.tau)

        return {
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "pi_loss": pi_loss,
            "alpha": self.alpha,
            "logp": tf.reduce_mean(logp),
        }

    def save(self, path_prefix):
        self.actor.save_weights(path_prefix + "_actor.h5")
        self.critic1.save_weights(path_prefix + "_q1.h5")
        self.critic2.save_weights(path_prefix + "_q2.h5")

    def load(self, path_prefix):
        dummy_s = tf.zeros((1, self.obs_dim)); dummy_a = tf.zeros((1, self.act_dim))
        self.actor.sample(dummy_s); self.critic1(dummy_s, dummy_a); self.critic2(dummy_s, dummy_a)
        self.actor.load_weights(path_prefix + "_actor.h5")
        self.critic1.load_weights(path_prefix + "_q1.h5")
        self.critic2.load_weights(path_prefix + "_q2.h5")
        # refresh targets
        self.target1.set_weights(self.critic1.get_weights())
        self.target2.set_weights(self.critic2.get_weights())
