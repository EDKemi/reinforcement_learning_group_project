import gymnasium as gym
import random
import tensorflow as tf

import numpy as np

from networks import ActorController, CriticController
from utils import ReplayMemory, update_target_network, OrnsteinUhlenbeckActionNoise, ddpg_add_exploration_noise

random.seed(42)
np.random.seed(42)


env = gym.make("BipedalWalker-v3", render_mode='human')
replay = ReplayMemory(100)

num_episodes = 10
max_episode_steps = 500
minibatch_size = 64
gamma = tf.cast(0.95, tf.float32)

actor_controller = ActorController(env.observation_space.shape[0], env.action_space.shape[0])
critic_controller = CriticController(env.observation_space.shape[0], env.action_space.shape[0])

exploration_noise = OrnsteinUhlenbeckActionNoise(size=env.action_space.shape[0])

actor = actor_controller.actor
critic = critic_controller.critic

predict_array = np.zeros((1, env.observation_space.shape[0]))

rewards = []

for episode in range(num_episodes):
    episode_complete = False
    episode_reward = 0
    step = 0
    state, _ = env.reset()

    while not episode_complete:
        predict_array[0] = state
        action = actor.predict(predict_array, verbose=0)[0]
        action = ddpg_add_exploration_noise(exploration_noise, action, noise_scale=0.1)

        state_prime, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        episode_complete = terminated or truncated or step >= max_episode_steps

        replay.add_experience(state, action, reward, state_prime)

        experience_batch = replay.sample(minibatch_size)
        s_array = tf.convert_to_tensor([x[0] for x in experience_batch], dtype=tf.float32)
        a_array = tf.convert_to_tensor([x[1] for x in experience_batch], dtype=tf.float32)
        r_array = tf.convert_to_tensor([x[2] for x in experience_batch], dtype=tf.float32)
        s_prime_array = tf.convert_to_tensor([x[3] for x in experience_batch], dtype=tf.float32)

        target_q_actions = actor_controller.actor_target(s_prime_array)
        target_q = r_array + gamma * tf.cast(critic_controller.critic_target(tf.concat([s_prime_array, target_q_actions], 1)), tf.float32)

        with tf.GradientTape() as tape:
            current_q = critic(tf.concat([s_array, a_array], 1))
            current_q = tf.cast(current_q, tf.float32)
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))

        critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
        critic_controller.optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        with tf.GradientTape() as tape:
            current_actions = actor(s_array, training=True)
            current_q = critic(tf.concat([s_array, current_actions], 1), training=True)
            actor_loss = -tf.reduce_mean(current_q)

        actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
        actor_controller.optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))

        update_target_network(actor, actor_controller.actor_target)
        update_target_network(critic, critic_controller.critic_target)

        step += 1
        episode_reward += reward
        state = state_prime

    rewards.append(episode_reward)

print(rewards)
env.close()