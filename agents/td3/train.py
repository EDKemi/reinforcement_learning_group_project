# Training loop for TD3
# Run  "python -m agents.td3.train_td3" to train the agent

import numpy as np
import gymnasium as gym

from .config import TD3Config
from .td3_agent import TD3Agent

def main():

    # Setup the environment and the agent
    env = gym.make("BipedalWalker-v3")
    cfg = TD3Config()
    agent = TD3Agent(env, config=cfg)

    # Training loop
    total_episodes = 1_000_000
    episode_return = 0.0
    episode_len = 0

    obs, _ = env.reset(seed=cfg.seed)

    try:
        for episode in range(total_episodes):

            # Warmup with random actions to fill buffer
            if episode < cfg.start_steps:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(obs, action, reward, next_obs, done)

            # Start training once we have enough data
            if episode >= cfg.start_steps and len(agent.replay) >= cfg.batch_size:
                agent.train_step()

            obs = next_obs
            episode_return += reward
            episode_len += 1

            if done:
                print("{:.2f}%: episode={} | return={:.2f} | len={}".format(episode/total_episodes * 100, episode, episode_return, episode_len))
                obs, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

            if (episode + 1) % 25_000 == 0:
                step_dir = "agents/td3/checkpoints/episode_{}".format(episode + 1)
                agent.save(step_dir)
                print("Saved model at episode {} to {}".format(episode + 1, step_dir))


            # Save first run too
            if episode == 0:
                step_dir = "agents/td3/checkpoints/first"
                agent.save(step_dir)
                print("Saved model at episode {} to {}".format(episode + 1, step_dir))

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        env.close()
        agent.save("agents/td3/checkpoints/final")
        print("Saved model at episode {} to agents/td3/checkpoints".format(episode + 1))

if __name__ == "__main__":
    main()
