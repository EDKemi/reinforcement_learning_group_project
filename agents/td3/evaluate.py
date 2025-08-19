import argparse
import gymnasium as gym
from .config import TD3Config
from .td3_agent import TD3Agent

def main():

    # Parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    # Create the environment and the agent
    env = gym.make("BipedalWalker-v3", render_mode="human")
    agent = TD3Agent(env, config=TD3Config())

    # Load the model
    agent.load(args.model_dir)

    # Evaluate
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r
            ep_len += 1
        print(f"eval_ep={ep+1} | return={ep_ret:.2f} | len={ep_len}")
    env.close()

if __name__ == "__main__":
    main()
