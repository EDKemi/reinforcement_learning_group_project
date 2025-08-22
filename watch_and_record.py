import os, time, argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from sac_agent import SACAgent
from sac_utils import get_logger
logger = get_logger(name="sac.watch")

FPS = 60


def run_episode(env, policy_fn, seed=123, sleep_when_human=True):
    obs, _ = env.reset(seed=seed)
    done, ret = False, 0.0
    while not done:
        action = policy_fn(obs)
        obs, r, term, trunc, _ = env.step(action)
        ret += r
        if sleep_when_human and getattr(env, "render_mode", None) == "human":
            time.sleep(1.0 / FPS)
        done = term or trunc
    return ret


def before_watch(env_id: str):
    env = gym.make(env_id, render_mode="human")
    policy = lambda _obs: env.action_space.sample()
    ret = run_episode(env, policy, seed=42)
    logger.info(f"[BEFORE: WATCH] Return: {ret:.1f}")
    env.close()


def before_record(env_id: str, video_dir: str = "videos", episodes: int = 1, name_prefix: str = "random_before"):
    os.makedirs(video_dir, exist_ok=True)
    base = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(base, video_folder=video_dir, episode_trigger=lambda ep: True, name_prefix=name_prefix)
    policy = lambda _obs: env.action_space.sample()
    for ep in range(episodes):
        ret = run_episode(env, policy, seed=100 + ep, sleep_when_human=False)
        logger.info(f"[BEFORE: RECORDED] Episode {ep+1} return: {ret:.1f}")
    env.close()


def after_watch(env_id: str, weights_prefix: str):
    env = gym.make(env_id, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim)
    agent.actor.load_weights(f"{weights_prefix}_actor.weights.h5")
    agent.critic1.load_weights(f"{weights_prefix}_q1.weights.h5")
    agent.critic2.load_weights(f"{weights_prefix}_q2.weights.h5")
    agent.target1.set_weights(agent.critic1.get_weights())
    agent.target2.set_weights(agent.critic2.get_weights())
    policy = lambda obs: agent.act(obs, eval_mode=True).numpy()
    ret = run_episode(env, policy, seed=123)
    logger.info(f"[AFTER: WATCH] Return: {ret:.1f}")
    env.close()


def after_record(env_id: str, weights_prefix: str, video_dir: str = "videos", episodes: int = 1, name_prefix="sac_after"):
    os.makedirs(video_dir, exist_ok=True)
    base = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(base, video_folder=video_dir, episode_trigger=lambda ep: True, name_prefix=name_prefix)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim)
    agent.actor.load_weights(f"{weights_prefix}_actor.weights.h5")
    agent.critic1.load_weights(f"{weights_prefix}_q1.weights.h5")
    agent.critic2.load_weights(f"{weights_prefix}_q2.weights.h5")
    agent.target1.set_weights(agent.critic1.get_weights())
    agent.target2.set_weights(agent.critic2.get_weights())
    policy = lambda obs: agent.act(obs, eval_mode=True).numpy()
    for ep in range(episodes):
        ret = run_episode(env, policy, seed=456 + ep, sleep_when_human=False)
        logger.info(f"[AFTER: RECORDED] Episode {ep+1} return: {ret:.1f}")
    env.close()


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="BipedalWalker-v3")
    ap.add_argument("--before_watch", action="store_true")
    ap.add_argument("--before_record", action="store_true")
    ap.add_argument("--after_watch", action="store_true")
    ap.add_argument("--after_record", action="store_true")
    ap.add_argument("--video_dir", default="videos")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--weights_prefix", default="results/BipedalWalker-v3/sac")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.before_watch:
        before_watch(args.env_id)
    if args.before_record:
        before_record(args.env_id, args.video_dir, args.episodes)
    if args.after_watch:
        after_watch(args.env_id, args.weights_prefix)
    if args.after_record:
        after_record(args.env_id, args.weights_prefix, args.video_dir, args.episodes)
