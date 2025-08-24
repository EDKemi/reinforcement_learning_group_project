import os
import csv
import argparse
import numpy as np
import tensorflow as tf
from sac_utils import set_seed, make_env, ReplayBuffer
from sac_agent import SACAgent
from sac_utils import get_logger
import json


def evaluate(env, agent, episodes=5):
    rets = []
    for _ in range(episodes):
        s, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a = agent.act(s, eval_mode=True).numpy()
            s, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
    return float(np.mean(rets)), float(np.std(rets)), rets


def train(
    env_id: str = "BipedalWalker-v3",
    results_dir: str | None = None,
    total_steps: int = 300_000,
    start_steps: int = 10_000,
    batch_size: int = 256,
    eval_every: int = 10_000,
    seed: int = 0,
):
    results_dir = results_dir or os.path.join("results", env_id)
    os.makedirs(results_dir, exist_ok=True)

    logger = get_logger(name="train", log_path=os.path.join(results_dir, "training.log"))
    logger.info(f"Starting training | env={env_id} | results_dir={results_dir}")

    set_seed(seed)
    env = make_env(env_id, seed=seed)
    eval_env = make_env(env_id, seed=seed + 1)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=-0.5 * act_dim)
    buf = ReplayBuffer(obs_dim, act_dim, size=1_000_000)

    csv_path = os.path.join(results_dir, "eval_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "episodes", "avg_return", "std_return", "all_returns_json"
        ])

    # Warmup
    s, _ = env.reset()
    for _ in range(start_steps):
        a = env.action_space.sample()
        s2, r, term, trunc, _ = env.step(a)
        d = float(term or trunc)
        buf.store(s, a, r, s2, d)
        s, _ = env.reset() if d else (s2, {})
    logger.info(f"Warmup complete: {start_steps} random steps")

    # Train
    episodes = 0
    s, _ = env.reset()
    for t in range(1, total_steps + 1):
        a = agent.act(s, eval_mode=False).numpy()
        s2, r, term, trunc, _ = env.step(a)
        d = float(term or trunc)
        buf.store(s, a, r, s2, d)

        if d:  # episode finished
            episodes += 1
            s, _ = env.reset()
        else:
            s = s2

        batch = buf.sample(batch_size)
        metrics = agent.update(batch)

        if t % 1000 == 0:
            m = {k: float(v.numpy() if isinstance(v, tf.Tensor) else v) for k, v in metrics.items()}
            logger.info(
                f"t={t:6d} | episodes={episodes} | "
                f"q1={m['q1_loss']:.3f} q2={m['q2_loss']:.3f} "
                f"pi={m['pi_loss']:.3f} alpha={m['alpha']:.3f}"
            )

        if t % eval_every == 0:
            avg, std, all_rets = evaluate(eval_env, agent, episodes=5)
            logger.info(f"[EVAL] step={t} | episodes={episodes} | avg_return={avg:.1f} ± {std:.1f}")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    t, episodes, avg, std,
                    json.dumps(all_rets)
                ])

            agent.save(os.path.join(results_dir, f"sac_{t}"))

    # Final weights
    agent.actor.save_weights(os.path.join(results_dir, "sac_actor.weights.h5"))
    agent.critic1.save_weights(os.path.join(results_dir, "sac_q1.weights.h5"))
    agent.critic2.save_weights(os.path.join(results_dir, "sac_q2.weights.h5"))
    logger.info("Final model weights saved")

    env.close(); eval_env.close()
    return results_dir


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", default="BipedalWalker-v3")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--start_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(**vars(args))
