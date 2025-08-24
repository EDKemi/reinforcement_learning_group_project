import os
import time
from train_sac_tf import train
from watch_and_record import before_record, after_watch, after_record
from sac_utils import get_logger
logger = get_logger(name="sac.runner")

ENV_IDS = ["BipedalWalkerHardcore-v3"]  # "BipedalWalker-v3. Add one at a time to update parameters for environment,
# or have both in list to run with same parameters


def run_all(total_steps=300_000, start_steps=10_000, batch_size=256, eval_every=10_000, seed=0,
    episodes_before=1, do_watch_after=False, do_record_after=False,):

    os.makedirs("results", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    for env_id in ENV_IDS:
        logger.info(f"Running pipeline for {env_id}")
        results_dir = os.path.join("results", env_id)
        video_dir = os.path.join("videos", env_id)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        # BEFORE: record random
        np = f"random_before_{env_id}"
        before_record(env_id=env_id, video_dir=video_dir, episodes=episodes_before, name_prefix=np)

        # TRAIN
        start_time = time.time()
        train(
            env_id=env_id,
            results_dir=results_dir,
            total_steps=total_steps,
            start_steps=start_steps,
            batch_size=batch_size,
            eval_every=eval_every,
            seed=seed,
        )
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"Training took {elapsed_min:.2f} min")

        # AFTER: live watch
        if do_watch_after:
            after_watch(env_id=env_id, weights_prefix=os.path.join(results_dir, "sac"))

        # AFTER: record trained
        if do_record_after:
            after_record(env_id=env_id, weights_prefix=os.path.join(results_dir, "sac"),
                         video_dir=video_dir, episodes=1, name_prefix=f"sac_after_{env_id}")


if __name__ == "__main__":
    run_all(
        total_steps=2_000_000, #1_000_000 first time training BipedalWalkerHardcore-v3, 300_000 training BipedalWalker-v3
        start_steps=30_000, # 10_000 training BipedalWalker-v3
        batch_size=256,
        eval_every=10_000,
        seed=0,
        episodes_before=1,
        do_watch_after=True,   # set True to pop windows after each training
        do_record_after=True,  # set True to save "after" MP4s
    )
