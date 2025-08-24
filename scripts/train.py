from src.ppo_bipedal import train

if __name__ == "__main__":
    ac, history = train(
        seed=42,
        total_updates=250,
        steps_per_update=2048,
        gamma=0.99, lam=0.95,
        clip_eps=0.2, vf_coef=0.5,
        pi_lr=3e-4, v_lr=8e-4,
        log_every=1,
        save_path="results/ppo_actorcritic.pt"
    )