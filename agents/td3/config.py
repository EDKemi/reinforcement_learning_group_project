# TD3 Agent Config

class TD3Config:
    def __init__(self):
        # This just stores all the config for the TD3 agent

        # Discount and target update
        self.gamma = 0.99
        self.tau = 0.005

        # TD3 specific
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2

        # Optimisers
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003

        # Exploration
        self.exploration_noise = 0.1

        # Training
        self.batch_size = 256
        self.buffer_size = 1_000_000
        self.start_steps = 25_000

        # Seed for reproducibility
        self.seed = 123