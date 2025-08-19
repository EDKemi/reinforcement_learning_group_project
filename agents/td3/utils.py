# Helpers for setting seeds and soft target update.

import random
import numpy as np
import torch

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def soft_update(online, target, tau):
    # Soft update target network parameters.

    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.copy_(tau * p_online.data + (1.0 - tau) * p_target.data)
