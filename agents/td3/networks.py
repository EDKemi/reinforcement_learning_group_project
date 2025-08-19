# TD3 Networks for Actor and Critic
import torch
import torch.nn as nn

class Actor(nn.Module):
        # Actor network as the policy

        def __init__(self, obs_dim, act_dim, action_limit, hidden=(256, 256)):
            super().__init__()
            # Create the model
            self.model = nn.Sequential(
                nn.Linear(obs_dim, hidden[0]), 
                nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]), 
                nn.ReLU(),
                nn.Linear(hidden[1], act_dim), 
                nn.Tanh()
            )
            self.action_limit = float(action_limit)

        def forward(self, obs):
            return self.model(obs) * self.action_limit


class Critic(nn.Module):
    # Q-value network as critic

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.model(x)