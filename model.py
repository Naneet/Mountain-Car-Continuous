import torch
from torch import nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        self.log_std = nn.Parameter(torch.ones(1) * 0.5)

        self.critic = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def get_action_and_value(self, state):

        mean = self.actor(state)
        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state).squeeze(-1)
        return action, log_prob, value