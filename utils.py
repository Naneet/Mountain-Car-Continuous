import torch

def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    G = 0

    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done == True:
            G = 0
        G = reward + gamma * G
        returns.insert(0, G)

    return torch.tensor(returns, dtype=torch.float32)

def compute_advantages(returns, values):
    advantages = returns-values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages