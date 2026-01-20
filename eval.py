import gymnasium as gym
import torch
from torch.distributions import Normal

from model import ActorCritic

model = ActorCritic()
checkpoint = torch.load("ActorCritic.pth")
model.load_state_dict(checkpoint)
model.eval()
env = gym.make("MountainCarContinuous-v0", render_mode="human")
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.tensor(state).float()
        with torch.no_grad():
            action = model.actor(state)
        action = torch.clamp(action, -1, 1).numpy()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
    print(f"Episode: {episode+1} | Reward: {total_reward}")