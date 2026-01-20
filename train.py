import torch
from torch import optim
import gymnasium as gym
from torch.distributions import Normal

from model import ActorCritic
from buffer import RolloutBuffer
from utils import compute_returns, compute_advantages

num_updates = 500
rollout_steps = 2048
mini_batch_size = 128
ppo_epochs = 10
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01
episode_no = 1
reward_tracker = []
episode_reward = 0

buffer = RolloutBuffer()
env = gym.make("MountainCarContinuous-v0")
model = ActorCritic()
model.train()
actor_optimizer = optim.Adam(list(model.actor.parameters())+[model.log_std], lr = 3e-4)
critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)

for iteration in range(num_updates):

    buffer.clear()
    state, _ = env.reset() 

    for rollout in range(rollout_steps):

        state = torch.tensor(state).float()
        with torch.no_grad():
            action, log_prob, value = model.get_action_and_value(state)
        clipped_action = torch.clip(action, -1, 1).numpy()
        
        next_state, reward, terminated, truncated, info = env.step(clipped_action)
        done = terminated or truncated
        episode_reward += reward

        buffer.push(state, action, log_prob, reward, value, done)
        
        state = next_state
        if done:
            print(f"Episode: {episode_no} | Reward: {episode_reward:.5f}")
            episode_no += 1
            episode_reward = 0
            reward_tracker.append(reward)
            state, _ = env.reset()

    states, actions, old_log_probs, rewards, values, dones = buffer.get()
    returns = compute_returns(rewards, dones)
    advantages = compute_advantages(returns, values)

    for epoch in range(ppo_epochs):

        indices = torch.randperm(rollout_steps)

        for start in range(0, rollout_steps, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            mean = model.actor(mb_states)
            std = torch.exp(model.log_std)
            dist = Normal(mean, std)

            mb_new_log_probs = dist.log_prob(mb_actions).sum(-1)
            mb_values = model.critic(mb_states).squeeze(-1)
            mb_entropy = dist.entropy().sum(-1)

            ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
            unclipped = ratio * mb_advantages
            clipped = torch.clip(ratio, 1-clip_eps, 1+clip_eps) * mb_advantages

            actor_loss = -torch.min(unclipped, clipped).mean()
            entropy_loss = -mb_entropy.mean()
            critic_loss = (mb_values - mb_returns).pow(2).mean()

            loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            actor_optimizer.step()
            critic_optimizer.step()

torch.save(model.state_dict(), "ActorCritic.pth")