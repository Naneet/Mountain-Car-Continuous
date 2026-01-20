import torch

class RolloutBuffer:
    def __init__(self):
        self.buffer = []

    def clear(self):
        self.buffer = []

    def push(self, state, action, log_prob, reward, value, done):
        self.buffer.append((state, action, log_prob, reward, value, done))

    def get(self):
        states, actions, log_probs, rewards, values, dones = zip(*self.buffer)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(log_probs),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(values),
            torch.tensor(dones, dtype=torch.float32),
        )