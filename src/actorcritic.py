import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch
import numpy as np
from collections import deque
import random

class ImagePreprocessor:
    def __init__(self):
        pass
    
    def normalize_image(self, img):
        return img.astype(np.float32) / 255.0

    def transpose_image(self, img):
        return np.transpose(img, (2, 0, 1))
    
    def preprocess_image(self, img):
        img = self.normalize_image(img)
        img = self.transpose_image(img)
        return torch.from_numpy(img).unsqueeze(0)
        # return torch.from_numpy(img)

class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.l1 = nn.Conv2d(3, 3, 3)
        self.l2 = nn.Conv2d(3, 1, 4)
        self.l3 = nn.Linear(124*124, 124)
        self.mean_linear = nn.Linear(124, action_size)
        self.log_std_linear = nn.Linear(124, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.l3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        return mean, log_std
    
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(3, 3, 3)
        self.l2 = nn.Conv2d(3, 1, 4)
        self.l3 = nn.Linear(124*124, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.l3(x))
        return x
    
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.002
        self.action_size = 4

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.reply = ReplayBuffer(buffer_size=500, batch_size=32)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr = self.lr)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr = self.lr)

    def get_action(self, state):
        mean, log_std = self.pi.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        mean = torch.tanh(mean)
        return action, mean
    
    # def update(self):
    #     if len(self.reply.buffer) < 500:
    #         return
    #     state_batch, action_batach, reward_batch, next_state_bach, done_batch =  self.reply.get_batch()
    #     v_targets = reward_batch + self.gamma * self.v(next_state_bach) * (1 - done_batch)
    #     v_targets.detach()
    #     v_values = self.v(state_batch)
    #     critic_loss = F.mse_loss(v_values, v_targets)

    #     self.optimizer_v.zero_grad()
    #     critic_loss.backward()
    #     self.optimizer_v.step()

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add_memory(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        # 各要素を ndarray に揃えて返す
        states = np.stack([b[0] for b in batch], axis=0).squeeze(1)
        actions = np.stack([b[1].detach().numpy() for b in batch], axis=0).squeeze(1)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch], axis=0).squeeze(1)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


if __name__ == '__main__':
    policy = PolicyNet(action_size=3)