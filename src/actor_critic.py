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

class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.l1 = nn.Linear(action_size, 500)
        self.l2 = nn.Linear(500, 124)
        self.mean_linear = nn.Linear(124, action_size)
        self.log_std_linear = nn.Linear(124, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        return mean, log_std
    
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 500)
        self.l2 = nn.Linear(500, 124)
        self.l3 = nn.Linear(124, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        # print(std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t*0.01
        mean = torch.tanh(mean)*0.01
        # print(action)
        # mean = torch.tanh(mean)*2*np.pi*0.002
        # print(action)
        # print(action)
        return action, mean
    
    # def update_batch(self):
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

    def update(self, state, action, mean, reward, next_state, done):
        # ---- critic ----
        v_target = reward + self.gamma * self.v(next_state) * (1 - done)

        v_value = self.v(state)
        loss_v = F.mse_loss(v_value, v_target)

        # ---- update ----
        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        # ---- actor ----
        loss_pi = - self.v(state).mean()

        # ---- update ----
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()

    def reward(self, current_pos, box_pos, t):
        r1 = 0
        done = False
        # print(current_pos, box_pos)
        d  =  np.linalg.norm(current_pos-box_pos)
        r0 = -d*10
        print(r0)
        if d < 0.1:
            done = True
            r1 = 300
        elif t > 5:
            done = True
            r1 = -100

        return r0+r1, done



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