import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch

class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.l1 = nn.Conv2d(3, 3, 3)
        self.l2 = nn.Conv2d(3, 3, 3)
        self.l3 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_size)
        self.log_std = nn.Linear(128, action_size)

        self.relu = F.relu()
        self.pool = F.max_pool2d(2, 2)
        self.softmax = F.softmax(dim=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        return mean, log_std
    
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(3, 3, 3)
        self.l2 = nn.Conv2d(3, 3, 3)
        self.l3 = nn.Linear(128, 128)

        self.relu = F.relu()
        self.pool = F.max_pool2d(2, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.l3
        return x
    
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.002
        self.action_size = 4

        self.pi = PolicyNet()
        self.v = ValueNet()
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
    
    # def updaet(self, state, action, )

if __name__ == '__main__':
    policy = PolicyNet(action_size=3)