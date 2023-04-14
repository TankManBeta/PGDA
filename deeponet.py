import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DeepONet(nn.Module):
    def __init__(self,b_dim,t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim
        
        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
        )
        
        self.b = Parameter(torch.zeros(1))
        
    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)
        
        res = torch.einsum("bi,bi->b", x, l)
        res = res.unsqueeze(1) + self.b
        
        return res
