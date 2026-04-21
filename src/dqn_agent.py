import torch
import torch.nn as nn
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, state):
        return self.net(state)

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(2)

        with torch.no_grad():
            return self(state).argmax().item()