import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        return x + self.dropout(torch.relu(self.second(torch.relu(self.first(x)))))


class PersonalNetwork(nn.Module):
    def __init__(self, n_features, n_channels=0):
        super().__init__()
        self.temporal = None
        if n_channels:
            self.temporal = nn.Sequential(nn.Conv1d(n_channels, 24, 1), *[TemporalBlock(24, d) for d in [1, 2, 4, 8, 16]])
        self.head = nn.Sequential(nn.Linear(n_features + (24 if n_channels else 0), 128), nn.SiLU(),
                                  nn.Dropout(.15), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, sequence):
        if self.temporal is not None:
            features = torch.cat([features, self.temporal(sequence)[:, :, -1]], dim=-1)
        return self.head(features)
