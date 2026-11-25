"""Past-only convolutional encoder and a 36-step direct prediction head."""

import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.first = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=self.padding)
        self.second = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=self.padding)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        a = torch.relu(self.first(x)[..., :-self.padding])
        return x + self.dropout(torch.relu(self.second(a)[..., :-self.padding]))


class TrajectoryNetwork(nn.Module):
    def __init__(self, features, channels, steps=36):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(channels, 24, 1), *[CausalBlock(24, d) for d in [1, 2, 4, 8, 16, 32]])
        self.head = nn.Sequential(nn.Linear(features + 48, 128), nn.SiLU(), nn.Dropout(.15),
                                  nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, steps))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, sequence):
        temporal = self.encoder(sequence)
        return self.head(torch.cat([x, temporal[..., -1], temporal.mean(dim=-1)], dim=-1))
