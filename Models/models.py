import torch
import torch.nn as nn


class DriftNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 == 1 or dim == 1:
            self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, dim)
        )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, dim)
            )
        
    def forward(self, x, t):
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

class FreeEnergyNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 == 1 or dim == 1:
            self.net = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        else:
            self.net = nn.Sequential(
                nn.Linear(1, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 1)
            )
    def forward(self, t):
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        return self.net(t).squeeze(-1)

class LogitsNet(nn.Module):
    def __init__(self, dim, num_components):
        super().__init__()
        if dim == 1 or dim == 2:
            self.net = nn.Sequential(
            nn.Linear(1, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
            nn.Tanh(),
            nn.Linear(3, num_components)
        )
        else:
            self.net = nn.Sequential(
                nn.Linear(1, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, num_components)
            )
    def forward(self, t):
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        return self.net(t).squeeze(-1)

## PiecewiseLinearLogits
class PiecewiseLinearLogits(nn.Module):
    """
    Piecewise linear logits(t) schedule.

    Input:
        t : [B] or scalar

    Output:
        logits(t) : [B, M]
    """

    def __init__(self, num_modes, num_segments=10):
        super().__init__()

        self.num_modes = num_modes
        self.num_segments = num_segments

        # control points θ_0 ... θ_S
        self.logits = nn.Parameter(
            torch.zeros(num_segments + 1, num_modes)
        )  # [S+1, M]

    def forward(self, t):
        """
        t: [B] or scalar

        returns:
            logits(t): [B, M]
        """

        if t.ndim == 0:
            t = t.unsqueeze(0)

        t = torch.clamp(t, 0.0, 1.0)  # [B]

        s = t * self.num_segments     # [B]

        idx = torch.floor(s).long()   # [B]
        idx = torch.clamp(idx, 0, self.num_segments - 1)

        alpha = s - idx.float()       # [B]

        logit0 = self.logits[idx]     # [B, M]
        logit1 = self.logits[idx + 1] # [B, M]

        logits_t = (
            (1 - alpha.unsqueeze(-1)) * logit0
            + alpha.unsqueeze(-1) * logit1
        )  # [B, M]

        return logits_t


class PotentialNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 == 1 or dim == 1:
            self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 1)
            )

    def forward(self, x, t):
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt).squeeze(-1)
