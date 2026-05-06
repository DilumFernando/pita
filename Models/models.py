import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period, device=t.device, dtype=torch.float32))
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size).to(t.device))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * hidden_size)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa[:, None, :] * self.dropout(attn_out)
        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.dropout(self.mlp(mlp_in))
        return x


class DiTDriftNet(nn.Module):
    """
    DiT-style drift model for particle systems stored as flattened coordinates.

    Inputs match DriftNet.forward: x is [batch, n_particles * spatial_dim] and
    t is scalar, [batch], or [batch, 1]. Output is a flattened drift with the
    same shape as x.
    """

    def __init__(
        self,
        dim,
        n_particles,
        spatial_dim=3,
        hidden_size=192,
        cond_dim=64,
        n_heads=6,
        n_blocks=6,
        dropout=0.1,
    ):
        super().__init__()
        if dim != n_particles * spatial_dim:
            raise ValueError(
                f"DiTDriftNet expected dim == n_particles * spatial_dim, "
                f"got dim={dim}, n_particles={n_particles}, spatial_dim={spatial_dim}"
            )
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")

        self.dim = dim
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.input_proj = nn.Linear(spatial_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_particles, hidden_size))
        self.time_embed = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        self.output_proj = nn.Linear(hidden_size, spatial_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t):
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        t = t.reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        elif t.numel() != x.shape[0]:
            raise ValueError(f"t should have {x.shape[0]} entries or 1 entry, got {tuple(t.shape)}")

        tokens = x.reshape(-1, self.n_particles, self.spatial_dim)
        tokens = self.input_proj(tokens) + self.pos_embed
        c = F.silu(self.time_embed(t.to(device=x.device, dtype=x.dtype)))

        for block in self.blocks:
            tokens = block(tokens, c)

        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        return self.output_proj(tokens).reshape(x.shape[0], self.dim)

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
