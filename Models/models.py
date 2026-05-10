import torch
import torch.nn as nn


def _as_batch_time(t, batch_size, device, dtype):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.expand(batch_size)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    return t


def _remove_center_of_mass(x):
    return x - x.mean(dim=1, keepdim=True)


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


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, coords, edge_index):
        row, col = edge_index
        coord_diff = coords[row] - coords[col]
        radial = coord_diff.pow(2).sum(dim=-1, keepdim=True)
        edge_input = torch.cat([h[row], h[col], radial], dim=-1)
        messages = self.edge_mlp(edge_input)

        coord_weights = self.coord_mlp(messages)
        coord_updates = coord_diff * coord_weights
        coords = coords + coords.new_zeros(coords.shape).index_add_(0, row, coord_updates)

        aggregated = h.new_zeros(h.shape).index_add_(0, row, messages)
        h = h + self.node_mlp(torch.cat([h, aggregated], dim=-1))
        return h, coords


class _BaseEGNNNet(nn.Module):
    def __init__(
        self,
        dim,
        n_particles=None,
        spatial_dim=3,
        hidden_dim=64,
        n_layers=4,
        remove_mean=True,
    ):
        super().__init__()
        if n_particles is None:
            if dim % spatial_dim != 0:
                raise ValueError(
                    f"Cannot infer n_particles because dim={dim} is not divisible by spatial_dim={spatial_dim}."
                )
            n_particles = dim // spatial_dim
        if n_particles * spatial_dim != dim:
            raise ValueError(
                f"EGNN expects dim == n_particles * spatial_dim, got {dim} != {n_particles} * {spatial_dim}."
            )

        self.dim = dim
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.remove_mean = remove_mean
        self.node_embedding = nn.Embedding(n_particles, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(EGNNLayer(hidden_dim) for _ in range(n_layers))
        self.register_buffer("edges", self._create_edges(n_particles), persistent=False)

    @staticmethod
    def _create_edges(n_particles):
        rows = []
        cols = []
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                rows.append(i)
                cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long)

    def _batched_edges(self, batch_size, device):
        base_edges = self.edges.to(device)
        offsets = torch.arange(batch_size, device=device).view(-1, 1, 1) * self.n_particles
        return (base_edges.view(1, 2, -1) + offsets).permute(1, 0, 2).reshape(2, -1)

    def _encode(self, x, t):
        batch_size = x.shape[0]
        coords = x.reshape(batch_size, self.n_particles, self.spatial_dim)
        if self.remove_mean:
            coords = _remove_center_of_mass(coords)

        node_ids = torch.arange(self.n_particles, device=x.device)
        h = self.node_embedding(node_ids).unsqueeze(0).expand(batch_size, -1, -1)
        h = h + self.time_embedding(_as_batch_time(t, batch_size, x.device, x.dtype)).unsqueeze(1)

        h = h.reshape(batch_size * self.n_particles, self.hidden_dim)
        coords = coords.reshape(batch_size * self.n_particles, self.spatial_dim)
        edge_index = self._batched_edges(batch_size, x.device)
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)
        h = h.reshape(batch_size, self.n_particles, self.hidden_dim)
        coords = coords.reshape(batch_size, self.n_particles, self.spatial_dim)
        return h, coords


class EGNNDriftNet(_BaseEGNNNet):
    def forward(self, x, t):
        input_coords = x.reshape(x.shape[0], self.n_particles, self.spatial_dim)
        if self.remove_mean:
            input_coords = _remove_center_of_mass(input_coords)
        _, coords = self._encode(x, t)
        velocity = coords - input_coords
        if self.remove_mean:
            velocity = _remove_center_of_mass(velocity)
        return velocity.reshape(x.shape[0], self.dim)


class EGNNEnergyNet(_BaseEGNNNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x, t):
        h, _ = self._encode(x, t)
        return self.readout(h.sum(dim=1)).squeeze(-1)


class EGNNFreeEnergyNet(FreeEnergyNet):
    pass


def build_model_bundle(dim, device, model_type="mlp", model_kwargs=None):
    model_kwargs = dict(model_kwargs or {})
    model_type = str(model_type or "mlp").lower()
    if model_type == "mlp":
        return DriftNet(dim).to(device), FreeEnergyNet(dim).to(device), PotentialNet(dim).to(device)
    if model_type == "egnn":
        drift = EGNNDriftNet(dim, **model_kwargs).to(device)
        free_energy = EGNNFreeEnergyNet(dim).to(device)
        potential = EGNNEnergyNet(dim, **model_kwargs).to(device)
        return drift, free_energy, potential
    raise ValueError(f"Unsupported model_type: {model_type}")
