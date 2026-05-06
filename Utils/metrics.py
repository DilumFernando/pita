import torch
import math
import numpy as np
from Utils.misc import normalize_weights

try:
    import ot
except ModuleNotFoundError:
    ot = None


def _require_ot():
    if ot is None:
        raise ModuleNotFoundError(
            "The optional dependency `ot` is required for Wasserstein metrics. "
            "Install POT or avoid calling the Wasserstein helper functions."
        )


def ess_from_weights(w, eps=1e-10):
    """
    Numerically stable ESS from log-weights.

    logw: [N] tensor of log-weights
    """
    mean_w = w.mean()
    mean_w2 = (w ** 2).mean()
    return (mean_w ** 2) / (mean_w2 + eps)

def ess_eval(A_trajectory, eps=1e-10):
    """
    Compute ESS for each timestep of log-weight trajectories.
    A_trajectory: [T, N] tensor of log-weights.
    """
    ess_vals = []
    for t in range(A_trajectory.shape[0]):
        ess_vals.append(ess_from_weights(A_trajectory[t], eps))
    return torch.stack(ess_vals)

def unweighted_w2_from_samples(p_samples: torch.Tensor, q_samples: torch.Tensor) -> torch.Tensor:
    """
    Compute the Wasserstein-2 distance between two sets of samples.
    Args:
        p_samples: Samples from the first distribution, shape (num_samples, dim).
        q_samples: Samples from the second distribution, shape (num_samples, dim).
    Returns:
        torch.Tensor: Wasserstein-2 distance.
    """
    _require_ot()
    cost_matrix = ot.dist(p_samples, q_samples, metric="sqeuclidean")
    p_weights = torch.ones_like(p_samples[:, 0]) / p_samples.shape[0]  # (num_samples,)
    q_weights = torch.ones_like(q_samples[:, 0]) / q_samples.shape[0]  # (num_samples,)
    return ot.emd2(p_weights, q_weights, cost_matrix) ** 0.5

def weighted_w2(generated_samples, true_samples, weights):
    # x: (N, d)
    # y: (M, d)
    # weights: (N,) importance weights (not necessarily normalized)

    _require_ot()

    generated_samples_np = generated_samples.detach().cpu().numpy()
    true_samples_np = true_samples.detach().cpu().numpy()
    # w_np = w.detach().cpu().numpy()

    normalized_weights_generated = normalize_weights(weights).detach().cpu().numpy()
    normalized_weights_true = normalize_weights(np.ones(len(true_samples_np)))


    # cost matrix
    C = ot.dist(generated_samples_np, true_samples_np, metric='euclidean') ** 2

    # Wasserstein-2 squared
    w2_sq = ot.emd2(normalized_weights_generated, normalized_weights_true, C)

    w2 = math.sqrt(w2_sq)
    
    return w2


def mmd_rbf(x, y, bandwidth=None):
    x = x.detach()
    y = y.detach()

    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if bandwidth is None:
        combined = torch.cat([x, y], dim=0)
        pairwise = torch.cdist(combined, combined).pow(2)
        bandwidth = torch.median(pairwise[pairwise > 0])
        if not torch.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    gamma = 1.0 / (2.0 * bandwidth)
    k_xx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return k_xx + k_yy - 2.0 * k_xy
