import torch
import torch.nn.functional as F
from Energies.interpolations import U_t

# @torch.inference_mode()
def mode_weights_from_particles(x: torch.Tensor,
                                mode_centers: torch.Tensor,
                                particle_weights: torch.Tensor | None = None):
    """
    x:              [N, 2]
    mode_centers:   [M, 2]  (e.g. true_modes)
    particle_weights: [N] or None (importance weights; can be unnormalized)

    returns:
      mode_w: [M] normalized to sum to 1
      labels: [N] argmin mode index
    """
    # assign by nearest center
    d2 = torch.cdist(x, mode_centers)  # [N, M]
    labels = torch.argmin(d2, dim=1)   # [N]

    M = mode_centers.shape[0]
    if particle_weights is None:
        w = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    else:
        w = particle_weights.to(device=x.device, dtype=x.dtype)
        w = w / (w.sum() + 1e-12)

    mode_w = torch.zeros(M, device=x.device, dtype=x.dtype)
    mode_w.scatter_add_(0, labels, w)
    mode_w = mode_w / (mode_w.sum() + 1e-12)
    return mode_w, labels

def soft_mode_weights_from_particles(x, mode_centers, particle_weights=None, tau=10.0, eps=1e-12):
    d2 = torch.cdist(x, mode_centers).pow(2)   # [N, M]
    resp = torch.softmax(-tau * d2, dim=1)     # [N, M]

    if particle_weights is None:
        w = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    else:
        w = particle_weights / (particle_weights.sum() + eps)

    mode_w = (w[:, None] * resp).sum(dim=0)
    mode_w = mode_w / (mode_w.sum() + eps)
    return mode_w, resp

def mode_weights_conditional(x, modes, logits, beta, particle_weights=None, eps=1e-12):
    """
    Conditional membership estimator of modal probabilities.

    x: [N, D] particles
    modes: [M, D] mode centers
    logits: [M] or [N, M] mode logits θ_k(t)
    beta: scalar or [N]
    particle_weights: [N] importance weights

    returns
        mode_weights: [M]
        responsibilities: [N, M]
    """

    N, D = x.shape
    M = modes.shape[0]

    device = x.device
    dtype = x.dtype

    # normalize particle weights
    if particle_weights is None:
        w = torch.ones(N, device=device, dtype=dtype) / N
    else:
        w = particle_weights / (particle_weights.sum() + eps)

    # ensure beta shape
    if torch.is_tensor(beta):
        beta = beta.view(-1)
        if beta.numel() == 1:
            beta = beta.expand(N)
    else:
        beta = torch.full((N,), beta, device=device, dtype=dtype)

    # pairwise squared distances
    diff2 = torch.cdist(x, modes).pow(2)        # [N, M]

    # energy term
    energy = -0.5 * beta[:, None] * diff2       # [N, M]

    # handle logits shape
    if logits.ndim == 1:
        logits = logits[None, :].expand(N, M)   # [N, M]

    # log responsibilities
    log_eta = logits + energy

    # normalize
    log_eta = log_eta - torch.logsumexp(log_eta, dim=1, keepdim=True)
    eta = log_eta.exp()                         # [N, M]

    # weighted modal probabilities
    mode_weights = (w[:, None] * eta).sum(dim=0)  # [M]
    # mode_weights = mode_weights / (mode_weights.sum() + eps)

    return mode_weights, eta

@torch.inference_mode
def p_t(x, t, means, U_net, modes, target, prior, energy_model):
    density = torch.exp(-U_t(x, t, means, U_net, modes, target, prior, energy_model))
    return density

def delta_t_at_T(beta, device):
    delta_t_min = torch.tensor(10**(-4)).to(device)
    delta_t = delta_t_min + 0.01 * (1 / beta)
    max_indices = torch.where(delta_t >= 0.005)
    delta_t[max_indices] = 0.005
    return delta_t

def beta_schedule(t, beta_max=10, beta_min=1e-3):
    # return beta_max + (beta_min - beta_max) * t
    return beta_max * (2**(-10*t))

# @torch.inference_mode
def normalize_weights(weights):
  normalized_weights = weights / weights.sum()
  return normalized_weights

@torch.inference_mode
def label_assignment_hard(x_final: torch.Tensor, true_modes: torch.Tensor):
    """
    x_final:    [N, D]
    true_modes: [K, D]
    returns:
      weights: [K]   (sum to 1)
      counts:  [K]
      assign:  [N]   mode index for each sample
    """
    x_ = x_final
    m = true_modes.to(x_.device)

    # squared distances: [N, K]
    d2 = torch.cdist(x_, m, p=2).pow(2)
    labels = d2.argmin(dim=1)
    return labels
