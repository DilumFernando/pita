import math

import torch


ENERGY_DERIVATIVE_CLIP = 100.0


def _clip_energy_derivative(value):
    return torch.clamp(value, min=-ENERGY_DERIVATIVE_CLIP, max=ENERGY_DERIVATIVE_CLIP)


def _optional_potential(U_net, x, t):
    if U_net is None:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    return U_net(x, t)


def _flatten_energy(values):
    return values.reshape(-1)

def target_U(x, target):
    return -target.log_prob(x)

def U_gmm_means(x, t, means):
    """
    x:      [N, D]
    t:      [N] or scalar
    means:  [C, D]
    beta(t): precision schedule (1/variance)
    returns U_t(x): [N, 1]
    """
    device = x.device
    N, D = x.shape
    C = means.shape[0]

    if torch.is_tensor(t):
        t = t.view(-1, 1, 1).to(device)
    else:
        t = torch.tensor(t, device=device, dtype=x.dtype).view(1, 1, 1)

    means_ = means[None, :, :]                       # [1,C,D]
    diff2 = (x[:, None, :] - t * means_).pow(2).sum(dim=-1)  # [N,C]
    energy = 0.5 * diff2  # [N,C]
    log_norm = -0.5 * D * math.log(2 * math.pi) * 0
    # ---- log-sum-exp mixture + normalization
    logp =  - math.log(C) + log_norm + torch.logsumexp(-energy, dim=1) # [N]
    Ut = -logp.unsqueeze(-1)                                       # [N,1]
    return Ut


def energy_components(x, t, means, U_net, modes, target, prior, energy_model):
    del prior

    t_flat = t.reshape(-1)
    U_net_t = _flatten_energy(_optional_potential(U_net, x, t))

    if means is not None:
        U_gmm_t = _flatten_energy(U_gmm_means(x, t, means))
        U1 = target_U(x, target)
        U1 = _flatten_energy(U1) if target is not None else torch.full_like(U_gmm_t, float("nan"))
        U_t_total = U_gmm_t
    elif modes is not None:
        U_gmm_t = _flatten_energy(energy_model(x, t * 0))
        U1 = _flatten_energy(target_U(x, target))
        U_t_total = t_flat * U1 + (1 - t_flat) * U_gmm_t + t_flat * (1 - t_flat) * U_net_t
    else:
        U_gmm_t = 0.5 * (x ** 2).sum(dim=1)
        U1 = _flatten_energy(target_U(x, target))
        if U_net is not None:
            U_t_total = (1 - t_flat) * U_gmm_t + t_flat * U1 + t_flat * (1 - t_flat) * U_net_t
        else:
            U_t_total = (1 - t_flat) * U_gmm_t + t_flat * U1

    return {
        "U1": U1,
        "U_gmm_t": U_gmm_t,
        "U_net": U_net_t,
        "U_t": U_t_total,
    }

def U_t(x, t, means, U_net, modes, target, prior, energy_model):
    return energy_components(x, t, means, U_net, modes, target, prior, energy_model)["U_t"]


def grad_U_t(x, t, means, U_net, modes, target, prior, energy_model):
    U = U_t(x, t, means, U_net, modes, target, prior, energy_model)
    grad = torch.autograd.grad(U.sum(), x, create_graph=True)[0]
    return _clip_energy_derivative(grad)

def partial_t_U(x, t, means, U_net, modes, target, prior, energy_model):
    U = U_t(x, t, means, U_net, modes, target, prior, energy_model)
    partial_t = torch.autograd.grad(U.sum(), t, create_graph=True)[0]
    return _clip_energy_derivative(partial_t)

## This should be put into the vector field module?
def divergence(y, x):
    noise = torch.randn_like(x)
    grad_y = torch.autograd.grad(y, x, grad_outputs=noise, create_graph=True, retain_graph=True)[0]
    return (grad_y * noise).sum(dim=1)
