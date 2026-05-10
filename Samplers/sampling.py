import torch
from Energies.interpolations import *
from Energies.gmm import create_gaussian_mixture
from Utils.metrics import ess_from_weights
from Utils.plotting import plot_walkers
from Training.train import resample_particles


EGNN_DIVERGENCE_BATCH_SIZE = 5
FINITE_DIFF_DIVERGENCE_EPS = 1e-3


def _finite_difference_drift_and_divergence(drift, x, t_tensor, eps=FINITE_DIFF_DIVERGENCE_EPS):
    b = drift(x, t_tensor)
    noise = torch.randn_like(x)
    x_plus = (x + eps * noise).detach().requires_grad_(True)
    x_minus = (x - eps * noise).detach().requires_grad_(True)
    b_plus = drift(x_plus, t_tensor)
    b_minus = drift(x_minus, t_tensor)
    div = ((b_plus - b_minus) * noise).sum(dim=1) / (2.0 * eps)
    return b, torch.nan_to_num(div, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)


def _drift_and_divergence(drift, x, t_tensor):
    if hasattr(drift, "n_particles"):
        return _finite_difference_drift_and_divergence(drift, x, t_tensor)

    if not hasattr(drift, "n_particles") or x.shape[0] <= EGNN_DIVERGENCE_BATCH_SIZE:
        b = drift(x, t_tensor)
        return b, divergence(b, x)

    drift_chunks = []
    div_chunks = []
    for start in range(0, x.shape[0], EGNN_DIVERGENCE_BATCH_SIZE):
        stop = min(start + EGNN_DIVERGENCE_BATCH_SIZE, x.shape[0])
        x_chunk = x[start:stop]
        t_chunk = t_tensor[start:stop]
        b_chunk = drift(x_chunk, t_chunk)
        drift_chunks.append(b_chunk)
        div_chunks.append(divergence(b_chunk, x_chunk))
    return torch.cat(drift_chunks, dim=0), torch.cat(div_chunks, dim=0)


def langevin_sampler(
    x0,
    A0,
    dt,
    steps,
    epsilon,
    means,
    U_net,
    modes,
    mixture,
    prior,
    energy_model,
    true_modes,
    true_samples,
    device,
    plot_every=None,
):
    x = x0.detach().requires_grad_(True)
    dim = x.shape[-1]
    A = A0
    x_trajectory = [x0.detach().cpu()] 
    A_trajectory = [A.cpu()]

    for step in range(0, steps+1):
            # print("weights:", torch.softmax(energy_model.logits, dim=0).detach().cpu())
            # print("grad_U norm:", grad_U.norm().item())

        t_tensor = torch.full((x.shape[0],), step * dt, device=device).requires_grad_(True)

        grad_U = grad_U_t(x, t_tensor, means, U_net, modes, mixture, prior, energy_model) 
        dtU = partial_t_U(x, t_tensor, means, U_net, modes, mixture, prior, energy_model)
        noise = torch.randn_like(x)
        x = x - epsilon * grad_U * dt + torch.sqrt(torch.tensor(2 * epsilon * dt)) * noise
        A = A - dtU * dt

        # ess = ess_from_weights(A.exp())

        # if ess < 0.3 * x.shape[0]:
        #     x, weights, A, idx = resample_particles(x, A.exp(), A)
        #     A = torch.zeros_like(A)
        #     x = x.detach().requires_grad_(True)
        
        x_trajectory.append(x.detach().cpu())
        A_trajectory.append(A.detach().cpu())
        x = x.detach().requires_grad_(True)
        # A = A.detach().requires_grad_(True)
        plot_interval = 10 if plot_every is None else plot_every
        if plot_interval and step % plot_interval == 0:
            if dim == 1 or dim == 2:
                plot_walkers(x, dt * step, means, U_net, step, modes, mixture, prior, energy_model, true_modes, true_samples, particle_weights=A.exp())


    return torch.stack(x_trajectory), torch.stack(A_trajectory)  # (steps, batch, 2)

def NET_Sampler(
    drift,
    x0,
    A0,
    dt,
    steps,
    epsilon,
    means,
    U_net,
    modes,
    mixture,
    prior,
    energy_model,
    true_modes,
    true_samples,
    device,
    plot_every=None,
):
    x = x0.detach().requires_grad_(True)
    A = A0.detach().requires_grad_(True)
    x_trajectory = [x0.detach().cpu()]  # list of (batch, 2)
    A_trajectory = [A0.detach().cpu()]
    dim = x.shape[-1]
    for step in range(steps+1):
        # if step == 0 or step == 250:
        plot_interval = 50 if plot_every is None else plot_every
        if plot_interval and step % plot_interval == 0:
            if dim == 1 or dim == 2:
                plot_walkers(x, dt * step, means, U_net, step, modes, mixture, prior, energy_model, true_modes, true_samples, particle_weights=A.exp())
        noise = torch.randn_like(x)
        t_tensor = torch.full((x.shape[0],), step * dt, device=device).requires_grad_(True)

        b, div_b = _drift_and_divergence(drift, x, t_tensor)
        grad_U = grad_U_t(x, t_tensor, means, U_net, modes, mixture, prior, energy_model)
        dtU = partial_t_U(x, t_tensor, means, U_net, modes, mixture, prior, energy_model)
        
        x = x - epsilon * grad_U * dt + b * dt + (2 * epsilon * dt) ** 0.5 * noise
        A = A + div_b * dt - dtU * dt - (b * grad_U).sum(dim=1) * dt 

        # ess = ess_from_weights(A.exp())

        # if ess < 0.3 * x.shape[0]:
        #     x, weights, A, idx = resample_particles(x, A.exp(), A)
        #     A = torch.zeros_like(A)
        #     x = x.detach().requires_grad_(True)

        x_trajectory.append(x.detach().cpu())
        A_trajectory.append(A.detach().cpu())
        x = x.detach().requires_grad_(True)
    
    return torch.stack(x_trajectory), torch.stack(A_trajectory)


def prior_samples(modes, num_samples, device):
    # means_ = torch.tensor([[0, 0]], device=device, dtype=torch.float32)
    # beta_max = torch.tensor(1, device=device)
    means_ = modes
    beta_max = torch.tensor(10, device=device)
    K, dim = means_.shape  
    mix = create_gaussian_mixture(dim, K, means=means_, covs=1/beta_max, device=device)
    x = torch.stack([mix.sample() for i in range(0, num_samples)]).to(device).requires_grad_(True)

    # x = torch.randn(num_samples, dim, device=device).requires_grad_(True)
    return x
