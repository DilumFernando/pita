import torch
from Energies.interpolations import *
from Energies.gmm import create_gaussian_mixture
from Utils.metrics import ess_from_weights
from Utils.plotting import plot_walkers
from Training.train import resample_particles

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

        b = drift(x, t_tensor)
        grad_U = grad_U_t(x, t_tensor, means, U_net, modes, mixture, prior, energy_model)
        dtU = partial_t_U(x, t_tensor, means, U_net, modes, mixture, prior, energy_model)
        div_b = divergence(b, x)
        
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
