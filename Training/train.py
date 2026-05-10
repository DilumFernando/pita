import csv
import os

import torch

from Energies.interpolations import divergence, grad_U_t, partial_t_U
from Energies.targets import target_payload
from Models.models import build_model_bundle
from Utils.metrics import mmd_rbf, unweighted_w2_from_samples, weighted_w2, ess_from_weights
from Utils.misc import soft_mode_weights_from_particles
from Utils.plotting import plot_training_metrics, plot_walkers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA_T = 0.1
TRAIN_T_MAX = 1.0
TRAIN_T_MIN = 0.05
ESS_RESAMPLE_THRESHOLD = 0.3
DRIFT_CLIP = 100.0
EGNN_DIVERGENCE_BATCH_SIZE = 5
FINITE_DIFF_DIVERGENCE_EPS = 1e-3


def _clip_update_vector(value, clip=DRIFT_CLIP):
    value = torch.nan_to_num(value, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(value, min=-clip, max=clip)


def _drift_and_divergence(drift_net, x, t, divergence_batch_size=None):
    if divergence_batch_size is None or divergence_batch_size <= 0 or divergence_batch_size >= x.shape[0]:
        b = _clip_update_vector(drift_net(x, t))
        return b, divergence(b, x)

    drift_chunks = []
    div_chunks = []
    for start in range(0, x.shape[0], divergence_batch_size):
        stop = min(start + divergence_batch_size, x.shape[0])
        x_chunk = x[start:stop]
        t_chunk = t[start:stop]
        b_chunk = _clip_update_vector(drift_net(x_chunk, t_chunk))
        drift_chunks.append(b_chunk)
        div_chunks.append(divergence(b_chunk, x_chunk))
    return torch.cat(drift_chunks, dim=0), torch.cat(div_chunks, dim=0)


def _drift_and_finite_difference_divergence(drift_net, x, t, eps=FINITE_DIFF_DIVERGENCE_EPS):
    b = _clip_update_vector(drift_net(x, t))
    noise = torch.randn_like(x)
    x_plus = (x + eps * noise).detach()
    x_minus = (x - eps * noise).detach()
    b_plus = _clip_update_vector(drift_net(x_plus, t))
    b_minus = _clip_update_vector(drift_net(x_minus, t))
    div = ((b_plus - b_minus) * noise).sum(dim=1) / (2.0 * eps)
    div = torch.nan_to_num(div, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)
    return b, torch.clamp(div, min=-DRIFT_CLIP, max=DRIFT_CLIP)


def interpolation_dir(kind, dim, num_components, run_name=None):
    if run_name:
        return os.path.join(PROJECT_ROOT, f"{kind}_interpolation", run_name)
    return os.path.join(PROJECT_ROOT, f"{kind}_interpolation", f"{dim}d_{num_components}gmm")


def artifact_dirs(kind, dim, num_components, run_name=None):
    run_dir = interpolation_dir(kind, dim, num_components, run_name=run_name)
    return {
        "run": run_dir,
        "models": os.path.join(run_dir, "models"),
        "plots": os.path.join(run_dir, "plots"),
        "metrics": os.path.join(run_dir, "metrics"),
    }


def _mode_summary(mode_weights):
    mode_weights = mode_weights.detach()
    return {f"mode_weight_{idx}": float(value.item()) for idx, value in enumerate(mode_weights)}


def _build_metrics_header(num_components):
    base = [
        "step",
        "loss",
        "manual_loss",
        "ctds_loss",
        "ctds_pinn_loss",
        "ctds_component_bal_loss",
        "modal_loss",
        "effective_modal_loss_weight",
        "resample_count",
        "lr_drift",
        "lr_free_energy",
        "lr_potential",
        "lr_energy_model",
        "ess_count",
        "w2",
        "weighted_w2",
        "mmd",
        "drift_grad_norm",
        "free_energy_grad_norm",
        "potential_grad_norm",
        "energy_model_grad_norm",
        "total_grad_norm",
        "beta_max",
    ]
    base.extend(f"mode_weight_{idx}" for idx in range(num_components))
    return base


def _resolve_interpolation_kind(interpolation_kind, means, modes, U_net):
    if interpolation_kind is not None:
        return interpolation_kind
    if means is not None and U_net is not None:
        return "learned"
    if means is not None:
        return "mean"
    if modes is not None:
        return "alps"
    return "fixed"


def _hyperparameter_summary(
    dim,
    num_components,
    n_walkers,
    steps,
    epsilon,
    K,
    interpolation_kind,
    modal_loss_weight,
    loss_type,
    modal_loss_end_fraction,
    energy_model=None,
    model_type="mlp",
    model_kwargs=None,
):
    return {
        "dim": dim,
        "num_components": num_components,
        "n_walkers": n_walkers,
        "steps": steps,
        "epsilon": epsilon,
        "K": K,
        "delta_t": 1.0 / K,
        "interpolation": interpolation_kind,
        "modal_loss_weight": modal_loss_weight,
        "loss_type": loss_type,
        "modal_loss_end_fraction": modal_loss_end_fraction,
        "beta_max_learnable": _energy_model_beta_max_learnable(energy_model),
        "model_type": model_type,
        "model_kwargs": dict(model_kwargs or {}),
    }


def _save_training_checkpoint(
    checkpoint_dir,
    drift_net,
    F_net,
    U_net,
    energy_model,
    prior,
    mixture,
    step,
    metrics,
    model_type="mlp",
    model_kwargs=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    _sync_alps_prior_with_energy_model(prior, energy_model)

    torch.save(drift_net.state_dict(), os.path.join(checkpoint_dir, "drift.pth"))
    torch.save(F_net.state_dict(), os.path.join(checkpoint_dir, "free_energy.pth"))
    if energy_model is not None:
        torch.save(energy_model, os.path.join(checkpoint_dir, "energy_model.pth"))
    if U_net is not None:
        torch.save(U_net.state_dict(), os.path.join(checkpoint_dir, "potential.pth"))
    if prior is not None:
        torch.save(prior, os.path.join(checkpoint_dir, "prior.pth"))

    torch.save(target_payload(mixture), os.path.join(checkpoint_dir, "mixture.pth"))
    torch.save({"step": step, "metrics": dict(metrics)}, os.path.join(checkpoint_dir, "metadata.pth"))
    torch.save(
        {"model_type": str(model_type or "mlp").lower(), "model_kwargs": dict(model_kwargs or {})},
        os.path.join(checkpoint_dir, "architecture.pth"),
    )


def _current_training_horizon(step, steps, gamma=1.0):
    progress = 0.0 if steps <= 0 else step / steps
    return TRAIN_T_MIN + (TRAIN_T_MAX - TRAIN_T_MIN) * (progress ** gamma)


def _sync_alps_prior_with_energy_model(prior, energy_model):
    if prior is None or energy_model is None or not hasattr(energy_model, "beta_max"):
        return
    with torch.no_grad():
        cov = (1.0 / energy_model.beta_max.detach()).to(device=prior.covs.device, dtype=prior.covs.dtype)
        eye = torch.eye(prior.dim, device=prior.covs.device, dtype=prior.covs.dtype)
        prior.covs.copy_(cov * eye.view(1, prior.dim, prior.dim).expand(prior.nmodes, -1, -1))


def _energy_model_beta_max(energy_model):
    if energy_model is None or not hasattr(energy_model, "beta_max"):
        return float("nan")
    return float(energy_model.beta_max.detach().item())


def _energy_model_beta_max_learnable(energy_model):
    if energy_model is None:
        return False
    return bool(getattr(energy_model, "beta_max_learnable", False))


def _trainable_parameters(module):
    if module is None:
        return []
    return [param for param in module.parameters() if param.requires_grad]


def _effective_modal_loss_weight(
    step,
    steps,
    modal_loss_weight,
    modal_loss_end_fraction,
):
    if modal_loss_weight == 0.0:
        return 0.0
    progress = 0.0 if steps <= 0 else step / steps
    return modal_loss_weight if progress < modal_loss_end_fraction else 0.0



def systematic_resample(weights):
    N = weights.shape[0]
    device = weights.device
    weights = weights / (weights.sum() + 1e-12)
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0

    u0 = torch.rand(1, device=device) / N
    positions = u0 + torch.arange(N, device=device, dtype=weights.dtype) / N
    return torch.searchsorted(cdf, positions)


def resample_particles(x, weights, A=None):
    N = x.shape[0]
    idx = systematic_resample(weights)
    x_new = x[idx]
    weights_new = torch.full((N,), 1.0 / N, device=x.device, dtype=x.dtype)

    if A is None:
        return x_new, weights_new, idx

    A_new = A[idx]
    return x_new, weights_new, A_new, idx


def kl_loss(p, nu, eps=1e-12):
    nu = nu / (nu.sum() + eps)
    return torch.sum(nu * (torch.log(nu + eps) - torch.log(p + eps)))


def modal_weights_loss(mode_weights, eps=1e-12):
    num_components = mode_weights.shape[-1]
    target = torch.ones(num_components, device=mode_weights.device, dtype=mode_weights.dtype)
    return kl_loss(mode_weights, target, eps=eps)


def _path_mode_weights(x, true_modes, weights):
    if true_modes is None:
        return None, torch.tensor(0.0, device=x.device)
    mode_weights, _ = soft_mode_weights_from_particles(x, true_modes, particle_weights=weights)
    return mode_weights, modal_weights_loss(mode_weights)


def _ctds_extract_weights(log_weights, log_clamp_val=4.0):
    log_weights = torch.clamp(log_weights, max=log_clamp_val)
    weights = torch.exp(log_weights)
    return weights / (weights.mean(dim=0, keepdim=True) + 1e-12)


def _compute_ctds_loss_terms(
    drift_net,
    F_net,
    U_net,
    means,
    modes,
    mixture,
    prior,
    energy_model,
    xs,
    ts,
    log_weights,
    true_modes,
    component_bal_lambda,
):
    device = xs.device
    weights = _ctds_extract_weights(log_weights)

    xs_flat = xs.reshape(-1, xs.shape[-1]).detach().clone().requires_grad_(True)
    ts_flat = ts.reshape(-1).detach().clone().requires_grad_(True)
    weights_flat = weights.reshape(-1, 1)

    control = drift_net(xs_flat, ts_flat)
    score = -grad_U_t(xs_flat, ts_flat, means, U_net, modes, mixture, prior, energy_model)
    div = divergence(control, xs_flat)
    dt_ln_pt = -partial_t_U(xs_flat, ts_flat, means, U_net, modes, mixture, prior, energy_model)
    dt_Ft = torch.autograd.grad(F_net(ts_flat).sum(), ts_flat, create_graph=True)[0]

    raw_loss = (dt_Ft + div + (control * score).sum(dim=1) + dt_ln_pt).pow(2).unsqueeze(1)
    pinn_loss = torch.mean(weights_flat * raw_loss)

    if true_modes is None:
        component_bal_loss = torch.zeros((), device=device, dtype=xs.dtype)
    else:
        nmodes = true_modes.shape[0]
        uniform = torch.ones(nmodes, device=device, dtype=xs.dtype) / nmodes
        eps = torch.finfo(xs.dtype).eps
        assignment_temperature = 1.0
        mode_losses = []

        for time_idx in range(xs.shape[1]):
            xt = xs[:, time_idx, :]
            wt = weights[:, time_idx]
            wt = wt / (torch.sum(wt) + eps)
            distances_sq = torch.cdist(xt, true_modes.to(xt)).square()
            soft_assignments = torch.softmax(-distances_sq / assignment_temperature, dim=1)
            mode_mass = torch.sum(wt.unsqueeze(1) * soft_assignments, dim=0)
            mode_mass = torch.clamp(mode_mass, min=eps)
            mode_mass = mode_mass / torch.sum(mode_mass)
            kl = torch.sum(mode_mass * (torch.log(mode_mass) - torch.log(uniform)))
            mode_losses.append(kl)

        component_bal_loss = torch.stack(mode_losses).mean()

    total_loss = pinn_loss + component_bal_lambda * component_bal_loss
    return total_loss, pinn_loss, component_bal_loss


def _train_path(
    drift_net,
    F_net,
    U_net,
    epsilon,
    mixture,
    step,
    steps,
    dim,
    n_walkers,
    means,
    modes,
    prior,
    energy_model,
    prior_samples,
    true_modes,
    true_samples,
    plot_every,
    modal_loss_weight,
    modal_loss_end_fraction,
    K,
    loss_type,
    backward_per_step=False,
    divergence_batch_size=None,
    finite_difference_divergence=False,
):
    device = next(drift_net.parameters()).device
    delta_t = 1.0 / K
    x = prior_samples.to(device)
    A = torch.zeros(n_walkers, device=device).requires_grad_(True)
    t_k = torch.zeros(size=(n_walkers,), requires_grad=True, device=device)
    current_T = _current_training_horizon(step, steps)
    effective_modal_weight = _effective_modal_loss_weight(
        step,
        steps,
        modal_loss_weight,
        modal_loss_end_fraction,
    )

    compute_ctds = loss_type == "ctds"
    manual_total_loss = torch.tensor(0.0, device=device)
    did_backward = False
    final_mode_weights = None
    final_weights = torch.ones(n_walkers, device=device)
    final_modal_loss = torch.tensor(0.0, device=device)
    resample_count = 0
    path_xs = [] if compute_ctds else None
    path_ts = [] if compute_ctds else None
    path_log_weights = [] if compute_ctds else None

    while t_k.mean() <= current_T:
        if finite_difference_divergence:
            b, div_b = _drift_and_finite_difference_divergence(drift_net, x, t_k)
        else:
            b, div_b = _drift_and_divergence(drift_net, x, t_k, divergence_batch_size)
        gradU = grad_U_t(x, t_k, means, U_net, modes, mixture, prior, energy_model)
        dtU = partial_t_U(x, t_k, means, U_net, modes, mixture, prior, energy_model)

        noise = torch.randn_like(x)
        x_next = x - epsilon * gradU * delta_t + b * delta_t + ((2 * epsilon * delta_t) ** 0.5) * noise
        x_next = torch.nan_to_num(x_next, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)
        A = A + div_b * delta_t - (b * gradU).sum(dim=1) * delta_t - dtU * delta_t
        A = torch.nan_to_num(A, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)

        x_det = x.detach().requires_grad_(True)
        t_det = t_k.detach().requires_grad_(True)
        if finite_difference_divergence:
            b_eval, div_b_eval = _drift_and_finite_difference_divergence(drift_net, x_det, t_det)
        else:
            b_eval, div_b_eval = _drift_and_divergence(drift_net, x_det, t_det, divergence_batch_size)
        gradU_eval = grad_U_t(x_det, t_det, means, U_net, modes, mixture, prior, energy_model)
        dtU_eval = partial_t_U(x_det, t_det, means, U_net, modes, mixture, prior, energy_model)
        dF_dt = _clip_update_vector(torch.autograd.grad(F_net(t_det).sum(), t_det, create_graph=True)[0])

        err = div_b_eval - (gradU_eval * b_eval).sum(dim=1) - dtU_eval + dF_dt
        err = torch.nan_to_num(err, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)
        weights = torch.exp(A - A.max())
        loss = (weights * err.pow(2)).mean() / (weights.mean() + 1e-12)

        mode_weights, modal_loss = _path_mode_weights(x_det, true_modes, weights)
        if mode_weights is not None:
            final_mode_weights = mode_weights.detach()
        final_modal_loss = modal_loss.detach()
        final_weights = weights.detach()
        t_next = t_k + delta_t
        if compute_ctds:
            path_xs.append(x_next)
            path_ts.append(t_next)
            path_log_weights.append(A)
        combined_loss = loss + effective_modal_weight * modal_loss
        step_loss = delta_t * combined_loss
        if backward_per_step and loss_type == "manual":
            step_loss.backward()
            did_backward = True
            manual_total_loss = manual_total_loss + step_loss.detach()
        else:
            manual_total_loss = manual_total_loss + step_loss

        ess_value = ess_from_weights(weights)
        if float(ess_value.detach().item()) < ESS_RESAMPLE_THRESHOLD:
            x_next_resampled, _, _, _ = resample_particles(x_next.detach(), weights.detach(), A.detach())
            x_next = x_next_resampled.requires_grad_(True)
            A = torch.zeros(n_walkers, device=device, dtype=x_next.dtype).requires_grad_(True)
            resample_count += 1

        x = x_next.detach().requires_grad_(True)
        t_k = t_next.detach().requires_grad_(True)
        A = A.detach().requires_grad_(True)

    if plot_every and step % plot_every == 0 and dim in (1, 2):
        plot_walkers(
            x,
            current_T,
            means,
            U_net,
            step,
            modes,
            mixture,
            prior,
            energy_model,
            true_modes,
            true_samples,
            particle_weights=final_weights,
        )

    if compute_ctds:
        ctds_total_loss, ctds_pinn_loss, ctds_component_bal_loss = _compute_ctds_loss_terms(
            drift_net=drift_net,
            F_net=F_net,
            U_net=U_net,
            means=means,
            modes=modes,
            mixture=mixture,
            prior=prior,
            energy_model=energy_model,
            xs=torch.stack(path_xs, dim=1),
            ts=torch.stack(path_ts, dim=1),
            log_weights=torch.stack(path_log_weights, dim=1),
            true_modes=true_modes,
            component_bal_lambda=effective_modal_weight,
        )
    else:
        ctds_total_loss = torch.tensor(float("nan"), device=device)
        ctds_pinn_loss = torch.tensor(float("nan"), device=device)
        ctds_component_bal_loss = torch.tensor(float("nan"), device=device)

    if loss_type == "manual":
        optimize_loss = manual_total_loss
    elif loss_type == "ctds":
        optimize_loss = ctds_total_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    metrics = {
        "step": int(step),
        "loss": float(optimize_loss.detach().item()),
        "manual_loss": float(manual_total_loss.detach().item()),
        "ctds_loss": float(ctds_total_loss.detach().item()),
        "ctds_pinn_loss": float(ctds_pinn_loss.detach().item()),
        "ctds_component_bal_loss": float(ctds_component_bal_loss.detach().item()),
        "modal_loss": float(final_modal_loss.item()),
        "effective_modal_loss_weight": float(effective_modal_weight),
        "resample_count": int(resample_count),
        "ess_count": float(ess_from_weights(final_weights).item()),
        "w2": float("nan"),
        "weighted_w2": float("nan"),
        "mmd": float("nan"),
    }
    if final_mode_weights is not None:
        metrics.update(_mode_summary(final_mode_weights))

    return optimize_loss, final_weights, x.detach(), metrics, did_backward


def train_step(
    drift_net,
    F_net,
    U_net,
    epsilon,
    mixture,
    step,
    steps,
    dim=2,
    n_walkers=1000,
    K=50,
    T=1.0,
    plot_every=5,
    means=None,
    modes=None,
    prior=None,
    energy_model=None,
    prior_samples=None,
    true_modes=None,
    true_samples=None,
    modal_loss_weight=0.0,
    modal_loss_end_fraction=0.6,
    loss_type="manual",
    backward_per_step=False,
    divergence_batch_size=None,
    finite_difference_divergence=False,
):
    del T
    return _train_path(
        drift_net=drift_net,
        F_net=F_net,
        U_net=U_net,
        epsilon=epsilon,
        mixture=mixture,
        step=step,
        steps=steps,
        dim=dim,
        n_walkers=n_walkers,
        means=means,
        modes=modes,
        prior=prior,
        energy_model=energy_model,
        prior_samples=prior_samples,
        true_modes=true_modes,
        true_samples=true_samples,
        plot_every=plot_every,
        modal_loss_weight=modal_loss_weight,
        modal_loss_end_fraction=modal_loss_end_fraction,
        K=K,
        loss_type=loss_type,
        backward_per_step=backward_per_step,
        divergence_batch_size=divergence_batch_size,
        finite_difference_divergence=finite_difference_divergence,
    )


def grad_global_norm(params):
    sq = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        sq += grad.pow(2).sum().item()
    return sq ** 0.5


def _configure_models_and_optimizer(
    dim,
    device,
    interpolation_kind,
    U_net,
    energy_model,
    num_components,
    model_type="mlp",
    model_kwargs=None,
):
    del num_components

    model_kwargs = dict(model_kwargs or {})
    drift_net, F_net, default_potential_net = build_model_bundle(
        dim,
        device,
        model_type=model_type,
        model_kwargs=model_kwargs,
    )
    potential_net = U_net.to(device) if U_net is not None else None
    energy_model = energy_model.to(device) if energy_model is not None else None
    scheduler = None

    if interpolation_kind == "learned":
        if potential_net is None:
            potential_net = default_potential_net
        optimizer = torch.optim.Adam(
            list(drift_net.parameters()) + list(F_net.parameters()) + list(potential_net.parameters()),
            lr=1e-4,
        )
        print("using the potential network")
    elif interpolation_kind == "mean":
        optimizer = torch.optim.Adam(list(drift_net.parameters()) + list(F_net.parameters()), lr=1e-4)
        print("using the mean interpolation")
    elif interpolation_kind == "alps":
        lr = 1e-3
        if potential_net is None:
            potential_net = default_potential_net
        if energy_model is None:
            raise ValueError("ALPS interpolation requires an energy_model with learnable parameters.")
        energy_model_params = _trainable_parameters(energy_model)
        optimizer = torch.optim.Adam(
            [
                {"params": drift_net.parameters(), "lr": lr},
                {"params": F_net.parameters(), "lr": lr},
                {"params": potential_net.parameters(), "lr": lr},
                {"params": energy_model_params, "lr": lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        beta_max_status = "learnable" if _energy_model_beta_max_learnable(energy_model) else "fixed"
        print(f"using the alps interpolation with {beta_max_status} beta_max and {model_type} nets")
    else:
        optimizer = torch.optim.Adam(list(drift_net.parameters()) + list(F_net.parameters()), lr=1e-4)
        print("using the fixed interpolation")

    return drift_net, F_net, potential_net, energy_model, optimizer, scheduler


def _initialize_run_dirs(interpolation_kind, dim, num_components, run_name=None):
    dirs = artifact_dirs(interpolation_kind, dim, num_components, run_name=run_name)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _initialize_logging(metrics_dir, num_components, hyperparams):
    metrics_csv = os.path.join(metrics_dir, "checkpoint_log.csv")
    metrics_history_path = os.path.join(metrics_dir, "training_metrics_history.pth")
    hyperparams_path = os.path.join(metrics_dir, "hyperparameters.pth")
    hyperparams_txt_path = os.path.join(metrics_dir, "hyperparameters.txt")
    best_checkpoints_path = os.path.join(metrics_dir, "best_checkpoints_summary.pth")
    metrics_header = _build_metrics_header(num_components)

    with open(metrics_csv, "w", newline="") as handle:
        csv.writer(handle).writerow(metrics_header)

    return {
        "metrics_csv": metrics_csv,
        "metrics_history_path": metrics_history_path,
        "hyperparams_path": hyperparams_path,
        "hyperparams_txt_path": hyperparams_txt_path,
        "best_checkpoints_path": best_checkpoints_path,
        "metrics_header": metrics_header,
        "hyperparams": hyperparams,
    }


def _optimizer_lrs(optimizer, interpolation_kind, U_net, energy_model):
    lrs = [float(group["lr"]) for group in optimizer.param_groups]

    if interpolation_kind == "alps":
        drift_lr = lrs[0] if len(lrs) > 0 else float("nan")
        free_energy_lr = lrs[1] if len(lrs) > 1 else drift_lr
        potential_lr = lrs[2] if len(lrs) > 2 else free_energy_lr
        energy_model_lr = lrs[3] if len(lrs) > 3 else potential_lr
    else:
        shared_lr = lrs[0] if lrs else float("nan")
        drift_lr = shared_lr
        free_energy_lr = shared_lr
        potential_lr = shared_lr if U_net is not None else 0.0
        energy_model_lr = shared_lr if energy_model is not None else 0.0

    return {
        "lr_drift": drift_lr,
        "lr_free_energy": free_energy_lr,
        "lr_potential": potential_lr,
        "lr_energy_model": energy_model_lr,
    }


def _update_step_metrics(
    step_metrics,
    drift_net,
    F_net,
    U_net,
    energy_model,
    samples,
    weights,
    true_samples,
    optimizer,
    interpolation_kind,
):
    drift_grad_norm = grad_global_norm(drift_net.parameters())
    free_energy_grad_norm = grad_global_norm(F_net.parameters())
    potential_grad_norm = grad_global_norm(U_net.parameters()) if U_net is not None else 0.0
    energy_model_grad_norm = grad_global_norm(energy_model.parameters()) if energy_model is not None else 0.0
    total_grad_norm = (
        drift_grad_norm ** 2
        + free_energy_grad_norm ** 2
        + potential_grad_norm ** 2
        + energy_model_grad_norm ** 2
    ) ** 0.5
    beta_max = _energy_model_beta_max(energy_model)

    step_metrics.update(
        {
            "drift_grad_norm": drift_grad_norm,
            "free_energy_grad_norm": free_energy_grad_norm,
            "potential_grad_norm": potential_grad_norm,
            "energy_model_grad_norm": energy_model_grad_norm,
            "total_grad_norm": total_grad_norm,
            "beta_max": beta_max,
        }
    )
    step_metrics.update(_optimizer_lrs(optimizer, interpolation_kind, U_net, energy_model))

    if true_samples is None:
        step_metrics["w2"] = float("nan")
        step_metrics["weighted_w2"] = float("nan")
        step_metrics["mmd"] = float("nan")
        return step_metrics

    generated_samples = samples.to(weights.device)
    reference_samples = true_samples.to(weights.device)
    try:
        step_metrics["w2"] = float(unweighted_w2_from_samples(generated_samples, reference_samples))
        step_metrics["weighted_w2"] = float(weighted_w2(generated_samples, reference_samples, weights.detach()))
    except ModuleNotFoundError:
        step_metrics["w2"] = float("nan")
        step_metrics["weighted_w2"] = float("nan")
    step_metrics["mmd"] = float(mmd_rbf(generated_samples, reference_samples).item())
    return step_metrics


def _maybe_save_best_checkpoint(
    best_weighted_w2,
    step_metrics,
    best_models_dir,
    drift_net,
    F_net,
    U_net,
    energy_model,
    prior,
    mixture,
    step,
    model_type="mlp",
    model_kwargs=None,
):
    current_weighted_w2 = step_metrics["weighted_w2"]
    best_checkpoint_summary = {}
    if current_weighted_w2 == current_weighted_w2 and current_weighted_w2 < best_weighted_w2:
        best_weighted_w2 = current_weighted_w2
        weighted_w2_dir = os.path.join(best_models_dir, "best_weighted_w2")
        _save_training_checkpoint(
            weighted_w2_dir,
            drift_net,
            F_net,
            U_net,
            energy_model,
            prior,
            mixture,
            step,
            step_metrics,
            model_type=model_type,
            model_kwargs=model_kwargs,
        )
        best_checkpoint_summary["best_weighted_w2"] = {
            "step": step,
            "weighted_w2": best_weighted_w2,
            "path": weighted_w2_dir,
        }
    return best_weighted_w2, best_checkpoint_summary


def _log_step(step, step_metrics, num_components):
    if step % 5 != 0:
        return

    mode_values = [f"{step_metrics.get(f'mode_weight_{idx}', 0.0):.4f}" for idx in range(num_components)]
    print(
        f"Step {step:5d} | loss {step_metrics['loss']:.4f} | "
        f"manual {step_metrics['manual_loss']:.4f} | "
        f"ctds {step_metrics['ctds_loss']:.4f} | "
        f"ESS {step_metrics['ess_count']:.2f} | "
        f"modal {step_metrics['modal_loss']:.4f} | "
        f"modal_w {step_metrics['effective_modal_loss_weight']:.2e} | "
        f"resamples {int(step_metrics['resample_count'])} | "
        f"grad_d {step_metrics['drift_grad_norm']:.4f} | "
        f"grad_f {step_metrics['free_energy_grad_norm']:.4f} | "
        f"grad_u {step_metrics['potential_grad_norm']:.4f} | "
        f"grad_tot {step_metrics['total_grad_norm']:.4f}"
    )
    print(
        f"           | mode_weights [{', '.join(mode_values)}] | "
        f"w2 {step_metrics['w2']:.4f} | "
        f"weighted_w2 {step_metrics['weighted_w2']:.4f} | "
        f"mmd {step_metrics['mmd']:.4f} | "
        f"lr [{step_metrics['lr_drift']:.2e}, {step_metrics['lr_free_energy']:.2e}, {step_metrics['lr_potential']:.2e}]"
    )


def _final_model_paths(models_dir, interpolation_kind, U_net):
    if interpolation_kind == "learned":
        return {
            "drift": os.path.join(models_dir, "final_drift_means.pth"),
            "free_energy": os.path.join(models_dir, "final_free_energy_means.pth"),
            "energy_model": None,
            "potential": os.path.join(models_dir, "final_potential_means.pth"),
            "mixture": os.path.join(models_dir, "mixture_means.pth"),
            "prior": os.path.join(models_dir, "prior_means.pth"),
        }
    if interpolation_kind == "mean":
        return {
            "drift": os.path.join(models_dir, "final_drift_mean_interp.pth"),
            "free_energy": os.path.join(models_dir, "final_free_energy_mean_interp.pth"),
            "energy_model": None,
            "potential": None,
            "mixture": os.path.join(models_dir, "mixture_mean_interp.pth"),
            "prior": os.path.join(models_dir, "prior_mean_interp.pth"),
        }
    if interpolation_kind == "alps":
        prefix = "covs_"
        return {
            "drift": os.path.join(models_dir, f"{prefix}final_drift_alps.pth"),
            "free_energy": os.path.join(models_dir, f"{prefix}final_free_energy_alps.pth"),
            "energy_model": os.path.join(models_dir, f"{prefix}final_energy_model_alps.pth"),
            "potential": os.path.join(models_dir, f"{prefix}final_potential_alps.pth"),
            "mixture": os.path.join(models_dir, f"{prefix}mixture_weights_exp.pth"),
            "prior": os.path.join(models_dir, f"{prefix}prior_alps_exp.pth"),
        }
    return {
        "drift": os.path.join(models_dir, "final_drift.pth"),
        "free_energy": os.path.join(models_dir, "final_free_energy.pth"),
        "energy_model": None,
        "potential": os.path.join(models_dir, "final_potential.pth") if U_net is not None else None,
        "mixture": os.path.join(models_dir, "mixture.pth"),
        "prior": os.path.join(models_dir, "prior.pth"),
    }


def train_and_save(
    dim,
    num_components,
    mixture,
    means,
    U_net,
    modes,
    device,
    prior,
    energy_model,
    prior_samples,
    n_walkers=1000,
    steps=1000,
    epsilon=0,
    K=50,
    modal_loss_weight=0.0,
    modal_loss_end_fraction=0.4,
    loss_type="manual",
    true_modes=None,
    true_samples=None,
    interpolation_kind=None,
    run_name=None,
    model_type="mlp",
    model_kwargs=None,
):
    interpolation_kind = _resolve_interpolation_kind(interpolation_kind, means, modes, U_net)
    model_type = str(model_type or "mlp").lower()
    model_kwargs = dict(model_kwargs or {})
    dirs = _initialize_run_dirs(interpolation_kind, dim, num_components, run_name=run_name)
    drift_net, F_net, U_net, energy_model, optimizer, scheduler = _configure_models_and_optimizer(
        dim=dim,
        device=device,
        interpolation_kind=interpolation_kind,
        U_net=U_net,
        energy_model=energy_model,
        num_components=num_components,
        model_type=model_type,
        model_kwargs=model_kwargs,
    )

    hyperparams = _hyperparameter_summary(
        dim,
        num_components,
        n_walkers,
        steps,
        epsilon,
        K,
        interpolation_kind,
        modal_loss_weight,
        loss_type,
        modal_loss_end_fraction,
        energy_model,
        model_type,
        model_kwargs,
    )
    logging_paths = _initialize_logging(dirs["metrics"], num_components, hyperparams)
    metrics_history = []
    best_models_dir = os.path.join(dirs["models"], "best_checkpoints")
    os.makedirs(best_models_dir, exist_ok=True)
    best_weighted_w2 = float("inf")
    best_checkpoint_summary = {}

    for step in range(steps):
        backward_per_step = model_type == "egnn" and loss_type == "manual"
        divergence_batch_size = EGNN_DIVERGENCE_BATCH_SIZE if model_type == "egnn" else None
        finite_difference_divergence = model_type == "egnn"
        loss, weights, samples, step_metrics, did_backward = train_step(
            drift_net=drift_net,
            F_net=F_net,
            U_net=U_net,
            epsilon=epsilon,
            mixture=mixture,
            step=step,
            steps=steps,
            dim=dim,
            n_walkers=n_walkers,
            K=K,
            modal_loss_weight=modal_loss_weight,
            modal_loss_end_fraction=modal_loss_end_fraction,
            loss_type=loss_type,
            means=means,
            modes=modes,
            prior=prior,
            energy_model=energy_model,
            prior_samples=prior_samples,
            true_modes=true_modes,
            true_samples=true_samples,
            backward_per_step=backward_per_step,
            divergence_batch_size=divergence_batch_size,
            finite_difference_divergence=finite_difference_divergence,
        )

        if not did_backward:
            loss.backward()
        step_metrics = _update_step_metrics(
            step_metrics,
            drift_net,
            F_net,
            U_net,
            energy_model,
            samples,
            weights,
            true_samples,
            optimizer,
            interpolation_kind,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step_metrics["beta_max"] = _energy_model_beta_max(energy_model)
        optimizer.zero_grad()

        best_weighted_w2, maybe_summary = _maybe_save_best_checkpoint(
            best_weighted_w2,
            step_metrics,
            best_models_dir,
            drift_net,
            F_net,
            U_net,
            energy_model,
            prior,
            mixture,
            step,
            model_type,
            model_kwargs,
        )
        if maybe_summary:
            best_checkpoint_summary.update(maybe_summary)

        metrics_history.append(dict(step_metrics))
        with open(logging_paths["metrics_csv"], "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=logging_paths["metrics_header"], extrasaction="ignore")
            writer.writerow(step_metrics)
        _log_step(step, step_metrics, num_components)

    torch.save(logging_paths["hyperparams"], logging_paths["hyperparams_path"])
    with open(logging_paths["hyperparams_txt_path"], "w") as handle:
        for key, value in logging_paths["hyperparams"].items():
            handle.write(f"{key}: {value}\n")
    torch.save(best_checkpoint_summary, logging_paths["best_checkpoints_path"])
    torch.save(metrics_history, logging_paths["metrics_history_path"])
    plot_training_metrics(metrics_history, dirs["plots"], logging_paths["hyperparams"])

    final_paths = _final_model_paths(dirs["models"], interpolation_kind, U_net)
    _sync_alps_prior_with_energy_model(prior, energy_model)
    torch.save(drift_net.state_dict(), final_paths["drift"])
    torch.save(F_net.state_dict(), final_paths["free_energy"])
    if final_paths["energy_model"] is not None and energy_model is not None:
        torch.save(energy_model, final_paths["energy_model"])
    if final_paths["potential"] is not None and U_net is not None:
        torch.save(U_net.state_dict(), final_paths["potential"])
    torch.save(target_payload(mixture), final_paths["mixture"])
    if prior is not None:
        torch.save(prior, final_paths["prior"])
    torch.save(
        {"model_type": model_type, "model_kwargs": model_kwargs},
        os.path.join(dirs["models"], "architecture.pth"),
    )
